"""The nearline generation worker (design §11 step 5): history changes in, MiniOneRec top-50 out.

A native process on the Mac (MPS needs the host, not a container):

  1. registers itself: SET gen:worker ... PX 30 s, refreshed every 10 s. apply.lua puts a user
     into gen:queue only while that key exists, so nothing is queued when no worker runs;
  2. claims due users (Lua, atomic: ZRANGEBYSCORE <= now, ZREM, take and delete their pending
     changes), at most --claim at a time;
  3. reads their recent history (the last 10 items, oldest first) and history version in one
     MULTI, so the two belong together; users without history are skipped (cold start);
  4. sends the histories as SIDs to the generator process (minionerec.serve_gen in the minionerec
     environment: evaluate.py's prompts, constraint and 50-beam search, prompt encoded once);
  5. maps the SIDs back to items (items that share a SID in the offline tie order: training
     popularity, then item id), keeps the top 50, and writes gen:{uid}:{ver} unless a result of a
     newer history version is already there;
  6. logs every job to jobs.jsonl: the changes it answers (event time and the time apply.lua
     applied them), when it was due, claimed, read, generated and written.

A change that arrives while its user is being generated queues the user again (apply.lua runs
after the claim), so no change is lost; the result written is the one of the history read in
step 3. With --only-users only those users are generated (the A/B's gen arm); the others'
changes are claimed and dropped. --backfill queues users as due now (the A/B's warm start).

Writes <out>/jobs.jsonl, <out>/queue.jsonl (queue length and due users every 10 s), <out>/run.json
(settings, the generator's ready line, counts, throughput, the generator's memory footprint,
swap-out and load at the start and the end).

Usage: python -m ml.nearline.worker --out artifacts/nearline/<run> [--only-users u.json]
           [--backfill u.json] [--seconds 3600]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import signal
import subprocess
import threading
import time
from pathlib import Path

import redis

from ml import paths
from ml.eval import footprint, machine
from ml.eval.splits import item_sids
from ml.nearline import keys

ROOT = Path(__file__).resolve().parents[2]
GENERATOR_CMD = "uv run --project minionerec --frozen python -m minionerec.serve_gen"
DEFAULT_BUNDLE = paths.ARTIFACTS / "bundles" / "current"
QUEUE_SAMPLE_S = 10.0

# KEYS 1 gen:queue. ARGV 1 now (ms), 2 at most this many users.
# Returns uid, due, "<event>:<applied>,..." per user. The pending lists are named from the
# uids (not declared in KEYS): fine on the single state instance, not on a cluster.
CLAIM_LUA = """
local due = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[1], 'WITHSCORES', 'LIMIT', 0, tonumber(ARGV[2]))
local out = {}
for i = 1, #due, 2 do
  local uid = due[i]
  redis.call('ZREM', KEYS[1], uid)
  local pk = 'gen:pending:' .. uid
  local changes = redis.call('LRANGE', pk, 0, -1)
  redis.call('DEL', pk)
  table.insert(out, uid)
  table.insert(out, due[i + 1])
  table.insert(out, table.concat(changes, ','))
end
return out
"""

# KEYS 1 gen:{uid}:{ver}. ARGV 1 hv, 2 payload, 3 TTL (ms). 1 = written, 0 = a newer one is there.
WRITE_LUA = """
local cur = redis.call('GET', KEYS[1])
if cur then
  local ok, obj = pcall(cjson.decode, cur)
  if ok and type(obj) == 'table' and obj.hv and tonumber(obj.hv) > tonumber(ARGV[1]) then
    return 0
  end
end
redis.call('SET', KEYS[1], ARGV[2], 'PX', ARGV[3])
return 1
"""


def redis_ms(r: redis.Redis) -> int:
    sec, usec = r.time()
    return int(sec) * 1000 + int(usec) // 1000


def sid_items(sids: list[str], tie_order: list[int]) -> dict[str, list[int]]:
    """SID -> the items that have it, in tie order (ml.eval.minionerec.sid_items)."""
    out: dict[str, list[int]] = {}
    for item in tie_order:
        out.setdefault(sids[item], []).append(item)
    return out


def to_items(
    sids: list[str], scores: list[float], by_sid: dict[str, list[int]], k: int = keys.TOP_K
) -> tuple[list[int], list[float]]:
    """Generated SIDs (best first) -> the first k distinct items; each item gets its SID's beam
    score. SIDs that name no item (none are expected: the search is constrained) are skipped."""
    items, out_scores, seen = [], [], set()
    for sid, score in zip(sids, scores, strict=True):
        for item in by_sid.get(sid.strip('"\n').strip(), ()):
            if item not in seen:
                seen.add(item)
                items.append(item)
                out_scores.append(score)
                if len(items) == k:
                    return items, out_scores
    return items, out_scores


class GeneratorProcess:
    """minionerec.serve_gen behind a pipe: one JSON line per request and per answer."""

    def __init__(self, cmd: list[str], log: Path):
        self.log = log.open("a")
        env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
        env.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)  # an op MPS lacks must fail, not fall back
        self.proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
        )
        self.ready = self._read()
        if not self.ready.get("ready"):
            raise SystemExit(f"generator did not start: {self.ready}")
        self._next = 0

    def _read(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise SystemExit(f"generator exited ({self.proc.poll()}); see {self.log.name}")
        return json.loads(line)

    def generate(self, histories: list[list[str]]) -> dict:
        self._next += 1
        self.proc.stdin.write(json.dumps({"id": self._next, "histories": histories}) + "\n")
        self.proc.stdin.flush()
        out = self._read()
        if out.get("error"):
            raise RuntimeError(out["error"])
        return out

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                self.proc.wait(timeout=30)
        self.log.close()


class Worker:
    def __init__(self, args):
        self.args = args
        self.r = redis.Redis.from_url(args.redis_url)
        self.claim = self.r.register_script(CLAIM_LUA)
        self.write = self.r.register_script(WRITE_LUA)
        self.sids = item_sids()
        pop = json.loads((Path(args.bundle) / "pop.json").read_text())
        self.by_sid = sid_items(self.sids, pop["items"])
        self.only = None
        if args.only_users:
            self.only = {int(u) for u in json.loads(Path(args.only_users).read_text())}
        self.stop = threading.Event()
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.jobs = (self.out / "jobs.jsonl").open("a")
        self.counts = {"jobs": 0, "written": 0, "older": 0, "no_history": 0, "dropped": 0}
        self.batches: list[dict] = []

    # ---------------------------------------------------------------- registration

    def heartbeat(self, info: dict) -> None:
        payload = json.dumps(info, separators=(",", ":"))
        while not self.stop.is_set():
            try:
                self.r.set(keys.WORKER_KEY, payload, px=keys.HEARTBEAT_TTL_MS)
            except redis.RedisError as e:
                print(f"worker   heartbeat failed: {e}", flush=True)
            self.stop.wait(keys.HEARTBEAT_EVERY_S)

    def sample_queue(self) -> None:
        with (self.out / "queue.jsonl").open("a") as f:
            while not self.stop.is_set():
                try:
                    now = redis_ms(self.r)
                    pipe = self.r.pipeline(transaction=False)
                    pipe.zcard(keys.QUEUE_KEY)
                    pipe.zcount(keys.QUEUE_KEY, "-inf", now)
                    queued, due = pipe.execute()
                    f.write(json.dumps({"t": now, "queued": queued, "due": due}) + "\n")
                    f.flush()
                except redis.RedisError as e:
                    print(f"worker   queue sample failed: {e}", flush=True)
                self.stop.wait(QUEUE_SAMPLE_S)

    def backfill(self, users: list[int]) -> int:
        now = redis_ms(self.r)
        pipe = self.r.pipeline(transaction=False)
        for u in users:
            pipe.zadd(keys.QUEUE_KEY, {str(u): now})
        pipe.execute()
        return len(users)

    # ---------------------------------------------------------------- one round

    def round(self, gen: GeneratorProcess) -> int:
        now = redis_ms(self.r)
        flat = self.claim(keys=[keys.QUEUE_KEY], args=[now, self.args.claim])
        claimed_ms = redis_ms(self.r)
        jobs = []
        for i in range(0, len(flat), 3):
            uid = int(flat[i])
            if self.only is not None and uid not in self.only:
                self.counts["dropped"] += 1
                continue
            jobs.append(
                {
                    "uid": uid,
                    "due_ms": int(float(flat[i + 1])),
                    "changes": keys.parse_changes(flat[i + 2].decode()),
                }
            )
        if not jobs:
            return 0
        try:
            self._answer(gen, jobs, claimed_ms)
        except Exception:
            # Claimed but not written (Redis or the generator failed): queue them again, due now,
            # so their changes are answered by a later round. Best effort: Redis may be the
            # problem; the changes' times are in this round's log line either way.
            try:
                now = redis_ms(self.r)
                self.r.zadd(keys.QUEUE_KEY, {str(j["uid"]): now for j in jobs if "status" not in j})
            except redis.RedisError:
                pass
            for j in jobs:
                if "status" not in j:
                    j["status"] = "failed"
                    self._log(j)
            raise
        return len(jobs)

    def _answer(self, gen: GeneratorProcess, jobs: list[dict], claimed_ms: int) -> None:
        pipe = self.r.pipeline(transaction=True)  # MULTI: history and hv of the same moment
        for j in jobs:
            pipe.zrange(keys.hist_key(j["uid"]), -keys.HISTORY_WINDOW, -1)
            pipe.get(keys.hv_key(j["uid"]))
        state = pipe.execute()
        read_ms = redis_ms(self.r)
        todo = []
        for n, j in enumerate(jobs):
            hist = [int(x) for x in state[2 * n]]
            j["hv"] = int(state[2 * n + 1] or 0)
            j["claimed_ms"], j["read_ms"] = claimed_ms, read_ms
            j["history"] = len(hist)
            if not hist:
                j["status"] = "no_history"
                self.counts["no_history"] += 1
                self._log(j)
                continue
            j["sids"] = [self.sids[i] for i in hist]
            todo.append(j)
        if not todo:
            return
        t0 = time.perf_counter()
        res = gen.generate([j["sids"] for j in todo])
        gen_seconds = time.perf_counter() - t0
        generated_ms = redis_ms(self.r)
        pipe = self.r.pipeline(transaction=False)
        for n, j in enumerate(todo):
            items, scores = to_items(res["sids"][n], res["scores"][n], self.by_sid)
            j["items"] = len(items)
            value = keys.encode(j["hv"], generated_ms, items, scores)
            self.write(
                keys=[keys.result_key(j["uid"], self.version)],
                args=[j["hv"], value, keys.RESULT_TTL_MS],
                client=pipe,
            )
        written = pipe.execute()
        written_ms = redis_ms(self.r)
        self.batches.append(
            {"users": len(todo), "seconds": gen_seconds, "generator": res.get("batches")}
        )
        for j, ok in zip(todo, written, strict=True):
            j.pop("sids")
            j.update(
                generated_ms=generated_ms,
                written_ms=written_ms,
                status="written" if ok == 1 else "older",
                batch_users=len(todo),
                gen_seconds=round(gen_seconds, 3),
            )
            self.counts["written" if ok == 1 else "older"] += 1
            self._log(j)

    def _log(self, j: dict) -> None:
        self.counts["jobs"] += 1
        self.jobs.write(json.dumps(j, separators=(",", ":")) + "\n")
        self.jobs.flush()

    # ---------------------------------------------------------------- main loop

    def run(self) -> dict:
        args = self.args
        started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        load_start, vm_start = os.getloadavg(), machine.vm_counters()
        host_ms = int(time.time() * 1000)
        clock_offset_ms = redis_ms(self.r) - host_ms  # Redis (the VM's clock) - host clock
        cmd = shlex.split(args.generator_cmd) + [
            "--max-batch",
            str(args.max_batch),
            "--mps-memory-gb",
            str(args.mps_memory_gb),
        ]
        t_setup = time.monotonic()
        gen = GeneratorProcess(cmd, self.out / "generator.log")
        setup_seconds = time.monotonic() - t_setup
        sampler = footprint.Sampler(gen.proc.pid, interval=1.0).start()
        self.version = gen.ready["version"]
        info = {"version": self.version, "pid": os.getpid(), "started": started}
        threads = [
            threading.Thread(target=self.heartbeat, args=(info,), daemon=True),
            threading.Thread(target=self.sample_queue, daemon=True),
        ]
        for t in threads:
            t.start()
        backfilled = 0
        if args.backfill:
            users = [int(u) for u in json.loads(Path(args.backfill).read_text())]
            backfilled = self.backfill(users)
            print(f"worker   backfill: {backfilled:,} users queued", flush=True)
        print(f"worker   {self.version} ready ({setup_seconds:.0f} s); claiming", flush=True)
        deadline = time.monotonic() + args.seconds if args.seconds else None
        t_loop = time.monotonic()
        try:
            while not self.stop.is_set():
                if deadline is not None and time.monotonic() > deadline:
                    break
                try:
                    n = self.round(gen)
                except (redis.RedisError, RuntimeError) as e:
                    # RuntimeError: the generator answered with an error (the process lives on).
                    self.counts["failed_rounds"] = self.counts.get("failed_rounds", 0) + 1
                    print(f"worker   round failed: {e}; retrying in 1 s", flush=True)
                    self.stop.wait(1.0)
                    continue
                if n == 0:
                    self.stop.wait(args.idle_ms / 1000)
        finally:
            self.stop.set()
            loop_seconds = time.monotonic() - t_loop
            with contextlib.suppress(redis.RedisError):
                self.r.delete(keys.WORKER_KEY)
            memory = sampler.stop()
            gen.close()
            self.jobs.close()
        gen_seconds = sum(b["seconds"] for b in self.batches)
        users = sum(b["users"] for b in self.batches)
        summary = {
            "started": started,
            "settings": {
                "claim": args.claim,
                "max_batch": args.max_batch,
                "mps_memory_gb": args.mps_memory_gb,
                "only_users": len(self.only) if self.only is not None else None,
                "backfill": backfilled,
                "debounce_ms": keys.DEBOUNCE_MS,
                "max_wait_ms": keys.MAX_WAIT_MS,
                "top_k": keys.TOP_K,
                "history_window": keys.HISTORY_WINDOW,
                "bundle": str(args.bundle),
            },
            "generator": gen.ready,
            "generator_setup_seconds": setup_seconds,
            "clock_offset_ms": clock_offset_ms,
            "counts": self.counts,
            "batches": len(self.batches),
            "generated_users": users,
            "generate_seconds": gen_seconds,
            "seconds_per_user": gen_seconds / users if users else None,
            "loop_seconds": loop_seconds,
            "memory": memory,
            "loadavg_start": list(load_start),
            "loadavg_end": list(os.getloadavg()),
            "swapped_out_bytes": machine.swapped_out_bytes(vm_start, machine.vm_counters()),
            "containers_end": machine.containers(),
        }
        (self.out / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(
            f"worker   done: {self.counts}; {users:,} users in {gen_seconds:.0f} s of generation",
            flush=True,
        )
        return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--redis-url", default=os.environ.get("REDIS_URL", "redis://localhost:16390/0"))
    ap.add_argument(
        "--bundle", type=Path, default=Path(os.environ.get("RECLOOP_BUNDLE", DEFAULT_BUNDLE))
    )
    ap.add_argument("--only-users", type=Path, help="JSON list: generate only for these users")
    ap.add_argument("--backfill", type=Path, help="JSON list: queue these users as due now")
    ap.add_argument("--claim", type=int, default=8, help="users per round")
    ap.add_argument("--max-batch", type=int, default=8, help="prompts per generate() call")
    ap.add_argument("--mps-memory-gb", type=float, default=8.0)
    ap.add_argument("--idle-ms", type=float, default=200.0, help="pause when nothing is due")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this long (0: on signal)")
    ap.add_argument("--generator-cmd", default=GENERATOR_CMD)
    args = ap.parse_args(argv)
    w = Worker(args)

    def _stop(signum, frame):
        w.stop.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    w.run()


if __name__ == "__main__":
    main()
