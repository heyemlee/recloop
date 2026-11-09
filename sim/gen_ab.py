"""The nearline A/B (design §11 step 6): three recall sources against three + MiniOneRec (gen).

One phase in the recall layer over all users, 50 / 50:

  recall-gen-ab-v1   control: rrf_weights {sasrec 1, itemcf 1, pop 0.3}
                     gen:     rrf_weights {sasrec 1, itemcf 1, pop 0.3, gen 1.0}, gen_version

The gen weight is fixed beforehand at 1.0, SASRec's and ItemCF's, not tuned. The recall layer's
running experiments (recall-rrf-weights-v1 since M1) are stopped for the phase and put back as
they were afterwards. The rank layer runs no experiment: both arms are ordered by RRF (the api's
RANKER default). LightGBM was trained on the three sources' features only -- a gen item has none
of them -- so it is not used here; the served log still records its features and scores.

Sequence (one command):
  1. the consumers catch up and Redis is reseeded (as sim.ab: the dataset users' training
     histories), so both arms start from the same state;
  2. the generation worker (ml.nearline.worker, started here as a child process) registers and
     generates every gen-arm user from the seeded history (backfill; its duration is recorded);
     only gen-arm users are generated all along (--only-users: the Python twin of the bucketing);
  3. the experiment starts; the traffic starts once the api serves its config;
  4. --seconds of simulated traffic (sim.traffic: think times as M4, --concurrency sessions,
     lower than M4's 16 so the worker keeps up; see the plan);
  5. the experiment stops and the stopped ones come back; the worker stops.
Then `make samples` (attributed impressions) and `make nearline-report`.

Writes artifacts/nearline/ab/: run.json, gen_users.json, worker/ (jobs.jsonl, queue.jsonl,
run.json, generator.log), traffic/ledger.jsonl.

Usage: python -m sim.gen_ab [--seconds 3600] [--concurrency 4] [--backfill-timeout 5400]
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from analysis import srm as srm_mod
from sim import ab as ab_mod
from sim import ads as ads_mod
from sim import redis_stats, standing
from sim import traffic as traffic_mod
from sim import world as world_mod
from sim.admin import Admin

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "nearline" / "ab"
EXP_ID = "recall-gen-ab-v1"
CONTROL = {"sasrec": 1.0, "itemcf": 1.0, "pop": 0.3}
GEN_WEIGHT = 1.0


def experiment(gen_version: str) -> dict:
    return {
        "exp_id": EXP_ID,
        "layer": "recall",
        "status": "running",
        "bucket_from": 0,
        "bucket_to": 1000,
        "variants": [
            {"name": "control", "share": 50, "params": {"rrf_weights": CONTROL}},
            {
                "name": "gen",
                "share": 50,
                "params": {
                    "rrf_weights": {**CONTROL, "gen": GEN_WEIGHT},
                    "gen_version": gen_version,
                },
            },
        ],
    }


def gen_users(exp: dict, users: list[int]) -> list[int]:
    """The users the bucketing puts into the gen arm (Python twin of serving/internal/experiment)."""
    return [u for u in users if srm_mod.assign([exp], "recall", str(u)) == "gen"]


def wait_for(cond, timeout: float, what: str, every: float = 2.0, proc=None) -> float:
    t0 = time.time()
    while not cond():
        if proc is not None and proc.poll() is not None:
            raise SystemExit(f"gen_ab: the worker exited ({proc.returncode}) before {what}")
        if time.time() - t0 > timeout:
            raise SystemExit(f"gen_ab: {what} did not happen within {timeout:.0f} s")
        time.sleep(every)
    return time.time() - t0


def count_jobs(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for _ in f)


def export_served(path: Path, t0: float, t1: float) -> int:
    """The phase's served home first pages from events_raw (after the log sink caught up): the
    arm, the history version, the candidate cache result and what the request saw of the user's
    generation result (served.gen). The report reads this file, not the database."""
    import psycopg

    agg = ROOT / ".cache" / "bin" / "aggregator"
    subprocess.run([str(agg), "wait-lag", "log-sink"], check=True)
    sql = """
        SELECT user_id, request_id::text, extract(epoch FROM server_ts),
               payload -> 'exp' -> 'recall' ->> 'variant', (payload ->> 'hv')::bigint,
               payload ->> 'candidate_cache', payload -> 'gen', payload -> 'degraded'
        FROM events_raw
        WHERE type = 'served' AND payload ->> 'surface' = 'home'
          AND (payload ->> 'offset')::int = 0
          AND payload -> 'exp' -> 'recall' ->> 'exp_id' = %s
          AND server_ts >= to_timestamp(%s) AND server_ts < to_timestamp(%s) + interval '1 minute'
    """
    n = 0
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, path.open("w") as f:
        for uid, rid, ts, arm, hv, cache, gen, degraded in conn.execute(sql, (EXP_ID, t0, t1)):
            row = {"user_id": uid, "request_id": rid, "t": float(ts), "arm": arm, "hv": hv,
                   "cache": cache, "gen": gen, "degraded": degraded or []}  # fmt: skip
            f.write(json.dumps(row) + "\n")
            n += 1
    return n


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--api", default=os.environ.get("RECLOOP_API_URL", "http://127.0.0.1:18080"))
    p.add_argument("--seconds", type=float, default=3600)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument(
        "--think-ms",
        type=traffic_mod.think_arg,
        default=traffic_mod.THINK_MS,
        help="min,max pause between a session's page views",
    )
    p.add_argument("--backfill-timeout", type=float, default=5400)
    p.add_argument("--worker-args", default="", help="extra arguments for ml.nearline.worker")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)
    if (args.out / "run.json").exists() or (args.out / "worker" / "jobs.jsonl").exists():
        raise SystemExit(f"{args.out} holds an earlier run; move it away first")
    args.out.mkdir(parents=True, exist_ok=True)
    admin = Admin(args.api, actor="gen-ab-run")
    ads = ads_mod.Ads(args.api, os.environ.get("DATABASE_URL", ""))
    campaigns = standing.ensure(ads)
    state = redis_stats.client("state")

    # 1. the same start as every A/B phase
    before = [e for e in admin.experiments() if e["layer"] == "recall" and e["status"] == "running"]
    for e in before:
        if e["exp_id"] != EXP_ID:
            admin.stop_experiment(e["exp_id"])
    ab_mod.reset_state()

    # 2. the worker, generating the gen arm's users
    world = world_mod.load(world_mod.DEFAULT_WORLD)
    users = [u for u, _ in world.users]
    probe = experiment("?")
    arm = gen_users(probe, users)
    (args.out / "gen_users.json").write_text(json.dumps(arm) + "\n")
    worker_dir = args.out / "worker"
    cmd = [
        sys.executable,
        "-m",
        "ml.nearline.worker",
        "--out",
        str(worker_dir),
        "--only-users",
        str(args.out / "gen_users.json"),
        "--backfill",
        str(args.out / "gen_users.json"),
        *shlex.split(args.worker_args),
    ]
    log = (args.out / "worker.log").open("a")
    worker = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    t_start = time.time()
    try:
        wait_for(
            lambda: state.exists("gen:worker") == 1,
            900,
            "the worker's registration (model load)",
            proc=worker,
        )
        registered = time.time()
        gen_version = json.loads(state.get("gen:worker"))["version"]
        print(
            f"gen_ab   {time.strftime('%H:%M:%S')} worker {gen_version} up; backfill of {len(arm):,} users"
        )
        jobs = worker_dir / "jobs.jsonl"
        backfill_s = wait_for(
            lambda: count_jobs(jobs) >= len(arm),
            args.backfill_timeout,
            "the backfill",
            every=5,
            proc=worker,
        )
        print(f"gen_ab   backfill done in {backfill_s:.0f} s")

        # 3. the experiment
        exp = experiment(gen_version)
        version = admin.put_experiment(exp)
        admin.wait_config(version)
        redis_before = redis_stats.counters()

        # 4. traffic
        traffic = traffic_mod.Traffic(
            args.out / "traffic", args.api, args.concurrency, think_ms=args.think_ms
        )
        t0 = traffic.start(args.seconds)
        print(
            f"gen_ab   {time.strftime('%H:%M:%S')} {EXP_ID} running, {args.seconds:.0f} s of traffic"
        )
        traffic.thread.join()
        traffic.stop()
        t1 = time.time()

        # 5. back as it was
        admin.stop_experiment(EXP_ID)
        restored = []
        for e in before:
            if e["exp_id"] == EXP_ID:
                continue
            admin.put_experiment(
                {k: e[k] for k in ("exp_id", "layer", "bucket_from", "bucket_to", "variants")}
                | {"status": "running"}
            )
            restored.append(e["exp_id"])
        redis = redis_stats.delta(redis_before, redis_stats.counters())
        print(f"gen_ab   {redis_stats.describe(redis)}")
        served = export_served(args.out / "served_gen.jsonl", t0, t1)
        print(f"gen_ab   {served:,} served home first pages of {EXP_ID} exported")
    finally:
        worker.terminate()  # SIGTERM: the worker deregisters and writes its run.json
        try:
            worker.wait(timeout=180)
        except subprocess.TimeoutExpired:
            worker.kill()
        log.close()
    s = traffic.stats
    run = {
        "api": args.api,
        "bundle": admin.overview()["active"],
        "campaigns": len(campaigns),
        "concurrency": args.concurrency,
        "think_ms": list(args.think_ms),
        "exp_id": EXP_ID,
        "experiment": exp,
        "config_version": version,
        "stopped_and_restored": restored,
        "gen_version": gen_version,
        "gen_weight": GEN_WEIGHT,
        "gen_users": len(arm),
        "worker_started": t_start,
        "worker_registered": registered,
        "backfill_seconds": backfill_s,
        "from": t0,
        "to": t1,
        "sessions": s.sessions,
        "page_views": dict(s.page_views),
        "get_status": {str(k): v for k, v in s.status.items()},
        "events": dict(s.events),
        "ads": dict(s.ads),
        "redis": redis,
    }
    (args.out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(f"gen_ab   done: {s.sessions:,} sessions, {sum(s.page_views.values()):,} page views")
    admin.close()


if __name__ == "__main__":
    main()
