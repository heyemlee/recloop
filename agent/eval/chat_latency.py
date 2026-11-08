"""The chat path's latency and cost, measured through the running services (design §10.4).

  python -m agent.eval.chat_latency [--agent-url URL] [--repeats 3] [--warmup 30]   (make agent-latency)

Sends the evaluation's messages to POST /v1/chat one at a time, as the store would see them:
each from a dataset user with a training-period history (a fixed sample), so the chat surface's
recommendations run SASRec and the purchase history is the one the feed counts. Per reply it
records the client's time to the reply event and to the end of the stream, the service's stage
timings (search, checks, the recommendations call, the ads call, rendering), whether
personalisation changed the shown items, whether the ads module was asked and filled the slot,
degradations, and the language model's calls and cost (none with the rules parser).

The first `--warmup` replies are not recorded (the query encoder is slow on its first queries).
Before starting it waits, checking every minute for up to 20, for a 1-minute load average below
2; it records the load at the start and the end and every Docker container outside the stack's
compose project seen during the run, and marks the result indicative when the start was busy.
Needs the agent service, the api, the model server and the stack; writes
artifacts/eval/agent/chat/latency.json, which `agent.eval.report` renders into agent.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import http.client
import json
import os
import random
import subprocess
import sys
import time
import urllib.parse
from collections import Counter
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq

from agent import paths
from agent.eval import cases as case_io

USER_SEED = 20261108
MIN_HISTORY = 3
OUT = paths.RUNS / "chat" / "latency.json"
BUSY_LOAD = 2.0


def containers() -> list[tuple[str, str]] | None:
    """(name, compose project) of the running containers; None when Docker does not answer."""
    try:
        out = subprocess.run(
            ["docker", "ps", "--format", '{{.Names}}\t{{.Label "com.docker.compose.project"}}'],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return [tuple(line.split("\t", 1)) for line in out.splitlines() if line]  # type: ignore[misc]


def others(project: str) -> set[str]:
    return {name for name, proj in containers() or [] if proj != project}


def wait_for_load(max_wait_s: float, poll_s: float) -> tuple[float, float]:
    """Wait until the 1-minute load average is below BUSY_LOAD (at most max_wait_s); return the
    load when the wait ended and the seconds waited."""
    t0 = time.monotonic()
    load = os.getloadavg()[0]
    while load >= BUSY_LOAD and time.monotonic() - t0 < max_wait_s:
        print(f"waiting for a 1-minute load below {BUSY_LOAD:g} (now {load:.2f})", file=sys.stderr)
        time.sleep(poll_s)
        load = os.getloadavg()[0]
    return load, time.monotonic() - t0


def sample_users(n: int, seed: int = USER_SEED) -> list[int]:
    """n dataset users with at least MIN_HISTORY training-period interactions (fixed seed)."""
    t = pq.read_table(paths.interactions_parquet(), columns=["user_id", "split"])
    train = t.filter(pc.equal(t["split"], "train"))["user_id"].to_pylist()
    counts = Counter(train)
    users = sorted(u for u, c in counts.items() if c >= MIN_HISTORY)
    return random.Random(seed).sample(users, min(n, len(users)))


class ChatClient:
    """POST /v1/chat on a kept-alive connection; reads the event stream as it arrives."""

    def __init__(self, url: str, timeout_s: float = 120.0):
        u = urllib.parse.urlsplit(url)
        self.host, self.port, self.timeout_s = u.hostname or "127.0.0.1", u.port or 80, timeout_s
        self.conn: http.client.HTTPConnection | None = None

    def _connection(self) -> http.client.HTTPConnection:
        if self.conn is None:
            self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_s)
        return self.conn

    def health(self) -> dict[str, Any]:
        conn = self._connection()
        conn.request("GET", "/healthz")
        return json.loads(conn.getresponse().read())

    def reset(self) -> None:
        if self.conn is not None:
            self.conn.close()
        self.conn = None

    def _post(self, body: str) -> tuple[http.client.HTTPResponse, float]:
        for attempt in (0, 1):
            reused = self.conn is not None
            conn = self._connection()
            t0 = time.perf_counter()
            try:
                conn.request("POST", "/v1/chat", body, {"Content-Type": "application/json"})
                return conn.getresponse(), t0
            except (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError):
                self.reset()  # the server closed the idle connection: once more, fresh
                if not reused or attempt == 1:
                    raise
        raise AssertionError("unreachable")

    def turn(self, user_id: int, message: str) -> dict[str, Any]:
        body = json.dumps({"user_id": user_id, "message": message})
        resp, t0 = self._post(body)
        if resp.status != 200:
            raise RuntimeError(f"/v1/chat: HTTP {resp.status} {resp.read()[:200]!r}")
        events: dict[str, Any] = {}
        at: dict[str, float] = {}
        name, data = None, []
        for raw in resp:
            line = raw.decode().rstrip("\r\n")
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
            elif line == "" and name:
                events[name] = json.loads("\n".join(data))
                at[name] = (time.perf_counter() - t0) * 1000
                name, data = None, []
        total = (time.perf_counter() - t0) * 1000
        if "error" in events:
            raise RuntimeError(f"/v1/chat failed: {events['error']}")
        return {"events": events, "at_ms": at, "total_ms": total}


def record(case: case_io.Case, user_id: int, repeat: int, out: dict[str, Any]) -> dict[str, Any]:
    reply, done = out["events"]["reply"], out["events"]["done"]
    return {
        "case_id": case.id,
        "template": case.template,
        "user_id": user_id,
        "repeat": repeat,
        "status": reply["status"],
        "items": len(reply["items"]),
        "sponsored": reply["sponsored"] is not None,
        "ads": reply["ads"],
        "personal": None
        if reply["personal"] is None
        else {"hits": reply["personal"]["hits"], "moved": reply["personal"]["moved"]},
        "degraded": reply["degraded"],
        "parse_source": reply["parse"]["source"],
        "tools": reply["tools"],
        "client_reply_ms": round(out["at_ms"]["reply"], 3),
        "client_total_ms": round(out["total_ms"], 3),
        "server_ms": done["server_ms"],
        "timings_ms": done["timings_ms"],
        "llm": done["llm"],
    }


def campaigns(api_url: str) -> str:
    """What the ads stage had to work with: the api's active campaigns."""
    u = urllib.parse.urlsplit(api_url)
    try:
        conn = http.client.HTTPConnection(u.hostname or "127.0.0.1", u.port or 80, timeout=5)
        conn.request("GET", "/v1/campaigns")
        listed = json.loads(conn.getresponse().read())
    except (OSError, ValueError) as e:
        return f"unknown ({e})"
    active = [c for c in listed if c.get("status") == "active"]
    if not active:
        return "none active."
    items = sum(len(c.get("item_ids") or []) for c in active)
    bids = sorted(c["bid_cpc_cents"] for c in active)
    return (
        f"{len(active)} active, {items} items in all, bids {bids[0]}–{bids[-1]}¢ per click, "
        f"daily budgets {min(c['daily_budget_cents'] for c in active) / 100:.0f}–"
        f"{max(c['daily_budget_cents'] for c in active) / 100:.0f} USD."
    )


def git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "describe", "--always", "--dirty"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    default_url = os.environ.get("RECLOOP_AGENT_URL", "http://127.0.0.1:18083")
    ap.add_argument("--agent-url", default=default_url)
    ap.add_argument(
        "--api-url", default=os.environ.get("RECLOOP_API_URL", "http://127.0.0.1:18080")
    )
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--limit", type=int, help="only the first N cases (smoke test)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument(
        "--compose-project",
        default=os.environ.get("COMPOSE_PROJECT_NAME", "recloop"),
        help="the stack's compose project: its containers are not 'other' load",
    )
    ap.add_argument("--quiet-wait", type=float, default=20 * 60, help="seconds (0: no wait)")
    ap.add_argument("--quiet-poll", type=float, default=60.0, help="seconds between checks")
    args = ap.parse_args()

    cases = case_io.load()[: args.limit]
    users = sample_users(len(cases))
    client = ChatClient(args.agent_url)
    health = client.health()
    load_ready, waited = wait_for_load(args.quiet_wait, args.quiet_poll)
    client.reset()  # the service closes idle connections after a few seconds
    seen = others(args.compose_project)
    started = dt.datetime.now(dt.UTC)
    load_start = os.getloadavg()
    print(f"agent {args.agent_url} ({health.get('parser')}); {len(cases)} cases x {args.repeats}")

    for i in range(args.warmup):
        c = cases[i % len(cases)]
        client.turn(users[i % len(users)], c.message)
    rows = []
    for rep in range(1, args.repeats + 1):
        for c, u in zip(cases, users, strict=False):
            rows.append(record(c, u, rep, client.turn(u, c.message)))
            print(f"\rrepeat {rep}: {len(rows)} replies", end="", file=sys.stderr)
        seen |= others(args.compose_project)
    print(file=sys.stderr)
    load_end = os.getloadavg()

    run = {
        "meta": {
            "started": started.isoformat(timespec="seconds"),
            "finished": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "git": git_rev(),
            "agent_url": args.agent_url,
            "parser": health.get("parser"),
            "cases": len(cases),
            "repeats": args.repeats,
            "warmup": args.warmup,
            "users": f"{len(set(users))} dataset users with >= {MIN_HISTORY} training-period "
            f"interactions (seed {USER_SEED})",
            "campaigns": campaigns(args.api_url),
            "cpus": os.cpu_count(),
            "load_start": [round(x, 2) for x in load_start],
            "load_end": [round(x, 2) for x in load_end],
            "waited_s": round(waited),
            "quiet_start": load_ready < BUSY_LOAD,
            "compose_project": args.compose_project,
            "other_containers": sorted(seen),
        },
        "replies": rows,
    }
    out = paths.ROOT / args.out if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(run, f, indent=1)
    p50 = sorted(r["client_total_ms"] for r in rows)[len(rows) // 2]
    print(f"wrote {out}: {len(rows)} replies, client P50 {p50:.1f} ms")


if __name__ == "__main__":
    main()
