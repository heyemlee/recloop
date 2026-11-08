"""Billing chaos drill (docs/design.md §9.3): simulated traffic for 30 minutes while billing is
killed with SIGKILL at random moments, Redpanda restarts and PostgreSQL answers billing 200 ms
late; then the charge ledger is checked against the simulator's truth, charge by charge.

Setup: the run creates its own campaigns through the advertiser api (one advertiser each,
different bids and budgets: some run out during the drill, some do not) and serves them with a
safety margin of 0 clicks (an ads-layer experiment), the setting that lets the most clicks
arrive after a budget ran out. Every delivery fault of the simulator is on, plus forged tokens.

Chaos (random times from --seed, written to chaos.json):
    billing     kill -9, started again 2-10 s later (8 times)
    redpanda    restarted (3 times; graceful: the dev-container broker skips fsync, so a kill -9
                could lose acknowledged writes, a broker setting and not a pipeline property)
    postgres    +200 ms on every reply to billing, through Toxiproxy (pg-proxy), for 3 minutes
                (twice)

When the traffic stops: wait until billing, the log-sink and the aggregator have consumed
everything, run one reconciliation pass, then sim.billing_verify (the ledger) and sim.verify
(the event pipeline, as in make drill).

Usage: python -m sim.billing_drill [--minutes 30]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from sim import ads as ads_mod
from sim import billing_verify
from sim import run as run_mod
from sim import verify as verify_mod

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / ".cache" / "bin"
FAULTS = "duplicate=0.1,resend=0.05,reorder=0.1,late=0.05,double_click=0.05,forge=0.02"
EXP_ID = "ads-billing-drill"
# (bid, daily budget) in cents. The auction ranks by bid (the stub pCTR is the same for every
# ad): the first campaigns run out of budget during the drill, the last ones never.
CAMPAIGNS = [(60, 1500), (50, 3000), (40, 4000), (30, 2500), (20, 50000), (10, 50000)]
ITEMS_PER_CAMPAIGN = 10
# Impressions per user and campaign per day: high enough that the ~2,000 simulated users keep
# clicking for 30 minutes, low enough that some of them reach it (the cap is exercised too).
FREQ_CAP = 20


def compose_cmd() -> list[str]:
    """docker compose on the repo's file; the project and ports come from the environment
    (COMPOSE_PROJECT_NAME) and, if there is one, the repo's .env."""
    cmd = ["docker", "compose", "-f", str(ROOT / "deploy" / "docker-compose.yml")]
    if (ROOT / ".env").exists():
        cmd += ["--env-file", str(ROOT / ".env")]
    return cmd


def compose(*args: str) -> None:
    subprocess.run([*compose_cmd(), *args], check=True, capture_output=True)


def toxiproxy(method: str, path: str, doc: dict | None = None) -> None:
    url = os.environ.get("PG_PROXY_URL", "http://localhost:18475") + path
    data = json.dumps(doc).encode() if doc is not None else None
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        r.read()


def pg_slow(on: bool, ms: int) -> None:
    if on:
        toxiproxy("POST", "/proxies/postgres/toxics",
                  {"name": "pg_latency", "type": "latency", "stream": "downstream",
                   "attributes": {"latency": ms, "jitter": 0}})  # fmt: skip
    else:
        toxiproxy("DELETE", "/proxies/postgres/toxics/pg_latency")


def plan(seconds: float, seed: int) -> list[dict]:
    """The chaos schedule: start times (s from the start of the traffic) and actions."""
    rng = random.Random(seed)

    def spaced(n: int, lo: float, hi: float, gap: float) -> list[float]:
        for _ in range(10_000):
            ts = sorted(rng.uniform(lo, hi) for _ in range(n))
            if all(b - a >= gap for a, b in zip(ts, ts[1:], strict=False)):
                return ts
        raise SystemExit("cannot place the chaos events: the drill is too short")

    s = seconds
    steps = [{"t": t, "action": "kill_billing", "down": round(rng.uniform(2, 10), 1)}
             for t in spaced(8, 0.03 * s, 0.95 * s, 0.06 * s)]  # fmt: skip
    steps += [
        {"t": t, "action": "restart_redpanda"} for t in spaced(3, 0.07 * s, 0.92 * s, 0.17 * s)
    ]
    slow = min(180.0, 0.1 * s)
    steps += [{"t": t, "action": "pg_slow", "seconds": slow}
              for t in spaced(2, 0.1 * s, 0.85 * s - slow, slow + 0.05 * s)]  # fmt: skip
    for st in steps:
        st["t"] = round(st["t"], 1)
    return sorted(steps, key=lambda x: x["t"])


def chaos(steps: list[dict], pg_ms: int, log: list, stop: threading.Event) -> None:
    t0 = time.time()

    def note(what: str) -> None:
        log.append({"t": round(time.time() - t0, 1), "action": what})
        print(f"  t={log[-1]['t']:7.1f}s  {what}", flush=True)

    timers = []
    for st in steps:
        if stop.wait(max(0.0, t0 + st["t"] - time.time())):
            break
        if st["action"] == "kill_billing":
            compose("kill", "-s", "SIGKILL", "billing")
            note("kill -9 billing")
            time.sleep(st["down"])
            compose("start", "billing")
            note(f"started billing ({st['down']} s down)")
        elif st["action"] == "restart_redpanda":
            compose("restart", "redpanda")
            note("restarted redpanda")
        elif st["action"] == "pg_slow":
            pg_slow(True, pg_ms)
            note(f"postgres +{pg_ms} ms for billing")

            def back(seconds=st["seconds"]) -> None:
                pg_slow(False, pg_ms)
                note(f"postgres back to normal ({seconds:.0f} s)")

            tm = threading.Timer(st["seconds"], back)
            tm.start()
            timers.append(tm)
    for tm in timers:
        tm.join()


def reconcile() -> dict:
    """One reconciliation pass (billing reconcile): the ledger, the budget rows and Redis."""
    p = subprocess.run([str(BIN / "billing"), "reconcile"], capture_output=True, text=True)
    # Exit 1 = the pass found a disagreement: its report (stdout) says which; the log is stderr.
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        raise SystemExit(f"billing reconcile failed ({p.returncode}): {p.stderr}") from None


def repairs_during(since: float) -> dict:
    """What billing's own reconciler (every minute) repaired since `since`, from its log: the
    passes that found a Redis counter behind the ledger."""
    p = subprocess.run(
        [
            *compose_cmd(),
            "logs",
            "--no-color",
            "--no-log-prefix",
            "--since",
            str(int(since)),
            "billing",
        ],
        capture_output=True,
        text=True,
    )
    passes, cents, starts = 0, 0, 0
    for line in p.stdout.splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("msg") == "reconciled":
            passes += 1
            cents += r.get("repaired_cents", 0)
        starts += r.get("msg") == "billing consuming"
    return {"passes_with_repairs": passes, "repaired_cents": cents, "billing_starts": starts}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--minutes", type=float, default=30)
    # 8 sessions at a time, ~40 page views/s: 30 minutes of result pages (5 KB each, kept 10
    # minutes) and dedup markers (one per event, 7 days) stay well under Redis's 384 MB, so
    # nothing is evicted (at 12, ~64/s, the dev Redis filled up and evicted keys).
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--think-ms", default="50,300")
    p.add_argument("--faults", default=FAULTS)
    p.add_argument("--pg-latency-ms", type=int, default=200)
    p.add_argument("--seed", type=int, default=20261107)
    p.add_argument("--out", default=str(ROOT / "artifacts" / "billing" / "drill"))
    args = p.parse_args(argv)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    api_url = os.environ.get("RECLOOP_API_URL", "http://127.0.0.1:18080")
    db_url = os.environ.get(
        "DATABASE_URL", "postgres://recloop:recloop@localhost:15432/recloop?sslmode=disable"
    )
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:16390/0")
    seconds = args.minutes * 60

    ads = ads_mod.Ads(api_url, db_url)
    ads.require_no_active_campaigns()
    cache = ads.candidate_cache()
    # The seeded user state, and Redis far from its memory cap: an evicted spend counter, dedup
    # marker or frequency counter would change the run (the run fails if any key is evicted).
    ads_mod.reset_redis()
    evicted = ads_mod.evicted_keys()
    # Toxiproxy reachable, and no toxic left from an earlier run.
    with contextlib.suppress(urllib.error.HTTPError):  # 404: none
        pg_slow(False, 0)
    ads.set_policy(EXP_ID, margin=0, pacing=False)
    items = ads.pick_items(len(CAMPAIGNS) * ITEMS_PER_CAMPAIGN, args.seed)
    specs = [
        {
            "bid": bid,
            "budget": budget,
            "freq_cap": FREQ_CAP,
            "items": items[i * ITEMS_PER_CAMPAIGN : (i + 1) * ITEMS_PER_CAMPAIGN],
        }
        for i, (bid, budget) in enumerate(CAMPAIGNS)
    ]
    camps = ads.create("Billing drill", specs)
    (out / "campaigns.json").write_text(json.dumps(camps, indent=2))
    ids = [c["campaign_id"] for c in camps]
    steps = plan(seconds, args.seed)
    (out / "chaos_plan.json").write_text(json.dumps(steps, indent=2))
    print(f"billing drill  {args.minutes:.0f} min, campaigns {ids}, margin 0, faults {args.faults}")
    try:
        ads.wait_campaigns(ids, user=1)
        ads.wait_policy(0, False, user=1)
        sim_args = run_mod.parser().parse_args(
            ["--sessions", "100000000", "--max-seconds", str(seconds),
             "--concurrency", str(args.concurrency), "--think-ms", args.think_ms,
             "--faults", args.faults, "--seed", str(args.seed), "--out", str(out), "--api", api_url]
        )  # fmt: skip
        log: list = []
        stop = threading.Event()
        th = threading.Thread(target=chaos, args=(steps, args.pg_latency_ms, log, stop))
        started = time.time()
        summary = run_mod.run_main(sim_args, on_start=th.start)
        stop.set()
        th.join()
    finally:
        with contextlib.suppress(urllib.error.HTTPError):  # 404: no toxic left
            pg_slow(False, 0)
        compose("start", "billing")  # in case the traffic ended while it was down
        ads.end(ids)
        ads.stop_policy(EXP_ID)
        ads.close()

    print("  waiting for billing, the log-sink and the aggregator to catch up ...", flush=True)
    t = time.time()
    verify_mod.wait_for_consumers(("billing", "log-sink", "aggregator"))
    caught_up = time.time() - t
    evicted = ads_mod.evicted_keys() - evicted
    during = repairs_during(started)
    rec = reconcile()
    rep = billing_verify.verify(out, db_url, redis_url)
    pipeline = verify_mod.verify(out, db_url, redis_url)
    rep.update({
        "minutes": args.minutes, "seed": args.seed, "faults": args.faults,
        "pg_latency_ms": args.pg_latency_ms, "concurrency": args.concurrency,
        "think_ms": args.think_ms, "chaos": log, "chaos_plan": steps, "sim": summary,
        "freq_cap": FREQ_CAP, "safety_margin_clicks": 0,
        "campaign_settings": [{"campaign_id": c["campaign_id"], "bid": c["bid_cpc_cents"],
                               "budget": c["daily_budget_cents"]} for c in camps],
        "caught_up_seconds": round(caught_up, 1), "reconcile": rec, "reconciler_during": during,
        "evicted_keys": evicted, "candidate_cache": cache,
        "pipeline": {k: pipeline[k] for k in ("events", "pages", "checks", "ok")},
    })  # fmt: skip
    rep["ok"] = (
        rep["ok"]
        and pipeline["ok"]
        and rec["redis_ahead"] == 0
        and rec["row_mismatch"] == 0
        and evicted == 0
    )
    (out / "verify.json").write_text(json.dumps(rep, indent=2))
    billing_verify.print_report(rep)
    verify_mod.print_report(pipeline)
    raise SystemExit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
