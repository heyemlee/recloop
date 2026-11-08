"""Release drill (design §6, §16 M4): a canary rolled out in full, then a bad release rolled back.

Simulated traffic runs throughout (sim.traffic); every step goes through the api's admin
endpoints, the same ones the admin page's buttons call:

  1. register and validate the new version (both sides load it and replay their golden queries;
     the active version keeps serving);
  2. canary: the release-canary experiment pins 5 % of the users (rank-layer buckets) to it for
     --canary-seconds; then the guardrails, canary arm against the rest over that time:
       failed page views    none on either side
       degraded share       canary at most 1 point above the rest
       client P99           under the 100 ms SLO
       CTR                  not significantly lower (user-level bootstrap, 95 %: the upper end
                            of canary - rest above 0)
     pass: full rollout (activate); fail: abort the canary;
  3. full rollout: --full-seconds with every user on the new version;
  4. a bad release: --bad, built from a weaker SASRec checkpoint, passes validation (it is
     consistent; validation does not judge quality) and is activated without a canary;
  5. after --bad-seconds the CTR of the bad release is compared with the full-rollout phase (the
     numbers go into the rollback's note) and it is rolled back: the previous version is still
     loaded, so the switch back is a pointer switch; --after-seconds more traffic show the
     recovery.

Recorded: every step's wall-clock time and the api's answer, the release record
(model_version_events), the traffic per phase and per 10 s (versions served, failed page
views, P99), and the guardrail decision. artifacts/release/drill.json; `make release-report`
renders docs/results/release.md.

Usage: python -m sim.release_drill --new <version> --bad <version>
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import psycopg

from sim import ads as ads_mod
from sim import redis_stats, standing
from sim import traffic as traffic_mod
from sim.admin import Admin

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "release"
SLO_P99_MS = 100.0


def events_since(t0: float) -> list[dict]:
    url = os.environ.get(
        "DATABASE_URL", "postgres://recloop:recloop@localhost:15432/recloop?sslmode=disable"
    )
    with psycopg.connect(url) as conn:
        rows = conn.execute(
            "SELECT version, from_status, to_status, at, actor, note FROM model_version_events "
            "WHERE at >= to_timestamp(%s) ORDER BY at, event_id",
            (t0,),
        ).fetchall()
    return [
        {"version": v, "from": f, "to": t, "at": a.timestamp(), "actor": ac, "note": n}
        for v, f, t, a, ac, n in rows
    ]


def version_arm(pg: dict) -> str | None:
    """A page view's arm when the arms are the versions that served it."""
    return pg.get("model_version")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--api", default=os.environ.get("RECLOOP_API_URL", "http://127.0.0.1:18080"))
    p.add_argument("--new", help="the version to release (built under artifacts/bundles)")
    p.add_argument("--bad", help="the bad version for the rollback")
    p.add_argument(
        "--analyse",
        action="store_true",
        help="only redo the tables of the drill in --out from its ledger (no traffic, no steps)",
    )
    p.add_argument("--canary-percent", type=int, default=5)
    p.add_argument("--warmup-seconds", type=float, default=60)
    p.add_argument("--canary-seconds", type=float, default=600)
    p.add_argument("--full-seconds", type=float, default=300)
    p.add_argument("--bad-seconds", type=float, default=300)
    p.add_argument("--after-seconds", type=float, default=300)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument(
        "--think-ms",
        type=traffic_mod.think_arg,
        default=traffic_mod.THINK_MS,
        help="min,max pause between a session's page views",
    )
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)
    if args.analyse:
        log = json.loads((args.out / "drill.json").read_text())
        n, e = analyse(log, args.out / "traffic" / "ledger.jsonl")
        log["analysed"] = {"at": time.time(), "by": "sim.release_drill --analyse"}
        _write(args.out, log)
        print(f"drill    re-analysed: {n:,} page views, {e} failed")
        return
    if not (args.new and args.bad):
        p.error("--new and --bad are required (or --analyse)")

    admin = Admin(args.api, actor="release-drill")
    ads = ads_mod.Ads(args.api, os.environ.get("DATABASE_URL", ""))
    campaigns = standing.ensure(ads)
    admin.stop_layer("rank")  # the drill's users are split by the canary only
    before = admin.overview()
    old = before["active"]
    log = {
        "api": args.api,
        "old": old,
        "new": args.new,
        "bad": args.bad,
        "steps": [],
        "args": vars(args) | {"out": str(args.out)},
    }
    log["campaigns"] = len(campaigns)

    def step(name: str, f):
        t = time.time()
        try:
            resp = f()
            ok = True
        except SystemExit as e:
            resp, ok = str(e), False
        log["steps"].append(
            {"step": name, "t": t, "seconds": round(time.time() - t, 3), "ok": ok, "response": resp}
        )
        print(
            f"drill    {time.strftime('%H:%M:%S')} {name}: {'ok' if ok else 'FAILED'} ({time.time() - t:.2f}s)"
        )
        if not ok:
            raise SystemExit(f"{name} failed: {resp}")
        return resp

    total = (
        args.warmup_seconds
        + args.canary_seconds
        + args.full_seconds
        + args.bad_seconds
        + args.after_seconds
        + 600
    )
    redis_before = redis_stats.counters()
    traffic = traffic_mod.Traffic(
        args.out / "traffic", args.api, args.concurrency, think_ms=args.think_ms
    )
    t0 = traffic.start(total)
    log["t0"] = t0
    try:
        time.sleep(args.warmup_seconds)
        for v in (args.new, args.bad):
            step(f"register {v}", lambda v=v: admin.register(v))
        step(f"validate {args.new}", lambda: admin.validate(args.new))
        t_canary = time.time()
        step(
            f"canary {args.new} {args.canary_percent}%",
            lambda: admin.canary(args.new, args.canary_percent),
        )
        time.sleep(args.canary_seconds)
        t_canary_end = time.time()

        # Guardrails over the canary phase (from 15 s after the start: the experiment reload).
        pages, events, errors = _snapshot(traffic)
        lo = t_canary + 15

        def arm(pg: dict) -> str | None:
            v = pg.get("model_version")
            return "canary" if v == args.new else "rest" if v == old else None

        summary = traffic_mod.summarize(pages, events, errors, lo, t_canary_end, arm)
        users = traffic_mod.ctr_by_user(pages, events, lo, t_canary_end, arm)
        diff = traffic_mod.bootstrap_ctr_diff(users.get("rest", {}), users.get("canary", {}))
        c, r = summary["arms"].get("canary"), summary["arms"].get("rest")
        checks = {
            "failed_page_views": summary["failed_page_views"] == 0,
            "degraded": c is not None
            and r is not None
            and c["degraded_share"] <= r["degraded_share"] + 0.01,
            "p99": c is not None and c["p99_ms"] < SLO_P99_MS,
            "ctr_not_lower": diff[2] > 0,
        }
        log["canary"] = {
            "from": lo,
            "to": t_canary_end,
            "summary": summary,
            "ctr_diff": diff,
            "checks": checks,
            "pass": all(checks.values()),
        }
        print(
            f"drill    canary guardrails: {checks}; CTR canary - rest {diff[0]:+.4f} [{diff[1]:+.4f}, {diff[2]:+.4f}]"
        )
        if not all(checks.values()):
            step(
                "abort canary",
                lambda: admin.abort_canary("guardrails failed: " + json.dumps(checks)),
            )
            raise SystemExit("canary failed its guardrails: aborted")
        t_full = time.time()
        step(
            f"full rollout {args.new}",
            lambda: admin.activate(args.new, "canary passed its guardrails"),
        )
        time.sleep(args.full_seconds)
        t_full_end = time.time()

        step(f"validate {args.bad}", lambda: admin.validate(args.bad))
        t_bad = time.time()
        step(
            f"activate {args.bad} (no canary)",
            lambda: admin.activate(args.bad, "bad release: pushed without a canary"),
        )
        time.sleep(args.bad_seconds)
        # The evidence goes into the release record with the decision: the bad release's CTR
        # against the full rollout before it (user-level bootstrap), over what had been served
        # when it was read. The rollback's own time is its step's (reading takes a few seconds).
        t_decide = time.time()
        pages, events, errors = _snapshot(traffic)
        full_users = traffic_mod.ctr_by_user(
            pages, events, t_full + 15, t_full_end, version_arm
        ).get(args.new, {})
        bad_users = traffic_mod.ctr_by_user(pages, events, t_bad + 5, t_decide, version_arm).get(
            args.bad, {}
        )
        bad = traffic_mod.bootstrap_ctr_diff(full_users, bad_users)
        log["bad_vs_full_ctr"] = bad
        log["bad_vs_full_windows"] = [[t_full + 15, t_full_end], [t_bad + 5, t_decide]]
        step(
            "rollback",
            lambda: admin.rollback(
                "drill: the release pushed without a canary goes back; its CTR against the full "
                f"rollout {bad[0]:+.4f} [{bad[1]:+.4f}, {bad[2]:+.4f}]"
            ),
        )
        time.sleep(args.after_seconds)
    finally:
        log["t_end"] = time.time()
        traffic.stop()
        log["redis"] = redis_stats.delta(redis_before, redis_stats.counters())
        print(f"drill    {redis_stats.describe(log['redis'])}")
        # What happened so far, kept even if the drill failed or the analysis below fails (the
        # ledger next to it holds the traffic).
        _write(args.out, log)

    pages, errors = analyse(log, traffic.ledger.f.name)
    log["events"] = events_since(log["t0"] - 1)
    log["after"] = admin.overview()
    _write(args.out, log)
    print(
        f"drill    done: {pages:,} page views, {errors} failed -> {(args.out / 'drill.json').relative_to(ROOT)}"
    )


# Steps that change which version serves, by the phase they start.
PHASE_STEPS = (
    ("canary ", "canary"),
    ("abort canary", "aborted"),
    ("full rollout ", "full"),
    ("activate ", "bad"),
    ("rollback", "after"),
)


def analyse(log: dict, ledger: str | Path) -> tuple[int, int]:
    """The drill's tables, from its recorded steps and the traffic ledger: phases (each from
    the step that started it to the next one: the steps' own times), the switches, the traffic
    per 10 s, CTR by a user's page count, every failed page view with the degradations around
    it. Re-runnable on a finished drill: `python -m sim.release_drill --analyse`."""
    pages, events, errors = traffic_mod.read_ledger(Path(ledger))
    t0, t_end = log["t0"], log["t_end"]
    marks = [("before", t0)]
    for s in log["steps"]:
        for prefix, phase in PHASE_STEPS:
            if s["step"].startswith(prefix) and s["ok"]:
                marks.append((phase, s["t"]))
    bounds = [
        (name, a, b) for (name, a), (_, b) in zip(marks, marks[1:] + [("", t_end)], strict=True)
    ]
    log["phases"] = {
        name: {
            "from": a,
            "to": b,
            **traffic_mod.summarize(pages, events, errors, a, b, version_arm),
        }
        for name, a, b in bounds
    }
    at = dict(marks)
    log["switch"] = {
        key: _switch_time(pages, at[phase], v)
        for key, phase, v in (
            ("full", "full", log["new"]),
            ("bad", "bad", log["bad"]),
            ("rollback", "after", log["new"]),
        )
        if phase in at
    }
    log["timeline"] = traffic_mod.timeline(pages, errors, t0, t_end)
    log["ctr_by_page_count"] = traffic_mod.ctr_by_page_count(pages, events, t0, t_end)
    log["failures"] = [
        {
            "t": e["t"],
            "surface": e.get("surface"),
            "status": e.get("status"),
            "ms": e.get("ms"),
            "degraded_around": traffic_mod.degraded_reasons(pages, e["t"] - 5, e["t"] + 5),
        }
        for e in sorted(errors, key=lambda e: e["t"])
        if t0 <= e["t"] < t_end
    ]
    return len(pages), len(errors)


def _write(out: Path, log: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "drill.json").write_text(json.dumps(log, indent=2, default=str) + "\n")


def _snapshot(traffic: traffic_mod.Traffic):
    return traffic_mod.snapshot(traffic.ledger)


def _switch_time(pages: dict, t: float, version: str) -> dict:
    """After a step at t: seconds until the first page and until the last page of another
    version (requests in flight finish on the version they started with). First pages only: a
    next page keeps the version of the list it pages through."""
    after = sorted(
        (p["t"], p.get("model_version"))
        for p in pages.values()
        if p["t"] >= t and p["surface"] != "next_page"
    )
    first = next((ts for ts, v in after if v == version), None)
    last_other = max((ts for ts, v in after if v != version and ts < t + 120), default=None)
    return {
        "first_page_s": round(first - t, 3) if first is not None else None,
        "last_other_page_s": round(last_other - t, 3) if last_other is not None else None,
    }


if __name__ == "__main__":
    main()
