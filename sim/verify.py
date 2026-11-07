"""Check what the event pipeline made of a simulator run (docs/design.md §7.3).

The ledger lists every unique event the simulator created; the fault switches delivered some of
them several times, late or out of order, and consumers may have crashed in between. With
at-least-once delivery and idempotent consumers the result must be as if each event had arrived
exactly once:

    events_raw   every ledger event is there exactly once, no extra rows for the run's users
    served       every page the simulator was served has its served row
    orders       one order per purchase event; stock went down by one per purchase
    Redis        hv went up by exactly the number of click / add-to-cart / purchase events;
                 every event has its dedup marker; purchases are in `bought`; each user's last
                 clicked item is in the history

Usage: python -m sim.verify --run artifacts/sim/run   (after the consumers caught up)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import psycopg
import redis

ROOT = Path(__file__).resolve().parent.parent
AGGREGATOR = ROOT / ".cache" / "bin" / "aggregator"
HISTORY_TYPES = {"click", "add_to_cart", "purchase"}


def read_ledger(path: Path) -> tuple[dict[str, dict], list[dict]]:
    events, pages = {}, []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if r["kind"] == "event":
                events[r["event_id"]] = r
            elif r["kind"] == "page":
                pages.append(r)
    return events, pages


def wait_for_consumers(groups=("log-sink", "aggregator")) -> None:
    """Block until each consumer group has committed everything produced so far."""
    for g in groups:
        subprocess.run([str(AGGREGATOR), "wait-lag", g], check=True, stdout=subprocess.DEVNULL)


def check(name: str, expected, actual, detail: str = "") -> dict:
    return {
        "check": name,
        "expected": expected,
        "actual": actual,
        "ok": expected == actual,
        "detail": detail,
    }


def verify(run: Path, database_url: str, redis_url: str) -> dict:
    events, pages = read_ledger(run / "ledger.jsonl")
    before = json.loads((run / "before.json").read_text())
    users = sorted({e["user_id"] for e in events.values()} | {p["user_id"] for p in pages})
    checks = []

    with psycopg.connect(database_url) as conn:
        conn.execute("CREATE TEMP TABLE ledger_events (event_id uuid PRIMARY KEY)")
        with conn.cursor().copy("COPY ledger_events (event_id) FROM STDIN") as cp:
            for eid in events:
                cp.write_row((eid,))
        rows = conn.execute(
            """SELECT r.event_id::text, count(*) FROM events_raw AS r
               JOIN ledger_events USING (event_id) GROUP BY 1"""
        ).fetchall()
        stored = dict(rows)
        missing = [e for e in events if e not in stored]
        repeated = {e: n for e, n in stored.items() if n > 1}
        checks.append(
            check(
                "events_raw: ledger events stored",
                len(events),
                len(stored),
                f"missing {missing[:3]}",
            )
        )
        checks.append(
            check("events_raw: events stored twice", 0, len(repeated), str(list(repeated)[:3]))
        )
        # No client event of the run's users and time window that the simulator did not create.
        started = json.loads((run / "summary.json").read_text())["started"]
        extra = conn.execute(
            """SELECT count(*) FROM events_raw AS r
               WHERE r.type <> 'served' AND r.user_id = ANY(%s) AND r.server_ts >= to_timestamp(%s)
                 AND NOT EXISTS (SELECT 1 FROM ledger_events AS l WHERE l.event_id = r.event_id)""",
            (users, started),
        ).fetchone()[0]
        checks.append(check("events_raw: rows the simulator did not send", 0, extra))

        want_pages = Counter(p["request_id"] for p in pages)
        conn.execute("CREATE TEMP TABLE ledger_pages (request_id uuid PRIMARY KEY, n int)")
        with conn.cursor().copy("COPY ledger_pages (request_id, n) FROM STDIN") as cp:
            for rid, n in want_pages.items():
                cp.write_row((rid, n))
        served = dict(
            conn.execute(
                """SELECT p.request_id::text, count(r.event_id) FROM ledger_pages AS p
                   LEFT JOIN events_raw AS r ON r.request_id = p.request_id AND r.type = 'served'
                   GROUP BY 1"""
            ).fetchall()
        )
        short = {r: (n, served.get(r, 0)) for r, n in want_pages.items() if served.get(r, 0) < n}
        over = {r: (n, served.get(r, 0)) for r, n in want_pages.items() if served.get(r, 0) > n}
        checks.append(
            check(
                "served: pages with their served row",
                sum(want_pages.values()),
                sum(min(served.get(r, 0), n) for r, n in want_pages.items()),
                f"{len(short)} requests short",
            )
        )
        checks.append(check("served: extra served rows", 0, sum(s - n for n, s in over.values())))

        purchases = [e for e in events.values() if e["type"] == "purchase"]
        orders = conn.execute(
            "SELECT count(*), count(DISTINCT o.event_id) FROM orders AS o JOIN ledger_events USING (event_id)"
        ).fetchone()
        checks.append(
            check("orders: one per purchase event", [len(purchases), len(purchases)], list(orders))
        )
        bought_per_item = Counter(e["item_id"] for e in purchases)
        stock_after = dict(conn.execute("SELECT item_id, stock FROM items").fetchall())
        wrong_stock = [
            i
            for i, n in bought_per_item.items()
            if stock_after[i] != max(0, before["stock"][str(i)] - n)
        ]
        checks.append(
            check("stock: one unit per purchase", 0, len(wrong_stock), str(wrong_stock[:5]))
        )

    r = redis.Redis.from_url(redis_url)
    per_user = defaultdict(list)
    for e in sorted(events.values(), key=lambda e: e["event_id"]):  # UUIDv7: time order
        per_user[e["user_id"]].append(e)
    pipe = r.pipeline(transaction=False)
    for u in users:
        pipe.get(f"u:{u}:hv")
        pipe.smembers(f"u:{u}:bought")
        pipe.zrange(f"u:{u}:hist", 0, -1)
    res = pipe.execute()
    bad_hv, bad_bought, bad_hist = [], [], []
    for k, u in enumerate(users):
        hv, bought, hist = res[3 * k], res[3 * k + 1], res[3 * k + 2]
        evs = per_user.get(u, [])
        want = before["hv"].get(str(u), 0) + sum(e["type"] in HISTORY_TYPES for e in evs)
        if int(hv or 0) != want:
            bad_hv.append((u, want, int(hv or 0)))
        got_bought = {int(x) for x in bought}
        if any(e["item_id"] not in got_bought for e in evs if e["type"] == "purchase"):
            bad_bought.append(u)
        clicks = [e for e in evs if e["type"] in HISTORY_TYPES]
        if clicks and str(clicks[-1]["item_id"]).encode() not in hist:
            bad_hist.append(u)
    checks.append(
        check("redis: hv = before + history events (users)", 0, len(bad_hv), str(bad_hv[:3]))
    )
    checks.append(
        check("redis: purchases in bought (users)", 0, len(bad_bought), str(bad_bought[:3]))
    )
    checks.append(
        check("redis: last clicked item in history (users)", 0, len(bad_hist), str(bad_hist[:3]))
    )
    pipe = r.pipeline(transaction=False)
    for eid in events:
        pipe.exists(f"dedup:aggregator:{eid}")
    markers = sum(pipe.execute())
    checks.append(check("redis: dedup marker per event", len(events), markers))

    return {
        "run": str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run),
        "users": len(users),
        "events": len(events),
        "events_by_type": dict(Counter(e["type"] for e in events.values())),
        "pages": len(pages),
        "checks": checks,
        "ok": all(c["ok"] for c in checks),
    }


def print_report(rep: dict) -> None:
    print(f"verify   {rep['events']:,} events of {rep['users']:,} users, {rep['pages']:,} pages")
    for c in rep["checks"]:
        mark = "ok  " if c["ok"] else "FAIL"
        extra = "" if c["ok"] or not c["detail"] else f"   ({c['detail']})"
        print(f"  {mark} {c['check']}: expected {c['expected']}, got {c['actual']}{extra}")
    print("  all checks passed" if rep["ok"] else "  SOME CHECKS FAILED")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", type=Path, default=ROOT / "artifacts" / "sim" / "run")
    p.add_argument("--no-wait", action="store_true", help="do not wait for consumer lag 0")
    args = p.parse_args(argv)
    if not args.no_wait:
        wait_for_consumers()
    rep = verify(
        args.run.resolve(),
        os.environ.get(
            "DATABASE_URL", "postgres://recloop:recloop@localhost:15432/recloop?sslmode=disable"
        ),
        os.environ.get("REDIS_URL", "redis://localhost:16390/0"),
    )
    (args.run / "verify.json").write_text(json.dumps(rep, indent=2))
    print_report(rep)
    raise SystemExit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
