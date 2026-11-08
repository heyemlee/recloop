"""Check the charge ledger against what the simulator did (docs/design.md §9.3).

The truth comes from the simulator's own ledger: every sponsored slot it was served (it reads
the claims of the ad_token it got, as any client can), every click it sent with a token, and
which of those tokens it forged. However the events were delivered (duplicates, resends, late,
out of order) and whatever happened to billing meanwhile (kill -9, broker restarts, a slow
database), the result must be exactly:

    clicks       one ad_clicks row per sponsored click event; forged tokens invalid, the others
                 valid (expired if the api received the click after the token's exp)
    charges      one charge per sponsored impression with a valid click, at the campaign, UTC
                 day and price of its token, for one of that impression's genuine clicks; no
                 charge for anything else
    amounts      per campaign-day: billed = min(budget, sum of prices); every charge in full
                 except at most one partial one, and nothing billed (0) only once the budget
                 is spent; no budget row above its budget
    agreement    budget row = ledger = Redis spend counter (after reconciliation)

Usage: python -m sim.billing_verify --run artifacts/billing/drill   (after the consumers caught up)
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import psycopg
import redis

from sim import ads as ads_mod
from sim import verify as verify_mod

ROOT = Path(__file__).resolve().parent.parent


def read_run(run: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Sponsored click events, sponsored impression events and pages with sponsored slots."""
    clicks, imps, pages = [], [], []
    with (run / "ledger.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r["kind"] == "event" and r.get("ad_token"):
                (clicks if r["type"] == "click" else imps).append(r)
            elif r["kind"] == "page" and r.get("sponsored"):
                pages.append(r)
    return clicks, imps, pages


def expected_charges(clicks: list[dict], received: dict[str, float]) -> tuple[dict, dict]:
    """impression_id -> expected charge, and event_id -> expected token status. received maps
    an event to the api's receive time (unix seconds)."""
    status, charged = {}, {}
    for c in clicks:
        if c.get("forged"):
            status[c["event_id"]] = "invalid"
            continue
        cl = ads_mod.claims(c["ad_token"])
        expired = received.get(c["event_id"], 0) >= cl["exp"]
        status[c["event_id"]] = "expired" if expired else "valid"
        if expired:
            continue
        imp = f"{cl['rid']}:{cl['pos']}"
        e = charged.setdefault(
            imp,
            {"campaign_id": cl["cid"], "day": ads_mod.utc_day(cl["iat"]), "price": cl["price"],
             "clicks": set()},
        )  # fmt: skip
        e["clicks"].add(c["event_id"])
    return charged, status


def check(name: str, expected, actual, detail: str = "") -> dict:
    return verify_mod.check(name, expected, actual, detail)


def verify(run: Path, database_url: str, redis_url: str) -> dict:
    clicks, imps, pages = read_run(run)
    campaigns = {c["campaign_id"]: c for c in json.loads((run / "campaigns.json").read_text())}
    ids = sorted(campaigns)
    click_ids = [c["event_id"] for c in clicks]
    checks = []

    with psycopg.connect(database_url) as conn:
        received = {
            str(e): ts.timestamp()
            for e, ts in conn.execute(
                "SELECT event_id, server_ts FROM events_raw WHERE type = 'click' AND event_id = ANY(%s::uuid[])",
                (click_ids,),
            )
        }
        charged, want_status = expected_charges(clicks, received)
        got_status = {
            str(e): s
            for e, s in conn.execute(
                "SELECT event_id, token_status FROM ad_clicks WHERE event_id = ANY(%s::uuid[])",
                (click_ids,),
            )
        }
        stray_clicks = conn.execute(
            "SELECT count(*) FROM ad_clicks WHERE campaign_id = ANY(%s) AND NOT event_id = ANY(%s::uuid[])",
            (ids, click_ids),
        ).fetchone()[0]
        charges = {
            imp: {"click": str(ev), "campaign_id": cid, "day": day.isoformat(), "price": price,
                  "billed": billed}
            for imp, ev, cid, day, price, billed in conn.execute(
                """SELECT impression_id, click_event_id, campaign_id, day, price_cents, billed_cents
                   FROM ad_charges WHERE campaign_id = ANY(%s)""",
                (ids,),
            )
        }  # fmt: skip
        rows = {
            (cid, day.isoformat()): (budget, spent)
            for cid, day, budget, spent in conn.execute(
                "SELECT campaign_id, day, budget_cents, spent_cents FROM campaign_budget_daily WHERE campaign_id = ANY(%s)",
                (ids,),
            )
        }

    # Clicks: each processed once, with the status the truth gives it.
    checks.append(
        check(
            "ad_clicks: one row per sponsored click event",
            len(click_ids),
            len(got_status),
            f"missing {[e for e in click_ids if e not in got_status][:3]}",
        )
    )
    checks.append(check("ad_clicks: rows for clicks the simulator did not send", 0, stray_clicks))
    forged = [c["event_id"] for c in clicks if c.get("forged")]
    checks.append(
        check(
            "forged tokens refused (invalid)",
            len(forged),
            sum(got_status.get(e) == "invalid" for e in forged),
        )
    )
    genuine = [c["event_id"] for c in clicks if not c.get("forged")]
    wrong = [e for e in genuine if got_status.get(e) != want_status[e]]
    checks.append(
        check(
            "genuine tokens verified (valid, or expired after exp)",
            len(genuine),
            len(genuine) - len(wrong),
            str([(e, want_status[e], got_status.get(e)) for e in wrong[:3]]),
        )
    )

    # Charges, one by one.
    missing = [i for i in charged if i not in charges]
    unexpected = [i for i in charges if i not in charged]
    mismatched, foreign = [], []
    for imp, want in charged.items():
        got = charges.get(imp)
        if got is None:
            continue
        if (got["campaign_id"], got["day"], got["price"]) != (
            want["campaign_id"],
            want["day"],
            want["price"],
        ):
            mismatched.append((imp, want, got))
        if got["click"] not in want["clicks"]:
            foreign.append(imp)
    checks.append(
        check("charges: clicked impressions with no charge", 0, len(missing), str(missing[:3]))
    )
    checks.append(
        check(
            "charges: charges with no clicked impression", 0, len(unexpected), str(unexpected[:3])
        )
    )
    checks.append(
        check(
            "charges: campaign, day and price of the token",
            0,
            len(mismatched),
            str(mismatched[:2]),
        )
    )
    checks.append(
        check(
            "charges: billed click is a genuine click of the impression",
            0,
            len(foreign),
            str(foreign[:3]),
        )
    )

    # Amounts per campaign-day.
    by_day: dict[tuple, list] = defaultdict(list)
    for c in charges.values():
        by_day[(c["campaign_id"], c["day"])].append(c)
    # The budget is the advertiser's setting (campaigns.json; the run never changes it), not the
    # budget row billing wrote: a row created with the wrong budget must fail here.
    days, bad_sum, over, bad_row, bad_order, bad_budget = [], [], [], [], [], []
    for key in sorted(set(by_day) | set(rows)):
        cs = by_day.get(key, [])
        budget = campaigns[key[0]]["daily_budget_cents"]
        row_budget, spent = rows.get(key, (budget, 0))
        if row_budget != budget:
            bad_budget.append((key, budget, row_budget))
        prices = sum(c["price"] for c in cs)
        billed = sum(c["billed"] for c in cs)
        partial = sum(0 < c["billed"] < c["price"] for c in cs)
        short = any(c["billed"] < c["price"] for c in cs)
        days.append(
            {"campaign_id": key[0], "day": key[1], "budget": budget, "spent": spent,
             "charges": len(cs), "price": prices, "billed": billed,
             "zero_billed": sum(c["billed"] == 0 for c in cs), "partial": partial}
        )  # fmt: skip
        if billed != min(budget, prices):
            bad_sum.append((key, budget, prices, billed))
        if billed > budget or spent > budget or any(c["billed"] > c["price"] for c in cs):
            over.append(key)
        if spent != billed:
            bad_row.append((key, spent, billed))
        if partial > 1 or (short and billed != budget):
            bad_order.append(key)
    checks.append(
        check(
            "amounts: billed = min(budget, sum of prices) per campaign-day",
            0,
            len(bad_sum),
            str(bad_sum[:3]),
        )
    )
    checks.append(check("amounts: charges above budget", 0, len(over), str(over[:3])))
    checks.append(
        check(
            "amounts: one partial charge at most, zero only once the budget is spent",
            0,
            len(bad_order),
            str(bad_order[:3]),
        )
    )
    checks.append(check("budget rows = ledger", 0, len(bad_row), str(bad_row[:3])))
    checks.append(
        check(
            "budget rows: the campaign's daily budget",
            0,
            len(bad_budget),
            str(bad_budget[:3]),
        )
    )

    # Redis: the counters ad serving reads, after reconciliation.
    r = redis.Redis.from_url(redis_url)
    keys = [f"spend:{d['campaign_id']}:{d['day'].replace('-', '')}" for d in days]
    vals = r.mget(keys) if keys else []
    bad_redis = [
        (k, int(v or 0), d["billed"])
        for k, v, d in zip(keys, vals, days, strict=True)
        if int(v or 0) != d["billed"]
    ]
    for d, v in zip(days, vals, strict=True):
        d["redis"] = int(v or 0)
    checks.append(check("Redis spend = ledger", 0, len(bad_redis), str(bad_redis[:3])))

    slots = [s for p in pages for s in p["sponsored"]]
    valid = sum(s == "valid" for s in got_status.values())
    return {
        "run": str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run),
        "campaigns": len(ids),
        "slots_served": len(slots),
        "sponsored_impressions": len(imps),
        "sponsored_click_events": len(clicks),
        "forged_click_events": len(forged),
        "valid_clicks": valid,
        "expired_clicks": sum(s == "expired" for s in got_status.values()),
        "charges_expected": len(charged),
        "charges": len(charges),
        "repeat_clicks": valid - len(charges),  # valid clicks on an impression already charged
        "billed_cents": sum(c["billed"] for c in charges.values()),
        "price_cents": sum(c["price"] for c in charges.values()),
        "unbilled_cents": sum(c["price"] - c["billed"] for c in charges.values()),
        "zero_billed": sum(c["billed"] == 0 for c in charges.values()),
        "budgets_exhausted": sum(d["billed"] == d["budget"] for d in days),
        "days": days,
        "freq": frequency(imps),
        "checks": checks,
        "ok": all(c["ok"] for c in checks),
    }


def frequency(imps: list[dict]) -> dict:
    """Sponsored impressions per user, campaign and serving day (the frequency cap counts them)."""
    n: dict[tuple, int] = defaultdict(int)
    for e in imps:
        cl = ads_mod.claims(e["ad_token"])
        n[(e["user_id"], cl["cid"], ads_mod.utc_day(cl["iat"]))] += 1
    counts = sorted(n.values())
    return {
        "user_campaign_days": len(counts),
        "max": counts[-1] if counts else 0,
        "histogram": {str(k): counts.count(k) for k in sorted(set(counts))},
    }


def print_report(rep: dict) -> None:
    print(
        f"billing  {rep['sponsored_click_events']:,} sponsored click events "
        f"({rep['forged_click_events']} forged) -> {rep['charges']:,} charges, "
        f"{rep['billed_cents'] / 100:,.2f} $ billed, {rep['unbilled_cents'] / 100:,.2f} $ not billed"
    )
    for c in rep["checks"]:
        mark = "ok  " if c["ok"] else "FAIL"
        extra = "" if c["ok"] or not c["detail"] else f"   ({c['detail']})"
        print(f"  {mark} {c['check']}: expected {c['expected']}, got {c['actual']}{extra}")
    print("  all checks passed" if rep["ok"] else "  SOME CHECKS FAILED")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", type=Path, default=ROOT / "artifacts" / "billing" / "drill")
    args = p.parse_args(argv)
    rep = verify(
        args.run.resolve(),
        os.environ.get(
            "DATABASE_URL", "postgres://recloop:recloop@localhost:15432/recloop?sslmode=disable"
        ),
        os.environ.get("REDIS_URL", "redis://localhost:16390/0"),
    )
    (args.run / "billing_verify.json").write_text(json.dumps(rep, indent=2))
    print_report(rep)
    raise SystemExit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
