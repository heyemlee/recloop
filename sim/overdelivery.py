"""Overdelivery study (docs/design.md §9.4): safety margins of 0, 5 and 20 clicks, each with and
without pacing, on the same simulated traffic.

Overdelivery: a slot served while the campaign still had budget is clicked after the budget ran
out; billing charges what is left (often 0) and the platform absorbs the rest. It can only be
made smaller: ad serving reads the spend in Redis and stops a campaign once its budget left is
at most `margin` clicks at its bid; pacing spreads the budget over the campaign's flight
window, so the last part of the budget is spent slowly.

Each configuration gets fresh campaigns (the same bids, budgets and items every time) with a
flight window of --window seconds, and the simulator runs over the window plus --tail seconds
for the clicks still on their way. Users stay on a page for --dwell-ms before acting, so a click
reaches billing seconds after its slot was served. The policy is set through an ads-layer
experiment, which the api reloads every 10 s.

Measured per configuration, from the ledger (ad_charges, ad_clicks) and the simulator's pages:
    overdelivery      sum of (price - billed) / sum of budgets
    spent             sum of billed / sum of budgets
    budget ran out    when the last billed click of a campaign that spent its whole budget
                      arrived, as a share of the window
    serving stopped   when the campaign's last sponsored slot was served, as a share of the window

Usage: python -m sim.overdelivery [--configs 0,0p,5,5p,20,20p]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import psycopg

from sim import ads as ads_mod
from sim import run as run_mod
from sim import verify as verify_mod

ROOT = Path(__file__).resolve().parent.parent
EXP_ID = "ads-overdelivery"
# (bid, daily budget) in cents, one advertiser each. With the stub pCTR the auction ranks by
# bid, so the first campaign is served the most and runs out first. The budgets are 50 bids
# each, so a margin of 20 clicks holds back 40 % of every budget and one of 5 clicks 10 %; with
# the default traffic, unpaced and with no margin, the first two run out inside the window.
CAMPAIGNS = [(50, 2500), (40, 2000), (30, 1500), (20, 1000)]
ITEMS_PER_CAMPAIGN = 10
CONFIGS = ["0", "0p", "5", "5p", "20", "20p"]


def parse_config(c: str) -> tuple[int, bool]:
    return int(c.rstrip("p")), c.endswith("p")


def measure(conn, ids: list[int], t0: float, window: float, pages: list[dict]) -> list[dict]:
    """Per campaign: budget, spend, overdelivery, when the budget ran out, when serving stopped."""
    charges = defaultdict(list)
    for cid, price, billed, at in conn.execute(
        """SELECT ch.campaign_id, ch.price_cents, ch.billed_cents, k.received_at
           FROM ad_charges AS ch JOIN ad_clicks AS k ON k.event_id = ch.click_event_id
           WHERE ch.campaign_id = ANY(%s)""",
        (ids,),
    ):
        charges[cid].append((price, billed, at.timestamp()))
    budgets = dict(
        conn.execute(
            "SELECT campaign_id, daily_budget_cents FROM campaigns WHERE campaign_id = ANY(%s)",
            (ids,),
        ).fetchall()
    )
    served = defaultdict(list)
    for p in pages:
        for s in p.get("sponsored", []):
            served[s["campaign_id"]].append(p["t"])
    out = []
    for cid in ids:
        cs = charges.get(cid, [])
        budget = budgets[cid]
        billed = sum(b for _, b, _ in cs)
        paid = [at for _, b, at in cs if b > 0]
        ran_out = (max(paid) - t0) / window if billed == budget and paid else None
        slots = served.get(cid, [])
        curve = {}
        for q in (0.25, 0.5, 0.75, 1.0):
            curve[str(q)] = sum(b for _, b, at in cs if at <= t0 + q * window) / budget
        out.append({
            "campaign_id": cid, "budget": budget, "clicks": len(cs),
            "price": sum(p for p, _, _ in cs), "billed": billed,
            "unbilled": sum(p - b for p, b, _ in cs), "zero_billed": sum(b == 0 for _, b, _ in cs),
            "slots_served": len(slots),
            "ran_out_at": None if ran_out is None else round(ran_out, 3),
            "stopped_at": round((max(slots) - t0) / window, 3) if slots else None,
            "spend_curve": {k: round(v, 3) for k, v in curve.items()},
        })  # fmt: skip
    return out


def summarize(camps: list[dict]) -> dict:
    budget = sum(c["budget"] for c in camps)
    ran_out = [c["ran_out_at"] for c in camps if c["ran_out_at"] is not None]
    stopped = [c["stopped_at"] for c in camps if c["stopped_at"] is not None]
    return {
        "budget": budget,
        "billed": sum(c["billed"] for c in camps),
        "unbilled": sum(c["unbilled"] for c in camps),
        "overdelivery": round(sum(c["unbilled"] for c in camps) / budget, 4),
        "spent": round(sum(c["billed"] for c in camps) / budget, 4),
        "clicks": sum(c["clicks"] for c in camps),
        "zero_billed": sum(c["zero_billed"] for c in camps),
        "exhausted": len(ran_out),
        "campaigns": len(camps),
        "ran_out_median": round(statistics.median(ran_out), 3) if ran_out else None,
        "stopped_median": round(statistics.median(stopped), 3) if stopped else None,
    }


def read_pages(ledger: Path) -> list[dict]:
    with ledger.open() as f:
        return [r for r in map(json.loads, f) if r["kind"] == "page" and r.get("sponsored")]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--configs", default=",".join(CONFIGS), help="margins in clicks, p = paced")
    p.add_argument("--window", type=float, default=240, help="flight window (s)")
    p.add_argument("--tail", type=float, default=30, help="traffic after the window (s)")
    p.add_argument("--lead", type=float, default=15, help="flight starts this long after setup (s)")
    p.add_argument("--concurrency", type=int, default=400)
    p.add_argument("--dwell-ms", default="1000,9000")
    p.add_argument("--think-ms", default="0,0")
    p.add_argument("--seed", type=int, default=20261107)
    p.add_argument("--out", default=str(ROOT / "artifacts" / "billing" / "overdelivery"))
    args = p.parse_args(argv)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    api_url = os.environ.get("RECLOOP_API_URL", "http://127.0.0.1:18080")
    db_url = os.environ.get(
        "DATABASE_URL", "postgres://recloop:recloop@localhost:15432/recloop?sslmode=disable"
    )

    ads = ads_mod.Ads(api_url, db_url)
    ads.require_no_active_campaigns()
    cache = ads.candidate_cache()
    items = ads.pick_items(len(CAMPAIGNS) * ITEMS_PER_CAMPAIGN, args.seed)
    results = []
    try:
        for name in args.configs.split(","):
            margin, paced = parse_config(name)
            # Every configuration starts from the same user state, with Redis far from its
            # memory cap (an evicted spend counter would corrupt the measurement).
            ads_mod.reset_redis()
            ads.set_policy(EXP_ID, margin, paced)
            ads.wait_policy(margin, paced, user=1)
            t0 = time.time() + args.lead
            specs = [
                {"bid": bid, "budget": budget, "freq_cap": 100, "start": t0, "end": t0 + args.window,
                 "items": items[i * ITEMS_PER_CAMPAIGN : (i + 1) * ITEMS_PER_CAMPAIGN]}
                for i, (bid, budget) in enumerate(CAMPAIGNS)
            ]  # fmt: skip
            camps = ads.create(f"Overdelivery {name}", specs)
            ids = [c["campaign_id"] for c in camps]
            ads.wait_campaigns(ids, user=1)
            run_dir = out / f"m{name}"
            seconds = t0 + args.window + args.tail - time.time()
            print(f"overdelivery  margin {margin}, pacing {'on' if paced else 'off'}: campaigns {ids}, "
                  f"{seconds:.0f}s of traffic", flush=True)  # fmt: skip
            sim_args = run_mod.parser().parse_args(
                ["--sessions", "100000000", "--max-seconds", str(seconds),
                 "--concurrency", str(args.concurrency), "--think-ms", args.think_ms,
                 "--dwell-ms", args.dwell_ms, "--seed", str(args.seed), "--out", str(run_dir),
                 "--api", api_url]
            )  # fmt: skip
            evicted = ads_mod.evicted_keys()
            summary = run_mod.run_main(sim_args)
            ads.end(ids)
            verify_mod.wait_for_consumers(("billing",))
            evicted = ads_mod.evicted_keys() - evicted
            with psycopg.connect(db_url) as conn:
                per = measure(conn, ids, t0, args.window, read_pages(run_dir / "ledger.jsonl"))
            res = {"config": name, "margin": margin, "pacing": paced, "t0": t0,
                   "campaigns": per, "summary": summarize(per), "sim": summary,
                   "evicted_keys": evicted}  # fmt: skip
            results.append(res)
            s = res["summary"]
            print(f"  overdelivery {s['overdelivery']:.2%}, spent {s['spent']:.1%}, "
                  f"{s['exhausted']}/{s['campaigns']} ran out (median {s['ran_out_median']}), "
                  f"serving stopped at {s['stopped_median']}, Redis evicted {evicted} keys",
                  flush=True)  # fmt: skip
    finally:
        ads.stop_policy(EXP_ID)
        ads.close()
    doc = {
        "window_s": args.window, "tail_s": args.tail, "concurrency": args.concurrency,
        "dwell_ms": args.dwell_ms, "think_ms": args.think_ms, "seed": args.seed,
        "campaigns": [{"bid": b, "budget": bud} for b, bud in CAMPAIGNS],
        "items_per_campaign": ITEMS_PER_CAMPAIGN, "freq_cap": 100, "candidate_cache": cache,
        "results": results,
    }  # fmt: skip
    path = ROOT / "artifacts" / "billing" / "overdelivery.json"
    path.write_text(json.dumps(doc, indent=2))
    print(f"overdelivery  {len(results)} configurations -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
