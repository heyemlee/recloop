"""Render docs/results/billing.md (design §9.3, §9.4, §16 M3) from the billing runs' artifacts:

    artifacts/billing/drill/verify.json      sim.billing_drill: the ledger against the truth
    artifacts/billing/overdelivery.json      sim.overdelivery: safety margins and pacing

Either may be missing (its section then says how to produce it). The output depends only on
those files, so re-rendering gives the same bytes.

Usage: python -m analysis.billing_report [--drill ...] [--overdelivery ...] [--out ...]
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "billing"
DEFAULT_OUT = ROOT / "docs" / "results" / "billing.md"


def dollars(cents: float) -> str:
    return f"${cents / 100:,.2f}"


def pct(x: float | None, digits: int = 1) -> str:
    return "–" if x is None else f"{100 * x:.{digits}f} %"


def share(x: float | None) -> str:
    """A point of the flight window, as a share of it."""
    return "–" if x is None else f"{100 * x:.0f} %"


def when(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d %H:%M UTC")


def table(head: list[str], rows: list[list], align: str | None = None) -> list[str]:
    align = align or "l" + "r" * (len(head) - 1)
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(head) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


# ---------------------------------------------------------------- drill


def chaos_counts(chaos: list[dict]) -> dict[str, int]:
    n = {"kill": 0, "redpanda": 0, "slow": 0}
    for c in chaos:
        a = c["action"]
        n["kill"] += a == "kill -9 billing"
        n["redpanda"] += a == "restarted redpanda"
        n["slow"] += a.startswith("postgres +")
    return n


def drill_summary(d: dict) -> list[str]:
    n = chaos_counts(d["chaos"])
    failed = [c["check"] for c in d["checks"] if not c["ok"]]
    verdict = (
        "**The ledger equals the simulator's truth charge by charge**"
        if d["ok"]
        else f"**{len(failed)} check(s) failed**: " + "; ".join(failed)
    )
    over = next(c for c in d["checks"] if c["check"] == "amounts: charges above budget")
    return [
        f"- **Chaos drill** ({d['minutes']:.0f} minutes of simulated traffic, {n['kill']} × "
        f"`kill -9` of billing, {n['redpanda']} Redpanda restarts, PostgreSQL {d['pg_latency_ms']} "
        f"ms slower for billing {n['slow']} times): {d['sponsored_click_events']:,} sponsored "
        f"click events ({d['forged_click_events']} with a forged token) became "
        f"{d['charges']:,} charges. {verdict}: {d['charges_expected']:,} expected, "
        f"{d['charges']:,} in the ledger. **{over['actual']} charges above budget**; the "
        f"budget rows and the Redis counters equal the ledger.",
    ]


def drill_section(d: dict) -> list[str]:
    sim = d["sim"]
    n = chaos_counts(d["chaos"])
    plan = d.get("chaos_plan", [])
    downs = [st["down"] for st in plan if st["action"] == "kill_billing"]
    down = f"{min(downs):.0f}–{max(downs):.0f}" if downs else "2–10"
    slows = [st["seconds"] for st in plan if st["action"] == "pg_slow"]
    slow = f"{max(slows):.0f}" if slows else "180"
    bids = {c["campaign_id"]: c["bid"] for c in d.get("campaign_settings", [])}
    out = ["## 1. Chaos drill: the ledger against the truth", ""]
    out += [
        f"Run {when(sim['started'])}, `make billing-drill` (`sim/billing_drill.py`, seed "
        f"{d['seed']}). The simulator ({d['concurrency']} sessions at a time, "
        f"{d['think_ms'].replace(',', '–')} ms between page views) sent {sim['events_unique']:,} "
        f"events ({sim['event_copies_delivered']:,} deliveries with its faults "
        f"`{d['faults']}`) over {sum(sim['page_views'].values()):,} page views; "
        f"{d['slots_served']:,} sponsored slots were served and "
        f"{d['sponsored_impressions']:,} of them seen. The run's {d['campaigns']} campaigns were "
        "served with a safety margin of 0 clicks, the setting that lets the most clicks "
        "arrive after a budget is spent.",
        "",
        f"During the traffic: {n['kill']} × `kill -9` of the billing container (started again "
        f"{down} s later), {n['redpanda']} restarts of Redpanda, and {n['slow']} windows of "
        f"{slow} s in which every reply PostgreSQL sent billing came {d['pg_latency_ms']} ms late "
        "(Toxiproxy between them). Times (s from the start of the traffic):",
        "",
    ]
    out += table(["t (s)", "action"], [[f"{c['t']:.1f}", c["action"]] for c in d["chaos"]], "rl")
    out += [
        "",
        f"After the traffic, billing, the log-sink and the aggregator caught up in "
        f"{d['caught_up_seconds']:.0f} s; then one reconciliation pass and the checks below. The "
        "truth comes from the simulator's own records: the token of every sponsored slot it was "
        "served (any client can read a token's claims), every click it sent, and which tokens it "
        "forged.",
        "",
    ]
    out += table(
        ["check", "expected", "actual", ""],
        [
            [c["check"], c["expected"], c["actual"], "ok" if c["ok"] else "**FAIL**"]
            for c in d["checks"]
        ],
        "lrrl",
    )
    out += [
        "",
        f"Clicks: {d['valid_clicks']:,} valid, {d['forged_click_events']} forged (refused), "
        f"{d['expired_clicks']} expired. {d['repeat_clicks']:,} valid clicks were repeat clicks on "
        "an impression already charged (double clicks; the impression is charged once). "
        "Deliveries of one event beyond the first (duplicates, resent batches, redelivery after "
        "a crash) never reach the ledger: billing claims each event_id once.",
        "",
        "Per campaign (UTC day of serving; *not billed* is the auction price of clicks that came "
        "in after the budget was spent, which the platform absorbs):",
        "",
    ]
    rows = []
    for x in d["days"]:
        rows.append([
            x["campaign_id"], x["day"], f"{bids[x['campaign_id']]} ¢" if x["campaign_id"] in bids else "–",
            dollars(x["budget"]), x["charges"], dollars(x["price"]), dollars(x["billed"]),
            dollars(x["price"] - x["billed"]), x["zero_billed"], dollars(x["spent"]),
            dollars(x["redis"]),
        ])  # fmt: skip
    out += table(
        ["campaign", "day", "bid", "budget", "charges", "prices", "billed", "not billed",
         "billed 0", "budget row", "Redis"],
        rows,
        "llrrrrrrrrr",
    )  # fmt: skip
    rec = d["reconcile"]
    during = d.get("reconciler_during")
    out += [
        "",
        f"{d['budgets_exhausted']} of {len(d['days'])} campaign-days spent their whole budget. "
        f"Overall {dollars(d['billed_cents'])} billed and {dollars(d['unbilled_cents'])} not "
        f"billed ({d['zero_billed']} clicks billed 0).",
        "",
        f"Reconciliation after the drill (one pass over every campaign of today and yesterday): "
        f"{len(rec['days'])} campaign-days, {rec['match']} in agreement, {rec['redis_behind']} "
        f"Redis counters behind the ledger and repaired (+{rec['repaired_cents']} ¢), "
        f"{rec['redis_ahead']} ahead, {rec['row_mismatch']} budget rows different from the "
        "ledger.",
    ]
    if during:
        out += [
            "",
            f"During the drill billing's own reconciler (every minute, restarted with billing "
            f"{during['billing_starts']} times) found a Redis counter behind the ledger in "
            f"{during['passes_with_repairs']} passes and added {during['repaired_cents']} ¢: "
            "increments lost when billing was killed between its commit and the Redis update.",
        ]
    out += [
        "",
        "The same run, checked as `make drill` checks the event pipeline:",
        "",
    ]
    out += table(
        ["check", "expected", "actual", ""],
        [[c["check"], c["expected"], c["actual"], "ok" if c["ok"] else "**FAIL**"] for c in d["pipeline"]["checks"]],
        "lrrl",
    )  # fmt: skip
    f = d["freq"]
    cap = d.get("freq_cap", 20)
    over_cap = sum(v for k, v in f["histogram"].items() if int(k) > cap)
    at_cap = sum(v for k, v in f["histogram"].items() if int(k) == cap)
    out += [
        "",
        f"Frequency cap ({cap} impressions per user, campaign and day in this run, counted from "
        f"impression events): {f['user_campaign_days']:,} user-campaign-days had a sponsored "
        f"impression, {at_cap:,} reached the cap and "
        + (
            f"{over_cap:,} went over it (at most {f['max']}): slots served before the earlier "
            "impressions had reached the counter, which §9.2 accepts."
            if over_cap
            else "none went over it."
        ),
        "",
    ]
    return out


# ---------------------------------------------------------------- overdelivery


def config_name(r: dict) -> tuple[str, str]:
    return str(r["margin"]), "on" if r["pacing"] else "off"


def overdelivery_summary(o: dict) -> list[str]:
    by = {(r["margin"], r["pacing"]): r["summary"] for r in o["results"]}
    parts = []
    for m in sorted({r["margin"] for r in o["results"]}):
        s = by.get((m, False))
        if s:
            parts.append(f"{m} clicks {pct(s['overdelivery'], 2)}")
    paced = [
        f"{m} clicks {pct(by[(m, True)]['overdelivery'], 2)}"
        for m in sorted({r["margin"] for r in o["results"]})
        if (m, True) in by
    ]
    line = (
        "- **Overdelivery** (clicks the platform absorbs, as a share of the budgets) by safety margin: "
        + ", ".join(parts)
    )
    if paced:
        line += "; with pacing " + ", ".join(paced)
    return [line + ". Details and the cost in unspent budget in §2."]


def overdelivery_section(o: dict) -> list[str]:
    camps = ", ".join(f"{dollars(c['budget'])} at {c['bid']} ¢" for c in o["campaigns"])
    out = ["## 2. Overdelivery: safety margin and pacing", ""]
    first = min(r["t0"] for r in o["results"])
    out += [
        f"Run {when(first)}, `make overdelivery` (`sim/overdelivery.py`, seed {o['seed']}). Each "
        f"configuration gets four new campaigns with the same bids, daily budgets and items "
        f"({camps}; one advertiser each, frequency cap {o['freq_cap']}) and a flight window of "
        f"{o['window_s']:.0f} s, then {o['window_s'] + o['tail_s']:.0f} s of traffic: "
        f"{o['concurrency']} simulated users at a time, each staying "
        f"{o['dwell_ms'].replace(',', '–')} ms on a page before acting, so a click reaches "
        "billing seconds after its slot was served. The stub pCTR is the same for every ad, so "
        "the auction ranks by bid: the first campaign is shown first and runs out first, and the "
        "next one moves up. The safety margin and pacing are an ads-layer experiment (§8), "
        "changed between configurations.",
        "",
        "- *overdelivery*: the auction price of clicks not billed (the budget was spent when they "
        "came in) / the budgets;",
        "- *spent*: billed / the budgets;",
        "- *ran out at*: when the last billed click of a campaign that spent its whole budget "
        "came in, as a share of the window (median over those campaigns);",
        "- *stopped at*: when a campaign's last sponsored slot was served (median over the four).",
        "",
    ]
    rows = []
    for r in o["results"]:
        s = r["summary"]
        m, pace = config_name(r)
        rows.append([
            m, pace, pct(s["overdelivery"], 2), dollars(s["unbilled"]), pct(s["spent"]),
            f"{s['exhausted']} / {s['campaigns']}", share(s["ran_out_median"]),
            share(s["stopped_median"]), s["clicks"], s["zero_billed"],
        ])  # fmt: skip
    out += table(
        ["margin (clicks)", "pacing", "overdelivery", "not billed", "spent", "budgets spent",
         "ran out at", "stopped at", "clicks", "billed 0"],
        rows,
        "rlrrrrrrrr",
    )  # fmt: skip
    out += [
        "",
        "Per campaign (spend by the end of each quarter of the window, as a share of its budget):",
        "",
    ]
    rows = []
    for r in o["results"]:
        m, pace = config_name(r)
        for c in r["campaigns"]:
            curve = " / ".join(
                f"{100 * c['spend_curve'][q]:.0f}" for q in ("0.25", "0.5", "0.75", "1.0")
            )
            rows.append([
                m, pace, c["campaign_id"], dollars(c["budget"]), c["slots_served"], c["clicks"],
                dollars(c["billed"]), dollars(c["unbilled"]), share(c["ran_out_at"]),
                share(c["stopped_at"]), curve,
            ])  # fmt: skip
    out += table(
        ["margin", "pacing", "campaign", "budget", "slots", "clicks", "billed", "not billed",
         "ran out at", "stopped at", "spent by quarter (%)"],
        rows,
        "rlrrrrrrrrl",
    )  # fmt: skip
    out.append("")
    return out


# ---------------------------------------------------------------- page


def render(drill: dict | None, over: dict | None) -> str:
    out = [
        "# Billing correctness and overdelivery",
        "",
        "Generated by `make billing` (`analysis/billing_report.py` over `artifacts/billing/`); "
        "do not edit by hand. Design §9.3 (billing) and §9.4 (budget vs. overdelivery); the "
        "M3 acceptance of §16: after the chaos drill the ledger equals the truth charge by "
        "charge, no charge is above budget, and the overdelivery of the three safety margins.",
        "",
        "## Summary",
        "",
    ]
    if drill:
        out += drill_summary(drill)
    if over:
        out += overdelivery_summary(over)
    out.append("")
    if drill:
        out += drill_section(drill)
    else:
        out += ["## 1. Chaos drill", "", "Not run yet: `make billing-drill`.", ""]
    if over:
        out += overdelivery_section(over)
    else:
        out += ["## 2. Overdelivery", "", "Not run yet: `make overdelivery`.", ""]
    out += [
        "## How billing stays correct",
        "",
        "- One transaction per click (`serving/internal/billing`): claim the click's event_id in "
        "`ad_clicks` (a redelivered event stops here), lock the campaign's budget row for the "
        "day the slot was served, bill min(token price, budget left), insert the charge keyed by "
        "the impression (a second click on it stops here), add to the row. "
        "`CHECK (spent_cents <= budget_cents)` backs the arithmetic.",
        "- The consumer commits a batch's offsets only after every click in it was handled, so a "
        "`kill -9` at any point redelivers the batch and the claims make the redelivery "
        "harmless. Token expiry is judged at the time the api received the click, so a click "
        "processed late (billing down, PostgreSQL slow) is judged as it was then.",
        "- The Redis spend counter that ad serving reads is updated after the commit, once per "
        "click (a dedup marker in the same Lua call). An update lost to a crash is re-applied by "
        "the reconciliation every minute, with the same marker, so it can never count twice.",
        "",
    ]
    return "\n".join(out)


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--drill", type=Path, default=ART / "drill" / "verify.json")
    p.add_argument("--overdelivery", type=Path, default=ART / "overdelivery.json")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    text = render(load(args.drill), load(args.overdelivery))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(
        f"billing report -> {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}"
    )


if __name__ == "__main__":
    main()
