"""Nearline MiniOneRec: the simulated A/B and the freshness of the results -> docs/results/nearline.md.

Inputs (all files; no database):
  artifacts/nearline/ab/run.json            the phase (sim.gen_ab): arms, times, traffic, backfill
  artifacts/nearline/ab/served_gen.jsonl    the phase's served home first pages: arm, hv, cache,
                                            what each request saw of the generation result
  artifacts/nearline/ab/traffic/ledger.jsonl  the simulator's view: client latency per request
  artifacts/nearline/ab/worker/{jobs,queue}.jsonl, run.json   the generation worker
  artifacts/samples/impressions             the sample job: every impression joined to its served
                                            record (source, ranks, arm) and labelled by a click
                                            within 30 minutes (`make samples`)

The effect metrics are M4's (analysis/ab_report.py): users as the unit, ratio metrics with the
delta-method variance and a user-level bootstrap interval, SRM against the configured split and
the Python twin of the bucketing. Freshness is measured per history change: from the moment
apply.lua applied it (and from the event's own time) to the moment the worker wrote a result
that includes it, plus what the requests saw (fresh, stale by k history versions, missing).

Usage: python -m analysis.nearline_report
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import ab_report
from analysis import srm as srm_mod

ROOT = Path(__file__).resolve().parent.parent
AB = ROOT / "artifacts" / "nearline" / "ab"
IMPRESSIONS = ROOT / "artifacts" / "samples" / "impressions"
OUT = ROOT / "docs" / "results" / "nearline.md"
NUM_ITEMS = 3459
SEED = 20261108
ARMS = ("control", "gen")
PCTS = (50, 90, 99)


# ----------------------------------------------------------------------------- inputs


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # a line cut off when a process was stopped
                continue
    return rows


def load_impressions(exp_id: str, directory: Path = IMPRESSIONS) -> pd.DataFrame:
    days = sorted(directory.glob("day=*"))
    if not days:
        raise SystemExit(f"no impressions under {directory}: run `make samples`")
    cols = [
        "impression_id",
        "request_id",
        "user_id",
        "item_id",
        "pos",
        "surface",
        "slot",
        "source",
        "ranks",
        "price",
        "candidate_cache",
        "exp_id_recall",
        "exp_recall",
        "degraded",
        "clicked",
        "carted",
        "final",
    ]
    df = pd.concat([pd.read_parquet(d, columns=cols) for d in days], ignore_index=True)
    df = df[df.exp_id_recall == exp_id].copy()
    # The ranker's report works on exp_rank; the arm here is the recall layer's.
    df["exp_rank"] = df["exp_recall"]
    df["rank_sources"] = df.ranks.map(_rank_sources)
    return df


def _rank_sources(ranks) -> frozenset:
    """The sources that returned an impression's item (the served record's ranks map)."""
    if ranks is None:
        return frozenset()
    if isinstance(ranks, dict):
        return frozenset(ranks)
    return frozenset(k for k, _ in ranks)  # a parquet map reads as (key, value) pairs


# ----------------------------------------------------------------------------- effect


def srm(df: pd.DataFrame, exp: dict) -> dict:
    users = df.groupby("user_id").exp_recall.agg(["nunique", "first"])
    counts = users["first"].value_counts()
    names = [v["name"] for v in exp["variants"]]
    total = int(counts.reindex(names).fillna(0).sum())
    expected = [total * v["share"] / 100 for v in exp["variants"]]
    observed = [int(counts.get(n, 0)) for n in names]
    from scipy import stats

    chi2, p = stats.chisquare(observed, expected)
    mismatch = int(
        sum(
            srm_mod.assign([exp], "recall", str(uid)) != row["first"]
            for uid, row in users.iterrows()
        )
    )
    switchers = int((users["nunique"] > 1).sum())
    return {
        "observed": dict(zip(names, observed, strict=True)),
        "chi2": float(chi2),
        "p": float(p),
        "switchers": switchers,
        "bucketing_mismatch": mismatch,
        "ok": bool(p >= ab_report.SRM_ALPHA and switchers == 0 and mismatch == 0),
    }


def effect(df: pd.DataFrame) -> dict:
    totals = ab_report.user_totals(df)
    rng = np.random.default_rng(SEED)
    a = totals[totals.exp_rank == ARMS[0]]
    b = totals[totals.exp_rank == ARMS[1]]
    out = {
        key: ab_report.compare(a, b, num, den, rng) | {"label": label}
        for key, label, num, den in ab_report.METRICS
    }
    rpm = ab_report.compare(a, b, "revenue", "impressions", rng)
    for k in ("a", "b", "diff"):
        rpm[k] *= 1000
    rpm["ci_delta"] = tuple(x * 1000 for x in rpm["ci_delta"])
    rpm["ci_boot"] = tuple(x * 1000 for x in rpm["ci_boot"])
    return {
        "metrics": out,
        "rpm": rpm,
        "clicks_per_user": ab_report._welch(a.clicks.to_numpy(float), b.clicks.to_numpy(float)),
    }


def coverage(df: pd.DataFrame, reps: int = 20) -> dict:
    """Organic home-feed impressions per arm: distinct items at equal impression counts (the
    larger arm subsampled to the smaller one's count, `reps` seeded draws), the share of
    impressions whose item gen returned, where gen was the item's main source, and items only
    gen returned, with their CTR."""
    org = df[(df.surface == "home") & (df.slot != "sponsored")]
    n = int(org.groupby("exp_recall").size().min())
    rng = np.random.default_rng(SEED)
    out = {"impressions_compared": n, "arms": {}}
    for arm in ARMS:
        d = org[org.exp_recall == arm]
        items = d.item_id.to_numpy()
        draws = [
            len(np.unique(rng.choice(items, size=n, replace=False))) for _ in range(reps)
        ] if len(items) > n else [len(np.unique(items))]  # fmt: skip
        via_gen = d.rank_sources.map(lambda s: "gen" in s)
        only_gen = d.rank_sources.map(lambda s: s == frozenset({"gen"}))
        out["arms"][arm] = {
            "impressions": int(len(d)),
            "distinct_items": int(d.item_id.nunique()),
            "distinct_items_at_n": float(np.mean(draws)),
            "catalog_share_at_n": float(np.mean(draws)) / NUM_ITEMS,
            "gen_share": float(via_gen.mean()),
            "gen_primary_share": float((d.source == "gen").mean()),
            "gen_only_share": float(only_gen.mean()),
            "gen_only_ctr": float(d.clicked[only_gen].mean()) if only_gen.any() else float("nan"),
            "other_ctr": float(d.clicked[~only_gen].mean()),
        }
    return out


def latency(ledger: list[dict], served: list[dict]) -> dict:
    """Per arm: client P50 / P99 over every page view, degraded share; candidate-cache results
    of the home first pages (served log)."""
    by: dict[str, dict] = {}
    for r in ledger:
        if r.get("kind") != "page" or "ms" not in r:
            continue
        d = by.setdefault(r["variants"].get("recall"), {"ms": [], "degraded": 0})
        d["ms"].append(r["ms"])
        d["degraded"] += bool(r["degraded"])
    cache = {arm: Counter(s["cache"] for s in served if s["arm"] == arm) for arm in ARMS}
    out = {}
    for arm in ARMS:
        d = by.get(arm, {"ms": [np.nan], "degraded": 0})
        c = cache[arm]
        looked = sum(v for k, v in c.items() if k in ("hit", "miss"))
        out[arm] = {
            "requests": len(d["ms"]),
            "p50": float(np.nanpercentile(d["ms"], 50)),
            "p99": float(np.nanpercentile(d["ms"], 99)),
            "degraded": d["degraded"] / max(1, len(d["ms"])),
            "cache_hit_rate": c.get("hit", 0) / looked if looked else float("nan"),
        }
    return out


# ----------------------------------------------------------------------------- freshness


def _pcts(x) -> dict:
    x = np.asarray(x, dtype=float)
    if not len(x):
        return {"n": 0}
    return {
        "n": int(len(x)),
        **{f"p{p}": float(np.percentile(x, p)) for p in PCTS},
        "max": float(x.max()),
    }


def freshness(
    jobs: list[dict], queue: list[dict], served: list[dict], t0: float, t1: float
) -> dict:
    """Per history change answered during the traffic: written - applied, written - event time
    (seconds); jobs per round; the queue over time; what the gen arm's requests saw."""
    lo, hi = t0 * 1000, t1 * 1000 + 3_600_000  # changes applied during the traffic, answered later
    applied, event, per_job, waits = [], [], [], []
    statuses = Counter(j.get("status") for j in jobs)
    for j in jobs:
        if j.get("status") != "written":
            continue
        changes = [c for c in j.get("changes", []) if t0 * 1000 <= c[1] <= hi]
        if not changes:
            continue
        per_job.append(len(changes))
        waits.append((j["claimed_ms"] - j["due_ms"]) / 1000)
        for ev, ap in changes:
            applied.append((j["written_ms"] - ap) / 1000)
            event.append((j["written_ms"] - ev) / 1000)
    q = [r for r in queue if lo <= r["t"] <= t1 * 1000]
    gen_served = [s for s in served if s["arm"] == "gen" and s.get("gen")]
    seen = Counter(s["gen"]["status"] for s in gen_served)
    lags = Counter(s["gen"].get("lag", 0) for s in gen_served if s["gen"]["status"] == "stale")
    ages = [s["gen"]["age_ms"] / 1000 for s in gen_served if "age_ms" in s["gen"]]
    return {
        "job_statuses": dict(statuses),
        "since_applied_s": _pcts(applied),
        "since_event_s": _pcts(event),
        "changes_per_job": _pcts(per_job),
        "queue_wait_s": _pcts(waits),
        "queue_max": max((r["queued"] for r in q), default=0),
        "due_max": max((r["due"] for r in q), default=0),
        "due_last": q[-1]["due"] if q else None,
        "served": {
            "requests": len(gen_served),
            **{k: v / max(1, len(gen_served)) for k, v in seen.items()},
        },
        "stale_lag": dict(sorted(lags.items())),
        "result_age_s": _pcts(ages),
    }


def session_gaps(ledger: list[dict], gap_s: float = 10.0) -> dict:
    """Seconds between a user's visits (page views more than gap_s apart): the time scale the
    freshness compares with. The simulator compresses time: its users come back within minutes."""
    by: dict[int, list[float]] = {}
    for r in ledger:
        if r.get("kind") == "page":
            by.setdefault(r["user_id"], []).append(r["t"])
    gaps = []
    for ts in by.values():
        ts.sort()
        gaps.extend(b - a for a, b in zip(ts, ts[1:], strict=False) if b - a > gap_s)
    return _pcts(gaps)


# ----------------------------------------------------------------------------- report


def _s(x: float, d: int = 1) -> str:
    return "n/a" if x != x else f"{x:,.{d}f}"


def _dist(p: dict, unit: str = "s") -> str:
    if not p.get("n"):
        return "none"
    return (
        f"P50 {_s(p['p50'])} {unit}, P90 {_s(p['p90'])} {unit}, P99 {_s(p['p99'])} {unit}, "
        f"max {_s(p['max'])} {unit} ({p['n']:,})"
    )


def render(run: dict, worker: dict, res: dict) -> str:
    exp = run["experiment"]
    L = [
        "# Nearline MiniOneRec: simulated A/B and freshness",
        "",
        "<!-- Generated by `make nearline-report` (analysis/nearline_report.py). Do not edit by hand. -->",
        "",
        "MiniOneRec does not run on the request path: one 50-beam request takes 3.1 s on this Mac "
        "(minionerec.md, Cost). It runs nearline (design §11 step 5): a history change queues the "
        "user (apply.lua, in the same script as the history update), the worker generates once the "
        f"history has been quiet for {worker['settings']['debounce_ms'] / 1000:.0f} s (at most "
        f"{worker['settings']['max_wait_ms'] / 1000:.0f} s after the first unanswered change), "
        f"writes the top {worker['settings']['top_k']} items to the state Redis, and the home "
        "feed fuses them as a fourth RRF source (`gen`) when the user has a result.",
        "",
        "## Setup",
        "",
        f"- Bundle `{run['bundle']}`; generator `{run['gen_version']}` "
        f"({worker['generator']['device']}, {worker['generator']['settings']['dtype']}, "
        f"{worker['generator']['settings']['num_beams']} beams, prompt "
        f"{'encoded once' if worker['generator']['settings']['prefill'] == 'once' else 'per beam'}, "
        f"batches of up to {worker['settings']['max_batch']}).",
        f"- Experiment `{run['exp_id']}` in the recall layer, 50 / 50 over all users: control = "
        f"{_weights(exp['variants'][0])}; gen = {_weights(exp['variants'][1])}. The gen weight was "
        "fixed beforehand (SASRec's and ItemCF's), not tuned. "
        + (
            f"The recall layer's running experiments ({', '.join(f'`{e}`' for e in run['stopped_and_restored'])}) "
            "were stopped for the phase and restored afterwards. "
            if run["stopped_and_restored"]
            else ""
        )
        + "Both arms are ordered by RRF: the LightGBM ranker was trained on the three sources' "
        "features only (an item only gen returned has none of them), so the rank layer ran no "
        "experiment.",
        f"- Traffic: {(run['to'] - run['from']) / 60:.0f} min, {run['concurrency']} sessions at a "
        f"time with pauses of {run['think_ms'][0]:.0f}-{run['think_ms'][1]:.0f} ms between page "
        f"views ({run['sessions']:,} sessions, {sum(run['page_views'].values()):,} page views), "
        "fewer than M4's 16 so that the worker keeps up. Standing campaigns fill the sponsored slots.",
        f"- Start: consumers caught up, Redis reseeded with the dataset users' training histories, "
        f"then the worker generated the gen arm's {run['gen_users']:,} users from them "
        f"(backfill: {run['backfill_seconds'] / 60:.1f} min) before the traffic started. The worker "
        "generates only for the gen arm's users.",
        f"- State Redis: {_redis(run.get('redis'), 'state')}; cache Redis: {_redis(run.get('redis'), 'cache')}.",
        "",
        "## Split",
        "",
    ]
    s = res["srm"]
    L.append(
        f"{sum(s['observed'].values()):,} users: "
        + ", ".join(f"{k} {v:,}" for k, v in s["observed"].items())
        + f"; chi-square p = {s['p']:.3f}; {s['switchers']} users in two arms; "
        f"{s['bucketing_mismatch']} users whose arm differs from the Python bucketing"
        + (" (sound)." if s["ok"] else " — **the split is not sound: do not read the effect.**")
    )
    L += [
        "",
        "## Effect",
        "",
        "| Metric (gen vs control) | control | gen | Difference | 95% CI (delta) | 95% CI (bootstrap) | p |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in res["effect"]["metrics"].values():
        L.append(_metric_row(m["label"], m))
    L.append(
        _metric_row("Ad revenue per 1,000 home impressions (cents)", res["effect"]["rpm"], d=1)
    )
    L += ["", "## Coverage", ""]
    c = res["coverage"]
    L.append(
        f"Organic home-feed impressions; distinct items compared at {c['impressions_compared']:,} "
        "impressions per arm (the larger arm subsampled, mean of 20 seeded draws)."
    )
    L += [
        "",
        "| Arm | Impressions | Distinct items | ... at equal impressions | Share of the catalog | gen returned the item | gen its main source | Only gen returned it | CTR, only gen | CTR, the rest |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        a = c["arms"][arm]
        L.append(
            f"| {arm} | {a['impressions']:,} | {a['distinct_items']:,} | {a['distinct_items_at_n']:,.0f} "
            f"| {a['catalog_share_at_n']:.1%} | {a['gen_share']:.1%} | {a['gen_primary_share']:.1%} "
            f"| {a['gen_only_share']:.1%} | {_s(a['gen_only_ctr'], 4)} | {_s(a['other_ctr'], 4)} |"
        )
    L += [
        "",
        "## Latency",
        "",
        "| Arm | Requests | Client P50 (ms) | Client P99 (ms) | Degraded | Candidate cache hit rate (home, first pages) |",
        "|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        a = res["latency"][arm]
        L.append(
            f"| {arm} | {a['requests']:,} | {a['p50']:.1f} | {a['p99']:.1f} | {a['degraded']:.2%} | {_s(a['cache_hit_rate'] * 100)}% |"
        )
    L += [
        "",
        "The gen arm reads its result in the same Redis round trip as the rest of the user context. "
        "Its candidate cache key carries the result's history version, so a new result is used by "
        "the next request instead of waiting out the cache's 5 minutes; that costs the gen arm one "
        "more cache miss per written result.",
        "",
        "## Freshness",
        "",
    ]
    f = res["freshness"]
    L += [
        f"- History change applied → result written: {_dist(f['since_applied_s'])}.",
        f"- The event itself → result written: {_dist(f['since_event_s'])}.",
        f"- Changes answered per generation: {_dist(f['changes_per_job'], 'changes')}; due → claimed "
        f"(time in the queue after the debounce): {_dist(f['queue_wait_s'])}.",
        f"- Queue: at most {f['queue_max']:,} users queued and {f['due_max']:,} due at once during the "
        f"traffic; {f['due_last'] if f['due_last'] is not None else 'n/a'} due at the end.",
        "- The gen arm's home first pages: "
        + ", ".join(f"{k} {v:.1%}" for k, v in f["served"].items() if k != "requests")
        + f" of {f['served']['requests']:,}"
        + (
            "; stale by "
            + ", ".join(
                f"{k} version{'s' if k != 1 else ''}: {v:,}" for k, v in f["stale_lag"].items()
            )
            if f["stale_lag"]
            else ""
        )
        + f". Age of the result when served: {_dist(f['result_age_s'])}.",
        f"- For scale: a simulated user comes back after {_dist(res['gaps'])} (page views more "
        "than 10 s apart). The simulator compresses time; against real visit gaps of hours the "
        "debounce and the generation time would be small.",
        "",
        "## Worker",
        "",
        f"- {worker['generated_users']:,} users generated in {worker['batches']:,} rounds, "
        f"{_s(worker['seconds_per_user'] or float('nan'), 2)} s per user of generation; generator "
        f"setup {worker['generator_setup_seconds']:.0f} s; peak footprint "
        f"{worker['memory']['lifetime_max_footprint'] / 2**30:.1f} GiB (MPS cap "
        f"{worker['settings']['mps_memory_gb']:.0f} GB); job statuses: "
        + ", ".join(f"{k} {v:,}" for k, v in f["job_statuses"].items())
        + ".",
        f"- 1-minute load {worker['loadavg_start'][0]:.1f} at the start, {worker['loadavg_end'][0]:.1f} "
        f"at the end; swapped out while it ran: {_gib(worker.get('swapped_out_bytes'))}.",
        "",
    ]
    return "\n".join(L) + "\n"


def _weights(v: dict) -> str:
    return "RRF " + ", ".join(f"{k} {w:g}" for k, w in v["params"]["rrf_weights"].items())


def _redis(r: dict | None, name: str) -> str:
    if not r or name not in r:
        return "not recorded"
    x = r[name]
    return (
        f"{x['evicted_keys']:,} evicted, {x['oom_errors']:,} refused writes, "
        f"{x['used_memory_mb']:,} MB used at the end"
    )


def _gib(b) -> str:
    return "unknown" if b is None else f"{b / 2**30:.1f} GiB"


def _metric_row(label: str, m: dict, d: int = 4) -> str:
    return (
        f"| {label} | {m['a']:.{d}f} | {m['b']:.{d}f} | {m['diff']:+.{d}f} ({m['rel']:+.1%}) "
        f"| [{m['ci_delta'][0]:+.{d}f}, {m['ci_delta'][1]:+.{d}f}] "
        f"| [{m['ci_boot'][0]:+.{d}f}, {m['ci_boot'][1]:+.{d}f}] | {m['p']:.3f} |"
    )


def build(ab_dir: Path = AB, impressions: Path = IMPRESSIONS) -> tuple[dict, dict, dict]:
    run = json.loads((ab_dir / "run.json").read_text())
    worker = json.loads((ab_dir / "worker" / "run.json").read_text())
    df = load_impressions(run["exp_id"], impressions)
    served = read_jsonl(ab_dir / "served_gen.jsonl")
    ledger = read_jsonl(ab_dir / "traffic" / "ledger.jsonl")
    res = {
        "srm": srm(df, run["experiment"]),
        "effect": effect(df),
        "coverage": coverage(df),
        "latency": latency(ledger, served),
        "freshness": freshness(
            read_jsonl(ab_dir / "worker" / "jobs.jsonl"),
            read_jsonl(ab_dir / "worker" / "queue.jsonl"),
            served,
            run["from"],
            run["to"],
        ),
        "gaps": session_gaps(ledger),
    }
    return run, worker, res


def main() -> None:
    run, worker, res = build()
    OUT.write_text(render(run, worker, res))
    (AB / "report.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
