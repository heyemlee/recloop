"""The M4 drills' bookkeeping on made-up runs: sim.traffic's summaries, the A/B report (delta
method against a direct computation, SRM, the A/A re-split) and the release report."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from analysis import ab_report, release_report
from sim import traffic


def ledger(tmp_path):
    rows = []
    t0 = 1_000_000.0
    for i in range(40):
        rid = f"r{i}"
        version = "vA" if i < 20 else "vB"
        rows.append(
            {
                "kind": "page",
                "request_id": rid,
                "user_id": i % 8,
                "surface": "home",
                "t": t0 + i,
                "variants": {"rank": "a" if i % 2 else "b"},
                "degraded": [] if i % 10 else ["rank_timeout"],
                "model_version": version,
                "ms": 5.0 + i,
            }
        )
        for pos in range(1, 6):
            rows.append(
                {
                    "kind": "event",
                    "event_id": "00000000-0000-7000-8000-000000000000",
                    "type": "impression",
                    "user_id": i % 8,
                    "item_id": pos,
                    "impression_id": f"{rid}:{pos}",
                }
            )
        if i % 4 == 0:
            rows.append(
                {
                    "kind": "event",
                    "event_id": "00000000-0000-7000-8000-000000000001",
                    "type": "click",
                    "user_id": i % 8,
                    "item_id": 1,
                    "impression_id": f"{rid}:1",
                }
            )
            rows.append(
                {
                    "kind": "event",
                    "event_id": "00000000-0000-7000-8000-000000000002",
                    "type": "click",
                    "user_id": i % 8,
                    "item_id": 1,
                    "impression_id": f"{rid}:1",
                }
            )  # a double click
    rows.append(
        {"kind": "error", "user_id": 1, "surface": "home", "t": t0 + 5, "status": 503, "ms": 100.0}
    )
    path = tmp_path / "ledger.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + '\n{"kind": "pa')  # a torn last line
    return path, t0


def test_traffic_summaries(tmp_path):
    path, t0 = ledger(tmp_path)
    pages, events, errors = traffic.read_ledger(path)
    assert len(pages) == 40 and len(errors) == 1
    s = traffic.summarize(pages, events, errors, t0, t0 + 40, lambda p: p["model_version"])
    a, b = s["arms"]["vA"], s["arms"]["vB"]
    assert (
        a["page_views"] == 20 and a["impressions"] == 100 and a["clicks"] == 5
    )  # double clicks once
    assert a["degraded_share"] == pytest.approx(0.1) and s["failed_page_views"] == 1
    assert b["p50_ms"] == pytest.approx(np.percentile(np.arange(20, 40) + 5.0, 50))
    tl = traffic.timeline(pages, errors, t0, t0 + 40, step=10)
    assert [r["versions"] for r in tl] == [{"vA": 10}, {"vA": 10}, {"vB": 10}, {"vB": 10}] and tl[
        0
    ]["failed"] == 1
    users = traffic.ctr_by_user(pages, events, t0, t0 + 40, lambda p: p["model_version"])
    point, lo, hi = traffic.bootstrap_ctr_diff(users["vA"], users["vB"], reps=200)
    assert point == pytest.approx(5 / 100 - 5 / 100) and lo <= point <= hi


def test_ratio_delta_matches_its_formula():
    rng = np.random.default_rng(0)
    i = rng.integers(5, 50, 400).astype(float)
    c = rng.binomial(i.astype(int), 0.05).astype(float)
    r, v = ab_report.ratio_delta(c, i)
    assert r == pytest.approx(c.sum() / i.sum())
    n, mc, mi = len(c), c.mean(), i.mean()
    cov = np.cov(c, i)
    want = (cov[0, 0] / mi**2 - 2 * mc * cov[0, 1] / mi**3 + mc**2 * cov[1, 1] / mi**4) / n
    assert v == pytest.approx(want)
    # The delta-method standard error agrees with a user bootstrap of the same ratio.
    reps = [c[k].sum() / i[k].sum() for k in (rng.integers(0, n, n) for _ in range(3000))]
    assert math.sqrt(v) == pytest.approx(np.std(reps), rel=0.1)


def impressions(n_users: int = 400, seed: int = 1, lift: float = 0.0) -> pd.DataFrame:
    """Home impressions of an A/A (lift 0) or an A/B; arms by the real bucketing."""
    from analysis import srm as srm_mod

    rng = np.random.default_rng(seed)
    exp = {
        "layer": "rank",
        "bucket_from": 0,
        "bucket_to": 1000,
        "variants": [{"name": "a", "share": 50}, {"name": "b", "share": 50}],
    }
    rows = []
    for u in range(n_users):
        arm = srm_mod.assign([exp], "rank", str(u))
        base = rng.uniform(0.005, 0.03)  # users differ: impressions within a user are correlated
        for k in range(int(rng.integers(20, 120))):
            p = base * (1 + lift) if arm == "b" else base
            rows.append(
                {
                    "impression_id": f"{u}-{k}",
                    "request_id": f"{u}-{k // 10}",
                    "user_id": u,
                    "item_id": k,
                    "pos": k % 20 + 1,
                    "surface": "home",
                    "slot": "sponsored" if k % 10 == 2 else "organic",
                    "price": 30 if k % 10 == 2 else None,
                    "exp_id_rank": "rank-aa-v1",
                    "exp_rank": arm,
                    "bucket_rank": 0,
                    "degraded": False,
                    "clicked": int(rng.random() < p),
                    "carted": 0,
                    "purchased": 0,
                    "final": True,
                    "t_impression": pd.Timestamp("2026-11-08"),
                    "ranker": "rrf",
                }
            )
    return pd.DataFrame(rows)


def test_ab_report_on_a_made_up_aa(tmp_path):
    df = impressions()
    phase = {
        "exp_id": "rank-aa-v1",
        "variants": [
            {"name": "a", "share": 50, "params": {"ranker": "rrf"}},
            {"name": "b", "share": 50, "params": {"ranker": "rrf"}},
        ],
        "from": 0,
        "to": 1200,
        "sessions": 10,
        "page_views": {"home": 100},
        "evicted_keys": 0,
    }
    res = ab_report.analyse(df, phase, tmp_path / "none.jsonl")
    assert (
        res["srm"]["bucketing_mismatch"] == 0 and res["srm"]["switchers"] == 0 and res["srm"]["ok"]
    )
    m = res["metrics"]["ctr"]
    assert m["ci_delta"][0] < m["diff"] < m["ci_delta"][1]
    assert 0 <= m["p"] <= 1 and res["resalt"]["n"] == ab_report.RESALTS
    assert sum(res["resalt"]["deciles"]) == ab_report.RESALTS
    md = ab_report.render(
        {"bundle": "v1", "campaigns": 16, "phases": {"aa": phase}}, {"aa": res}, None
    )
    assert "## A/A" in md and "Re-split 100 times" in md


def test_ab_report_finds_a_real_lift(tmp_path):
    df = impressions(n_users=800, lift=0.6, seed=3)
    phase = {
        "exp_id": "rank-aa-v1",
        "variants": [{"name": "a", "share": 50}, {"name": "b", "share": 50}],
    }
    res = ab_report.analyse(df, phase, tmp_path / "none.jsonl")
    assert res["metrics"]["ctr"]["ci_delta"][0] > 0


def test_release_report_renders(tmp_path, monkeypatch):
    summary = {
        "arms": {
            "vA": {
                "page_views": 10,
                "users": 3,
                "degraded_share": 0.0,
                "p50_ms": 5.0,
                "p99_ms": 9.0,
                "impressions": 50,
                "clicks": 1,
                "ctr": 0.02,
            }
        },
        "failed_page_views": 0,
        "failed_by_status": {},
    }
    drill = {
        "t0": 1000.0,
        "old": "vA",
        "new": "vB",
        "bad": "vC",
        "args": {"concurrency": 16, "canary_percent": 5},
        "steps": [
            {"step": "full rollout vB", "t": 1100.0, "seconds": 0.1, "ok": True, "response": {}}
        ],
        "canary": {
            "from": 1000.0,
            "to": 1100.0,
            "summary": summary,
            "ctr_diff": [0.001, -0.01, 0.012],
            "checks": {
                "failed_page_views": True,
                "degraded": True,
                "p99": True,
                "ctr_not_lower": True,
            },
            "pass": True,
        },
        "switch": {"full": {"first_page_s": 0.01, "last_other_page_s": 0.2}},
        "phases": {"before": {"from": 1000.0, "to": 1100.0, **summary}},
        "bad_vs_full_ctr": [-0.004, -0.007, -0.001],
        "timeline": [{"t": 90.0, "versions": {"vA": 5, "vB": 5}, "failed": 0, "p99_ms": 9.0}],
        "events": [
            {
                "version": "vB",
                "from": "canary",
                "to": "active",
                "at": 1100.0,
                "actor": "drill",
                "note": "ok",
            }
        ],
    }
    monkeypatch.setattr(release_report, "BUNDLES", tmp_path)
    md = release_report.render(drill)
    assert (
        "## Canary" in md and "Decision: full rollout." in md and "| `vB` | canary | active |" in md
    )
