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
    pc = traffic.ctr_by_page_count(pages, events, t0, t0 + 40)
    # 8 users with 5 home pages each: pages 1-3 and 4-15 of each user; 50 impressions per
    # user... 5 per page, a click (counted once) on every fourth page.
    assert [(r["pages"], r["impressions"]) for r in pc] == [([1, 3], 120), ([4, 15], 80)]
    assert sum(r["clicks"] for r in pc) == 10
    users = traffic.ctr_by_user(pages, events, t0, t0 + 40, lambda p: p["model_version"])
    point, lo, hi = traffic.bootstrap_ctr_diff(users["vA"], users["vB"], reps=200)
    assert point == pytest.approx(5 / 100 - 5 / 100) and lo <= point <= hi


def test_snapshot_reads_an_open_and_a_closed_ledger(tmp_path):
    from sim import run as run_mod

    led = run_mod.Ledger(tmp_path / "ledger.jsonl")
    led.write("page", request_id="r1", user_id=1, surface="home", t=1.0)
    pages, _, _ = traffic.snapshot(led)  # still open: flushed first
    assert list(pages) == ["r1"]
    led.write("error", user_id=1, surface="home", t=2.0, status=503, ms=1.0)
    led.close()  # as Traffic.stop() does
    pages, _, errors = traffic.snapshot(led)
    assert list(pages) == ["r1"] and len(errors) == 1


def test_drill_phases_follow_the_steps_times(tmp_path):
    from sim import release_drill

    led, t0 = ledger(tmp_path)  # vA serves pages 0-19, vB pages 20-39 (one per second)
    log = {
        "t0": t0,
        "t_end": t0 + 40,
        "new": "vB",
        "bad": "vC",
        "steps": [
            {"step": "canary vB 5%", "t": t0 + 10, "ok": True},
            # the rollout's step: phases start at a step's own time, not at a variable taken
            # before some analysis
            {"step": "full rollout vB", "t": t0 + 19.5, "ok": True},
        ],
    }
    n, e = release_drill.analyse(log, led)
    assert (n, e) == (40, 1)
    ph = log["phases"]
    assert list(ph) == ["before", "canary", "full"]
    assert ph["full"]["from"] == t0 + 19.5 and ph["full"]["to"] == t0 + 40
    assert ph["full"]["arms"]["vB"]["page_views"] == 20 and "vA" not in ph["full"]["arms"]
    assert log["switch"]["full"]["first_page_s"] == pytest.approx(0.5)
    assert [f["status"] for f in log["failures"]] == [503]


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
        "redis": {
            "state": {
                "policy": "noeviction",
                "maxmemory_mb": 1024,
                "evicted_keys": 0,
                "oom_errors": 0,
                "used_memory_mb": 40.0,
            },
            "cache": {
                "policy": "allkeys-lru",
                "maxmemory_mb": 384,
                "evicted_keys": 7,
                "oom_errors": 0,
                "used_memory_mb": 300.0,
            },
        },
    }
    led, _ = ledger(tmp_path)  # a real ledger shape, torn last line included
    res = ab_report.analyse(df, phase, led)
    assert (
        res["srm"]["bucketing_mismatch"] == 0 and res["srm"]["switchers"] == 0 and res["srm"]["ok"]
    )
    assert res["guardrails"]["a"]["requests"] == 20  # the ledger's pages by their rank variant
    m = res["metrics"]["ctr"]
    assert m["ci_delta"][0] < m["diff"] < m["ci_delta"][1]
    assert 0 <= m["p"] <= 1 and res["resalt"]["n"] == ab_report.RESALTS
    assert sum(res["resalt"]["deciles"]) == ab_report.RESALTS
    md = ab_report.render(
        {
            "bundle": "v1",
            "campaigns": 16,
            "concurrency": 16,
            "think_ms": [50.0, 300.0],
            "phases": {"aa": phase},
        },
        {"aa": res},
        None,
    )
    assert "## A/A" in md and "Re-split 100 times" in md and "| 0 / 0 | 7 |" in md
    assert "16 sessions at a time with pauses of 50-300 ms" in md


def test_srm_fails_when_the_bucketing_disagrees():
    df = impressions(n_users=200)
    df["exp_rank"] = df["exp_rank"].map({"a": "b", "b": "a"})  # arms swapped: ratio still fine
    variants = [{"name": "a", "share": 50}, {"name": "b", "share": 50}]
    s = ab_report.srm(df, variants)
    assert s["ratio_ok"] and s["bucketing_mismatch"] == 200 and not s["ok"]


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
                "degraded_reasons": {"user_state_timeout": 2},
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
        "args": {"concurrency": 16, "canary_percent": 5, "think_ms": [50.0, 300.0]},
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
        "switch": {"full": {"first_page_s": 0.01, "last_other_page_s": None}},
        "ctr_by_page_count": [
            {"pages": [1, 3], "arm": None, "impressions": 100, "clicks": 2, "ctr": 0.02},
            {"pages": [64, None], "arm": None, "impressions": 50, "clicks": 0, "ctr": 0.0},
        ],
        "phases": {"before": {"from": 1000.0, "to": 1100.0, **summary}},
        "failures": [
            {
                "t": 1050.0,
                "surface": "next_page",
                "status": 503,
                "ms": 52.0,
                "degraded_around": {"pages_timeout": 3},
            }
        ],
        "redis": {
            "state": {
                "policy": "noeviction",
                "maxmemory_mb": 1024,
                "evicted_keys": 0,
                "oom_errors": 0,
                "used_memory_mb": 40.0,
            },
            "cache": {
                "policy": "allkeys-lru",
                "maxmemory_mb": 384,
                "evicted_keys": 0,
                "oom_errors": 0,
                "used_memory_mb": 30.0,
            },
        },
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
            },
            {
                "version": "vB",
                "from": None,
                "to": "built",
                "at": 1000.0,
                "actor": "drill",
                "note": "/Users/someone/repo/artifacts/bundles/vB",
            },
        ],
    }
    monkeypatch.setattr(release_report, "BUNDLES", tmp_path)
    md = release_report.render(drill)
    assert (
        "## Canary" in md
        and "Decision: full rollout." in md
        and "| `vB` | canary | active |" in md
        and "evicted 0 keys and refused 0 writes" in md
        and "16 concurrent sessions pausing 50-300 ms between page views" in md
        and "/Users/" not in md
        and "| drill | vB |" in md
        and "| full rollout | 0.01 | none |" in md
        and "CTR: -0.0040, 95% CI [-0.0070, -0.0010] (user-level bootstrap over the two phases' users); lower."
        in md
        and "| 64 and later | 50 | 0 | 0.00% |" in md
        and "| 50.0 | next_page | 503 | 52.0 | pages_timeout 3 |" in md
        and "| user_state_timeout 2 |" in md
    )
