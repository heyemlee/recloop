"""The billing runs' logic without the stack: ad_token helpers, the truth of sim.billing_verify,
the chaos plan, the overdelivery summary and the report (analysis/billing_report.py)."""

import base64
import json
import random

import pytest

from analysis import billing_report
from sim import ads as ads_mod
from sim import billing_drill, billing_verify, overdelivery
from sim import run as run_mod


def b64(doc: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(doc).encode()).rstrip(b"=").decode()


def token(
    rid="01a1b88d-0000-7000-8000-000000000001",
    pos=3,
    cid=7,
    item=42,
    price=31,
    uid=5,
    iat=1_794_000_000,
):
    claims = {"rid": rid, "pos": pos, "cid": cid, "item": item, "price": price, "uid": uid,
              "iat": iat, "exp": iat + 1800}  # fmt: skip
    return f"{b64({'alg': 'HS256', 'kid': 't1'})}.{b64(claims)}.c2lnbmF0dXJl"


def test_token_claims_and_tamper():
    tok = token()
    c = ads_mod.claims(tok)
    assert (c["rid"], c["pos"], c["cid"], c["price"]) == (
        "01a1b88d-0000-7000-8000-000000000001",
        3,
        7,
        31,
    )
    forged = ads_mod.tamper(tok, random.Random(1))
    assert forged != tok
    head, payload, sig = forged.split(".")
    assert (head, sig) == (tok.split(".")[0], tok.split(".")[2])  # only the payload changed
    assert sum(a != b for a, b in zip(payload, tok.split(".")[1], strict=True)) == 1
    assert ads_mod.utc_day(1_794_000_000) == "2026-11-06"  # 21:20 UTC


def click(event_id, tok, forged=False):
    return {
        "event_id": event_id,
        "type": "click",
        "ad_token": tok,
        **({"forged": True} if forged else {}),
    }


def test_truth_one_charge_per_clicked_impression():
    a, b = token(pos=3), token(pos=9, cid=8, price=5)
    clicks = [
        click("e1", a),
        click("e2", a),  # a double click: the same impression
        click("e3", ads_mod.tamper(b, random.Random(2)), forged=True),  # never billed
        click("e4", token(pos=4, iat=1_794_000_000 - 3600)),  # received after exp
        click("e5", b),
    ]
    received = {"e1": 1_794_000_010, "e2": 1_794_000_011, "e4": 1_794_000_100, "e5": 1_794_001_799}
    charged, status = billing_verify.expected_charges(clicks, received)
    assert status == {"e1": "valid", "e2": "valid", "e3": "invalid", "e4": "expired", "e5": "valid"}
    rid = "01a1b88d-0000-7000-8000-000000000001"
    assert set(charged) == {f"{rid}:3", f"{rid}:9"}
    assert charged[f"{rid}:3"] == {
        "campaign_id": 7,
        "day": "2026-11-06",
        "price": 31,
        "clicks": {"e1", "e2"},
    }
    assert charged[f"{rid}:9"]["clicks"] == {"e5"}  # the forged click on the same slot is not one


def test_chaos_plan():
    steps = billing_drill.plan(1800, seed=3)
    assert steps == billing_drill.plan(1800, seed=3)  # the seed fixes it
    kinds = [s["action"] for s in steps]
    assert (
        kinds.count("kill_billing") == 8
        and kinds.count("restart_redpanda") == 3
        and kinds.count("pg_slow") == 2
    )
    assert all(0 < s["t"] < 1800 for s in steps)
    assert [s["t"] for s in steps] == sorted(s["t"] for s in steps)
    kills = [s["t"] for s in steps if s["action"] == "kill_billing"]
    assert min(b - a for a, b in zip(kills, kills[1:], strict=False)) >= 0.06 * 1800
    slow = [s for s in steps if s["action"] == "pg_slow"]
    assert all(s["seconds"] == 180 for s in slow)
    assert slow[1]["t"] - slow[0]["t"] >= 180  # the two slow windows do not overlap
    assert all(2 <= s["down"] <= 10 for s in steps if s["action"] == "kill_billing")


def test_overdelivery_summary():
    assert overdelivery.parse_config("20p") == (20, True) and overdelivery.parse_config("5") == (
        5,
        False,
    )
    camps = [
        {"budget": 1000, "billed": 1000, "unbilled": 31, "clicks": 30, "zero_billed": 1, "ran_out_at": 0.5, "stopped_at": 0.52},
        {"budget": 500, "billed": 200, "unbilled": 0, "clicks": 8, "zero_billed": 0, "ran_out_at": None, "stopped_at": 1.0},
    ]  # fmt: skip
    s = overdelivery.summarize(camps)
    assert s["overdelivery"] == pytest.approx(31 / 1500, abs=1e-4)
    assert s["spent"] == pytest.approx(1200 / 1500, abs=1e-4)
    assert (s["exhausted"], s["campaigns"], s["ran_out_median"], s["stopped_median"]) == (
        1,
        2,
        0.5,
        0.76,
    )
    assert (s["clicks"], s["zero_billed"]) == (38, 1)


def test_forged_clicks_in_the_simulator():
    """With forge=1 every sponsored click carries a tampered token, marked in the ledger."""
    from ml.tests.test_sim import tiny_world

    class Ledger:
        def __init__(self):
            self.rows = []

        def write(self, kind, **rec):
            self.rows.append({"kind": kind, **rec})

    w, _ = tiny_world()
    ledger = Ledger()
    sim = run_mod.Simulator(w, "http://unused", run_mod.Faults.parse("forge=1"), ledger)
    s = run_mod.Session(sim, random.Random(1), None, run_mod.Stats())
    tok = token()
    ev = s.click(1, {"item_id": 4, "impression_id": "r:3", "ad_token": tok})
    assert ev["ad_token"] != tok and ledger.rows[-1]["forged"] is True
    organic = s.click(1, {"item_id": 4, "impression_id": "r:2"})
    assert "ad_token" not in organic and "forged" not in ledger.rows[-1]
    assert s.stats.faults["forge"] == 1 and s.stats.ads["clicks"] == 1


def drill_doc():
    checks = [{"check": c, "expected": 0, "actual": 0, "ok": True, "detail": ""}
              for c in ("charges: clicked impressions with no charge", "amounts: charges above budget",
                        "budget rows = ledger", "Redis spend = ledger")]  # fmt: skip
    return {
        "ok": True, "minutes": 30, "seed": 1, "faults": "duplicate=0.1", "pg_latency_ms": 200,
        "concurrency": 12, "think_ms": "50,300", "freq_cap": 20, "caught_up_seconds": 4.2,
        "chaos": [{"t": 60.0, "action": "kill -9 billing"}, {"t": 64.0, "action": "started billing (4.0 s down)"},
                  {"t": 300.0, "action": "restarted redpanda"}, {"t": 400.0, "action": "postgres +200 ms for billing"},
                  {"t": 580.0, "action": "postgres back to normal (180 s)"}],
        "chaos_plan": [{"t": 60.0, "action": "kill_billing", "down": 4.0}, {"t": 400.0, "action": "pg_slow", "seconds": 180}],
        "campaign_settings": [{"campaign_id": 3, "bid": 60, "budget": 1500}],
        "sim": {"started": 1_794_000_000, "events_unique": 1000, "event_copies_delivered": 1200,
                "page_views": {"home": 80, "next_page": 10, "item": 10}},
        "campaigns": 1, "slots_served": 150, "sponsored_impressions": 120, "sponsored_click_events": 12,
        "forged_click_events": 1, "valid_clicks": 11, "expired_clicks": 0, "charges_expected": 10,
        "charges": 10, "repeat_clicks": 1, "billed_cents": 1500, "price_cents": 1560, "unbilled_cents": 60,
        "zero_billed": 1, "budgets_exhausted": 1,
        "days": [{"campaign_id": 3, "day": "2026-11-07", "budget": 1500, "spent": 1500, "charges": 10,
                  "price": 1560, "billed": 1500, "zero_billed": 1, "partial": 1, "redis": 1500}],
        "freq": {"user_campaign_days": 100, "max": 21, "histogram": {"1": 90, "20": 8, "21": 2}},
        "checks": checks,
        "reconcile": {"days": [{}], "match": 1, "redis_behind": 0, "repaired_cents": 0, "redis_ahead": 0, "row_mismatch": 0},
        "reconciler_during": {"passes_with_repairs": 1, "repaired_cents": 51, "billing_starts": 2},
        "pipeline": {"checks": checks, "ok": True},
    }  # fmt: skip


def over_doc():
    def res(m, p, over):
        camp = {"campaign_id": 1, "budget": 2500, "slots_served": 900, "clicks": 60, "billed": 2500,
                "unbilled": over, "zero_billed": 1, "ran_out_at": 0.5, "stopped_at": 0.5,
                "spend_curve": {"0.25": 0.5, "0.5": 1.0, "0.75": 1.0, "1.0": 1.0}}  # fmt: skip
        return {"config": f"{m}{'p' if p else ''}", "margin": m, "pacing": p, "t0": 1_794_000_000,
                "campaigns": [camp], "summary": overdelivery.summarize([camp])}  # fmt: skip

    return {
        "window_s": 240, "tail_s": 30, "concurrency": 400, "dwell_ms": "1000,9000", "think_ms": "0,0",
        "seed": 1, "campaigns": [{"bid": 50, "budget": 2500}], "items_per_campaign": 10, "freq_cap": 100,
        "results": [res(0, False, 82), res(0, True, 41), res(5, False, 0)],
    }  # fmt: skip


def test_report_renders_both_runs_deterministically():
    text = billing_report.render(drill_doc(), over_doc())
    assert text == billing_report.render(drill_doc(), over_doc())
    assert (
        "**The ledger equals the simulator's truth charge by charge**: 10 expected, 10 in the ledger"
        in text
    )
    assert (
        "**0 campaign-days with charges above budget**; the budget rows and the Redis counters equal the ledger"
        in text
    )
    assert (
        "0 clicks 3.28 % (the first campaign spent 100 % of its budget), 5 clicks 0.00 % (the first "
        "campaign spent 100 % of its budget); with pacing: 0 clicks 1.64 % (the first campaign spent "
        "100 % of its budget)"
    ) in text
    assert "| 60.0 | kill -9 billing |" in text
    assert (
        "| 3 | 2026-11-07 | 60 ¢ | $15.00 | 10 | $15.60 | $15.00 | $0.60 | 1 | $15.00 | $15.00 |"
        in text
    )
    assert "2 went over it (at most 21)" in text
    assert "added 51 ¢" in text
    assert "| 0 | on | 1.64 % | $0.41 | 1 | 100.0 % | 1 / 1 | 100 % | 50 % | 50 % | 60 |" in text
    assert "margin 0: $0.82 not billed, 1 clicks billed 0" in text
    failed = drill_doc()
    failed["ok"] = False
    failed["checks"][0] = {**failed["checks"][0], "actual": 2, "ok": False}
    assert "**1 ledger check(s) failed**" in billing_report.render(failed, None)
    assert "Not run yet: `make overdelivery`." in billing_report.render(failed, None)


def split_redis(state_evicted=0, oom=0, cache_evicted=12):
    return {
        "state": {"maxmemory_mb": 1024, "policy": "noeviction", "evicted_keys": state_evicted,
                  "oom_errors": oom, "used_memory_mb": 117.5},
        "cache": {"maxmemory_mb": 384, "policy": "allkeys-lru", "evicted_keys": cache_evicted,
                  "oom_errors": 0, "used_memory_mb": 300.0},
    }  # fmt: skip


def test_report_names_each_runs_redis_setup():
    # Runs before the split recorded no setup: the single instance of then is named.
    text = billing_report.render(drill_doc(), over_doc())
    assert "on one Redis instance (384 MB, `volatile-ttl`" in text
    # A drill on the two instances: their setup, and what it did to each.
    d = drill_doc()
    d["redis"], d["candidate_cache"] = split_redis(), True
    text = billing_report.render(d, over_doc())
    assert (
        "on two Redis instances (design §19, 2026-11-08): state (`noeviction`, 1024 MB) and cache "
        "(`allkeys-lru`, 384 MB), with the api's candidate cache on."
    ) in text
    assert "It evicted 0 keys and refused 0 writes during the run and held 117.5 MB" in text
    assert (
        "the state Redis evicted 0 keys and refused 0 writes during it, the cache Redis evicted 12."
        in text
    )


def test_state_redis_must_not_evict_or_refuse():
    from sim import redis_stats

    before = split_redis(cache_evicted=0)
    after = split_redis(cache_evicted=40)
    d = redis_stats.delta(before, after)
    assert d["cache"]["evicted_keys"] == 40 and redis_stats.state_ok(d)
    assert not redis_stats.state_ok(redis_stats.delta(before, split_redis(state_evicted=1)))
    assert not redis_stats.state_ok(redis_stats.delta(before, split_redis(oom=3)))
    assert "state Redis (noeviction, 1024 MB): evicted 0, refused 0" in redis_stats.describe(d)


def test_compose_project_of_this_checkout(tmp_path, monkeypatch):
    from sim import compose

    env = tmp_path / ".env"
    monkeypatch.setattr(compose, "ENV_FILE", env)
    monkeypatch.delenv("COMPOSE_PROJECT_NAME", raising=False)
    assert compose.project() == "recloop"  # no .env: the shared stack
    env.write_text("# private stack\nRECLOOP_REDIS_PORT=26392\nCOMPOSE_PROJECT_NAME=recloop-x\n")
    assert compose.project() == "recloop-x"
    cmd = compose.command("bench")
    assert cmd[:4] == ["docker", "compose", "-p", "recloop-x"]
    assert ["--env-file", str(env)] == cmd[6:8] and cmd[-2:] == ["--profile", "bench"]
    assert compose.container("redis-cache") == "recloop-x-redis-cache-1"
    assert compose.service_of("recloop-x-redis-cache-1") == "redis-cache"
    monkeypatch.setenv("COMPOSE_PROJECT_NAME", "recloop-y")  # the environment wins
    assert compose.project() == "recloop-y"
