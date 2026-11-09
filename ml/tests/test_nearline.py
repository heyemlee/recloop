"""Nearline generation (design §11 step 5): payload format, SID -> item mapping, the worker's
Redis round, and the report's freshness arithmetic.

The worker test needs a Redis it may write to: set RECLOOP_TEST_REDIS_URL (e.g. a private
stack's state instance, database 15); it flushes that database.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from analysis import nearline_report as rep
from ml import paths
from ml.nearline import keys, worker

PAYLOAD = paths.ROOT / "testdata" / "gen_payload.json"


def test_payload_is_the_one_go_reads():
    # serving/internal/userstate/gen_test.go decodes the same file.
    got = keys.encode(12, 1762600000123, [5, 17, 3458], [-0.5, -1.25, -7.0625])
    assert got == PAYLOAD.read_text().strip()
    assert keys.decode(got)["items"] == [5, 17, 3458]
    with pytest.raises(ValueError):
        keys.encode(1, 0, [1, 2], [0.0])


def test_key_names_match_go():
    assert keys.result_key(7, "v1") == "gen:7:v1"
    assert keys.pending_key(7) == "gen:pending:7"
    assert (keys.WORKER_KEY, keys.QUEUE_KEY) == ("gen:worker", "gen:queue")
    assert (keys.DEBOUNCE_MS, keys.MAX_WAIT_MS) == (
        30_000,
        120_000,
    )  # userstate.GenDebounce, GenMaxWait


def test_parse_changes():
    assert keys.parse_changes("") == []
    assert keys.parse_changes("10:12,20:25") == [(10, 12), (20, 25)]


def test_shared_sids_expand_in_tie_order_and_stop_at_k():
    sids = ["<a>", "<b>", "<a>", "<c>"]  # items 0 and 2 share <a>
    by_sid = worker.sid_items(sids, tie_order=[2, 0, 3, 1])
    assert by_sid == {"<a>": [2, 0], "<c>": [3], "<b>": [1]}
    items, scores = worker.to_items(
        ['"<a>"\n', "<c>", "<b>", "<x>"], [-1.0, -2.0, -3.0, -4.0], by_sid
    )
    assert items == [2, 0, 3, 1]
    assert scores == [-1.0, -1.0, -2.0, -3.0]
    items, _ = worker.to_items(["<a>", "<c>"], [-1.0, -2.0], by_sid, k=2)
    assert items == [2, 0]


class FakeGenerator:
    """minionerec.serve_gen's answers: every history gets the same SIDs."""

    def __init__(self, sids):
        self.sids, self.calls = sids, []

    def generate(self, histories):
        self.calls.append(histories)
        return {
            "sids": [list(self.sids) for _ in histories],
            "scores": [[-float(i) for i in range(len(self.sids))] for _ in histories],
            "batches": [{"size": len(histories), "seconds": 0.01}],
        }


@pytest.fixture
def test_redis():
    url = os.environ.get("RECLOOP_TEST_REDIS_URL")
    if not url:
        pytest.skip("set RECLOOP_TEST_REDIS_URL to a Redis database this test may flush")
    import redis

    r = redis.Redis.from_url(url)
    r.flushdb()
    yield url, r
    r.flushdb()


def test_worker_round_claims_generates_and_keeps_the_newest(tmp_path, test_redis):
    url, r = test_redis
    sids = worker.item_sids()
    pop = {"items": list(range(len(sids))), "counts": [0] * len(sids)}
    (tmp_path / "bundle").mkdir()
    (tmp_path / "bundle" / "pop.json").write_text(json.dumps(pop))
    args = SimpleNamespace(
        redis_url=url,
        bundle=tmp_path / "bundle",
        only_users=None,
        out=tmp_path / "out",
        claim=8,
        max_batch=8,
    )
    w = worker.Worker(args)
    w.version = "vtest"
    now = worker.redis_ms(r)
    # user 1: history 3 items, hv 5, two changes, due; user 2: no history; user 3: not due yet.
    r.zadd("u:1:hist", {"10": 1, "11": 2, "12": 3})
    r.set("u:1:hv", 5)
    r.rpush("gen:pending:1", f"{now - 40_000}:{now - 39_000}", f"{now - 31_000}:{now - 30_500}")
    r.zadd("gen:queue", {"1": now - 500, "2": now - 100, "3": now + 60_000})
    gen = FakeGenerator([sids[7], sids[8]])
    assert w.round(gen) == 2
    assert gen.calls == [[[sids[10], sids[11], sids[12]]]]
    got = keys.decode(r.get("gen:1:vtest"))
    want, _ = worker.to_items([sids[7], sids[8]], [0.0, -1.0], w.by_sid)
    assert got["hv"] == 5 and got["items"] == want and got["scores"][0] == 0.0
    assert r.ttl("gen:1:vtest") > 0
    assert not r.exists("gen:pending:1") and r.zscore("gen:queue", "3") is not None
    assert r.zscore("gen:queue", "1") is None
    lines = [json.loads(x) for x in (tmp_path / "out" / "jobs.jsonl").read_text().splitlines()]
    by_uid = {j["uid"]: j for j in lines}
    assert by_uid[2]["status"] == "no_history"
    assert by_uid[1]["status"] == "written" and len(by_uid[1]["changes"]) == 2
    assert by_uid[1]["written_ms"] >= by_uid[1]["claimed_ms"] >= by_uid[1]["due_ms"]

    # A result from a newer history version is never overwritten by an older one.
    r.set("gen:1:vtest", keys.encode(9, now, [1], [0.0]))
    r.zadd("gen:queue", {"1": now - 1})
    assert w.round(gen) == 1
    assert keys.decode(r.get("gen:1:vtest"))["hv"] == 9
    last = json.loads((tmp_path / "out" / "jobs.jsonl").read_text().splitlines()[-1])
    assert last["status"] == "older"

    # --only-users: other users' changes are claimed and dropped.
    w.only = {99}
    r.zadd("gen:queue", {"1": now - 1})
    assert w.round(gen) == 0 and w.counts["dropped"] == 1


def test_freshness_counts_each_change_once_from_apply_and_from_the_event():
    t0 = 1_000.0
    jobs = [
        {"status": "written", "due_ms": 1_031_000, "claimed_ms": 1_032_000, "written_ms": 1_040_000,
         "changes": [[1_000_000, 1_001_000], [1_009_000, 1_010_000]]},
        {"status": "no_history", "changes": [[1_002_000, 1_002_500]]},
        {"status": "written", "due_ms": 0, "claimed_ms": 0, "written_ms": 999_000, "changes": []},
    ]  # fmt: skip
    served = [
        {"arm": "gen", "gen": {"status": "fresh", "age_ms": 2000}},
        {"arm": "gen", "gen": {"status": "stale", "lag": 1, "age_ms": 4000}},
        {"arm": "gen", "gen": {"status": "missing"}},
        {"arm": "control", "gen": None},
    ]
    f = rep.freshness(jobs, [{"t": 1_005_000, "queued": 3, "due": 1}], served, t0, 1_100.0)
    assert f["since_applied_s"]["n"] == 2
    assert f["since_applied_s"]["max"] == pytest.approx(39.0)
    assert f["since_event_s"]["max"] == pytest.approx(40.0)
    assert f["event_to_applied_ms"]["max"] == pytest.approx(1000.0)
    assert f["queue_wait_s"]["p50"] == pytest.approx(1.0)
    assert f["served"]["requests"] == 3
    assert f["served"]["fresh"] == pytest.approx(1 / 3)
    assert f["stale_lag"] == {1: 1}
    assert f["queue_max"] == 3


def test_session_gaps_ignore_page_views_of_one_visit():
    ledger = [
        {"kind": "page", "user_id": 1, "t": 0.0},
        {"kind": "page", "user_id": 1, "t": 1.0},
        {"kind": "page", "user_id": 1, "t": 100.0},
        {"kind": "page", "user_id": 2, "t": 5.0},
        {"kind": "event", "user_id": 2, "t": 500.0},
    ]
    g = rep.session_gaps(ledger)
    assert g["n"] == 1 and g["max"] == pytest.approx(99.0)
