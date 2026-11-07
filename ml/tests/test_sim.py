"""Simulator (sim/) and SRM check (analysis/srm.py), without the api or a database."""

import json
import random

import numpy as np
import pandas as pd
import pytest

from analysis import srm
from ml.experiment import bucketing
from sim import run as run_mod
from sim import world as world_mod
from sim.client import uuid7, uuid7_ms


def tiny_world():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(30, 8)).astype(np.float32)
    emb[5] = emb[4] + 0.05 * rng.normal(size=8)  # item 5 is close to item 4
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    samples = pd.DataFrame(
        {
            "split": ["train", "test", "test", "test"],
            "user_id": [1, 1, 1, 2],
            "item_id": [0, 4, 7, 9],
            "seq_idx": [0, 1, 2, 0],
        }
    )
    users, relevance = world_mod.build(samples, emb)
    return world_mod.World(users=users, relevance=relevance), emb


def test_world_targets_and_relevance():
    w, _ = tiny_world()
    assert w.users == [(1, [4, 7]), (2, [9])]  # test-period items only, in order
    c = w.click
    assert w.rel(1, 4) == c.rel_target and w.rel(1, 7) == c.rel_target
    assert c.rel_floor < w.rel(1, 5) < c.rel_target  # similar to target 4
    assert w.rel(2, 4) <= w.rel(1, 4)
    assert w.rel(1, 29) >= c.rel_floor
    # examination falls with the position
    assert w.p_click(1, 4, 1) > w.p_click(1, 4, 2) > w.p_click(1, 4, 10)
    assert w.p_click(1, 4, 1) == pytest.approx(c.rel_target)


def test_world_json_round_trip():
    w, _ = tiny_world()
    back = world_mod.from_json(json.loads(json.dumps(world_mod.to_json(w))))
    assert back.users == w.users
    assert back.relevance == w.relevance
    assert back.click == w.click and back.sessions == w.sessions


def test_uuid7():
    ids = [uuid7() for _ in range(100)]
    assert all(i[14] == "7" for i in ids)  # version nibble
    assert all(i[19] in "89ab" for i in ids)  # variant 10
    assert len(set(ids)) == 100
    assert uuid7_ms(uuid7(1_800_000_000_123)) == 1_800_000_000_123


def test_fault_spec():
    f = run_mod.Faults.parse("duplicate=0.1, late=0.05")
    assert f.duplicate == 0.1 and f.late == 0.05 and f.resend == 0
    with pytest.raises(SystemExit):
        run_mod.Faults.parse("explode=1")


class FakeApi:
    """Accepts every POST and records the bodies."""

    def __init__(self):
        self.bodies = []

    def post_json(self, path, doc):
        self.bodies.append(json.loads(json.dumps(doc)))
        return 202, {"accepted": len(doc["events"])}


class NullLedger:
    def write(self, kind, **rec):
        pass


def session(faults: str, seed=1):
    w, _ = tiny_world()
    sim = run_mod.Simulator(w, "http://unused", run_mod.Faults.parse(faults), NullLedger())
    api = FakeApi()
    return run_mod.Session(sim, random.Random(seed), api, run_mod.Stats()), api


def delivered_ids(api):
    return [e["event_id"] for b in api.bodies for e in b["events"]]


def test_faults_change_delivery_not_content():
    s, api = session("")
    evs = [s.event("impression", 1, i, f"r:{i + 1}") for i in range(5)]
    s.deliver(evs)
    s.flush()
    assert delivered_ids(api) == [e["event_id"] for e in evs]

    s, api = session("resend=1")
    evs = [s.event("click", 1, 3, "r:1")]
    s.deliver(evs)
    assert delivered_ids(api) == [evs[0]["event_id"]] * 2  # the same batch twice

    s, api = session("late=1")
    evs = [s.event("impression", 1, i, f"r:{i + 1}") for i in range(3)]
    s.deliver(evs)
    assert api.bodies == []  # held back...
    s.flush()
    assert sorted(delivered_ids(api)) == sorted(e["event_id"] for e in evs)  # ...until the end

    s, api = session("duplicate=1")
    first = [s.event("impression", 1, 1, "r:1")]
    second = [s.event("impression", 1, 2, "r:2")]
    s.deliver(first)
    s.deliver(second)
    s.flush()
    ids = delivered_ids(api)
    assert ids.count(first[0]["event_id"]) == 2 and ids.count(second[0]["event_id"]) == 2


def test_variant_buckets_match_the_go_split():
    exp = {"layer": "recall", "bucket_from": 0, "bucket_to": 1000,
           "variants": [{"name": "control", "share": 50}, {"name": "sasrec_heavy", "share": 50}]}  # fmt: skip
    assert srm.variant_buckets(exp) == {"control": 500, "sasrec_heavy": 500}
    odd = {"layer": "rank", "bucket_from": 100, "bucket_to": 433,
           "variants": [{"name": "a", "share": 33}, {"name": "b", "share": 33}, {"name": "c", "share": 34}]}  # fmt: skip
    vb = srm.variant_buckets(odd)
    assert sum(vb.values()) == 333 and vb == {"a": 109, "b": 110, "c": 114}
    shares = srm.expected_shares([exp, odd])
    assert shares["recall"] == {"control": 0.5, "sasrec_heavy": 0.5}
    assert shares["rank"]["default"] == pytest.approx(0.667)
    assert shares["ads"] == {"default": 1.0}
    # assign() splits [from, to) exactly as Go's pickVariant does
    counts = {"a": 0, "b": 0, "c": 0, "default": 0}
    for u in range(20000):
        counts[srm.assign([odd], "rank", str(u))] += 1
    in_range = sum(
        100 <= bucketing.bucket(bucketing.SALTS["rank"], str(u)) < 433 for u in range(20000)
    )
    assert counts["default"] == 20000 - in_range


def test_chi_square_detects_a_mismatch():
    shares = {"control": 0.5, "treatment": 0.5}
    fine = srm.chi_square({"control": 10_050, "treatment": 9_950}, shares)
    assert not fine["srm"] and fine["p"] > 0.4
    broken = srm.chi_square({"control": 10_400, "treatment": 9_600}, shares)  # 2 % lost on one side
    assert broken["srm"] and broken["p"] < 1e-6
    stray = srm.chi_square({"default": 10, "oops": 1}, {"default": 1.0})
    assert stray["srm"] and stray["unexpected_variants"] == ["oops"]


def test_analyse_flags_users_in_two_groups():
    exp = [{"layer": "recall", "bucket_from": 0, "bucket_to": 1000,
            "variants": [{"name": "control", "share": 50}, {"name": "sasrec_heavy", "share": 50}]}]  # fmt: skip
    rows = []
    for u in range(400):
        v = srm.assign(exp, "recall", str(u))
        rows.append(
            (
                u,
                {
                    layer: {"variant": v if layer == "recall" else "default"}
                    for layer in bucketing.LAYERS
                },
            )
        )
    rep = srm.analyse(rows, exp)
    assert rep["ok"], rep["failed_layers"]
    rows.append(
        (
            7,
            {
                "recall": {
                    "variant": "control"
                    if srm.assign(exp, "recall", "7") != "control"
                    else "sasrec_heavy"
                }
            },
        )
    )
    rep = srm.analyse(rows, exp)
    assert not rep["ok"] and rep["layers"]["recall"]["users_in_several_variants"] == 1
