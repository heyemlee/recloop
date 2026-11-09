"""minionerec.replay's diff helpers: where two beam lists part and by how much (no model)."""

from __future__ import annotations

import json

import pytest

from minionerec import replay


def _shard(path, i, predicts):
    rows = [{"input": f"p{i}-{k}", "predict": p} for k, p in enumerate(predicts)]
    (path / "shards").mkdir(exist_ok=True)
    (path / "shards" / f"{i}.json").write_text(json.dumps(rows))
    return rows


def test_differing_walks_the_shards_in_order(tmp_path):
    a = _shard(tmp_path, 0, [["x", "y"], ["y", "x"]])
    b = _shard(tmp_path, 1, [["x", "y"], ["x", "y"], ["y", "x"]])
    ours = [dict(r) for r in a + b]
    ours[1] = {**ours[1], "predict": ["x", "y"]}  # shard 0, sample 1
    ours[4] = {**ours[4], "predict": ["x", "y"]}  # shard 1, sample 2
    assert replay.differing(ours, tmp_path, [(0, 2), (2, 5)]) == {0: [1], 1: [2]}
    # A replay cut short covers the first shards only.
    assert replay.differing(ours[:3], tmp_path, [(0, 2), (2, 5)]) == {0: [1]}
    ours[2] = {**ours[2], "input": "another prompt"}
    with pytest.raises(SystemExit, match="not the reference's prompt"):
        replay.differing(ours, tmp_path, [(0, 2), (2, 5)])


def test_swap_reports_the_first_parting_and_both_gaps():
    ref = (["a", "b", "c", "d"], [-1.0, -2.0, -2.00001, -3.0])
    s = replay.swap(ref, (["a", "b", "d", "c"], [-1.0, -2.0, -2.0, -2.000004]))
    assert s["position"] == 3 and (s["reference_sid"], s["replay_sid"]) == ("c", "d")
    assert s["gap_reference"] == pytest.approx(-2.00001 - -3.0)
    assert s["gap_replay"] == pytest.approx(-2.0 - -2.000004)
    assert (s["reference_position_of_replay_sid"], s["replay_position_of_reference_sid"]) == (4, 4)
    # A SID missing from the other list leaves its gap open.
    s = replay.swap(ref, (["a", "e", "b", "c"], [-1.0, -1.5, -2.0, -2.1]))
    assert s["position"] == 2 and s["gap_reference"] is None
    assert s["gap_replay"] == pytest.approx(-1.5 - -2.0)
    assert replay.swap(ref, ref) == {"position": None}
    # The largest move of one SID's score between the paths, over the SIDs both have.
    ours = (["a", "d", "c", "e"], [-1.25, -2.5, -2.0, -9.0])
    assert replay.score_difference((["a", "c", "d"], [-1.0, -2.0, -3.0]), ours) == 0.5
    assert replay.score_difference(ref, (["e"], [0.0])) == 0.0


def test_score_differences_are_per_shared_sid():
    ref = (["a", "b", "c"], [-1.0, -2.0, -3.0])
    ours = (["c", "a", "d"], [-2.5, -1.25, -4.0])
    assert replay.score_differences(ref, ours) == [0.25, 0.5]
    q = replay._quantiles([0.1, 0.4, 0.2, 0.3])
    assert q == {"count": 4, "median": 0.3, "p90": 0.4, "p99": 0.4, "max": 0.4}


def test_near_tie_needs_both_gaps_within_the_samples_own_moves():
    assert replay.near_tie({"gap_reference": 0.1, "gap_replay": 0.2}, [0.0, 0.25])
    assert not replay.near_tie({"gap_reference": 0.1, "gap_replay": 0.3}, [0.0, 0.25])
    assert not replay.near_tie({"gap_reference": None, "gap_replay": 0.01}, [0.25])
    assert not replay.near_tie({"gap_reference": 0.0, "gap_replay": 0.01}, [])
