"""The offline replica of the api's fusion, the ranker's features and the ONNX tree export."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.ranking import candidates as cm
from ml.ranking import features as feat
from ml.ranking import onnx_trees


def go_rrf(lists: dict, weights: dict, depth: int) -> list[tuple[int, float]]:
    """mixer.RRF, line by line, for one request: (item, score) best first."""
    score, best, order = {}, {}, []
    for s in cm.SOURCES:
        if s not in lists or weights.get(s, 0) <= 0:
            continue
        for pos, item in enumerate(lists[s]):
            rank = pos + 1
            if item not in score:
                score[item] = 0.0
                best[item] = 1 << 30
                order.append(item)
            score[item] += weights[s] / float(cm.RRF_K + rank)
            best[item] = min(best[item], rank)
    out = sorted(order, key=lambda i: (-score[i], best[i], i))
    return [(i, score[i]) for i in out[:depth]]


def test_fuse_orders_like_the_go_mixer():
    rng = np.random.default_rng(1)
    n, num_items, k = 40, 60, 12
    lists = {}
    for s in cm.SOURCES:
        items = np.stack([rng.permutation(num_items)[:k] for _ in range(n)])
        if s == "itemcf":  # ItemCF lists can be short
            items[:, rng.integers(3, k) :] = -1
        lists[s] = (items, np.where(items >= 0, rng.random((n, k)), np.nan))
    for weights in (cm.CONTROL_WEIGHTS, cm.SASREC_HEAVY_WEIGHTS, {"sasrec": 1.0, "itemcf": 1.0, "pop": 1.0}):
        c = cm.fuse(lists, weights, num_items, depth=20)
        for i in range(n):
            want = go_rrf({s: [int(x) for x in v[0][i] if x >= 0] for s, v in lists.items()}, weights, 20)
            got = [(int(it), float(sc)) for it, sc in zip(c.items[i], c.rrf[i], strict=True) if it >= 0]
            assert [g[0] for g in got] == [w[0] for w in want]
            assert [g[1] for g in got] == [w[1] for w in want]  # same sums in the same order


def test_fuse_keeps_each_sources_rank_and_score():
    lists = {
        "sasrec": (np.array([[5, 3, 7]]), np.array([[0.9, 0.8, 0.1]])),
        "itemcf": (np.array([[3, -1, -1]]), np.array([[2.5, np.nan, np.nan]])),
        "pop": (np.array([[1, 5, 9]]), np.array([[100.0, 90.0, 80.0]])),
    }
    c = cm.fuse(lists, cm.CONTROL_WEIGHTS, 10, depth=6)
    items = [int(x) for x in c.items[0] if x >= 0]
    assert sorted(items) == [1, 3, 5, 7, 9]
    j = items.index(3)
    assert c.ranks["sasrec"][0][j] == 2 and c.ranks["itemcf"][0][j] == 1 and c.ranks["pop"][0][j] == 0
    assert c.scores["itemcf"][0][j] == 2.5 and np.isnan(c.scores["pop"][0][j])
    assert c.n_sources()[0][j] == 2
    assert c.items[0][-1] == -1 and np.isnan(c.rrf[0][-1])  # only 5 distinct items


def test_top_lists_breaks_ties_by_item_id_and_drops_non_positive():
    scores = np.array([[0.0, 2.0, 1.0, 2.0, -1.0]])
    items, top = cm.top_lists(scores, 4)
    assert items.tolist() == [[1, 3, 2, 0]]
    items, top = cm.top_lists(scores, 4, positive_only=True)
    assert items.tolist() == [[1, 3, 2, -1]] and np.isnan(top[0, 3])


def tables() -> feat.ItemTables:
    catalog = pd.DataFrame(
        {
            "item_id": [0, 1, 2, 3, 4],
            "categories": [
                ["Office Products", "Office & School Supplies", "Paper", "Copy Paper"],
                ["Office Products", "Office & School Supplies", "Paper", "Notebooks"],
                ["Office Products", "Office & School Supplies", "Pens"],
                [],
                ["Office Products", "Office & School Supplies", "Paper", "Copy Paper"],
            ],
            "price_cents": pd.array([499, 500, None, 25000, 1999], dtype="Int64"),
        }
    )
    return feat.ItemTables.build(np.array([10.0, 0.0, 3.0, 1.0, 7.0]), catalog)


def test_item_tables():
    t = tables()
    assert t.price_bucket[[0, 1, 3, 4]].tolist() == [0.0, 1.0, 5.0, 2.0] and np.isnan(t.price_bucket[2])
    assert t.cat2[0] == t.cat2[1] == t.cat2[4] != t.cat2[2] and t.cat2[3] == -1
    assert t.leaf[0] == t.leaf[4] != t.leaf[1] and t.leaf[3] == -1
    doc = t.to_json()
    assert doc["price_bucket"][2] is None and doc["features"] == list(feat.FEATURES)


def test_compute_features():
    t = tables()
    c = cm.Candidates(
        items=np.array([[4, 2, 3, -1]]),
        rrf=np.array([[0.03, 0.02, 0.01, np.nan]]),
        ranks={"sasrec": np.array([[1, 0, 2, 0]]), "itemcf": np.array([[0, 3, 0, 0]]), "pop": np.array([[5, 0, 0, 0]])},
        scores={
            "sasrec": np.array([[1.5, np.nan, 0.5, np.nan]]),
            "itemcf": np.array([[np.nan, 0.2, np.nan, np.nan]]),
            "pop": np.array([[7.0, np.nan, np.nan, np.nan]]),
        },
    )
    x = feat.compute(c, [np.array([0, 1, 2])], t)
    f = {name: x[0, :, i] for i, name in enumerate(feat.FEATURES)}
    assert f["sasrec_score"][0] == pytest.approx(1.5) and np.isnan(f["sasrec_score"][1])
    assert f["sasrec_rank"][2] == 2 and np.isnan(f["sasrec_rank"][1])
    assert f["itemcf_rank"][1] == 3 and f["pop_rank"][0] == 5
    assert f["n_sources"][:3].tolist() == [2, 1, 1]
    assert f["pop_log"][0] == pytest.approx(np.log1p(7.0))
    # item 4 (Paper / Copy Paper) against history 0 (Paper / Copy Paper), 1 (Paper / Notebooks), 2 (Pens)
    assert f["cat2_overlap"][0] == pytest.approx(2 / 3) and f["leaf_overlap"][0] == pytest.approx(1 / 3)
    assert f["cat2_overlap"][1] == pytest.approx(1 / 3)  # item 2 (Pens): itself in the history
    assert np.isnan(f["cat2_overlap"][2]) and np.isnan(f["price_bucket"][1])  # no category; no price
    assert np.isnan(x[0, 3]).all()  # padding
    assert x.dtype == np.float32


def test_float32_floor_keeps_the_decision():
    rng = np.random.default_rng(3)
    for t in rng.normal(size=200).tolist() + [0.0, 1.0, 10.5, 2.0**-130]:
        f = onnx_trees.float32_floor(t)
        assert f <= t and np.float32(f) == f
        up = float(np.nextafter(np.float32(f), np.float32(np.inf)))
        assert up > t  # nothing float32 lies between f and t


def test_onnx_ranker_matches_lightgbm():
    lgb = pytest.importorskip("lightgbm")
    pytest.importorskip("onnxruntime")
    rng = np.random.default_rng(7)
    q, d, f = 120, 30, 5
    x = rng.normal(size=(q * d, f)).astype(np.float32)
    x[rng.random(x.shape) < 0.2] = np.nan  # missing values, as for absent sources
    x[:, 4] = np.round(x[:, 4] * 3).astype(np.float32)  # a discrete column, like ranks
    x[:, 4] = np.where(np.isnan(x[:, 4]), 0, x[:, 4])  # never missing: LightGBM type None
    rel = np.nan_to_num(x[:, 0]) + 0.5 * np.nan_to_num(x[:, 1]) + 0.3 * x[:, 4]
    y = np.zeros(q * d, dtype=np.float32)
    for i in range(q):
        y[i * d + int(np.argmax(rel[i * d : (i + 1) * d]))] = 1
    params = {"objective": "lambdarank", "num_leaves": 15, "min_data_in_leaf": 5, "verbose": -1, "num_threads": 1, "deterministic": True}
    booster = lgb.train(params, lgb.Dataset(x, y, group=[d] * q), num_boost_round=40)
    model = onnx_trees.booster_to_onnx(booster, f)
    # Inputs exactly on, just above and just below every threshold, plus NaNs.
    dump = booster.dump_model()
    probes = []
    stack = [t["tree_structure"] for t in dump["tree_info"]]
    while stack:
        node = stack.pop()
        if "split_index" in node:
            t = node["threshold"]
            for v in (t, np.nextafter(np.float32(t), np.float32(np.inf)), np.nextafter(np.float32(t), np.float32(-np.inf))):
                row = rng.normal(size=f).astype(np.float32)
                row[node["split_feature"]] = np.float32(v)
                probes.append(row)
            stack += [node["left_child"], node["right_child"]]
    probe = np.vstack([x, np.asarray(probes, dtype=np.float32), np.full((1, f), np.nan, dtype=np.float32)])
    want = booster.predict(probe, raw_score=True)
    got = onnx_trees.onnx_scores(model.SerializeToString(), probe)
    assert np.abs(got - want).max() < 1e-5
