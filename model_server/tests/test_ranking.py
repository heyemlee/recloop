"""Rank (M4): features, ranker scores, pCTR and ad relevance; several bundle versions at once."""

import json
import math
import time
import urllib.request
from pathlib import Path

import faiss
import grpc
import numpy as np
import pytest

from model_server import admin
from model_server import bundle as bundle_mod
from model_server import ranking as rk
from model_server.registry import Registry
from model_server.rpc import pb, pb_grpc
from model_server.server import serve

N, DIM, L = 8, 4, 10
F = {name: i for i, name in enumerate(rk.FEATURES)}


class SumEncoder:
    def __init__(self, vectors):
        self.table = np.vstack([np.zeros((1, vectors.shape[1]), np.float32), vectors])

    def run(self, output_names, feed):
        return [self.table[feed["history"]].sum(axis=1)]


class LinearRanker:
    """Stand-in for the ONNX ranker: score = 2 * sasrec_score (NaN -> 0) + n_sources."""

    def run(self, output_names, feed):
        x = feed["features"].astype(np.float64)
        s = 2 * np.nan_to_num(x[:, F["sasrec_score"]]) + x[:, F["n_sources"]]
        return [s[:, None].astype(np.float32)]


TABLES = {
    "features": list(rk.FEATURES),
    "pop_count": [10, 0, 3, 1, 7, 5, 2, 9],
    "price_bucket": [0, 1, None, 5, 2, 2, 3, 0],
    "cat2": [0, 0, 1, -1, 0, 1, 1, 2],
    "leaf": [0, 1, 2, -1, 0, 3, 3, 4],
}


def pctr_model():
    columns = (
        "sasrec_score",
        "pop_log",
        "n_sources",
        "cat2_overlap",
        "leaf_overlap",
        "sasrec_missing",
        *[f"price_{b}" for b in range(6)],
        "price_unknown",
        "sponsored",
        *[f"pos_{p}" for p in range(1, 11)],
        "pos_11_20",
        "pos_21_up",
    )
    coef = np.zeros(len(columns))
    coef[columns.index("sasrec_score")] = 0.5
    coef[columns.index("pos_1")] = 0.7
    coef[columns.index("pos_3")] = 0.2
    return rk.PCTRModel(
        columns,
        frozenset(columns[:5]),
        np.zeros(len(columns)),
        np.ones(len(columns)),
        coef,
        -4.0,
        3,
        "logreg",
    )


def ctr_pctr_doc() -> dict:
    """pctr.json of a model with the CTR columns: only the day's log-odds and the position."""
    base = pctr_model()
    columns = (*base.columns[:5], *rk.CTR_COLUMNS, *base.columns[5:])
    coef = np.zeros(len(columns))
    coef[columns.index("ictr_24h_logit")] = 1.0
    coef[columns.index("pos_3")] = 0.2
    return {
        "model": "logreg",
        "features": list(rk.FEATURES),
        "columns": list(columns),
        "continuous": [*columns[:5], *rk.CTR_COLUMNS],
        "mean": [0.0] * len(columns),
        "scale": [1.0] * len(columns),
        "coef": coef.tolist(),
        "intercept": 0.0,
        "reference_position": 3,
        "ctr": {"prior": 0.01, "strength": 100.0},
    }


def test_pctr_ctr_columns_follow_the_counts_and_the_golden_rows():
    m = rk.PCTRModel.from_json(ctr_pctr_doc())
    assert m.uses_ctr
    x = np.full((3, len(rk.FEATURES)), np.nan)
    counts = np.asarray([[0, 0, 0, 0], [50, 2, 4000, 200], [np.nan] * 4])
    p = m.predict(x, counts=counts)
    # No impressions: the prior (logit 0.01 + 0.2 at position 3); 200 clicks of 4000: about 0.05.
    expected0 = 1 / (1 + np.exp(-(np.log(0.01 / 0.99) + 0.2)))
    assert p[0] == pytest.approx(expected0) and p[2] == pytest.approx(expected0)
    r1 = (200 + 1.0) / (4000 + 100)
    assert p[1] == pytest.approx(1 / (1 + np.exp(-(np.log(r1 / (1 - r1)) + 0.2))))
    golden = {
        "tolerance": 1e-9,
        "rows": [
            {
                "features": [None] * len(rk.FEATURES),
                "pos": 3,
                "sponsored": 1,
                "counts": c,
                "pctr": v,
            }
            for c, v in ((None, float(p[0])), ([50, 2, 4000, 200], float(p[1])))
        ],
    }
    assert rk.golden_pctr(m, golden)["rows"] == 2
    golden["rows"][1]["counts"] = [50, 2, 4000, 201]
    with pytest.raises(rk.RankingError):
        rk.golden_pctr(m, golden)
    doc = ctr_pctr_doc()
    del doc["ctr"]
    with pytest.raises(rk.RankingError):
        rk.PCTRModel.from_json(doc)


def toy(version="v1", seed=0, ranker=True, pctr=True):
    vectors = np.random.default_rng(seed).normal(size=(N, DIM)).astype(np.float32)
    index = faiss.IndexFlatIP(DIM)
    index.add(vectors)
    ranking = rk.Ranking(TABLES, LinearRanker() if ranker else None, pctr_model() if pctr else None)
    return bundle_mod.Bundle(
        version=version,
        path=Path("/bundles") / version,
        num_items=N,
        dim=DIM,
        history_len=L,
        encoder=SumEncoder(vectors),
        index=index,
        item_ids=np.arange(N),
        item_vectors=vectors,
        ranking=ranking,
    ), vectors


def test_features_follow_the_offline_definitions():
    r = rk.Ranking(TABLES, LinearRanker())
    hits = [
        [("sasrec", 1, 1.5), ("pop", 5, 7.0)],
        [("itemcf", 3, 0.2)],
        [("sasrec", 2, 0.5), ("text", 1, 0.9)],
    ]
    x = r.features([0, 1, 2], [4, 2, 3], hits)
    assert x.dtype == np.float32 and x.shape == (3, len(rk.FEATURES))
    assert x[0, F["sasrec_score"]] == pytest.approx(1.5) and x[0, F["pop_rank"]] == 5
    assert math.isnan(x[1, F["sasrec_score"]]) and x[1, F["itemcf_rank"]] == 3
    assert x[:, F["n_sources"]].tolist() == [2, 1, 1]  # "text" is not a home-feed source
    assert x[0, F["pop_log"]] == pytest.approx(np.log1p(7))
    # item 4: cat2 0 (history: 0, 0, 1) -> 2/3; leaf 0 (history leaves 0, 1, 2) -> 1/3
    assert x[0, F["cat2_overlap"]] == pytest.approx(2 / 3) and x[
        0, F["leaf_overlap"]
    ] == pytest.approx(1 / 3)
    assert math.isnan(x[2, F["cat2_overlap"]]) and math.isnan(x[1, F["price_bucket"]])
    np.testing.assert_allclose(r.scores(x), [2 * 1.5 + 2, 0 + 1, 2 * 0.5 + 1], rtol=1e-6)


def test_pctr_fixes_the_position_at_the_reference():
    m = pctr_model()
    x = np.full((2, len(rk.FEATURES)), np.nan)
    x[:, F["sasrec_score"]] = [0.0, 2.0]
    x[:, F["pop_log"]] = 1.0
    x[:, F["n_sources"]] = 0.0
    x[:, F["price_bucket"]] = [1.0, np.nan]
    at_ref = m.predict(x)
    expected = 1 / (1 + np.exp(-(-4.0 + 0.2 + 0.5 * np.array([0.0, 2.0]))))
    np.testing.assert_allclose(at_ref, expected)
    at_top = m.predict(x, pos=np.array([1, 1]), sponsored=np.array([1, 1]))
    assert (at_top > at_ref).all()  # position 1 has the larger coefficient


def test_golden_checks_refuse_drift():
    r = rk.Ranking(TABLES, LinearRanker(), pctr_model())
    hits = [[["sasrec", 1, 1.5]], [["itemcf", 2, 0.25]]]
    x = r.features([0, 1], [4, 2], hits)
    golden = {
        "feature_tolerance": 1e-5,
        "score_tolerance": 1e-3,
        "queries": [
            {
                "valid_row": 0,
                "history": [0, 1],
                "items": [4, 2],
                "hits": hits,
                "features": [[None if math.isnan(v) else float(v) for v in row] for row in x],
                "scores": r.scores(x).tolist(),
            }
        ],
    }
    assert rk.golden_ranker(r, golden)["max_score_diff"] < 1e-6
    golden["queries"][0]["scores"][0] += 1
    with pytest.raises(rk.RankingError):
        rk.golden_ranker(r, golden)


def start(registry):
    try:
        return serve(registry, "127.0.0.1", 0, workers=2)
    except (RuntimeError, OSError) as e:
        pytest.skip(f"cannot bind a local port: {e}")


def test_grpc_rank_and_ad_relevance():
    b, vectors = toy()
    server, port = start(b)
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as ch:
            s = pb_grpc.ModelServerStub(ch)
            ret = s.Retrieve(
                pb.RetrieveRequest(history=[1, 2], k=3, ad_item_ids=[5, 6, 99]), timeout=5
            )
            user = vectors[1] + vectors[2]
            assert [a.item_id for a in ret.ads] == [5, 6]  # 99 is not in the catalog
            np.testing.assert_allclose(
                [a.score for a in ret.ads], vectors[[5, 6]] @ user, rtol=1e-5
            )
            req = pb.RankRequest(
                history=[1, 2],
                candidates=[
                    pb.Candidate(
                        item_id=4, hits=[pb.SourceHit(source="sasrec", rank=1, score=1.5)]
                    ),
                    pb.Candidate(
                        item_id=2, hits=[pb.SourceHit(source="itemcf", rank=1, score=0.3)]
                    ),
                ],
                ads=[pb.AdCandidate(item_id=5, relevance=0.25), pb.AdCandidate(item_id=6)],
            )
            resp = s.Rank(req, timeout=5)
            assert resp.model_version == "v1" and list(resp.feature_names) == list(rk.FEATURES)
            assert resp.ranker == "lightgbm" and resp.pctr_model == "logreg"
            x = np.asarray(resp.features).reshape(2, -1)
            assert x[0, F["sasrec_score"]] == pytest.approx(1.5)
            np.testing.assert_allclose(resp.scores, [2 * 1.5 + 1, 1.0], rtol=1e-6)
            xa = np.asarray(resp.ad_features).reshape(2, -1)
            assert xa[0, F["sasrec_score"]] == pytest.approx(0.25)  # given by the caller
            assert xa[1, F["sasrec_score"]] == pytest.approx(
                float(vectors[6] @ user), rel=1e-5
            )  # computed here
            assert math.isnan(xa[0, F["sasrec_rank"]]) and xa[0, F["n_sources"]] == 0
            assert len(resp.ad_pctr) == 2 and all(0 < p < 1 for p in resp.ad_pctr)
    finally:
        server.stop(None)


def test_grpc_rank_passes_the_item_counts_to_the_pctr():
    b, _ = toy()
    b.ranking.pctr = rk.PCTRModel.from_json(ctr_pctr_doc())
    server, port = start(b)
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as ch:
            s = pb_grpc.ModelServerStub(ch)
            counts = pb.ItemCounts(
                impressions_hour=50, clicks_hour=2, impressions_day=4000, clicks_day=200
            )
            req = pb.RankRequest(
                history=[1, 2],
                ads=[
                    pb.AdCandidate(item_id=5, relevance=0.25, counts=counts),
                    pb.AdCandidate(item_id=6, relevance=0.25),
                ],
            )
            resp = s.Rank(req, timeout=5)
            r1 = (200 + 1.0) / (4000 + 100)
            assert resp.ad_pctr[0] == pytest.approx(
                1 / (1 + np.exp(-(np.log(r1 / (1 - r1)) + 0.2))), rel=1e-6
            )
            assert resp.ad_pctr[1] == pytest.approx(
                1 / (1 + np.exp(-(np.log(0.01 / 0.99) + 0.2))), rel=1e-6
            )  # no counts: the prior
    finally:
        server.stop(None)


def test_registry_pins_versions_and_switches_the_default(tmp_path):
    v1, _ = toy("v1", seed=1)
    v2, _ = toy("v2", seed=2, ranker=False, pctr=False)
    (tmp_path / "v2").mkdir()
    reg = Registry(tmp_path, v1, loader=lambda path: v2)
    assert reg.load("v2", wait=True) == "ready"
    assert reg.load("../etc", wait=True).startswith("failed")
    assert reg.state("v3") == "absent"
    server, port = start(reg)
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as ch:
            s = pb_grpc.ModelServerStub(ch)
            assert s.Retrieve(pb.RetrieveRequest(history=[1], k=2), timeout=5).model_version == "v1"
            assert (
                s.Retrieve(
                    pb.RetrieveRequest(history=[1], k=2, pin_version="v2"), timeout=5
                ).model_version
                == "v2"
            )
            reg.set_default("v2")
            assert s.Retrieve(pb.RetrieveRequest(history=[1], k=2), timeout=5).model_version == "v2"
            # v2 has the feature tables but no ranker: features only.
            resp = s.Rank(
                pb.RankRequest(history=[1], candidates=[pb.Candidate(item_id=3)], pin_version="v2"),
                timeout=5,
            )
            assert resp.model_version == "v2" and resp.ranker == "" and list(resp.scores) == []
            with pytest.raises(grpc.RpcError) as e:
                s.Rank(pb.RankRequest(history=[1], pin_version="v3"), timeout=5)
            assert e.value.code() == grpc.StatusCode.NOT_FOUND
    finally:
        server.stop(None)
    with pytest.raises(ValueError):
        reg.unload("v2")  # the default
    reg.unload("v1")
    assert [e["version"] for e in reg.status()["loaded"]] == ["v2"]


def test_rank_without_a_ranker_returns_features_only():
    b, _ = toy("v2", ranker=False, pctr=False)
    server, port = start(b)
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as ch:
            resp = pb_grpc.ModelServerStub(ch).Rank(
                pb.RankRequest(
                    history=[1],
                    candidates=[pb.Candidate(item_id=3)],
                    ads=[pb.AdCandidate(item_id=4, relevance=1.0)],
                ),
                timeout=5,
            )
        assert resp.ranker == "" and list(resp.scores) == [] and list(resp.ad_pctr) == []
        assert len(resp.features) == len(rk.FEATURES) and len(resp.ad_features) == len(rk.FEATURES)
    finally:
        server.stop(None)


def test_admin_loads_switches_and_unloads(tmp_path):
    v1, _ = toy("v1")
    v2, _ = toy("v2", seed=3)
    (tmp_path / "v2").mkdir()
    reg = Registry(tmp_path, v1, loader=lambda path: (time.sleep(0.05), v2)[1])
    try:
        httpd = admin.serve_admin(admin.Faults(), "127.0.0.1", 0, reg)
    except OSError as e:
        pytest.skip(f"cannot bind a local port: {e}")
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def call(method, path, body=None):
        req = urllib.request.Request(
            base + path, method=method, data=json.dumps(body).encode() if body is not None else None
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    try:
        assert call("POST", "/bundles", {"version": "v2", "wait": True}) == (
            200,
            {"version": "v2", "state": "ready"},
        )
        assert call("PUT", "/bundles/default", {"version": "v2"}) == (200, {"default": "v2"})
        assert call("DELETE", "/bundles/v2")[0] == 409  # the default
        assert call("DELETE", "/bundles/v1")[0] == 200
        code, status = call("GET", "/bundles")
        assert (
            code == 200
            and status["default"] == "v2"
            and [e["version"] for e in status["loaded"]] == ["v2"]
        )
        assert call("PUT", "/bundles/default", {"version": "nope"})[0] == 404
    finally:
        httpd.shutdown()
