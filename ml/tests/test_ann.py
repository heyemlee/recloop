import json
import sys

import faiss
import numpy as np
import pytest

from ml.ann import bench, incremental, indexes, metrics, report, vectors


def _unit(rng, n, d=16):
    x = rng.standard_normal((n, d))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


# ---------------------------------------------------------------- ground truth and recall


def test_exact_topk_matches_full_sort_and_chunking():
    rng = np.random.default_rng(0)
    data, q = _unit(rng, 500), _unit(rng, 37)
    ids, scores = metrics.exact_topk(data, q, 20)
    full = q.astype(np.float64) @ data.astype(np.float64).T
    expected = np.argsort(-full, axis=1, kind="stable")[:, :20]
    np.testing.assert_array_equal(ids, expected)
    np.testing.assert_allclose(scores, np.take_along_axis(full, expected, axis=1))
    small_ids, _ = metrics.exact_topk(data, q, 20, chunk_bytes=8 * 500 * 3)  # 3 queries/chunk
    np.testing.assert_array_equal(small_ids, ids)


def test_exact_topk_orders_ties_by_id():
    data = np.array([[1, 0], [0, 1], [1, 0], [1, 0]], dtype=np.float32)
    ids, _ = metrics.exact_topk(data, np.array([[1, 0]], dtype=np.float32), 3)
    np.testing.assert_array_equal(ids[0], [0, 2, 3])


def test_recall_equals_intersection_without_ties():
    rng = np.random.default_rng(1)
    data, q = _unit(rng, 400), _unit(rng, 50)
    truth, scores = metrics.exact_topk(data, q, 10)
    worst, _ = metrics.exact_topk(data, -q, 3)  # the 3 lowest-scoring items of each query
    found = truth.copy()
    found[:, 7:] = worst  # an ANN that misses the last 3 neighbours
    found[0, :] = -1  # nothing returned for query 0
    found[1, 5:] = -1  # only 5 returned for query 1
    got = metrics.recall_at_k(metrics.exact_scores(data, q, found), scores[:, 9], 10)
    np.testing.assert_allclose(got, metrics.intersection_recall(found, truth, 10))
    assert got[0] == 0.0
    assert got[1] == 0.5


def test_recall_counts_items_tied_with_the_kth():
    # Items 1 and 2 are the same vector, tied at the 2nd place: either one is a correct answer.
    data = np.array([[1, 0], [0.8, 0.6], [0.8, 0.6], [0, 1]], dtype=np.float32)
    q = np.array([[1, 0]], dtype=np.float32)
    truth, scores = metrics.exact_topk(data, q, 2)
    np.testing.assert_array_equal(truth[0], [0, 1])
    found = np.array([[0, 2]])
    assert metrics.recall_at_k(metrics.exact_scores(data, q, found), scores[:, 1], 2)[0] == 1.0
    assert metrics.intersection_recall(found, truth, 2)[0] == 0.5
    worse = np.array([[0, 3]])
    assert metrics.recall_at_k(metrics.exact_scores(data, q, worse), scores[:, 1], 2)[0] == 0.5


def test_has_duplicates_ignores_padding():
    assert not metrics.has_duplicates(np.array([[3, 1, -1, -1], [0, 1, 2, 3]]))
    assert metrics.has_duplicates(np.array([[3, 1, 3, -1]]))


def test_exact_scores_marks_missing_results():
    data = np.eye(3, dtype=np.float32)
    got = metrics.exact_scores(data, data[:2], np.array([[0, -1], [2, 1]]))
    np.testing.assert_array_equal(got, [[1.0, -np.inf], [0.0, 1.0]])


# ---------------------------------------------------------------- list balance


def test_list_balance_definitions():
    b = metrics.list_balance(np.array([0, 2, 4, 6]))
    assert b["mean"] == 3.0
    assert b["max"] == 6 and b["max_over_mean"] == 2.0 and b["empty"] == 1
    assert b["cv"] == pytest.approx(np.sqrt(5.0) / 3.0)
    assert b["imbalance"] == pytest.approx(1 + b["cv"] ** 2)
    even = metrics.list_balance(np.full(8, 5))
    assert even["cv"] == 0.0 and even["imbalance"] == 1.0


def test_list_balance_matches_faiss_imbalance_factor():
    rng = np.random.default_rng(2)
    x = _unit(rng, 3000)
    built = indexes.build_ivf_flat(x, 16, seed=3)
    sizes = indexes.list_sizes(built.index)
    assert sizes.sum() == 3000
    b = metrics.list_balance(sizes)
    assert b["imbalance"] == pytest.approx(built.index.invlists.imbalance_factor())


def test_latency_summaries():
    assert metrics.percentiles_us(np.arange(1, 101) * 1000, (50, 99)) == pytest.approx(
        [50.5, 99.01]
    )
    assert metrics.median_of_runs([[1, 10, 3], [3, 30, 1], [2, 20, 2]]) == [2, 20, 2]


# ---------------------------------------------------------------- vectors


def _two_clusters(rng, n=600):
    means = np.array([[3.0, 0.0], [-3.0, 1.0]])
    labels = (rng.random(n) < 0.3).astype(int)
    return means[labels] + 0.5 * rng.standard_normal((n, 2)), means


def test_mixture_em_recovers_a_separated_mixture():
    rng = np.random.default_rng(4)
    x, means = _two_clusters(rng)
    fit = vectors.fit_mixture(x, 2, reg=0.0, seed=5)
    got = fit.mixture.means[np.argsort(fit.mixture.means[:, 0])[::-1]]
    np.testing.assert_allclose(got, means, atol=0.15)
    assert sorted(fit.mixture.weights) == pytest.approx([0.3, 0.7], abs=0.05)
    assert np.all(np.diff(fit.trace) > -1e-9)  # EM never lowers the likelihood
    assert fit.mixture.log_prob(x).mean() == pytest.approx(fit.log_likelihood)


def test_select_mixture_prefers_the_right_size():
    rng = np.random.default_rng(6)
    x, _ = _two_clusters(rng)
    fit, trials = vectors.select_mixture(x, ks=(1, 2), regs=(1e-3,), seed=7)
    assert fit.mixture.k == 2
    assert {t["k"] for t in trials} == {1, 2}


def test_mixture_sampling_is_seeded_and_respects_components():
    mix = vectors.synthetic_mixture(dim=8, k=5, seed=8)
    a, la = vectors.make_catalog(mix, 1000, seed=9)
    b, lb = vectors.make_catalog(mix, 1000, seed=9)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(la, lb)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-6)
    _, lc = vectors.make_catalog(mix, 500, seed=10, components=np.array([1, 3]))
    assert set(np.unique(lc)) <= {1, 3}
    assert vectors.array_digest(a) == vectors.array_digest(b)


def test_make_queries():
    rng = np.random.default_rng(11)
    pool = _unit(rng, 300, d=8)
    labels = rng.integers(0, 4, size=300)
    q, home = vectors.make_queries(pool, labels, 200, seed=12)
    q2, _ = vectors.make_queries(pool, labels, 200, seed=12)
    np.testing.assert_array_equal(q, q2)
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, rtol=1e-6)
    _, home = vectors.make_queries(pool, labels, 200, seed=13, homes=np.array([2]))
    assert set(np.unique(home)) == {2}
    # A history of one item is that item.
    one, _ = vectors.make_queries(pool, labels, 50, seed=14, params=vectors.QueryParams(max_len=1))
    best = (one @ pool.T).max(axis=1)
    np.testing.assert_allclose(best, 1.0, rtol=1e-6)


def test_participation_ratio():
    assert vectors.participation_ratio(np.ones(10)) == pytest.approx(10)
    assert vectors.participation_ratio(np.array([1.0, 0.0, 0.0])) == pytest.approx(1)


# ---------------------------------------------------------------- indexes and the harness


def test_flat_and_exhaustive_ivf_are_exact_and_timing_returns_search_results():
    rng = np.random.default_rng(15)
    data, q = _unit(rng, 2000), _unit(rng, 40)
    run = bench.Run(k=10, runs=2, warmup=5, build_threads=2)
    kth = bench.kth_scores(data, q, run.k)
    flat = indexes.build_flat(data)
    assert bench.evaluate(flat.index, data, q, kth, run)["recall"] == 1.0
    ivf = indexes.build_ivf_flat(data, 8, seed=1)
    indexes.set_param(ivf.index, "nprobe", 8)  # all lists: exhaustive
    res = bench.evaluate(ivf.index, data, q, kth, run)
    assert res["recall"] == 1.0
    assert res["ndis"] == 2000  # every vector compared once per query
    assert len(res["runs_us"]) == 2
    t = indexes.time_queries(ivf.index, q, run.k, runs=1, warmup=0)
    np.testing.assert_array_equal(t.ids, ivf.index.search(q, run.k)[1])


def test_hnsw_and_refined_pq_reach_high_recall():
    rng = np.random.default_rng(16)
    data, q = _unit(rng, 4000), _unit(rng, 30)
    run = bench.Run(k=10, runs=1, warmup=0)
    kth = bench.kth_scores(data, q, run.k)
    h = indexes.build_hnsw(data, 16, 80)
    indexes.set_param(h.index, "efSearch", 200)
    assert bench.evaluate(h.index, data, q, kth, run)["recall"] > 0.9
    with pytest.raises(RuntimeError):
        h.index.remove_ids(faiss.IDSelectorBatch(np.array([1], dtype=np.int64)))
    pq = indexes.build_ivf_pq(data, 8, 4, 8, seed=2, refine=True)
    indexes.set_param(pq.index, "nprobe", 8)
    indexes.set_param(pq.index, "k_factor", 1)
    plain = bench.evaluate(pq.index, data, q, kth, run)["recall"]
    indexes.set_param(pq.index, "k_factor", 8)
    assert bench.evaluate(pq.index, data, q, kth, run)["recall"] > max(plain, 0.9)


def test_k_sweep_measures_every_setting_at_each_k():
    rng = np.random.default_rng(20)
    data, q = _unit(rng, 3000), _unit(rng, 25)
    run = bench.Run(runs=1, warmup=0, build_threads=2)
    sweep = bench.KSweep(ks=(10, 20), nlist=8, nprobes=(2, 8), hnsw=(8, 40), ef_searches=(32,))
    rows = bench.k_sweep(data, q, run, sweep)
    assert len(rows) == 2 * (1 + 2 + 1)
    assert {(r["index"], r["k"]) for r in rows} >= {("Flat", 10), ("Flat", 20), ("HNSW", 20)}
    exhaustive = [r for r in rows if r["index"] == "IVF-Flat" and r["params"]["nprobe"] == 8]
    assert [r["recall"] for r in exhaustive] == [1.0, 1.0]
    assert all(r["recall"] == 1.0 for r in rows if r["index"] == "Flat")


def test_bench_selection_helpers():
    rows = [
        {"index": "A", "recall": 0.96, "p99_us": 50.0, "p50_us": 10.0},
        {"index": "B", "recall": 0.99, "p99_us": 80.0, "p50_us": 30.0},
        {"index": "A", "recall": 0.90, "p99_us": 20.0, "p50_us": 5.0},
    ]
    assert bench.cheapest(rows, 0.95) is rows[0]
    assert bench.cheapest(rows, 0.99) is rows[1]
    assert bench.cheapest(rows, 0.999) is None
    assert [r["recall"] for r in bench.pareto(rows)] == [0.90, 0.96, 0.99]


# ---------------------------------------------------------------- updates


def test_new_components_are_a_neighbourhood():
    mix = vectors.synthetic_mixture(dim=8, k=10, seed=17)
    new = incremental.new_components(mix, 0.3, seed=18)
    assert len(new) == 3
    mu = mix.means / np.linalg.norm(mix.means, axis=1, keepdims=True)
    first = np.random.default_rng(18).integers(10)
    assert first in new
    sims = mu @ mu[first]
    assert sims[new].min() >= np.delete(sims, new).max()


def test_deletion_ids():
    rng = np.random.default_rng(19)
    cats = np.array([0] * 30 + [1] * 50 + [2] * 20 + [-1] * 100)
    dead = incremental.deletion_ids("random", 0.25, cats, rng)
    assert len(dead) == 50 and len(np.unique(dead)) == 50
    dead = incremental.deletion_ids("category", 0.2, cats, rng)
    assert len(dead) >= 40
    removed = set(np.unique(cats[dead]))
    assert -1 not in removed
    for c in removed:  # whole categories only
        assert set(np.flatnonzero(cats == c)) <= set(dead.tolist())


# ---------------------------------------------------------------- end to end


def test_quick_run_renders_a_report(tmp_path, monkeypatch):
    from ml.ann import run as ann_run

    out_json, out_md = tmp_path / "ann.json", tmp_path / "ann.md"
    argv = ["run", "--quick", "--queries", "60", "--runs", "1", "--build-threads", "2"]
    argv += ["--source-emb", str(tmp_path / "missing.npy"), "--json", str(out_json)]
    argv += ["--report", str(out_md)]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        ann_run,
        "QUICK_PLAN",
        incremental.Plan(
            initial=4000, batches=2, batch_size=1000, nlist=32, nprobe=4, hnsw_m=8,
            hnsw_efc=20, ef_search=32, queries=40, query_pool=4000, delete_fractions=(0.3,),
        ),
    )  # fmt: skip
    ann_run.main()
    out = json.loads(out_json.read_text())
    assert out["setup"]["status"] == "preliminary"
    assert "synthetic clustered Gaussian" in out["data"]["kind"]
    text = out_md.read_text()
    assert text.startswith("# ANN benchmark and index updates")
    assert "Preliminary numbers" in text
    assert "## Top-200" in text and out["ksweep"]["rows"]
    assert report.render(out) == text  # rendering is deterministic


def test_flat_cost_model_and_switch_point():
    def scale(n, mean):
        flat = {"index": "Flat", "mean_us": mean, "p99_us": 2 * mean}
        return {"n": n, "source": "mixture", "rows": [flat]}

    scales = [scale(1_000, 12.0), scale(100_000, 210.0), scale(1_000_000, 2010.0)]
    cost = report.flat_cost(scales)
    assert cost["mean_us"]["b_ns_per_item"] == pytest.approx(2.0)
    assert cost["mean_us"]["a_us"] == pytest.approx(10.0)
    assert report.max_items(cost["mean_us"], 1010.0) == pytest.approx(500_000)
    assert report._round_items(213_000) == "200K"
    assert report._round_items(2_900_000) == "3M"
