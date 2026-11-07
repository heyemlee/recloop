import math

import numpy as np

from ml.eval import metrics


def _tie_pos(order: np.ndarray) -> np.ndarray:
    pos = np.empty_like(order)
    pos[order] = np.arange(len(order))
    return pos


def test_ranks_of_matches_full_sort_with_ties():
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 4, size=(200, 30)).astype(float)  # many ties
    target = rng.integers(0, 30, size=200)
    tie_order = rng.permutation(30)
    ranked = metrics.rank_items(scores, tie_order)
    expected = metrics.target_ranks(ranked, target)
    got = metrics.ranks_of(scores, target, _tie_pos(tie_order))
    np.testing.assert_array_equal(got, expected)


def test_rank_items_breaks_ties_by_tie_order():
    scores = np.array([[1.0, 2.0, 2.0, 0.0]])
    ranked = metrics.rank_items(scores, np.array([3, 2, 1, 0]))
    np.testing.assert_array_equal(ranked[0], [2, 1, 0, 3])


def test_target_ranks_missing_target_ranks_past_the_list():
    ranked = np.array([[5, 6, 7], [7, 6, 5]])
    np.testing.assert_array_equal(metrics.target_ranks(ranked, np.array([9, 5])), [4, 3])


def test_hr_and_ndcg_definitions():
    ranks = np.array([1, 3, 10, 11])
    v = metrics.per_sample(ranks, ks=(3, 10))
    np.testing.assert_array_equal(v["HR@3"], [1, 1, 0, 0])
    np.testing.assert_array_equal(v["HR@10"], [1, 1, 1, 0])
    np.testing.assert_allclose(v["NDCG@3"], [1.0, 0.5, 0.0, 0.0])
    np.testing.assert_allclose(v["NDCG@10"], [1.0, 0.5, 1 / math.log2(11), 0.0])


def test_sid_rank_takes_best_item_with_the_target_sid():
    # items 1 and 3 share a SID
    codes = np.array([0, 1, 2, 1])
    groups = metrics.sid_group_matrix(codes)
    scores = np.array([[0.1, 0.2, 0.9, 0.8]])  # order: 2, 3, 1, 0
    tie_pos = _tie_pos(np.arange(4))
    assert metrics.ranks_of(scores, np.array([1]), tie_pos)[0] == 3
    assert metrics.sid_ranks(scores, np.array([1]), tie_pos, groups)[0] == 2
    assert metrics.sid_ranks(scores, np.array([2]), tie_pos, groups)[0] == 1


def test_sid_rule_on_full_list_equals_code_comparison():
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 15, size=20)
    groups = metrics.sid_group_matrix(codes)
    scores = rng.random((50, 20))
    target = rng.integers(0, 20, size=50)
    tie_order = np.arange(20)
    ranked = metrics.rank_items(scores, tie_order)
    expected = metrics.target_ranks(ranked, target, codes)
    got = metrics.sid_ranks(scores, target, _tie_pos(tie_order), groups)
    np.testing.assert_array_equal(got, expected)
