import itertools
import math

import numpy as np
import pytest

from ml.eval.splits import Split, train_sequences
from ml.models.itemcf import ItemCF, ItemCFParams, cooccurrence, top_k_rows
from ml.models.pop import Pop

SEQS = {
    0: np.array([0, 1, 2, 1]),
    1: np.array([1, 2]),
    2: np.array([3, 0, 1, 4, 2]),
    3: np.array([4]),
}
N = 5


def _brute_cooc(seqs, lam, iuf):
    c = np.zeros((N, N))
    for s in seqs.values():
        w = 1 / math.log1p(len(s)) if iuf else 1.0
        for p, q in itertools.permutations(range(len(s)), 2):
            if s[p] != s[q]:
                c[s[p], s[q]] += w * lam ** (abs(p - q) - 1)
    return c


@pytest.mark.parametrize(("lam", "iuf"), [(1.0, False), (0.5, True), (0.7, False)])
def test_cooccurrence_matches_brute_force(lam, iuf):
    got = cooccurrence(SEQS, N, lam, iuf).toarray()
    np.testing.assert_allclose(got, _brute_cooc(SEQS, lam, iuf))
    assert np.allclose(np.diag(got), 0)


def test_itemcf_similarity_and_scores_match_brute_force():
    params = ItemCFParams(beta=0.3, history_decay=0.5, proximity_decay=0.7, iuf=True, neighbors=2)
    model = ItemCF(params).fit(SEQS, N)
    counts = np.bincount(np.concatenate(list(SEQS.values())), minlength=N).astype(float)
    sim = _brute_cooc(SEQS, 0.7, True) / np.outer(counts**0.7, counts**0.3)
    # keep the top-2 per row (lower column first on ties)
    top = np.zeros_like(sim)
    for i in range(N):
        order = np.lexsort((np.arange(N), -sim[i]))[:2]
        order = [j for j in order if sim[i, j] > 0]
        top[i, order] = sim[i, order]
    np.testing.assert_allclose(model.sim.toarray(), top)

    split = Split("t", np.array([0, 1]), np.array([2, 0]), [np.array([0, 1]), np.array([3])])
    expected = np.stack([0.5 * top[0] + 1.0 * top[1], top[3]])
    np.testing.assert_allclose(model.score(split), expected)


def test_top_k_rows_keeps_largest():
    import scipy.sparse as sp

    m = sp.csr_matrix(np.array([[0, 3, 1, 3], [5, 0, 0, 0]], dtype=float))
    np.testing.assert_array_equal(top_k_rows(m, 2).toarray(), [[0, 3, 0, 3], [5, 0, 0, 0]])


def test_pop_counts_training_interactions():
    pop = Pop().fit(SEQS, N)
    np.testing.assert_array_equal(pop.counts, [2, 4, 3, 1, 2])
    split = Split("t", np.array([0]), np.array([0]), [np.array([1])])
    np.testing.assert_array_equal(pop.score(split)[0], pop.counts)


def test_train_sequences_rebuilds_user_sequences():
    # user 5: items 10, 11, 12, 13 (rows target 11, 12, 13); user 6: items 20, 21
    split = Split(
        "train",
        user=np.array([5, 6, 5, 5]),
        target=np.array([11, 21, 12, 13]),
        history=[np.array([10]), np.array([20]), np.array([10, 11]), np.array([10, 11, 12])],
    )
    seqs = train_sequences(split)
    np.testing.assert_array_equal(seqs[5], [10, 11, 12, 13])
    np.testing.assert_array_equal(seqs[6], [20, 21])


def test_train_sequences_rejects_inconsistent_rows():
    split = Split(
        "train", np.array([5, 5]), np.array([11, 12]), [np.array([10]), np.array([99, 11])]
    )
    with pytest.raises(ValueError):
        train_sequences(split)


def test_history_matrix_right_aligns_and_pads():
    split = Split("t", np.array([0, 1]), np.array([5, 6]), [np.array([1]), np.array([2, 3])])
    m = split.history_matrix()
    assert m.shape == (2, 10)
    np.testing.assert_array_equal(m[0], [-1] * 9 + [1])
    np.testing.assert_array_equal(m[1], [-1] * 8 + [2, 3])
