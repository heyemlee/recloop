"""Item-based collaborative filtering (docs/design.md §4.2).

Offline:
  * co-occurrence over each user's training sequence: every ordered pair of positions (p, q),
    p != q, with different items adds  w_u * lambda^(|p - q| - 1)  to C[s_p, s_q], where
    w_u = 1 / log(1 + |s|) down-weights very active users (IUF) and lambda < 1 favours pairs
    that are close in the sequence;
  * similarity with a popularity penalty on the candidate:
        sim(i -> j) = C[i, j] / (n_i^(1 - beta) * n_j^beta),   n_i = training count of item i
    (beta = 0.5 is cosine; a larger beta pushes popular candidates down);
  * keep the top `neighbors` (50) items per source item.
Online / scoring: for a history h_1..h_L (most recent last)
        score(j) = sum_k gamma^(L - k) * sim(h_k -> j)
Items no history item points to score 0; the evaluation breaks those ties by popularity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from ml.eval.splits import Split


@dataclass(frozen=True)
class ItemCFParams:
    beta: float = 0.5  # candidate popularity penalty
    history_decay: float = 1.0  # gamma: weight of the k-th most recent history item, gamma^k
    proximity_decay: float = 1.0  # lambda: pair weight by distance in the training sequence
    iuf: bool = True  # 1 / log(1 + sequence length) per user
    neighbors: int = 50

    def label(self) -> str:
        return (
            f"beta={self.beta}, gamma={self.history_decay}, lambda={self.proximity_decay}, "
            f"iuf={'on' if self.iuf else 'off'}, top-{self.neighbors}"
        )


def cooccurrence(
    sequences: dict[int, np.ndarray], num_items: int, proximity_decay: float, iuf: bool
) -> sp.csr_matrix:
    rows, cols, vals = [], [], []
    for seq in sequences.values():
        n = len(seq)
        if n < 2:
            continue
        p, q = np.triu_indices(n, k=1)
        a, b = seq[p], seq[q]
        keep = a != b
        w = proximity_decay ** (q - p - 1).astype(np.float64)
        if iuf:
            w = w / np.log1p(n)
        a, b, w = a[keep], b[keep], w[keep]
        rows += [a, b]
        cols += [b, a]
        vals += [w, w]
    c = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(num_items, num_items),
    )
    return c.tocsr()  # duplicates are summed


def top_k_rows(m: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Keep the k largest entries of each row (ties: lower column index first)."""
    m = m.tocsr()
    m.sort_indices()
    indptr, indices, data = [0], [], []
    for r in range(m.shape[0]):
        lo, hi = m.indptr[r], m.indptr[r + 1]
        cols, vals = m.indices[lo:hi], m.data[lo:hi]
        if len(vals) > k:
            order = np.lexsort((cols, -vals))[:k]
            cols, vals = cols[order], vals[order]
        indices.append(cols)
        data.append(vals)
        indptr.append(indptr[-1] + len(cols))
    out = sp.csr_matrix(
        (np.concatenate(data), np.concatenate(indices), np.asarray(indptr)), shape=m.shape
    )
    out.sort_indices()
    return out


@dataclass
class ItemCF:
    params: ItemCFParams = field(default_factory=ItemCFParams)
    name: str = "ItemCF"
    sim: sp.csr_matrix | None = None
    counts: np.ndarray | None = None

    def fit(
        self, sequences: dict[int, np.ndarray], num_items: int, cooc: sp.csr_matrix | None = None
    ) -> ItemCF:
        p = self.params
        self.counts = np.bincount(
            np.concatenate(list(sequences.values())), minlength=num_items
        ).astype(np.float64)
        if cooc is None:
            cooc = cooccurrence(sequences, num_items, p.proximity_decay, p.iuf)
        n = np.where(self.counts > 0, self.counts, 1.0)
        sim = sp.diags(n ** -(1.0 - p.beta)) @ cooc @ sp.diags(n**-p.beta)
        self.sim = top_k_rows(sim.tocsr(), p.neighbors)
        return self

    def history_weights(self, split: Split) -> sp.csr_matrix:
        rows, cols, vals = [], [], []
        for i, h in enumerate(split.history):
            n = len(h)
            rows.append(np.full(n, i))
            cols.append(h)
            vals.append(self.params.history_decay ** np.arange(n - 1, -1, -1, dtype=np.float64))
        return sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(len(split), self.sim.shape[0]),
        ).tocsr()

    def score(self, split: Split) -> np.ndarray:
        assert self.sim is not None, "fit() first"
        return (self.history_weights(split) @ self.sim).toarray()

    def describe(self) -> str:
        return self.params.label()
