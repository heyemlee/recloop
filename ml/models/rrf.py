"""Weighted reciprocal rank fusion of SASRec, ItemCF and Pop: the api's v1 ranking (design §4.3).

    score(i) = sum over the sources s that returned i of  w_s / (60 + rank_s(i))

over each source's top 200, as `ml.ranking.candidates` replays it. Items that are not among
the 300 fused candidates score -inf, so the evaluation's tie order (training popularity, then
item_id) places them after every candidate; HR / NDCG at K <= 10 only ever see candidates.
"""

from __future__ import annotations

import numpy as np

from ml.eval.splits import NUM_ITEMS, Split
from ml.ranking import candidates as cand_mod


class RRF:
    def __init__(
        self, sources: dict, weights: dict[str, float], name: str = "RRF", variant: str = "control"
    ):
        self.sources = sources  # source name -> fitted model with score(split)
        self.weights = dict(weights)
        self.name = name
        self.variant = variant
        self._cache: dict[str, cand_mod.Candidates] = {}

    def fit(self, sequences, num_items: int) -> RRF:
        return self  # nothing to fit: the sources are fitted already

    def candidates(self, split: Split) -> cand_mod.Candidates:
        if split.name not in self._cache:
            lists = cand_mod.source_lists(self.sources, split)
            self._cache[split.name] = cand_mod.fuse(lists, self.weights, NUM_ITEMS)
        return self._cache[split.name]

    def score(self, split: Split) -> np.ndarray:
        c = self.candidates(split)
        return dense(c.items, c.rrf)

    def describe(self) -> str:
        w = ", ".join(f"{s} {self.weights[s]:g}" for s in cand_mod.SOURCES if s in self.weights)
        return (
            f"weights {w} (the recall experiment's {self.variant} variant), k = {cand_mod.RRF_K}, "
            f"top {cand_mod.SOURCE_K} per source, {cand_mod.CANDIDATE_DEPTH} candidates"
        )


def dense(items: np.ndarray, values: np.ndarray) -> np.ndarray:
    """(n, NUM_ITEMS) scores: `values` at the candidates, -inf everywhere else."""
    out = np.full((items.shape[0], NUM_ITEMS), -np.inf)
    rows, cols = np.nonzero(items >= 0)
    out[rows, items[rows, cols]] = values[rows, cols]
    return out
