"""The home feed's candidates, offline (docs/design.md §4.1, §4.3).

This replays what the api does for `surface=home` (serving/cmd/api/main.go, internal/recall,
internal/mixer/fuse.go) on a split, so the RRF baseline and the LightGBM ranker are evaluated
on exactly the candidates they would see online:

  * three sources, each returning its top 200: SASRec (dense scores; the model server's FAISS
    IndexFlatIP is exact), ItemCF (items with a positive score only, like the Go scorer) and Pop
    (training counts). Each list is ordered by score, then item id;
  * weighted reciprocal rank fusion, score(i) = sum_s w_s / (60 + rank_s(i)), summed in the
    sources' order (sasrec, itemcf, pop) as in Go; order by score, then the best single-source
    rank, then item id; the first 300 are the candidates (cache.CandidateDepth).

The weights are the recall-layer experiment's (migration 00003): control is the default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SOURCES = ("sasrec", "itemcf", "pop")  # the home surface's spec order
SOURCE_K = 200
RRF_K = 60
CANDIDATE_DEPTH = 300
CONTROL_WEIGHTS = {"sasrec": 1.0, "itemcf": 1.0, "pop": 0.3}
SASREC_HEAVY_WEIGHTS = {"sasrec": 1.5, "itemcf": 0.75, "pop": 0.3}
CHUNK = 1024  # rows per block: dense (rows, 3459) matrices stay small


def top_lists(scores: np.ndarray, k: int, positive_only: bool = False):
    """(items, scores) of each row's top k, by score desc then item id asc; -1 / NaN padded.

    With `positive_only`, items scoring <= 0 are left out (ItemCF returns only neighbours).
    """
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]  # stable: lower id first on ties
    top = np.take_along_axis(scores, order, axis=1).astype(np.float64)
    items = order.astype(np.int64)
    if positive_only:
        keep = top > 0
        items = np.where(keep, items, -1)
        top = np.where(keep, top, np.nan)
    return items, top


@dataclass(frozen=True)
class Candidates:
    """Per sample: the fused candidates and what each source said about them."""

    items: np.ndarray  # (n, depth) item ids in RRF order, -1 padded
    rrf: np.ndarray  # (n, depth) fused score, NaN padded
    ranks: dict[str, np.ndarray]  # source -> (n, depth) 1-based rank in its list, 0 = absent
    scores: dict[str, np.ndarray]  # source -> (n, depth) the source's score, NaN = absent

    def __len__(self) -> int:
        return self.items.shape[0]

    def valid(self) -> np.ndarray:
        return self.items >= 0

    def n_sources(self) -> np.ndarray:
        return sum((r > 0).astype(np.int64) for r in self.ranks.values())

    def row(self, i: int) -> dict:
        """One sample's candidates as plain lists (golden queries, tests)."""
        keep = self.items[i] >= 0
        return {
            "items": self.items[i][keep].tolist(),
            "hits": [
                [
                    [s, int(self.ranks[s][i][j]), float(self.scores[s][i][j])]
                    for s in self.ranks
                    if self.ranks[s][i][j] > 0
                ]
                for j in np.flatnonzero(keep)
            ],
        }


def fuse(
    lists: dict[str, tuple[np.ndarray, np.ndarray]],
    weights: dict[str, float],
    num_items: int,
    depth: int = CANDIDATE_DEPTH,
) -> Candidates:
    """Weighted RRF of per-source (items, scores) lists, exactly as mixer.RRF orders them."""
    n = next(iter(lists.values()))[0].shape[0]
    out_items = np.full((n, depth), -1, dtype=np.int64)
    out_rrf = np.full((n, depth), np.nan)
    out_ranks = {s: np.zeros((n, depth), dtype=np.int64) for s in lists}
    out_scores = {s: np.full((n, depth), np.nan) for s in lists}
    big = np.iinfo(np.int64).max
    for lo in range(0, n, CHUNK):
        hi = min(n, lo + CHUNK)
        m = hi - lo
        rows = np.arange(m)[:, None]
        fused = np.zeros((m, num_items))
        hit = np.zeros((m, num_items), dtype=bool)
        best = np.full((m, num_items), big, dtype=np.int64)
        rank_of, score_of = {}, {}
        # Go sums the contributions in the sources' order: the home spec's three, then any
        # other source in the order given (M7: gen, after pop, as in serving/cmd/api).
        for s in (*SOURCES, *(x for x in lists if x not in SOURCES)):
            if s not in lists or weights.get(s, 0.0) <= 0:
                continue
            items, scores = lists[s][0][lo:hi], lists[s][1][lo:hi]
            ok = items >= 0
            rank = np.broadcast_to(np.arange(1, items.shape[1] + 1), items.shape)
            contrib = weights[s] / (RRF_K + rank).astype(np.float64)
            r_idx, c_idx = np.nonzero(ok)
            it = items[r_idx, c_idx]
            fused[r_idx, it] += contrib[r_idx, c_idx]
            hit[r_idx, it] = True
            best[r_idx, it] = np.minimum(best[r_idx, it], rank[r_idx, c_idx])
            rk = np.zeros((m, num_items), dtype=np.int64)
            rk[r_idx, it] = rank[r_idx, c_idx]
            sc = np.full((m, num_items), np.nan)
            sc[r_idx, it] = scores[r_idx, c_idx]
            rank_of[s], score_of[s] = rk, sc
        ids = np.broadcast_to(np.arange(num_items), (m, num_items))
        # Order: hits first, then fused score desc, best rank asc, item id asc.
        order = np.lexsort((ids, best, -fused, ~hit), axis=1)[:, :depth]
        sel_hit = hit[rows, order]
        out_items[lo:hi] = np.where(sel_hit, order, -1)
        out_rrf[lo:hi] = np.where(sel_hit, fused[rows, order], np.nan)
        for s in rank_of:
            out_ranks[s][lo:hi] = np.where(sel_hit, rank_of[s][rows, order], 0)
            out_scores[s][lo:hi] = np.where(sel_hit, score_of[s][rows, order], np.nan)
    for s in lists:  # a source with weight 0 contributes nothing (and is not a hit)
        out_ranks.setdefault(s, np.zeros((n, depth), dtype=np.int64))
    return Candidates(out_items, out_rrf, out_ranks, out_scores)


def source_lists(models: dict, split, k: int = SOURCE_K) -> dict:
    """Each source's top-k lists for a split; `models` maps source name -> fitted model."""
    out = {}
    for s in SOURCES:
        if s in models:
            scores = np.asarray(models[s].score(split), dtype=np.float64)
            out[s] = top_lists(scores, k, positive_only=(s == "itemcf"))
    return out
