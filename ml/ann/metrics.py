"""Exact ground truth, recall, inverted-list balance and latency summaries for the ANN runs.

Ground truth is brute force in float64 with numpy, independent of FAISS. Recall@k counts a
returned item as a hit when its exact inner product with the query is at least the k-th best
exact inner product, so items tied with the k-th neighbour (duplicate vectors) count as found;
without ties this equals |returned ∩ exact top-k| / k. FAISS pads missing results with -1;
those are misses.
"""

from __future__ import annotations

import numpy as np

TIE_TOL = 1e-9  # float64 rounding between two ways of computing the same dot product


def exact_topk(
    data: np.ndarray, queries: np.ndarray, k: int, chunk_bytes: int = 1 << 28
) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k by inner product: (ids, scores), both (nq, k), best first.

    Ties are ordered by item id. Computed in float64 from the given (float32) vectors, a chunk
    of queries at a time (about `chunk_bytes` of scores per chunk).
    """
    data64 = np.asarray(data, dtype=np.float64)
    nq = len(queries)
    chunk = max(1, chunk_bytes // (8 * len(data64)))
    ids = np.empty((nq, k), dtype=np.int64)
    scores = np.empty((nq, k), dtype=np.float64)
    for lo in range(0, nq, chunk):
        q = np.asarray(queries[lo : lo + chunk], dtype=np.float64)
        s = q @ data64.T
        top = np.argpartition(-s, k - 1, axis=1)[:, :k]
        top_s = np.take_along_axis(s, top, axis=1)
        order = np.lexsort((top, -top_s), axis=1)
        ids[lo : lo + chunk] = np.take_along_axis(top, order, axis=1)
        scores[lo : lo + chunk] = np.take_along_axis(top_s, order, axis=1)
    return ids, scores


def exact_scores(data: np.ndarray, queries: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """float64 inner product of every query with each of its returned ids (-inf for -1)."""
    out = np.full(ids.shape, -np.inf)
    valid = ids >= 0
    rows = np.nonzero(valid)[0]
    out[valid] = np.einsum(
        "nd,nd->n",
        np.asarray(queries, dtype=np.float64)[rows],
        np.asarray(data[ids[valid]], dtype=np.float64),
    )
    return out


def recall_at_k(found_scores: np.ndarray, kth_score: np.ndarray, k: int) -> np.ndarray:
    """Per-query recall@k from the exact scores of the returned ids (see the module doc).

    found_scores: (nq, >= k) exact scores of the returned ids, -inf where nothing was returned.
    The ids must be distinct per query, as FAISS returns them (see `has_duplicates`).
    kth_score: (nq,) the k-th best exact score of each query.
    """
    hits = found_scores[:, :k] >= (np.asarray(kth_score)[:, None] - TIE_TOL)
    return hits.sum(axis=1) / k


def has_duplicates(found: np.ndarray) -> bool:
    """Whether any query's result list repeats a (valid) id."""
    s = np.sort(found, axis=1)
    return bool(((s[:, 1:] == s[:, :-1]) & (s[:, 1:] >= 0)).any())


def intersection_recall(found: np.ndarray, truth: np.ndarray, k: int) -> np.ndarray:
    """|found[:, :k] ∩ truth[:, :k]| / k per query (-1 never matches)."""
    return np.array(
        [
            len(set(f[f >= 0].tolist()) & set(t.tolist())) / k
            for f, t in zip(found[:, :k], truth[:, :k], strict=True)
        ]
    )


def list_balance(sizes: np.ndarray) -> dict:
    """Balance of IVF inverted lists (empty lists included).

    cv is std / mean (population std). imbalance is FAISS's invlists.imbalance_factor(),
    nlist * sum(size^2) / (sum size)^2 = 1 + cv^2: when queries land on lists in proportion to
    their size, a probe scans `imbalance` times the average list.
    """
    s = np.asarray(sizes, dtype=np.float64)
    mean = s.mean()
    return {
        "nlist": len(s),
        "items": int(s.sum()),
        "mean": float(mean),
        "cv": float(s.std() / mean),
        "max": int(s.max()),
        "max_over_mean": float(s.max() / mean),
        "imbalance": float(len(s) * (s**2).sum() / s.sum() ** 2),
        "empty": int((s == 0).sum()),
    }


def percentiles_us(latency_ns: np.ndarray, qs: tuple[float, ...] = (50, 99)) -> list[float]:
    """Latency percentiles in microseconds (numpy's linear interpolation)."""
    return [float(v) / 1e3 for v in np.percentile(latency_ns, qs)]


def median_of_runs(per_run: list[list[float]]) -> list[float]:
    """Element-wise median over runs, e.g. of [p50, p99, mean] measured in each of 3 runs."""
    return [float(v) for v in np.median(np.asarray(per_run, dtype=np.float64), axis=0)]
