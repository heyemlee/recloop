"""Flat vs IVF-Flat vs HNSW (and IVF-PQ at 1M) at one catalog size.

For every configuration: recall@k against the exact top-k (mean over queries, plus the 5th
percentile, since a mean can hide users whose results are mostly wrong), single-query P50 / P99
latency (one thread, median of `runs` runs), vectors compared per query, index size, and
training / build time (`build_threads` threads).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ml.ann import indexes, metrics


@dataclass(frozen=True)
class Grid:
    ivf_nlists: tuple[int, ...]
    nprobes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    hnsw: tuple[tuple[int, int], ...] = ((16, 200), (32, 40), (32, 200), (64, 200))  # (M, efC)
    ef_searches: tuple[int, ...] = (64, 128, 256, 512, 1024)
    ivfpq_nlist: int | None = None
    ivfpq_m: tuple[int, ...] = (16, 32)
    ivfpq_nprobes: tuple[int, ...] = (16, 32, 64, 128)
    ivfpq_k_factors: tuple[int, ...] = (1, 4)


@dataclass(frozen=True)
class Run:
    k: int = 100
    runs: int = 3
    warmup: int = 100
    build_threads: int = 8
    search_threads: int = 1
    seed: int = 20261106


def kth_scores(data: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """The k-th best exact score of every query (the recall threshold)."""
    return metrics.exact_topk(data, queries, k)[1][:, k - 1]


def evaluate(index, data, queries, kth, run: Run, search_params=None, span=None) -> dict:
    """Recall and single-query latency of `index` as currently configured.

    With `span` (per query, how many components its exact top-k covers), also the median span
    of the queries whose recall is below 0.9.
    """
    t = indexes.time_queries(index, queries, run.k, run.runs, run.warmup, search_params)
    assert not metrics.has_duplicates(t.ids), "an index returned the same item twice"
    rec = metrics.recall_at_k(metrics.exact_scores(data, queries, t.ids), kth, run.k)
    p50, p99, mean = metrics.median_of_runs(t.runs)
    low = rec < 0.9
    out = {
        "recall": float(rec.mean()),
        "recall_p5": float(np.percentile(rec, 5)),
        "recall_lt_0.9": float(low.mean()),
        "p50_us": p50,
        "p99_us": p99,
        "mean_us": mean,
        "runs_us": t.runs,
        "ndis": t.ndis,
    }
    if span is not None:
        out["span_lt_0.9"] = float(np.median(span[low])) if low.any() else None
    return out


def build_with(run: Run, fn, *args):
    """Call an index builder with `run.build_threads` threads, then go back to search threads."""
    indexes.set_threads(run.build_threads)
    try:
        return fn(*args)
    finally:
        indexes.set_threads(run.search_threads)


class Bench:
    """Holds a catalog, its queries and their exact top-k; measures index configurations."""

    def __init__(self, data: np.ndarray, queries: np.ndarray, run: Run, labels: np.ndarray):
        self.data, self.queries, self.run = data, queries, run
        ids, scores = metrics.exact_topk(data, queries, run.k)
        self.kth = scores[:, run.k - 1]
        # How many components (categories) each query's exact top-k covers.
        self.span = np.array([len(np.unique(labels[row])) for row in ids])
        self.rows: list[dict] = []

    def _build(self, fn, *args):
        return build_with(self.run, fn, *args)

    def measure(self, index, family: str, params: dict, built: dict, extra: dict | None = None):
        for name, value in params.items():
            if name in ("nprobe", "efSearch", "k_factor"):
                indexes.set_param(index, name, value)
        row = {
            "index": family,
            "params": params,
            **evaluate(index, self.data, self.queries, self.kth, self.run, span=self.span),
            **built,
            **(extra or {}),
        }
        self.rows.append(row)
        return row

    @staticmethod
    def built_info(b: indexes.Built, size: int) -> dict:
        return {"train_s": b.train_s, "add_s": b.add_s, "bytes": size}

    def flat(self) -> None:
        b = self._build(indexes.build_flat, self.data)
        self.measure(b.index, "Flat", {}, self.built_info(b, indexes.index_bytes(b.index)))

    def ivf_flat(self, nlists: tuple[int, ...], nprobes: tuple[int, ...]) -> None:
        for nlist in nlists:
            b = self._build(indexes.build_ivf_flat, self.data, nlist, self.run.seed)
            info = self.built_info(b, indexes.index_bytes(b.index))
            balance = metrics.list_balance(indexes.list_sizes(b.index))
            for nprobe in (p for p in nprobes if p <= nlist):
                self.measure(
                    b.index, "IVF-Flat", {"nlist": nlist, "nprobe": nprobe}, info,
                    {"balance": balance},
                )  # fmt: skip
            del b

    def hnsw(self, configs: tuple[tuple[int, int], ...], ef_searches: tuple[int, ...]) -> None:
        for m, efc in configs:
            b = self._build(indexes.build_hnsw, self.data, m, efc)
            info = self.built_info(b, indexes.index_bytes(b.index))
            for ef in ef_searches:
                self.measure(b.index, "HNSW", {"M": m, "efC": efc, "efSearch": ef}, info)
            del b

    def ivf_pq(self, nlist: int, ms: tuple[int, ...], nprobes, k_factors) -> None:
        for m in ms:
            b = self._build(indexes.build_ivf_pq, self.data, nlist, m, 8, self.run.seed, True)
            base = indexes.index_bytes(b.index.base_index)
            info = self.built_info(b, indexes.index_bytes(b.index))
            for kf in k_factors:
                for nprobe in nprobes:
                    # k_factor 1 re-scores only the PQ top-k: the same items as plain IVF-PQ,
                    # whose memory is the codes alone.
                    size = base if kf == 1 else info["bytes"]
                    self.measure(
                        b.index, "IVF-PQ" if kf == 1 else "IVF-PQ + refine",
                        {"nlist": nlist, "m": m, "nbits": 8, "nprobe": nprobe, "k_factor": kf},
                        {**info, "bytes": size},
                    )  # fmt: skip
            del b

    def all(self, grid: Grid) -> list[dict]:
        self.flat()
        self.ivf_flat(grid.ivf_nlists, grid.nprobes)
        self.hnsw(grid.hnsw, grid.ef_searches)
        if grid.ivfpq_nlist:
            self.ivf_pq(grid.ivfpq_nlist, grid.ivfpq_m, grid.ivfpq_nprobes, grid.ivfpq_k_factors)
        return self.rows


@dataclass(frozen=True)
class KSweep:
    """The same indexes measured at several result sizes k (the retrieval stage asks for 200)."""

    ks: tuple[int, ...] = (100, 200)
    nlist: int = 4096
    nprobes: tuple[int, ...] = (32, 64, 128, 256)
    hnsw: tuple[int, int] = (32, 200)  # (M, efConstruction)
    ef_searches: tuple[int, ...] = (128, 256, 512, 1024, 2048)


def k_sweep(data: np.ndarray, queries: np.ndarray, run: Run, sweep: KSweep) -> list[dict]:
    """Flat, one IVF-Flat and one HNSW index; every search setting is measured at each k in
    turn, so the k = 100 and k = 200 numbers come from the same index under the same load."""
    _, scores = metrics.exact_topk(data, queries, max(sweep.ks))
    kth = {k: scores[:, k - 1] for k in sweep.ks}
    rows: list[dict] = []

    def at_each_k(index, family: str, params: dict, built: dict) -> None:
        for k in sweep.ks:
            res = evaluate(index, data, queries, kth[k], replace(run, k=k))
            rows.append({"index": family, "params": params, "k": k, **res, **built})

    flat = build_with(run, indexes.build_flat, data)
    at_each_k(flat.index, "Flat", {}, Bench.built_info(flat, indexes.index_bytes(flat.index)))
    del flat
    ivf = build_with(run, indexes.build_ivf_flat, data, sweep.nlist, run.seed)
    info = Bench.built_info(ivf, indexes.index_bytes(ivf.index))
    for nprobe in sweep.nprobes:
        indexes.set_param(ivf.index, "nprobe", nprobe)
        at_each_k(ivf.index, "IVF-Flat", {"nlist": sweep.nlist, "nprobe": nprobe}, info)
    del ivf
    m, efc = sweep.hnsw
    hnsw = build_with(run, indexes.build_hnsw, data, m, efc)
    info = Bench.built_info(hnsw, indexes.index_bytes(hnsw.index))
    for ef in sweep.ef_searches:
        indexes.set_param(hnsw.index, "efSearch", ef)
        at_each_k(hnsw.index, "HNSW", {"M": m, "efC": efc, "efSearch": ef}, info)
    return rows


def cheapest(rows: list[dict], target: float, key: str = "p99_us") -> dict | None:
    """The configuration with the lowest `key` among those reaching mean recall >= target."""
    ok = [r for r in rows if r["recall"] >= target]
    return min(ok, key=lambda r: (r[key], r["p50_us"])) if ok else None


def pareto(rows: list[dict], key: str = "p99_us") -> list[dict]:
    """Rows not dominated in (higher recall, lower `key`), sorted by recall."""
    out = []
    for r in sorted(rows, key=lambda r: (r[key], -r["recall"])):
        if not out or r["recall"] > out[-1]["recall"]:
            out.append(r)
    return out
