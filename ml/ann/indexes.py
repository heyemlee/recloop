"""FAISS index builders and the single-query latency harness.

Builds run with `build_threads` OpenMP threads; searches run with one thread and one query per
`search` call, which is how the model-server calls FAISS for a request (from Python, so the
timings include the SWIG call overhead the server also pays).
"""

from __future__ import annotations

import ctypes
import gc
import sys
import time
from dataclasses import dataclass

import faiss
import numpy as np

METRIC = faiss.METRIC_INNER_PRODUCT


def set_threads(n: int) -> None:
    faiss.omp_set_num_threads(n)


def prefer_performance_cores() -> bool:
    """macOS: raise this thread's QoS class to user-interactive.

    The scheduler then keeps the (single-threaded) search loop on a performance core instead of
    moving it to an efficiency core when other work is running -- the closest macOS has to
    pinning a core. Returns whether the call succeeded.
    """
    if sys.platform != "darwin":
        return False
    try:
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        return libc.pthread_set_qos_class_self_np(0x21, 0) == 0  # QOS_CLASS_USER_INTERACTIVE
    except (OSError, AttributeError):
        return False


@dataclass
class Built:
    index: faiss.Index
    train_s: float  # k-means / PQ training (0 for Flat and HNSW)
    add_s: float  # adding the vectors (graph construction for HNSW)

    @property
    def build_s(self) -> float:
        return self.train_s + self.add_s


def index_bytes(index: faiss.Index) -> int:
    """Serialized size, which tracks the index's memory (vectors / codes, ids, graph links)."""
    return int(faiss.serialize_index(index).size)


def _train_add(index: faiss.Index, x: np.ndarray) -> Built:
    t0 = time.perf_counter()
    if not index.is_trained:
        index.train(x)
    t1 = time.perf_counter()
    index.add(x)
    return Built(index, t1 - t0, time.perf_counter() - t1)


def build_flat(x: np.ndarray) -> Built:
    return _train_add(faiss.IndexFlatIP(x.shape[1]), x)


def ivf_flat(dim: int, nlist: int, seed: int) -> faiss.IndexIVFFlat:
    """Empty IVF-Flat; k-means uses FAISS's defaults for IVF (10 iterations, spherical for
    inner product, a sample of 256 points per centroid) with our seed."""
    index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, nlist, METRIC)
    index.cp.seed = seed
    return index


def build_ivf_flat(x: np.ndarray, nlist: int, seed: int) -> Built:
    return _train_add(ivf_flat(x.shape[1], nlist, seed), x)


def build_hnsw(x: np.ndarray, m: int, ef_construction: int) -> Built:
    index = faiss.IndexHNSWFlat(x.shape[1], m, METRIC)
    index.hnsw.efConstruction = ef_construction
    return _train_add(index, x)


def build_ivf_pq(
    x: np.ndarray, nlist: int, m: int, nbits: int, seed: int, refine: bool = False
) -> Built:
    """IVF-PQ (m sub-quantizers of nbits each, residual encoding); with `refine`, the top
    k * k_factor PQ results are re-scored with the full vectors (IndexRefineFlat)."""
    dim = x.shape[1]
    base = faiss.IndexIVFPQ(faiss.IndexFlatIP(dim), dim, nlist, m, nbits, METRIC)
    base.cp.seed = seed
    base.pq.cp.seed = seed
    return _train_add(faiss.IndexRefineFlat(base) if refine else base, x)


def ivf_of(index: faiss.Index) -> faiss.IndexIVF:
    return faiss.extract_index_ivf(index)


def list_sizes(index: faiss.Index) -> np.ndarray:
    ivf = ivf_of(index)
    invlists = ivf.invlists
    return np.array([invlists.list_size(i) for i in range(ivf.nlist)], dtype=np.int64)


def set_param(index: faiss.Index, name: str, value: int) -> None:
    if name == "nprobe":
        ivf_of(index).nprobe = value
    elif name == "efSearch":
        index.hnsw.efSearch = value
    elif name == "k_factor":
        index.k_factor = value
    else:
        raise ValueError(f"unknown search parameter {name}")


def stats_for(index: faiss.Index):
    """FAISS's global search counters for this index type (ndis = vectors compared)."""
    if isinstance(index, faiss.IndexHNSW):
        return faiss.cvar.hnsw_stats
    try:
        ivf_of(index)
    except RuntimeError:
        return None
    return faiss.cvar.indexIVF_stats


@dataclass
class Timing:
    ids: np.ndarray  # (nq, k) results of the first run
    runs: list[list[float]]  # [p50, p99, mean] in microseconds, per run
    ndis: float | None  # mean vectors compared per query (IVF / HNSW counters), first run


def time_queries(
    index: faiss.Index,
    queries: np.ndarray,
    k: int,
    runs: int = 3,
    warmup: int = 100,
    params: faiss.SearchParameters | None = None,
) -> Timing:
    """Search every query alone (nq = 1), `runs` times; per-query wall time around `search`.

    The caller fixes the OpenMP thread count. Garbage collection is off while timing.
    """
    nq = len(queries)
    rows = [queries[i : i + 1] for i in range(nq)]
    kw = {} if params is None else {"params": params}
    search = index.search
    for q in rows[:warmup]:
        search(q, k, **kw)
    stats = stats_for(index)
    lat = np.empty(nq, dtype=np.int64)
    ids = np.empty((nq, k), dtype=np.int64)
    per_run: list[list[float]] = []
    ndis = None
    gc_on = gc.isenabled()
    gc.disable()
    try:
        for r in range(runs):
            if r == 0 and stats is not None:
                stats.reset()
            for i, q in enumerate(rows):
                t0 = time.perf_counter_ns()
                _, found = search(q, k, **kw)
                lat[i] = time.perf_counter_ns() - t0
                if r == 0:
                    ids[i] = found[0]
            if r == 0 and stats is not None:
                ndis = stats.ndis / nq
            p50, p99 = np.percentile(lat, (50, 99))
            per_run.append([float(p50) / 1e3, float(p99) / 1e3, float(lat.mean()) / 1e3])
    finally:
        if gc_on:
            gc.enable()
    return Timing(ids, per_run, ndis)


def call_overhead_us(dim: int, k: int, n: int = 2000, runs: int = 3) -> list[float]:
    """Median [p50, p99, mean] of `search` on a one-vector Flat index: the per-call floor."""
    index = faiss.IndexFlatIP(dim)
    index.add(np.ones((1, dim), dtype=np.float32) / np.sqrt(dim))
    q = np.ones((n, dim), dtype=np.float32) / np.sqrt(dim)
    t = time_queries(index, q, k, runs=runs)
    return [float(v) for v in np.median(np.asarray(t.runs), axis=0)]
