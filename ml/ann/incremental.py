"""Index updates while new items arrive: incremental adds vs. periodic full rebuilds.

Scenario: an initial catalog drawn from the "old" mixture components, then `batches` batches.
Each batch mixes old-category items with items of new categories: a contiguous group of
held-out components (a random component and its nearest components by mean direction), i.e. a
region of the space the initial catalog does not cover. A control stream adds batches of old
categories only (no drift).

After every batch:
  * IVF trained on the initial catalog, new items only added (drift and control streams):
    inverted-list balance, recall@k and latency at a fixed nprobe, vectors compared per query;
  * IVF rebuilt from scratch on the current catalog (k-means retrained): rebuild time, balance,
    recall, latency -- what a full rebuild at that point gives;
  * HNSW with the batch added (time per batch), recall and latency at a fixed efSearch.
At the end, HNSW built from scratch on the final catalog. Then deletions: FAISS's HNSW cannot
remove vectors, so deleted items stay in the graph and are filtered out at search time with an
IDSelector; recall and latency as the deleted share grows (random items, whole categories),
against an HNSW rebuilt on the live items; and IVF's in-place `remove_ids`, timed.

Queries are users of old categories ("old") and of new categories ("new"), fixed for the whole
run and drawn from a separate sample of the final distribution; each checkpoint scores them
against the exact top-k of the catalog at that point.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import faiss
import numpy as np

from ml.ann import indexes, metrics, vectors
from ml.ann.bench import Run, build_with, evaluate, kth_scores


@dataclass(frozen=True)
class Plan:
    initial: int = 500_000
    batches: int = 5
    batch_size: int = 100_000
    new_share: float = 0.6  # share of new-category items in a drift batch
    new_fraction: float = 0.2  # share of the mixture components held out as new categories
    nlist: int = 4096
    nprobe: int = 32
    hnsw_m: int = 32
    hnsw_efc: int = 200
    ef_search: int = 256
    queries: int = 1000  # per group
    query_pool: int = 200_000
    delete_fractions: tuple[float, ...] = (0.1, 0.3, 0.5)
    ivf_remove_fraction: float = 0.1


def new_components(mixture: vectors.Mixture, fraction: float, seed: int) -> np.ndarray:
    """A random component and its nearest components by mean direction (the new categories)."""
    mu = mixture.means / np.linalg.norm(mixture.means, axis=1, keepdims=True)
    first = np.random.default_rng(seed).integers(mixture.k)
    order = np.argsort(-(mu @ mu[first]), kind="stable")
    return np.sort(order[: max(1, round(fraction * mixture.k))])


def batches(mixture, plan: Plan, old, new, seed: int, drift: bool) -> list[tuple]:
    """(vectors, components) of each batch, old and new categories interleaved."""
    out = []
    for b in range(plan.batches):
        n_new = round(plan.new_share * plan.batch_size) if drift else 0
        parts = [vectors.make_catalog(mixture, plan.batch_size - n_new, seed + 10 * b + 1, old)]
        if n_new:
            parts.append(vectors.make_catalog(mixture, n_new, seed + 10 * b + 2, new))
        x = np.concatenate([p[0] for p in parts])
        labels = np.concatenate([p[1] for p in parts])
        perm = np.random.default_rng(seed + 10 * b + 3).permutation(len(x))
        out.append((x[perm], labels[perm]))
    return out


def ivf_state(index, data, groups: dict, run: Run, nprobe: int) -> dict:
    indexes.set_param(index, "nprobe", nprobe)
    return {
        "balance": metrics.list_balance(indexes.list_sizes(index)),
        "groups": {g: evaluate(index, data, q, kth, run) for g, (q, kth) in groups.items()},
    }


def hnsw_groups(index, data, groups: dict, run: Run, ef: int) -> dict:
    indexes.set_param(index, "efSearch", ef)
    return {g: evaluate(index, data, q, kth, run) for g, (q, kth) in groups.items()}


def _with_kth(data: np.ndarray, queries: dict[str, np.ndarray], k: int) -> dict:
    return {g: (q, kth_scores(data, q, k)) for g, q in queries.items()}


def run(mixture: vectors.Mixture, plan: Plan, run: Run, log=print) -> dict:
    seed = run.seed
    new = new_components(mixture, plan.new_fraction, seed)
    old = np.setdiff1d(np.arange(mixture.k), new)

    pool, pool_labels = vectors.make_catalog(mixture, plan.query_pool, seed + 1)
    queries = {
        "old": vectors.make_queries(pool, pool_labels, plan.queries, seed + 2, homes=old)[0],
        "new": vectors.make_queries(pool, pool_labels, plan.queries, seed + 3, homes=new)[0],
    }
    data, labels = vectors.make_catalog(mixture, plan.initial, seed + 4, old)
    drift = batches(mixture, plan, old, new, seed + 100, drift=True)
    control = batches(mixture, plan, old, new, seed + 200, drift=False)
    ctrl_data = data
    groups = _with_kth(data, queries, run.k)
    log(f"  t=0: {len(data):,} items")
    ivf = build_with(run, indexes.build_ivf_flat, data, plan.nlist, seed)
    ivf_ctrl = faiss.clone_index(ivf.index)
    hnsw = build_with(run, indexes.build_hnsw, data, plan.hnsw_m, plan.hnsw_efc)
    state0 = ivf_state(ivf.index, data, groups, run, plan.nprobe)
    checkpoints = [
        {
            "t": 0,
            "items": len(data),
            "new_items": 0,
            "ivf_incr": {**state0, "add_s": ivf.build_s},
            "ivf_control": {
                "balance": state0["balance"],
                "groups": {"old": state0["groups"]["old"]},
            },
            "ivf_rebuilt": {**state0, "train_s": ivf.train_s, "add_s": ivf.add_s},
            "hnsw_incr": {
                "add_s": hnsw.add_s,
                "groups": hnsw_groups(hnsw.index, data, groups, run, plan.ef_search),
            },
        }
    ]
    n_new = 0
    for t, ((batch, batch_labels), (ctrl_batch, _)) in enumerate(
        zip(drift, control, strict=True), start=1
    ):
        data = np.concatenate([data, batch])
        labels = np.concatenate([labels, batch_labels])
        ctrl_data = np.concatenate([ctrl_data, ctrl_batch])
        n_new += round(plan.new_share * plan.batch_size)
        log(f"  t={t}: {len(data):,} items ({n_new:,} in new categories)")
        groups = _with_kth(data, queries, run.k)
        ctrl_groups = _with_kth(ctrl_data, {"old": queries["old"]}, run.k)

        t0 = time.perf_counter()
        build_with(run, ivf.index.add, batch)
        ivf_add = time.perf_counter() - t0
        build_with(run, ivf_ctrl.add, ctrl_batch)
        t0 = time.perf_counter()
        build_with(run, hnsw.index.add, batch)
        hnsw_add = time.perf_counter() - t0
        rebuilt = build_with(run, indexes.build_ivf_flat, data, plan.nlist, seed)

        checkpoints.append(
            {
                "t": t,
                "items": len(data),
                "new_items": n_new,
                "ivf_incr": {
                    **ivf_state(ivf.index, data, groups, run, plan.nprobe),
                    "add_s": ivf_add,
                },
                "ivf_control": ivf_state(ivf_ctrl, ctrl_data, ctrl_groups, run, plan.nprobe),
                "ivf_rebuilt": {
                    **ivf_state(rebuilt.index, data, groups, run, plan.nprobe),
                    "train_s": rebuilt.train_s,
                    "add_s": rebuilt.add_s,
                },
                "hnsw_incr": {
                    "add_s": hnsw_add,
                    "groups": hnsw_groups(hnsw.index, data, groups, run, plan.ef_search),
                },
            }
        )
        del rebuilt

    log("  HNSW rebuilt on the final catalog")
    fresh = build_with(run, indexes.build_hnsw, data, plan.hnsw_m, plan.hnsw_efc)
    hnsw_rebuilt = {
        "build_s": fresh.add_s,
        "bytes": indexes.index_bytes(fresh.index),
        "groups": hnsw_groups(fresh.index, data, groups, run, plan.ef_search),
    }
    del fresh

    log("  deletions")
    categories = np.where(np.isin(labels, old), labels, -1)  # only old categories are deleted
    deletion = deletions(hnsw.index, data, queries, categories, plan, run)
    t0 = time.perf_counter()
    victims = np.random.default_rng(seed + 7).choice(
        len(data), round(plan.ivf_remove_fraction * len(data)), replace=False
    )
    removed = ivf.index.remove_ids(faiss.IDSelectorBatch(victims.astype(np.int64)))
    ivf_remove = {
        "fraction": plan.ivf_remove_fraction,
        "removed": int(removed),
        "seconds": time.perf_counter() - t0,
    }
    return {
        "plan": asdict(plan),
        "new_components": new.tolist(),
        "new_weight": float(mixture.weights[new].sum()),
        "checkpoints": checkpoints,
        "hnsw_incr_bytes": indexes.index_bytes(hnsw.index),
        "hnsw_rebuilt": hnsw_rebuilt,
        "deletion": deletion,
        "ivf_remove": ivf_remove,
    }


def deletion_ids(pattern: str, fraction: float, categories: np.ndarray, rng) -> np.ndarray:
    """Ids to delete: `random` -- a uniform subset of all items; `category` -- whole categories
    (items with a category >= 0, categories in random order) until `fraction` of all items."""
    n = len(categories)
    target = round(fraction * n)
    if pattern == "random":
        return np.sort(rng.choice(n, target, replace=False))
    chosen: list[np.ndarray] = []
    total = 0
    for c in rng.permutation(np.unique(categories[categories >= 0])):
        members = np.flatnonzero(categories == c)
        chosen.append(members)
        total += len(members)
        if total >= target:
            break
    return np.sort(np.concatenate(chosen))


def deletions(index, data, queries, categories, plan: Plan, run: Run) -> dict:
    """Filter-at-search-time vs. rebuild, for random and whole-category deletions."""
    all_q = np.concatenate([queries["old"], queries["new"]])
    rows = []
    for pattern in ("random", "category"):
        rng = np.random.default_rng(run.seed + (11 if pattern == "random" else 13))
        for fraction in plan.delete_fractions:
            dead = deletion_ids(pattern, fraction, categories, rng)
            live = np.ones(len(data), dtype=bool)
            live[dead] = False
            live_ids = np.flatnonzero(live)
            kth = kth_scores(data[live_ids], all_q, run.k)
            bits = np.packbits(live, bitorder="little")
            sel = faiss.IDSelectorBitmap(len(bits), faiss.swig_ptr(bits))
            out = {
                "pattern": pattern,
                "fraction": fraction,
                "items": len(data),
                "deleted": int(len(dead)),
            }
            for name, ef in (
                ("filter", plan.ef_search),
                ("filter_scaled_ef", round(plan.ef_search * len(data) / len(live_ids))),
            ):
                params = faiss.SearchParametersHNSW()
                params.efSearch = ef
                params.sel = sel
                out[name] = {"ef_search": ef, **evaluate(index, data, all_q, kth, run, params)}
            sub = data[live_ids]
            fresh = build_with(run, indexes.build_hnsw, sub, plan.hnsw_m, plan.hnsw_efc)
            indexes.set_param(fresh.index, "efSearch", plan.ef_search)
            out["rebuild"] = {"build_s": fresh.add_s, **evaluate(fresh.index, sub, all_q, kth, run)}
            del fresh
            rows.append(out)
            del sel, bits
    return {"ef_search": plan.ef_search, "rows": rows}
