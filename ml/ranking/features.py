"""Features of the LightGBM ranker (docs/design.md §4.3), one row per (sample, candidate).

    sasrec_score, sasrec_rank   SASRec's inner product and rank, if SASRec returned the item
    itemcf_score, itemcf_rank   the same for ItemCF
    pop_rank                    rank in the popular list, if the item is in its top 200
    pop_log                     log(1 + training interactions of the item)
    n_sources                   how many of the three sources returned the item
    price_bucket                0..5 for < $5, < $10, < $20, < $50, < $100, >= $100
    cat2_overlap                share of the history items in the item's level-2 category
    leaf_overlap                share of the history items with the item's full category path

A missing value is NaN (LightGBM's default handling): a source that did not return the item,
an unknown price, an item without a category at that level. Ranks are 1-based.

Everything here is computable for the dataset's validation and test periods as well as online:
the item tables (`ItemTables`) ship in the bundle, the source ranks and scores come with the
candidates, the history comes with the request. The model server re-implements `compute` and
checks itself against the bundle's golden queries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml import paths

FEATURES = (
    "sasrec_score",
    "sasrec_rank",
    "itemcf_score",
    "itemcf_rank",
    "pop_rank",
    "pop_log",
    "n_sources",
    "price_bucket",
    "cat2_overlap",
    "leaf_overlap",
)
PRICE_EDGES_CENTS = (500, 1000, 2000, 5000, 10000)
CATEGORY_LEVEL = 2  # categories[2]: 37 values ("Writing & Correction Supplies", "Paper", ...)


@dataclass(frozen=True)
class ItemTables:
    """Per-item inputs of the features, indexed by item_id."""

    pop_count: np.ndarray  # (num_items,) training interactions
    price_bucket: np.ndarray  # (num_items,) float, NaN = unknown price
    cat2: np.ndarray  # (num_items,) int, -1 = no category at that level
    leaf: np.ndarray  # (num_items,) int, -1 = no category

    @classmethod
    def build(cls, pop_count: np.ndarray, catalog: pd.DataFrame | None = None) -> ItemTables:
        if catalog is None:
            catalog = pd.read_parquet(
                paths.CATALOG_PARQUET, columns=["item_id", "categories", "price_cents"]
            )
        catalog = catalog.sort_values("item_id")
        assert np.array_equal(catalog.item_id.to_numpy(), np.arange(len(pop_count)))
        price = catalog.price_cents.astype("Float64").to_numpy(dtype=np.float64, na_value=np.nan)
        bucket = np.where(
            np.isnan(price), np.nan, np.searchsorted(PRICE_EDGES_CENTS, price, side="right")
        )
        cats = [list(c) if c is not None else [] for c in catalog.categories]
        cat2 = _codes([c[CATEGORY_LEVEL] if len(c) > CATEGORY_LEVEL else None for c in cats])
        leaf = _codes([" > ".join(c) if c else None for c in cats])
        return cls(np.asarray(pop_count, dtype=np.float64), bucket.astype(np.float64), cat2, leaf)

    def to_json(self) -> dict:
        """The bundle's copy (features.json): what the model server computes the features from."""
        return {
            "features": list(FEATURES),
            "price_edges_cents": list(PRICE_EDGES_CENTS),
            "category_level": CATEGORY_LEVEL,
            "pop_count": [int(x) for x in self.pop_count],
            "price_bucket": [None if np.isnan(x) else int(x) for x in self.price_bucket],
            "cat2": [int(x) for x in self.cat2],
            "leaf": [int(x) for x in self.leaf],
        }


def _codes(values: list[str | None]) -> np.ndarray:
    """Dense integer codes in sorted order of the distinct values; None -> -1."""
    distinct = sorted({v for v in values if v is not None})
    index = {v: i for i, v in enumerate(distinct)}
    return np.asarray([index[v] if v is not None else -1 for v in values], dtype=np.int64)


def _overlap(cand: np.ndarray, hist: np.ndarray, hist_len: np.ndarray) -> np.ndarray:
    """Share of the history items whose code equals the candidate's; NaN if the candidate has no
    code. cand (n, d), hist (n, h) padded with -2 (never matches), hist_len (n,)."""
    same = (cand[:, :, None] == hist[:, None, :]).sum(axis=2).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = same / hist_len[:, None]
    return np.where(cand < 0, np.nan, share)


def compute(cands, history: list[np.ndarray], tables: ItemTables) -> np.ndarray:
    """(n, depth, len(FEATURES)) float32; padded candidate slots are all NaN."""
    items = cands.items
    valid = items >= 0
    safe = np.where(valid, items, 0)
    n, depth = items.shape
    h = max(1, max((len(x) for x in history), default=1))
    hist = np.full((n, h), -1, dtype=np.int64)
    for i, x in enumerate(history):
        hist[i, : len(x)] = x
    hist_len = np.asarray([len(x) for x in history], dtype=np.float64)
    hist_ok = hist >= 0
    hist_cat2 = np.where(hist_ok, tables.cat2[np.where(hist_ok, hist, 0)], -2)
    hist_cat2 = np.where(hist_cat2 < 0, -2, hist_cat2)
    hist_leaf = np.where(hist_ok, tables.leaf[np.where(hist_ok, hist, 0)], -2)
    hist_leaf = np.where(hist_leaf < 0, -2, hist_leaf)

    def rank(s: str) -> np.ndarray:
        r = cands.ranks[s].astype(np.float64)
        return np.where(r > 0, r, np.nan)

    cols = {
        "sasrec_score": cands.scores["sasrec"],
        "sasrec_rank": rank("sasrec"),
        "itemcf_score": cands.scores["itemcf"],
        "itemcf_rank": rank("itemcf"),
        "pop_rank": rank("pop"),
        "pop_log": np.log1p(tables.pop_count[safe]),
        "n_sources": cands.n_sources().astype(np.float64),
        "price_bucket": tables.price_bucket[safe],
        "cat2_overlap": _overlap(tables.cat2[safe], hist_cat2, hist_len),
        "leaf_overlap": _overlap(tables.leaf[safe], hist_leaf, hist_len),
    }
    x = np.stack([np.asarray(cols[f], dtype=np.float64) for f in FEATURES], axis=2)
    x[~valid] = np.nan
    return x.astype(np.float32)
