"""Purchase history as the store counts it: orders plus the dataset users' training period.

The store treats each dataset user's training-period interactions as having happened, and a
review as a purchase (design §2.2): `make seed` puts those items in `u:{uid}:bought`, so the feed
never offers them again. They are not in `orders`, which holds only the purchases made in the
running store (one per purchase event). `StorePurchases` reads both, so "the one I bought last
time" is the same item in the chat as in the feed. The evaluation does not use it: each case
brings its own purchase list.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from agent import paths
from agent.catalog import Catalog, Purchase

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class DatasetPurchases:
    """The training-period rows of ml/'s interactions table, read once, sorted by user and time."""

    def __init__(self, user_ids: np.ndarray, item_ids: np.ndarray, times_ms: np.ndarray):
        order = np.lexsort((times_ms, user_ids))
        self.user_ids = user_ids[order]
        self.item_ids = item_ids[order]
        self.times_ms = times_ms[order]

    @classmethod
    def load(cls, path: Path | None = None) -> DatasetPurchases:
        table = pq.read_table(
            path or paths.interactions_parquet(),
            columns=["user_id", "item_id", "ts", "seq_idx", "split"],
        )
        train = table.filter(pc.equal(table["split"], "train"))

        def column(name: str) -> np.ndarray:
            return train[name].to_numpy().astype(np.int64)

        # The time `make seed` gives u:{uid}:hist: the review time (whole days) in ms plus the
        # position in the user's sequence, which keeps the dataset's order inside a day.
        return cls(column("user_id"), column("item_id"), column("ts") * 1000 + column("seq_idx"))

    def __len__(self) -> int:
        return len(self.user_ids)

    def __call__(self, user_id: int, limit: int) -> list[Purchase]:
        lo, hi = np.searchsorted(self.user_ids, [user_id, user_id + 1])
        newest_first = range(hi - 1, max(lo, hi - limit) - 1, -1)
        return [
            Purchase(int(self.item_ids[k]), EPOCH + timedelta(milliseconds=int(self.times_ms[k])))
            for k in newest_first
        ]


class StorePurchases:
    """Orders plus DatasetPurchases, most recent first: what a store-facing agent passes as
    `purchases`. Without it the agent reads the orders table only."""

    def __init__(self, catalog: Catalog, dataset: DatasetPurchases):
        self.catalog = catalog
        self.dataset = dataset

    def __call__(self, user_id: int, limit: int) -> list[Purchase]:
        rows = self.catalog.purchases(user_id, limit) + self.dataset(user_id, limit)
        rows.sort(key=lambda p: p.purchased_at, reverse=True)  # store orders are the newer ones
        return rows[:limit]
