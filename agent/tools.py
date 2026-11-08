"""The agent's tools (design §10.2), with a trace of every call.

  * search_items(query, filters): hybrid search with SQL filters (`Catalog.search`).
  * get_item_details(ids): current rows of the items table.
  * get_purchase_history(user_id): the user's purchases, most recent first (orders by default;
    in the store also the dataset users' training period, see history.py).
  * get_recommendations(user_id, seed_items, limit): the store's chat surface
    (/v1/recommendations?surface=chat, no ads), the personalisation signal. Only the store's
    assistant has a recommender; the evaluation's cases have no user state, so it is not called.

The trace is what the evaluation scores as tool-call correctness: the expected calls follow from
the gold constraints (history only when the request refers to an earlier purchase, the filters
equal to the gold constraints, no repeated or extra calls).

The sponsored item is not a tool: the agent's code asks the ads module for it after the checks
(`agent.agent`, design §10.2).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent.catalog import Catalog, Item, Purchase, SearchFilters, SearchHit
from agent.store_api import Recommendations, Recommender

PurchaseSource = Callable[[int, int], list[Purchase]]


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    result_ids: list[int]
    ms: float


@dataclass
class Tools:
    catalog: Catalog
    # Where get_purchase_history reads from: the orders table by default; the store passes
    # history.StorePurchases (orders plus the dataset users' training period), the evaluation
    # each case's own purchase list.
    purchases: PurchaseSource | None = None
    search_mode: str = "hybrid"
    recommender: Recommender | None = None
    trace: list[ToolCall] = field(default_factory=list)

    def _record(self, name: str, args: dict[str, Any], ids: list[int], t0: float) -> None:
        self.trace.append(ToolCall(name, args, ids, (time.perf_counter() - t0) * 1000))

    def search_items(self, query: str, filters: SearchFilters, limit: int = 30) -> list[SearchHit]:
        t0 = time.perf_counter()
        hits = self.catalog.search(query, filters, limit, mode=self.search_mode)  # type: ignore[arg-type]
        args = {"query": query, "filters": filters.model_dump(), "limit": limit}
        self._record("search_items", args, [h.item_id for h in hits], t0)
        return hits

    def get_item_details(self, item_ids: Sequence[int]) -> list[Item]:
        t0 = time.perf_counter()
        items = self.catalog.items(list(item_ids))
        self._record("get_item_details", {"ids": list(item_ids)}, [i.item_id for i in items], t0)
        return items

    def get_purchase_history(self, user_id: int, limit: int = 20) -> list[tuple[Purchase, Item]]:
        t0 = time.perf_counter()
        source = self.purchases or self.catalog.purchases
        rows = source(user_id, limit)
        items = {i.item_id: i for i in self.catalog.items([p.item_id for p in rows])}
        out = [(p, items[p.item_id]) for p in rows if p.item_id in items]
        args = {"user_id": user_id, "limit": limit}
        self._record("get_purchase_history", args, [p.item_id for p, _ in out], t0)
        return out

    def get_recommendations(
        self, user_id: int, seed_items: Sequence[int], limit: int = 50
    ) -> Recommendations:
        assert self.recommender is not None
        t0 = time.perf_counter()
        recs = self.recommender.recommend(user_id, list(seed_items), limit)
        args = {"user_id": user_id, "seed_items": list(seed_items), "limit": limit}
        self._record("get_recommendations", args, [h.item_id for h in recs.hits], t0)
        return recs
