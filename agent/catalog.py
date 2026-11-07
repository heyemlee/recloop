"""The catalog behind the tools: items, purchases and hybrid search (design §10.2).

`search` fuses two rankings with reciprocal rank fusion (k = 60, as in the recommendation
mixer): BM25 over the item text, and the exact pgvector KNN of the query embedding against
`items.text_emb`. Both rankings only see items that pass the filters, and the price, stock and
category filters are applied in SQL on every call, so a stale index can never let a sold-out or
over-budget item through. (The code checks every item again before it is shown.)

BM25 runs in-process on the agent's own tokens (`agent.text`): the catalog text is read once at
start-up (3,459 items, 3.4 MB), the tokens keep model numbers intact, and a query costs about a
millisecond. PostgreSQL's `to_tsvector('english', ...)` splits model numbers inconsistently and,
without a stored tsvector column, costs ~150 ms per query over the whole table.

`MemoryCatalog` is the same logic over Python objects, for tests.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from agent import taxonomy, text
from agent.encoder import QueryEncoder
from agent.schema import CategoryKey

RRF_K = 60
CANDIDATES_PER_RETRIEVER = 100
FIELD_WEIGHTS = {"title": 3.0, "brand": 3.0, "category": 2.0, "features": 1.5, "description": 1.0}
BM25_K1, BM25_B = 1.2, 0.75

ItemField = Literal["title", "description", "features"]

ITEM_COLUMNS = (
    "item_id, asin, title, brand, categories, price_cents, description, features, image_url, stock"
)


@dataclass(frozen=True)
class Item:
    item_id: int
    asin: str
    title: str
    brand: str | None
    categories: tuple[str, ...]
    price_cents: int | None
    description: str | None
    features: tuple[str, ...]
    image_url: str | None
    stock: int

    def field_text(self, name: ItemField) -> str:
        if name == "title":
            return self.title
        if name == "description":
            return self.description or ""
        return "\n".join(self.features)

    @classmethod
    def from_row(cls, row: Sequence) -> Item:
        item_id, asin, title, brand, cats, price, desc, feats, image, stock = row
        return cls(
            int(item_id),
            asin,
            title,
            brand,
            tuple(cats or ()),
            price,
            desc,
            tuple(feats or ()),
            image,
            int(stock),
        )


@dataclass(frozen=True)
class Purchase:
    item_id: int
    purchased_at: datetime


class SearchFilters(BaseModel):
    """The `filters` argument of search_items."""

    model_config = ConfigDict(extra="forbid")

    category: CategoryKey | None = None
    max_price_cents: int | None = Field(default=None, ge=1)
    in_stock: bool = True
    brands: list[str] = Field(default_factory=list)
    exclude_brands: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class SearchHit:
    item_id: int
    score: float  # reciprocal rank fusion score
    text_rank: int | None  # 1-based rank in the BM25 list, None if not in it
    vector_rank: int | None


# --- Brands ---------------------------------------------------------------------------------

_BRAND_ALIASES = {"hewlett packard": "hp"}


def norm_brand(brand: str | None) -> str:
    b = text.norm_phrase(brand or "")
    return _BRAND_ALIASES.get(b, b)


def brand_matches(item_brand: str | None, wanted: str) -> bool:
    """'Brother Reseller' and 'CANON USA, INC.' match 'Brother' and 'Canon'; word prefixes only."""
    a, b = norm_brand(item_brand), norm_brand(wanted)
    if not a or not b:
        return False
    return a == b or a.startswith(b + " ") or b.startswith(a + " ")


def brand_allowed(item_brand: str | None, brands: Sequence[str], exclude: Sequence[str]) -> bool:
    if brands and not any(brand_matches(item_brand, w) for w in brands):
        return False
    return not any(brand_matches(item_brand, w) for w in exclude)


# --- BM25 -------------------------------------------------------------------------------------


def _item_fields(item: Item) -> dict[str, str]:
    return {
        "title": item.title,
        "brand": item.brand or "",
        # The two top levels ('Office Products', 'Office & School Supplies') say nothing.
        "category": " ".join(item.categories[2:]),
        "features": " ".join(item.features),
        "description": item.description or "",
    }


class BM25Index:
    """BM25 with per-field weights on term frequency and length (a light BM25F)."""

    def __init__(self, items: Iterable[Item]):
        ids: list[int] = []
        lengths: list[float] = []
        postings: dict[str, dict[int, float]] = defaultdict(dict)
        for row, item in enumerate(items):
            ids.append(item.item_id)
            length = 0.0
            for name, value in _item_fields(item).items():
                w = FIELD_WEIGHTS[name]
                toks = text.tokens(value)
                length += w * len(toks)
                for t in toks:
                    postings[t][row] = postings[t].get(row, 0.0) + w
            lengths.append(length)
        self.ids = np.asarray(ids, dtype=np.int64)
        self.row_of = {item_id: row for row, item_id in enumerate(ids)}
        n = len(ids)
        avg = (sum(lengths) / n) if n else 1.0
        norm = BM25_K1 * (1 - BM25_B + BM25_B * np.asarray(lengths) / avg)
        self._postings: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        for t, rows in postings.items():
            r = np.fromiter(rows.keys(), dtype=np.int64, count=len(rows))
            tf = np.fromiter(rows.values(), dtype=np.float64, count=len(rows))
            idf = math.log(1 + (n - len(rows) + 0.5) / (len(rows) + 0.5))
            self._postings[t] = (r, idf * tf * (BM25_K1 + 1) / (tf + norm[r]), idf)

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.ids))
        for t in text.query_tokens(query):
            hit = self._postings.get(t)
            if hit is not None:
                s[hit[0]] += hit[1]
        return s

    def rank(self, query: str, eligible: np.ndarray, k: int) -> list[int]:
        """Item ids of the top-k eligible rows with a positive score (ties: lower item_id)."""
        s = self.scores(query)
        rows = np.flatnonzero(eligible & (s > 0))
        order = np.lexsort((self.ids[rows], -s[rows]))[:k]
        return self.ids[rows[order]].tolist()


def rrf(rankings: Sequence[Sequence[int]], limit: int) -> list[SearchHit]:
    """Reciprocal rank fusion of item-id rankings (best first); ties broken by item_id."""
    score: dict[int, float] = defaultdict(float)
    ranks: dict[int, list[int | None]] = defaultdict(lambda: [None] * len(rankings))
    for li, ranking in enumerate(rankings):
        for r, item_id in enumerate(ranking, start=1):
            score[item_id] += 1.0 / (RRF_K + r)
            ranks[item_id][li] = r
    best = sorted(score, key=lambda i: (-score[i], i))[:limit]
    hits = []
    for i in best:
        rs = ranks[i] + [None, None]
        hits.append(SearchHit(i, score[i], rs[0], rs[1]))
    return hits


# --- Catalogs ---------------------------------------------------------------------------------


@dataclass
class SearchStats:
    """What the last search saw (for the trace and the reply)."""

    eligible: int = 0
    text_candidates: int = 0
    vector_candidates: int = 0


class Catalog(ABC):
    encoder: QueryEncoder
    index: BM25Index
    last_stats: SearchStats

    def search(
        self,
        query: str,
        filters: SearchFilters,
        limit: int = 30,
        mode: Literal["hybrid", "text", "vector"] = "hybrid",
    ) -> list[SearchHit]:
        eligible_ids = [
            item_id
            for item_id, brand in self._eligible(filters)
            if brand_allowed(brand, filters.brands, filters.exclude_brands)
        ]
        stats = SearchStats(eligible=len(eligible_ids))
        rankings: list[list[int]] = []
        if eligible_ids and mode in ("hybrid", "text"):
            mask = np.zeros(len(self.index.ids), dtype=bool)
            mask[[self.index.row_of[i] for i in eligible_ids if i in self.index.row_of]] = True
            ranked = self.index.rank(query, mask, CANDIDATES_PER_RETRIEVER)
            stats.text_candidates = len(ranked)
            rankings.append(ranked)
        elif mode == "vector":
            rankings.append([])
        if eligible_ids and mode in ("hybrid", "vector"):
            ranked = self._nearest(
                self.encoder.encode(query), eligible_ids, CANDIDATES_PER_RETRIEVER
            )
            stats.vector_candidates = len(ranked)
            rankings.append(ranked)
        self.last_stats = stats
        return rrf(rankings, limit) if rankings else []

    @abstractmethod
    def _eligible(self, filters: SearchFilters) -> list[tuple[int, str | None]]:
        """(item_id, brand) of the items passing the price, stock and category filters."""

    @abstractmethod
    def _nearest(self, vector: np.ndarray, item_ids: Sequence[int], k: int) -> list[int]:
        """Item ids among `item_ids` by descending cosine similarity (ties: lower item_id)."""

    @abstractmethod
    def items(self, item_ids: Sequence[int]) -> list[Item]:
        """Current rows for these ids, in the given order (unknown ids are skipped)."""

    @abstractmethod
    def all_items(self) -> list[Item]:
        """Every item, current rows (for the evaluation's satisfiable sets)."""

    @abstractmethod
    def purchases(self, user_id: int, limit: int) -> list[Purchase]:
        """The user's orders, most recent first."""


def _passes_sql_filters(item: Item, f: SearchFilters) -> bool:
    """What PgCatalog's WHERE clause does, for MemoryCatalog."""
    if f.in_stock and item.stock <= 0:
        return False
    if f.max_price_cents is not None and (
        item.price_cents is None or item.price_cents > f.max_price_cents
    ):
        return False
    return f.category is None or taxonomy.in_category(item.categories, f.category)


@dataclass
class MemoryCatalog(Catalog):
    """In-memory catalog: tests and offline experiments."""

    catalog_items: list[Item]
    embeddings: np.ndarray  # (n, 384), row i = catalog_items[i]
    encoder: QueryEncoder
    orders: dict[int, list[Purchase]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.index = BM25Index(self.catalog_items)
        self._by_id = {it.item_id: it for it in self.catalog_items}
        self._row = {it.item_id: r for r, it in enumerate(self.catalog_items)}
        self.last_stats = SearchStats()

    def _eligible(self, filters: SearchFilters) -> list[tuple[int, str | None]]:
        return [
            (it.item_id, it.brand) for it in self.catalog_items if _passes_sql_filters(it, filters)
        ]

    def _nearest(self, vector: np.ndarray, item_ids: Sequence[int], k: int) -> list[int]:
        ids = np.asarray(item_ids, dtype=np.int64)
        sims = self.embeddings[[self._row[i] for i in item_ids]] @ vector
        order = np.lexsort((ids, -sims))[:k]
        return ids[order].tolist()

    def items(self, item_ids: Sequence[int]) -> list[Item]:
        return [self._by_id[i] for i in item_ids if i in self._by_id]

    def all_items(self) -> list[Item]:
        return list(self.catalog_items)

    def purchases(self, user_id: int, limit: int) -> list[Purchase]:
        rows = sorted(self.orders.get(user_id, []), key=lambda p: p.purchased_at, reverse=True)
        return rows[:limit]


class PgCatalog(Catalog):
    """The `items` and `orders` tables of the RecLoop database, read-only."""

    def __init__(self, conn, encoder: QueryEncoder):
        # Every statement on this session runs read-only, enforced by the server.
        conn.autocommit = True
        conn.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
        self.conn = conn
        self.encoder = encoder
        self.index = BM25Index(self.all_items())
        self.last_stats = SearchStats()

    @classmethod
    def connect(cls, url: str, encoder: QueryEncoder) -> PgCatalog:
        import psycopg

        return cls(psycopg.connect(url, connect_timeout=5), encoder)

    def _eligible(self, filters: SearchFilters) -> list[tuple[int, str | None]]:
        where, params = ["TRUE"], {}
        if filters.in_stock:
            where.append("stock > 0")
        if filters.max_price_cents is not None:
            where.append("price_cents IS NOT NULL AND price_cents <= %(max_price)s")
            params["max_price"] = filters.max_price_cents
        if filters.category is not None:
            cat = taxonomy.CATEGORIES[filters.category]
            where.append("categories && %(cat_include)s::text[]")
            params["cat_include"] = list(cat.include)
            if cat.exclude:
                where.append("NOT (categories && %(cat_exclude)s::text[])")
                params["cat_exclude"] = list(cat.exclude)
        sql = f"SELECT item_id, brand FROM items WHERE {' AND '.join(where)} ORDER BY item_id"
        return [(int(i), b) for i, b in self.conn.execute(sql, params).fetchall()]

    def _nearest(self, vector: np.ndarray, item_ids: Sequence[int], k: int) -> list[int]:
        literal = "[" + ",".join(f"{x:.7g}" for x in vector.tolist()) + "]"
        rows = self.conn.execute(
            "SELECT item_id FROM items WHERE item_id = ANY(%s) AND text_emb IS NOT NULL "
            "ORDER BY text_emb <=> %s::vector, item_id LIMIT %s",
            (list(item_ids), literal, k),
        ).fetchall()
        return [int(r[0]) for r in rows]

    def items(self, item_ids: Sequence[int]) -> list[Item]:
        rows = self.conn.execute(
            f"SELECT {ITEM_COLUMNS} FROM items WHERE item_id = ANY(%s)", (list(item_ids),)
        ).fetchall()
        by_id = {int(r[0]): Item.from_row(r) for r in rows}
        return [by_id[i] for i in item_ids if i in by_id]

    def all_items(self) -> list[Item]:
        rows = self.conn.execute(f"SELECT {ITEM_COLUMNS} FROM items ORDER BY item_id").fetchall()
        return [Item.from_row(r) for r in rows]

    def purchases(self, user_id: int, limit: int) -> list[Purchase]:
        rows = self.conn.execute(
            "SELECT item_id, created_at FROM orders WHERE user_id = %s "
            "ORDER BY created_at DESC, order_id DESC LIMIT %s",
            (user_id, limit),
        ).fetchall()
        return [Purchase(int(i), ts) for i, ts in rows]
