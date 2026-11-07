"""PgCatalog against the RecLoop database (read-only). Skipped when it is not reachable."""

import numpy as np
import pytest

from agent import paths
from agent.catalog import PgCatalog, SearchFilters
from agent.encoder import HashEncoder

psycopg = pytest.importorskip("psycopg")


@pytest.fixture(scope="module")
def catalog():
    try:
        conn = psycopg.connect(paths.database_url(), connect_timeout=3, autocommit=True)
    except psycopg.OperationalError as e:
        pytest.skip(f"database not reachable ({e.__class__.__name__}): run `make up`")
    try:
        n = conn.execute("SELECT count(*) FROM items").fetchone()[0]
    except psycopg.errors.UndefinedTable:
        pytest.skip("no items table: run `make ingest`")
    if n == 0:
        pytest.skip("items is empty: run `make ingest`")
    cat = PgCatalog(conn, HashEncoder())
    yield cat
    conn.close()


def test_session_is_read_only(catalog):
    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        catalog.conn.execute("UPDATE items SET stock = stock WHERE item_id = -1")


def test_search_applies_the_sql_filters(catalog):
    f = SearchFilters(category="ink_cartridges", max_price_cents=3000)
    hits = catalog.search("canon ink cartridge MG3620", f, limit=20)
    assert hits
    rows = catalog.items([h.item_id for h in hits])
    assert all(r.stock > 0 and r.price_cents is not None and r.price_cents <= 3000 for r in rows)
    assert all("Inkjet Printer Ink" in r.categories for r in rows)


def test_vector_neighbours_come_from_pgvector(catalog):
    rows = catalog.conn.execute("SELECT text_emb::text FROM items WHERE item_id = 0").fetchone()
    vec = np.array([float(x) for x in rows[0].strip("[]").split(",")], dtype=np.float32)
    assert catalog._nearest(vec, [0, 1, 2, 3], 2)[0] == 0  # an item is its own nearest neighbour


def test_purchases_query_runs(catalog):
    assert isinstance(catalog.purchases(-1, 5), list)
