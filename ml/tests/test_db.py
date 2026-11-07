"""The ingested catalog in PostgreSQL. Skipped when the database is not reachable."""

import os

import pandas as pd
import pytest

from ml import paths
from ml.data.ingest import DEFAULT_DATABASE_URL

psycopg = pytest.importorskip("psycopg")


@pytest.fixture(scope="module")
def conn():
    url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    try:
        c = psycopg.connect(url, connect_timeout=3)
    except psycopg.OperationalError as e:
        pytest.skip(f"database not reachable ({e.__class__.__name__}): run `make up`")
    with c:
        yield c


def _one(conn, sql):
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchone()


def test_items_table_matches_catalog(conn):
    n, with_asin, distinct_asin, lo, hi = _one(
        conn,
        "SELECT count(*), count(asin), count(DISTINCT asin), min(item_id), max(item_id) FROM items",
    )
    if n == 0:
        pytest.skip("items is empty: run `make ingest`")
    assert (n, with_asin, distinct_asin, lo, hi) == (3459, 3459, 3459, 0, 3458)
    if paths.CATALOG_PARQUET.exists():
        items = pd.read_parquet(paths.CATALOG_PARQUET)
        with conn.cursor() as cur:
            cur.execute("SELECT item_id, asin, title, price_cents FROM items ORDER BY item_id")
            db = pd.DataFrame(cur.fetchall(), columns=["item_id", "asin", "title", "price_cents"])
        assert db.asin.tolist() == items.asin.tolist()
        assert db.title.tolist() == items.title.tolist()
        assert db.price_cents.isna().sum() == items.price_cents.isna().sum()


def test_users_table(conn):
    n, distinct_ext = _one(conn, "SELECT count(*), count(DISTINCT external_id) FROM users")
    if n == 0:
        pytest.skip("users is empty: run `make ingest`")
    assert n == distinct_ext == 8328
