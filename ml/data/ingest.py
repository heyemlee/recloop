"""Load the catalog and the dataset users into PostgreSQL.

Reads data/processed/items.parquet and users.parquet (written by `make data`) and upserts them
into `items` and `users` (schema: serving/migrations). Re-running is safe: rows are matched on
the primary key, and runtime columns (items.stock, users.created_at) are left alone on update.

  * items.text_emb comes from data/processed/text_emb.npy (ml.data.text_embed) and is rewritten
    on every run (it is derived from the catalog and a pinned model).
  * items.stock is set when an item is first inserted: simulated inventory (design §2.2), drawn
    with a fixed seed; about 3% of the items start sold out so the sold-out filter has work.

Usage: DATABASE_URL=postgres://... python -m ml.data.ingest
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import psycopg

from ml import paths

DEFAULT_DATABASE_URL = "postgres://recloop:recloop@localhost:15432/recloop?sslmode=disable"

STOCK_SEED = 20261106
SOLD_OUT_SHARE = 0.03
MAX_STOCK = 200

ITEM_COLUMNS = [
    "item_id",
    "asin",
    "title",
    "brand",
    "categories",
    "price_cents",
    "description",
    "features",
    "image_url",
]
ITEM_TYPES = ["int4", "text", "text", "text", "text[]", "int4", "text", "text[]", "text"]


def _none(value):
    """pandas missing values -> None; numpy arrays (list columns) -> list."""
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    if pd.isna(value):
        return None
    return value


def item_rows(items: pd.DataFrame):
    for row in items[ITEM_COLUMNS].itertuples(index=False):
        yield tuple(_none(v) for v in row)


def initial_stock(n: int) -> np.ndarray:
    """Simulated starting inventory per item_id (fixed seed): 1..MAX_STOCK, ~3% sold out."""
    rng = np.random.default_rng(STOCK_SEED)
    stock = rng.integers(1, MAX_STOCK + 1, size=n)
    stock[rng.random(n) < SOLD_OUT_SHARE] = 0
    return stock


def _vector_literal(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in v.tolist()) + "]"


def ingest(
    conn: psycopg.Connection,
    items: pd.DataFrame,
    users: pd.DataFrame,
    text_emb: np.ndarray | None = None,
) -> None:
    with conn.cursor() as cur:
        columns = ", ".join(f"{c} {t}" for c, t in zip(ITEM_COLUMNS, ITEM_TYPES, strict=True))
        cur.execute(f"CREATE TEMP TABLE items_stage ({columns}, stock int4) ON COMMIT DROP")
        stock = initial_stock(len(items))
        with cur.copy(f"COPY items_stage ({', '.join(ITEM_COLUMNS)}, stock) FROM STDIN") as copy:
            copy.set_types([*ITEM_TYPES, "int4"])
            for row, st in zip(item_rows(items), stock, strict=True):
                copy.write_row((*row, int(st)))
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in ITEM_COLUMNS[1:])
        cur.execute(
            f"INSERT INTO items ({', '.join(ITEM_COLUMNS)}, stock) "
            f"SELECT {', '.join(ITEM_COLUMNS)}, stock FROM items_stage "
            f"ON CONFLICT (item_id) DO UPDATE SET {updates}"
        )
        if text_emb is not None:
            cur.execute("CREATE TEMP TABLE emb_stage (item_id int4, emb text) ON COMMIT DROP")
            with cur.copy("COPY emb_stage (item_id, emb) FROM STDIN") as copy:
                copy.set_types(["int4", "text"])
                for item_id, v in zip(items.item_id, text_emb, strict=True):
                    copy.write_row((int(item_id), _vector_literal(v)))
            cur.execute(
                "UPDATE items i SET text_emb = s.emb::vector FROM emb_stage s "
                "WHERE i.item_id = s.item_id"
            )

        cur.execute(
            "CREATE TEMP TABLE users_stage (user_id BIGINT, external_id TEXT) ON COMMIT DROP"
        )
        with cur.copy("COPY users_stage (user_id, external_id) FROM STDIN") as copy:
            copy.set_types(["int8", "text"])
            for uid, ext in zip(users.user_id, users.external_id, strict=True):
                copy.write_row((int(uid), ext))
        cur.execute(
            "INSERT INTO users (user_id, external_id) SELECT user_id, external_id FROM users_stage "
            "ON CONFLICT (user_id) DO UPDATE SET external_id = EXCLUDED.external_id"
        )


def summary(conn: psycopg.Connection) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*), count(asin), count(price_cents), count(brand), count(image_url), "
            "count(description), min(item_id), max(item_id) FROM items"
        )
        n, asin, price, brand, image, desc, lo, hi = cur.fetchone()
        cur.execute("SELECT count(*) FROM users")
        (n_users,) = cur.fetchone()
        cur.execute("SELECT count(text_emb), count(*) FILTER (WHERE stock = 0) FROM items")
        with_emb, sold_out = cur.fetchone()
    return {
        "with_text_emb": with_emb,
        "sold_out": sold_out,
        "items": n,
        "with_asin": asin,
        "with_price": price,
        "with_brand": brand,
        "with_image": image,
        "with_description": desc,
        "item_id_range": (lo, hi),
        "users": n_users,
    }


def main() -> None:
    url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    items = pd.read_parquet(paths.CATALOG_PARQUET)
    users = pd.read_parquet(paths.USERS_PARQUET)
    text_emb = np.load(paths.TEXT_EMB_NPY) if paths.TEXT_EMB_NPY.exists() else None
    if text_emb is None:
        print("no data/processed/text_emb.npy (run `make data`): items.text_emb left as is")
    elif text_emb.shape != (len(items), 384):
        raise SystemExit(f"text_emb.npy has shape {text_emb.shape}, expected ({len(items)}, 384)")
    with psycopg.connect(url) as conn:
        ingest(conn, items, users, text_emb)
        conn.commit()
        s = summary(conn)
    n = s["items"]
    print(
        f"items  {n:,} rows, item_id {s['item_id_range'][0]}..{s['item_id_range'][1]}, "
        f"ASIN on {s['with_asin']:,}"
    )
    print(
        f"       price known for {s['with_price']:,} ({s['with_price'] / n:.1%}); "
        f"brand {s['with_brand'] / n:.1%}, image {s['with_image'] / n:.1%}, "
        f"description {s['with_description'] / n:.1%}"
    )
    print(
        f"       text_emb on {s['with_text_emb']:,}; {s['sold_out']:,} sold out "
        "(simulated stock, set on first insert)"
    )
    print(f"users  {s['users']:,} rows")
    if n != len(items) or s["with_asin"] != n:
        raise SystemExit("items table does not match the catalog")


if __name__ == "__main__":
    main()
