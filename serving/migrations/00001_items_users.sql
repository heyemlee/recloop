-- Catalog and users (docs/design.md §12.1). Loaded by `make ingest` from data/processed/.

-- +goose Up
CREATE EXTENSION IF NOT EXISTS vector;

-- Product catalog
CREATE TABLE items (
  item_id     INT PRIMARY KEY,              -- same id as MiniOneRec's item index
  asin        TEXT UNIQUE NOT NULL,
  title       TEXT NOT NULL,
  brand       TEXT,
  categories  TEXT[],
  price_cents INT,                          -- NULL = price unknown (missing, a range, or unparseable)
  description TEXT,
  features    TEXT[],
  image_url   TEXT,
  stock       INT NOT NULL DEFAULT 100,
  text_emb    vector(384)                   -- search / agent; filled in M1
);

CREATE TABLE users (
  user_id     BIGINT PRIMARY KEY,           -- dataset users: MiniOneRec's user index
  external_id TEXT UNIQUE,                  -- dataset users: reviewerID
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- +goose Down
DROP TABLE users;
DROP TABLE items;
