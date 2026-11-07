-- Events, orders, experiments and model versions (docs/design.md §7, §8, §12.1).

-- +goose Up
CREATE EXTENSION IF NOT EXISTS btree_gist;

-- Raw events, partitioned by day. The partition key is the day of the timestamp inside the
-- UUIDv7 event_id (computed by the api while validating the event), never the arrival time:
-- a retried event always lands in the same partition, so the primary key deduplicates it
-- (§7.3). Partitions (events_raw_YYYYMMDD) are created by the api when first needed.
CREATE TABLE events_raw (
  event_id      UUID NOT NULL,
  event_day     DATE NOT NULL,
  type          TEXT NOT NULL
                CHECK (type IN ('served', 'impression', 'click', 'add_to_cart', 'purchase')),
  user_id       BIGINT NOT NULL,
  request_id    UUID,
  impression_id TEXT,                       -- {request_id}:{position}
  item_id       INT,
  payload       JSONB,                      -- served: the page, experiment groups, model_ver
  client_ts     TIMESTAMPTZ,
  server_ts     TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (event_id, event_day),
  -- Attribution (§8): every served record names its request, its experiment assignment and the
  -- model version that produced it. COALESCE(..., false): a CHECK whose expression is NULL
  -- (missing key) would otherwise pass.
  CONSTRAINT served_is_attributable CHECK (
    type <> 'served' OR (
      request_id IS NOT NULL
      AND COALESCE(jsonb_typeof(payload -> 'model_ver') = 'string'
                   AND payload ->> 'model_ver' <> '', false)
      AND COALESCE(jsonb_typeof(payload -> 'exp' -> 'recall') = 'object', false)
      AND COALESCE(jsonb_typeof(payload -> 'config_version') = 'number', false)
    )
  )
) PARTITION BY RANGE (event_day);

CREATE INDEX events_raw_request ON events_raw (request_id);
CREATE INDEX events_raw_user ON events_raw (user_id, event_day);

-- Purchases are idempotent too: one order per purchase event.
CREATE TABLE orders (
  order_id   BIGSERIAL PRIMARY KEY,
  user_id    BIGINT NOT NULL,
  item_id    INT NOT NULL REFERENCES items,
  event_id   UUID UNIQUE NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Layered experiments (§8): bucket = xxhash64(layer salt + ":" + unit id) % 1000. Running
-- experiments of one layer may not share a bucket.
CREATE TABLE experiments (
  exp_id         TEXT PRIMARY KEY,
  layer          TEXT NOT NULL CHECK (layer IN ('recall', 'rank', 'ads', 'agent')),
  status         TEXT NOT NULL CHECK (status IN ('draft', 'running', 'stopped')),
  bucket_from    INT NOT NULL,
  bucket_to      INT NOT NULL,              -- [bucket_from, bucket_to) within [0, 1000)
  variants       JSONB NOT NULL,            -- [{"name": "control", "share": 50, "params": {...}}]
  config_version INT NOT NULL,
  start_at       TIMESTAMPTZ,
  end_at         TIMESTAMPTZ,
  CHECK (0 <= bucket_from AND bucket_from < bucket_to AND bucket_to <= 1000),
  EXCLUDE USING gist (layer WITH =, int4range(bucket_from, bucket_to) WITH &&)
    WHERE (status = 'running')
);

CREATE TABLE model_versions (
  version      TEXT PRIMARY KEY,            -- e.g. 2026-11-06-sasrec64-1a2b3c4d
  status       TEXT NOT NULL CHECK (status IN ('built', 'validated', 'canary', 'active', 'retired')),
  artifact_uri TEXT NOT NULL,
  manifest     JSONB NOT NULL,              -- data snapshot, offline metrics, checksums
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  activated_at TIMESTAMPTZ
);

-- +goose Down
DROP TABLE model_versions;
DROP TABLE experiments;
DROP TABLE orders;
DROP TABLE events_raw;
