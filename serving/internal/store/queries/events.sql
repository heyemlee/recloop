-- One statement per event, sent as a single pgx batch. A duplicate event_id (same event_day,
-- since the day comes from the event_id itself) returns no row. server_ts is when the api
-- accepted the event (now() if unknown). A new purchase also creates its order and takes one
-- unit of stock, in the same statement (trigger, migration 00005).
-- name: InsertEvent :batchone
INSERT INTO events_raw (
  event_id, event_day, type, user_id, request_id, impression_id, item_id, payload, client_ts,
  server_ts
) VALUES (
  $1, $2, $3, $4, $5, $6, $7, $8, $9, COALESCE(sqlc.narg(server_ts)::timestamptz, now())
)
ON CONFLICT (event_id, event_day) DO NOTHING
RETURNING event_id;

-- name: StockOf :many
SELECT item_id, stock FROM items WHERE item_id = ANY(sqlc.arg(item_ids)::int[]);

-- Client events of one day in event_id (= UUIDv7 time) order, for rebuilding Redis. The
-- ad_token of a sponsored impression (kept in the payload) rebuilds its frequency-cap counter.
-- name: ReplayEvents :many
SELECT event_id, type, user_id, item_id, impression_id, server_ts,
       COALESCE(payload ->> 'ad_token', '')::text AS ad_token
FROM events_raw
WHERE event_day = sqlc.arg(event_day)
  AND type IN ('impression', 'click', 'add_to_cart', 'purchase')
  AND event_id > sqlc.arg(after)
ORDER BY event_id
LIMIT sqlc.arg(n);
