-- One statement per event, sent as a single pgx batch. A duplicate event_id (same event_day,
-- since the day comes from the event_id itself) returns no row.
-- name: InsertEvent :batchone
INSERT INTO events_raw (
  event_id, event_day, type, user_id, request_id, impression_id, item_id, payload, client_ts
) VALUES (
  $1, $2, $3, $4, $5, $6, $7, $8, $9
)
ON CONFLICT (event_id, event_day) DO NOTHING
RETURNING event_id;

-- name: InsertOrder :one
INSERT INTO orders (user_id, item_id, event_id)
VALUES ($1, $2, $3)
ON CONFLICT (event_id) DO NOTHING
RETURNING order_id;

-- name: DecrementStock :one
UPDATE items SET stock = stock - 1
WHERE item_id = $1 AND stock > 0
RETURNING stock;
