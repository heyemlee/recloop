-- Orders from purchase events, inside the database (docs/design.md §7.2, §12.1).
--
-- From M2 the api no longer writes PostgreSQL: events go to Redpanda and the log-sink (Redpanda
-- Connect) bulk-inserts them into events_raw with ON CONFLICT DO NOTHING. A purchase event is
-- also an order in this stub shop: the trigger creates it and takes one unit of stock in the
-- same transaction as the event row. A redelivered purchase conflicts on the primary key, no
-- row is inserted and the trigger does not fire, so an event never orders twice; the unique
-- orders.event_id stays as the second guard.

-- +goose Up
-- +goose StatementBegin
CREATE FUNCTION events_raw_purchase_order() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  INSERT INTO orders (user_id, item_id, event_id, created_at)
  VALUES (NEW.user_id, NEW.item_id, NEW.event_id, NEW.server_ts)
  ON CONFLICT (event_id) DO NOTHING;
  IF FOUND THEN
    UPDATE items SET stock = stock - 1 WHERE item_id = NEW.item_id AND stock > 0;
  END IF;
  RETURN NULL;
END
$$;
-- +goose StatementEnd

CREATE TRIGGER events_raw_purchase_order
AFTER INSERT ON events_raw
FOR EACH ROW WHEN (NEW.type = 'purchase')
EXECUTE FUNCTION events_raw_purchase_order();

-- +goose Down
DROP TRIGGER events_raw_purchase_order ON events_raw;
DROP FUNCTION events_raw_purchase_order();
