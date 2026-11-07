-- Keep items.text_emb inline (docs/design.md §4.1, item surface).
--
-- Item rows are wide (description, features), so with the default EXTENDED storage PostgreSQL
-- moves the 1.5 KB vector out of line into TOAST, and the exact KNN of the item page read
-- ~12 buffers per row (~41K for 3.5K items, 15-35 ms). PLAIN keeps the vector in the row (the
-- long text columns are toasted instead). Existing rows change when text_emb is next written
-- (`make ingest` rewrites it).

-- +goose Up
ALTER TABLE items ALTER COLUMN text_emb SET STORAGE PLAIN;

-- +goose Down
ALTER TABLE items ALTER COLUMN text_emb SET STORAGE EXTENDED;
