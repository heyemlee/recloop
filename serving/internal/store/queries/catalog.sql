-- name: ListItems :many
SELECT item_id, title, brand, categories, price_cents, image_url, stock, description, features
FROM items
ORDER BY item_id;

-- name: ListStock :many
SELECT item_id, stock FROM items;

-- Text-similar items for the item surface (pgvector, exact cosine KNN over 3.5K rows).
-- name: TextNeighbors :many
SELECT i.item_id, (1 - (i.text_emb <=> s.text_emb))::float8 AS similarity
FROM items AS i, items AS s
WHERE s.item_id = sqlc.arg(item_id)
  AND i.item_id <> s.item_id
  AND i.text_emb IS NOT NULL
  AND s.text_emb IS NOT NULL
ORDER BY i.text_emb <=> s.text_emb
LIMIT sqlc.arg(k);
