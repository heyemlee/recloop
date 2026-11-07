-- Served-log attribution checks (docs/design.md §8). Run with:
--   docker exec -i recloop-postgres-1 psql -U recloop -d recloop < analysis/served_attribution.sql
-- The served_is_attributable CHECK constraint on events_raw already refuses a served row
-- without model_ver / exp.recall / config_version; these queries show it on the data.

-- 1. Every served row carries a model version and the assignment of all four layers.
SELECT count(*)                                                      AS served_rows,
       count(*) FILTER (WHERE payload ->> 'model_ver' <> '')         AS with_model_ver,
       count(*) FILTER (WHERE payload -> 'exp' -> 'recall' ->> 'variant' IS NOT NULL
                          AND payload -> 'exp' -> 'rank'   ->> 'variant' IS NOT NULL
                          AND payload -> 'exp' -> 'ads'    ->> 'variant' IS NOT NULL
                          AND payload -> 'exp' -> 'agent'  ->> 'variant' IS NOT NULL) AS with_all_layers,
       count(*) FILTER (WHERE (payload ->> 'config_version') IS NOT NULL) AS with_config_version,
       count(DISTINCT request_id)                                    AS requests
FROM events_raw
WHERE type = 'served';

-- 2. Served pages by model version and recall-layer variant.
SELECT payload ->> 'model_ver'                         AS model_ver,
       payload -> 'exp' -> 'recall' ->> 'exp_id'       AS exp_id,
       payload -> 'exp' -> 'recall' ->> 'variant'      AS variant,
       count(*)                                        AS served_rows,
       count(DISTINCT user_id)                         AS users
FROM events_raw
WHERE type = 'served'
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3;

-- 3. Attribution of clicks: click -> impression_id -> served(request_id) -> group, model, source.
SELECT c.event_id                                      AS click,
       c.user_id,
       c.item_id,
       c.impression_id,
       s.payload ->> 'model_ver'                       AS model_ver,
       s.payload -> 'exp' -> 'recall' ->> 'variant'    AS variant,
       i ->> 'source'                                  AS source
FROM events_raw AS c
JOIN events_raw AS s
  ON s.type = 'served' AND s.request_id = c.request_id
CROSS JOIN LATERAL jsonb_array_elements(s.payload -> 'items') AS i
WHERE c.type = 'click'
  AND (i ->> 'pos')::int = split_part(c.impression_id, ':', 2)::int
ORDER BY c.event_id DESC
LIMIT 10;
