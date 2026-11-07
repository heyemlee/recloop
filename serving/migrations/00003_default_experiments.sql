-- The experiment running from M1 on (docs/design.md §8): RRF weights of the recall sources,
-- 50 / 50 over the whole recall layer. Managed through the admin API from M4 on.

-- +goose Up
INSERT INTO experiments (exp_id, layer, status, bucket_from, bucket_to, variants, config_version, start_at)
VALUES (
  'recall-rrf-weights-v1', 'recall', 'running', 0, 1000,
  '[
     {"name": "control", "share": 50,
      "params": {"rrf_weights": {"sasrec": 1.0, "itemcf": 1.0, "pop": 0.3, "text": 1.0}}},
     {"name": "sasrec_heavy", "share": 50,
      "params": {"rrf_weights": {"sasrec": 1.5, "itemcf": 0.75, "pop": 0.3, "text": 1.0}}}
   ]'::jsonb,
  1, now()
)
ON CONFLICT (exp_id) DO NOTHING;

-- +goose Down
DELETE FROM experiments WHERE exp_id = 'recall-rrf-weights-v1';
