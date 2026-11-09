-- Model releases and experiment management (M4, docs/design.md §6, §8).

-- name: ListModelVersions :many
SELECT version, status, artifact_uri, manifest, created_at, activated_at, validated_at, canary_at, retired_at
FROM model_versions
ORDER BY created_at DESC, version;

-- name: GetModelVersionForUpdate :one
SELECT version, status, artifact_uri, manifest, created_at, activated_at, validated_at, canary_at, retired_at
FROM model_versions
WHERE version = $1
FOR UPDATE;

-- name: ActiveModelVersionForUpdate :one
SELECT version, status, artifact_uri, manifest, created_at, activated_at, validated_at, canary_at, retired_at
FROM model_versions
WHERE status = 'active'
FOR UPDATE;

-- name: InsertBuiltModelVersion :execrows
INSERT INTO model_versions (version, status, artifact_uri, manifest)
VALUES ($1, 'built', $2, $3)
ON CONFLICT (version) DO NOTHING;

-- name: SetModelVersionStatus :exec
UPDATE model_versions
SET status       = sqlc.arg(status),
    validated_at = CASE WHEN sqlc.arg(status) = 'validated' THEN now() ELSE validated_at END,
    canary_at    = CASE WHEN sqlc.arg(status) = 'canary' THEN now() ELSE canary_at END,
    activated_at = CASE WHEN sqlc.arg(status) = 'active' THEN now() ELSE activated_at END,
    retired_at   = CASE WHEN sqlc.arg(status) = 'retired' THEN now()
                        WHEN sqlc.arg(status) = 'active' THEN NULL
                        ELSE retired_at END
WHERE version = sqlc.arg(version);

-- name: InsertModelVersionEvent :exec
INSERT INTO model_version_events (version, from_status, to_status, actor, note)
VALUES ($1, $2, $3, $4, $5);

-- name: ListModelVersionEvents :many
SELECT event_id, version, from_status, to_status, at, actor, note
FROM model_version_events
ORDER BY at DESC, event_id DESC
LIMIT $1;

-- name: ListExperiments :many
SELECT exp_id, layer, status, bucket_from, bucket_to, variants, config_version, start_at, end_at
FROM experiments
ORDER BY layer, bucket_from, exp_id;

-- name: NextConfigVersion :one
SELECT (COALESCE(MAX(config_version), 0) + 1)::int AS next FROM experiments;

-- name: UpsertExperiment :exec
INSERT INTO experiments (exp_id, layer, status, bucket_from, bucket_to, variants, config_version, start_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, now())
ON CONFLICT (exp_id) DO UPDATE
SET layer = EXCLUDED.layer,
    status = EXCLUDED.status,
    bucket_from = EXCLUDED.bucket_from,
    bucket_to = EXCLUDED.bucket_to,
    variants = EXCLUDED.variants,
    config_version = EXCLUDED.config_version,
    start_at = CASE WHEN experiments.status = 'running' AND EXCLUDED.status = 'running'
                    THEN experiments.start_at ELSE now() END,
    end_at = NULL;

-- name: StopExperiment :execrows
UPDATE experiments
SET status = 'stopped', end_at = now(), config_version = sqlc.arg(config_version)
WHERE exp_id = sqlc.arg(exp_id) AND status <> 'stopped';
