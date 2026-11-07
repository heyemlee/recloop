-- name: ListRunningExperiments :many
SELECT exp_id, layer, bucket_from, bucket_to, variants, config_version
FROM experiments
WHERE status = 'running'
  AND (start_at IS NULL OR start_at <= now())
  AND (end_at IS NULL OR end_at > now())
ORDER BY layer, bucket_from;

-- name: RetireOtherModelVersions :exec
UPDATE model_versions SET status = 'retired'
WHERE status = 'active' AND version <> $1;

-- name: UpsertActiveModelVersion :exec
INSERT INTO model_versions (version, status, artifact_uri, manifest, activated_at)
VALUES ($1, 'active', $2, $3, now())
ON CONFLICT (version) DO UPDATE
SET status = 'active',
    artifact_uri = EXCLUDED.artifact_uri,
    manifest = EXCLUDED.manifest,
    activated_at = COALESCE(model_versions.activated_at, EXCLUDED.activated_at);
