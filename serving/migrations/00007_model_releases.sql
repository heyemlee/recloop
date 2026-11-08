-- Model releases (docs/design.md §6, M4): a bundle version moves built -> validated -> canary ->
-- active -> retired through the api's release endpoints. The timestamps say when each step
-- happened (a rollback needs to know when the previous version was retired), at most one
-- version is active, and every transition is a row of model_version_events: the record of each
-- canary, activation and rollback.

-- +goose Up
ALTER TABLE model_versions
  ADD COLUMN validated_at TIMESTAMPTZ,
  ADD COLUMN canary_at    TIMESTAMPTZ,
  ADD COLUMN retired_at   TIMESTAMPTZ;

CREATE UNIQUE INDEX model_versions_one_active ON model_versions ((status)) WHERE status = 'active';

CREATE TABLE model_version_events (
  event_id    BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  version     TEXT NOT NULL REFERENCES model_versions,
  from_status TEXT,                          -- NULL when the version was registered
  to_status   TEXT NOT NULL,
  at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  actor       TEXT NOT NULL,                 -- who asked: admin-ui, release-drill, api-startup
  note        TEXT NOT NULL DEFAULT ''
);
CREATE INDEX model_version_events_version ON model_version_events (version, at);

-- +goose Down
DROP TABLE model_version_events;
DROP INDEX model_versions_one_active;
ALTER TABLE model_versions DROP COLUMN validated_at, DROP COLUMN canary_at, DROP COLUMN retired_at;
