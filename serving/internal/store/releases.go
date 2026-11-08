package store

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"

	"github.com/heyemlee/recloop/serving/internal/store/db"
)

// ---------------------------------------------------------------- model releases (design §6)

// ModelVersion is one row of model_versions.
type ModelVersion struct {
	Version     string          `json:"version"`
	Status      string          `json:"status"`
	ArtifactURI string          `json:"artifact_uri"`
	Manifest    json.RawMessage `json:"manifest,omitempty"`
	CreatedAt   time.Time       `json:"created_at"`
	ValidatedAt *time.Time      `json:"validated_at,omitempty"`
	CanaryAt    *time.Time      `json:"canary_at,omitempty"`
	ActivatedAt *time.Time      `json:"activated_at,omitempty"`
	RetiredAt   *time.Time      `json:"retired_at,omitempty"`
}

// ModelEvent is one status transition (model_version_events): the release record.
type ModelEvent struct {
	ID      int64     `json:"id"`
	Version string    `json:"version"`
	From    string    `json:"from,omitempty"`
	To      string    `json:"to"`
	At      time.Time `json:"at"`
	Actor   string    `json:"actor"`
	Note    string    `json:"note,omitempty"`
}

// ErrTransition: the version is not in a state the requested step can start from.
var ErrTransition = errors.New("model version is not in a state this step starts from")

// ErrNoVersion: no such model version.
var ErrNoVersion = errors.New("no such model version")

func tsPtr(t pgtype.Timestamptz) *time.Time {
	if !t.Valid {
		return nil
	}
	v := t.Time
	return &v
}

func fromRow(r db.ModelVersion) ModelVersion {
	return ModelVersion{
		Version: r.Version, Status: r.Status, ArtifactURI: r.ArtifactUri, Manifest: r.Manifest,
		CreatedAt: r.CreatedAt.Time, ValidatedAt: tsPtr(r.ValidatedAt), CanaryAt: tsPtr(r.CanaryAt),
		ActivatedAt: tsPtr(r.ActivatedAt), RetiredAt: tsPtr(r.RetiredAt),
	}
}

// ModelVersions lists every version, newest first.
func (s *Store) ModelVersions(ctx context.Context) ([]ModelVersion, error) {
	rows, err := s.Q.ListModelVersions(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]ModelVersion, len(rows))
	for i, r := range rows {
		out[i] = fromRow(r)
	}
	return out, nil
}

// ModelEvents lists the latest status transitions, newest first.
func (s *Store) ModelEvents(ctx context.Context, limit int32) ([]ModelEvent, error) {
	rows, err := s.Q.ListModelVersionEvents(ctx, limit)
	if err != nil {
		return nil, err
	}
	out := make([]ModelEvent, len(rows))
	for i, r := range rows {
		out[i] = ModelEvent{ID: r.EventID, Version: r.Version, From: r.FromStatus.String, To: r.ToStatus, At: r.At.Time, Actor: r.Actor, Note: r.Note}
	}
	return out, nil
}

// RegisterBuilt records a bundle as built; false if the version was already registered.
func (s *Store) RegisterBuilt(ctx context.Context, version, uri string, manifest []byte, actor string) (bool, error) {
	var added bool
	err := pgx.BeginFunc(ctx, s.Pool, func(tx pgx.Tx) error {
		q := s.Q.WithTx(tx)
		n, err := q.InsertBuiltModelVersion(ctx, db.InsertBuiltModelVersionParams{Version: version, ArtifactUri: uri, Manifest: manifest})
		if err != nil || n == 0 {
			return err
		}
		added = true
		return q.InsertModelVersionEvent(ctx, db.InsertModelVersionEventParams{Version: version, ToStatus: "built", Actor: actor, Note: "registered: manifest found (artifact_uri)"})
	})
	return added, err
}

// Transition moves a version from one of `from` to `to` and records it. With `to` = active,
// the version that was active becomes retired in the same transaction (one active version at a
// time, the switch is atomic); the returned string is that previous version ("" if none).
func (s *Store) Transition(ctx context.Context, version string, from []string, to, actor, note string) (string, error) {
	var previous string
	err := pgx.BeginFunc(ctx, s.Pool, func(tx pgx.Tx) error {
		q := s.Q.WithTx(tx)
		if to == "active" {
			cur, err := q.ActiveModelVersionForUpdate(ctx)
			switch {
			case errors.Is(err, pgx.ErrNoRows):
			case err != nil:
				return err
			case cur.Version != version:
				previous = cur.Version
				if err := q.SetModelVersionStatus(ctx, db.SetModelVersionStatusParams{Status: "retired", Version: cur.Version}); err != nil {
					return err
				}
				if err := q.InsertModelVersionEvent(ctx, db.InsertModelVersionEventParams{
					Version: cur.Version, FromStatus: pgtype.Text{String: "active", Valid: true}, ToStatus: "retired",
					Actor: actor, Note: "replaced by " + version,
				}); err != nil {
					return err
				}
			}
		}
		row, err := q.GetModelVersionForUpdate(ctx, version)
		if errors.Is(err, pgx.ErrNoRows) {
			return ErrNoVersion
		}
		if err != nil {
			return err
		}
		if !slices.Contains(from, row.Status) {
			return fmt.Errorf("%w: %s is %s, %s needs %v", ErrTransition, version, row.Status, to, from)
		}
		if err := q.SetModelVersionStatus(ctx, db.SetModelVersionStatusParams{Status: to, Version: version}); err != nil {
			return err
		}
		return q.InsertModelVersionEvent(ctx, db.InsertModelVersionEventParams{
			Version: version, FromStatus: pgtype.Text{String: row.Status, Valid: true}, ToStatus: to, Actor: actor, Note: note,
		})
	})
	return previous, err
}

// ---------------------------------------------------------------- experiments (design §8)

// ExperimentRow is one row of the experiments table.
type ExperimentRow struct {
	ID            string          `json:"exp_id"`
	Layer         string          `json:"layer"`
	Status        string          `json:"status"`
	BucketFrom    int32           `json:"bucket_from"`
	BucketTo      int32           `json:"bucket_to"`
	Variants      json.RawMessage `json:"variants"`
	ConfigVersion int32           `json:"config_version"`
	StartAt       *time.Time      `json:"start_at,omitempty"`
	EndAt         *time.Time      `json:"end_at,omitempty"`
}

// Experiments lists every experiment, running or not.
func (s *Store) Experiments(ctx context.Context) ([]ExperimentRow, error) {
	rows, err := s.Q.ListExperiments(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]ExperimentRow, len(rows))
	for i, r := range rows {
		out[i] = ExperimentRow{
			ID: r.ExpID, Layer: r.Layer, Status: r.Status, BucketFrom: r.BucketFrom, BucketTo: r.BucketTo,
			Variants: r.Variants, ConfigVersion: r.ConfigVersion, StartAt: tsPtr(r.StartAt), EndAt: tsPtr(r.EndAt),
		}
	}
	return out, nil
}

// PutExperiment creates or replaces an experiment (status running or stopped) under the next
// config_version; the api instances pick it up at their next reload (every 10 s). PostgreSQL's
// exclusion constraint refuses a running experiment that overlaps another of its layer.
func (s *Store) PutExperiment(ctx context.Context, e ExperimentRow) (int32, error) {
	var version int32
	err := pgx.BeginFunc(ctx, s.Pool, func(tx pgx.Tx) error {
		q := s.Q.WithTx(tx)
		if _, err := tx.Exec(ctx, "LOCK TABLE experiments IN SHARE ROW EXCLUSIVE MODE"); err != nil {
			return err
		}
		v, err := q.NextConfigVersion(ctx)
		if err != nil {
			return err
		}
		version = v
		return q.UpsertExperiment(ctx, db.UpsertExperimentParams{
			ExpID: e.ID, Layer: e.Layer, Status: e.Status, BucketFrom: e.BucketFrom, BucketTo: e.BucketTo,
			Variants: e.Variants, ConfigVersion: v,
		})
	})
	return version, err
}

// StopExperiment stops a running experiment; false if it was not running.
func (s *Store) StopExperiment(ctx context.Context, id string) (bool, error) {
	var stopped bool
	err := pgx.BeginFunc(ctx, s.Pool, func(tx pgx.Tx) error {
		q := s.Q.WithTx(tx)
		if _, err := tx.Exec(ctx, "LOCK TABLE experiments IN SHARE ROW EXCLUSIVE MODE"); err != nil {
			return err
		}
		v, err := q.NextConfigVersion(ctx)
		if err != nil {
			return err
		}
		n, err := q.StopExperiment(ctx, db.StopExperimentParams{ConfigVersion: v, ExpID: id})
		stopped = n > 0
		return err
	})
	return stopped, err
}
