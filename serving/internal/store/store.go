// Package store is the api's PostgreSQL access: sqlc queries (internal/store/db) plus the few
// statements sqlc cannot express (partition DDL, transactions spanning several queries).
package store

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/heyemlee/recloop/serving/internal/catalog"
	"github.com/heyemlee/recloop/serving/internal/experiment"
	"github.com/heyemlee/recloop/serving/internal/store/db"
)

// Store wraps a connection pool.
type Store struct {
	Pool *pgxpool.Pool
	Q    *db.Queries

	mu         sync.Mutex
	partitions map[string]bool // events_raw partitions known to exist
}

// Open connects and pings.
func Open(ctx context.Context, url string) (*Store, error) {
	cfg, err := pgxpool.ParseConfig(url)
	if err != nil {
		return nil, err
	}
	cfg.MaxConns = 16
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, err
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("postgres: %w", err)
	}
	return &Store{Pool: pool, Q: db.New(pool), partitions: map[string]bool{}}, nil
}

func (s *Store) Close() { s.Pool.Close() }

// ---------------------------------------------------------------- catalog.Source

func (s *Store) Items(ctx context.Context) ([]catalog.Item, []int32, error) {
	rows, err := s.Q.ListItems(ctx)
	if err != nil {
		return nil, nil, err
	}
	items := make([]catalog.Item, len(rows))
	stock := make([]int32, len(rows))
	for i, r := range rows {
		it := catalog.Item{
			ID:          r.ItemID,
			Title:       r.Title,
			Brand:       r.Brand.String,
			Categories:  r.Categories,
			ImageURL:    r.ImageUrl.String,
			Description: r.Description.String,
			Features:    r.Features,
		}
		if r.PriceCents.Valid {
			p := r.PriceCents.Int32
			it.PriceCents = &p
		}
		items[i], stock[i] = it, r.Stock
	}
	return items, stock, nil
}

func (s *Store) Stock(ctx context.Context) (map[int32]int32, error) {
	rows, err := s.Q.ListStock(ctx)
	if err != nil {
		return nil, err
	}
	out := make(map[int32]int32, len(rows))
	for _, r := range rows {
		out[r.ItemID] = r.Stock
	}
	return out, nil
}

// ---------------------------------------------------------------- experiments, models

// RunningExperiments implements experiment.FetchFunc.
func (s *Store) RunningExperiments(ctx context.Context) ([]experiment.Experiment, error) {
	rows, err := s.Q.ListRunningExperiments(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]experiment.Experiment, 0, len(rows))
	for _, r := range rows {
		var variants []experiment.Variant
		if err := json.Unmarshal(r.Variants, &variants); err != nil {
			return nil, fmt.Errorf("experiment %s: variants: %w", r.ExpID, err)
		}
		out = append(out, experiment.Experiment{
			ID: r.ExpID, Layer: r.Layer, From: int(r.BucketFrom), To: int(r.BucketTo),
			Variants: variants, ConfigVersion: int(r.ConfigVersion),
		})
	}
	return out, nil
}

// RegisterActiveModel records the bundle the api serves as the active model version.
func (s *Store) RegisterActiveModel(ctx context.Context, version, uri string, manifest []byte) error {
	return pgx.BeginFunc(ctx, s.Pool, func(tx pgx.Tx) error {
		q := s.Q.WithTx(tx)
		if err := q.RetireOtherModelVersions(ctx, version); err != nil {
			return err
		}
		return q.UpsertActiveModelVersion(ctx, db.UpsertActiveModelVersionParams{
			Version: version, ArtifactUri: uri, Manifest: manifest,
		})
	})
}

// ---------------------------------------------------------------- text similarity

// Neighbor is a text-similar item.
type Neighbor struct {
	ItemID     int32
	Similarity float64
}

// TextNeighbors returns the k items whose text embedding is closest (cosine) to the item's.
func (s *Store) TextNeighbors(ctx context.Context, item int32, k int) ([]Neighbor, error) {
	rows, err := s.Q.TextNeighbors(ctx, db.TextNeighborsParams{ItemID: item, K: int32(k)})
	if err != nil {
		return nil, err
	}
	out := make([]Neighbor, len(rows))
	for i, r := range rows {
		out[i] = Neighbor{ItemID: r.ItemID, Similarity: r.Similarity}
	}
	return out, nil
}

// ---------------------------------------------------------------- events

// PartitionName is the events_raw partition of a UTC day.
func PartitionName(day time.Time) string { return "events_raw_" + day.UTC().Format("20060102") }

// EnsurePartition creates the events_raw partition of a UTC day if it does not exist.
func (s *Store) EnsurePartition(ctx context.Context, day time.Time) error {
	day = time.Date(day.Year(), day.Month(), day.Day(), 0, 0, 0, 0, time.UTC)
	name := PartitionName(day)
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.partitions[name] {
		return nil
	}
	sql := fmt.Sprintf(
		"CREATE TABLE IF NOT EXISTS %s PARTITION OF events_raw FOR VALUES FROM ('%s') TO ('%s')",
		name, day.Format(time.DateOnly), day.AddDate(0, 0, 1).Format(time.DateOnly),
	)
	if _, err := s.Pool.Exec(ctx, sql); err != nil {
		var pgErr *pgconn.PgError
		// Another process created it between our check and the CREATE.
		if !errors.As(err, &pgErr) || (pgErr.Code != "42P07" && pgErr.Code != "23505") {
			return fmt.Errorf("create partition %s: %w", name, err)
		}
	}
	s.partitions[name] = true
	return nil
}

// EventRow is one event to store; Order is set for purchases.
type EventRow struct {
	Params db.InsertEventParams
	Order  *db.InsertOrderParams
}

// WriteResult reports which events were new and the stock left after new purchases.
type WriteResult struct {
	Inserted []bool
	Stock    map[int32]int32
}

// WriteEvents inserts the events (duplicates of an earlier event_id are skipped) and, in the
// same transaction, creates an order and takes one unit of stock for every new purchase.
func (s *Store) WriteEvents(ctx context.Context, rows []EventRow) (WriteResult, error) {
	res := WriteResult{Inserted: make([]bool, len(rows)), Stock: map[int32]int32{}}
	if len(rows) == 0 {
		return res, nil
	}
	for _, r := range rows {
		if err := s.EnsurePartition(ctx, r.Params.EventDay.Time); err != nil {
			return res, err
		}
	}
	params := make([]db.InsertEventParams, len(rows))
	for i, r := range rows {
		params[i] = r.Params
	}
	err := pgx.BeginFunc(ctx, s.Pool, func(tx pgx.Tx) error {
		q := s.Q.WithTx(tx)
		var batchErr error
		br := q.InsertEvent(ctx, params)
		br.QueryRow(func(i int, _ pgtype.UUID, err error) {
			switch {
			case err == nil:
				res.Inserted[i] = true
			case errors.Is(err, pgx.ErrNoRows): // duplicate event_id
			default:
				if batchErr == nil {
					batchErr = err
				}
			}
		})
		if err := br.Close(); err != nil && batchErr == nil {
			batchErr = err
		}
		if batchErr != nil {
			return batchErr
		}
		for i, r := range rows {
			if !res.Inserted[i] || r.Order == nil {
				continue
			}
			if _, err := q.InsertOrder(ctx, *r.Order); err != nil {
				if errors.Is(err, pgx.ErrNoRows) {
					continue // the order of this event exists already
				}
				return err
			}
			left, err := q.DecrementStock(ctx, r.Order.ItemID)
			switch {
			case err == nil:
				res.Stock[r.Order.ItemID] = left
			case errors.Is(err, pgx.ErrNoRows):
				res.Stock[r.Order.ItemID] = 0 // already sold out
			default:
				return err
			}
		}
		return nil
	})
	if err != nil {
		return WriteResult{Inserted: make([]bool, len(rows))}, err
	}
	return res, nil
}

// InsertServed writes server-side events (served); duplicates are ignored.
func (s *Store) InsertServed(ctx context.Context, params []db.InsertEventParams) error {
	rows := make([]EventRow, len(params))
	for i, p := range params {
		rows[i] = EventRow{Params: p}
	}
	_, err := s.WriteEvents(ctx, rows)
	return err
}
