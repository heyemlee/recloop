package events

import (
	"context"
	"time"

	"github.com/jackc/pgx/v5/pgtype"

	"github.com/heyemlee/recloop/serving/internal/store"
	"github.com/heyemlee/recloop/serving/internal/store/db"
	"github.com/heyemlee/recloop/serving/internal/userstate"
)

// Writer stores events durably (store.Store).
type Writer interface {
	WriteEvents(ctx context.Context, rows []store.EventRow) (store.WriteResult, error)
}

// StateApplier applies an event to the user's state (userstate.Store).
type StateApplier interface {
	Apply(ctx context.Context, c userstate.Change, now time.Time) (bool, error)
}

// Service ingests validated events.
type Service struct {
	DB    Writer
	State StateApplier
	// OnStock is called with the stock left after new purchases (to update the catalog).
	OnStock func(item, stock int32)
	Now     func() time.Time
}

// Result of one batch.
type Result struct {
	Accepted   int // new events
	Duplicates int // event_id seen before
}

// Ingest writes the events to PostgreSQL (one transaction; new purchases create their order and
// take stock in it), then applies every event to Redis. Both steps are idempotent, so a client
// that gets an error retries the whole batch with the same event_ids.
func (s *Service) Ingest(ctx context.Context, evs []Event) (Result, error) {
	rows := make([]store.EventRow, len(evs))
	for i, ev := range evs {
		rows[i] = store.EventRow{Params: Params(ev)}
		if ev.Type == TypePurchase {
			rows[i].Order = &db.InsertOrderParams{
				UserID: ev.UserID, ItemID: ev.ItemID, EventID: pgtype.UUID{Bytes: ev.ID, Valid: true},
			}
		}
	}
	res, err := s.DB.WriteEvents(ctx, rows)
	if err != nil {
		return Result{}, err
	}
	if s.OnStock != nil {
		for item, left := range res.Stock {
			s.OnStock(item, left)
		}
	}
	var out Result
	for _, ins := range res.Inserted {
		if ins {
			out.Accepted++
		} else {
			out.Duplicates++
		}
	}
	// Redis is updated for duplicates too: if an earlier attempt stored the event but failed
	// before (or while) updating Redis, the retry completes it; the dedup marker prevents a
	// second application.
	now := s.Now()
	for _, ev := range evs {
		kind := KindOf(ev.Type)
		if kind == "" {
			continue
		}
		c := userstate.Change{EventID: ev.ID.String(), UserID: ev.UserID, ItemID: ev.ItemID, Kind: kind, At: ev.Time}
		if _, err := s.State.Apply(ctx, c, now); err != nil {
			return out, err
		}
	}
	return out, nil
}

// KindOf maps an event type to its effect on user state ("" for none).
func KindOf(eventType string) string {
	switch eventType {
	case TypeClick, TypeAddToCart:
		return userstate.KindHistory
	case TypePurchase:
		return userstate.KindPurchase
	case TypeImpression:
		return userstate.KindImpression
	}
	return ""
}

// Params converts an event to its events_raw row.
func Params(ev Event) db.InsertEventParams {
	p := db.InsertEventParams{
		EventID:  pgtype.UUID{Bytes: ev.ID, Valid: true},
		EventDay: pgtype.Date{Time: ev.Day, Valid: true},
		Type:     ev.Type,
		UserID:   ev.UserID,
		Payload:  ev.Payload,
	}
	if ev.Type != TypeServed {
		p.ItemID = pgtype.Int4{Int32: ev.ItemID, Valid: true}
	}
	if ev.RequestID.Valid {
		p.RequestID = pgtype.UUID{Bytes: ev.RequestID.UUID, Valid: true}
	}
	if ev.ImpressionID != "" {
		p.ImpressionID = pgtype.Text{String: ev.ImpressionID, Valid: true}
	}
	if !ev.ClientTS.IsZero() {
		p.ClientTs = pgtype.Timestamptz{Time: ev.ClientTS, Valid: true}
	}
	return p
}
