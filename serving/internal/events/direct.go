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

// StateApplier applies events to user state (userstate.Store).
type StateApplier interface {
	ApplyBatch(ctx context.Context, cs []userstate.Change, now time.Time) ([]bool, error)
}

// Direct is M1's synchronous path, without an event log: PostgreSQL first (events_raw; the
// purchase trigger creates the order and takes stock in the same transaction), then Redis.
// The api uses it when EVENTS_MODE=direct (no Redpanda) and the tests use it to see an event's
// effect immediately. With Redpanda the same two writes are done by the log-sink and the
// aggregator.
type Direct struct {
	DB    Writer
	State StateApplier
	// OnStock is called with the stock left after new purchases (to update the catalog).
	OnStock func(item, stock int32)
	Now     func() time.Time
	// FreqKey maps the impression of a sponsored slot to its frequency-cap counter (the
	// aggregator's rule, aggregator.FreqKeys); nil counts no frequency caps.
	FreqKey func(Event) string
}

// Publish writes the events to PostgreSQL in one transaction, then applies every event to
// Redis. Both steps are idempotent, so a client that gets an error retries the whole batch with
// the same event_ids.
func (s *Direct) Publish(ctx context.Context, evs []Event) (Result, error) {
	rows := make([]store.EventRow, len(evs))
	for i, ev := range evs {
		rows[i] = store.EventRow{Params: Params(ev)}
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
	if _, err := s.State.ApplyBatch(ctx, Changes(evs, s.FreqKey), s.Now()); err != nil {
		return out, err
	}
	return out, nil
}

// PGServed writes served events straight into events_raw (the synchronous path).
type PGServed struct{ DB Writer }

func (p PGServed) WriteServed(ctx context.Context, evs []Event) error {
	rows := make([]store.EventRow, len(evs))
	for i, ev := range evs {
		rows[i] = store.EventRow{Params: Params(ev)}
	}
	_, err := p.DB.WriteEvents(ctx, rows)
	return err
}

// Changes are the user-state effects of a batch of events, in order (served events and other
// types without an effect are left out). freqKey, if set, names the frequency-cap counter a
// sponsored impression adds to ("" for none).
func Changes(evs []Event, freqKey func(Event) string) []userstate.Change {
	out := make([]userstate.Change, 0, len(evs))
	for _, ev := range evs {
		kind := KindOf(ev.Type)
		if kind == "" {
			continue
		}
		c := userstate.Change{EventID: ev.ID.String(), UserID: ev.UserID, ItemID: ev.ItemID, Kind: kind, At: ev.Time}
		if freqKey != nil && kind == userstate.KindImpression && ev.AdToken != "" {
			c.FreqKey = freqKey(ev)
		}
		out = append(out, c)
	}
	return out
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

// Params converts an event to its events_raw row. The ad_token of a sponsored click is kept in
// the payload (events_raw has no column for it), as the log-sink does.
func Params(ev Event) db.InsertEventParams {
	p := db.InsertEventParams{
		EventID:  pgtype.UUID{Bytes: ev.ID, Valid: true},
		EventDay: pgtype.Date{Time: ev.Day, Valid: true},
		Type:     ev.Type,
		UserID:   ev.UserID,
		Payload:  PayloadWithAdToken(ev.Payload, ev.AdToken),
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
	if !ev.Received.IsZero() {
		p.ServerTs = pgtype.Timestamptz{Time: ev.Received, Valid: true}
	}
	return p
}
