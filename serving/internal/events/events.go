// Package events validates client events and writes them (design §7): PostgreSQL events_raw
// first (the durable log, deduplicated by event_id), then the user's state in Redis (deduplicated
// by its own marker). M1 writes both synchronously from the api; M2 puts Redpanda in between.
package events

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
)

// Client event types (served is written by the server only).
const (
	TypeServed     = "served"
	TypeImpression = "impression"
	TypeClick      = "click"
	TypeAddToCart  = "add_to_cart"
	TypePurchase   = "purchase"
)

// MaxBatch is the largest batch POST /v1/events accepts.
const MaxBatch = 100

// Accepted event-time window relative to the server clock.
const (
	MaxAge  = 7 * 24 * time.Hour
	MaxSkew = 5 * time.Minute
)

// ClientEvent is the wire format of POST /v1/events.
type ClientEvent struct {
	EventID      string          `json:"event_id"`
	Type         string          `json:"type"`
	UserID       *int64          `json:"user_id"`
	ItemID       *int32          `json:"item_id"`
	RequestID    string          `json:"request_id,omitempty"`
	ImpressionID string          `json:"impression_id,omitempty"`
	ClientTS     string          `json:"client_ts,omitempty"` // RFC 3339
	Props        json.RawMessage `json:"props,omitempty"`     // small object, stored as payload
}

// Event is a validated event.
type Event struct {
	ID           uuid.UUID
	Day          time.Time // UTC day of the UUIDv7 time: the partition key
	Time         time.Time // event time = the UUIDv7 time
	Type         string
	UserID       int64
	ItemID       int32
	RequestID    uuid.NullUUID
	ImpressionID string
	ClientTS     time.Time // zero if not sent
	Payload      []byte    // nil if no props
}

// V7Time returns the Unix-millisecond timestamp stored in the first 48 bits of a UUIDv7.
func V7Time(id uuid.UUID) time.Time {
	var b [8]byte
	copy(b[2:], id[:6])
	return time.UnixMilli(int64(binary.BigEndian.Uint64(b[:]))).UTC()
}

// Day truncates to the UTC day.
func Day(t time.Time) time.Time {
	t = t.UTC()
	return time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, time.UTC)
}

// ParseImpressionID splits "{request_id}:{position}" (position >= 1).
func ParseImpressionID(s string) (uuid.UUID, int, error) {
	rid, pos, ok := strings.Cut(s, ":")
	if !ok {
		return uuid.Nil, 0, errors.New("impression_id must be {request_id}:{position}")
	}
	id, err := uuid.Parse(rid)
	if err != nil {
		return uuid.Nil, 0, fmt.Errorf("impression_id: %w", err)
	}
	p, err := strconv.Atoi(pos)
	if err != nil || p < 1 {
		return uuid.Nil, 0, errors.New("impression_id: position must be a positive integer")
	}
	return id, p, nil
}

// ImpressionID builds "{request_id}:{position}".
func ImpressionID(requestID uuid.UUID, position int) string {
	return requestID.String() + ":" + strconv.Itoa(position)
}

// Validate checks a client event against the rules of design §7.1 / §7.3.
func Validate(ce ClientEvent, now time.Time, knownItem func(int32) bool) (Event, error) {
	id, err := uuid.Parse(ce.EventID)
	if err != nil {
		return Event{}, fmt.Errorf("event_id: %w", err)
	}
	if id.Version() != 7 {
		return Event{}, fmt.Errorf("event_id must be a UUIDv7, got version %d", id.Version())
	}
	at := V7Time(id)
	if at.Before(now.Add(-MaxAge)) || at.After(now.Add(MaxSkew)) {
		return Event{}, fmt.Errorf("event_id time %s is outside the accepted window", at.Format(time.RFC3339))
	}
	switch ce.Type {
	case TypeImpression, TypeClick, TypeAddToCart, TypePurchase:
	case TypeServed:
		return Event{}, errors.New("served events are written by the server")
	default:
		return Event{}, fmt.Errorf("unknown type %q", ce.Type)
	}
	if ce.UserID == nil || *ce.UserID < 0 {
		return Event{}, errors.New("user_id is required and must be >= 0")
	}
	if ce.ItemID == nil || !knownItem(*ce.ItemID) {
		return Event{}, errors.New("item_id is required and must be a catalog item")
	}
	ev := Event{ID: id, Day: Day(at), Time: at, Type: ce.Type, UserID: *ce.UserID, ItemID: *ce.ItemID}

	if ce.ImpressionID != "" {
		rid, _, err := ParseImpressionID(ce.ImpressionID)
		if err != nil {
			return Event{}, err
		}
		ev.ImpressionID = ce.ImpressionID
		ev.RequestID = uuid.NullUUID{UUID: rid, Valid: true}
	} else if ce.Type == TypeImpression {
		return Event{}, errors.New("impression events need impression_id")
	}
	if ce.RequestID != "" {
		rid, err := uuid.Parse(ce.RequestID)
		if err != nil {
			return Event{}, fmt.Errorf("request_id: %w", err)
		}
		if ev.RequestID.Valid && ev.RequestID.UUID != rid {
			return Event{}, errors.New("request_id does not match impression_id")
		}
		ev.RequestID = uuid.NullUUID{UUID: rid, Valid: true}
	}
	if ce.ClientTS != "" {
		ts, err := time.Parse(time.RFC3339Nano, ce.ClientTS)
		if err != nil {
			return Event{}, fmt.Errorf("client_ts: %w", err)
		}
		ev.ClientTS = ts
	}
	if len(ce.Props) > 0 && string(ce.Props) != "null" {
		if len(ce.Props) > 2048 || ce.Props[0] != '{' {
			return Event{}, errors.New("props must be a JSON object of at most 2 KB")
		}
		ev.Payload = ce.Props
	}
	return ev, nil
}
