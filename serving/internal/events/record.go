package events

import (
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"time"

	"github.com/google/uuid"
)

// RecordVersion is the version of the record format on the `events` topic.
const RecordVersion = 1

// Header names on every record: consumers that only care about some types (billing: clicks;
// the aggregator: not served) decide from the header without decoding the value.
const (
	HeaderType    = "type"
	HeaderVersion = "v"
)

// Record is one event on the `events` topic (design §7.2, §12.3), JSON-encoded, key = user_id
// so one user's events stay in order on one partition. The fields map one to one onto the
// events_raw columns (the log-sink only renames: ad_token goes into payload); event_day is
// computed by the api from the UUIDv7 event_id, never from arrival time (§7.3).
type Record struct {
	V            int             `json:"v"`
	EventID      string          `json:"event_id"`
	EventDay     string          `json:"event_day"` // YYYY-MM-DD, UTC
	EventTS      string          `json:"event_ts"`  // the UUIDv7 time, RFC 3339 with ms
	Type         string          `json:"type"`
	UserID       int64           `json:"user_id"`
	ItemID       *int32          `json:"item_id,omitempty"`
	RequestID    string          `json:"request_id,omitempty"`
	ImpressionID string          `json:"impression_id,omitempty"`
	AdToken      string          `json:"ad_token,omitempty"`
	ClientTS     string          `json:"client_ts,omitempty"`
	ServerTS     string          `json:"server_ts"` // accepted by the api
	Payload      json.RawMessage `json:"payload,omitempty"`
}

const tsLayout = "2006-01-02T15:04:05.000Z07:00"

// EncodeRecord renders an event as the topic's record value.
func EncodeRecord(ev Event) ([]byte, error) {
	r := Record{
		V: RecordVersion, EventID: ev.ID.String(), EventDay: ev.Day.Format(time.DateOnly),
		EventTS: ev.Time.UTC().Format(tsLayout), Type: ev.Type, UserID: ev.UserID,
		ImpressionID: ev.ImpressionID, AdToken: ev.AdToken,
	}
	if ev.Type != TypeServed {
		id := ev.ItemID
		r.ItemID = &id
	}
	if ev.RequestID.Valid {
		r.RequestID = ev.RequestID.UUID.String()
	}
	if !ev.ClientTS.IsZero() {
		r.ClientTS = ev.ClientTS.UTC().Format(time.RFC3339Nano)
	}
	received := ev.Received
	if received.IsZero() {
		received = time.Now()
	}
	r.ServerTS = received.UTC().Format(time.RFC3339Nano)
	if len(ev.Payload) > 0 {
		r.Payload = ev.Payload
	}
	return json.Marshal(r)
}

// DecodeRecord parses a record value back into an event. It checks what consumers rely on:
// a UUIDv7 event_id whose day is event_day, a known type, a user, and an item for client events.
func DecodeRecord(b []byte) (Event, error) {
	var r Record
	if err := json.Unmarshal(b, &r); err != nil {
		return Event{}, fmt.Errorf("record: %w", err)
	}
	if r.V != RecordVersion {
		return Event{}, fmt.Errorf("record: version %d, this code reads %d", r.V, RecordVersion)
	}
	id, err := uuid.Parse(r.EventID)
	if err != nil || id.Version() != 7 {
		return Event{}, fmt.Errorf("record: event_id %q is not a UUIDv7", r.EventID)
	}
	at := V7Time(id)
	ev := Event{ID: id, Day: Day(at), Time: at, Type: r.Type, UserID: r.UserID,
		ImpressionID: r.ImpressionID, AdToken: r.AdToken, Payload: r.Payload}
	if r.EventDay != ev.Day.Format(time.DateOnly) {
		return Event{}, fmt.Errorf("record: event_day %s is not the day of event_id (%s)", r.EventDay, ev.Day.Format(time.DateOnly))
	}
	switch r.Type {
	case TypeServed:
	case TypeImpression, TypeClick, TypeAddToCart, TypePurchase:
		if r.ItemID == nil {
			return Event{}, errors.New("record: client event without item_id")
		}
		ev.ItemID = *r.ItemID
	default:
		return Event{}, fmt.Errorf("record: unknown type %q", r.Type)
	}
	if r.UserID < 0 {
		return Event{}, errors.New("record: negative user_id")
	}
	if r.RequestID != "" {
		rid, err := uuid.Parse(r.RequestID)
		if err != nil {
			return Event{}, fmt.Errorf("record: request_id: %w", err)
		}
		ev.RequestID = uuid.NullUUID{UUID: rid, Valid: true}
	}
	if r.ClientTS != "" {
		if ev.ClientTS, err = time.Parse(time.RFC3339Nano, r.ClientTS); err != nil {
			return Event{}, fmt.Errorf("record: client_ts: %w", err)
		}
	}
	if r.ServerTS != "" {
		if ev.Received, err = time.Parse(time.RFC3339Nano, r.ServerTS); err != nil {
			return Event{}, fmt.Errorf("record: server_ts: %w", err)
		}
	}
	return ev, nil
}

// Key is the record key of a user's events.
func Key(userID int64) []byte { return strconv.AppendInt(nil, userID, 10) }

// PayloadWithAdToken adds "ad_token" to a JSON-object payload (events_raw keeps the token of a
// sponsored click in its payload; the log-sink does the same mapping).
func PayloadWithAdToken(payload []byte, token string) []byte {
	if token == "" {
		return payload
	}
	m := map[string]json.RawMessage{}
	if len(payload) > 0 {
		if err := json.Unmarshal(payload, &m); err != nil {
			m = map[string]json.RawMessage{}
		}
	}
	t, _ := json.Marshal(token)
	m["ad_token"] = t
	out, _ := json.Marshal(m)
	return out
}
