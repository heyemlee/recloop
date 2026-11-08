// Package aggregator turns the event log into per-user state in Redis (design §7.2): recent
// history, history version, purchases, recent impressions and the frequency-cap counters of
// sponsored slots. It is the only writer of that state in normal operation; the api reads it on
// every request.
//
// Each event is applied by the Lua script of userstate, which sets the dedup marker
// dedup:aggregator:{event_id} and updates the state in one atomic step, so a redelivered event
// (producer retry, client retry, consumer restart before its commit) changes nothing.
package aggregator

import (
	"context"
	"errors"
	"log/slog"
	"sync/atomic"
	"time"

	"github.com/twmb/franz-go/pkg/kgo"

	"github.com/heyemlee/recloop/serving/internal/adtoken"
	"github.com/heyemlee/recloop/serving/internal/events"
	"github.com/heyemlee/recloop/serving/internal/metrics"
	"github.com/heyemlee/recloop/serving/internal/store"
	"github.com/heyemlee/recloop/serving/internal/userstate"
)

// FreqKeys is the rule that maps the impression of a sponsored slot to its frequency-cap
// counter (design §9.2: the cap counts impression events, not serves). The impression carries
// the slot's ad_token; it counts when the token verifies (an expired one too: the impression
// happened) and names this impression, user and item, on the counter of the day the slot was
// served. Anything else counts nothing. nil keys: no caps are counted.
func FreqKeys(keys *adtoken.Keys) func(events.Event) string {
	if keys == nil {
		return nil
	}
	return func(ev events.Event) string {
		at := ev.Received
		if at.IsZero() {
			at = ev.Time
		}
		c, err := keys.Verify(ev.AdToken, at)
		if (err != nil && !errors.Is(err, adtoken.ErrExpired)) || !c.Matches(ev.ImpressionID, ev.UserID, ev.ItemID) {
			metrics.FreqCapTokens.WithLabelValues("rejected").Inc()
			return ""
		}
		metrics.FreqCapTokens.WithLabelValues("counted").Inc()
		return userstate.FreqKey(ev.UserID, c.CampaignID, c.Day())
	}
}

// Group is the aggregator's consumer group (and its dedup group in Redis).
const Group = userstate.DedupGroup

// Applier applies user-state changes (userstate.Store).
type Applier interface {
	ApplyBatch(ctx context.Context, cs []userstate.Change, now time.Time) ([]bool, error)
}

// Aggregator applies events to user state.
type Aggregator struct {
	State Applier
	// FreqKey maps sponsored impressions to frequency-cap counters (FreqKeys); nil: none.
	FreqKey func(events.Event) string
	Now     func() time.Time
	Log     *slog.Logger

	invalidLogged atomic.Int64 // unix seconds of the last "invalid record" log line
}

// Counts of one batch.
type Counts struct {
	Applied, Duplicate, Ignored, Invalid int
}

// Handle is the consumer.Handler: decode, apply in order, count. Malformed records are counted
// and skipped (retrying cannot fix them); only a Redis failure returns an error.
func (a *Aggregator) Handle(ctx context.Context, recs []*kgo.Record) error {
	_, err := a.HandleRecords(ctx, recs)
	return err
}

// HandleRecords is Handle with the counts.
func (a *Aggregator) HandleRecords(ctx context.Context, recs []*kgo.Record) (Counts, error) {
	var c Counts
	evs := make([]events.Event, 0, len(recs))
	for _, r := range recs {
		if events.Header(r, events.HeaderType) == events.TypeServed {
			c.Ignored++ // served events change no user state; skip them without decoding
			continue
		}
		ev, err := events.DecodeRecord(r.Value)
		if err != nil {
			c.Invalid++
			if now := time.Now().Unix(); a.invalidLogged.Swap(now) != now {
				a.Log.Warn("skipping an invalid record", "partition", r.Partition, "offset", r.Offset, "err", err)
			}
			continue
		}
		evs = append(evs, ev)
	}
	changes := events.Changes(evs, a.FreqKey)
	c.Ignored += len(evs) - len(changes)
	if err := a.apply(ctx, changes, &c); err != nil {
		return c, err
	}
	a.record(c)
	return c, nil
}

func (a *Aggregator) apply(ctx context.Context, changes []userstate.Change, c *Counts) error {
	if len(changes) == 0 {
		return nil
	}
	now := a.Now()
	applied, err := a.State.ApplyBatch(ctx, changes, now)
	if err != nil {
		return err
	}
	delay := metrics.EventToState.WithLabelValues(Group)
	for i, ok := range applied {
		if ok {
			c.Applied++
			delay.Observe(now.Sub(changes[i].At).Seconds())
		} else {
			c.Duplicate++
		}
	}
	return nil
}

func (a *Aggregator) record(c Counts) {
	m := metrics.ConsumerRecords
	m.WithLabelValues(Group, "applied").Add(float64(c.Applied))
	m.WithLabelValues(Group, "duplicate").Add(float64(c.Duplicate))
	m.WithLabelValues(Group, "ignored").Add(float64(c.Ignored))
	m.WithLabelValues(Group, "invalid").Add(float64(c.Invalid))
}

// Rebuild replays the client events stored in events_raw into user state (design §7.2: the
// derived state can be rebuilt from the durable log). Run it on a freshly seeded Redis (`make
// seed`), or next to a running aggregator: both use the same dedup markers, so an event is
// applied once either way.
func (a *Aggregator) Rebuild(ctx context.Context, st *store.Store, page int) (Counts, error) {
	var total Counts
	err := st.Replay(ctx, page, func(rows []store.ReplayRow) error {
		evs := make([]events.Event, 0, len(rows))
		for _, r := range rows {
			if !r.ItemID.Valid {
				total.Invalid++
				continue
			}
			at := events.V7Time(r.EventID.Bytes)
			evs = append(evs, events.Event{
				ID: r.EventID.Bytes, Time: at, Day: events.Day(at), Type: r.Type, UserID: r.UserID, ItemID: r.ItemID.Int32,
				ImpressionID: r.ImpressionID.String, AdToken: r.AdToken, Received: r.ServerTs.Time,
			})
		}
		changes := events.Changes(evs, a.FreqKey)
		var c Counts
		c.Ignored = len(evs) - len(changes)
		if err := a.apply(ctx, changes, &c); err != nil {
			return err
		}
		total.Applied += c.Applied
		total.Duplicate += c.Duplicate
		total.Ignored += c.Ignored
		return nil
	})
	return total, err
}
