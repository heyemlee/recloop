package billing

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/twmb/franz-go/pkg/kgo"

	"github.com/heyemlee/recloop/serving/internal/events"
	"github.com/heyemlee/recloop/serving/internal/metrics"
)

// Group is billing's consumer group on the event log (design §7.2): it reads only the clicks
// that carry an ad_token.
const Group = DedupGroup

// ClickTimeout bounds one Handle call. A click that runs out of it (PostgreSQL hanging) fails
// its batch, which is retried; the transaction was rolled back.
const ClickTimeout = 30 * time.Second

// Consumer turns the sponsored clicks of the event log into charges. It is a consumer.Handler:
// the loop commits a batch's offsets only after Handle returned nil, that is after every click
// of the batch was charged, found to be a duplicate, or recorded as not billable. A crash
// before that redelivers the batch, and Service.Handle makes the redelivery harmless.
type Consumer struct {
	Service *Service
	// Workers handle the clicks of one batch concurrently (1 = in order). The ledger does not
	// depend on the order: every click on an impression is billed against the budget row of
	// the day it was served, and the row lock serialises one campaign-day's charges.
	Workers int
	Log     *slog.Logger

	logged atomic.Int64 // unix second of the last "skipping" log line
}

// Counts of one batch.
type Counts struct {
	Clicks, Ignored, Invalid, UnknownCampaign int
	Outcomes                                  map[Outcome]int
}

// Handle is the consumer.Handler.
func (c *Consumer) Handle(ctx context.Context, recs []*kgo.Record) error {
	_, err := c.HandleRecords(ctx, recs)
	return err
}

// HandleRecords is Handle with the counts. It returns an error, and the batch must be retried,
// when a click could not be processed (the database failed); malformed records and tokens of
// campaigns this database does not know are logged, counted and skipped (retrying cannot fix
// them).
func (c *Consumer) HandleRecords(ctx context.Context, recs []*kgo.Record) (Counts, error) {
	n := Counts{Outcomes: map[Outcome]int{}}
	var clicks []Click
	for _, r := range recs {
		if events.Header(r, events.HeaderType) != events.TypeClick {
			n.Ignored++ // everything but clicks is skipped without decoding
			continue
		}
		ev, err := events.DecodeRecord(r.Value)
		if err != nil {
			n.Invalid++
			c.skip("skipping an invalid record", "partition", r.Partition, "offset", r.Offset, "err", err)
			continue
		}
		if ev.AdToken == "" {
			n.Ignored++ // an organic click
			continue
		}
		received := ev.Received // the api's receive time (server_ts)
		if received.IsZero() {
			received = r.Timestamp
		}
		clicks = append(clicks, Click{
			EventID: ev.ID, ImpressionID: ev.ImpressionID, UserID: ev.UserID, ItemID: ev.ItemID,
			AdToken: ev.AdToken, ReceivedAt: received,
		})
	}
	n.Clicks = len(clicks)
	results, errs := c.handleAll(ctx, clicks)
	var failed error
	for i, err := range errs {
		switch {
		case err == nil:
			res := results[i]
			n.Outcomes[res.Outcome]++
			metrics.BillingClicks.WithLabelValues(string(res.Outcome)).Inc()
			if res.Outcome == Charged {
				metrics.BillingCents.WithLabelValues("billed").Add(float64(res.BilledCents))
				metrics.BillingCents.WithLabelValues("unbilled").Add(float64(res.PriceCents - res.BilledCents))
			}
		case errors.Is(err, ErrUnknownCampaign):
			n.UnknownCampaign++
			metrics.BillingClicks.WithLabelValues("unknown_campaign").Inc()
			c.skip("skipping a click for an unknown campaign", "event_id", clicks[i].EventID, "err", err)
		case failed == nil:
			failed = err
		}
	}
	m := metrics.ConsumerRecords
	m.WithLabelValues(Group, "applied").Add(float64(n.Clicks))
	m.WithLabelValues(Group, "ignored").Add(float64(n.Ignored))
	m.WithLabelValues(Group, "invalid").Add(float64(n.Invalid))
	return n, failed
}

// handleAll runs Handle on every click, Workers at a time.
func (c *Consumer) handleAll(ctx context.Context, clicks []Click) ([]Result, []error) {
	results := make([]Result, len(clicks))
	errs := make([]error, len(clicks))
	workers := max(c.Workers, 1)
	var wg sync.WaitGroup
	sem := make(chan struct{}, workers)
	for i := range clicks {
		sem <- struct{}{}
		wg.Add(1)
		go func(i int) {
			defer func() { <-sem; wg.Done() }()
			cctx, cancel := context.WithTimeout(ctx, ClickTimeout)
			defer cancel()
			t0 := time.Now()
			results[i], errs[i] = c.Service.Handle(cctx, clicks[i])
			metrics.BillingHandle.Observe(time.Since(t0).Seconds())
		}(i)
	}
	wg.Wait()
	return results, errs
}

func (c *Consumer) skip(msg string, args ...any) {
	if now := time.Now().Unix(); c.logged.Swap(now) != now {
		c.Log.Warn(msg, args...)
	}
}
