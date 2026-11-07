// Package consumer is the consume loop shared by the event-log consumers (design §7.2, §7.3):
// poll a batch, apply it, and only then commit its offsets. A crash between the side effects
// and the commit redelivers the batch, which the handlers make harmless (each one deduplicates
// by event_id in the same atomic step as its side effect): at-least-once delivery plus
// idempotent handling.
package consumer

import (
	"context"
	"errors"
	"log/slog"
	"strings"
	"time"

	"github.com/twmb/franz-go/pkg/kadm"
	"github.com/twmb/franz-go/pkg/kgo"

	"github.com/heyemlee/recloop/serving/internal/metrics"
)

// Handler applies a batch of records (in partition order within each partition). It returns an
// error only for failures worth retrying (the store is down); records it cannot use are its own
// business to count and skip.
type Handler func(ctx context.Context, recs []*kgo.Record) error

// Options of a group consumer client.
func Options(brokers []string, group, topic string) []kgo.Opt {
	return []kgo.Opt{
		kgo.SeedBrokers(brokers...),
		kgo.ConsumerGroup(group),
		kgo.ConsumeTopics(topic),
		// A new group starts from the oldest retained event: that is what makes "replay the
		// log into an empty Redis" work (design §7.2).
		kgo.ConsumeResetOffset(kgo.NewOffset().AtStart()),
		kgo.DisableAutoCommit(),
		// Rebalances wait until the current batch is applied and committed.
		kgo.BlockRebalanceOnPoll(),
		kgo.FetchMaxWait(250 * time.Millisecond),
		// A member that died without leaving is dropped after 15 s (static members rejoin
		// at once anyway).
		kgo.SessionTimeout(15 * time.Second),
		kgo.RebalanceTimeout(15 * time.Second),
	}
}

// Loop consumes until ctx is done.
type Loop struct {
	Client  *kgo.Client
	Group   string
	Handle  Handler
	MaxPoll int
	Log     *slog.Logger
	// Retry backoff after a failed batch: doubles from MinBackoff up to MaxBackoff.
	MinBackoff, MaxBackoff time.Duration
}

// Run polls, applies and commits batch by batch. It returns nil when ctx is done; a batch that
// was being applied at that moment is not committed and comes back after a restart.
func (l *Loop) Run(ctx context.Context) error {
	if l.MaxPoll <= 0 {
		l.MaxPoll = 1000
	}
	if l.MinBackoff <= 0 {
		l.MinBackoff = 100 * time.Millisecond
	}
	if l.MaxBackoff <= 0 {
		l.MaxBackoff = 5 * time.Second
	}
	for {
		fetches := l.Client.PollRecords(ctx, l.MaxPoll)
		if fetches.IsClientClosed() || ctx.Err() != nil {
			return nil
		}
		fetches.EachError(func(topic string, p int32, err error) {
			if !errors.Is(err, context.Canceled) {
				l.Log.Warn("fetch error", "group", l.Group, "topic", topic, "partition", p, "err", err)
			}
		})
		recs := fetches.Records()
		if len(recs) > 0 {
			t0 := time.Now()
			if !l.apply(ctx, recs) {
				return nil // shutting down; the batch is not committed
			}
			// Synchronous commit of exactly this batch's offsets. If it fails, the batch is
			// applied again after the next restart or rebalance: harmless.
			if err := l.Client.CommitRecords(ctx, recs...); err != nil && ctx.Err() == nil {
				l.Log.Warn("offset commit failed; the batch may be redelivered", "group", l.Group, "records", len(recs), "err", err)
			}
			metrics.ConsumerBatch.WithLabelValues(l.Group).Observe(time.Since(t0).Seconds())
		}
		l.Client.AllowRebalance()
	}
}

// apply retries the handler until it succeeds (true) or ctx is done (false).
func (l *Loop) apply(ctx context.Context, recs []*kgo.Record) bool {
	backoff := l.MinBackoff
	for {
		err := l.Handle(ctx, recs)
		if err == nil {
			return true
		}
		if ctx.Err() != nil {
			return false
		}
		metrics.ConsumerRetries.WithLabelValues(l.Group).Inc()
		l.Log.Warn("applying a batch failed; retrying", "group", l.Group, "records", len(recs), "backoff", backoff, "err", err)
		select {
		case <-ctx.Done():
			return false
		case <-time.After(backoff):
		}
		backoff = min(2*backoff, l.MaxBackoff)
	}
}

// ExportLag publishes the lag of the given groups on `topic` every `every` until ctx is done
// (the aggregator runs it for every consumer group, including the log-sink's).
func ExportLag(ctx context.Context, adm *kadm.Client, topic string, groups []string, every time.Duration, log *slog.Logger) {
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		lags, err := adm.Lag(ctx, groups...)
		if err != nil && ctx.Err() == nil {
			log.Debug("lag", "err", err)
		}
		for g, l := range lags {
			if l.Error() != nil {
				continue
			}
			var total int64
			for _, pl := range l.Lag[topic] {
				if pl.Lag > 0 {
					total += pl.Lag
				}
			}
			metrics.ConsumerLag.WithLabelValues(g, topic).Set(float64(total))
		}
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}

// ResetToStart moves a consumer group's committed offsets on `topic` to the oldest retained
// record, so the next run replays the whole log. The group must have no active members.
func ResetToStart(ctx context.Context, adm *kadm.Client, group, topic string) (int, error) {
	described, err := adm.DescribeGroups(ctx, group)
	if err != nil {
		return 0, err
	}
	if g, ok := described[group]; ok && len(g.Members) > 0 {
		return 0, errors.New("consumer group " + group + " has active members (" + strings.Join(memberIDs(g), ", ") + "): stop it first")
	}
	starts, err := adm.ListStartOffsets(ctx, topic)
	if err != nil {
		return 0, err
	}
	if err := starts.Error(); err != nil {
		return 0, err
	}
	offsets := make(kadm.Offsets)
	starts.Each(func(o kadm.ListedOffset) {
		offsets.Add(kadm.Offset{Topic: o.Topic, Partition: o.Partition, At: o.Offset, LeaderEpoch: -1})
	})
	resp, err := adm.CommitOffsets(ctx, group, offsets)
	if err != nil {
		return 0, err
	}
	return len(offsets[topic]), resp.Error()
}

func memberIDs(g kadm.DescribedGroup) []string {
	out := make([]string, len(g.Members))
	for i, m := range g.Members {
		out[i] = m.MemberID
	}
	return out
}
