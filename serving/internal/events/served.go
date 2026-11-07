package events

import (
	"context"
	"log/slog"
	"sync/atomic"
	"time"

	"github.com/google/uuid"

	"github.com/heyemlee/recloop/serving/internal/metrics"
)

// ServedSink stores served events: the Redpanda producer (Kafka) in production, PostgreSQL
// directly (PGServed) on the synchronous path.
type ServedSink interface {
	WriteServed(ctx context.Context, evs []Event) error
}

// ServedWriter writes served events off the request path (design §4.1 ⑦: logging never blocks
// the response). Enqueue never blocks: when the buffer is full the event is dropped and
// counted. A background loop sends batches of up to maxBatch events, or whatever has
// accumulated every `every`. A batch the sink refuses (the broker restarting) is retried, with
// the same event_ids, for up to RetryFor; meanwhile new events wait in the buffer, and only
// when it is full are they dropped. The request path never waits.
type ServedWriter struct {
	sink    ServedSink
	ch      chan Event
	log     *slog.Logger
	dropped atomic.Int64
	written atomic.Int64
	done    chan struct{}
	// SendTimeout bounds one attempt to write a batch; RetryFor all attempts of one batch.
	SendTimeout time.Duration
	RetryFor    time.Duration
}

func NewServedWriter(sink ServedSink, buffer int, log *slog.Logger) *ServedWriter {
	return &ServedWriter{sink: sink, ch: make(chan Event, buffer), log: log, done: make(chan struct{}),
		SendTimeout: 5 * time.Second, RetryFor: 30 * time.Second}
}

// Served builds the served event of a request: a fresh UUIDv7, partitioned by its own time.
func Served(userID int64, requestID uuid.UUID, payload []byte) (Event, error) {
	id, err := uuid.NewV7()
	if err != nil {
		return Event{}, err
	}
	at := V7Time(id)
	return Event{
		ID: id, Day: Day(at), Time: at, Type: TypeServed, UserID: userID,
		RequestID: uuid.NullUUID{UUID: requestID, Valid: true}, Payload: payload, Received: time.Now().UTC(),
	}, nil
}

// Enqueue never blocks; it reports false if the event was dropped.
func (w *ServedWriter) Enqueue(ev Event) bool {
	select {
	case w.ch <- ev:
		return true
	default:
		w.dropped.Add(1)
		metrics.ServedEvents.WithLabelValues("dropped").Inc()
		return false
	}
}

// Stats returns how many events were written and dropped so far.
func (w *ServedWriter) Stats() (written, dropped int64) { return w.written.Load(), w.dropped.Load() }

// Run writes batches until ctx is done, then flushes what is buffered and returns.
func (w *ServedWriter) Run(ctx context.Context, maxBatch int, every time.Duration) {
	defer close(w.done)
	t := time.NewTicker(every)
	defer t.Stop()
	batch := make([]Event, 0, maxBatch)
	flush := func(parent context.Context) {
		if len(batch) == 0 {
			return
		}
		giveUp := time.Now().Add(w.RetryFor)
		backoff := 200 * time.Millisecond
		var err error
		for {
			fctx, cancel := context.WithTimeout(parent, w.SendTimeout)
			err = w.sink.WriteServed(fctx, batch)
			cancel()
			if err == nil || time.Now().Add(backoff).After(giveUp) {
				break
			}
			metrics.ServedEvents.WithLabelValues("retried").Add(float64(len(batch)))
			w.log.Warn("writing served events failed; retrying", "events", len(batch), "backoff", backoff, "err", err)
			select {
			case <-parent.Done():
			case <-time.After(backoff):
			}
			if parent.Err() != nil {
				break
			}
			backoff = min(2*backoff, 2*time.Second)
		}
		if err != nil {
			w.dropped.Add(int64(len(batch)))
			metrics.ServedEvents.WithLabelValues("failed").Add(float64(len(batch)))
			w.log.Error("writing served events failed", "events", len(batch), "err", err)
		} else {
			w.written.Add(int64(len(batch)))
			metrics.ServedEvents.WithLabelValues("produced").Add(float64(len(batch)))
		}
		batch = batch[:0]
	}
	for {
		select {
		case ev := <-w.ch:
			batch = append(batch, ev)
			if len(batch) >= maxBatch {
				flush(ctx)
			}
		case <-t.C:
			flush(ctx)
		case <-ctx.Done():
			// Drain what is buffered with a fresh context: shutdown should not lose the log.
			bg := context.Background()
			for {
				select {
				case ev := <-w.ch:
					batch = append(batch, ev)
					if len(batch) >= maxBatch {
						flush(bg)
					}
					continue
				default:
				}
				break
			}
			flush(bg)
			return
		}
	}
}

// Wait blocks until Run has flushed and returned.
func (w *ServedWriter) Wait() { <-w.done }
