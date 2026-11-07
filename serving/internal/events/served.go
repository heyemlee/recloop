package events

import (
	"context"
	"log/slog"
	"sync/atomic"
	"time"

	"github.com/google/uuid"

	"github.com/heyemlee/recloop/serving/internal/store/db"
)

// ServedSink stores served events (store.Store).
type ServedSink interface {
	InsertServed(ctx context.Context, params []db.InsertEventParams) error
}

// ServedWriter writes served events off the request path (design §4.1 ⑦: logging never blocks
// the response). Events are batched; when the buffer is full an event is dropped and counted.
// In M2 this becomes a Redpanda producer.
type ServedWriter struct {
	sink    ServedSink
	ch      chan db.InsertEventParams
	log     *slog.Logger
	dropped atomic.Int64
	written atomic.Int64
	done    chan struct{}
}

func NewServedWriter(sink ServedSink, buffer int, log *slog.Logger) *ServedWriter {
	return &ServedWriter{sink: sink, ch: make(chan db.InsertEventParams, buffer), log: log, done: make(chan struct{})}
}

// Served builds the served event of a request: a fresh UUIDv7, partitioned by its own time.
func Served(userID int64, requestID uuid.UUID, payload []byte) (db.InsertEventParams, error) {
	id, err := uuid.NewV7()
	if err != nil {
		return db.InsertEventParams{}, err
	}
	at := V7Time(id)
	return Params(Event{
		ID: id, Day: Day(at), Time: at, Type: TypeServed, UserID: userID,
		RequestID: uuid.NullUUID{UUID: requestID, Valid: true}, Payload: payload,
	}), nil
}

// Enqueue never blocks; it reports false if the event was dropped.
func (w *ServedWriter) Enqueue(p db.InsertEventParams) bool {
	select {
	case w.ch <- p:
		return true
	default:
		w.dropped.Add(1)
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
	batch := make([]db.InsertEventParams, 0, maxBatch)
	flush := func(fctx context.Context) {
		if len(batch) == 0 {
			return
		}
		if err := w.sink.InsertServed(fctx, batch); err != nil {
			w.dropped.Add(int64(len(batch)))
			w.log.Error("writing served events failed", "events", len(batch), "err", err)
		} else {
			w.written.Add(int64(len(batch)))
		}
		batch = batch[:0]
	}
	for {
		select {
		case p := <-w.ch:
			batch = append(batch, p)
			if len(batch) >= maxBatch {
				flush(ctx)
			}
		case <-t.C:
			flush(ctx)
		case <-ctx.Done():
			fctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			for {
				select {
				case p := <-w.ch:
					batch = append(batch, p)
					if len(batch) >= maxBatch {
						flush(fctx)
					}
					continue
				default:
				}
				break
			}
			flush(fctx)
			cancel()
			return
		}
	}
}

// Wait blocks until Run has flushed and returned.
func (w *ServedWriter) Wait() { <-w.done }
