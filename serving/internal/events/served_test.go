package events

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
)

// flakySink refuses the first `fail` writes (a broker restarting), then stores everything.
type flakySink struct {
	mu   sync.Mutex
	fail int
	got  map[uuid.UUID]int
}

func (f *flakySink) WriteServed(_ context.Context, evs []Event) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.fail > 0 {
		f.fail--
		return errors.New("broker unavailable")
	}
	for _, ev := range evs {
		f.got[ev.ID]++
	}
	return nil
}

func TestServedWriterRetriesAFailedBatch(t *testing.T) {
	sink := &flakySink{fail: 2, got: map[uuid.UUID]int{}}
	w := NewServedWriter(sink, 100, slog.New(slog.NewTextHandler(io.Discard, nil)))
	w.SendTimeout, w.RetryFor = time.Second, 10*time.Second
	ctx, cancel := context.WithCancel(context.Background())
	go w.Run(ctx, 10, 5*time.Millisecond)
	var ids []uuid.UUID
	for i := 0; i < 25; i++ {
		ev, _ := Served(1, uuid.Must(uuid.NewV7()), []byte(`{}`))
		ids = append(ids, ev.ID)
		if !w.Enqueue(ev) {
			t.Fatal("dropped with room in the buffer")
		}
	}
	time.Sleep(1500 * time.Millisecond) // two failed attempts: 200 + 400 ms of backoff
	cancel()
	w.Wait()
	written, dropped := w.Stats()
	if written != 25 || dropped != 0 {
		t.Fatalf("written %d, dropped %d", written, dropped)
	}
	for _, id := range ids {
		if sink.got[id] != 1 {
			t.Fatalf("event stored %d times", sink.got[id])
		}
	}
}
