package aggregator

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
	"github.com/twmb/franz-go/pkg/kgo"

	"github.com/heyemlee/recloop/serving/internal/adtoken"
	"github.com/heyemlee/recloop/serving/internal/events"
	"github.com/heyemlee/recloop/serving/internal/userstate"
)

func record(t *testing.T, ev events.Event) *kgo.Record {
	t.Helper()
	r, err := events.KafkaRecord(events.Topic, ev)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func clientEvent(typ string, user int64, item int32) events.Event {
	id := uuid.Must(uuid.NewV7())
	at := events.V7Time(id)
	return events.Event{ID: id, Day: events.Day(at), Time: at, Type: typ, UserID: user, ItemID: item, Received: time.Now()}
}

func newAggregator(t *testing.T) (*Aggregator, userstate.Store, *miniredis.Miniredis) {
	t.Helper()
	mr := miniredis.NewMiniRedis()
	if err := mr.Start(); err != nil {
		t.Skipf("cannot start an in-process Redis on a local port: %v", err)
	}
	t.Cleanup(mr.Close)
	st := userstate.Store{R: redis.NewClient(&redis.Options{Addr: mr.Addr()})}
	return &Aggregator{State: st, Now: time.Now, Log: slog.New(slog.NewTextHandler(io.Discard, nil))}, st, mr
}

func TestRedeliveredBatchChangesNothing(t *testing.T) {
	a, st, _ := newAggregator(t)
	ctx := context.Background()
	click := clientEvent(events.TypeClick, 5, 10)
	buy := clientEvent(events.TypePurchase, 5, 11)
	imp := clientEvent(events.TypeImpression, 5, 12)
	served, _ := events.Served(5, uuid.Must(uuid.NewV7()), []byte(`{}`))
	batch := []*kgo.Record{record(t, click), record(t, imp), record(t, served), record(t, buy),
		{Topic: events.Topic, Value: []byte("not json")}}

	c, err := a.HandleRecords(ctx, batch)
	if err != nil {
		t.Fatal(err)
	}
	if c.Applied != 3 || c.Ignored != 1 || c.Invalid != 1 || c.Duplicate != 0 {
		t.Fatalf("first delivery: %+v", c)
	}
	// The consumer crashed before committing: the whole batch comes again.
	c, err = a.HandleRecords(ctx, batch)
	if err != nil {
		t.Fatal(err)
	}
	if c.Applied != 0 || c.Duplicate != 3 {
		t.Fatalf("redelivery: %+v", c)
	}
	uc, err := st.Read(ctx, 5, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if uc.HV != 2 || len(uc.History) != 2 || !uc.Bought[11] || !uc.Seen[12] {
		t.Fatalf("state %+v", uc)
	}
}

type failingApplier struct{ calls int }

func (f *failingApplier) ApplyBatch(context.Context, []userstate.Change, time.Time) ([]bool, error) {
	f.calls++
	return nil, errors.New("redis down")
}

func TestStoreFailureIsReturnedForRetry(t *testing.T) {
	f := &failingApplier{}
	a := &Aggregator{State: f, Now: time.Now, Log: slog.New(slog.NewTextHandler(io.Discard, nil))}
	if err := a.Handle(context.Background(), []*kgo.Record{record(t, clientEvent(events.TypeClick, 1, 2))}); err == nil {
		t.Fatal("a Redis failure must fail the batch so it is retried, not committed")
	}
	// A batch with nothing to apply does not touch Redis at all.
	served, _ := events.Served(1, uuid.Must(uuid.NewV7()), []byte(`{}`))
	if err := a.Handle(context.Background(), []*kgo.Record{record(t, served)}); err != nil || f.calls != 1 {
		t.Fatalf("served-only batch: err %v, calls %d", err, f.calls)
	}
}

// Sponsored impressions count towards the campaign's frequency cap of the serving day when
// their token verifies and names the impression; expired tokens count (the impression
// happened), forged or misplaced ones do not, and a redelivered record counts once.
func TestSponsoredImpressionsCountTowardsTheCap(t *testing.T) {
	a, state, _ := newAggregator(t)
	keys, err := adtoken.ParseKeys("k1:aggregator-test-key-0123456789abcdef01234")
	if err != nil {
		t.Fatal(err)
	}
	a.FreqKey = FreqKeys(keys)
	served := time.Now().Add(-time.Minute)
	imp := func(pos int, item int32, token func(adtoken.Claims) string) events.Event {
		ev := clientEvent(events.TypeImpression, 5, item)
		rid := uuid.Must(uuid.NewV7())
		ev.RequestID = uuid.NullUUID{UUID: rid, Valid: true}
		ev.ImpressionID = events.ImpressionID(rid, pos)
		ev.AdToken = token(adtoken.Claims{RequestID: rid, Position: pos, CampaignID: 9, ItemID: item, PriceCents: 12, UserID: 5})
		return ev
	}
	sign := func(c adtoken.Claims) string {
		tok, err := keys.Sign(c, served)
		if err != nil {
			t.Fatal(err)
		}
		return tok
	}
	ok := imp(3, 40, sign)
	expired := imp(9, 41, func(c adtoken.Claims) string {
		tok, _ := keys.Sign(c, served.Add(-2*adtoken.TTL))
		return tok
	})
	misplaced := imp(3, 42, func(c adtoken.Claims) string { c.Position = 9; return sign(c) })
	forged := imp(3, 43, func(c adtoken.Claims) string { return sign(c) + "x" })
	recs := []*kgo.Record{record(t, ok), record(t, ok), record(t, expired), record(t, misplaced), record(t, forged)}
	if _, err := a.HandleRecords(context.Background(), recs); err != nil {
		t.Fatal(err)
	}
	day := events.Day(served)
	n, err := state.Frequency(context.Background(), []string{userstate.FreqKey(5, 9, day)})
	if err != nil {
		t.Fatal(err)
	}
	// ok once (redelivered) + expired (served two TTLs earlier, possibly the day before).
	want := int64(1)
	if events.Day(served.Add(-2 * adtoken.TTL)).Equal(day) {
		want = 2
	}
	if n[0] != want {
		t.Fatalf("frequency %d, want %d", n[0], want)
	}
	// Every impression still marks its item as seen.
	uc, _ := state.Read(context.Background(), 5, time.Now())
	if !uc.Seen[40] || !uc.Seen[41] || !uc.Seen[42] || !uc.Seen[43] {
		t.Fatalf("seen %v", uc.Seen)
	}
}
