package aggregator_test

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"math/rand/v2"
	"strconv"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/twmb/franz-go/pkg/kadm"
	"github.com/twmb/franz-go/pkg/kgo"

	"github.com/heyemlee/recloop/serving/internal/aggregator"
	"github.com/heyemlee/recloop/serving/internal/consumer"
	"github.com/heyemlee/recloop/serving/internal/events"
	"github.com/heyemlee/recloop/serving/internal/testutil"
	"github.com/heyemlee/recloop/serving/internal/userstate"
)

// The real path: api producer -> Redpanda -> consumer loop -> aggregator -> Redis. Events are
// published with duplicates, the consumer is stopped half way (its last batch may be applied but
// not committed) and started again; every event must be applied exactly once.
func TestRedeliveryThroughRedpanda(t *testing.T) {
	brokers := testutil.Brokers(t)
	rdb := testutil.RedisDB(t, 14)
	ctx := context.Background()
	topic := "events-test-" + strconv.FormatUint(rand.Uint64(), 36)
	group := "aggregator-test-" + strconv.FormatUint(rand.Uint64(), 36)

	admin, err := kgo.NewClient(kgo.SeedBrokers(brokers...))
	if err != nil {
		t.Fatal(err)
	}
	defer admin.Close()
	adm := kadm.NewClient(admin)
	if _, err := adm.CreateTopic(ctx, 3, 1, nil, topic); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _, _ = adm.DeleteTopics(context.Background(), topic) })

	prod, err := kgo.NewClient(kgo.SeedBrokers(brokers...), kgo.ProducerLinger(0))
	if err != nil {
		t.Fatal(err)
	}
	defer prod.Close()
	pub := events.Kafka{Client: prod, Topic: topic}

	// 30 users x 20 clicks, each batch also re-sent (a client retry): 1200 deliveries of 600 events.
	const users, clicks = 30, 20
	var batches [][]events.Event
	for c := 0; c < clicks; c++ {
		var b []events.Event
		for u := 0; u < users; u++ {
			id := uuid.Must(uuid.NewV7())
			at := events.V7Time(id)
			b = append(b, events.Event{ID: id, Day: events.Day(at), Time: at, Type: events.TypeClick,
				UserID: int64(u), ItemID: int32(c), Received: time.Now()})
		}
		batches = append(batches, b)
	}
	publish := func(from, to int) {
		for _, b := range batches[from:to] {
			for range 2 {
				if _, err := pub.Publish(ctx, b); err != nil {
					t.Fatal(err)
				}
			}
		}
	}

	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	state := userstate.Store{R: rdb, Group: "test"}
	consume := func(stopWhen func() bool) {
		cl, err := kgo.NewClient(consumer.Options(brokers, group, topic)...)
		if err != nil {
			t.Fatal(err)
		}
		defer cl.CloseAllowingRebalance()
		cctx, cancel := context.WithCancel(ctx)
		defer cancel()
		agg := &aggregator.Aggregator{State: state, Now: time.Now, Log: log}
		loop := &consumer.Loop{Client: cl, Group: group, Handle: agg.Handle, MaxPoll: 50, Log: log}
		done := make(chan error, 1)
		go func() { done <- loop.Run(cctx) }()
		deadline := time.Now().Add(30 * time.Second)
		for !stopWhen() {
			if time.Now().After(deadline) {
				t.Fatal("consumer did not get there in 30 s")
			}
			time.Sleep(20 * time.Millisecond)
		}
		cancel()
		if err := <-done; err != nil {
			t.Fatal(err)
		}
	}
	total := func() int64 {
		var n int64
		for u := 0; u < users; u++ {
			v, _ := rdb.Get(ctx, userstate.HVKey(int64(u))).Int64()
			n += v
		}
		return n
	}

	publish(0, clicks/2)
	consume(func() bool { return total() >= users*clicks/4 }) // stop somewhere in the middle
	publish(clicks/2, clicks)
	consume(func() bool { return total() >= users*clicks })
	time.Sleep(300 * time.Millisecond) // anything still arriving would push hv past 20

	for u := 0; u < users; u++ {
		uc, err := state.Read(ctx, int64(u), time.Now())
		if err != nil {
			t.Fatal(err)
		}
		if uc.HV != clicks || len(uc.History) != clicks {
			t.Fatalf("user %d: hv %d, %d history items; want %d each", u, uc.HV, len(uc.History), clicks)
		}
	}
	lags, err := adm.Lag(ctx, group)
	if err != nil {
		t.Fatal(err)
	}
	if l := lags[group]; l.Error() != nil || l.Lag.Total() != 0 {
		t.Fatalf("group lag after consuming everything: %v %v", l.Lag.Total(), l.Error())
	}
	fmt.Fprintf(io.Discard, "%d", total())
}
