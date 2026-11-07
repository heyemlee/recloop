package userstate

import (
	"context"
	"strconv"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
)

func newStore(t *testing.T) (Store, *miniredis.Miniredis) {
	t.Helper()
	mr := miniredis.NewMiniRedis()
	if err := mr.Start(); err != nil {
		// e.g. a sandbox that forbids listening on local ports
		t.Skipf("cannot start an in-process Redis on a local port: %v", err)
	}
	t.Cleanup(mr.Close)
	return Store{R: redis.NewClient(&redis.Options{Addr: mr.Addr()})}, mr
}

func TestApplyIsIdempotentPerEvent(t *testing.T) {
	s, _ := newStore(t)
	ctx := context.Background()
	now := time.Now()
	c := Change{EventID: "e1", UserID: 7, ItemID: 42, Kind: KindPurchase, At: now}
	for i := 0; i < 3; i++ {
		applied, err := s.Apply(ctx, c, now)
		if err != nil {
			t.Fatal(err)
		}
		if applied != (i == 0) {
			t.Fatalf("attempt %d: applied=%v", i, applied)
		}
	}
	uc, err := s.Read(ctx, 7, now)
	if err != nil {
		t.Fatal(err)
	}
	if uc.HV != 1 || len(uc.History) != 1 || uc.History[0] != 42 || !uc.Bought[42] || !uc.Seen[42] {
		t.Fatalf("state after a purchase applied 3 times: %+v", uc)
	}
}

func TestHistoryIsOrderedByEventTimeAndCapped(t *testing.T) {
	s, _ := newStore(t)
	ctx := context.Background()
	base := time.UnixMilli(1_800_000_000_000)
	for i := 0; i < HistoryMax+5; i++ {
		c := Change{EventID: "h" + strconv.Itoa(i), UserID: 1, ItemID: int32(i), Kind: KindHistory, At: base.Add(time.Duration(i) * time.Second)}
		if _, err := s.Apply(ctx, c, base); err != nil {
			t.Fatal(err)
		}
	}
	// A late event for item 10 with an old time must not move it to the front.
	if _, err := s.Apply(ctx, Change{EventID: "late", UserID: 1, ItemID: 54, Kind: KindHistory, At: base}, base); err != nil {
		t.Fatal(err)
	}
	uc, _ := s.Read(ctx, 1, base)
	if len(uc.History) != HistoryMax {
		t.Fatalf("history has %d items", len(uc.History))
	}
	if uc.History[0] != 5 || uc.History[len(uc.History)-1] != 54 {
		t.Fatalf("history oldest %d, newest %d", uc.History[0], uc.History[len(uc.History)-1])
	}
	if uc.HV != int64(HistoryMax+6) {
		t.Fatalf("hv %d", uc.HV)
	}
}

func TestSeenWindow(t *testing.T) {
	s, _ := newStore(t)
	ctx := context.Background()
	now := time.Now()
	old := Change{EventID: "i-old", UserID: 3, ItemID: 1, Kind: KindImpression, At: now.Add(-25 * time.Hour)}
	recent := Change{EventID: "i-new", UserID: 3, ItemID: 2, Kind: KindImpression, At: now.Add(-time.Hour)}
	for _, c := range []Change{old, recent} {
		if _, err := s.Apply(ctx, c, now); err != nil {
			t.Fatal(err)
		}
	}
	uc, _ := s.Read(ctx, 3, now)
	if uc.Seen[1] || !uc.Seen[2] || uc.HV != 0 || len(uc.History) != 0 {
		t.Fatalf("state %+v", uc)
	}
	// A click counts as seen even without an impression event.
	if _, err := s.Apply(ctx, Change{EventID: "c1", UserID: 3, ItemID: 9, Kind: KindHistory, At: now}, now); err != nil {
		t.Fatal(err)
	}
	if uc, _ = s.Read(ctx, 3, now); !uc.Seen[9] || uc.HV != 1 {
		t.Fatalf("after a click: %+v", uc)
	}
}

func TestApplyBatchMatchesApplyAndReloadsTheScript(t *testing.T) {
	s, mr := newStore(t)
	ctx := context.Background()
	now := time.Now()
	batch := []Change{
		{EventID: "b1", UserID: 3, ItemID: 10, Kind: KindHistory, At: now.Add(-3 * time.Second)},
		{EventID: "b2", UserID: 3, ItemID: 11, Kind: KindImpression, At: now.Add(-2 * time.Second)},
		{EventID: "b3", UserID: 3, ItemID: 12, Kind: KindPurchase, At: now.Add(-time.Second)},
		{EventID: "b1", UserID: 3, ItemID: 10, Kind: KindHistory, At: now.Add(-3 * time.Second)}, // redelivered in the same batch
	}
	got, err := s.ApplyBatch(ctx, batch, now)
	if err != nil {
		t.Fatal(err)
	}
	if want := []bool{true, true, true, false}; len(got) != 4 || got[0] != want[0] || got[1] != want[1] || got[2] != want[2] || got[3] != want[3] {
		t.Fatalf("applied %v", got)
	}
	// Redis lost its script cache (restart): the next batch reloads it; the old events are
	// still recognised as duplicates.
	mr.FlushAll()
	if _, err := s.ApplyBatch(ctx, batch[:1], now); err != nil {
		t.Fatalf("after a script flush: %v", err)
	}
	s.R.ScriptFlush(ctx)
	got, err = s.ApplyBatch(ctx, []Change{batch[0], {EventID: "b4", UserID: 3, ItemID: 13, Kind: KindHistory, At: now}}, now)
	if err != nil {
		t.Fatalf("NOSCRIPT was not handled: %v", err)
	}
	if got[0] || !got[1] {
		t.Fatalf("applied %v after reload", got)
	}
}
