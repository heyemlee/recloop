package userstate

import (
	"context"
	"os"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"
)

func historyChange(id string, uid int64, item int32, at time.Time) Change {
	return Change{EventID: id, UserID: uid, ItemID: item, Kind: KindHistory, At: at}
}

func TestNoWorkerQueuesNothing(t *testing.T) {
	s, mr := newStore(t)
	ctx := context.Background()
	now := time.UnixMilli(1_800_000_000_000)
	if _, err := s.Apply(ctx, historyChange("e1", 7, 42, now), now); err != nil {
		t.Fatal(err)
	}
	if mr.Exists(GenQueueKey) || mr.Exists(GenPendingKey(7)) {
		t.Fatalf("queued without a registered worker: %v", mr.Keys())
	}
}

func TestQueueDebounceAndMaxWait(t *testing.T) {
	s, mr := newStore(t)
	ctx := context.Background()
	if err := mr.Set(GenWorkerKey, "{}"); err != nil {
		t.Fatal(err)
	}
	t0 := time.UnixMilli(1_800_000_000_000)
	due := func() float64 {
		t.Helper()
		v, err := mr.ZScore(GenQueueKey, "7")
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	ms := func(d time.Duration) float64 { return float64(t0.Add(d).UnixMilli()) }

	if _, err := s.Apply(ctx, historyChange("a", 7, 1, t0.Add(-time.Second)), t0); err != nil {
		t.Fatal(err)
	}
	if got := due(); got != ms(GenDebounce) {
		t.Fatalf("first change: due %v, want now + debounce %v", got, ms(GenDebounce))
	}
	// Another change 10 s later pushes it back by the debounce...
	if _, err := s.Apply(ctx, historyChange("b", 7, 2, t0.Add(10*time.Second)), t0.Add(10*time.Second)); err != nil {
		t.Fatal(err)
	}
	if got := due(); got != ms(10*time.Second+GenDebounce) {
		t.Fatalf("second change: due %v", got)
	}
	// ... but never past max wait after the first change still pending.
	at := t0.Add(100 * time.Second)
	if _, err := s.Apply(ctx, historyChange("c", 7, 3, at), at); err != nil {
		t.Fatal(err)
	}
	if got := due(); got != ms(GenMaxWait) {
		t.Fatalf("third change: due %v, want first + max wait %v", got, ms(GenMaxWait))
	}
	// A duplicate event and an impression change nothing.
	if _, err := s.Apply(ctx, historyChange("c", 7, 3, at), at.Add(time.Second)); err != nil {
		t.Fatal(err)
	}
	imp := Change{EventID: "i", UserID: 7, ItemID: 9, Kind: KindImpression, At: at}
	if _, err := s.Apply(ctx, imp, at.Add(2*time.Second)); err != nil {
		t.Fatal(err)
	}
	pending, err := mr.List(GenPendingKey(7))
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		strings.Join([]string{i64(t0.Add(-time.Second).UnixMilli()), i64(t0.UnixMilli())}, ":"),
		strings.Join([]string{i64(t0.Add(10 * time.Second).UnixMilli()), i64(t0.Add(10 * time.Second).UnixMilli())}, ":"),
		strings.Join([]string{i64(at.UnixMilli()), i64(at.UnixMilli())}, ":"),
	}
	if !reflect.DeepEqual(pending, want) {
		t.Fatalf("pending %v, want %v", pending, want)
	}
	if ttl := mr.TTL(GenPendingKey(7)); ttl != GenPendingTTL {
		t.Fatalf("pending TTL %v", ttl)
	}
	if got := due(); got != ms(GenMaxWait) {
		t.Fatalf("after a duplicate and an impression: due %v", got)
	}
}

func TestPurchaseQueuesAndSponsoredImpressionKeepsItsCounter(t *testing.T) {
	s, mr := newStore(t)
	ctx := context.Background()
	if err := mr.Set(GenWorkerKey, "{}"); err != nil {
		t.Fatal(err)
	}
	now := time.UnixMilli(1_800_000_000_000)
	buy := Change{EventID: "p", UserID: 8, ItemID: 4, Kind: KindPurchase, At: now}
	if _, err := s.Apply(ctx, buy, now); err != nil {
		t.Fatal(err)
	}
	if _, err := mr.ZScore(GenQueueKey, "8"); err != nil {
		t.Fatalf("a purchase must queue the user: %v", err)
	}
	// The frequency-cap key moved behind the generation keys: it must still be counted.
	fk := FreqKey(8, 5, now)
	imp := Change{EventID: "si", UserID: 8, ItemID: 4, Kind: KindImpression, At: now, FreqKey: fk}
	if _, err := s.Apply(ctx, imp, now); err != nil {
		t.Fatal(err)
	}
	if v, _ := mr.Get(fk); v != "1" {
		t.Fatalf("frequency counter %q", v)
	}
}

func TestReadWithGen(t *testing.T) {
	s, mr := newStore(t)
	ctx := context.Background()
	now := time.UnixMilli(1_800_000_000_000)
	raw, err := os.ReadFile("../../../testdata/gen_payload.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := mr.Set(GenKey(5, "v1"), strings.TrimSpace(string(raw))); err != nil {
		t.Fatal(err)
	}
	if err := mr.Set(GenKey(6, "v1"), "not json"); err != nil {
		t.Fatal(err)
	}
	uc, err := s.ReadWith(ctx, 5, now, ReadOptions{GenVersion: "v1"})
	if err != nil {
		t.Fatal(err)
	}
	want := Generated{HV: 12, At: 1762600000123, Items: []int32{5, 17, 3458}, Scores: []float64{-0.5, -1.25, -7.0625}}
	if uc.Gen == nil || !reflect.DeepEqual(*uc.Gen, want) {
		t.Fatalf("gen %+v, want %+v (testdata/gen_payload.json, written by ml/nearline/keys.py)", uc.Gen, want)
	}
	if uc, _ := s.ReadWith(ctx, 5, now, ReadOptions{GenVersion: "v2"}); uc.Gen != nil {
		t.Fatalf("another version read: %+v", uc.Gen)
	}
	if uc, _ := s.Read(ctx, 5, now); uc.Gen != nil {
		t.Fatalf("Read must not read generation results: %+v", uc.Gen)
	}
	uc, err = s.ReadWith(ctx, 6, now, ReadOptions{GenVersion: "v1"})
	if err != nil || uc.Gen != nil || !uc.GenInvalid {
		t.Fatalf("an invalid payload: err %v, gen %+v, invalid %v", err, uc.Gen, uc.GenInvalid)
	}
}

func i64(v int64) string { return strconv.FormatInt(v, 10) }
