package itemctr_test

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"sort"
	"strconv"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"

	"github.com/heyemlee/recloop/serving/internal/events"
	"github.com/heyemlee/recloop/serving/internal/itemctr"
)

// fakeRedis holds hashes and strings the way the aggregator's script leaves them.
type fakeRedis struct {
	hashes  map[string]map[string]int
	strings map[string]string
	reads   int
	fail    error
}

func newFake() *fakeRedis {
	return &fakeRedis{hashes: map[string]map[string]int{}, strings: map[string]string{}}
}

func (f *fakeRedis) Get(ctx context.Context, key string) *redis.StringCmd {
	if f.fail != nil {
		return redis.NewStringResult("", f.fail)
	}
	v, ok := f.strings[key]
	if !ok {
		return redis.NewStringResult("", redis.Nil)
	}
	return redis.NewStringResult(v, nil)
}

func (f *fakeRedis) HGetAll(ctx context.Context, key string) *redis.MapStringStringCmd {
	f.reads++
	if f.fail != nil {
		return redis.NewMapStringStringResult(nil, f.fail)
	}
	out := map[string]string{}
	for k, v := range f.hashes[key] {
		out[k] = strconv.Itoa(v)
	}
	return redis.NewMapStringStringResult(out, nil)
}

func (f *fakeRedis) incr(key, field string) {
	if f.hashes[key] == nil {
		f.hashes[key] = map[string]int{}
	}
	f.hashes[key][field]++
}

type vectors struct {
	Events []struct {
		EventID      string    `json:"event_id"`
		Item         int32     `json:"item"`
		Type         string    `json:"type"`
		Received     time.Time `json:"received"`
		ImpressionID *string   `json:"impression_id"`
	} `json:"events"`
	Tables []struct {
		AsOf   time.Time            `json:"asof"`
		Since  *time.Time           `json:"since"`
		Counts map[string][4]uint32 `json:"counts"`
	} `json:"tables"`
}

// The shared vectors (testdata/item_ctr_vectors.json, also checked against the sample job):
// the aggregator's rule (events.CTRCounter, the first received copy of an event_id, the buckets
// of TargetOf) and the table's windows give the expected counts.
func TestTablesMatchTheSharedVectors(t *testing.T) {
	raw, err := os.ReadFile("../../../testdata/item_ctr_vectors.json")
	if err != nil {
		t.Fatal(err)
	}
	var v vectors
	if err := json.Unmarshal(raw, &v); err != nil {
		t.Fatal(err)
	}
	evs := v.Events
	sort.SliceStable(evs, func(a, b int) bool { return evs[a].Received.Before(evs[b].Received) })
	fake := newFake()
	applied := map[string]bool{} // the dedup markers
	for _, e := range evs {
		if applied[e.EventID] {
			continue
		}
		applied[e.EventID] = true
		ev := events.Event{ID: uuid.New(), Type: e.Type, ItemID: e.Item, Received: e.Received}
		if e.ImpressionID != nil {
			ev.ImpressionID = *e.ImpressionID
		}
		kind, at := events.CTRCounter(ev)
		if kind == "" {
			continue
		}
		tg := itemctr.TargetOf(at)
		fake.incr(tg.Key5m, itemctr.Field(kind, e.Item))
		fake.incr(tg.Key1h, itemctr.Field(kind, e.Item))
	}
	for _, tb := range v.Tables {
		var since time.Time
		if tb.Since != nil {
			since = *tb.Since
		}
		got, err := itemctr.Build(context.Background(), fake, tb.AsOf, since, tb.AsOf)
		if err != nil {
			t.Fatal(err)
		}
		for item := int32(0); item <= 9; item++ {
			want := tb.Counts[strconv.Itoa(int(item))]
			if g := got.Get(item).Array(); g != want {
				t.Errorf("table %s since %v, item %d: got %v, want %v", tb.AsOf.Format(time.RFC3339), tb.Since, item, g, want)
			}
		}
		if got.Items() != len(tb.Counts) {
			t.Errorf("table %s: %d items with counts, want %d", tb.AsOf.Format(time.RFC3339), got.Items(), len(tb.Counts))
		}
	}
}

func at(s string) time.Time {
	t, err := time.Parse(time.RFC3339Nano, s)
	if err != nil {
		panic(err)
	}
	return t
}

func TestBucketsKeysAndExpiry(t *testing.T) {
	tg := itemctr.TargetOf(at("2026-11-08T09:59:59.999999Z"))
	if tg.Key5m != "ictr:5m:202611080955" || tg.Key1h != "ictr:1h:2026110809" {
		t.Fatalf("keys %+v", tg)
	}
	// Read until 65 min 15 s (5-minute) or 25 h 15 s (hourly) after the bucket's start.
	if !tg.Expire5m.Equal(at("2026-11-08T11:25:00Z")) || !tg.Expire1h.Equal(at("2026-11-09T10:30:00Z")) {
		t.Fatalf("expiry %v %v", tg.Expire5m, tg.Expire1h)
	}
	if got := itemctr.Field(itemctr.Click, 42); got != "c:42" {
		t.Fatal(got)
	}
	// The table to read: the newest 5-minute boundary at least Grace ago.
	for now, want := range map[string]string{
		"2026-11-08T10:05:14.999Z": "2026-11-08T10:00:00Z",
		"2026-11-08T10:05:15Z":     "2026-11-08T10:05:00Z",
		"2026-11-08T10:09:59Z":     "2026-11-08T10:05:00Z",
	} {
		if got := itemctr.AsOfAt(at(now)); !got.Equal(at(want)) {
			t.Errorf("AsOfAt(%s) = %v, want %s", now, got, want)
		}
	}
	b5 := itemctr.Buckets5m(at("2026-11-08T10:05:00Z"), time.Time{})
	if len(b5) != 12 || !b5[0].Equal(at("2026-11-08T09:05:00Z")) || !b5[11].Equal(at("2026-11-08T10:00:00Z")) {
		t.Fatalf("5-minute buckets %v", b5)
	}
	b1 := itemctr.Buckets1h(at("2026-11-08T10:05:00Z"), time.Time{})
	if len(b1) != 24 || !b1[0].Equal(at("2026-11-07T10:00:00Z")) || !b1[23].Equal(at("2026-11-08T09:00:00Z")) {
		t.Fatalf("hourly buckets %v", b1)
	}
	// A seed exactly on a boundary keeps that bucket; one inside a bucket drops it.
	if b := itemctr.Buckets5m(at("2026-11-08T10:05:00Z"), at("2026-11-08T09:35:00Z")); len(b) != 6 {
		t.Fatalf("seed on a boundary: %v", b)
	}
	if b := itemctr.Buckets5m(at("2026-11-08T10:05:00Z"), at("2026-11-08T09:35:00.001Z")); len(b) != 5 {
		t.Fatalf("seed inside a bucket: %v", b)
	}
}

func TestRefreshBuildsOncePerAsOfAndAfterASeed(t *testing.T) {
	fake := newFake()
	fake.incr("ictr:5m:202611081000", "i:7")
	now := at("2026-11-08T10:05:20Z")
	f := &itemctr.Refresher{R: fake, Now: func() time.Time { return now }}
	ctx := context.Background()
	if f.Current().Fresh(now) {
		t.Fatal("no table yet")
	}
	built, err := f.Refresh(ctx)
	if err != nil || !built {
		t.Fatalf("first refresh: %v %v", built, err)
	}
	if c := f.Current().Get(7); c.ImpHour != 1 || !f.Current().AsOf.Equal(at("2026-11-08T10:05:00Z")) {
		t.Fatalf("table %+v", f.Current())
	}
	reads := fake.reads
	now = now.Add(4 * time.Minute) // same E
	if built, _ := f.Refresh(ctx); built || fake.reads != reads {
		t.Fatal("the table of the same E was rebuilt")
	}
	// A seed (make seed flushed the buckets) is seen at the next refresh, not at the next E.
	fake.hashes = map[string]map[string]int{}
	fake.strings[itemctr.SeededAtKey] = strconv.FormatInt(now.UnixMilli(), 10)
	if built, _ := f.Refresh(ctx); !built || f.Current().Get(7).ImpHour != 0 || f.Current().Since.IsZero() {
		t.Fatalf("after the seed: %+v", f.Current())
	}
	// A failed build keeps the last table; past MaxAge it is not fresh any more.
	fake.fail = errors.New("down")
	now = now.Add(5 * time.Minute)
	if _, err := f.Refresh(ctx); err == nil || f.Current() == nil {
		t.Fatal("a failed build must keep the last table")
	}
	if !f.Current().Fresh(now) || f.Current().Fresh(now.Add(itemctr.MaxAge)) {
		t.Fatal("freshness")
	}
	var nilTable *itemctr.Table
	if nilTable.Fresh(now) {
		t.Fatal("nil table")
	}
}
