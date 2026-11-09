// Package itemctr counts the items' recent impressions and clicks: the inputs of the pCTR's
// "item CTR over the last hour and day" (design §9.2 step 3: from the log, over windows before
// the impression, the same definition online and offline). ml/samples/job.py recomputes the same
// numbers from events_raw.
//
// What counts: impression and click events that carry an impression_id, once per event_id (the
// aggregator's dedup marker), on every surface, in the 5-minute and hourly UTC buckets of the time
// the api received them (Event.Received = events_raw.server_ts; a client clock cannot move an
// event into a bucket that is already closed or one that has not started).
//
// A Table has an as-of time E, a 5-minute boundary. Its hour sums the 12 five-minute buckets in
// [E - 60 min, E); its day sums the 24 hourly buckets in [H - 24 h, H), H = E truncated to the
// hour. Only complete buckets count, so the hour ends 15 s to 5 min 15 s before the request that
// reads it and the day up to 1 h 15 s before. After `make seed` (SeededAtKey) the buckets that
// start before the seed (rounded up to a bucket boundary) are left out: the seed flushed them.
//
// The aggregator increments the buckets inside apply.lua (package userstate), under the event's
// dedup marker; the api builds the table in the background after every 5-minute boundary and
// logs the counts it used, with E and the seed time, in the served event.
package itemctr

import (
	"context"
	"errors"
	"log/slog"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/heyemlee/recloop/serving/internal/metrics"
)

const (
	// Bucket5m and Bucket1h are the bucket widths of the hour's and the day's window.
	Bucket5m = 5 * time.Minute
	Bucket1h = time.Hour
	// Window5m and Window1h are how many complete buckets each window sums.
	Window5m = 12
	Window1h = 24
	// Grace: the table for E is built Grace after E, so the aggregator has applied the events
	// the api received just before E (its lag is well under a second when it keeps up).
	Grace = 15 * time.Second
	// MaxAge: a table whose E is older than this is not used (its builds keep failing); the
	// pCTR then gets no counts and uses its prior.
	MaxAge = 15 * time.Minute
	// TTL5m and TTL1h keep a bucket after its start: the last table that reads a 5-minute
	// bucket is built 65 min 15 s after its start, the last that reads an hourly one 25 h 15 s
	// after; the rest is room for late builds.
	TTL5m = 90 * time.Minute
	TTL1h = 25*time.Hour + 30*time.Minute
	// SeededAtKey holds the time of the last `make seed` (Unix ms), which flushed the buckets.
	// A full rebuild from events_raw (aggregator rebuild-from-db) deletes it.
	SeededAtKey = "state:seeded_at"
	// Field kinds: ictr buckets are hashes of "i:{item}" (impressions) and "c:{item}" (clicks).
	Impression = "i"
	Click      = "c"
	// RefreshEvery is how often the api checks whether a new table is due.
	RefreshEvery = 10 * time.Second
)

// Key5m is the hash of the 5-minute bucket that holds t: ictr:5m:{yyyymmddHHMM} (its start, UTC).
func Key5m(t time.Time) string {
	return "ictr:5m:" + t.UTC().Truncate(Bucket5m).Format("200601021504")
}

// Key1h is the hash of the hourly bucket that holds t: ictr:1h:{yyyymmddHH} (UTC).
func Key1h(t time.Time) string {
	return "ictr:1h:" + t.UTC().Truncate(Bucket1h).Format("2006010215")
}

// Field is the hash field of an item's impressions (Impression) or clicks (Click).
func Field(kind string, item int32) string { return kind + ":" + strconv.Itoa(int(item)) }

// Target is where an event received at a given time is counted.
type Target struct {
	Key5m, Key1h       string
	Expire5m, Expire1h time.Time // when each bucket's hash expires
}

// TargetOf names the buckets of an event the api received at t.
func TargetOf(received time.Time) Target {
	s5 := received.UTC().Truncate(Bucket5m)
	s1 := received.UTC().Truncate(Bucket1h)
	return Target{Key5m: Key5m(s5), Key1h: Key1h(s1), Expire5m: s5.Add(TTL5m), Expire1h: s1.Add(TTL1h)}
}

// AsOfAt is the table a request at now should read: the newest 5-minute boundary at least
// Grace ago.
func AsOfAt(now time.Time) time.Time { return now.Add(-Grace).UTC().Truncate(Bucket5m) }

// ceil rounds t up to a multiple of d (the zero time stays zero).
func ceil(t time.Time, d time.Duration) time.Time {
	if t.IsZero() {
		return t
	}
	f := t.UTC().Truncate(d)
	if f.Equal(t) {
		return f
	}
	return f.Add(d)
}

// Buckets5m are the starts of the 5-minute buckets the hour of a table at asOf sums, oldest
// first: [asOf - 60 min, asOf), none that starts before since (rounded up to 5 minutes).
func Buckets5m(asOf, since time.Time) []time.Time {
	return buckets(asOf.UTC().Truncate(Bucket5m), Bucket5m, Window5m, ceil(since, Bucket5m))
}

// Buckets1h are the starts of the hourly buckets the day of a table at asOf sums: [H - 24 h, H)
// with H = asOf truncated to the hour, none that starts before since (rounded up to the hour).
func Buckets1h(asOf, since time.Time) []time.Time {
	return buckets(asOf.UTC().Truncate(Bucket1h), Bucket1h, Window1h, ceil(since, Bucket1h))
}

func buckets(end time.Time, width time.Duration, n int, from time.Time) []time.Time {
	out := make([]time.Time, 0, n)
	for k := n; k >= 1; k-- {
		b := end.Add(-time.Duration(k) * width)
		if !from.IsZero() && b.Before(from) {
			continue
		}
		out = append(out, b)
	}
	return out
}

// Counts of one item in a table's windows.
type Counts struct {
	ImpHour, ClkHour, ImpDay, ClkDay uint32
}

// Array is the order the served event logs and the Rank request sends: impressions and clicks
// of the hour, then of the day.
func (c Counts) Array() [4]uint32 { return [4]uint32{c.ImpHour, c.ClkHour, c.ImpDay, c.ClkDay} }

// Table is one snapshot of every item's counts.
type Table struct {
	AsOf  time.Time // E
	Since time.Time // the last seed's time; zero: none recorded
	Built time.Time
	items map[int32]Counts
}

// NewTable is a table with the given counts (tests, tools).
func NewTable(asOf, since time.Time, counts map[int32]Counts) *Table {
	if counts == nil {
		counts = map[int32]Counts{}
	}
	return &Table{AsOf: asOf.UTC(), Since: since, Built: asOf, items: counts}
}

// Get returns an item's counts (zero for an item without impressions or clicks).
func (t *Table) Get(item int32) Counts { return t.items[item] }

// Items is how many items have a count.
func (t *Table) Items() int { return len(t.items) }

// Fresh reports whether the table may serve a request at now.
func (t *Table) Fresh(now time.Time) bool { return t != nil && now.Sub(t.AsOf) <= MaxAge }

// Reader is what Build reads from the state Redis (a redis.Cmdable).
type Reader interface {
	Get(ctx context.Context, key string) *redis.StringCmd
	HGetAll(ctx context.Context, key string) *redis.MapStringStringCmd
}

// SeededAt reads SeededAtKey (zero when absent).
func SeededAt(ctx context.Context, r Reader) (time.Time, error) {
	v, err := r.Get(ctx, SeededAtKey).Int64()
	if errors.Is(err, redis.Nil) {
		return time.Time{}, nil
	}
	if err != nil {
		return time.Time{}, err
	}
	return time.UnixMilli(v).UTC(), nil
}

// Build sums the buckets of a table at asOf after a seed at since. The hashes are read one at
// a time (each HGETALL holds Redis for about a millisecond; a pipeline of all of them would hold
// it for their sum).
func Build(ctx context.Context, r Reader, asOf, since, now time.Time) (*Table, error) {
	t := &Table{AsOf: asOf.UTC(), Since: since, Built: now, items: map[int32]Counts{}}
	for _, b := range Buckets5m(asOf, since) {
		m, err := r.HGetAll(ctx, Key5m(b)).Result()
		if err != nil {
			return nil, err
		}
		add(t.items, m, func(c *Counts) (*uint32, *uint32) { return &c.ImpHour, &c.ClkHour })
	}
	for _, b := range Buckets1h(asOf, since) {
		m, err := r.HGetAll(ctx, Key1h(b)).Result()
		if err != nil {
			return nil, err
		}
		add(t.items, m, func(c *Counts) (*uint32, *uint32) { return &c.ImpDay, &c.ClkDay })
	}
	return t, nil
}

func add(items map[int32]Counts, fields map[string]string, slots func(*Counts) (*uint32, *uint32)) {
	for f, v := range fields {
		kind, id, ok := strings.Cut(f, ":")
		if !ok {
			continue
		}
		item, err := strconv.ParseInt(id, 10, 32)
		n, err2 := strconv.ParseUint(v, 10, 32)
		if err != nil || err2 != nil {
			continue
		}
		c := items[int32(item)]
		imp, clk := slots(&c)
		switch kind {
		case Impression:
			*imp += uint32(n)
		case Click:
			*clk += uint32(n)
		default:
			continue
		}
		items[int32(item)] = c
	}
}

// Refresher keeps the current table of one api process.
type Refresher struct {
	R   Reader // the state instance
	Log *slog.Logger
	Now func() time.Time

	cur       atomic.Pointer[Table]
	lastError atomic.Int64 // unix second of the last logged failure
}

// Current is the newest table (nil before the first build).
func (f *Refresher) Current() *Table { return f.cur.Load() }

// Set installs a table (tests, and the first build at start).
func (f *Refresher) Set(t *Table) { f.cur.Store(t) }

func (f *Refresher) now() time.Time {
	if f.Now != nil {
		return f.Now()
	}
	return time.Now()
}

// Refresh builds the table for AsOfAt(now) unless the current one already is that table (same
// E, same seed). Reports whether it built one.
func (f *Refresher) Refresh(ctx context.Context) (bool, error) {
	now := f.now()
	asOf := AsOfAt(now)
	since, err := SeededAt(ctx, f.R)
	if err != nil {
		return false, err
	}
	if cur := f.cur.Load(); cur != nil && cur.AsOf.Equal(asOf) && cur.Since.Equal(since) {
		return false, nil
	}
	t, err := Build(ctx, f.R, asOf, since, now)
	if err != nil {
		return false, err
	}
	f.cur.Store(t)
	return true, nil
}

// Run refreshes every RefreshEvery until ctx ends.
func (f *Refresher) Run(ctx context.Context) {
	tick := time.NewTicker(RefreshEvery)
	defer tick.Stop()
	for {
		built, err := f.Refresh(ctx)
		switch {
		case err != nil && ctx.Err() == nil:
			metrics.ItemCTRBuilds.WithLabelValues("error").Inc()
			if now := time.Now().Unix(); f.lastError.Swap(now) != now && f.Log != nil {
				f.Log.Warn("item CTR table: build failed, keeping the last one", "err", err)
			}
		case built:
			metrics.ItemCTRBuilds.WithLabelValues("ok").Inc()
		}
		if t := f.cur.Load(); t != nil {
			metrics.ItemCTRAge.Set(f.now().Sub(t.AsOf).Seconds())
		}
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
		}
	}
}
