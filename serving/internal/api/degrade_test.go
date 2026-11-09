package api_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"slices"
	"sync/atomic"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"

	"github.com/heyemlee/recloop/serving/internal/api"
	"github.com/heyemlee/recloop/serving/internal/bundle"
	"github.com/heyemlee/recloop/serving/internal/cache"
	"github.com/heyemlee/recloop/serving/internal/catalog"
	"github.com/heyemlee/recloop/serving/internal/events"
	"github.com/heyemlee/recloop/serving/internal/experiment"
	"github.com/heyemlee/recloop/serving/internal/mixer"
	"github.com/heyemlee/recloop/serving/internal/recall"
	"github.com/heyemlee/recloop/serving/internal/resilience"
	"github.com/heyemlee/recloop/serving/internal/userstate"
)

// These tests run the recommendation handler in memory (no PostgreSQL, no listening port for
// the api) against fake recall sources, to check the degradation chain of design §5.1.

// fakeModel stands in for the SASRec source: it answers after `delay` or fails.
type fakeModel struct {
	delay time.Duration
	fail  atomic.Bool
	calls atomic.Int64
}

func (*fakeModel) Name() string { return "sasrec" }

// Applicable like recall.SASRec: nothing to encode without history.
func (*fakeModel) Applicable(req recall.Request) bool { return len(req.History) > 0 }

func (f *fakeModel) Recall(ctx context.Context, req recall.Request) (recall.Output, error) {
	f.calls.Add(1)
	if f.fail.Load() {
		return recall.Output{}, errors.New("unavailable")
	}
	select {
	case <-time.After(f.delay):
	case <-ctx.Done():
		return recall.Output{}, ctx.Err()
	}
	out := recall.Output{Version: "test-bundle-v1"}
	for i := 0; i < 30; i++ {
		out.Items = append(out.Items, recall.Candidate{ItemID: int32(39 - i), Score: float64(30 - i), Via: -1})
	}
	return out, nil
}

type nopSink struct{}

func (nopSink) WriteServed(context.Context, []events.Event) error { return nil }

type nopPublisher struct{}

func (nopPublisher) Publish(_ context.Context, evs []events.Event) (events.Result, error) {
	return events.Result{Accepted: len(evs)}, nil
}

func memServer(t *testing.T, rdb redis.UniversalClient, model *fakeModel, naive bool) *api.Server {
	t.Helper()
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	items := make([]catalog.Item, nItems)
	stock := make([]int32, nItems)
	b := &bundle.Bundle{Version: "test-bundle-v1", NumItems: nItems, ItemCF: bundle.ItemCF{HistoryDecay: 0.8}}
	for i := range items {
		items[i] = catalog.Item{ID: int32(i), Title: "item", Brand: []string{"A", "B", "C", "D"}[i%4]}
		stock[i] = 5
		b.Pop = append(b.Pop, int32(i))
		b.PopCounts = append(b.PopCounts, int32(nItems-i))
		b.ItemCF.Neighbors = append(b.ItemCF.Neighbors, []int32{int32((i + 1) % nItems), int32((i + 2) % nItems)})
		b.ItemCF.Scores = append(b.ItemCF.Scores, []float32{1, 0.5})
	}
	exps := experiment.NewLoader(func(context.Context) ([]experiment.Experiment, error) { return nil, nil }, log)
	if err := exps.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	served := events.NewServedWriter(nopSink{}, 64, log)
	ctx, cancel := context.WithCancel(context.Background())
	go served.Run(ctx, 10, time.Millisecond)
	t.Cleanup(func() { cancel(); served.Wait() })

	s := &api.Server{
		Catalog: catalog.New(items, stock), Bundle: b, Users: userstate.Store{R: rdb}, Pages: cache.Pages{R: rdb},
		Cands: &cache.CandidateCache{R: rdb}, Exps: exps, Events: nopPublisher{}, Served: served, Redis: rdb,
		Weights: map[string]map[string]float64{recall.SurfaceHome: {"sasrec": 1, "itemcf": 1, "pop": 0.3}},
		Plan:    resilience.DefaultPlan, Naive: naive, Log: log, Now: time.Now,
	}
	var msBreaker *resilience.Breaker
	if !naive {
		msBreaker = resilience.NewBreaker("model_server", resilience.DefaultBreaker, nil)
		s.RedisBreaker = resilience.NewBreaker("redis", resilience.DefaultBreaker, nil)
	}
	s.Sources = map[string][]mixer.Spec{recall.SurfaceHome: {
		{Source: model, Timeout: 35 * time.Millisecond, MinTimeout: 5 * time.Millisecond, K: 200, Breaker: msBreaker},
		{Source: recall.ItemCF{B: b}, Timeout: 10 * time.Millisecond, K: 200},
		{Source: recall.Pop{B: b}, Timeout: 10 * time.Millisecond, K: 200},
	}}
	return s
}

func miniRedis(t *testing.T) (*miniredis.Miniredis, redis.UniversalClient) {
	t.Helper()
	mr := miniredis.NewMiniRedis()
	if err := mr.Start(); err != nil {
		t.Skipf("cannot start an in-process Redis on a local port: %v", err)
	}
	t.Cleanup(mr.Close)
	return mr, redis.NewClient(&redis.Options{Addr: mr.Addr(), ContextTimeoutEnabled: true, MaxRetries: -1})
}

type memResp struct {
	code     int
	cache    string
	took     time.Duration
	degraded []string
	items    []api.RecItem
	cursor   string
}

func call(t *testing.T, s *api.Server, user string) memResp {
	t.Helper()
	rec := httptest.NewRecorder()
	t0 := time.Now()
	s.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/recommendations?user_id="+user+"&limit=10", nil))
	out := memResp{code: rec.Code, cache: rec.Header().Get("X-Candidate-Cache"), took: time.Since(t0)}
	if rec.Code == http.StatusOK {
		var body api.RecResponse
		if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
			t.Fatal(err)
		}
		out.degraded, out.items, out.cursor = body.Degraded, body.Items, body.NextCursor
	}
	return out
}

func seedHistory(t *testing.T, rdb redis.UniversalClient, uid int64) {
	t.Helper()
	st := userstate.Store{R: rdb}
	now := time.Now()
	for i, item := range []int32{3, 7} {
		c := userstate.Change{EventID: "seed-" + string(rune('a'+i)), UserID: uid, ItemID: item, Kind: userstate.KindHistory, At: now.Add(time.Duration(i) * time.Second)}
		if _, err := st.Apply(context.Background(), c, now); err != nil {
			t.Fatal(err)
		}
	}
}

func TestCandidateCacheHitUntilTheHistoryChanges(t *testing.T) {
	_, rdb := miniRedis(t)
	model := &fakeModel{delay: time.Millisecond}
	s := memServer(t, rdb, model, false)
	seedHistory(t, rdb, 9)

	first := call(t, s, "9")
	if first.code != 200 || first.cache != api.CacheMiss || len(first.degraded) != 0 {
		t.Fatalf("first request: %+v", first)
	}
	second := call(t, s, "9")
	if second.cache != api.CacheHit || model.calls.Load() != 1 {
		t.Fatalf("second request: cache %s, model called %d times", second.cache, model.calls.Load())
	}
	// A click bumps hv: the key changes, nothing had to be invalidated.
	st := userstate.Store{R: rdb}
	if _, err := st.Apply(context.Background(), userstate.Change{EventID: "click-1", UserID: 9, ItemID: 20, Kind: userstate.KindHistory, At: time.Now()}, time.Now()); err != nil {
		t.Fatal(err)
	}
	third := call(t, s, "9")
	if third.cache != api.CacheMiss || model.calls.Load() != 2 {
		t.Fatalf("after a click: cache %s, model called %d times", third.cache, model.calls.Load())
	}
}

func TestSlowModelServerIsCutAtTheBudget(t *testing.T) {
	_, rdb := miniRedis(t)
	model := &fakeModel{delay: 300 * time.Millisecond}
	s := memServer(t, rdb, model, false)
	seedHistory(t, rdb, 4)

	r := call(t, s, "4")
	if r.code != 200 || !slices.Contains(r.degraded, "sasrec_timeout") {
		t.Fatalf("slow model: %+v", r)
	}
	if r.took > 80*time.Millisecond {
		t.Fatalf("waited %v for a model the budget gives 35 ms", r.took)
	}
	if len(r.items) != 10 || r.cursor == "" {
		t.Fatalf("degraded response still pages from ItemCF + Pop: %d items, cursor %q", len(r.items), r.cursor)
	}
	// A degraded candidate list is not cached: the next request tries the model again.
	if call(t, s, "4").cache == api.CacheHit {
		t.Fatal("a degraded result was cached")
	}
}

func TestBreakerOpensAfterModelFailures(t *testing.T) {
	_, rdb := miniRedis(t)
	model := &fakeModel{delay: time.Millisecond}
	model.fail.Store(true)
	s := memServer(t, rdb, model, false)
	s.Cands = nil
	// Every call has to reach the model. A user-context read slower than its 5 ms (a busy CI
	// runner) skips SASRec and breaks the count, so this test gives that stage more room: it
	// tests the breaker, not Redis's timing.
	s.Plan.Total += 100 * time.Millisecond
	s.Plan.UserContext += 100 * time.Millisecond
	seedHistory(t, rdb, 5)
	// The default breaker opens after 10 failures in a row (or half of >= 20 calls in 10 s).
	for i := 0; i < int(resilience.DefaultBreaker.Consecutive); i++ {
		if r := call(t, s, "5"); !slices.Contains(r.degraded, "sasrec_error") {
			t.Fatalf("call %d: %v", i, r.degraded)
		}
	}
	before := model.calls.Load()
	r := call(t, s, "5")
	if !slices.Contains(r.degraded, "sasrec_circuit_open") || model.calls.Load() != before {
		t.Fatalf("breaker did not open: %v, calls %d -> %d", r.degraded, before, model.calls.Load())
	}
}

func TestRedisDownDegradesToPopularity(t *testing.T) {
	// Nothing listens on port 1: every Redis call fails at once.
	rdb := redis.NewClient(&redis.Options{Addr: "127.0.0.1:1", ContextTimeoutEnabled: true, MaxRetries: -1, DialTimeout: 50 * time.Millisecond})
	model := &fakeModel{delay: time.Millisecond}
	s := memServer(t, rdb, model, false)
	r := call(t, s, "6")
	if r.code != 200 || len(r.items) != 10 {
		t.Fatalf("Redis down: %+v", r)
	}
	if !slices.Contains(r.degraded, "user_state_error") && !slices.Contains(r.degraded, "user_state_timeout") {
		t.Fatalf("degraded %v", r.degraded)
	}
	for _, it := range r.items {
		if it.Source != "pop" {
			t.Fatalf("without history the feed is popularity, got %s", it.Source)
		}
	}
	if r.cursor != "" {
		t.Fatal("no page store, but a cursor")
	}
	// The same failure in naive mode is a failed request.
	n := memServer(t, rdb, model, true)
	if r := call(t, n, "6"); r.code != http.StatusServiceUnavailable {
		t.Fatalf("naive mode with Redis down: %d", r.code)
	}
}

func TestNaiveModeWaitsForTheSlowModel(t *testing.T) {
	_, rdb := miniRedis(t)
	model := &fakeModel{delay: 150 * time.Millisecond}
	s := memServer(t, rdb, model, true)
	s.Cands = nil
	seedHistory(t, rdb, 8)
	r := call(t, s, "8")
	if r.code != 200 || len(r.degraded) != 0 || r.took < 150*time.Millisecond {
		t.Fatalf("naive: %+v", r)
	}
	model.fail.Store(true)
	if r := call(t, s, "8"); r.code != http.StatusServiceUnavailable {
		t.Fatalf("naive mode with the model failing: %d", r.code)
	}
}
