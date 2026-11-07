package api_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/heyemlee/recloop/serving/internal/api"
	"github.com/heyemlee/recloop/serving/internal/bundle"
	"github.com/heyemlee/recloop/serving/internal/cache"
	"github.com/heyemlee/recloop/serving/internal/catalog"
	"github.com/heyemlee/recloop/serving/internal/events"
	"github.com/heyemlee/recloop/serving/internal/experiment"
	"github.com/heyemlee/recloop/serving/internal/mixer"
	"github.com/heyemlee/recloop/serving/internal/recall"
	"github.com/heyemlee/recloop/serving/internal/resilience"
	"github.com/heyemlee/recloop/serving/internal/store"
	"github.com/heyemlee/recloop/serving/internal/store/db"
	"github.com/heyemlee/recloop/serving/internal/testutil"
	"github.com/heyemlee/recloop/serving/internal/userstate"
)

const nItems = 40

// testPlan is generous: the integration tests check behaviour, not latency.
var testPlan = resilience.Plan{Total: 2 * time.Second, UserContext: 500 * time.Millisecond, CacheGet: 200 * time.Millisecond,
	Recall: 500 * time.Millisecond, Rank: 100 * time.Millisecond, Tail: 500 * time.Millisecond}

// fixture: an api on the test database and Redis DB 15, with in-process sources only.
type fixture struct {
	t      *testing.T
	st     *store.Store
	srv    *httptest.Server
	served *events.ServedWriter
	cat    *catalog.Catalog
	cancel context.CancelFunc
}

func newFixture(t *testing.T) *fixture {
	st := testutil.PG(t)
	rdb := testutil.Redis(t)
	testutil.SeedItems(t, st, nItems, func(i int) int {
		if i == nItems-1 {
			return 0 // sold out
		}
		return 3
	})
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	ctx := context.Background()
	cat, err := catalog.Load(ctx, st, log)
	if err != nil {
		t.Fatal(err)
	}
	// A small bundle: popularity = item id order, ItemCF neighbours i -> i+1..i+5.
	b := &bundle.Bundle{Version: "test-bundle-v1", NumItems: nItems, ItemCF: bundle.ItemCF{HistoryDecay: 0.8}}
	for i := 0; i < nItems; i++ {
		b.Pop = append(b.Pop, int32(i))
		b.PopCounts = append(b.PopCounts, int32(nItems-i))
		var ns []int32
		var ss []float32
		for d := 1; d <= 5; d++ {
			ns = append(ns, int32((i+d)%nItems))
			ss = append(ss, float32(1.0/float64(d)))
		}
		b.ItemCF.Neighbors = append(b.ItemCF.Neighbors, ns)
		b.ItemCF.Scores = append(b.ItemCF.Scores, ss)
	}
	exps := experiment.NewLoader(st.RunningExperiments, log)
	if err := exps.Refresh(ctx); err != nil {
		t.Fatal(err)
	}
	bg, cancel := context.WithCancel(context.Background())
	served := events.NewServedWriter(events.PGServed{DB: st}, 128, log)
	go served.Run(bg, 50, 10*time.Millisecond)
	users := userstate.Store{R: rdb}
	s := &api.Server{
		Catalog: cat, Bundle: b, Users: users, Pages: cache.Pages{R: rdb}, Exps: exps,
		Cands:  &cache.CandidateCache{R: rdb},
		Events: &events.Direct{DB: st, State: users, OnStock: cat.SetStock, Now: time.Now},
		Served: served, Redis: rdb, Plan: testPlan, Campaigns: st,
		Sources: map[string][]mixer.Spec{
			recall.SurfaceHome: {
				{Source: recall.ItemCF{B: b}, Timeout: 50 * time.Millisecond, K: 200},
				{Source: recall.Pop{B: b}, Timeout: 50 * time.Millisecond, K: 200},
			},
			recall.SurfaceItem: {{Source: recall.ItemCF{B: b}, Timeout: 50 * time.Millisecond, K: 100}},
		},
		Weights: map[string]map[string]float64{
			recall.SurfaceHome: {"sasrec": 1, "itemcf": 1, "pop": 0.3},
			recall.SurfaceItem: {"itemcf": 1, "text": 1},
		},
		Log: log, Now: time.Now,
	}
	f := &fixture{t: t, st: st, srv: httptest.NewServer(s.Handler()), served: served, cat: cat, cancel: cancel}
	t.Cleanup(func() {
		f.srv.Close()
		cancel()
		served.Wait()
	})
	return f
}

func (f *fixture) get(path string, out any) int {
	f.t.Helper()
	resp, err := http.Get(f.srv.URL + path)
	if err != nil {
		f.t.Fatal(err)
	}
	defer resp.Body.Close()
	if out != nil {
		if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
			f.t.Fatal(err)
		}
	}
	return resp.StatusCode
}

type ingestResp struct {
	Accepted   int `json:"accepted"`
	Duplicates int `json:"duplicates"`
	Rejected   []struct {
		Index int    `json:"index"`
		Error string `json:"error"`
	} `json:"rejected"`
}

func (f *fixture) post(evs ...map[string]any) ingestResp {
	f.t.Helper()
	body, _ := json.Marshal(map[string]any{"events": evs})
	resp, err := http.Post(f.srv.URL+"/v1/events", "application/json", bytes.NewReader(body))
	if err != nil {
		f.t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusAccepted {
		raw, _ := io.ReadAll(resp.Body)
		f.t.Fatalf("POST /v1/events: %d %s", resp.StatusCode, raw)
	}
	var out ingestResp
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		f.t.Fatal(err)
	}
	return out
}

func (f *fixture) count(sql string, args ...any) int {
	f.t.Helper()
	var n int
	if err := f.st.Pool.QueryRow(context.Background(), sql, args...).Scan(&n); err != nil {
		f.t.Fatal(err)
	}
	return n
}

func event(typ string, user, item int, impression string) map[string]any {
	ev := map[string]any{"event_id": uuid.Must(uuid.NewV7()).String(), "type": typ, "user_id": user, "item_id": item}
	if impression != "" {
		ev["impression_id"] = impression
	}
	return ev
}

func TestEventsAreIdempotent(t *testing.T) {
	f := newFixture(t)
	click := event("click", 7, 3, uuid.Must(uuid.NewV7()).String()+":2")
	buy := event("purchase", 7, 5, "")

	first := f.post(click, buy)
	if first.Accepted != 2 || first.Duplicates != 0 || len(first.Rejected) != 0 {
		t.Fatalf("first delivery: %+v", first)
	}
	// Retry of the same batch, plus the same click twice in one batch.
	second := f.post(click, buy, click)
	if second.Accepted != 0 || second.Duplicates != 3 {
		t.Fatalf("redelivery: %+v", second)
	}
	if n := f.count("SELECT count(*) FROM events_raw WHERE event_id = $1", click["event_id"]); n != 1 {
		t.Fatalf("click stored %d times", n)
	}
	if n := f.count("SELECT count(*) FROM orders WHERE event_id = $1", buy["event_id"]); n != 1 {
		t.Fatalf("%d orders for one purchase event", n)
	}
	if n := f.count("SELECT stock FROM items WHERE item_id = 5"); n != 2 {
		t.Fatalf("stock of item 5 is %d, want 3 - 1", n)
	}
	if f.cat.Stock(5) != 2 {
		t.Fatalf("catalog stock %d", f.cat.Stock(5))
	}
	var uc struct {
		HV      int64 `json:"hv"`
		History []struct {
			ItemID int32 `json:"item_id"`
		} `json:"history"`
		Bought int `json:"bought"`
	}
	f.get("/v1/users/7/context", &uc)
	if uc.HV != 2 || len(uc.History) != 2 || uc.History[0].ItemID != 5 || uc.Bought != 1 {
		t.Fatalf("user state after redelivery: %+v", uc)
	}
	// event_day comes from the UUIDv7 time.
	if n := f.count("SELECT count(*) FROM events_raw WHERE event_id = $1 AND event_day = (to_timestamp(($2::bigint)/1000.0) AT TIME ZONE 'UTC')::date",
		click["event_id"], events.V7Time(uuid.MustParse(click["event_id"].(string))).UnixMilli()); n != 1 {
		t.Fatal("event_day is not the day of the event_id")
	}

	bad := f.post(event("view", 7, 3, ""), event("click", 7, 999, ""))
	if bad.Accepted != 0 || len(bad.Rejected) != 2 {
		t.Fatalf("invalid events: %+v", bad)
	}
}

type recResp struct {
	RequestID    string                           `json:"request_id"`
	ModelVersion string                           `json:"model_version"`
	Experiments  map[string]experiment.Assignment `json:"experiments"`
	Degraded     []string                         `json:"degraded"`
	Items        []api.RecItem                    `json:"items"`
	NextCursor   string                           `json:"next_cursor"`
	Debug        *api.Debug                       `json:"debug"`
}

func TestClickChangesTheNextRecommendations(t *testing.T) {
	f := newFixture(t)
	var before recResp
	if code := f.get("/v1/recommendations?user_id=11&limit=10&debug=1", &before); code != 200 {
		t.Fatalf("status %d", code)
	}
	if before.Debug.HV != 0 || len(before.Debug.History) != 0 {
		t.Fatalf("new user has history: %+v", before.Debug)
	}
	for _, it := range before.Items {
		if it.Source != "pop" {
			t.Fatalf("cold user got %s candidates", it.Source)
		}
	}
	f.post(event("click", 11, 20, before.Items[0].ImpressionID))

	var after recResp
	f.get("/v1/recommendations?user_id=11&limit=10&debug=1", &after)
	if after.Debug.HV != 1 || after.Debug.History[0].ItemID != 20 {
		t.Fatalf("history not updated: %+v", after.Debug)
	}
	viaClick := 0
	for _, it := range after.Items {
		for _, h := range it.Sources {
			if h.Source == "itemcf" && h.Via == 20 {
				viaClick++
			}
		}
	}
	if viaClick == 0 || after.Items[0].ItemID != 21 {
		t.Fatalf("neighbours of the clicked item are missing: %+v", after.Items)
	}
}

func TestServedLogsAndPaging(t *testing.T) {
	f := newFixture(t)
	var page1 recResp
	f.get("/v1/recommendations?user_id=3&limit=15&debug=1", &page1)
	if len(page1.Items) != 15 || page1.NextCursor == "" || page1.ModelVersion != "test-bundle-v1" {
		t.Fatalf("page 1: %d items, cursor %q", len(page1.Items), page1.NextCursor)
	}
	if page1.Experiments["recall"].ExpID != "recall-rrf-weights-v1" {
		t.Fatalf("assignment %+v", page1.Experiments)
	}
	for _, it := range page1.Items {
		if it.ItemID == nItems-1 {
			t.Fatal("sold-out item recommended")
		}
	}
	var page2 recResp
	f.get("/v1/recommendations?user_id=3&limit=15&cursor="+page1.NextCursor, &page2)
	if page2.RequestID != page1.RequestID || len(page2.Items) == 0 || page2.Items[0].Position != 16 {
		t.Fatalf("page 2: %+v", page2)
	}
	seen := map[int32]bool{}
	for _, it := range append(page1.Items, page2.Items...) {
		if seen[it.ItemID] {
			t.Fatalf("item %d on two pages", it.ItemID)
		}
		seen[it.ItemID] = true
	}
	if code := f.get("/v1/recommendations?user_id=4&cursor="+page1.NextCursor, nil); code != 400 {
		t.Fatalf("another user's cursor: status %d", code)
	}

	// Both pages are logged as served events with the experiment groups and model version.
	deadline := time.Now().Add(3 * time.Second)
	for f.count("SELECT count(*) FROM events_raw WHERE type = 'served' AND request_id = $1", page1.RequestID) < 2 {
		if time.Now().After(deadline) {
			t.Fatal("served events not written")
		}
		time.Sleep(20 * time.Millisecond)
	}
	n := f.count(`SELECT count(*) FROM events_raw
		WHERE type = 'served' AND request_id = $1
		  AND payload->>'model_ver' = 'test-bundle-v1'
		  AND payload->'exp'->'recall'->>'variant' IN ('control', 'sasrec_heavy')
		  AND (payload->>'config_version')::int = 1
		  AND jsonb_array_length(payload->'items') = 15`, page1.RequestID)
	if n != 2 {
		t.Fatalf("%d attributable served rows", n)
	}
	if offset := f.count("SELECT max((payload->>'offset')::int) FROM events_raw WHERE request_id = $1", page1.RequestID); offset != 15 {
		t.Fatalf("page 2 offset %d", offset)
	}

	// The database refuses a served row without attribution.
	bad, _ := events.Served(3, uuid.Must(uuid.NewV7()), []byte(`{"exp": {}}`))
	if err := f.st.InsertServed(context.Background(), []db.InsertEventParams{events.Params(bad)}); err == nil {
		t.Fatal("served event without model_ver was accepted")
	}
}

// The default recall-layer experiment (migration 00003) has to reach fusion: each variant's
// rrf_weights replace the surface defaults, and the fused scores are computed with them.
// Neither the SRM check nor an A/A test would notice variant parameters that are silently
// ignored; the A/B would then compare two identical arms.
func TestRecallVariantsFuseWithTheirWeights(t *testing.T) {
	f := newFixture(t)
	var raw []byte
	if err := f.st.Pool.QueryRow(context.Background(),
		"SELECT variants FROM experiments WHERE exp_id = 'recall-rrf-weights-v1'").Scan(&raw); err != nil {
		t.Fatal(err)
	}
	var variants []struct {
		Name   string `json:"name"`
		Params struct {
			RRFWeights map[string]float64 `json:"rrf_weights"`
		} `json:"params"`
	}
	if err := json.Unmarshal(raw, &variants); err != nil {
		t.Fatal(err)
	}
	defaults := map[string]float64{"sasrec": 1, "itemcf": 1, "pop": 0.3} // the fixture's home weights
	want := map[string]map[string]float64{}
	for _, v := range variants {
		w := maps.Clone(defaults)
		for src := range w {
			if x, ok := v.Params.RRFWeights[src]; ok {
				w[src] = x
			}
		}
		want[v.Name] = w
	}
	if len(want) != 2 || maps.Equal(want["control"], want["sasrec_heavy"]) {
		t.Fatalf("the default experiment should compare two different weightings: %v", want)
	}

	checked := map[string]bool{}
	for uid := 100; len(checked) < len(want) && uid < 200; uid++ {
		f.post(event("click", uid, 5, "")) // a history, so that ItemCF (weighted differently) contributes
		var r recResp
		f.get(fmt.Sprintf("/v1/recommendations?user_id=%d&limit=20&debug=1", uid), &r)
		variant := r.Experiments["recall"].Variant
		w, ok := want[variant]
		if !ok {
			t.Fatalf("user %d: variant %q is not in the experiment", uid, variant)
		}
		if !maps.Equal(r.Debug.Weights, w) {
			t.Fatalf("user %d (%s): weights %v, want %v", uid, variant, r.Debug.Weights, w)
		}
		fromItemCF := 0
		for _, it := range r.Items {
			if it.Slot != "organic" {
				continue
			}
			score := 0.0
			for _, h := range it.Sources {
				score += w[h.Source] / float64(mixer.RRFK+h.Rank)
				if h.Source == "itemcf" {
					fromItemCF++
				}
			}
			if math.Abs(it.Score-score) > 1e-12 {
				t.Fatalf("user %d (%s): item %d scored %v, want %v from %+v", uid, variant, it.ItemID, it.Score, score, it.Sources)
			}
		}
		if fromItemCF == 0 {
			t.Fatalf("user %d: no ItemCF candidates on the page", uid)
		}
		checked[variant] = true
	}
	if len(checked) != len(want) {
		t.Fatalf("users 100-199 only reached %v", checked)
	}
}
