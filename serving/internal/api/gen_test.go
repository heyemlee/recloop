package api_test

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/heyemlee/recloop/serving/internal/api"
	"github.com/heyemlee/recloop/serving/internal/experiment"
	"github.com/heyemlee/recloop/serving/internal/mixer"
	"github.com/heyemlee/recloop/serving/internal/recall"
	"github.com/heyemlee/recloop/serving/internal/userstate"
)

// The nearline gen source (design §11 step 5): an arm whose recall params give "gen" a weight
// and a "gen_version" reads the user's generation result with the user context, fuses it as a
// fourth source, and keys its candidate cache by the result's history version; every other arm
// reads nothing and keys its cache as before.

type genView struct {
	cache string
	items []api.RecItem
	debug struct {
		Gen *struct {
			Version string `json:"ver"`
			Status  string `json:"status"`
			HV      int64  `json:"hv"`
			Lag     int64  `json:"lag"`
			Items   int    `json:"items"`
		} `json:"gen"`
		Sources []struct {
			Source     string  `json:"source"`
			Weight     float64 `json:"weight"`
			Candidates int     `json:"candidates"`
		} `json:"sources"`
	}
}

func genCall(t *testing.T, s *api.Server, user string) genView {
	t.Helper()
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/recommendations?user_id="+user+"&limit=20&debug=1", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	var body struct {
		Items []api.RecItem `json:"items"`
		Debug json.RawMessage
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	v := genView{cache: rec.Header().Get("X-Candidate-Cache"), items: body.Items}
	if err := json.Unmarshal(body.Debug, &v.debug); err != nil {
		t.Fatal(err)
	}
	return v
}

// genServer is memServer with the gen source and, when params is set, a recall experiment that
// puts every user into one variant with those params.
func genServer(t *testing.T, rdb redis.UniversalClient, params string) *api.Server {
	t.Helper()
	s := memServer(t, rdb, &fakeModel{delay: time.Millisecond}, false)
	s.Weights[recall.SurfaceHome]["gen"] = 0
	s.Sources[recall.SurfaceHome] = append(s.Sources[recall.SurfaceHome],
		mixer.Spec{Source: recall.Gen{}, Timeout: 10 * time.Millisecond, K: 50})
	if params != "" {
		exps := []experiment.Experiment{{
			ID: "recall-gen-test", Layer: "recall", From: 0, To: experiment.Buckets, ConfigVersion: 3,
			Variants: []experiment.Variant{{Name: "gen", Share: 100, Params: json.RawMessage(params)}},
		}}
		log := slog.New(slog.NewTextHandler(io.Discard, nil))
		s.Exps = experiment.NewLoader(func(context.Context) ([]experiment.Experiment, error) { return exps, nil }, log)
		if err := s.Exps.Refresh(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	return s
}

func putGen(t *testing.T, rdb redis.UniversalClient, uid int64, ver string, g userstate.Generated) {
	t.Helper()
	raw, err := json.Marshal(g)
	if err != nil {
		t.Fatal(err)
	}
	if err := rdb.Set(context.Background(), userstate.GenKey(uid, ver), raw, 0).Err(); err != nil {
		t.Fatal(err)
	}
}

func hasSource(items []api.RecItem, src string) bool {
	for _, it := range items {
		for _, h := range it.Sources {
			if h.Source == src {
				return true
			}
		}
	}
	return false
}

func TestGenArmFusesTheResultAndKeysTheCacheByIt(t *testing.T) {
	_, rdb := miniRedis(t)
	s := genServer(t, rdb, `{"rrf_weights": {"gen": 1.0}, "gen_version": "v1"}`)
	seedHistory(t, rdb, 9) // hv 2

	first := genCall(t, s, "9")
	if first.cache != api.CacheMiss || first.debug.Gen == nil || first.debug.Gen.Status != api.GenMissing {
		t.Fatalf("no result yet: cache %s, gen %+v", first.cache, first.debug.Gen)
	}
	if hasSource(first.items, "gen") {
		t.Fatal("gen hits without a result")
	}
	if again := genCall(t, s, "9"); again.cache != api.CacheHit {
		t.Fatalf("same state: cache %s", again.cache)
	}

	// The worker writes a result for hv 2: the cache key changes, the next request uses it.
	putGen(t, rdb, 9, "v1", userstate.Generated{HV: 2, At: time.Now().UnixMilli(), Items: []int32{25, 26, 27}, Scores: []float64{-0.1, -0.2, -0.3}})
	fresh := genCall(t, s, "9")
	if fresh.cache != api.CacheMiss || fresh.debug.Gen.Status != api.GenFresh || fresh.debug.Gen.Items != 3 {
		t.Fatalf("after the write: cache %s, gen %+v", fresh.cache, fresh.debug.Gen)
	}
	if !hasSource(fresh.items, "gen") {
		t.Fatalf("no item came from gen: %+v", fresh.items)
	}
	var genSrc bool
	for _, src := range fresh.debug.Sources {
		if src.Source == "gen" {
			genSrc = src.Weight == 1 && src.Candidates == 3
		}
	}
	if !genSrc {
		t.Fatalf("debug sources: %+v", fresh.debug.Sources)
	}

	// A click bumps hv to 3: the result is one version behind until the worker answers.
	st := userstate.Store{R: rdb}
	if _, err := st.Apply(context.Background(), userstate.Change{EventID: "click-g", UserID: 9, ItemID: 20, Kind: userstate.KindHistory, At: time.Now()}, time.Now()); err != nil {
		t.Fatal(err)
	}
	stale := genCall(t, s, "9")
	if stale.cache != api.CacheMiss || stale.debug.Gen.Status != api.GenStale || stale.debug.Gen.Lag != 1 {
		t.Fatalf("after a click: cache %s, gen %+v", stale.cache, stale.debug.Gen)
	}
	if !hasSource(stale.items, "gen") {
		t.Fatal("a stale result is still used")
	}
}

func TestArmWithoutGenReadsNothing(t *testing.T) {
	_, rdb := miniRedis(t)
	for _, params := range []string{
		"", // the default config: no recall experiment
		`{"rrf_weights": {"sasrec": 1.5}}`,
		`{"rrf_weights": {"gen": 0}, "gen_version": "v1"}`, // weight 0: not read
		`{"rrf_weights": {"gen": 1.0}}`,                    // no version: nothing to read
	} {
		s := genServer(t, rdb, params)
		uid := int64(11)
		if err := rdb.FlushDB(context.Background()).Err(); err != nil {
			t.Fatal(err)
		}
		seedHistory(t, rdb, uid)
		putGen(t, rdb, uid, "v1", userstate.Generated{HV: 2, At: time.Now().UnixMilli(), Items: []int32{25}, Scores: []float64{-0.1}})
		r := genCall(t, s, "11")
		if r.debug.Gen != nil || hasSource(r.items, "gen") {
			t.Fatalf("params %q: gen %+v", params, r.debug.Gen)
		}
		if again := genCall(t, s, "11"); again.cache != api.CacheHit {
			t.Fatalf("params %q: cache %s on the second request", params, again.cache)
		}
	}
}
