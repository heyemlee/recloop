package mixer

import (
	"context"
	"errors"
	"math"
	"testing"
	"time"

	"github.com/heyemlee/recloop/serving/internal/recall"
	"github.com/heyemlee/recloop/serving/internal/resilience"
)

func ids(fs []Fused) []int32 {
	out := make([]int32, len(fs))
	for i, f := range fs {
		out[i] = f.ItemID
	}
	return out
}

func equal(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func cands(items ...int32) []recall.Candidate {
	out := make([]recall.Candidate, len(items))
	for i, id := range items {
		out[i] = recall.Candidate{ItemID: id, Score: float64(len(items) - i), Via: -1}
	}
	return out
}

func TestRRFWeightsAndOrder(t *testing.T) {
	results := []Result{
		{Source: "sasrec", Items: cands(1, 2, 3)},
		{Source: "itemcf", Items: cands(3, 1)},
		{Source: "pop", Items: cands(9)},
		{Source: "broken", Items: cands(7), Err: errors.New("down")},
	}
	got := RRF(results, map[string]float64{"sasrec": 1, "itemcf": 2, "pop": 0.5, "broken": 1})
	want := map[int32]float64{
		1: 1.0/61 + 2.0/62,
		2: 1.0 / 62,
		3: 1.0/63 + 2.0/61,
		9: 0.5 / 61,
	}
	if len(got) != len(want) {
		t.Fatalf("got %v", ids(got))
	}
	for _, f := range got {
		if math.Abs(f.Score-want[f.ItemID]) > 1e-12 {
			t.Errorf("item %d: score %v, want %v", f.ItemID, f.Score, want[f.ItemID])
		}
	}
	if !equal(ids(got), []int32{3, 1, 2, 9}) {
		t.Fatalf("order %v", ids(got))
	}
	if got[0].Primary != "itemcf" || got[1].Primary != "itemcf" || got[2].Primary != "sasrec" {
		t.Fatalf("primary sources %s %s %s", got[0].Primary, got[1].Primary, got[2].Primary)
	}
	if len(got[0].Hits) != 2 || got[0].Hits[0].Source != "sasrec" || got[0].Hits[0].Rank != 3 {
		t.Fatalf("hits of item 3: %+v", got[0].Hits)
	}
}

func TestRRFTiesAndZeroWeight(t *testing.T) {
	results := []Result{{Source: "a", Items: cands(5)}, {Source: "b", Items: cands(4)}, {Source: "c", Items: cands(8)}}
	got := RRF(results, map[string]float64{"a": 1, "b": 1, "c": 0})
	// equal scores and equal best rank: lower item id first; weight 0 drops a source
	if !equal(ids(got), []int32{4, 5}) {
		t.Fatalf("order %v", ids(got))
	}
}

func TestFilterReasons(t *testing.T) {
	in := RRF([]Result{{Source: "s", Items: cands(1, 2, 3, 4, 5, 6)}}, map[string]float64{"s": 1})
	ex := Exclusions{
		Bought:  map[int32]bool{2: true},
		Seen:    map[int32]bool{3: true},
		Seed:    4,
		Known:   func(id int32) bool { return id != 6 },
		InStock: func(id int32) bool { return id != 5 },
	}
	got, st := Filter(in, ex)
	if !equal(ids(got), []int32{1}) {
		t.Fatalf("kept %v", ids(got))
	}
	if st != (FilterStats{Bought: 1, Seen: 1, SoldOut: 1, Seed: 1, Unknown: 1}) {
		t.Fatalf("stats %+v", st)
	}
}

func TestDiversifyCapsBrandsInTheWindow(t *testing.T) {
	brand := map[int32]string{1: "A", 2: "A", 3: "A", 4: "B", 5: "", 6: "", 7: "", 8: "A"}
	in := RRF([]Result{{Source: "s", Items: cands(1, 2, 3, 8, 4, 5, 6, 7)}}, map[string]float64{"s": 1})
	got, promoted := Diversify(in, func(id int32) string { return brand[id] }, 4, 2)
	// window 4: A, A, then 3 and 8 (A) are capped -> B (4) and 5 move up
	if !equal(ids(got), []int32{1, 2, 4, 5, 3, 8, 6, 7}) {
		t.Fatalf("order %v", ids(got))
	}
	if promoted != 2 {
		t.Fatalf("promoted %d", promoted)
	}
	countA := 0
	for _, f := range got[:4] {
		if brand[f.ItemID] == "A" {
			countA++
		}
	}
	if countA > 2 {
		t.Fatalf("%d items of brand A in the window", countA)
	}
	// unknown brands are not capped
	in2 := RRF([]Result{{Source: "s", Items: cands(5, 6, 7)}}, map[string]float64{"s": 1})
	got2, p2 := Diversify(in2, func(id int32) string { return brand[id] }, 3, 1)
	if !equal(ids(got2), []int32{5, 6, 7}) || p2 != 0 {
		t.Fatalf("unknown brands: %v promoted %d", ids(got2), p2)
	}
}

func TestDiversifyKeepsOrderWhenOnlyOneBrandIsLeft(t *testing.T) {
	in := RRF([]Result{{Source: "s", Items: cands(1, 2, 3)}}, map[string]float64{"s": 1})
	got, _ := Diversify(in, func(int32) string { return "A" }, 3, 1)
	if !equal(ids(got), []int32{1, 2, 3}) {
		t.Fatalf("order %v", ids(got))
	}
}

func TestBackfillSkipsPresentAndExcluded(t *testing.T) {
	list := RRF([]Result{{Source: "s", Items: cands(1, 2)}}, map[string]float64{"s": 1})
	ex := Exclusions{Bought: map[int32]bool{3: true}, Seed: -1}
	got, added := Backfill(list, []int32{2, 3, 4, 5, 6}, 4, ex)
	if !equal(ids(got), []int32{1, 2, 4, 5}) || added != 2 {
		t.Fatalf("got %v added %d", ids(got), added)
	}
	if got[2].Primary != "backfill" || got[2].Hits[0].Rank != 3 {
		t.Fatalf("backfill item %+v", got[2])
	}
}

type fakeSource struct {
	name  string
	delay time.Duration
	err   error
	items []recall.Candidate
	block bool // ignore ctx: a misbehaving source
}

func (f fakeSource) Name() string { return f.name }

func (f fakeSource) Recall(ctx context.Context, _ recall.Request) (recall.Output, error) {
	if f.block {
		time.Sleep(f.delay)
		return recall.Output{Items: f.items}, nil
	}
	select {
	case <-time.After(f.delay):
		return recall.Output{Items: f.items, Version: "v1"}, f.err
	case <-ctx.Done():
		return recall.Output{}, ctx.Err()
	}
}

func TestFanOutReturnsPartialResults(t *testing.T) {
	specs := []Spec{
		{Source: fakeSource{name: "fast", items: cands(1, 2)}, Timeout: 50 * time.Millisecond, K: 10},
		{Source: fakeSource{name: "slow", delay: time.Second, items: cands(3)}, Timeout: 30 * time.Millisecond, K: 10},
		{Source: fakeSource{name: "stuck", delay: time.Second, block: true}, Timeout: 40 * time.Millisecond, K: 10},
		{Source: fakeSource{name: "failing", err: errors.New("boom")}, Timeout: 50 * time.Millisecond, K: 10},
	}
	t0 := time.Now()
	res := FanOut(context.Background(), specs, recall.Request{}, time.Second)
	if took := time.Since(t0); took > 200*time.Millisecond {
		t.Fatalf("fan-out waited %v for slow sources", took)
	}
	if res[0].Err != nil || len(res[0].Items) != 2 || res[0].Version != "v1" {
		t.Fatalf("fast: %+v", res[0])
	}
	for _, i := range []int{1, 2} {
		if !errors.Is(res[i].Err, ErrTimeout) || len(res[i].Items) != 0 || res[i].Status != StatusTimeout {
			t.Fatalf("%s: %+v", res[i].Source, res[i])
		}
	}
	if res[3].Err == nil || errors.Is(res[3].Err, ErrTimeout) || res[3].Items != nil || res[3].Status != StatusError {
		t.Fatalf("failing: %+v", res[3])
	}
	fused := RRF(res, map[string]float64{"fast": 1, "slow": 1, "stuck": 1, "failing": 1})
	if !equal(ids(fused), []int32{1, 2}) {
		t.Fatalf("merged %v", ids(fused))
	}
}

func TestFanOutTakesTheStageBudget(t *testing.T) {
	specs := []Spec{
		{Source: fakeSource{name: "remote", delay: 30 * time.Millisecond, items: cands(1)}, Timeout: 35 * time.Millisecond, MinTimeout: 5 * time.Millisecond, K: 10},
		{Source: fakeSource{name: "local", items: cands(2)}, Timeout: 10 * time.Millisecond, K: 10},
	}
	// Only 12 ms are left for recall: the 30 ms source times out at 12 ms, not at its own 35.
	t0 := time.Now()
	res := FanOut(context.Background(), specs, recall.Request{}, 12*time.Millisecond)
	if took := time.Since(t0); took > 25*time.Millisecond {
		t.Fatalf("waited %v with a 12ms budget", took)
	}
	if res[0].Status != StatusTimeout || res[1].Status != StatusOK {
		t.Fatalf("statuses %s %s", res[0].Status, res[1].Status)
	}
	// 3 ms left: below the remote source's minimum, it is not called at all.
	res = FanOut(context.Background(), specs, recall.Request{}, 3*time.Millisecond)
	if res[0].Status != StatusSkipped || res[1].Status != StatusOK {
		t.Fatalf("statuses %s %s", res[0].Status, res[1].Status)
	}
}

func TestOpenBreakerSkipsTheSource(t *testing.T) {
	b := resilience.NewBreaker("remote", resilience.BreakerConfig{Window: 4, FailureRatio: 0.5, OpenFor: time.Minute, HalfOpenProbes: 1}, nil)
	failing := []Spec{{Source: fakeSource{name: "remote", err: errors.New("unavailable")}, Timeout: 20 * time.Millisecond, K: 10, Breaker: b}}
	for i := 0; i < 4; i++ {
		FanOut(context.Background(), failing, recall.Request{}, time.Second)
	}
	if b.State() != resilience.Open {
		t.Fatalf("breaker %v after 4 failures", b.State())
	}
	t0 := time.Now()
	slow := []Spec{{Source: fakeSource{name: "remote", delay: time.Second}, Timeout: 30 * time.Millisecond, K: 10, Breaker: b}}
	res := FanOut(context.Background(), slow, recall.Request{}, time.Second)
	if res[0].Status != StatusOpen || time.Since(t0) > 5*time.Millisecond {
		t.Fatalf("open breaker: %+v after %v", res[0], time.Since(t0))
	}
}

func TestNaiveFanOutWaitsForEverySource(t *testing.T) {
	specs := []Spec{
		{Source: fakeSource{name: "slow", delay: 60 * time.Millisecond, items: cands(1)}, Timeout: 10 * time.Millisecond, K: 10},
		{Source: fakeSource{name: "failing", err: errors.New("boom")}, Timeout: 10 * time.Millisecond, K: 10},
	}
	t0 := time.Now()
	res := FanOutNaive(context.Background(), specs, recall.Request{})
	if took := time.Since(t0); took < 60*time.Millisecond {
		t.Fatalf("naive fan-out returned after %v, before the slow source", took)
	}
	if res[0].Status != StatusOK || len(res[0].Items) != 1 || res[1].Status != StatusError {
		t.Fatalf("results %+v", res)
	}
}
