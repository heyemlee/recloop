package api_test

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"
	"google.golang.org/grpc"

	pb "github.com/heyemlee/recloop/serving/gen/recloop/modelserver/v1"
	"github.com/heyemlee/recloop/serving/internal/api"
	"github.com/heyemlee/recloop/serving/internal/bundle"
	"github.com/heyemlee/recloop/serving/internal/events"
	"github.com/heyemlee/recloop/serving/internal/ranking"
)

// rankMS stands in for the model server's Rank: score = item id (so LightGBM's order is the
// item ids, highest first), or an error, or a delay longer than the ranking budget.
type rankMS struct {
	pb.ModelServerClient
	fail  bool
	delay time.Duration
	mu    sync.Mutex
	calls int
}

func (f *rankMS) Rank(ctx context.Context, in *pb.RankRequest, _ ...grpc.CallOption) (*pb.RankResponse, error) {
	f.mu.Lock()
	f.calls++
	f.mu.Unlock()
	if f.fail {
		return nil, errors.New("unavailable")
	}
	select {
	case <-time.After(f.delay):
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	resp := &pb.RankResponse{ModelVersion: "test-bundle-v1", FeatureNames: []string{"sasrec_score", "n_sources"}, Ranker: "lightgbm"}
	for _, c := range in.Candidates {
		resp.Features = append(resp.Features, 0.5, float32(len(c.Hits)))
		resp.Scores = append(resp.Scores, float32(c.ItemId))
	}
	return resp, nil
}

type captureSink struct {
	mu  sync.Mutex
	evs []events.Event
}

func (c *captureSink) WriteServed(_ context.Context, evs []events.Event) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.evs = append(c.evs, evs...)
	return nil
}

func itemIDs(r memResp) []int32 {
	out := make([]int32, len(r.items))
	for i, it := range r.items {
		out[i] = it.ItemID
	}
	return out
}

func TestRankStageOrdersByLightGBMAndFallsBackToRRF(t *testing.T) {
	_, rdb := miniRedis(t)
	seed := func(user string) {
		for i, it := range []string{"3", "4", "5"} {
			rdb.ZAdd(context.Background(), "u:"+user+":hist", redis.Z{Score: float64(i + 1), Member: it})
		}
	}
	// Same history for every user; one user per case, so no case reads another's cached list.
	for _, u := range []string{"7", "8", "9", "10", "12"} {
		seed(u)
	}

	// RRF only (no Rank client): the baseline order.
	base := memServer(t, rdb, &fakeModel{delay: time.Millisecond}, false)
	rrf := call(t, base, "7")

	newServer := func(ms *rankMS, ranker string) (*api.Server, *rankMS) {
		s := memServer(t, rdb, &fakeModel{delay: time.Millisecond}, false)
		s.Bundle.Ranking = &bundle.Ranking{Features: []string{"sasrec_score", "n_sources"}, Ranker: "lightgbm"}
		s.Rank = &ranking.Client{MS: ms, Fixed: 0.01}
		s.DefaultRanker = ranker
		return s, ms
	}

	lgbm, ms := newServer(&rankMS{}, ranking.LightGBM)
	got := call(t, lgbm, "8")
	if got.code != 200 || len(got.degraded) != 0 || ms.calls != 1 {
		t.Fatalf("lightgbm: %d %v (%d Rank calls)", got.code, got.degraded, ms.calls)
	}
	if slices.Equal(itemIDs(got), itemIDs(rrf)) || got.items[0].ItemID < got.items[1].ItemID {
		t.Fatalf("not LightGBM's order: %v (RRF %v)", itemIDs(got), itemIDs(rrf))
	}

	// The RRF arm makes the same call (features for the served log) and keeps RRF's order.
	same, ms2 := newServer(&rankMS{}, ranking.RRF)
	if r := call(t, same, "9"); !slices.Equal(itemIDs(r), itemIDs(rrf)) || ms2.calls != 1 || len(r.degraded) != 0 {
		t.Fatalf("rrf arm: %v vs %v, %d calls, %v", itemIDs(r), itemIDs(rrf), ms2.calls, r.degraded)
	}

	// A failed or slow Rank call: RRF's order, the degradation recorded.
	for _, tc := range []struct {
		ms   *rankMS
		want string
		user string
	}{{&rankMS{fail: true}, "rank_error", "10"}, {&rankMS{delay: 200 * time.Millisecond}, "rank_timeout", "12"}} {
		srv, _ := newServer(tc.ms, ranking.LightGBM)
		r := call(t, srv, tc.user)
		if r.code != 200 || !slices.Contains(r.degraded, tc.want) || !slices.Equal(itemIDs(r), itemIDs(rrf)) {
			t.Fatalf("%s: %d %v %v", tc.want, r.code, r.degraded, itemIDs(r))
		}
		if r.took > 150*time.Millisecond {
			t.Fatalf("%s took %v: the ranking budget must cap it", tc.want, r.took)
		}
	}
}

func TestServedEventLogsTheFeaturesAndTheRanker(t *testing.T) {
	_, rdb := miniRedis(t)
	for i, it := range []string{"3", "4", "5"} {
		rdb.ZAdd(context.Background(), "u:11:hist", redis.Z{Score: float64(i + 1), Member: it})
	}
	s := memServer(t, rdb, &fakeModel{delay: time.Millisecond}, false)
	sink := &captureSink{}
	ctx, cancel := context.WithCancel(context.Background())
	s.Served = events.NewServedWriter(sink, 64, s.Log)
	go s.Served.Run(ctx, 10, time.Millisecond)
	defer func() { cancel(); s.Served.Wait() }()
	s.Bundle.Ranking = &bundle.Ranking{Features: []string{"sasrec_score", "n_sources"}, Ranker: "lightgbm"}
	s.Rank = &ranking.Client{MS: &rankMS{}, Fixed: 0.01}
	s.DefaultRanker = ranking.LightGBM
	if r := call(t, s, "11"); r.code != 200 {
		t.Fatal(r.code)
	}
	deadline := time.Now().Add(2 * time.Second)
	for {
		sink.mu.Lock()
		n := len(sink.evs)
		sink.mu.Unlock()
		if n > 0 || time.Now().After(deadline) {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	sink.mu.Lock()
	defer sink.mu.Unlock()
	if len(sink.evs) == 0 {
		t.Fatal("no served event")
	}
	var p struct {
		Ranker   string   `json:"ranker"`
		Features []string `json:"features"`
		Items    []struct {
			F    []*float64 `json:"f"`
			LGBM *float64   `json:"lgbm"`
		} `json:"items"`
	}
	if err := json.Unmarshal(sink.evs[0].Payload, &p); err != nil {
		t.Fatal(err)
	}
	if p.Ranker != "lightgbm" || len(p.Features) != 2 || len(p.Items) == 0 {
		t.Fatalf("payload %s", sink.evs[0].Payload)
	}
	for i, it := range p.Items {
		if len(it.F) != 2 || it.F[0] == nil || *it.F[0] != 0.5 || it.LGBM == nil {
			t.Fatalf("item %d: %s", i, sink.evs[0].Payload)
		}
	}
}
