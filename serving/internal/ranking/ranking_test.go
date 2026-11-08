package ranking_test

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"testing"

	"google.golang.org/grpc"

	pb "github.com/heyemlee/recloop/serving/gen/recloop/modelserver/v1"
	"github.com/heyemlee/recloop/serving/internal/ads"
	"github.com/heyemlee/recloop/serving/internal/mixer"
	"github.com/heyemlee/recloop/serving/internal/ranking"
)

// fakeMS answers Rank from fixed data and records the request.
type fakeMS struct {
	pb.ModelServerClient
	resp *pb.RankResponse
	err  error
	last *pb.RankRequest
}

func (f *fakeMS) Rank(_ context.Context, in *pb.RankRequest, _ ...grpc.CallOption) (*pb.RankResponse, error) {
	f.last = in
	return f.resp, f.err
}

var names = []string{"sasrec_score", "n_sources"}

func TestOrganicSendsHitsAndSplitsFeatureRows(t *testing.T) {
	ms := &fakeMS{resp: &pb.RankResponse{
		ModelVersion: "v1", FeatureNames: names, Features: []float32{1.5, 2, float32(math.NaN()), 1},
		Scores: []float32{0.1, 0.9}, Ranker: "lightgbm",
	}}
	c := &ranking.Client{MS: ms}
	cands := []mixer.Fused{
		{ItemID: 4, Hits: []mixer.Hit{{Source: "sasrec", Rank: 1, Score: 1.5}, {Source: "pop", Rank: 3, Score: 7}}},
		{ItemID: 9, Hits: []mixer.Hit{{Source: "itemcf", Rank: 2, Score: 0.25}}},
	}
	org, err := c.Organic(context.Background(), "v1", []int32{1, 2}, cands)
	if err != nil {
		t.Fatal(err)
	}
	if ms.last.PinVersion != "v1" || len(ms.last.Candidates) != 2 || len(ms.last.Candidates[0].Hits) != 2 ||
		ms.last.Candidates[1].Hits[0].Source != "itemcf" || ms.last.Candidates[1].Hits[0].Rank != 2 {
		t.Fatalf("request %v", ms.last)
	}
	if len(org.Features) != 2 || org.Features[0][0] != 1.5 || !math.IsNaN(float64(org.Features[1][0])) || org.Ranker != "lightgbm" {
		t.Fatalf("organic %+v", org)
	}
	ms.resp.Features = ms.resp.Features[:3] // a short reply is an error, not a misaligned row
	if _, err := c.Organic(context.Background(), "v1", nil, cands); err == nil {
		t.Fatal("ragged features accepted")
	}
}

func TestReorderIsStableOnTies(t *testing.T) {
	in := []mixer.Fused{{ItemID: 1, Score: 0.3}, {ItemID: 2, Score: 0.2}, {ItemID: 3, Score: 0.1}, {ItemID: 4}}
	out := ranking.Reorder(in, []float32{0.5, 0.9, 0.5, -1})
	got := []int32{out[0].ItemID, out[1].ItemID, out[2].ItemID, out[3].ItemID}
	if want := []int32{2, 1, 3, 4}; got[0] != want[0] || got[1] != want[1] || got[2] != want[2] || got[3] != want[3] {
		t.Fatalf("order %v, want %v", got, want)
	}
	if out[0].Score != float64(float32(0.9)) || in[0].ItemID != 1 {
		t.Fatalf("score %v, input changed %v", out[0].Score, in)
	}
}

func TestFeaturesJSONWritesNullForMissing(t *testing.T) {
	raw, err := json.Marshal(struct {
		F ranking.Features `json:"f"`
	}{ranking.Features{1.25, float32(math.NaN()), 3}})
	if err != nil || string(raw) != `{"f":[1.25,null,3]}` {
		t.Fatalf("%s %v", raw, err)
	}
}

func TestPredictCTRFallsBackToTheFixedRateWithoutAModel(t *testing.T) {
	ms := &fakeMS{resp: &pb.RankResponse{ModelVersion: "v1", FeatureNames: names, AdFeatures: []float32{0.5, 0, 0.7, 0}}}
	c := &ranking.Client{MS: ms, Fixed: 0.02}
	res, err := c.PredictCTR(context.Background(), ads.PCTRRequest{ItemIDs: []int32{5, 6}, Relevance: map[int32]float32{5: 0.5}, Version: "v1"})
	if err != nil || res.Model != ads.PCTRFixed || res.CTR[1] != 0.02 || res.Features[1][0] != 0.7 || res.FeatureNames[0] != "sasrec_score" {
		t.Fatalf("%+v %v", res, err)
	}
	if ms.last.Ads[0].Relevance == nil || *ms.last.Ads[0].Relevance != 0.5 || ms.last.Ads[1].Relevance != nil {
		t.Fatalf("relevance not passed through: %v", ms.last.Ads)
	}
	ms.resp.PctrModel, ms.resp.AdPctr = "logreg", []float32{0.03, 1.5}
	if _, err := c.PredictCTR(context.Background(), ads.PCTRRequest{ItemIDs: []int32{5, 6}}); err == nil {
		t.Fatal("a pCTR of 1.5 accepted")
	}
	ms.resp.AdPctr = []float32{0.03, 0.01}
	res, err = c.PredictCTR(context.Background(), ads.PCTRRequest{ItemIDs: []int32{5, 6}})
	if err != nil || res.Model != "logreg" || math.Abs(res.CTR[0]-0.03) > 1e-7 {
		t.Fatalf("%+v %v", res, err)
	}
	ms.err = errors.New("down")
	if _, err := c.PredictCTR(context.Background(), ads.PCTRRequest{ItemIDs: []int32{5}}); err == nil {
		t.Fatal("error swallowed")
	}
}

func TestPredictCTRSendsTheItemsCounts(t *testing.T) {
	ms := &fakeMS{resp: &pb.RankResponse{ModelVersion: "v1", FeatureNames: names, AdFeatures: []float32{0.5, 0, 0.7, 0},
		PctrModel: "logreg", AdPctr: []float32{0.02, 0.01}}}
	c := &ranking.Client{MS: ms}
	_, err := c.PredictCTR(context.Background(), ads.PCTRRequest{
		ItemIDs: []int32{5, 6}, Counts: map[int32][4]uint32{5: {120, 3, 2400, 31}},
	})
	if err != nil {
		t.Fatal(err)
	}
	got := ms.last.Ads[0].Counts
	if got == nil || got.ImpressionsHour != 120 || got.ClicksHour != 3 || got.ImpressionsDay != 2400 || got.ClicksDay != 31 {
		t.Fatalf("counts of item 5: %v", got)
	}
	if ms.last.Ads[1].Counts != nil {
		t.Fatalf("item 6 has no counts in the request, got %v", ms.last.Ads[1].Counts)
	}
}
