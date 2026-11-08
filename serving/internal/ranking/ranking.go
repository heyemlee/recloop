// Package ranking calls the model server's Rank (design §4.3, §9.2, §13.2; M4): the ranker's
// features and LightGBM score of every organic candidate, and the pCTR of the ad candidates.
// The features come back with the scores so the served event can log what the models saw.
package ranking

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"

	"google.golang.org/protobuf/proto"

	pb "github.com/heyemlee/recloop/serving/gen/recloop/modelserver/v1"
	"github.com/heyemlee/recloop/serving/internal/ads"
	"github.com/heyemlee/recloop/serving/internal/mixer"
	"github.com/heyemlee/recloop/serving/internal/resilience"
)

// Rankers a rank-layer experiment can ask for (params {"ranker": ...}).
const (
	RRF      = "rrf"
	LightGBM = "lightgbm"
)

// Client is safe for concurrent use.
type Client struct {
	MS      pb.ModelServerClient
	Breaker *resilience.Breaker // nil: no breaker
	// Fixed is the pCTR when the bundle has no pCTR model (ads.FixedPCTR, M3's stub).
	Fixed ads.FixedPCTR
}

// Organic is what Rank said about the organic candidates, in their order.
type Organic struct {
	Version  string
	Names    []string    // feature columns
	Features [][]float32 // per candidate
	Scores   []float32   // LightGBM score per candidate; nil when the bundle has no ranker
	Ranker   string      // "lightgbm" or ""
}

func (c *Client) call(ctx context.Context, req *pb.RankRequest) (*pb.RankResponse, error) {
	var resp *pb.RankResponse
	f := func() error {
		var err error
		resp, err = c.MS.Rank(ctx, req)
		return err
	}
	var err error
	if c.Breaker != nil {
		err = c.Breaker.Do(ctx, f)
	} else {
		err = f()
	}
	return resp, err
}

func rows(flat []float32, n, width int) ([][]float32, error) {
	if width == 0 || len(flat) != n*width {
		return nil, fmt.Errorf("rank: %d feature values for %d rows of %d", len(flat), n, width)
	}
	out := make([][]float32, n)
	for i := range out {
		out[i] = flat[i*width : (i+1)*width : (i+1)*width]
	}
	return out, nil
}

// Organic describes and scores the fused candidates. history is the recall window.
func (c *Client) Organic(ctx context.Context, version string, history []int32, cands []mixer.Fused) (Organic, error) {
	req := &pb.RankRequest{History: history, PinVersion: version, Candidates: make([]*pb.Candidate, len(cands))}
	for i, f := range cands {
		hits := make([]*pb.SourceHit, len(f.Hits))
		for j, h := range f.Hits {
			hits[j] = &pb.SourceHit{Source: h.Source, Rank: int32(h.Rank), Score: float32(h.Score)}
		}
		req.Candidates[i] = &pb.Candidate{ItemId: f.ItemID, Hits: hits}
	}
	resp, err := c.call(ctx, req)
	if err != nil {
		return Organic{}, err
	}
	feats, err := rows(resp.Features, len(cands), len(resp.FeatureNames))
	if err != nil {
		return Organic{}, err
	}
	out := Organic{Version: resp.ModelVersion, Names: resp.FeatureNames, Features: feats, Ranker: resp.Ranker}
	if resp.Ranker != "" {
		if len(resp.Scores) != len(cands) {
			return Organic{}, fmt.Errorf("rank: %d scores for %d candidates", len(resp.Scores), len(cands))
		}
		out.Scores = resp.Scores
	}
	return out, nil
}

// PredictCTR implements ads.PCTRModel through Rank: the model server computes the features of
// each ad item (SASRec relevance, popularity, price, category overlap with the history) and
// the logistic regression at the reference position. A bundle without a pCTR model still
// returns the features; the rate is then Fixed's.
func (c *Client) PredictCTR(ctx context.Context, r ads.PCTRRequest) (ads.PCTRResult, error) {
	req := &pb.RankRequest{History: r.History, PinVersion: r.Version, Ads: make([]*pb.AdCandidate, len(r.ItemIDs))}
	for i, id := range r.ItemIDs {
		a := &pb.AdCandidate{ItemId: id}
		if rel, ok := r.Relevance[id]; ok {
			a.Relevance = proto.Float32(rel)
		}
		req.Ads[i] = a
	}
	resp, err := c.call(ctx, req)
	if err != nil {
		return ads.PCTRResult{}, err
	}
	feats, err := rows(resp.AdFeatures, len(r.ItemIDs), len(resp.FeatureNames))
	if err != nil {
		return ads.PCTRResult{}, err
	}
	if resp.PctrModel == "" {
		res, err := c.Fixed.PredictCTR(ctx, r)
		res.Features, res.FeatureNames = feats, resp.FeatureNames
		return res, err
	}
	if len(resp.AdPctr) != len(r.ItemIDs) {
		return ads.PCTRResult{}, fmt.Errorf("rank: %d pCTRs for %d ad items", len(resp.AdPctr), len(r.ItemIDs))
	}
	ctr := make([]float64, len(resp.AdPctr))
	for i, p := range resp.AdPctr {
		if !(p >= 0 && p <= 1) {
			return ads.PCTRResult{}, fmt.Errorf("rank: pCTR %v of item %d is not a probability", p, r.ItemIDs[i])
		}
		ctr[i] = float64(p)
	}
	return ads.PCTRResult{CTR: ctr, Features: feats, FeatureNames: resp.FeatureNames, Model: resp.PctrModel}, nil
}

// Reorder sorts the candidates by LightGBM score, best first; equal scores keep RRF's order.
// The Score of each item becomes the LightGBM score (what ordered the list).
func Reorder(cands []mixer.Fused, scores []float32) []mixer.Fused {
	type pair struct {
		f mixer.Fused
		s float32
	}
	ps := make([]pair, len(cands))
	for i, f := range cands {
		ps[i] = pair{f, scores[i]}
	}
	sort.SliceStable(ps, func(a, b int) bool { return ps[a].s > ps[b].s }) // ties: RRF order
	out := make([]mixer.Fused, len(ps))
	for i, p := range ps {
		p.f.Score = float64(p.s)
		out[i] = p.f
	}
	return out
}

// Features is a feature row for JSON: NaN (missing) is written as null.
type Features []float32

// MarshalJSON writes 7 significant digits per value, null for NaN.
func (f Features) MarshalJSON() ([]byte, error) {
	var b strings.Builder
	b.WriteByte('[')
	for i, v := range f {
		if i > 0 {
			b.WriteByte(',')
		}
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			b.WriteString("null")
			continue
		}
		b.WriteString(strconv.FormatFloat(float64(v), 'g', 7, 32))
	}
	b.WriteByte(']')
	return []byte(b.String()), nil
}
