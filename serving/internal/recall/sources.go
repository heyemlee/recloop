package recall

import (
	"context"
	"math"

	pb "github.com/heyemlee/recloop/serving/gen/recloop/modelserver/v1"
	"github.com/heyemlee/recloop/serving/internal/bundle"
	"github.com/heyemlee/recloop/serving/internal/store"
)

// HistoryWindow is how many recent items ItemCF and SASRec look at (design §4.1).
const HistoryWindow = 10

// ---------------------------------------------------------------- Pop

// Pop returns the most popular items of the bundle's training data.
type Pop struct{ B *bundle.Bundle }

func (Pop) Name() string { return "pop" }

func (p Pop) Recall(_ context.Context, req Request) (Output, error) {
	n := min(req.K, len(p.B.Pop))
	out := make([]Candidate, n)
	for i := 0; i < n; i++ {
		out[i] = Candidate{ItemID: p.B.Pop[i], Score: float64(p.B.PopCounts[i]), Via: -1}
	}
	return Output{Items: out, Version: p.B.Version}, nil
}

// ---------------------------------------------------------------- ItemCF

// ItemCF scores the neighbours of the recent history (home) or of the current item (item):
//
//	score(j) = sum_k gamma^(L-k) * sim(h_k -> j)   (h_L the most recent; gamma = HistoryDecay)
//
// the same scoring as ml/models/itemcf.py, with the offline-tuned gamma.
type ItemCF struct{ B *bundle.Bundle }

func (ItemCF) Name() string { return "itemcf" }

func (c ItemCF) Recall(ctx context.Context, req Request) (Output, error) {
	seeds := Recent(req.History, HistoryWindow)
	if req.Surface == SurfaceItem {
		seeds = []int32{req.Seed}
	}
	return Output{Items: c.Score(seeds, req.K), Version: c.B.Version}, ctx.Err()
}

// Score aggregates the neighbour lists of the seeds (oldest first).
func (c ItemCF) Score(seeds []int32, k int) []Candidate {
	type acc struct {
		score, best float64
		via         int32
	}
	scores := map[int32]*acc{}
	n := len(seeds)
	for pos, h := range seeds {
		if h < 0 || int(h) >= len(c.B.ItemCF.Neighbors) {
			continue
		}
		w := math.Pow(c.B.ItemCF.HistoryDecay, float64(n-1-pos))
		for i, j := range c.B.ItemCF.Neighbors[h] {
			contrib := w * float64(c.B.ItemCF.Scores[h][i])
			a := scores[j]
			if a == nil {
				a = &acc{via: h}
				scores[j] = a
			}
			a.score += contrib
			if contrib > a.best {
				a.best, a.via = contrib, h
			}
		}
	}
	out := make([]Candidate, 0, len(scores))
	for j, a := range scores {
		out = append(out, Candidate{ItemID: j, Score: a.score, Via: a.via})
	}
	return topK(out, k)
}

// ---------------------------------------------------------------- SASRec

// SASRec encodes the history in the model server and retrieves from its FAISS index.
type SASRec struct{ Client pb.ModelServerClient }

func (SASRec) Name() string { return "sasrec" }

func (s SASRec) Recall(ctx context.Context, req Request) (Output, error) {
	history := Recent(req.History, HistoryWindow)
	if len(history) == 0 {
		return Output{}, nil // cold start: nothing to encode
	}
	resp, err := s.Client.Retrieve(ctx, &pb.RetrieveRequest{History: history, K: int32(req.K)})
	if err != nil {
		return Output{}, err
	}
	out := make([]Candidate, len(resp.Organic))
	for i, o := range resp.Organic {
		out[i] = Candidate{ItemID: o.ItemId, Score: float64(o.Score), Via: -1}
	}
	return Output{Items: out, Version: resp.ModelVersion}, nil
}

// ---------------------------------------------------------------- text

// TextSearcher finds text-similar items (store.Store in production).
type TextSearcher interface {
	TextNeighbors(ctx context.Context, item int32, k int) ([]store.Neighbor, error)
}

// Text returns the items whose title + features embedding is closest to the current item's
// (item surface only; pgvector, design §4.1).
type Text struct {
	S       TextSearcher
	Version string // embedding model, e.g. bge-small-en-v1.5
}

func (Text) Name() string { return "text" }

func (t Text) Recall(ctx context.Context, req Request) (Output, error) {
	if req.Surface != SurfaceItem || req.Seed < 0 {
		return Output{}, nil
	}
	ns, err := t.S.TextNeighbors(ctx, req.Seed, req.K)
	if err != nil {
		return Output{}, err
	}
	out := make([]Candidate, len(ns))
	for i, n := range ns {
		out[i] = Candidate{ItemID: n.ItemID, Score: n.Similarity, Via: req.Seed}
	}
	return Output{Items: out, Version: t.Version}, nil
}
