package mixer

import (
	"sort"
)

// RRFK is the rank constant of reciprocal rank fusion: score = sum_s w_s / (RRFK + rank_s).
const RRFK = 60

// Hit is one source's opinion about an item.
type Hit struct {
	Source string  `json:"source"`
	Rank   int     `json:"rank"` // 1-based
	Score  float64 `json:"score"`
	Via    int32   `json:"via,omitempty"`
}

// Fused is an item after fusion.
type Fused struct {
	ItemID  int32
	Score   float64 // fused (RRF) score
	Primary string  // source with the largest weighted contribution
	Hits    []Hit
}

func (f Fused) bestRank() int {
	best := 1 << 30
	for _, h := range f.Hits {
		best = min(best, h.Rank)
	}
	return best
}

// RRF fuses the sources' ranked lists with weighted reciprocal rank fusion (design §4.3):
//
//	score(i) = sum over sources s that returned i of  w_s / (RRFK + rank_s(i))
//
// Sources without a weight (or with weight <= 0) are ignored; failed sources contribute
// nothing. Order: score desc, then best single-source rank, then item id.
func RRF(results []Result, weights map[string]float64) []Fused {
	byItem := map[int32]*Fused{}
	best := map[int32]float64{}
	var order []int32
	for _, r := range results {
		w := weights[r.Source]
		if r.Err != nil || w <= 0 {
			continue
		}
		for pos, c := range r.Items {
			rank := pos + 1
			contrib := w / float64(RRFK+rank)
			f := byItem[c.ItemID]
			if f == nil {
				f = &Fused{ItemID: c.ItemID}
				byItem[c.ItemID] = f
				order = append(order, c.ItemID)
			}
			f.Score += contrib
			hit := Hit{Source: r.Source, Rank: rank, Score: c.Score}
			if c.Via >= 0 {
				hit.Via = c.Via
			}
			f.Hits = append(f.Hits, hit)
			if contrib > best[c.ItemID] {
				best[c.ItemID], f.Primary = contrib, r.Source
			}
		}
	}
	out := make([]Fused, 0, len(order))
	for _, id := range order {
		out = append(out, *byItem[id])
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		if bi, bj := out[i].bestRank(), out[j].bestRank(); bi != bj {
			return bi < bj
		}
		return out[i].ItemID < out[j].ItemID
	})
	return out
}
