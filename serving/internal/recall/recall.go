// Package recall holds the candidate sources of the recommendation pipeline (design §4.2):
// Pop and ItemCF in process, SASRec through the model server, text similarity through pgvector.
package recall

import (
	"context"
	"sort"
)

// Surfaces.
const (
	SurfaceHome = "home"
	SurfaceItem = "item"
)

// Candidate is one retrieved item. Via is the history (or seed) item that contributed most,
// for sources that can tell (ItemCF); -1 otherwise.
type Candidate struct {
	ItemID int32
	Score  float64
	Via    int32
}

// Request is what every source sees.
type Request struct {
	UserID  int64
	Surface string
	History []int32 // oldest first
	Seed    int32   // the current item on the item surface; -1 otherwise
	K       int
}

// Output of one source: candidates best first, and the model version that produced them.
type Output struct {
	Items   []Candidate
	Version string
}

// Source retrieves candidates; it must return promptly once ctx is done.
type Source interface {
	Name() string
	Recall(ctx context.Context, req Request) (Output, error)
}

// Recent returns the last n items of a history (oldest first).
func Recent(history []int32, n int) []int32 {
	if len(history) <= n {
		return history
	}
	return history[len(history)-n:]
}

// topK sorts by score (desc), then item id, and keeps k.
func topK(c []Candidate, k int) []Candidate {
	sort.Slice(c, func(i, j int) bool {
		if c[i].Score != c[j].Score {
			return c[i].Score > c[j].Score
		}
		return c[i].ItemID < c[j].ItemID
	})
	if len(c) > k {
		c = c[:k]
	}
	return c
}
