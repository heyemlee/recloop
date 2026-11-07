package recall

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/heyemlee/recloop/serving/internal/bundle"
)

func bundleDir() string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..", "..", "..", "artifacts", "bundles", "current")
}

// The Go ItemCF scorer must reproduce the Python model the offline evaluation scored:
// itemcf.json carries validation histories with the Python top-20 and scores.
func TestItemCFMatchesPythonOnTheBundle(t *testing.T) {
	dir := bundleDir()
	if _, err := os.Stat(dir); err != nil {
		t.Skip("no bundle: run `make bundle`")
	}
	b, err := bundle.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(dir, "itemcf.json"))
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Golden []struct {
			History []int32   `json:"history"`
			Items   []int32   `json:"items"`
			Scores  []float64 `json:"scores"`
		} `json:"golden"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if len(f.Golden) == 0 {
		t.Skip("bundle has no ItemCF golden queries (rebuild it)")
	}
	cf := ItemCF{B: b}
	for qi, q := range f.Golden {
		got := cf.Score(q.History, len(q.Items))
		if len(got) != len(q.Items) {
			t.Fatalf("query %d: %d candidates, want %d", qi, len(got), len(q.Items))
		}
		for i := range got {
			if math.Abs(got[i].Score-q.Scores[i]) > 1e-6*math.Max(1, q.Scores[i]) {
				t.Fatalf("query %d rank %d: score %.8f, want %.8f", qi, i, got[i].Score, q.Scores[i])
			}
			// Ids must agree except where two scores tie to within rounding.
			if got[i].ItemID != q.Items[i] && math.Abs(got[i].Score-q.Scores[i]) > 1e-9 {
				t.Fatalf("query %d rank %d: item %d, want %d", qi, i, got[i].ItemID, q.Items[i])
			}
		}
	}
}

func TestItemCFItemSurfaceUsesTheSeedOnly(t *testing.T) {
	b := &bundle.Bundle{Version: "v", NumItems: 4, ItemCF: bundle.ItemCF{
		HistoryDecay: 0.5,
		Neighbors:    [][]int32{{1, 2}, {0}, {3}, {}},
		Scores:       [][]float32{{0.9, 0.1}, {0.8}, {0.7}, {}},
	}}
	out, err := ItemCF{B: b}.Recall(context.Background(), Request{Surface: SurfaceItem, Seed: 0, History: []int32{2}, K: 10})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Items) != 2 || out.Items[0].ItemID != 1 || out.Items[0].Via != 0 {
		t.Fatalf("item surface: %+v", out.Items)
	}
	home := ItemCF{B: b}.Score([]int32{0, 2}, 10) // 2 is most recent (weight 1), 0 gets 0.5
	want := map[int32]float64{3: 0.7, 1: 0.45, 2: 0.05}
	for _, c := range home {
		if math.Abs(c.Score-want[c.ItemID]) > 1e-6 {
			t.Fatalf("home scores %+v", home)
		}
	}
	if home[0].ItemID != 3 || home[0].Via != 2 {
		t.Fatalf("order / via: %+v", home)
	}
}
