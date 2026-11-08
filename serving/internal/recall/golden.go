package recall

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/heyemlee/recloop/serving/internal/bundle"
)

// CheckItemCFGolden replays the validation histories itemcf.json carries with the Python
// model's top-20 and scores: the Go scorer must reproduce them before a bundle may serve
// (design §6, the api's side of a release's validation). A bundle without golden queries passes.
func CheckItemCFGolden(b *bundle.Bundle) error {
	raw, err := os.ReadFile(filepath.Join(b.Dir, "itemcf.json"))
	if err != nil {
		return err
	}
	var f struct {
		Golden []struct {
			History []int32   `json:"history"`
			Items   []int32   `json:"items"`
			Scores  []float64 `json:"scores"`
		} `json:"golden"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		return err
	}
	cf := ItemCF{B: b}
	for qi, q := range f.Golden {
		got := cf.Score(q.History, len(q.Items))
		if len(got) != len(q.Items) {
			return fmt.Errorf("itemcf golden query %d: %d candidates, want %d", qi, len(got), len(q.Items))
		}
		for i := range got {
			if math.Abs(got[i].Score-q.Scores[i]) > 1e-6*math.Max(1, q.Scores[i]) {
				return fmt.Errorf("itemcf golden query %d rank %d: score %.8f, want %.8f", qi, i, got[i].Score, q.Scores[i])
			}
			// Ids must agree except where two scores tie to within rounding.
			if got[i].ItemID != q.Items[i] && math.Abs(got[i].Score-q.Scores[i]) > 1e-9 {
				return fmt.Errorf("itemcf golden query %d rank %d: item %d, want %d", qi, i, got[i].ItemID, q.Items[i])
			}
		}
	}
	return nil
}
