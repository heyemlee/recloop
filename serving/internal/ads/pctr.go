package ads

import (
	"context"
	"fmt"
)

// PCTRRequest asks for the click-through rate of sponsored candidates in one request.
type PCTRRequest struct {
	UserID  int64
	Surface string
	ItemIDs []int32
	// What the logistic regression of M4 needs (the model server computes the features):
	History   []int32           // the user's recent items, oldest first
	Relevance map[int32]float32 // SASRec relevance from the request's Retrieve call, if it made one
	Version   string            // the bundle version serving the request (a canary pins its own)
	// Counts: each item's impressions and clicks of the last hour and day (itemctr.Counts.Array:
	// hour impressions, hour clicks, day impressions, day clicks); nil when the api has no
	// current table, and the model then uses its prior.
	Counts map[int32][4]uint32
}

// PCTRResult has one rate per requested item, at the reference position.
type PCTRResult struct {
	CTR []float64
	// Features per item, as the model saw them: the served event logs them for the sponsored
	// slots (design §7.5: training uses the features of serving time). Nil if there are none.
	Features     [][]float32
	FeatureNames []string
	Model        string // "fixed", "logreg", ...
}

// PCTRModel predicts one click-through rate per requested item, at a fixed reference position
// (position bias is a training feature, not a serving one; §9.2 step 3).
type PCTRModel interface {
	PredictCTR(ctx context.Context, req PCTRRequest) (PCTRResult, error)
}

// FixedPCTR predicts the same rate for every item: M3's stand-in, and what a bundle without a
// pCTR model still gets. With one rate for all, the auction ranks by bid alone.
type FixedPCTR float64

// PCTRFixed is the Model name of FixedPCTR's results.
const PCTRFixed = "fixed"

// PredictCTR implements PCTRModel.
func (f FixedPCTR) PredictCTR(_ context.Context, req PCTRRequest) (PCTRResult, error) {
	if f < 0 || f > 1 {
		return PCTRResult{}, fmt.Errorf("fixed pCTR %v is not a probability", float64(f))
	}
	out := make([]float64, len(req.ItemIDs))
	for i := range out {
		out[i] = float64(f)
	}
	return PCTRResult{CTR: out, Model: PCTRFixed}, nil
}
