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
}

// PCTRModel predicts one click-through rate per requested item, at a fixed reference position
// (position bias is a training feature, not a serving one; §9.2 step 3). The logistic
// regression trained on the served logs replaces the stub below.
type PCTRModel interface {
	PredictCTR(ctx context.Context, req PCTRRequest) ([]float64, error)
}

// FixedPCTR predicts the same rate for every item: a stand-in until the pCTR model exists.
// With one rate for all, the auction ranks by bid alone.
type FixedPCTR float64

// PredictCTR implements PCTRModel.
func (f FixedPCTR) PredictCTR(_ context.Context, req PCTRRequest) ([]float64, error) {
	if f < 0 || f > 1 {
		return nil, fmt.Errorf("fixed pCTR %v is not a probability", float64(f))
	}
	out := make([]float64, len(req.ItemIDs))
	for i := range out {
		out[i] = float64(f)
	}
	return out, nil
}
