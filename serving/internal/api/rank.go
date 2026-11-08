package api

import (
	"context"
	"errors"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/heyemlee/recloop/serving/internal/ads"
	"github.com/heyemlee/recloop/serving/internal/bundle"
	"github.com/heyemlee/recloop/serving/internal/cache"
	"github.com/heyemlee/recloop/serving/internal/mixer"
	"github.com/heyemlee/recloop/serving/internal/ranking"
	"github.com/heyemlee/recloop/serving/internal/resilience"
	"github.com/heyemlee/recloop/serving/internal/userstate"
)

// RankMinTimeout: with less of the budget left, the Rank call is not made (rank_skipped), like
// the remote recall sources.
const RankMinTimeout = 5 * time.Millisecond

// active is the bundle unpinned requests get.
func (s *Server) active() *bundle.Bundle {
	if s.Bundles != nil {
		return s.Bundles.Active()
	}
	return s.Bundle
}

// rankOrganic is the model server's Rank for the fused candidates under the ranking stage's
// budget (design §4.1 ⑤: 25 ms); it returns the call's status (mixer.Status*) and error.
func (s *Server) rankOrganic(q *request, fused []mixer.Fused) (ranking.Organic, string, error) {
	call := func(ctx context.Context) (ranking.Organic, error) {
		return s.Rank.Organic(ctx, q.bundle.Version, q.history, fused)
	}
	if s.Naive {
		org, err := call(q.ctx)
		if err != nil {
			return org, mixer.StatusError, err
		}
		return org, mixer.StatusOK, nil
	}
	if b := s.Rank.Breaker; b != nil && b.State() == resilience.Open {
		return ranking.Organic{}, mixer.StatusOpen, resilience.ErrOpen
	}
	d := q.budget.For(s.Plan.Rank, s.Plan.Ads+s.Plan.Tail)
	if d < RankMinTimeout {
		return ranking.Organic{}, mixer.StatusSkipped, mixer.ErrTimeout
	}
	ctx, cancel := context.WithTimeout(q.ctx, d)
	defer cancel()
	_, span := q.span("rank")
	org, err := call(ctx)
	span.End()
	switch {
	case err == nil:
		return org, mixer.StatusOK, nil
	case errors.Is(err, resilience.ErrOpen):
		return org, mixer.StatusOpen, err
	case ctx.Err() != nil, status.Code(err) == codes.DeadlineExceeded:
		return org, mixer.StatusTimeout, err
	default:
		return org, mixer.StatusError, err
	}
}

// eligibleAdItems are the items of the in-flight campaigns that the ads stage could still pick
// for this user (in stock, not bought): Retrieve scores them with the request's user vector
// (design §9.2 step 3). The budget, frequency and pacing gates come later, in the ads stage.
func (s *Server) eligibleAdItems(q *request, uc userstate.Context) []int32 {
	if s.Ads == nil || !adsAllowed(q.degraded) {
		return nil
	}
	snap := s.Ads.Index.Current()
	if !snap.Fresh(q.now) {
		return nil
	}
	pend := snap.Prepare(ads.Request{Now: q.now, Allowed: func(it int32) bool {
		return s.Catalog.InStock(it) && !uc.Bought[it]
	}})
	seen := map[int32]bool{}
	var out []int32
	for i := range pend.Campaigns {
		for _, it := range pend.Items[i] {
			if !seen[it] {
				seen[it] = true
				out = append(out, it)
			}
		}
	}
	return out
}

// addRanking writes the ranking stage's view into the first page's served event: the feature
// names once, each organic item's features and LightGBM score, and each sponsored slot's
// features as the pCTR model saw them.
func addRanking(served *servedPayload, rk rendered, slots []cache.Slot) {
	if len(rk.names) == 0 {
		for _, sl := range slots {
			if len(sl.Features) > 0 {
				rk.names = sl.FeatureNames
				break
			}
		}
	}
	if len(rk.names) == 0 {
		return
	}
	served.Features = rk.names
	byPos := map[int]cache.Slot{}
	for _, sl := range slots {
		byPos[sl.Position] = sl
	}
	for i := range served.Items {
		it := &served.Items[i]
		if it.Slot == SlotSponsored {
			if sl, ok := byPos[it.Pos]; ok && len(sl.Features) > 0 {
				it.F = sl.Features
			}
			continue
		}
		if f, ok := rk.features[it.ItemID]; ok {
			it.F = f
		}
		if v, ok := rk.lgbm[it.ItemID]; ok {
			x := float64(v)
			it.LGBM = &x
		}
	}
}
