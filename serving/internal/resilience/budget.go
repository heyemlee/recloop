// Package resilience holds what keeps the recommendation path inside its latency budget
// (design §5.1): the request deadline, the stage budgets taken from what is left of it, and the
// circuit breakers in front of the remote dependencies.
package resilience

import (
	"context"
	"math"
	"time"
)

// Plan is the latency budget of one recommendation request (design §4.1, §5.1). Every stage
// takes min(its own budget, time left − what the later stages need), so a slow early stage
// shortens the later ones instead of pushing the response past Total.
type Plan struct {
	Total       time.Duration // whole request, server side
	UserContext time.Duration // ① Redis pipeline: history, hv, bought, seen
	CacheGet    time.Duration // candidate cache lookup (Redis)
	Recall      time.Duration // ② concurrent recall
	Rank        time.Duration // ⑤ reserved for ranking (RRF in process today; LightGBM from M4)
	Ads         time.Duration // ⑥ sponsored slots: one Redis read (spend, frequency), pCTR, auction
	Tail        time.Duration // ④⑥⑦ filters, diversity, page store, rendering
}

// DefaultPlan is the design's 100 ms request: 5 ms context, 35 ms recall, 25 ms ranking, 10 ms
// for the ads.
var DefaultPlan = Plan{
	Total:       100 * time.Millisecond,
	UserContext: 5 * time.Millisecond,
	CacheGet:    3 * time.Millisecond,
	Recall:      35 * time.Millisecond,
	Rank:        25 * time.Millisecond,
	Ads:         10 * time.Millisecond,
	Tail:        10 * time.Millisecond,
}

// Budget tracks the deadline of one request. The zero Budget has no deadline: every stage gets
// what it asks for (the naive mode, which does not use stage budgets at all).
type Budget struct {
	deadline time.Time
	now      func() time.Time
}

// Start sets the request deadline (now + plan.Total) on ctx.
func Start(ctx context.Context, plan Plan, now func() time.Time) (context.Context, context.CancelFunc, Budget) {
	if now == nil {
		now = time.Now
	}
	d := now().Add(plan.Total)
	ctx, cancel := context.WithDeadline(ctx, d)
	return ctx, cancel, Budget{deadline: d, now: now}
}

// Left is the time until the request deadline (negative once it has passed).
func (b Budget) Left() time.Duration {
	if b.now == nil {
		return math.MaxInt64
	}
	return b.deadline.Sub(b.now())
}

// For returns the budget of a stage that would like `want` and must leave `reserve` for the
// stages after it: min(want, Left() − reserve). It can be zero or negative: the stage is
// skipped then.
func (b Budget) For(want, reserve time.Duration) time.Duration {
	if b.now == nil {
		return want
	}
	return min(want, b.Left()-reserve)
}

// Stage derives the context of a stage with budget d (see For). The caller checks d first.
func Stage(ctx context.Context, d time.Duration) (context.Context, context.CancelFunc) {
	return context.WithTimeout(ctx, d)
}
