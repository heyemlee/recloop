// Package mixer is the recommendation pipeline after the user context is read (design §4.1):
// concurrent fan-out to the recall sources, weighted RRF over whatever came back, filtering,
// brand diversity and popularity backfill.
//
// FanOut is the resilient version (design §5.1): every source runs under the smaller of its own
// timeout and the recall stage's budget, a remote source sits behind a circuit breaker, and
// the merge takes whatever answered in time. FanOutNaive is the baseline it is measured against
// (§5.2): no timeouts, no breakers, wait for everything.
package mixer

import (
	"context"
	"errors"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/heyemlee/recloop/serving/internal/metrics"
	"github.com/heyemlee/recloop/serving/internal/recall"
	"github.com/heyemlee/recloop/serving/internal/resilience"
)

// Spec is a source with its timeout, candidate count and circuit breaker.
type Spec struct {
	Source  recall.Source
	Timeout time.Duration // the source's own budget; the recall stage's budget caps it
	// MinTimeout: when less than this is left the source is not called (skipped): an RPC with
	// a 1 ms deadline only burns the dependency's CPU. In-process sources leave it 0.
	MinTimeout time.Duration
	K          int
	Breaker    *resilience.Breaker // remote sources; nil for in-process ones
}

// Status of one source in one request; a source that is not ok shows up in the response as
// degraded "<source>_<status>".
const (
	StatusOK      = "ok"
	StatusTimeout = "timeout"
	StatusError   = "error"
	StatusOpen    = "circuit_open" // breaker open: not called
	StatusSkipped = "skipped"      // not enough of the request's budget left to call it
)

// Result of one source.
type Result struct {
	Source   string
	Items    []recall.Candidate
	Version  string
	AdScores map[int32]float32 // SASRec's relevance of the request's ad items
	Status   string
	Err      error
	Took     time.Duration
}

// OK reports whether the source answered.
func (r Result) OK() bool { return r.Status == StatusOK }

// ErrTimeout marks a source that did not answer within its timeout.
var ErrTimeout = errors.New("timeout")

// applicable lets a source say it has nothing to do for a request (SASRec without history):
// it is then not called and does not count towards its breaker.
type applicable interface {
	Applicable(req recall.Request) bool
}

func skip(sp Spec, req recall.Request) bool {
	a, ok := sp.Source.(applicable)
	return ok && !a.Applicable(req)
}

func observe(r Result) {
	metrics.RecallDuration.WithLabelValues(r.Source, r.Status).Observe(r.Took.Seconds())
}

// FanOut queries the sources concurrently and returns one Result per spec, in spec order.
// Source i gets min(spec.Timeout, limit); it is skipped if that is below spec.MinTimeout. The
// call returns once every called source has answered or the longest of their timeouts has
// passed, so a slow or stuck source never holds the response.
func FanOut(ctx context.Context, specs []Spec, req recall.Request, limit time.Duration) []Result {
	type done struct {
		i int
		r Result
	}
	results := make([]Result, len(specs))
	got := make([]bool, len(specs))
	ch := make(chan done, len(specs))
	var wait time.Duration
	pending := 0
	for i, sp := range specs {
		name := sp.Source.Name()
		if skip(sp, req) {
			results[i], got[i] = Result{Source: name, Status: StatusOK}, true
			continue
		}
		timeout := min(sp.Timeout, limit)
		if timeout <= 0 || timeout < sp.MinTimeout {
			results[i], got[i] = Result{Source: name, Status: StatusSkipped, Err: ErrTimeout}, true
			observe(results[i])
			continue
		}
		if sp.Breaker != nil && sp.Breaker.State() == resilience.Open {
			results[i], got[i] = Result{Source: name, Status: StatusOpen, Err: resilience.ErrOpen}, true
			observe(results[i])
			continue
		}
		wait = max(wait, timeout)
		pending++
		go func() {
			sctx, cancel := context.WithTimeout(ctx, timeout)
			defer cancel()
			r := req
			r.K = sp.K
			t0 := time.Now()
			var out recall.Output
			call := func() error {
				var err error
				out, err = sp.Source.Recall(sctx, r)
				return err
			}
			var err error
			if sp.Breaker != nil {
				err = sp.Breaker.Do(ctx, call)
			} else {
				err = call()
			}
			res := Result{Source: name, Items: out.Items, Version: out.Version, AdScores: out.AdScores, Status: StatusOK, Took: time.Since(t0)}
			switch {
			case err == nil:
			case errors.Is(err, resilience.ErrOpen):
				res.Status, res.Err, res.Items = StatusOpen, err, nil
			case sctx.Err() != nil, status.Code(err) == codes.DeadlineExceeded:
				// The model server gives up a moment before our own context fires when it
				// sees the propagated deadline run out: the same timeout.
				res.Status, res.Err, res.Items = StatusTimeout, ErrTimeout, nil
			default:
				res.Status, res.Err, res.Items = StatusError, err, nil
			}
			ch <- done{i, res}
		}()
	}
	if pending == 0 {
		return results
	}
	timer := time.NewTimer(wait)
	defer timer.Stop()
	for ; pending > 0; pending-- {
		select {
		case d := <-ch:
			results[d.i], got[d.i] = d.r, true
			observe(d.r)
		case <-timer.C:
			for i, ok := range got {
				if !ok {
					results[i] = Result{Source: specs[i].Source.Name(), Status: StatusTimeout, Err: ErrTimeout, Took: wait}
					observe(results[i])
				}
			}
			return results
		case <-ctx.Done():
			for i, ok := range got {
				if !ok {
					results[i] = Result{Source: specs[i].Source.Name(), Status: StatusTimeout, Err: ctx.Err()}
					observe(results[i])
				}
			}
			return results
		}
	}
	return results
}

// FanOutNaive queries the sources concurrently without timeouts or breakers and waits for all
// of them. This is how the path looks without §5.1, kept to measure the difference.
func FanOutNaive(ctx context.Context, specs []Spec, req recall.Request) []Result {
	results := make([]Result, len(specs))
	ch := make(chan struct{}, len(specs))
	for i, sp := range specs {
		go func() {
			defer func() { ch <- struct{}{} }()
			r := req
			r.K = sp.K
			t0 := time.Now()
			out, err := sp.Source.Recall(ctx, r)
			res := Result{Source: sp.Source.Name(), Items: out.Items, Version: out.Version, AdScores: out.AdScores, Status: StatusOK, Took: time.Since(t0)}
			if err != nil {
				res.Status, res.Err, res.Items = StatusError, err, nil
			}
			results[i] = res
			observe(res)
		}()
	}
	for range specs {
		<-ch
	}
	return results
}
