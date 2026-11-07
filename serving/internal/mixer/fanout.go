// Package mixer is the recommendation pipeline after the user context is read (design §4.1):
// concurrent fan-out to the recall sources with per-source timeouts, weighted RRF over whatever
// came back, filtering, brand diversity and popularity backfill.
//
// M1 has the skeleton of ADR-0001: fan-out + timeouts + merge of partial results. The request
// deadline budget, the degradation chain and circuit breakers are M2.
package mixer

import (
	"context"
	"errors"
	"time"

	"github.com/heyemlee/recloop/serving/internal/recall"
)

// Spec is a source with its timeout and candidate count.
type Spec struct {
	Source  recall.Source
	Timeout time.Duration
	K       int
}

// Result of one source.
type Result struct {
	Source  string
	Items   []recall.Candidate
	Version string
	Err     error
	Took    time.Duration
}

// ErrTimeout marks a source that did not answer within its timeout.
var ErrTimeout = errors.New("timeout")

// FanOut queries all sources concurrently, each under its own timeout, and returns one Result
// per spec, in spec order. It returns once every source has answered or the longest timeout has
// passed: a source that has not answered by then is reported with ErrTimeout and dropped, so a
// slow or stuck source never holds the response.
func FanOut(ctx context.Context, specs []Spec, req recall.Request) []Result {
	type done struct {
		i int
		r Result
	}
	ch := make(chan done, len(specs))
	var wait time.Duration
	for i, sp := range specs {
		wait = max(wait, sp.Timeout)
		go func() {
			sctx, cancel := context.WithTimeout(ctx, sp.Timeout)
			defer cancel()
			r := req
			r.K = sp.K
			t0 := time.Now()
			out, err := sp.Source.Recall(sctx, r)
			if err != nil && sctx.Err() != nil {
				err = ErrTimeout
			}
			ch <- done{i, Result{Source: sp.Source.Name(), Items: out.Items, Version: out.Version, Err: err, Took: time.Since(t0)}}
		}()
	}
	results := make([]Result, len(specs))
	got := make([]bool, len(specs))
	timer := time.NewTimer(wait)
	defer timer.Stop()
	for n := 0; n < len(specs); n++ {
		select {
		case d := <-ch:
			results[d.i], got[d.i] = d.r, true
			if d.r.Err != nil {
				results[d.i].Items = nil
			}
		case <-timer.C:
			for i, ok := range got {
				if !ok {
					results[i] = Result{Source: specs[i].Source.Name(), Err: ErrTimeout, Took: wait}
				}
			}
			return results
		case <-ctx.Done():
			for i, ok := range got {
				if !ok {
					results[i] = Result{Source: specs[i].Source.Name(), Err: ctx.Err()}
				}
			}
			return results
		}
	}
	return results
}
