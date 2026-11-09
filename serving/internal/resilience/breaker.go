package resilience

import (
	"context"
	"errors"
	"fmt"
	"net"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/sony/gobreaker/v2"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// BreakerConfig is design §5.1's rule: open when at least FailureRatio of the recent calls
// failed or timed out, or after Consecutive failures in a row; stay open for OpenFor, then let
// HalfOpenProbes calls through: that many successes in a row close it again, one failure
// re-opens it.
//
// "Recent" is the calls of the last Period if there were at least MinCalls of them (a rolling
// time window, as Hystrix and resilience4j's time-based windows do; the default since the user
// chose it on 2026-11-06), or the last Window calls (Period == 0, §5.1 as first written). The
// count window spans a few tens of milliseconds at hundreds of calls per second: under a
// partial fault (30 % slow calls) it often holds 10 failures of 20 and the breaker keeps
// opening. The consecutive-failure trip lets an outage open the breaker after a handful of calls
// even when traffic is too low to reach MinCalls quickly; a run of 10 failures is very unlikely
// at a 30 % failure rate (0.3^10).
type BreakerConfig struct {
	Window         int
	Period         time.Duration
	MinCalls       int
	FailureRatio   float64
	Consecutive    uint32 // 0: no consecutive-failure trip
	OpenFor        time.Duration
	HalfOpenProbes uint32
}

// DefaultBreaker: the last 10 s with at least 20 calls, 50 %, or 10 failures in a row; open
// 10 s, 5 probes.
var DefaultBreaker = BreakerConfig{
	Period: 10 * time.Second, MinCalls: 20, FailureRatio: 0.5, Consecutive: 10,
	OpenFor: 10 * time.Second, HalfOpenProbes: 5,
}

// ParseWindow reads a window spec onto DefaultBreaker: "time:10s:20" (the last 10 seconds, at
// least 20 calls), optionally ":C" for the consecutive-failure trip (default 10, 0 = off) and
// then ":P" for the half-open probes (default 5: that many successes in a row close it again),
// or "calls:20" (the last 20 calls, no consecutive trip: §5.1 as first written). The api reads
// it from BREAKER_WINDOW for every breaker (model server, rank, PostgreSQL text recall).
func ParseWindow(spec string) (BreakerConfig, error) {
	c := DefaultBreaker
	parts := strings.Split(spec, ":")
	switch {
	case len(parts) == 2 && parts[0] == "calls":
		n, err := strconv.Atoi(parts[1])
		if err != nil || n <= 0 {
			return c, fmt.Errorf("breaker window %q: bad call count", spec)
		}
		c.Window, c.Period, c.MinCalls, c.Consecutive = n, 0, 0, 0
		return c, nil
	case len(parts) >= 3 && len(parts) <= 5 && parts[0] == "time":
		d, err := time.ParseDuration(parts[1])
		if err != nil || d <= 0 {
			return c, fmt.Errorf("breaker window %q: bad period", spec)
		}
		n, err := strconv.Atoi(parts[2])
		if err != nil || n <= 0 {
			return c, fmt.Errorf("breaker window %q: bad minimum call count", spec)
		}
		c.Window, c.Period, c.MinCalls = 0, d, n
		if len(parts) >= 4 {
			k, err := strconv.ParseUint(parts[3], 10, 32)
			if err != nil {
				return c, fmt.Errorf("breaker window %q: bad consecutive-failure count", spec)
			}
			c.Consecutive = uint32(k)
		}
		if len(parts) == 5 {
			p, err := strconv.ParseUint(parts[4], 10, 32)
			if err != nil || p == 0 {
				return c, fmt.Errorf("breaker window %q: bad half-open probe count", spec)
			}
			c.HalfOpenProbes = uint32(p)
		}
		return c, nil
	}
	return c, fmt.Errorf("breaker window %q: want time:DURATION:N[:C[:P]] or calls:N", spec)
}

// String describes the window ("time:10s:20:10", "calls:20"); the half-open probes only when
// they differ from the default ("time:10s:20:25:2").
func (c BreakerConfig) String() string {
	if c.Period > 0 {
		s := fmt.Sprintf("time:%s:%d:%d", c.Period, c.MinCalls, c.Consecutive)
		if c.HalfOpenProbes != DefaultBreaker.HalfOpenProbes {
			s += fmt.Sprintf(":%d", c.HalfOpenProbes)
		}
		return s
	}
	return fmt.Sprintf("calls:%d", c.Window)
}

// ErrOpen is returned without calling the dependency while the breaker is open (or while the
// half-open probes are taken).
var ErrOpen = errors.New("circuit open")

// errIgnored tells gobreaker to leave a call out of its counts.
var errIgnored = errors.New("ignored")

// State of a breaker, as exported in metrics: 0 closed, 1 half-open, 2 open.
type State int

const (
	Closed   State = 0
	HalfOpen State = 1
	Open     State = 2
)

func (s State) String() string {
	switch s {
	case HalfOpen:
		return "half_open"
	case Open:
		return "open"
	}
	return "closed"
}

func fromGobreaker(s gobreaker.State) State {
	switch s {
	case gobreaker.StateHalfOpen:
		return HalfOpen
	case gobreaker.StateOpen:
		return Open
	}
	return Closed
}

// Breaker guards one dependency. gobreaker keeps the state machine. With a count window the
// trip decision looks at the last Window outcomes (kept here: gobreaker's own counts are
// time-based); with a time window it uses gobreaker's rolling counts.
type Breaker struct {
	name string
	cb   *gobreaker.TwoStepCircuitBreaker[struct{}]
	win  *window
}

// NewBreaker builds a breaker; onChange (may be nil) sees every state transition.
func NewBreaker(name string, cfg BreakerConfig, onChange func(name string, from, to State)) *Breaker {
	if cfg.Window <= 0 && cfg.Period <= 0 {
		cfg = DefaultBreaker
	}
	st := gobreaker.Settings{
		Name:         name,
		MaxRequests:  cfg.HalfOpenProbes,
		Timeout:      cfg.OpenFor,
		IsSuccessful: func(err error) bool { return err == nil },
		IsExcluded:   func(err error) bool { return errors.Is(err, errIgnored) },
	}
	b := &Breaker{name: name}
	streak := func(c gobreaker.Counts) bool {
		return cfg.Consecutive > 0 && c.ConsecutiveFailures >= cfg.Consecutive
	}
	if cfg.Period > 0 {
		// gobreaker's rolling window: Period in ten buckets.
		st.Interval, st.BucketPeriod = cfg.Period, cfg.Period/10
		st.ReadyToTrip = func(c gobreaker.Counts) bool {
			done := c.TotalSuccesses + c.TotalFailures
			return streak(c) || int(done) >= cfg.MinCalls && float64(c.TotalFailures) >= cfg.FailureRatio*float64(done)
		}
	} else {
		b.win = newWindow(cfg.Window)
		threshold := int(float64(cfg.Window)*cfg.FailureRatio + 0.999999)
		st.ReadyToTrip = func(c gobreaker.Counts) bool { return streak(c) || b.win.failures() >= threshold }
	}
	st.OnStateChange = func(_ string, from, to gobreaker.State) {
		if b.win != nil {
			b.win.reset()
		}
		if onChange != nil {
			onChange(name, fromGobreaker(from), fromGobreaker(to))
		}
	}
	b.cb = gobreaker.NewTwoStepCircuitBreaker[struct{}](st)
	return b
}

// Name of the dependency.
func (b *Breaker) Name() string { return b.name }

// State now.
func (b *Breaker) State() State { return fromGobreaker(b.cb.State()) }

// Do calls f unless the breaker is open. parent is the request's own context: a call that
// failed because the request itself was cancelled (the client went away) is not counted
// against the dependency.
func (b *Breaker) Do(parent context.Context, f func() error) error {
	done, err := b.cb.Allow()
	if err != nil {
		return ErrOpen
	}
	err = f()
	switch Classify(parent, err) {
	case Success:
		if b.win != nil {
			b.win.record(false)
		}
		done(nil)
	case Failure:
		if b.win != nil {
			b.win.record(true)
		}
		done(err)
	default: // neither a success nor a failure of the dependency
		done(errIgnored)
	}
	return err
}

// Outcome of a call, from the dependency's point of view.
type Outcome int

const (
	Success Outcome = iota
	Failure
	Ignored
)

// Classify decides whether err says something about the dependency's health. Timeouts,
// unavailability and server errors count as failures; a request the client cancelled is
// ignored; errors that blame the request (bad argument, unknown pinned version) are a healthy
// dependency answering.
func Classify(parent context.Context, err error) Outcome {
	if err == nil {
		return Success
	}
	if parent != nil && errors.Is(parent.Err(), context.Canceled) {
		return Ignored
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return Failure
	}
	if errors.Is(err, context.Canceled) {
		return Ignored
	}
	var ne net.Error
	if errors.As(err, &ne) {
		return Failure
	}
	if s, ok := status.FromError(err); ok {
		switch s.Code() {
		case codes.InvalidArgument, codes.NotFound, codes.FailedPrecondition, codes.OutOfRange,
			codes.AlreadyExists, codes.PermissionDenied, codes.Unauthenticated, codes.Unimplemented:
			return Success
		case codes.Canceled:
			return Ignored
		}
	}
	return Failure
}

// window remembers the outcomes of the last n calls.
type window struct {
	mu    sync.Mutex
	buf   []bool
	next  int
	count int
	fails int
}

func newWindow(n int) *window { return &window{buf: make([]bool, n)} }

func (w *window) record(failed bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.count == len(w.buf) && w.buf[w.next] {
		w.fails--
	}
	w.buf[w.next] = failed
	if failed {
		w.fails++
	}
	w.next = (w.next + 1) % len(w.buf)
	if w.count < len(w.buf) {
		w.count++
	}
}

// failures among the last n calls; 0 until n calls have been seen, so a handful of early
// failures cannot trip the breaker.
func (w *window) failures() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.count < len(w.buf) {
		return 0
	}
	return w.fails
}

func (w *window) reset() {
	w.mu.Lock()
	defer w.mu.Unlock()
	clear(w.buf)
	w.next, w.count, w.fails = 0, 0, 0
}
