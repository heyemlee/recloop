package resilience

import (
	"context"
	"errors"
	"math/rand/v2"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestBudgetLeavesTheReserveForLaterStages(t *testing.T) {
	t0 := time.Unix(1000, 0)
	now := t0
	clock := func() time.Time { return now }
	_, cancel, b := Start(context.Background(), DefaultPlan, clock)
	defer cancel()

	if got := b.For(35*time.Millisecond, 35*time.Millisecond); got != 35*time.Millisecond {
		t.Fatalf("fresh request: recall budget %v, want its full 35ms", got)
	}
	now = t0.Add(50 * time.Millisecond) // a slow first stage took half the request
	if got := b.For(35*time.Millisecond, 35*time.Millisecond); got != 15*time.Millisecond {
		t.Fatalf("after 50ms: recall budget %v, want 100-50-35 = 15ms", got)
	}
	now = t0.Add(80 * time.Millisecond)
	if got := b.For(35*time.Millisecond, 35*time.Millisecond); got > 0 {
		t.Fatalf("after 80ms: recall budget %v, want <= 0 (skip the stage)", got)
	}
	if got := b.Left(); got != 20*time.Millisecond {
		t.Fatalf("left %v", got)
	}
}

func TestBreakerOpensOnHalfOfTheLastTwentyCalls(t *testing.T) {
	var transitions []string
	cfg := BreakerConfig{Window: 20, FailureRatio: 0.5, OpenFor: 50 * time.Millisecond, HalfOpenProbes: 2} // no consecutive trip
	b := NewBreaker("dep", cfg, func(_ string, from, to State) {
		transitions = append(transitions, from.String()+">"+to.String())
	})
	ctx := context.Background()
	fail := func() error { return context.DeadlineExceeded }
	ok := func() error { return nil }

	// 11 successes, then 9 failures: 9 of the last 20 calls failed, below 50 %.
	for i := 0; i < 20; i++ {
		f := ok
		if i >= 11 {
			f = fail
		}
		_ = b.Do(ctx, f)
	}
	if b.State() != Closed {
		t.Fatalf("9/20 failures opened the breaker")
	}
	// The next failure pushes the oldest success out of the window: 10 of 20.
	_ = b.Do(ctx, fail)
	if b.State() != Open {
		t.Fatalf("state %v after 10/20 failures", b.State())
	}
	called := false
	if err := b.Do(ctx, func() error { called = true; return nil }); !errors.Is(err, ErrOpen) || called {
		t.Fatalf("open breaker called the dependency (err %v)", err)
	}
	time.Sleep(60 * time.Millisecond)
	if b.State() != HalfOpen {
		t.Fatalf("state %v after OpenFor", b.State())
	}
	_ = b.Do(ctx, ok)
	_ = b.Do(ctx, ok)
	if b.State() != Closed {
		t.Fatalf("state %v after two good probes", b.State())
	}
	want := []string{"closed>open", "open>half_open", "half_open>closed"}
	if len(transitions) != len(want) {
		t.Fatalf("transitions %v", transitions)
	}
	for i := range want {
		if transitions[i] != want[i] {
			t.Fatalf("transitions %v, want %v", transitions, want)
		}
	}
}

func TestBreakerNeedsAFullWindow(t *testing.T) {
	b := NewBreaker("dep", BreakerConfig{Window: 20, FailureRatio: 0.5, OpenFor: time.Second, HalfOpenProbes: 1}, nil)
	for i := 0; i < 19; i++ {
		_ = b.Do(context.Background(), func() error { return errors.New("boom") })
	}
	if b.State() != Closed {
		t.Fatal("opened before 20 calls were seen")
	}
	_ = b.Do(context.Background(), func() error { return errors.New("boom") })
	if b.State() != Open {
		t.Fatal("20 failures did not open it")
	}
}

func TestClassify(t *testing.T) {
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	cases := []struct {
		parent context.Context
		err    error
		want   Outcome
	}{
		{context.Background(), nil, Success},
		{context.Background(), context.DeadlineExceeded, Failure},
		{context.Background(), status.Error(codes.DeadlineExceeded, "slow"), Failure},
		{context.Background(), status.Error(codes.Unavailable, "down"), Failure},
		{context.Background(), status.Error(codes.NotFound, "no such bundle"), Success},
		{context.Background(), status.Error(codes.InvalidArgument, "k < 0"), Success},
		{cancelled, context.Canceled, Ignored},
		{cancelled, status.Error(codes.Canceled, "client gone"), Ignored},
		{context.Background(), errors.New("anything else"), Failure},
	}
	for _, c := range cases {
		if got := Classify(c.parent, c.err); got != c.want {
			t.Errorf("Classify(%v) = %v, want %v", c.err, got, c.want)
		}
	}
}

func TestTimeWindowBreaker(t *testing.T) {
	cfg, err := ParseWindow("time:1s:20")
	if err != nil || cfg.Period != time.Second || cfg.MinCalls != 20 || cfg.Consecutive != 10 || cfg.String() != "time:1s:20:10" {
		t.Fatalf("%+v %v", cfg, err)
	}
	if def := DefaultBreaker.String(); def != "time:10s:20:10" {
		t.Fatalf("default %s", def)
	}
	cfg.OpenFor = time.Minute
	b := NewBreaker("dep", cfg, nil)
	ctx := context.Background()
	// 30 % failures over many calls: a slow tail, not an outage. A count window of 20 would
	// sometimes see 10 failures in a row of 20; the time window sees 30 %.
	for i := 0; i < 1000; i++ {
		f := func() error { return nil }
		if i%10 < 3 {
			f = func() error { return context.DeadlineExceeded }
		}
		_ = b.Do(ctx, f)
	}
	if b.State() != Closed {
		t.Fatal("30 % failures opened the time-window breaker")
	}
	for i := 0; i < 2000 && b.State() == Closed; i++ {
		_ = b.Do(ctx, func() error { return context.DeadlineExceeded })
	}
	if b.State() != Open {
		t.Fatal("an outage did not open it")
	}
	for _, bad := range []string{"", "calls:0", "time:x:3", "time:1s", "days:3", "time:1s:20:x", "calls:20:5"} {
		if _, err := ParseWindow(bad); err == nil {
			t.Errorf("window %q accepted", bad)
		}
	}
	if c, _ := ParseWindow("calls:20"); c.Window != 20 || c.Period != 0 || c.Consecutive != 0 || c.String() != "calls:20" {
		t.Fatalf("calls:20 -> %+v", c)
	}
}

// ":P" sets the half-open probes: that many successes in a row close an open breaker again
// (the comparison rule of make load: 25 failures in a row, 2 probes).
func TestHalfOpenProbes(t *testing.T) {
	cfg, err := ParseWindow("time:10s:20:25:2")
	if err != nil || cfg.Consecutive != 25 || cfg.HalfOpenProbes != 2 || cfg.String() != "time:10s:20:25:2" {
		t.Fatalf("%+v %v", cfg, err)
	}
	if c, _ := ParseWindow("time:10s:20:10:5"); c.String() != "time:10s:20:10" {
		t.Fatalf("the default probe count is left out: %s", c.String())
	}
	for _, bad := range []string{"time:10s:20:25:0", "time:10s:20:25:x", "time:10s:20:25:2:1"} {
		if _, err := ParseWindow(bad); err == nil {
			t.Errorf("window %q accepted", bad)
		}
	}
	ctx := context.Background()
	fail := func() error { return context.DeadlineExceeded }
	ok := func() error { return nil }
	for _, probes := range []string{"2", "5"} {
		cfg, _ := ParseWindow("time:10s:20:3:" + probes)
		cfg.OpenFor = 20 * time.Millisecond
		b := NewBreaker("dep", cfg, nil)
		for i := 0; i < 3; i++ {
			_ = b.Do(ctx, fail)
		}
		if b.State() != Open {
			t.Fatalf("probes %s: not open after 3 failures in a row", probes)
		}
		time.Sleep(30 * time.Millisecond)
		for i := 0; i < 2; i++ {
			_ = b.Do(ctx, ok)
		}
		if closed := b.State() == Closed; closed != (probes == "2") {
			t.Fatalf("probes %s: closed after 2 successes = %v", probes, closed)
		}
	}
}

func TestConsecutiveFailuresOpenTheBreakerAtLowTraffic(t *testing.T) {
	cfg, _ := ParseWindow("time:10s:20:5")
	cfg.OpenFor = time.Minute
	b := NewBreaker("dep", cfg, nil)
	// Healthy calls, then an outage: 5 failures in a row open it although the window has not
	// seen its 20 calls yet.
	for i := 0; i < 3; i++ {
		_ = b.Do(context.Background(), func() error { return nil })
	}
	for i := 0; i < 4; i++ {
		_ = b.Do(context.Background(), func() error { return context.DeadlineExceeded })
	}
	if b.State() != Closed {
		t.Fatal("opened before the 5th failure in a row")
	}
	_ = b.Do(context.Background(), func() error { return context.DeadlineExceeded })
	if b.State() != Open {
		t.Fatal("5 failures in a row did not open it")
	}
	off, _ := ParseWindow("time:10s:20:0")
	b = NewBreaker("dep", off, nil)
	for i := 0; i < 19; i++ {
		_ = b.Do(context.Background(), func() error { return context.DeadlineExceeded })
	}
	if b.State() != Closed {
		t.Fatal("consecutive trip 0 still tripped before MinCalls")
	}
}

func TestCountWindowFlapsUnderAPartialFault(t *testing.T) {
	// The same 30 % failure rate, in random order, against §5.1's original count window: some
	// windows of 20 calls reach 10 failures and the breaker opens.
	cfg, _ := ParseWindow("calls:20")
	cfg.OpenFor = time.Hour
	b := NewBreaker("dep", cfg, nil)
	rng := rand.New(rand.NewPCG(1, 2))
	opened := false
	for i := 0; i < 5000 && !opened; i++ {
		failed := rng.Float64() < 0.3
		_ = b.Do(context.Background(), func() error {
			if failed {
				return context.DeadlineExceeded
			}
			return nil
		})
		opened = b.State() == Open
	}
	if !opened {
		t.Fatal("expected the count window to trip on a 30 % random failure rate within 5000 calls")
	}
}
