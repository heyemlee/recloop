package userstate

import (
	"context"
	"errors"
	"log/slog"
	"net"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/heyemlee/recloop/serving/internal/metrics"
)

// The state Redis never evicts (design §19, 2026-11-08: maxmemory-policy noeviction). When it is
// full it refuses writes ("OOM command not allowed when used memory > 'maxmemory'"); a script
// (apply.lua, spend.lua) is refused at its first write, so nothing is half applied. The writers
// retry or leave the repair to reconciliation; this file makes the condition visible.

// Error kinds of RedisErrors (metric recloop_redis_errors_total).
const (
	ErrKindOOM     = "oom"
	ErrKindTimeout = "timeout"
	ErrKindOther   = "other"
)

// ErrorKind classifies a failed Redis call for the metrics: oom when the instance is full, timeout
// for deadlines, other for the rest (connection refused, NOSCRIPT, ...).
func ErrorKind(err error) string {
	var ne net.Error
	switch {
	case err == nil:
		return ""
	case IsOOM(err):
		return ErrKindOOM
	case errors.Is(err, context.DeadlineExceeded), errors.As(err, &ne) && ne.Timeout():
		return ErrKindTimeout
	}
	return ErrKindOther
}

// IsOOM reports whether Redis refused a write because it is at maxmemory, also from inside a
// script (the error then names the script and line).
func IsOOM(err error) bool {
	return err != nil && (redis.HasErrorPrefix(err, "OOM") || strings.Contains(err.Error(), "used memory > 'maxmemory'"))
}

// CountError adds a failed write to recloop_redis_errors_total; nil does nothing.
func CountError(component, op string, err error) {
	if err != nil {
		metrics.RedisErrors.WithLabelValues(component, op, ErrorKind(err)).Inc()
	}
}

// Info is what Fullness reads from INFO memory and INFO errorstats.
type Info struct {
	UsedMemory int64 // used_memory
	MaxMemory  int64 // maxmemory (0: no limit)
	OOMErrors  int64 // errorstat_OOM count: commands refused at maxmemory since the start
}

// ParseInfo reads the fields of Info from the text of INFO; missing fields stay 0.
func ParseInfo(raw string) Info {
	var in Info
	for _, line := range strings.Split(raw, "\n") {
		k, v, ok := strings.Cut(strings.TrimSpace(line), ":")
		if !ok {
			continue
		}
		switch k {
		case "used_memory":
			in.UsedMemory, _ = strconv.ParseInt(v, 10, 64)
		case "maxmemory":
			in.MaxMemory, _ = strconv.ParseInt(v, 10, 64)
		case "errorstat_OOM": // count=12
			if n, ok := strings.CutPrefix(v, "count="); ok {
				n, _, _ = strings.Cut(n, ",")
				in.OOMErrors, _ = strconv.ParseInt(n, 10, 64)
			}
		}
	}
	return in
}

// FullAt is the share of maxmemory from which Fullness reports the state Redis as full: a little
// below the limit, since a write that would cross it is refused.
const FullAt = 0.98

// Fullness tells whether the state Redis is full. What the aggregator and billing cannot write
// then (frequency caps, spend counters) stops counting, so the api serves no ads while it is
// (design §5.1: when unsure, no ads; degradation ads_state_full).
type Fullness struct {
	R   redis.UniversalClient
	Log *slog.Logger

	full    atomic.Bool
	lastOOM int64 // errorstat_OOM at the previous check (only Check touches it)
	checked bool
}

// Full reports the result of the last successful check (false before the first one).
func (f *Fullness) Full() bool { return f.full.Load() }

// Check reads INFO once. Full: used_memory at FullAt of maxmemory or more, or a write refused
// since the previous check. An error keeps the previous answer: if the instance is unreachable,
// reading the user's state fails too, and the degradation chain rules ads out anyway.
func (f *Fullness) Check(ctx context.Context) error {
	raw, err := f.R.Info(ctx, "memory", "errorstats").Result()
	if err != nil {
		return err
	}
	f.observe(ParseInfo(raw))
	return nil
}

// observe updates the answer from one INFO reading.
func (f *Fullness) observe(in Info) {
	full := in.MaxMemory > 0 && float64(in.UsedMemory) >= FullAt*float64(in.MaxMemory)
	if f.checked && in.OOMErrors > f.lastOOM {
		full = true
	}
	f.lastOOM, f.checked = in.OOMErrors, true
	if full != f.full.Swap(full) && f.Log != nil {
		if full {
			f.Log.Warn("the state Redis is full: its writes fail, no ads until it has room",
				"used_memory", in.UsedMemory, "maxmemory", in.MaxMemory, "oom_errors", in.OOMErrors)
		} else {
			f.Log.Info("the state Redis has room again", "used_memory", in.UsedMemory, "maxmemory", in.MaxMemory)
		}
	}
	if full {
		metrics.RedisStateFull.Set(1)
	} else {
		metrics.RedisStateFull.Set(0)
	}
}

// Run checks every `every` until ctx ends.
func (f *Fullness) Run(ctx context.Context, every time.Duration) {
	t := time.NewTicker(every)
	defer t.Stop()
	failing := false
	for {
		cctx, cancel := context.WithTimeout(ctx, time.Second)
		err := f.Check(cctx)
		cancel()
		if err != nil && !failing && ctx.Err() == nil && f.Log != nil {
			f.Log.Warn("reading the state Redis's memory failed", "err", err)
		}
		failing = err != nil
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}
