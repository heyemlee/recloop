package experiment

import (
	"context"
	"log/slog"
	"sync/atomic"
	"time"
)

// FetchFunc returns the running experiments (from the experiments table).
type FetchFunc func(ctx context.Context) ([]Experiment, error)

// Loader keeps the current Config, refreshed periodically (design §8: every 10 s). A failed or
// invalid refresh keeps the previous snapshot.
type Loader struct {
	fetch FetchFunc
	cur   atomic.Pointer[Config]
	log   *slog.Logger
}

func NewLoader(fetch FetchFunc, log *slog.Logger) *Loader {
	return &Loader{fetch: fetch, log: log}
}

// Refresh loads a new snapshot; on error the previous one stays in place.
func (l *Loader) Refresh(ctx context.Context) error {
	exps, err := l.fetch(ctx)
	if err != nil {
		return err
	}
	cfg, err := NewConfig(exps)
	if err != nil {
		return err
	}
	if old := l.cur.Swap(cfg); old == nil || old.Version != cfg.Version {
		l.log.Info("experiment config loaded", "config_version", cfg.Version, "experiments", len(exps))
	}
	return nil
}

// Current returns the latest snapshot (nil before the first successful refresh).
func (l *Loader) Current() *Config { return l.cur.Load() }

// Run refreshes every `every` until ctx is done.
func (l *Loader) Run(ctx context.Context, every time.Duration) {
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			if err := l.Refresh(ctx); err != nil {
				l.log.Warn("experiment config refresh failed; keeping the previous one", "err", err)
			}
		}
	}
}
