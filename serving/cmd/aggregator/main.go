// Command aggregator consumes the event log into per-user state in Redis (design §7.2), and
// rebuilds that state when it is lost.
//
//	aggregator                  consume `events` as group "aggregator" (default)
//	aggregator rebuild-from-db  replay events_raw into Redis, then exit
//	aggregator reset-offsets    move the group to the oldest retained event (stop it first);
//	                            the next run replays the log into Redis
//	aggregator wait-lag GROUP [TIMEOUT]   wait (default 5m) until GROUP has consumed everything
//	                            produced so far
//
// Configuration (environment; defaults in parentheses):
//
//	KAFKA_BROKERS             Redpanda seed brokers (localhost:19092)
//	REDIS_URL                 Redis (redis://localhost:16390/0)
//	DATABASE_URL              PostgreSQL, for partitions and rebuild-from-db
//	AGGREGATOR_METRICS_ADDR   Prometheus /metrics (127.0.0.1:18090)
//	LAG_GROUPS                consumer groups whose lag is exported (aggregator,log-sink,billing)
//	AD_TOKEN_KEYS             ad_token keys: sponsored impressions count towards frequency caps
//	                          only when their token verifies (unset: no caps are counted)
package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/twmb/franz-go/pkg/kadm"
	"github.com/twmb/franz-go/pkg/kgo"

	"github.com/heyemlee/recloop/serving/internal/adtoken"
	"github.com/heyemlee/recloop/serving/internal/aggregator"
	"github.com/heyemlee/recloop/serving/internal/consumer"
	"github.com/heyemlee/recloop/serving/internal/events"
	"github.com/heyemlee/recloop/serving/internal/metrics"
	"github.com/heyemlee/recloop/serving/internal/store"
	"github.com/heyemlee/recloop/serving/internal/userstate"
)

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	cmd := "run"
	if len(os.Args) > 1 {
		cmd = os.Args[1]
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	var err error
	switch cmd {
	case "run":
		err = run(ctx, log)
	case "rebuild-from-db":
		err = rebuildFromDB(ctx, log)
	case "reset-offsets":
		err = resetOffsets(ctx, log)
	case "wait-lag":
		group, timeout := aggregator.Group, 5*time.Minute
		if len(os.Args) > 2 {
			group = os.Args[2]
		}
		if len(os.Args) > 3 {
			if timeout, err = time.ParseDuration(os.Args[3]); err != nil {
				break
			}
		}
		err = waitLag(ctx, group, timeout)
	default:
		err = fmt.Errorf("unknown command %q (run, rebuild-from-db, reset-offsets, wait-lag)", cmd)
	}
	if err != nil {
		log.Error("aggregator", "cmd", cmd, "err", err)
		os.Exit(1)
	}
}

func brokers() []string { return strings.Split(env("KAFKA_BROKERS", "localhost:19092"), ",") }

func redisStore() (userstate.Store, *redis.Client, error) {
	opts, err := redis.ParseURL(env("REDIS_URL", "redis://localhost:16390/0"))
	if err != nil {
		return userstate.Store{}, nil, err
	}
	rdb := redis.NewClient(opts)
	return userstate.Store{R: rdb, Group: aggregator.Group}, rdb, nil
}

// freqKeys is the frequency-cap rule of AD_TOKEN_KEYS (nil when unset).
func freqKeys(log *slog.Logger) (func(events.Event) string, error) {
	if os.Getenv(adtoken.EnvKeys) == "" {
		log.Warn(adtoken.EnvKeys + " is not set: sponsored impressions do not count towards frequency caps")
		return nil, nil
	}
	keys, err := adtoken.KeysFromEnv()
	if err != nil {
		return nil, err
	}
	return aggregator.FreqKeys(keys), nil
}

func run(ctx context.Context, log *slog.Logger) error {
	freq, err := freqKeys(log)
	if err != nil {
		return err
	}
	state, rdb, err := redisStore()
	if err != nil {
		return err
	}
	defer rdb.Close()

	// Static membership: a restarted aggregator (crash, deploy) takes its old place in the
	// group at once instead of waiting for the dead member's session to time out.
	cl, err := kgo.NewClient(append(consumer.Options(brokers(), aggregator.Group, events.Topic),
		kgo.ClientID("recloop-aggregator"), kgo.InstanceID(env("AGGREGATOR_INSTANCE_ID", "aggregator-0")))...)
	if err != nil {
		return err
	}
	defer cl.CloseAllowingRebalance()

	mux := http.NewServeMux()
	mux.Handle("GET /metrics", metrics.Handler())
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte("ok\n")) })
	addr := env("AGGREGATOR_METRICS_ADDR", "127.0.0.1:18090")
	go func() {
		if err := (&http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 5 * time.Second}).ListenAndServe(); err != nil {
			log.Error("metrics server", "err", err)
		}
	}()

	// The log-sink inserts into events_raw partitions by day: keep them in place here too, so
	// the pipeline works while the api is down.
	if url := os.Getenv("DATABASE_URL"); url != "" {
		if st, err := store.Open(ctx, url); err != nil {
			log.Warn("postgres unreachable; not maintaining events_raw partitions", "err", err)
		} else {
			defer st.Close()
			go st.MaintainPartitions(ctx, 10*time.Minute, func(err error) { log.Warn("events_raw partitions", "err", err) })
		}
	}

	adm := kadm.NewClient(cl)
	groups := strings.Split(env("LAG_GROUPS", "aggregator,log-sink,billing"), ",")
	go consumer.ExportLag(ctx, adm, events.Topic, groups, 5*time.Second, log)

	agg := &aggregator.Aggregator{State: state, FreqKey: freq, Now: time.Now, Log: log}
	loop := &consumer.Loop{Client: cl, Group: aggregator.Group, Handle: agg.Handle, MaxPoll: 1000, Log: log}
	log.Info("aggregator consuming", "topic", events.Topic, "group", aggregator.Group, "brokers", brokers(), "metrics", addr)
	return loop.Run(ctx)
}

func rebuildFromDB(ctx context.Context, log *slog.Logger) error {
	state, rdb, err := redisStore()
	if err != nil {
		return err
	}
	defer rdb.Close()
	st, err := store.Open(ctx, env("DATABASE_URL", "postgres://recloop:recloop@localhost:15432/recloop?sslmode=disable"))
	if err != nil {
		return err
	}
	defer st.Close()
	freq, err := freqKeys(log)
	if err != nil {
		return err
	}
	t0 := time.Now()
	agg := &aggregator.Aggregator{State: state, FreqKey: freq, Now: time.Now, Log: log}
	c, err := agg.Rebuild(ctx, st, 2000)
	if err != nil {
		return err
	}
	log.Info("rebuilt user state from events_raw", "applied", c.Applied, "already_applied", c.Duplicate,
		"no_effect", c.Ignored, "invalid", c.Invalid, "seconds", time.Since(t0).Seconds())
	fmt.Printf("rebuild-from-db: applied %d events (%d already applied, %d without effect) in %.1fs\n",
		c.Applied, c.Duplicate, c.Ignored, time.Since(t0).Seconds())
	return nil
}

func resetOffsets(ctx context.Context, log *slog.Logger) error {
	cl, err := kgo.NewClient(kgo.SeedBrokers(brokers()...), kgo.ClientID("recloop-aggregator-admin"))
	if err != nil {
		return err
	}
	defer cl.Close()
	n, err := consumer.ResetToStart(ctx, kadm.NewClient(cl), aggregator.Group, events.Topic)
	if err != nil {
		return err
	}
	if n == 0 {
		return errors.New("no partitions found for topic " + events.Topic)
	}
	log.Info("consumer group reset to the oldest retained events", "group", aggregator.Group, "partitions", n)
	fmt.Printf("reset-offsets: group %s now starts at the oldest retained event of %d partitions\n", aggregator.Group, n)
	return nil
}

// waitLag polls the group's lag on the events topic until it is 0 (every record produced so far
// has been committed by the group) or the timeout passes.
func waitLag(ctx context.Context, group string, timeout time.Duration) error {
	cl, err := kgo.NewClient(kgo.SeedBrokers(brokers()...), kgo.ClientID("recloop-aggregator-admin"))
	if err != nil {
		return err
	}
	defer cl.Close()
	adm := kadm.NewClient(cl)
	deadline := time.Now().Add(timeout)
	for {
		lags, err := adm.Lag(ctx, group)
		if err == nil {
			if l, ok := lags[group]; ok && l.Error() == nil {
				var total int64
				for _, pl := range l.Lag[events.Topic] {
					total += max(pl.Lag, 0)
				}
				if total == 0 {
					fmt.Printf("wait-lag: group %s has consumed everything\n", group)
					return nil
				}
				if time.Now().After(deadline) {
					return fmt.Errorf("group %s still %d records behind after %v", group, total, timeout)
				}
			}
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("lag of %s unknown after %v: %v", group, timeout, err)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(250 * time.Millisecond):
		}
	}
}
