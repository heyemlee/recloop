// Command billing charges sponsored clicks (design §9.3) and reconciles spend (§9.4).
//
//	billing             consume `events` as group "billing": every click with an ad_token goes
//	                    through billing.Service.Handle, and a batch's offsets are committed only
//	                    after all of its clicks were handled; reconcile every RECONCILE_EVERY
//	billing reconcile   one reconciliation pass, printed as JSON, then exit
//
// Configuration (environment; defaults in parentheses):
//
//	KAFKA_BROKERS          Redpanda seed brokers (localhost:19092)
//	DATABASE_URL           PostgreSQL, the ledger (postgres://recloop:recloop@localhost:15432/recloop?sslmode=disable)
//	REDIS_URL              Redis, the spend counters ad serving reads (redis://localhost:16390/0)
//	AD_TOKEN_KEYS          ad_token keys, kid:secret[,kid:secret...] (required)
//	BILLING_METRICS_ADDR   Prometheus /metrics (127.0.0.1:18092)
//	BILLING_INSTANCE_ID    static group membership (billing-0)
//	BILLING_WORKERS        clicks of one batch handled concurrently (8)
//	RECONCILE_EVERY        reconciliation period (1m; 0 turns it off)
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/twmb/franz-go/pkg/kgo"

	"github.com/heyemlee/recloop/serving/internal/adtoken"
	"github.com/heyemlee/recloop/serving/internal/billing"
	"github.com/heyemlee/recloop/serving/internal/consumer"
	"github.com/heyemlee/recloop/serving/internal/events"
	"github.com/heyemlee/recloop/serving/internal/metrics"
	"github.com/heyemlee/recloop/serving/internal/store"
)

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func main() {
	cmd := "run"
	if len(os.Args) > 1 {
		cmd = os.Args[1]
	}
	// `billing reconcile` prints its report on stdout: its log goes to stderr.
	out := os.Stdout
	if cmd != "run" {
		out = os.Stderr
	}
	log := slog.New(slog.NewJSONHandler(out, nil))
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	var err error
	switch cmd {
	case "run":
		err = run(ctx, log)
	case "reconcile":
		err = reconcileOnce(ctx, log)
	default:
		err = fmt.Errorf("unknown command %q (run, reconcile)", cmd)
	}
	if err != nil {
		log.Error("billing", "cmd", cmd, "err", err)
		os.Exit(1)
	}
}

type deps struct {
	st  *store.Store
	rdb *redis.Client
}

func open(ctx context.Context) (deps, error) {
	st, err := store.Open(ctx, env("DATABASE_URL", "postgres://recloop:recloop@localhost:15432/recloop?sslmode=disable"))
	if err != nil {
		return deps{}, err
	}
	opts, err := redis.ParseURL(env("REDIS_URL", "redis://localhost:16390/0"))
	if err != nil {
		st.Close()
		return deps{}, err
	}
	return deps{st: st, rdb: redis.NewClient(opts)}, nil
}

func (d deps) close() {
	d.rdb.Close()
	d.st.Close()
}

func run(ctx context.Context, log *slog.Logger) error {
	keys, err := adtoken.KeysFromEnv()
	if err != nil {
		return err
	}
	// The database may still be starting (compose restarts billing with the stack): retry.
	var d deps
	for attempt := 0; ; attempt++ {
		if d, err = open(ctx); err == nil {
			break
		}
		if attempt == 30 || ctx.Err() != nil {
			return err
		}
		log.Warn("postgres unreachable; retrying", "err", err)
		time.Sleep(time.Second)
	}
	defer d.close()

	workers, err := strconv.Atoi(env("BILLING_WORKERS", "8"))
	if err != nil || workers < 1 {
		return errors.New("BILLING_WORKERS must be a positive integer")
	}
	every, err := time.ParseDuration(env("RECONCILE_EVERY", billing.ReconcileEvery.String()))
	if err != nil {
		return fmt.Errorf("RECONCILE_EVERY: %w", err)
	}

	brokers := strings.Split(env("KAFKA_BROKERS", "localhost:19092"), ",")
	// Static membership, as the aggregator: a restarted billing (kill -9, deploy) takes its
	// partitions back at once instead of waiting for its old session to time out.
	cl, err := kgo.NewClient(append(consumer.Options(brokers, billing.Group, events.Topic),
		kgo.ClientID("recloop-billing"), kgo.InstanceID(env("BILLING_INSTANCE_ID", "billing-0")))...)
	if err != nil {
		return err
	}
	defer cl.CloseAllowingRebalance()

	mux := http.NewServeMux()
	mux.Handle("GET /metrics", metrics.Handler())
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte("ok\n")) })
	addr := env("BILLING_METRICS_ADDR", "127.0.0.1:18092")
	go func() {
		if err := (&http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 5 * time.Second}).ListenAndServe(); err != nil {
			log.Error("metrics server", "err", err)
		}
	}()

	svc := &billing.Service{DB: d.st.Pool, Keys: keys, Spend: billing.RedisSpend{R: d.rdb}, Log: log}
	if every > 0 {
		rec := &billing.Reconciler{DB: d.st.Pool, Redis: d.rdb, Log: log}
		go rec.Run(ctx, every)
	}
	cons := &billing.Consumer{Service: svc, Workers: workers, Log: log}
	loop := &consumer.Loop{Client: cl, Group: billing.Group, Handle: cons.Handle, MaxPoll: 1000, Log: log}
	log.Info("billing consuming", "topic", events.Topic, "group", billing.Group, "brokers", brokers,
		"workers", workers, "reconcile_every", every.String(), "metrics", addr, "signing_kid", keys.SigningKeyID())
	return loop.Run(ctx)
}

func reconcileOnce(ctx context.Context, log *slog.Logger) error {
	d, err := open(ctx)
	if err != nil {
		return err
	}
	defer d.close()
	rec := &billing.Reconciler{DB: d.st.Pool, Redis: d.rdb, Log: log}
	rep, err := rec.Pass(ctx)
	if err != nil {
		return err
	}
	out, _ := json.MarshalIndent(rep, "", "  ")
	fmt.Println(string(out))
	fmt.Fprintln(os.Stderr, "reconcile: "+rep.String())
	if !rep.OK() {
		return errors.New("the ledger, the budget rows and Redis disagree")
	}
	return nil
}
