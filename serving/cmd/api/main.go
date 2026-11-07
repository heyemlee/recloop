// Command api is RecLoop's online HTTP service (design §3, §4, §7, §8, §13.1).
//
// Configuration (environment):
//
//	API_ADDR             listen address                   default 127.0.0.1:18080
//	DATABASE_URL         PostgreSQL                       default postgres://recloop:recloop@localhost:15432/recloop?sslmode=disable
//	REDIS_URL            Redis                            default redis://localhost:16390/0
//	MODEL_SERVER_ADDR    gRPC model server                default 127.0.0.1:18081
//	RECLOOP_BUNDLE       bundle directory                 default artifacts/bundles/current
//	SASREC_TIMEOUT_MS    SASRec recall timeout            default 80
//	TEXT_TIMEOUT_MS      text-similarity recall timeout   default 50
package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/redis/go-redis/v9"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"

	pb "github.com/heyemlee/recloop/serving/gen/recloop/modelserver/v1"
	"github.com/heyemlee/recloop/serving/internal/api"
	"github.com/heyemlee/recloop/serving/internal/bundle"
	"github.com/heyemlee/recloop/serving/internal/cache"
	"github.com/heyemlee/recloop/serving/internal/catalog"
	"github.com/heyemlee/recloop/serving/internal/events"
	"github.com/heyemlee/recloop/serving/internal/experiment"
	"github.com/heyemlee/recloop/serving/internal/mixer"
	"github.com/heyemlee/recloop/serving/internal/recall"
	"github.com/heyemlee/recloop/serving/internal/store"
	"github.com/heyemlee/recloop/serving/internal/userstate"
)

// TextModel is the sentence encoder behind items.text_emb (ml/data/text_embed.py).
const TextModel = "bge-small-en-v1.5"

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envMillis(key string, def int) time.Duration {
	if n, err := strconv.Atoi(os.Getenv(key)); err == nil && n > 0 {
		return time.Duration(n) * time.Millisecond
	}
	return time.Duration(def) * time.Millisecond
}

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	if err := run(log); err != nil {
		log.Error("api stopped", "err", err)
		os.Exit(1)
	}
}

func run(log *slog.Logger) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	b, err := bundle.Load(env("RECLOOP_BUNDLE", "artifacts/bundles/current"))
	if err != nil {
		return err
	}
	log.Info("bundle loaded", "version", b.Version, "dir", b.Dir, "items", b.NumItems)

	st, err := store.Open(ctx, env("DATABASE_URL", "postgres://recloop:recloop@localhost:15432/recloop?sslmode=disable"))
	if err != nil {
		return err
	}
	defer st.Close()
	cat, err := catalog.Load(ctx, st, log)
	if err != nil {
		return err
	}
	if cat.Len() != b.NumItems {
		return errors.New("catalog and bundle disagree on the number of items: run `make ingest`")
	}
	if err := st.RegisterActiveModel(ctx, b.Version, b.Dir, b.Manifest); err != nil {
		return err
	}
	today := time.Now().UTC()
	for _, d := range []time.Time{today, today.AddDate(0, 0, 1)} {
		if err := st.EnsurePartition(ctx, d); err != nil {
			return err
		}
	}

	ropts, err := redis.ParseURL(env("REDIS_URL", "redis://localhost:16390/0"))
	if err != nil {
		return err
	}
	rdb := redis.NewClient(ropts)
	defer rdb.Close()
	if err := rdb.Ping(ctx).Err(); err != nil {
		log.Warn("redis unreachable; requests degrade until it is back", "err", err)
	}

	conn, err := grpc.NewClient(env("MODEL_SERVER_ADDR", "127.0.0.1:18081"),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return err
	}
	defer conn.Close()
	ms := pb.NewModelServerClient(conn)
	health := healthpb.NewHealthClient(conn)

	exps := experiment.NewLoader(st.RunningExperiments, log)
	if err := exps.Refresh(ctx); err != nil {
		return err
	}

	bg, cancelBG := context.WithCancel(context.Background())
	served := events.NewServedWriter(st, 4096, log)
	go served.Run(bg, 200, 200*time.Millisecond)
	go cat.Run(bg, time.Minute, 10*time.Minute)
	go exps.Run(bg, 10*time.Second)

	users := userstate.Store{R: rdb}
	srv := &api.Server{
		Catalog: cat,
		Bundle:  b,
		Users:   users,
		Pages:   cache.Pages{R: rdb},
		Exps:    exps,
		Events: &events.Service{
			DB: st, State: users, OnStock: cat.SetStock, Now: time.Now,
		},
		Served: served,
		Redis:  rdb,
		Sources: map[string][]mixer.Spec{
			recall.SurfaceHome: {
				{Source: recall.SASRec{Client: ms}, Timeout: envMillis("SASREC_TIMEOUT_MS", 80), K: 200},
				{Source: recall.ItemCF{B: b}, Timeout: 20 * time.Millisecond, K: 200},
				{Source: recall.Pop{B: b}, Timeout: 20 * time.Millisecond, K: 200},
			},
			recall.SurfaceItem: {
				{Source: recall.ItemCF{B: b}, Timeout: 20 * time.Millisecond, K: 100},
				{Source: recall.Text{S: st, Version: TextModel}, Timeout: envMillis("TEXT_TIMEOUT_MS", 50), K: 50},
			},
		},
		Weights: map[string]map[string]float64{
			recall.SurfaceHome: {"sasrec": 1.0, "itemcf": 1.0, "pop": 0.3},
			recall.SurfaceItem: {"itemcf": 1.0, "text": 1.0},
		},
		Ready: func(ctx context.Context) map[string]string {
			ctx, cancel := context.WithTimeout(ctx, time.Second)
			defer cancel()
			out := map[string]string{"postgres": "ok", "redis": "ok", "model_server": "ok"}
			if err := st.Pool.Ping(ctx); err != nil {
				out["postgres"] = err.Error()
			}
			if err := rdb.Ping(ctx).Err(); err != nil {
				out["redis"] = err.Error()
			}
			if r, err := health.Check(ctx, &healthpb.HealthCheckRequest{}); err != nil {
				out["model_server"] = err.Error()
			} else if r.Status != healthpb.HealthCheckResponse_SERVING {
				out["model_server"] = r.Status.String()
			}
			return out
		},
		Log: log,
		Now: time.Now,
	}

	addr := env("API_ADDR", "127.0.0.1:18080")
	httpSrv := &http.Server{Addr: addr, Handler: srv.Handler(), ReadHeaderTimeout: 5 * time.Second}
	errc := make(chan error, 1)
	go func() { errc <- httpSrv.ListenAndServe() }()
	log.Info("api listening", "addr", addr, "bundle", b.Version, "config_version", exps.Current().Version)

	select {
	case err := <-errc:
		cancelBG()
		return err
	case <-ctx.Done():
	}
	log.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	err = httpSrv.Shutdown(shutdownCtx)
	cancelBG()
	served.Wait()
	written, dropped := served.Stats()
	log.Info("served events", "written", written, "dropped", dropped)
	return err
}
