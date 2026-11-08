// Command api is RecLoop's online HTTP service (design §3, §4, §5, §7, §8, §13.1).
//
// Configuration (environment; defaults in parentheses):
//
//	API_ADDR               listen address (127.0.0.1:18080)
//	DATABASE_URL           PostgreSQL (postgres://recloop:recloop@localhost:15432/recloop?sslmode=disable)
//	REDIS_URL              Redis (redis://localhost:16390/0)
//	MODEL_SERVER_ADDR      gRPC model server (127.0.0.1:18081)
//	RECLOOP_BUNDLE         bundle directory (artifacts/bundles/current)
//	EVENTS_MODE            redpanda: events go to the event log (default); direct: written to
//	                       PostgreSQL and Redis synchronously, M1's path, no Redpanda needed
//	KAFKA_BROKERS          Redpanda seed brokers, comma-separated (localhost:19092)
//	RECLOOP_RESILIENCE     full (default) or naive: no deadline, budgets, breakers or
//	                       degradation (the baseline of the fault experiments, design §5.2)
//	CANDIDATE_CACHE        on (default) or off
//	BUDGET_*_MS            request budget: TOTAL 100, USER_CONTEXT 5, CACHE_GET 3, RECALL 35,
//	                       RANK 25, ADS 10, TAIL 10 (design §4.1, §5.1)
//	ADS                    on (default): sponsored slots 3 and 9 of the home feed and
//	                       /internal/ads/select (design §9.2); off: none
//	AD_TOKEN_KEYS          ad_token keys, kid:secret[,kid:secret...] (no keys: no ads)
//	ADS_SAFETY_MARGIN_CLICKS  stop a campaign when its budget left is at most this many clicks
//	                       at its bid (5); an ads-layer experiment's "safety_margin_clicks"
//	                       overrides it per user
//	ADS_PACING             off (default) or on: spread a campaign's daily budget over its
//	                       flight window ("pacing" in the ads-layer experiment)
//	ADS_FIXED_PCTR         the stub pCTR every ad gets (0.01); ADS_MIN_PCTR the floor (0.001)
//	BREAKER_WINDOW         time:10s:20:10 (default, design §5.1: >= 50 % of the calls of the
//	                       last 10 s with at least 20 calls, or 10 failures in a row) or
//	                       calls:20 (the last 20 calls); open 10 s, 5 half-open probes
//	SASREC_TIMEOUT_MS      SASRec recall timeout, capped by the recall budget (35)
//	TEXT_TIMEOUT_MS        text-similarity recall timeout (35)
//	ACCESS_LOG             1 (default): one log line per request; 0 for load tests
//	LOG_LEVEL              debug, info (default), warn, error
//	OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_TRACES_SAMPLER(_ARG)   tracing (off when unset)
package main

import (
	"context"
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
	"go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"

	pb "github.com/heyemlee/recloop/serving/gen/recloop/modelserver/v1"
	"github.com/heyemlee/recloop/serving/internal/ads"
	"github.com/heyemlee/recloop/serving/internal/adtoken"
	"github.com/heyemlee/recloop/serving/internal/aggregator"
	"github.com/heyemlee/recloop/serving/internal/api"
	"github.com/heyemlee/recloop/serving/internal/bundle"
	"github.com/heyemlee/recloop/serving/internal/cache"
	"github.com/heyemlee/recloop/serving/internal/catalog"
	"github.com/heyemlee/recloop/serving/internal/events"
	"github.com/heyemlee/recloop/serving/internal/experiment"
	"github.com/heyemlee/recloop/serving/internal/metrics"
	"github.com/heyemlee/recloop/serving/internal/mixer"
	"github.com/heyemlee/recloop/serving/internal/recall"
	"github.com/heyemlee/recloop/serving/internal/resilience"
	"github.com/heyemlee/recloop/serving/internal/store"
	"github.com/heyemlee/recloop/serving/internal/tracing"
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

func logLevel() slog.Level {
	var l slog.Level
	if err := l.UnmarshalText([]byte(env("LOG_LEVEL", "info"))); err != nil {
		return slog.LevelInfo
	}
	return l
}

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: logLevel()}))
	if err := run(log); err != nil {
		log.Error("api stopped", "err", err)
		os.Exit(1)
	}
}

func plan() resilience.Plan {
	d := resilience.DefaultPlan
	return resilience.Plan{
		Total:       envMillis("BUDGET_TOTAL_MS", int(d.Total.Milliseconds())),
		UserContext: envMillis("BUDGET_USER_CONTEXT_MS", int(d.UserContext.Milliseconds())),
		CacheGet:    envMillis("BUDGET_CACHE_GET_MS", int(d.CacheGet.Milliseconds())),
		Recall:      envMillis("BUDGET_RECALL_MS", int(d.Recall.Milliseconds())),
		Rank:        envMillis("BUDGET_RANK_MS", int(d.Rank.Milliseconds())),
		Ads:         envMillis("BUDGET_ADS_MS", int(d.Ads.Milliseconds())),
		Tail:        envMillis("BUDGET_TAIL_MS", int(d.Tail.Milliseconds())),
	}
}

func envFloat(key string, def float64) (float64, error) {
	v := os.Getenv(key)
	if v == "" {
		return def, nil
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		return 0, fmt.Errorf("%s: %w", key, err)
	}
	return f, nil
}

// adServer sets up the sponsored slots (design §9.2): the campaign index (refreshed every
// ads.RefreshEvery in the background), the token keys, the stub pCTR and the default policy.
func adServer(ctx context.Context, bg context.Context, st *store.Store, log *slog.Logger) (*api.AdServer, error) {
	if env("ADS", "on") == "off" {
		return nil, nil
	}
	if os.Getenv(adtoken.EnvKeys) == "" {
		log.Warn(adtoken.EnvKeys + " is not set: no sponsored slots")
		return nil, nil
	}
	keys, err := adtoken.KeysFromEnv()
	if err != nil {
		return nil, err
	}
	margin, err := strconv.Atoi(env("ADS_SAFETY_MARGIN_CLICKS", "5"))
	if err != nil || margin < 0 {
		return nil, errors.New("ADS_SAFETY_MARGIN_CLICKS must be a non-negative integer")
	}
	pctr, err := envFloat("ADS_FIXED_PCTR", 0.01)
	if err != nil {
		return nil, err
	}
	minPCTR, err := envFloat("ADS_MIN_PCTR", 0.001)
	if err != nil {
		return nil, err
	}
	index := ads.NewIndex(func(ctx context.Context, today time.Time) ([]ads.Campaign, error) {
		return st.Campaigns(ctx, nil, today)
	}, time.Now, log)
	if err := index.Refresh(ctx); err != nil {
		log.Warn("campaigns not loaded yet; no ads until they are", "err", err)
	}
	go index.Run(bg, ads.RefreshEvery)
	return &api.AdServer{
		Index: index, PCTR: ads.FixedPCTR(pctr), Keys: keys,
		Policy: ads.Policy{SafetyMarginClicks: int32(margin), Pacing: env("ADS_PACING", "off") == "on", MinPCTR: minPCTR},
	}, nil
}

// spanName keeps span names to the route, not the concrete path.
func spanName(_ string, r *http.Request) string {
	p := r.URL.Path
	switch {
	case strings.HasPrefix(p, "/v1/items/"):
		p = "/v1/items/{id}"
	case strings.HasPrefix(p, "/v1/users/"):
		p = "/v1/users/{id}/context"
	}
	return r.Method + " " + p
}

func run(log *slog.Logger) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	shutdownTracing, tracingOn, err := tracing.Setup(ctx, "recloop-api")
	if err != nil {
		return err
	}
	defer func() {
		sctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = shutdownTracing(sctx)
	}()

	naive := env("RECLOOP_RESILIENCE", "full") == "naive"
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

	ropts, err := redis.ParseURL(env("REDIS_URL", "redis://localhost:16390/0"))
	if err != nil {
		return err
	}
	if !naive {
		// Deadlines come from the request budget; a retry inside a 5 ms budget is pointless.
		ropts.ContextTimeoutEnabled = true
		ropts.MaxRetries = -1
		ropts.DialTimeout = 100 * time.Millisecond
		ropts.PoolTimeout = 50 * time.Millisecond
		ropts.MinIdleConns = 8
	}
	rdb := redis.NewClient(ropts)
	defer rdb.Close()
	if err := rdb.Ping(ctx).Err(); err != nil {
		log.Warn("redis unreachable; requests degrade until it is back", "err", err)
	}

	gopts := []grpc.DialOption{grpc.WithTransportCredentials(insecure.NewCredentials())}
	if tracingOn {
		gopts = append(gopts, grpc.WithStatsHandler(otelgrpc.NewClientHandler()))
	}
	conn, err := grpc.NewClient(env("MODEL_SERVER_ADDR", "127.0.0.1:18081"), gopts...)
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
	defer cancelBG()
	users := userstate.Store{R: rdb}

	// Event log: Redpanda (default) or the synchronous M1 path.
	var (
		publisher  events.Publisher
		servedSink events.ServedSink
		kafka      *kgo.Client
	)
	switch mode := env("EVENTS_MODE", "redpanda"); mode {
	case "redpanda":
		kafka, err = kgo.NewClient(
			kgo.SeedBrokers(strings.Split(env("KAFKA_BROKERS", "localhost:19092"), ",")...),
			kgo.ClientID("recloop-api"),
			kgo.DefaultProduceTopic(events.Topic),
			// POST /v1/events waits for the acknowledgement: send at once instead of lingering
			// for a fuller batch (concurrent requests still share produce requests).
			kgo.ProducerLinger(0),
			kgo.RecordDeliveryTimeout(30*time.Second),
		)
		if err != nil {
			return err
		}
		defer kafka.Close()
		k := events.Kafka{Client: kafka, Topic: events.Topic}
		publisher, servedSink = k, k
	case "direct":
		direct := &events.Direct{DB: st, State: users, OnStock: cat.SetStock, Now: time.Now}
		if keys, err := adtoken.KeysFromEnv(); err == nil {
			direct.FreqKey = aggregator.FreqKeys(keys) // the aggregator's frequency-cap rule
		}
		publisher = direct
		servedSink = events.PGServed{DB: st}
	default:
		return errors.New("EVENTS_MODE must be redpanda or direct, not " + mode)
	}
	served := events.NewServedWriter(servedSink, 8192, log)
	go served.Run(bg, 200, 100*time.Millisecond)
	// Stock changes arrive asynchronously (purchase -> log-sink -> trigger): poll it often.
	go cat.Run(bg, 5*time.Second, 10*time.Minute)
	go exps.Run(bg, 10*time.Second)
	go st.MaintainPartitions(bg, 10*time.Minute, func(err error) { log.Warn("events_raw partitions", "err", err) })

	breakerChange := func(name string, from, to resilience.State) {
		metrics.BreakerState.WithLabelValues(name).Set(float64(to))
		metrics.BreakerTransitions.WithLabelValues(name, to.String()).Inc()
		log.Warn("circuit breaker", "dependency", name, "from", from.String(), "to", to.String())
	}
	// Breakers guard the calls whose failure is slow: the model server (up to 35 ms of the
	// budget per call) and the text search in PostgreSQL. Redis gets none: its calls are capped
	// at 3-10 ms by their stage budgets and fail at once when Redis is down, so a breaker would
	// save little and could open on a short burst of slow calls (seen on a cold start through
	// Docker Desktop's port forwarder: 10 s without user state for a few late replies).
	breakerCfg, err := resilience.ParseWindow(env("BREAKER_WINDOW", resilience.DefaultBreaker.String()))
	if err != nil {
		return err
	}
	var msBreaker, pgBreaker *resilience.Breaker
	if !naive {
		msBreaker = resilience.NewBreaker("model_server", breakerCfg, breakerChange)
		pgBreaker = resilience.NewBreaker("postgres", breakerCfg, breakerChange)
		for _, name := range []string{"model_server", "postgres"} {
			metrics.BreakerState.WithLabelValues(name).Set(0)
		}
	}
	var cands *cache.CandidateCache
	if env("CANDIDATE_CACHE", "on") != "off" {
		cands = &cache.CandidateCache{R: rdb, TTL: cache.CandidateTTL}
	}
	adSrv, err := adServer(ctx, bg, st, log)
	if err != nil {
		return err
	}

	srv := &api.Server{
		Catalog: cat,
		Bundle:  b,
		Users:   users,
		Pages:   cache.Pages{R: rdb},
		Cands:   cands,
		Exps:    exps,
		Events:  publisher,
		Served:  served,
		Redis:   rdb,
		Sources: map[string][]mixer.Spec{
			recall.SurfaceHome: {
				{Source: recall.SASRec{Client: ms}, Timeout: envMillis("SASREC_TIMEOUT_MS", 35), MinTimeout: 5 * time.Millisecond, K: 200, Breaker: msBreaker},
				{Source: recall.ItemCF{B: b}, Timeout: 10 * time.Millisecond, K: 200},
				{Source: recall.Pop{B: b}, Timeout: 10 * time.Millisecond, K: 200},
			},
			recall.SurfaceItem: {
				{Source: recall.ItemCF{B: b}, Timeout: 10 * time.Millisecond, K: 100},
				{Source: recall.Text{S: st, Version: TextModel}, Timeout: envMillis("TEXT_TIMEOUT_MS", 35), MinTimeout: 5 * time.Millisecond, K: 50, Breaker: pgBreaker},
			},
		},
		Weights: map[string]map[string]float64{
			recall.SurfaceHome: {"sasrec": 1.0, "itemcf": 1.0, "pop": 0.3},
			recall.SurfaceItem: {"itemcf": 1.0, "text": 1.0},
		},
		Plan:      plan(),
		Naive:     naive,
		AccessLog: env("ACCESS_LOG", "1") != "0",
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
			if kafka != nil {
				out["redpanda"] = "ok"
				if err := kafka.Ping(ctx); err != nil {
					out["redpanda"] = err.Error()
				}
			}
			return out
		},
		Log: log,
		Now: time.Now,

		Campaigns: st,    // the advertiser api (design §9.5)
		Ads:       adSrv, // sponsored slots (design §9.2)
	}

	handler := srv.Handler()
	if tracingOn {
		handler = otelhttp.NewHandler(handler, "api", otelhttp.WithSpanNameFormatter(spanName),
			otelhttp.WithFilter(func(r *http.Request) bool {
				switch r.URL.Path {
				case "/metrics", "/healthz", "/readyz": // scrapes and probes, not traffic
					return false
				}
				return true
			}))
	}
	addr := env("API_ADDR", "127.0.0.1:18080")
	httpSrv := &http.Server{Addr: addr, Handler: handler, ReadHeaderTimeout: 5 * time.Second, IdleTimeout: 2 * time.Minute}
	errc := make(chan error, 1)
	go func() { errc <- httpSrv.ListenAndServe() }()
	log.Info("api listening", "addr", addr, "bundle", b.Version, "config_version", exps.Current().Version,
		"events", env("EVENTS_MODE", "redpanda"), "naive", naive, "candidate_cache", cands != nil, "tracing", tracingOn,
		"budget_ms", srv.Plan.Total.Milliseconds(), "breaker_window", breakerCfg.String(), "ads", adSrv != nil)

	select {
	case err := <-errc:
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
