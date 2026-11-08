// Package api is the HTTP surface of the online service (design §13.1).
package api

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"runtime/debug"
	"strconv"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/heyemlee/recloop/serving/internal/bundle"
	"github.com/heyemlee/recloop/serving/internal/cache"
	"github.com/heyemlee/recloop/serving/internal/catalog"
	"github.com/heyemlee/recloop/serving/internal/events"
	"github.com/heyemlee/recloop/serving/internal/experiment"
	"github.com/heyemlee/recloop/serving/internal/metrics"
	"github.com/heyemlee/recloop/serving/internal/mixer"
	"github.com/heyemlee/recloop/serving/internal/ranking"
	"github.com/heyemlee/recloop/serving/internal/resilience"
	"github.com/heyemlee/recloop/serving/internal/userstate"
)

// Server holds the api's dependencies.
type Server struct {
	Catalog *catalog.Catalog
	// Bundle is the bundle the api started with. Bundles, if set, holds every version it
	// serves (active, canary, the previous one for a rollback; design §6) and Bundle is then
	// only its first active version.
	Bundle  *bundle.Bundle
	Bundles *bundle.Registry
	// Rank calls the model server's Rank for bundles with ranking models (M4); nil: RRF only
	// and FixedPCTR. DefaultRanker orders the list when the rank variant does not say
	// ("rrf" when empty).
	Rank          *ranking.Client
	DefaultRanker string
	// Release runs the model release steps of the admin api (nil: no admin endpoints).
	Release Releaser
	Users   userstate.Store
	Pages   cache.Pages
	Cands   *cache.CandidateCache // nil: candidate cache off
	Exps    *experiment.Loader
	Events  events.Publisher
	Served  *events.ServedWriter
	// Redis is the state instance (user state, spend and frequency counters, demo users);
	// Pages and Cands hold the cache instance (design §19, 2026-11-08: two Redis).
	Redis redis.UniversalClient
	// Campaigns is the advertiser api's store; nil leaves its routes out.
	Campaigns CampaignStore
	// Ads serves the sponsored slots of the home feed and /internal/ads/select; nil: no ads.
	Ads *AdServer
	// Sources per surface, and the default RRF weights per surface (an experiment variant's
	// "rrf_weights" param overrides them).
	Sources map[string][]mixer.Spec
	Weights map[string]map[string]float64
	// Plan is the request budget. RedisBreaker, if set, guards the calls to both Redis
	// instances: user state, pages and the candidate cache (cmd/api leaves it nil); the model
	// server's and PostgreSQL's breakers sit in their Specs.
	Plan         resilience.Plan
	RedisBreaker *resilience.Breaker
	// Naive turns off the request deadline, the stage budgets, the breakers and degradation:
	// the baseline of the fault experiments (design §5.2). Never for real traffic.
	Naive     bool
	AccessLog bool
	Ready     func(ctx context.Context) map[string]string // dependency -> "ok" or an error
	Log       *slog.Logger
	Now       func() time.Time

	warned sync.Map // warning key -> unix second of its last log line
}

// warn logs at most one line per second per key: under a dependency failure every request
// fails the same way, and the metrics already count each one.
func (s *Server) warn(key, msg string, args ...any) {
	now := time.Now().Unix()
	if last, ok := s.warned.Load(key); ok && last.(int64) == now {
		return
	}
	s.warned.Store(key, now)
	s.Log.Warn(msg, args...)
}

// Handler routes the api.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/recommendations", s.recommendations)
	mux.HandleFunc("POST /v1/events", s.postEvents)
	mux.HandleFunc("GET /v1/items/{id}", s.getItem)
	mux.HandleFunc("GET /v1/users/{id}/context", s.userContext)
	mux.HandleFunc("GET /v1/demo/users", s.demoUsers)
	if s.Campaigns != nil {
		s.campaignRoutes(mux)
	}
	// Internal: for the shopping assistant's service, not the browser (the web app proxies
	// /api/v1/* only).
	mux.HandleFunc("POST /internal/ads/select", s.selectChatAds)
	s.adminRoutes(mux)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("GET /readyz", s.readyz)
	mux.Handle("GET /metrics", metrics.Handler())
	return s.middleware(mux)
}

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

func (s *Server) middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t0 := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		defer func() {
			if p := recover(); p != nil {
				s.Log.Error("panic", "path", r.URL.Path, "panic", p, "stack", string(debug.Stack()))
				writeError(rec, http.StatusInternalServerError, "internal error")
			}
			took := time.Since(t0)
			route := r.Pattern // set by the mux on the request it routed
			if route == "" {
				route = "unmatched"
			}
			if route != "GET /metrics" && route != "GET /healthz" {
				metrics.HTTPRequests.WithLabelValues(route, strconv.Itoa(rec.status)).Inc()
				metrics.HTTPDuration.WithLabelValues(route).Observe(took.Seconds())
				if s.AccessLog {
					s.Log.Info("http", "method", r.Method, "path", r.URL.Path, "status", rec.status,
						"ms", float64(took.Microseconds())/1000)
				}
			}
		}()
		next.ServeHTTP(rec, r)
	})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func (s *Server) readyz(w http.ResponseWriter, r *http.Request) {
	deps := map[string]string{}
	if s.Ready != nil {
		deps = s.Ready(r.Context())
	}
	status := http.StatusOK
	if deps["postgres"] != "ok" {
		status = http.StatusServiceUnavailable
	}
	mode := "resilient"
	if s.Naive {
		mode = "naive"
	}
	writeJSON(w, status, map[string]any{
		"bundle": s.active().Version, "dependencies": deps, "mode": mode, "candidate_cache": s.Cands != nil,
	})
}

// ---------------------------------------------------------------- items and users

// ItemCard is what a recommendation tile needs.
type ItemCard struct {
	Title      string `json:"title"`
	Brand      string `json:"brand,omitempty"`
	PriceCents *int32 `json:"price_cents"`
	ImageURL   string `json:"image_url,omitempty"`
	Stock      int32  `json:"stock"`
}

func (s *Server) card(id int32) *ItemCard {
	it, ok := s.Catalog.Get(id)
	if !ok {
		return nil
	}
	return &ItemCard{Title: it.Title, Brand: it.Brand, PriceCents: it.PriceCents, ImageURL: it.ImageURL, Stock: s.Catalog.Stock(id)}
}

func (s *Server) getItem(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 32)
	if err != nil {
		writeError(w, http.StatusBadRequest, "item id must be an integer")
		return
	}
	it, ok := s.Catalog.Get(int32(id))
	if !ok {
		writeError(w, http.StatusNotFound, "no such item")
		return
	}
	writeJSON(w, http.StatusOK, struct {
		catalog.Item
		Stock int32 `json:"stock"`
	}{it, s.Catalog.Stock(it.ID)})
}

type historyEntry struct {
	ItemID int32  `json:"item_id"`
	Title  string `json:"title"`
}

func (s *Server) userContext(w http.ResponseWriter, r *http.Request) {
	uid, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil || uid < 0 {
		writeError(w, http.StatusBadRequest, "user id must be a non-negative integer")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 200*time.Millisecond)
	defer cancel()
	uc, err := s.Users.Read(ctx, uid, s.Now())
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, "user state unavailable: "+err.Error())
		return
	}
	hist := make([]historyEntry, 0, len(uc.History))
	for i := len(uc.History) - 1; i >= 0; i-- { // most recent first
		id := uc.History[i]
		e := historyEntry{ItemID: id}
		if it, ok := s.Catalog.Get(id); ok {
			e.Title = it.Title
		}
		hist = append(hist, e)
	}
	cfg := s.Exps.Current()
	writeJSON(w, http.StatusOK, map[string]any{
		"user_id":     uid,
		"hv":          uc.HV,
		"history":     hist,
		"bought":      len(uc.Bought),
		"seen_24h":    len(uc.Seen),
		"experiments": cfg.Assign(experiment.UserUnit(uid)),
	})
}

func (s *Server) demoUsers(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 200*time.Millisecond)
	defer cancel()
	raw, err := s.Redis.Get(ctx, userstate.DemoUsersKey).Bytes()
	if errors.Is(err, redis.Nil) {
		writeJSON(w, http.StatusOK, []any{})
		return
	}
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(raw)
}
