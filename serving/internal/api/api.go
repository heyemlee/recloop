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
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/heyemlee/recloop/serving/internal/bundle"
	"github.com/heyemlee/recloop/serving/internal/cache"
	"github.com/heyemlee/recloop/serving/internal/catalog"
	"github.com/heyemlee/recloop/serving/internal/events"
	"github.com/heyemlee/recloop/serving/internal/experiment"
	"github.com/heyemlee/recloop/serving/internal/mixer"
	"github.com/heyemlee/recloop/serving/internal/userstate"
)

// Server holds the api's dependencies.
type Server struct {
	Catalog *catalog.Catalog
	Bundle  *bundle.Bundle
	Users   userstate.Store
	Pages   cache.Pages
	Exps    *experiment.Loader
	Events  *events.Service
	Served  *events.ServedWriter
	Redis   redis.UniversalClient
	// Sources per surface, and the default RRF weights per surface (an experiment variant's
	// "rrf_weights" param overrides them).
	Sources map[string][]mixer.Spec
	Weights map[string]map[string]float64
	Ready   func(ctx context.Context) map[string]string // dependency -> "ok" or an error
	Log     *slog.Logger
	Now     func() time.Time
}

// Handler routes the api.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/recommendations", s.recommendations)
	mux.HandleFunc("POST /v1/events", s.postEvents)
	mux.HandleFunc("GET /v1/items/{id}", s.getItem)
	mux.HandleFunc("GET /v1/users/{id}/context", s.userContext)
	mux.HandleFunc("GET /v1/demo/users", s.demoUsers)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("GET /readyz", s.readyz)
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
			if r.URL.Path != "/healthz" {
				s.Log.Info("http", "method", r.Method, "path", r.URL.Path, "status", rec.status,
					"ms", float64(time.Since(t0).Microseconds())/1000)
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
	writeJSON(w, status, map[string]any{"bundle": s.Bundle.Version, "dependencies": deps})
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
	uc, err := s.Users.Read(r.Context(), uid, s.Now())
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
	raw, err := s.Redis.Get(r.Context(), userstate.DemoUsersKey).Bytes()
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
