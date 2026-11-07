package api

import (
	"context"
	"encoding/json"
	"errors"
	"maps"
	"net/http"
	"strconv"
	"time"

	"github.com/google/uuid"

	"github.com/heyemlee/recloop/serving/internal/cache"
	"github.com/heyemlee/recloop/serving/internal/events"
	"github.com/heyemlee/recloop/serving/internal/experiment"
	"github.com/heyemlee/recloop/serving/internal/mixer"
	"github.com/heyemlee/recloop/serving/internal/recall"
	"github.com/heyemlee/recloop/serving/internal/userstate"
)

const (
	// PageDepth is how many ranked items one request keeps for paging.
	PageDepth = 200
	// Diversity: at most MaxPerBrand items of one brand in the first DiversityWindow positions.
	DiversityWindow = 10
	MaxPerBrand     = 2
	DefaultLimit    = 20
	MaxLimit        = 50
)

// RecItem is one recommended item.
type RecItem struct {
	Position     int         `json:"position"`
	ItemID       int32       `json:"item_id"`
	Slot         string      `json:"slot"`
	Source       string      `json:"source"`
	Score        float64     `json:"score"`
	ImpressionID string      `json:"impression_id"`
	Sources      []mixer.Hit `json:"sources"`
	Item         *ItemCard   `json:"item,omitempty"`
}

// RecResponse of GET /v1/recommendations.
type RecResponse struct {
	RequestID     string                           `json:"request_id"`
	ModelVersion  string                           `json:"model_version"`
	Surface       string                           `json:"surface"`
	Degraded      []string                         `json:"degraded"`
	Experiments   map[string]experiment.Assignment `json:"experiments"`
	ConfigVersion int                              `json:"config_version"`
	Items         []RecItem                        `json:"items"`
	NextCursor    string                           `json:"next_cursor,omitempty"`
	Debug         *Debug                           `json:"debug,omitempty"`
}

// Debug is the request-level view of the debug panel.
type Debug struct {
	UserID     int64              `json:"user_id"`
	HV         int64              `json:"hv"`
	History    []historyEntry     `json:"history"` // most recent first, the window the sources used
	Sources    []SourceDebug      `json:"sources"`
	Weights    map[string]float64 `json:"rrf_weights"`
	Candidates int                `json:"candidates"`
	Filters    mixer.FilterStats  `json:"filters"`
	Promoted   int                `json:"diversity_promotions"`
	Backfill   int                `json:"backfill"`
	Offset     int                `json:"offset"`
	TookMs     float64            `json:"took_ms"`
}

// SourceDebug describes one recall source's answer.
type SourceDebug struct {
	Source     string  `json:"source"`
	Weight     float64 `json:"weight"`
	Candidates int     `json:"candidates"`
	TookMs     float64 `json:"took_ms"`
	Version    string  `json:"version,omitempty"`
	Error      string  `json:"error,omitempty"`
}

// servedItem / servedPayload are the served event (design §7.1).
type servedItem struct {
	ItemID int32          `json:"item_id"`
	Pos    int            `json:"pos"`
	Slot   string         `json:"slot"`
	Source string         `json:"source"`
	Score  float64        `json:"score"`
	Ranks  map[string]int `json:"ranks"`
}

type servedPayload struct {
	Surface        string                           `json:"surface"`
	Seed           *int32                           `json:"seed_item,omitempty"`
	Offset         int                              `json:"offset"`
	ModelVer       string                           `json:"model_ver"`
	SourceVersions map[string]string                `json:"source_versions,omitempty"`
	Exp            map[string]experiment.Assignment `json:"exp"`
	ConfigVersion  int                              `json:"config_version"`
	Degraded       []string                         `json:"degraded"`
	BackfillCount  int                              `json:"backfill_count"`
	Filters        *mixer.FilterStats               `json:"filters,omitempty"`
	HV             int64                            `json:"hv"`
	Items          []servedItem                     `json:"items"`
}

func (s *Server) recommendations(w http.ResponseWriter, r *http.Request) {
	t0 := time.Now()
	q := r.URL.Query()
	uid, err := strconv.ParseInt(q.Get("user_id"), 10, 64)
	if err != nil || uid < 0 {
		writeError(w, http.StatusBadRequest, "user_id is required and must be a non-negative integer")
		return
	}
	limit := DefaultLimit
	if v := q.Get("limit"); v != "" {
		if limit, err = strconv.Atoi(v); err != nil || limit < 1 || limit > MaxLimit {
			writeError(w, http.StatusBadRequest, "limit must be between 1 and 50")
			return
		}
	}
	debugOn := q.Get("debug") == "1" || q.Get("debug") == "true"
	if c := q.Get("cursor"); c != "" {
		s.nextPage(w, r, uid, c, limit, debugOn, t0)
		return
	}

	surface := q.Get("surface")
	if surface == "" {
		surface = recall.SurfaceHome
	}
	specs, ok := s.Sources[surface]
	if !ok {
		writeError(w, http.StatusBadRequest, "surface must be home or item")
		return
	}
	seed := int32(-1)
	if surface == recall.SurfaceItem {
		id, err := strconv.ParseInt(q.Get("item_id"), 10, 32)
		if err != nil || !s.Catalog.Has(int32(id)) {
			writeError(w, http.StatusBadRequest, "surface=item needs a catalog item_id")
			return
		}
		seed = int32(id)
	}

	ctx := r.Context()
	now := s.Now()
	requestID, err := uuid.NewV7()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	cfg := s.Exps.Current()
	configVersion := 0
	if cfg != nil {
		configVersion = cfg.Version
	}
	assign := cfg.Assign(experiment.UserUnit(uid))
	weights := s.weights(surface, assign["recall"])
	degraded := []string{}

	uc, err := s.Users.Read(ctx, uid, now)
	if err != nil {
		// Without Redis: no history (popularity only) and no seen / bought filters.
		s.Log.Warn("user state unavailable", "user_id", uid, "err", err)
		degraded = append(degraded, "user_state_unavailable")
		uc = userstate.Context{}
	}
	req := recall.Request{UserID: uid, Surface: surface, History: uc.History, Seed: seed}
	results := mixer.FanOut(ctx, specs, req)
	versions := map[string]string{}
	for _, res := range results {
		if res.Err != nil {
			reason := "error"
			if errors.Is(res.Err, mixer.ErrTimeout) {
				reason = "timeout"
			}
			degraded = append(degraded, res.Source+"_"+reason)
			s.Log.Warn("recall source failed", "source", res.Source, "err", res.Err)
		} else if res.Version != "" {
			versions[res.Source] = res.Version
		}
	}
	if v := versions["sasrec"]; v != "" && v != s.Bundle.Version {
		// The model server serves another bundle than the api's ItemCF / Pop tables.
		degraded = append(degraded, "model_version_mismatch")
	}

	fused := mixer.RRF(results, weights)
	ex := mixer.Exclusions{Bought: uc.Bought, Seen: uc.Seen, Seed: seed, Known: s.Catalog.Has, InStock: s.Catalog.InStock}
	filtered, fstats := mixer.Filter(fused, ex)
	diverse, promoted := mixer.Diversify(filtered, s.Catalog.Brand, DiversityWindow, MaxPerBrand)
	list, backfill := mixer.Backfill(diverse, s.Bundle.Pop, PageDepth, ex)
	if len(list) > PageDepth {
		list = list[:PageDepth]
	}

	page := &cache.Page{
		RequestID: requestID, UserID: uid, Surface: surface, Seed: seed, ModelVer: s.Bundle.Version,
		Exp: assign, ConfigVersion: configVersion, Degraded: degraded, HV: uc.HV, Items: list, CreatedAt: now,
	}
	if err := s.Pages.Put(ctx, page); err != nil {
		s.Log.Warn("storing the page failed; no next page", "err", err)
		page = nil
	}

	resp, served := s.render(requestID, surface, seed, assign, configVersion, degraded, list, 0, limit, page != nil, uc.HV)
	served.SourceVersions = versions
	served.BackfillCount = backfill
	served.Filters = &fstats
	s.logServed(uid, requestID, served)

	if debugOn {
		window := recall.Recent(uc.History, recall.HistoryWindow)
		hist := make([]historyEntry, 0, len(window))
		for i := len(window) - 1; i >= 0; i-- {
			e := historyEntry{ItemID: window[i]}
			if it, ok := s.Catalog.Get(window[i]); ok {
				e.Title = it.Title
			}
			hist = append(hist, e)
		}
		srcs := make([]SourceDebug, len(results))
		for i, res := range results {
			srcs[i] = SourceDebug{
				Source: res.Source, Weight: weights[res.Source], Candidates: len(res.Items),
				TookMs: float64(res.Took.Microseconds()) / 1000, Version: res.Version,
			}
			if res.Err != nil {
				srcs[i].Error = res.Err.Error()
			}
		}
		resp.Debug = &Debug{
			UserID: uid, HV: uc.HV, History: hist, Sources: srcs, Weights: weights,
			Candidates: len(fused), Filters: fstats, Promoted: promoted, Backfill: backfill,
			TookMs: float64(time.Since(t0).Microseconds()) / 1000,
		}
	}
	writeJSON(w, http.StatusOK, resp)
}

// weights merges the surface defaults with the recall-layer variant's "rrf_weights".
func (s *Server) weights(surface string, a experiment.Assignment) map[string]float64 {
	out := maps.Clone(s.Weights[surface])
	if len(a.Params) == 0 {
		return out
	}
	var p struct {
		RRFWeights map[string]float64 `json:"rrf_weights"`
	}
	if err := json.Unmarshal(a.Params, &p); err != nil {
		s.Log.Warn("bad experiment params", "exp_id", a.ExpID, "err", err)
		return out
	}
	for src := range out {
		if w, ok := p.RRFWeights[src]; ok {
			out[src] = w
		}
	}
	return out
}

// render builds the response for items [offset, offset+limit) of the list, and the matching
// served event.
func (s *Server) render(
	requestID uuid.UUID, surface string, seed int32, assign map[string]experiment.Assignment,
	configVersion int, degraded []string, list []mixer.Fused, offset, limit int, pageable bool, hv int64,
) (RecResponse, servedPayload) {
	end := min(offset+limit, len(list))
	resp := RecResponse{
		RequestID: requestID.String(), ModelVersion: s.Bundle.Version, Surface: surface,
		Degraded: degraded, Experiments: assign, ConfigVersion: configVersion, Items: []RecItem{},
	}
	served := servedPayload{
		Surface: surface, Offset: offset, ModelVer: s.Bundle.Version, Exp: assign,
		ConfigVersion: configVersion, Degraded: degraded, HV: hv, Items: []servedItem{},
	}
	if seed >= 0 {
		served.Seed = &seed
	}
	for i := offset; i < end; i++ {
		f := list[i]
		pos := i + 1
		resp.Items = append(resp.Items, RecItem{
			Position: pos, ItemID: f.ItemID, Slot: "organic", Source: f.Primary, Score: f.Score,
			ImpressionID: events.ImpressionID(requestID, pos), Sources: f.Hits, Item: s.card(f.ItemID),
		})
		ranks := make(map[string]int, len(f.Hits))
		for _, h := range f.Hits {
			ranks[h.Source] = h.Rank
		}
		served.Items = append(served.Items, servedItem{
			ItemID: f.ItemID, Pos: pos, Slot: "organic", Source: f.Primary, Score: f.Score, Ranks: ranks,
		})
	}
	if pageable && end < len(list) {
		resp.NextCursor = cache.Cursor{RequestID: requestID, Offset: end}.Encode()
	}
	return resp, served
}

func (s *Server) logServed(uid int64, requestID uuid.UUID, p servedPayload) {
	raw, err := json.Marshal(p)
	if err != nil {
		s.Log.Error("served payload", "err", err)
		return
	}
	ev, err := events.Served(uid, requestID, raw)
	if err != nil {
		s.Log.Error("served event", "err", err)
		return
	}
	if !s.Served.Enqueue(ev) {
		s.Log.Warn("served event dropped: buffer full", "request_id", requestID)
	}
}

func (s *Server) nextPage(w http.ResponseWriter, r *http.Request, uid int64, cursor string, limit int, debugOn bool, t0 time.Time) {
	cur, err := cache.DecodeCursor(cursor)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 50*time.Millisecond)
	defer cancel()
	pg, err := s.Pages.Get(ctx, cur.RequestID)
	if errors.Is(err, cache.ErrNoPage) {
		writeError(w, http.StatusGone, "the result list expired; request the first page again")
		return
	}
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, "pages unavailable: "+err.Error())
		return
	}
	if pg.UserID != uid {
		writeError(w, http.StatusBadRequest, "cursor belongs to another user")
		return
	}
	if cur.Offset >= len(pg.Items) {
		writeError(w, http.StatusBadRequest, "cursor is past the end of the list")
		return
	}
	resp, served := s.render(pg.RequestID, pg.Surface, pg.Seed, pg.Exp, pg.ConfigVersion, pg.Degraded, pg.Items, cur.Offset, limit, true, pg.HV)
	resp.ModelVersion, served.ModelVer = pg.ModelVer, pg.ModelVer
	s.logServed(uid, pg.RequestID, served)
	if debugOn {
		resp.Debug = &Debug{UserID: uid, Offset: cur.Offset, TookMs: float64(time.Since(t0).Microseconds()) / 1000}
	}
	writeJSON(w, http.StatusOK, resp)
}
