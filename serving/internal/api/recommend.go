package api

import (
	"context"
	"encoding/json"
	"errors"
	"maps"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/heyemlee/recloop/serving/internal/bundle"
	"github.com/heyemlee/recloop/serving/internal/cache"
	"github.com/heyemlee/recloop/serving/internal/events"
	"github.com/heyemlee/recloop/serving/internal/experiment"
	"github.com/heyemlee/recloop/serving/internal/itemctr"
	"github.com/heyemlee/recloop/serving/internal/metrics"
	"github.com/heyemlee/recloop/serving/internal/mixer"
	"github.com/heyemlee/recloop/serving/internal/ranking"
	"github.com/heyemlee/recloop/serving/internal/recall"
	"github.com/heyemlee/recloop/serving/internal/resilience"
	"github.com/heyemlee/recloop/serving/internal/tracing"
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
	// MaxChatSeeds bounds surface=chat's seed_items (the sources look at that many items).
	MaxChatSeeds = recall.HistoryWindow
	// renderReserve is kept back from the page-store stage for encoding the response.
	renderReserve = 2 * time.Millisecond
)

// Candidate cache results (X-Candidate-Cache header, metrics, served log).
const (
	CacheHit   = "hit"
	CacheMiss  = "miss"
	CacheError = "error" // Redis failed or timed out: computed fresh
	CacheSkip  = "skip"  // not looked up: no user state, breaker open, or no budget left
	CacheOff   = "off"
)

// RecItem is one recommended item. A sponsored slot (design §9.2, §13.1) has slot "sponsored",
// label "Sponsored" and the ad_token its impression and click events must carry.
type RecItem struct {
	Position     int         `json:"position"`
	ItemID       int32       `json:"item_id"`
	Slot         string      `json:"slot"`
	Label        string      `json:"label,omitempty"`
	Source       string      `json:"source"`
	Score        float64     `json:"score"`
	ImpressionID string      `json:"impression_id"`
	CampaignID   int64       `json:"campaign_id,omitempty"`
	AdToken      string      `json:"ad_token,omitempty"`
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
	Cache      string             `json:"candidate_cache,omitempty"`
	Stages     map[string]float64 `json:"stages_ms,omitempty"`
	Mode       string             `json:"mode,omitempty"`
	Ranker     string             `json:"ranker,omitempty"` // what ordered the organic list (M4)
	Ads        *adDebug           `json:"ads,omitempty"`
	Gen        *genServed         `json:"gen,omitempty"` // M7: the nearline generation result
	TookMs     float64            `json:"took_ms"`
}

// adDebug is the ads stage of the debug panel: what it decided, and each slot's price.
type adDebug struct {
	adResult
	Slots []cache.Slot `json:"slots"`
}

// SourceDebug describes one recall source's answer.
type SourceDebug struct {
	Source     string  `json:"source"`
	Weight     float64 `json:"weight"`
	Candidates int     `json:"candidates"`
	Status     string  `json:"status"`
	TookMs     float64 `json:"took_ms"`
	Version    string  `json:"version,omitempty"`
	Error      string  `json:"error,omitempty"`
}

// servedItem / servedPayload are the served event (design §7.1). Sponsored items also carry
// their campaign, the auction price (what a click is billed at most), the pCTR they were
// ranked with (the feature at serving time) and the ad_token.
type servedItem struct {
	ItemID     int32          `json:"item_id"`
	Pos        int            `json:"pos"`
	Slot       string         `json:"slot"`
	Source     string         `json:"source"`
	Score      float64        `json:"score"`
	Ranks      map[string]int `json:"ranks"`
	CampaignID int64          `json:"campaign_id,omitempty"`
	Price      int32          `json:"price,omitempty"`
	PCTRPPM    int64          `json:"pctr_ppm,omitempty"`
	AdToken    string         `json:"ad_token,omitempty"`
	// M4: the ranker's features of this item at serving time (payload "features" names them;
	// null = missing) and its LightGBM score, logged for both rank variants.
	F    ranking.Features `json:"f,omitempty"`
	LGBM *float64         `json:"lgbm,omitempty"`
	// The item's impressions and clicks of the last hour and day at serving time
	// (itemctr.Counts.Array), logged with F: the pCTR's CTR inputs (design §7.5, §9.2 step 3).
	ICTR *[4]uint32 `json:"ictr,omitempty"`
}

type servedPayload struct {
	Surface        string                           `json:"surface"`
	Seed           *int32                           `json:"seed_item,omitempty"`
	SeedItems      []int32                          `json:"seed_items,omitempty"` // surface=chat
	Offset         int                              `json:"offset"`
	ModelVer       string                           `json:"model_ver"`
	SourceVersions map[string]string                `json:"source_versions,omitempty"`
	Exp            map[string]experiment.Assignment `json:"exp"`
	ConfigVersion  int                              `json:"config_version"`
	Degraded       []string                         `json:"degraded"`
	BackfillCount  int                              `json:"backfill_count"`
	Filters        *mixer.FilterStats               `json:"filters,omitempty"`
	Cache          string                           `json:"candidate_cache,omitempty"`
	HV             int64                            `json:"hv"`
	Items          []servedItem                     `json:"items"`
	Ads            *adResult                        `json:"ads,omitempty"` // the ads stage (first page only)
	// M4: what ordered the organic list ("rrf" or "lightgbm"), the names of the items' feature
	// columns, and the pCTR model of the sponsored slots ("fixed", "logreg").
	Ranker    string   `json:"ranker,omitempty"`
	Features  []string `json:"features,omitempty"`
	PCTRModel string   `json:"pctr_model,omitempty"`
	// M7: the nearline generation result the request saw (arms with the gen source only).
	Gen *genServed `json:"gen,omitempty"`
	// The item CTR table the items' ictr come from: its as-of time and the seed it starts
	// after (Unix ms; package itemctr), so the sample job can recompute the same windows.
	ICTRAsOf  int64 `json:"ictr_asof,omitempty"`
	ICTRSince int64 `json:"ictr_since,omitempty"`
}

// genServed is what a home request of an arm with the gen source saw of the user's nearline
// generation result (design §11 step 5): fresh (generated from the current history version),
// stale (from an older one; Lag versions behind), missing, invalid (did not decode) or
// unavailable (no user state). AgeMs is the result's age when the request was served.
type genServed struct {
	Version string `json:"ver"`
	Status  string `json:"status"`
	HV      int64  `json:"hv,omitempty"`
	Lag     int64  `json:"lag,omitempty"`
	AgeMs   int64  `json:"age_ms,omitempty"`
	Items   int    `json:"items,omitempty"`
}

// Gen result statuses (served log, metrics).
const (
	GenFresh       = "fresh"
	GenStale       = "stale"
	GenMissing     = "missing"
	GenInvalid     = "invalid"
	GenUnavailable = "unavailable"
)

// genVersion is the generator version a request reads (design §11 step 5): the recall variant's
// "gen_version", on the home feed, when the variant gives the gen source a weight. Empty: the
// request does not read a generation result (every other arm, and the default config).
func (s *Server) genVersion(surface string, a experiment.Assignment, weights map[string]float64) string {
	if surface != recall.SurfaceHome || weights["gen"] <= 0 || len(a.Params) == 0 {
		return ""
	}
	var p struct {
		GenVersion string `json:"gen_version"`
	}
	if err := json.Unmarshal(a.Params, &p); err != nil {
		return ""
	}
	return p.GenVersion
}

// genStatus describes the result a request read (genVersion non-empty).
func genStatus(ver string, uc userstate.Context, userOK bool, now time.Time) *genServed {
	g := &genServed{Version: ver}
	switch {
	case !userOK:
		g.Status = GenUnavailable
	case uc.GenInvalid:
		g.Status = GenInvalid
	case uc.Gen == nil:
		g.Status = GenMissing
	default:
		g.HV, g.Items = uc.Gen.HV, len(uc.Gen.Items)
		g.AgeMs = now.Sub(uc.Gen.AtTime()).Milliseconds()
		if lag := uc.HV - uc.Gen.HV; lag > 0 {
			g.Status, g.Lag = GenStale, lag
		} else {
			g.Status = GenFresh
		}
	}
	return g
}

// genStamp is the part of the candidate cache key that tells generation results apart: a new
// result (another history version) changes the key, so the next request recomputes and sees it
// instead of a cached list computed without it (CandidateTTL).
func genStamp(uc userstate.Context) string {
	if uc.Gen == nil {
		return ".g-"
	}
	return ".g" + strconv.FormatInt(uc.Gen.HV, 10)
}

// rankParams are the rank layer's experiment parameters: which order the organic list takes,
// and (the canary of a release, design §6) which bundle version serves the request.
type rankParams struct {
	Ranker     string `json:"ranker"`
	PinVersion string `json:"pin_version"`
}

func (s *Server) rankParams(a experiment.Assignment) rankParams {
	p := rankParams{Ranker: s.DefaultRanker}
	if p.Ranker == "" {
		p.Ranker = ranking.RRF
	}
	if len(a.Params) == 0 {
		return p
	}
	var v rankParams
	if err := json.Unmarshal(a.Params, &v); err != nil {
		s.warn("bad_rank_params", "bad rank experiment params", "exp_id", a.ExpID, "err", err)
		return p
	}
	if v.Ranker == ranking.RRF || v.Ranker == ranking.LightGBM {
		p.Ranker = v.Ranker
	}
	p.PinVersion = v.PinVersion
	return p
}

// bundleFor is the version serving a request: the rank variant's pinned version if it is
// loaded, else the active one. Read once per request, so a release switch never mixes versions
// inside one response.
func (s *Server) bundleFor(p rankParams) *bundle.Bundle {
	if s.Bundles == nil {
		return s.Bundle
	}
	if p.PinVersion != "" {
		if b := s.Bundles.Get(p.PinVersion); b != nil {
			return b
		}
		s.warn("pin_missing", "pinned bundle not loaded; serving the active one", "version", p.PinVersion)
	}
	return s.Bundles.Active()
}

// rendered is what the first page's served event adds about the ranking stage.
type rendered struct {
	features map[int32][]float32
	lgbm     map[int32]float32
	names    []string
	ranker   string
}

// request is the state of one recommendation request as it moves through the stages.
type request struct {
	s        *Server
	ctx      context.Context // carries the request deadline (resilient mode)
	budget   resilience.Budget
	surface  string
	seed     int32
	seeds    []int32 // surface=chat: the assistant's seed items, most relevant first
	uid      int64
	now      time.Time
	degraded []string
	stages   map[string]float64
	// M4: the bundle version serving the request, the recall window, and SASRec's relevance of
	// the eligible ad items (from Retrieve; empty when the candidates came from the cache).
	bundle    *bundle.Bundle
	history   []int32
	relevance map[int32]float32
	// The item CTR table of the request, read once (itemCTR) so the pCTR and the served event
	// use the same one.
	ictr      *itemctr.Table
	ictrTaken bool
}

// itemCTR is the request's item CTR table: nil without ads, before the first build, or when
// the current one is older than itemctr.MaxAge.
func (q *request) itemCTR() *itemctr.Table {
	if !q.ictrTaken {
		q.ictrTaken = true
		if a := q.s.Ads; a != nil && a.ItemCTR != nil {
			if t := a.ItemCTR.Current(); t.Fresh(q.now) {
				q.ictr = t
			}
		}
	}
	return q.ictr
}

// errFatal ends a naive-mode request with 503: without degradation a failed dependency is a
// failed request.
type errFatal struct {
	stage string
	err   error
}

func (e errFatal) Error() string { return e.stage + ": " + e.err.Error() }

func (q *request) stage(name string, t0 time.Time) {
	d := time.Since(t0)
	metrics.StageDuration.WithLabelValues(q.surface, name).Observe(d.Seconds())
	if q.stages != nil {
		q.stages[name] = float64(d.Microseconds()) / 1000
	}
}

func (q *request) span(name string) (context.Context, trace.Span) {
	return tracing.Tracer.Start(q.ctx, name)
}

func (q *request) degrade(reason string) { q.degraded = append(q.degraded, reason) }

// redis runs f against Redis under a stage budget d (and the Redis breaker, if one is set). It
// returns the status of the call (mixer.Status*) and its error.
func (q *request) redis(d time.Duration, f func(ctx context.Context) error) (string, error) {
	if q.s.Naive {
		if err := f(q.ctx); err != nil {
			return mixer.StatusError, err
		}
		return mixer.StatusOK, nil
	}
	if d <= 0 {
		return mixer.StatusSkipped, mixer.ErrTimeout
	}
	sctx, cancel := context.WithTimeout(q.ctx, d)
	defer cancel()
	call := func() error { return f(sctx) }
	var err error
	if q.s.RedisBreaker != nil {
		err = q.s.RedisBreaker.Do(q.ctx, call)
	} else {
		err = call()
	}
	var ne net.Error
	switch {
	case err == nil:
		return mixer.StatusOK, nil
	case errors.Is(err, resilience.ErrOpen):
		return mixer.StatusOpen, err
	case sctx.Err() != nil, errors.As(err, &ne) && ne.Timeout():
		// go-redis turns the context deadline into a socket deadline: an "i/o timeout" can
		// come back an instant before the context itself reports it.
		return mixer.StatusTimeout, err
	default:
		return mixer.StatusError, err
	}
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
	surface := q.Get("surface")
	if surface == "" {
		surface = recall.SurfaceHome
	}
	if c := q.Get("cursor"); c != "" {
		s.nextPage(w, r, uid, c, limit, debugOn, t0)
		return
	}
	specs, ok := s.Sources[surface]
	if !ok {
		writeError(w, http.StatusBadRequest, "surface must be home, item or chat")
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
	var seeds []int32
	if surface == recall.SurfaceChat {
		if seeds, err = parseSeeds(q.Get("seed_items"), s.Catalog.Has); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
	}

	req := &request{s: s, ctx: r.Context(), surface: surface, seed: seed, seeds: seeds, uid: uid, now: s.Now(), degraded: []string{}}
	if !s.Naive {
		var cancel context.CancelFunc
		req.ctx, cancel, req.budget = resilience.Start(r.Context(), s.Plan, time.Now)
		defer cancel()
	}
	if debugOn {
		req.stages = map[string]float64{}
	}
	resp, err := s.recommend(req, specs, limit, debugOn)
	if err != nil {
		var fatal errFatal
		if errors.As(err, &fatal) {
			s.warn("naive:"+fatal.stage, "request failed (naive mode)", "stage", fatal.stage, "err", fatal.err)
			writeError(w, http.StatusServiceUnavailable, err.Error())
			return
		}
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	req.stage("total", t0)
	if resp.Debug != nil {
		resp.Debug.TookMs = float64(time.Since(t0).Microseconds()) / 1000
	}
	w.Header().Set("X-Candidate-Cache", resp.cache)
	writeJSON(w, http.StatusOK, resp.RecResponse)
}

type recResult struct {
	RecResponse
	cache string
}

// recommend runs the stages of design §4.1 under the budget of §5.1. In resilient mode no
// dependency failure is fatal: each one is written into `degraded` and the request goes on
// with what is left, down to the in-process popular list. In naive mode the first failure is.
func (s *Server) recommend(q *request, specs []mixer.Spec, limit int, debugOn bool) (*recResult, error) {
	plan := s.Plan
	requestID, err := uuid.NewV7()
	if err != nil {
		return nil, err
	}
	cfg := s.Exps.Current()
	configVersion := 0
	if cfg != nil {
		configVersion = cfg.Version
	}
	assign := cfg.Assign(experiment.UserUnit(q.uid))
	weights := s.weights(q.surface, assign["recall"])
	genVer := s.genVersion(q.surface, assign["recall"], weights)
	rp := s.rankParams(assign["rank"])
	q.bundle = s.bundleFor(rp)
	b := q.bundle

	// ① user context: history, hv, bought, seen. Without it: no history (popularity), no
	// seen / bought filters, no ads (design §5.1).
	t := time.Now()
	_, span := q.span("user_context")
	var uc userstate.Context
	status, err := q.redis(q.budget.For(plan.UserContext, plan.CacheGet+plan.Recall+plan.Rank+plan.Ads+plan.Tail), func(c context.Context) error {
		var err error
		uc, err = s.Users.ReadWith(c, q.uid, q.now, userstate.ReadOptions{GenVersion: genVer})
		return err
	})
	span.End()
	q.stage("user_context", t)
	userOK := status == mixer.StatusOK
	if !userOK {
		if s.Naive {
			return nil, errFatal{"user_context", err}
		}
		q.degrade("user_state_" + status)
		s.warn("user_state_"+status, "user state unavailable", "err", err)
		uc = userstate.Context{}
	}

	// Candidate cache (design §4.5): recall + ranking output, keyed by the history version.
	// Chat requests skip it: their seed items are not part of the key.
	variant := assign["recall"].Variant + "." + assign["rank"].Variant
	var gen *genServed
	if genVer != "" {
		gen = genStatus(genVer, uc, userOK, q.now)
		metrics.GenResults.WithLabelValues(gen.Status).Inc()
		variant += genStamp(uc)
	}
	key := cache.CandidateKey(q.uid, q.surface, q.seed, variant, b.Version, uc.HV)
	cacheResult := CacheOff
	useCache := s.Cands != nil && q.surface != recall.SurfaceChat
	var cands *cache.Candidates
	if useCache {
		cacheResult = CacheSkip
		if userOK || q.surface == recall.SurfaceItem {
			t = time.Now()
			_, span = q.span("candidate_cache.get")
			status, err := q.redis(q.budget.For(plan.CacheGet, plan.Recall+plan.Rank+plan.Ads+plan.Tail), func(c context.Context) error {
				var err error
				cands, err = s.Cands.Get(c, key)
				if errors.Is(err, cache.ErrMiss) {
					return nil
				}
				return err
			})
			span.End()
			q.stage("cache_get", t)
			switch {
			case status == mixer.StatusOK && cands != nil:
				cacheResult = CacheHit
			case status == mixer.StatusOK:
				cacheResult = CacheMiss
			case s.Naive:
				return nil, errFatal{"candidate_cache", err}
			case status == mixer.StatusSkipped || status == mixer.StatusOpen:
				cacheResult = CacheSkip
			default:
				cacheResult = CacheError
			}
		}
		metrics.CandidateCache.WithLabelValues(q.surface, cacheResult).Inc()
	}

	var results []mixer.Result
	var rk rendered
	complete := true
	q.history = recall.Recent(chatHistory(uc.History, q.seeds), recall.HistoryWindow)
	ranks := q.surface == recall.SurfaceHome && b.Ranking != nil && s.Rank != nil
	if cands == nil {
		// ② recall, concurrently, each source under min(its timeout, the stage budget).
		t = time.Now()
		rctx, span := q.span("recall")
		rreq := recall.Request{UserID: q.uid, Surface: q.surface, History: chatHistory(uc.History, q.seeds), Seed: q.seed, Bundle: b}
		if genVer != "" {
			rreq.Gen, rreq.GenVersion = uc.Gen, genVer
		}
		if s.Bundles != nil {
			rreq.PinVersion = b.Version // the model server serves exactly this version
		}
		if ranks && userOK {
			rreq.AdItems = s.eligibleAdItems(q, uc) // scored by the same Retrieve call
		}
		if s.Naive {
			results = mixer.FanOutNaive(rctx, specs, rreq)
		} else {
			results = mixer.FanOut(rctx, specs, rreq, q.budget.For(plan.Recall, plan.Rank+plan.Ads+plan.Tail))
		}
		span.End()
		q.stage("recall", t)
		versions := map[string]string{}
		for _, res := range results {
			if !res.OK() {
				complete = false
				if s.Naive {
					return nil, errFatal{"recall_" + res.Source, res.Err}
				}
				q.degrade(res.Source + "_" + res.Status)
				s.warn(res.Source+"_"+res.Status, "recall source failed", "source", res.Source, "err", res.Err)
			} else if res.Version != "" {
				versions[res.Source] = res.Version
			}
		}
		for _, res := range results {
			if res.Source == "sasrec" && res.AdScores != nil {
				q.relevance = res.AdScores
			}
		}
		if v := versions["sasrec"]; v != "" && v != b.Version {
			// The model server serves another bundle than the api's ItemCF / Pop tables.
			q.degrade("model_version_mismatch")
			complete = false
		}
		// ⑤ ranking: weighted RRF in process (v1). With an M4 bundle the model server's Rank
		// adds every candidate's features and LightGBM score; the rank variant decides whether
		// LightGBM's order replaces RRF's. A failed Rank call leaves RRF's order (§5.1).
		t = time.Now()
		fused := mixer.RRF(results, weights)
		if len(fused) > cache.CandidateDepth {
			fused = fused[:cache.CandidateDepth]
		}
		rk.ranker = ranking.RRF
		if ranks && len(fused) > 0 {
			org, status, err := s.rankOrganic(q, fused)
			switch {
			case status != mixer.StatusOK && s.Naive:
				return nil, errFatal{"rank", err}
			case status != mixer.StatusOK:
				q.degrade("rank_" + status)
				s.warn("rank_"+status, "ranking failed: RRF order", "err", err)
				complete = false
			default:
				rk.names = org.Names
				rk.features = make(map[int32][]float32, len(fused))
				rk.lgbm = make(map[int32]float32, len(org.Scores))
				for i, f := range fused {
					rk.features[f.ItemID] = org.Features[i]
					if org.Scores != nil {
						rk.lgbm[f.ItemID] = org.Scores[i]
					}
				}
				if rp.Ranker == ranking.LightGBM && org.Scores != nil {
					fused = ranking.Reorder(fused, org.Scores)
					rk.ranker = ranking.LightGBM
				}
			}
		}
		cands = &cache.Candidates{Items: fused, Versions: versions}
		q.stage("rank", t)
	} else if rp.Ranker == ranking.LightGBM && ranks {
		rk.ranker = ranking.LightGBM // the cached list was ordered by this variant's ranker
	} else {
		rk.ranker = ranking.RRF
	}

	// ④ filters and re-ranking, every request (seen, bought and stock change quickly).
	t = time.Now()
	ex := mixer.Exclusions{Bought: uc.Bought, Seen: uc.Seen, Seed: q.seed, Known: s.Catalog.Has, InStock: s.Catalog.InStock}
	if q.surface == recall.SurfaceChat {
		// The assistant ranks items the shopper asked for: having seen one today is no reason
		// to leave it out (the feed filters them to stay fresh).
		ex.Seen = nil
	}
	filtered, fstats := mixer.Filter(cands.Items, ex)
	diverse, promoted := mixer.Diversify(filtered, s.Catalog.Brand, DiversityWindow, MaxPerBrand)
	list, backfill := mixer.Backfill(diverse, b.Pop, PageDepth, ex)
	if len(list) > PageDepth {
		list = list[:PageDepth]
	}
	if len(filtered) == 0 && !complete && !s.Naive {
		// Nothing came back in time: the in-process popular list, which needs no dependency.
		q.degrade("static_popular")
	}
	q.stage("filter", t)

	// ⑥ sponsored slots of the home feed (design §9.2): computed for every request, never
	// cached; a failure leaves the organic list.
	var ad *adResult
	if q.surface == recall.SurfaceHome && s.Ads != nil {
		t = time.Now()
		res, err := s.homeSlots(q, requestID, list, uc, assign["ads"])
		if err != nil {
			return nil, err
		}
		q.stage("ads", t)
		list = insertSponsored(list, res.Slots, PageDepth)
		ad = &res
	}

	// Page (for the cursor) and, after a fresh and complete recall, the candidate cache entry:
	// one round trip to the cache Redis (cmd/api builds Pages and Cands on the same client). The
	// assistant reads one list: chat results are not paged.
	page := &cache.Page{
		RequestID: requestID, UserID: q.uid, Surface: q.surface, Seed: q.seed, ModelVer: b.Version,
		Exp: assign, ConfigVersion: configVersion, Degraded: q.degraded, HV: uc.HV, Items: list, CreatedAt: q.now,
	}
	if ad != nil {
		page.Sponsored = ad.Slots
	}
	pageable := false
	if q.surface != recall.SurfaceChat {
		t = time.Now()
		_, span = q.span("page_store")
		storeCands := useCache && cacheResult != CacheHit && complete && (userOK || q.surface == recall.SurfaceItem)
		status, err = q.redis(q.budget.For(plan.Tail, renderReserve), func(c context.Context) error {
			pipe := s.Pages.R.Pipeline()
			if err := s.Pages.PutCmd(c, pipe, page); err != nil {
				return err
			}
			if storeCands {
				s.Cands.SetCmd(c, pipe, key, cands)
			}
			_, err := pipe.Exec(c)
			return err
		})
		span.End()
		q.stage("page_store", t)
		pageable = status == mixer.StatusOK
		if !pageable {
			if s.Naive {
				return nil, errFatal{"page_store", err}
			}
			q.degrade("pages_" + status)
			s.warn("pages_"+status, "storing the page failed; no next page", "err", err)
		}
	}

	// ⑦ response and served log (asynchronous, never blocks).
	t = time.Now()
	resp, served := s.render(requestID, q.surface, q.seed, assign, configVersion, q.degraded, list, page.Sponsored, 0, limit, pageable, uc.HV)
	resp.ModelVersion, served.ModelVer = b.Version, b.Version
	if q.surface == recall.SurfaceHome && b.Ranking != nil {
		served.Ranker = rk.ranker
	}
	addRanking(&served, rk, page.Sponsored, q.itemCTR())
	if ad != nil {
		served.PCTRModel = ad.pctrModel
	}
	served.SourceVersions = cands.Versions
	served.SeedItems = q.seeds
	served.BackfillCount = backfill
	served.Filters = &fstats
	served.Cache = cacheResult
	served.Ads = ad
	served.Gen = gen
	s.logServed(q.uid, requestID, served)
	q.stage("render", t)

	metrics.Recommendations.WithLabelValues(q.surface, strconv.FormatBool(len(q.degraded) > 0)).Inc()
	for _, d := range q.degraded {
		metrics.Degraded.WithLabelValues(q.surface, d).Inc()
	}
	trace.SpanFromContext(q.ctx).SetAttributes(
		attribute.String("recloop.surface", q.surface), attribute.String("recloop.candidate_cache", cacheResult),
		attribute.StringSlice("recloop.degraded", q.degraded),
	)

	if debugOn {
		window := recall.Recent(chatHistory(uc.History, q.seeds), recall.HistoryWindow)
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
				Source: res.Source, Weight: weights[res.Source], Candidates: len(res.Items), Status: res.Status,
				TookMs: float64(res.Took.Microseconds()) / 1000, Version: res.Version,
			}
			if res.Err != nil {
				srcs[i].Error = res.Err.Error()
			}
		}
		mode := "resilient"
		if s.Naive {
			mode = "naive"
		}
		resp.Debug = &Debug{
			UserID: q.uid, HV: uc.HV, History: hist, Sources: srcs, Weights: weights,
			Candidates: len(cands.Items), Filters: fstats, Promoted: promoted, Backfill: backfill,
			Cache: cacheResult, Stages: q.stages, Mode: mode, Ranker: served.Ranker, Gen: gen,
		}
		if ad != nil {
			resp.Debug.Ads = &adDebug{adResult: *ad, Slots: ad.Slots}
		}
	}
	return &recResult{RecResponse: resp, cache: cacheResult}, nil
}

// parseSeeds reads surface=chat's seed_items: comma-separated catalog item ids, most relevant
// first, at most MaxChatSeeds.
func parseSeeds(v string, known func(int32) bool) ([]int32, error) {
	if v == "" {
		return nil, nil
	}
	parts := strings.Split(v, ",")
	if len(parts) > MaxChatSeeds {
		return nil, errors.New("seed_items lists more than " + strconv.Itoa(MaxChatSeeds) + " items")
	}
	out := make([]int32, 0, len(parts))
	for _, p := range parts {
		id, err := strconv.ParseInt(strings.TrimSpace(p), 10, 32)
		if err != nil || !known(int32(id)) {
			return nil, errors.New("seed_items must be catalog item ids")
		}
		out = append(out, int32(id))
	}
	return out, nil
}

// chatHistory is what the sources see on the chat surface: the user's history (oldest first)
// followed by the seed items, the most relevant seed last, i.e. as the most recent item (it
// weighs most in ItemCF and SASRec). Other surfaces have no seeds: the history as it is.
func chatHistory(history, seeds []int32) []int32 {
	if len(seeds) == 0 {
		return history
	}
	out := make([]int32, 0, len(history)+len(seeds))
	out = append(out, history...)
	for i := len(seeds) - 1; i >= 0; i-- {
		out = append(out, seeds[i])
	}
	return out
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
		s.warn("bad_params", "bad experiment params", "exp_id", a.ExpID, "err", err)
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
// served event; the items at the slots' positions are sponsored.
func (s *Server) render(
	requestID uuid.UUID, surface string, seed int32, assign map[string]experiment.Assignment,
	configVersion int, degraded []string, list []mixer.Fused, slots []cache.Slot, offset, limit int, pageable bool, hv int64,
) (RecResponse, servedPayload) {
	end := min(offset+limit, len(list))
	resp := RecResponse{
		RequestID: requestID.String(), ModelVersion: s.active().Version, Surface: surface,
		Degraded: degraded, Experiments: assign, ConfigVersion: configVersion, Items: []RecItem{},
	}
	served := servedPayload{
		Surface: surface, Offset: offset, ModelVer: s.active().Version, Exp: assign,
		ConfigVersion: configVersion, Degraded: degraded, HV: hv, Items: []servedItem{},
	}
	if seed >= 0 {
		served.Seed = &seed
	}
	for i := offset; i < end; i++ {
		f := list[i]
		pos := i + 1
		it := RecItem{
			Position: pos, ItemID: f.ItemID, Slot: SlotOrganic, Source: f.Primary, Score: f.Score,
			ImpressionID: events.ImpressionID(requestID, pos), Sources: f.Hits, Item: s.card(f.ItemID),
		}
		ranks := make(map[string]int, len(f.Hits))
		for _, h := range f.Hits {
			ranks[h.Source] = h.Rank
		}
		sv := servedItem{ItemID: f.ItemID, Pos: pos, Slot: SlotOrganic, Source: f.Primary, Score: f.Score, Ranks: ranks}
		for _, sl := range slots {
			if sl.Position == pos && sl.ItemID == f.ItemID {
				sponsoredItem(&it, &sv, sl)
			}
		}
		if it.Sources == nil {
			it.Sources = []mixer.Hit{}
		}
		resp.Items = append(resp.Items, it)
		served.Items = append(served.Items, sv)
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
		s.warn("served_dropped", "served event dropped: buffer full", "request_id", requestID)
	}
}

func (s *Server) nextPage(w http.ResponseWriter, r *http.Request, uid int64, cursor string, limit int, debugOn bool, t0 time.Time) {
	cur, err := cache.DecodeCursor(cursor)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	q := &request{s: s, ctx: r.Context(), surface: "page", uid: uid, now: s.Now()}
	if !s.Naive {
		var cancel context.CancelFunc
		q.ctx, cancel, q.budget = resilience.Start(r.Context(), s.Plan, time.Now)
		defer cancel()
	}
	var pg *cache.Page
	t := time.Now()
	// The next page is one Redis read and no recall: it may use half of the request budget.
	// Past that the client gets 503 and retries with the same cursor (the page is still there).
	status, err := q.redis(q.budget.For(s.Plan.Total/2, renderReserve), func(c context.Context) error {
		var err error
		pg, err = s.Pages.Get(c, cur.RequestID)
		if errors.Is(err, cache.ErrNoPage) {
			return nil // a healthy Redis answering "no such key"
		}
		return err
	})
	q.stage("page_get", t)
	if status != mixer.StatusOK {
		writeError(w, http.StatusServiceUnavailable, "pages unavailable: "+err.Error())
		return
	}
	if pg == nil {
		writeError(w, http.StatusGone, "the result list expired; request the first page again")
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
	resp, served := s.render(pg.RequestID, pg.Surface, pg.Seed, pg.Exp, pg.ConfigVersion, pg.Degraded, pg.Items, pg.Sponsored, cur.Offset, limit, true, pg.HV)
	resp.ModelVersion, served.ModelVer = pg.ModelVer, pg.ModelVer
	s.logServed(uid, pg.RequestID, served)
	if debugOn {
		resp.Debug = &Debug{UserID: uid, Offset: cur.Offset, TookMs: float64(time.Since(t0).Microseconds()) / 1000}
	}
	q.stage("total", t0)
	w.Header().Set("X-Candidate-Cache", "page")
	writeJSON(w, http.StatusOK, resp)
}

// adsAllowed is the ads rule of the degradation chain (design §5.1): no sponsored items when
// the request lost user state (no budget or frequency-cap data), the SASRec recall (no ad
// relevance scores) or the ranker. "When in doubt, no ads: better to earn less than to charge
// wrongly." The ads stage (selectAds) consults it.
func adsAllowed(degraded []string) bool {
	for _, d := range degraded {
		if strings.HasPrefix(d, "user_state_") || strings.HasPrefix(d, "sasrec_") ||
			strings.HasPrefix(d, "rank_") || d == "static_popular" || d == "model_version_mismatch" {
			return false
		}
	}
	return true
}
