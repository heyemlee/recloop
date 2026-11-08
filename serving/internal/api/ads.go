package api

import (
	"context"
	"encoding/json"
	"math/rand/v2"
	"net/http"
	"slices"
	"strconv"
	"time"

	"github.com/google/uuid"

	"github.com/heyemlee/recloop/serving/internal/ads"
	"github.com/heyemlee/recloop/serving/internal/adtoken"
	"github.com/heyemlee/recloop/serving/internal/billing"
	"github.com/heyemlee/recloop/serving/internal/cache"
	"github.com/heyemlee/recloop/serving/internal/experiment"
	"github.com/heyemlee/recloop/serving/internal/metrics"
	"github.com/heyemlee/recloop/serving/internal/mixer"
	"github.com/heyemlee/recloop/serving/internal/resilience"
	"github.com/heyemlee/recloop/serving/internal/userstate"
)

// Sponsored slots (design §9.2): the item at a sponsored position is an ad, labelled for the
// user, and its impression and click events carry the slot's ad_token.
const (
	SlotSponsored  = "sponsored"
	SlotOrganic    = "organic"
	LabelSponsored = "Sponsored"
	SourceAds      = "ads"
	SurfaceChat    = "chat"
	// MaxChatAds: a chat reply shows at most one sponsored item (§10.3 (5)).
	MaxChatAds = 1
	// MaxAdCandidates bounds the candidate list of /internal/ads/select.
	MaxAdCandidates = 500
)

// AdServer is what the sponsored slots need.
type AdServer struct {
	Index *ads.Index    // active campaigns, refreshed every ads.RefreshEvery
	PCTR  ads.PCTRModel // ads.FixedPCTR until the pCTR model exists (M4)
	Keys  *adtoken.Keys // signs the ad_tokens
	// Policy is the default serving policy; the params of the user's ads-layer experiment
	// variant ("safety_margin_clicks", "pacing", "min_pctr") override it.
	Policy ads.Policy
	// Draw returns uniform numbers in [0, 1) for pacing (math/rand when nil).
	Draw func() float64
}

func (a *AdServer) draw() float64 {
	if a.Draw != nil {
		return a.Draw()
	}
	return rand.Float64()
}

// adParams are the ads layer's experiment parameters.
type adParams struct {
	SafetyMarginClicks *int32   `json:"safety_margin_clicks"`
	Pacing             *bool    `json:"pacing"`
	MinPCTR            *float64 `json:"min_pctr"`
}

func (s *Server) adPolicy(a experiment.Assignment) ads.Policy {
	p := s.Ads.Policy
	if len(a.Params) == 0 {
		return p
	}
	var v adParams
	if err := json.Unmarshal(a.Params, &v); err != nil {
		s.warn("bad_ads_params", "bad ads experiment params", "exp_id", a.ExpID, "err", err)
		return p
	}
	if v.SafetyMarginClicks != nil && *v.SafetyMarginClicks >= 0 {
		p.SafetyMarginClicks = *v.SafetyMarginClicks
	}
	if v.Pacing != nil {
		p.Pacing = *v.Pacing
	}
	if v.MinPCTR != nil {
		p.MinPCTR = *v.MinPCTR
	}
	return p
}

// Ads-stage outcomes (metric recloop_ad_requests_total, served event, debug panel).
const (
	adsFilled     = "filled"      // at least one sponsored slot
	adsEmpty      = "empty"       // no eligible ad
	adsNotAllowed = "not_allowed" // the degradation chain rules ads out (§5.1)
	adsStale      = "stale"       // the campaign snapshot is too old (PostgreSQL unreachable)
)

// adResult is what the ads stage decided for one request.
type adResult struct {
	Outcome string       `json:"outcome"`
	Policy  adPolicyView `json:"policy"`
	Stats   ads.Stats    `json:"stats"`
	Slots   []cache.Slot `json:"-"`
}

type adPolicyView struct {
	SafetyMarginClicks int32   `json:"safety_margin_clicks"`
	Pacing             bool    `json:"pacing"`
	MinPCTR            float64 `json:"min_pctr"`
}

// selectAds is the ads stage (design §9.2): the in-flight campaigns with an item `allowed`
// admits, one Redis read for their spend today and the user's impressions of them today
// (under the stage's budget), the budget / frequency / pacing gates, pCTR, the GSP auction for
// len(positions) slots, and a signed ad_token per winner (the auction's k-th place gets
// positions[k-1]). Any failure leaves the request without ads and, for a dependency, writes
// ads_<status> into `degraded`; only naive mode turns it into an error.
func (s *Server) selectAds(q *request, requestID uuid.UUID, positions []int, a experiment.Assignment, allowed func(int32) bool) (adResult, error) {
	res := adResult{Outcome: adsEmpty}
	defer func() { metrics.AdRequests.WithLabelValues(q.surface, res.Outcome).Inc() }()
	if !adsAllowed(q.degraded) {
		res.Outcome = adsNotAllowed
		return res, nil
	}
	snap := s.Ads.Index.Current()
	if !snap.Fresh(q.now) {
		res.Outcome = adsStale
		q.degrade("ads_stale")
		s.warn("ads_stale", "no fresh campaign snapshot: no ads")
		return res, nil
	}
	policy := s.adPolicy(a)
	res.Policy = adPolicyView{policy.SafetyMarginClicks, policy.Pacing, policy.MinPCTR}
	pend := snap.Prepare(ads.Request{Now: q.now, Allowed: allowed})
	res.Stats = pend.Stats
	if len(pend.Campaigns) == 0 || len(positions) == 0 {
		return res, nil
	}

	// Spend today (billing's counters) and the user's impressions today (the aggregator's
	// frequency counters), all campaigns in one round trip.
	day := ads.UTCDay(q.now)
	n := len(pend.Campaigns)
	keys := make([]string, 0, 2*n)
	for _, l := range pend.Campaigns {
		keys = append(keys, billing.SpendKey(l.CampaignID, day))
	}
	for _, l := range pend.Campaigns {
		keys = append(keys, userstate.FreqKey(q.uid, l.CampaignID, day))
	}
	var vals []any
	_, span := q.span("ads.usage")
	status, err := q.redis(q.budget.For(s.Plan.Ads, s.Plan.Tail), func(c context.Context) error {
		var err error
		vals, err = s.Redis.MGet(c, keys...).Result()
		return err
	})
	span.End()
	if status != mixer.StatusOK {
		if s.Naive {
			return res, errFatal{"ads", err}
		}
		res.Outcome = status
		q.degrade("ads_" + status)
		s.warn("ads_"+status, "ads usage unavailable: no ads", "err", err)
		return res, nil
	}
	counts := userstate.Ints(vals)
	usage := make([]ads.Usage, n)
	for i := range usage {
		usage[i] = ads.Usage{SpentCents: counts[i], Seen: counts[n+i]}
	}
	cands, st := policy.Candidates(pend, usage, q.now, s.Ads.draw)
	res.Stats = st
	for gate, k := range map[string]int{ads.GateBudget: st.Budget, ads.GateFreqCap: st.FreqCap, ads.GatePacing: st.Pacing} {
		if k > 0 {
			metrics.AdGates.WithLabelValues(gate).Add(float64(k))
		}
	}
	if len(cands) == 0 {
		return res, nil
	}

	// pCTR of every candidate item (at the reference position, §9.2 step 3).
	ids := make([]int32, 0, len(cands))
	for _, c := range cands {
		ids = append(ids, c.ItemID)
	}
	slices.Sort(ids)
	ids = slices.Compact(ids)
	ctrs, err := s.Ads.PCTR.PredictCTR(q.ctx, ads.PCTRRequest{UserID: q.uid, Surface: q.surface, ItemIDs: ids})
	if err != nil || len(ctrs) != len(ids) {
		if s.Naive {
			return res, errFatal{"pctr", err}
		}
		res.Outcome = mixer.StatusError
		q.degrade("pctr_error")
		s.warn("pctr_error", "pCTR failed: no ads", "err", err)
		return res, nil
	}
	ctr := make(map[int32]float64, len(ids))
	for i, id := range ids {
		ctr[id] = ctrs[i]
	}
	for i := range cands {
		cands[i].PCTR = ctr[cands[i].ItemID]
	}

	winners := ads.Run(cands, ads.Config{Slots: len(positions), ReserveCents: ads.ReserveCents, MinPCTR: policy.MinPCTR})
	for _, w := range winners {
		pos := positions[w.Rank-1]
		tok, err := s.Ads.Keys.Sign(adtoken.Claims{
			RequestID: requestID, Position: pos, CampaignID: w.CampaignID, ItemID: w.ItemID,
			PriceCents: w.PriceCents, UserID: q.uid,
		}, q.now)
		if err != nil {
			s.warn("ad_token", "signing an ad_token failed", "err", err)
			continue
		}
		res.Slots = append(res.Slots, cache.Slot{
			Position: pos, ItemID: w.ItemID, CampaignID: w.CampaignID, PriceCents: w.PriceCents,
			PCTRPPM: w.PCTRPPM, AdToken: tok,
		})
		metrics.AdSlots.WithLabelValues(q.surface, strconv.Itoa(pos)).Inc()
	}
	res.Stats.Winners = len(res.Slots)
	if len(res.Slots) > 0 {
		res.Outcome = adsFilled
	}
	return res, nil
}

// homeSlots runs the ads stage for the home feed: positions 3 and 9 of the list (both exist
// once the list is that long), never an item of the organic list itself.
func (s *Server) homeSlots(q *request, requestID uuid.UUID, list []mixer.Fused, uc userstate.Context, a experiment.Assignment) (adResult, error) {
	var positions []int
	for _, p := range ads.Slots {
		if p-1 <= len(list)+len(positions) {
			positions = append(positions, p)
		}
	}
	organic := make(map[int32]bool, len(list))
	for _, f := range list {
		organic[f.ItemID] = true
	}
	allowed := func(it int32) bool {
		return s.Catalog.Has(it) && s.Catalog.InStock(it) && !uc.Bought[it] && !organic[it]
	}
	return s.selectAds(q, requestID, positions, a, allowed)
}

// insertSponsored puts each slot's item at its position (slots in ascending position order);
// the organic items behind it move down, and the list keeps at most `depth` items.
func insertSponsored(list []mixer.Fused, slots []cache.Slot, depth int) []mixer.Fused {
	if len(slots) == 0 {
		return list
	}
	out := make([]mixer.Fused, 0, len(list)+len(slots))
	j := 0
	for _, sl := range slots {
		for len(out) < sl.Position-1 && j < len(list) {
			out = append(out, list[j])
			j++
		}
		out = append(out, mixer.Fused{ItemID: sl.ItemID, Primary: SourceAds})
	}
	out = append(out, list[j:]...)
	if len(out) > depth {
		out = out[:depth]
	}
	return out
}

// ---------------------------------------------------------------- /internal/ads/select

// AdSelectRequest is the body of POST /internal/ads/select: the sponsored item of a chat reply
// (design §10.3 (5)), chosen by the same auction and gates as the feed's slots. The assistant
// checked the request's hard constraints itself and sends the items that satisfy them; only
// those can be advertised, so a sponsored item meets the same constraints as the organic ones.
type AdSelectRequest struct {
	UserID *int64 `json:"user_id"`
	// CandidateItemIDs: the items that satisfy the request's validated hard constraints.
	CandidateItemIDs []int32 `json:"candidate_item_ids"`
	// ExcludeItemIDs: the organic items of the reply (an ad never repeats one).
	ExcludeItemIDs []int32 `json:"exclude_item_ids"`
	// MaxPriceCents, if set, is checked again here: an item without a known price, or above it,
	// is never advertised (an unknown price does not satisfy a budget, §10.3 (2)).
	MaxPriceCents *int32 `json:"max_price_cents"`
	// Slots wanted (default 1, at most MaxChatAds).
	Slots int `json:"slots"`
}

// AdSelectResponse: the sponsored items, labelled, with their ad_tokens. The items' impressions
// and clicks are reported to POST /v1/events like the feed's, with impression_id and ad_token.
type AdSelectResponse struct {
	RequestID     string                           `json:"request_id"`
	Surface       string                           `json:"surface"`
	Degraded      []string                         `json:"degraded"`
	Experiments   map[string]experiment.Assignment `json:"experiments"`
	ConfigVersion int                              `json:"config_version"`
	Items         []RecItem                        `json:"items"`
	Ads           *adResult                        `json:"ads,omitempty"`
}

func (s *Server) selectChatAds(w http.ResponseWriter, r *http.Request) {
	var body AdSelectRequest
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, "body: "+err.Error())
		return
	}
	switch {
	case body.UserID == nil || *body.UserID < 0:
		writeError(w, http.StatusBadRequest, "user_id is required and must be >= 0")
		return
	case len(body.CandidateItemIDs) > MaxAdCandidates:
		writeError(w, http.StatusBadRequest, "candidate_item_ids lists more than 500 items")
		return
	case body.Slots < 0 || body.Slots > MaxChatAds:
		writeError(w, http.StatusBadRequest, "slots must be 1 (a reply shows at most one sponsored item)")
		return
	}
	if body.Slots == 0 {
		body.Slots = 1
	}
	requestID, err := uuid.NewV7()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	q := &request{s: s, ctx: r.Context(), surface: SurfaceChat, seed: -1, uid: *body.UserID, now: s.Now(), degraded: []string{}}
	if !s.Naive {
		plan := s.Plan
		plan.Total = plan.UserContext + plan.Ads + plan.Tail
		var cancel context.CancelFunc
		q.ctx, cancel, q.budget = resilience.Start(r.Context(), plan, time.Now)
		defer cancel()
	}
	cfg := s.Exps.Current()
	configVersion := 0
	if cfg != nil {
		configVersion = cfg.Version
	}
	assign := cfg.Assign(experiment.UserUnit(q.uid))
	resp := AdSelectResponse{
		RequestID: requestID.String(), Surface: SurfaceChat, Experiments: assign, ConfigVersion: configVersion,
		Items: []RecItem{},
	}
	if s.Ads == nil {
		resp.Degraded = q.degraded
		writeJSON(w, http.StatusOK, resp)
		return
	}

	// The user's purchases: a bought item is never advertised.
	var uc userstate.Context
	status, err := q.redis(q.budget.For(s.Plan.UserContext, s.Plan.Ads+s.Plan.Tail), func(c context.Context) error {
		var err error
		uc, err = s.Users.Read(c, q.uid, q.now)
		return err
	})
	if status != mixer.StatusOK {
		if s.Naive {
			writeError(w, http.StatusServiceUnavailable, "user state: "+err.Error())
			return
		}
		q.degrade("user_state_" + status) // rules the ads out (adsAllowed)
	}
	candidates := make(map[int32]bool, len(body.CandidateItemIDs))
	for _, id := range body.CandidateItemIDs {
		candidates[id] = true
	}
	for _, id := range body.ExcludeItemIDs {
		delete(candidates, id)
	}
	allowed := func(it int32) bool {
		if !candidates[it] || !s.Catalog.Has(it) || !s.Catalog.InStock(it) || uc.Bought[it] {
			return false
		}
		if body.MaxPriceCents != nil {
			item, _ := s.Catalog.Get(it)
			if item.PriceCents == nil || *item.PriceCents > *body.MaxPriceCents {
				return false
			}
		}
		return true
	}
	positions := make([]int, body.Slots)
	for i := range positions {
		positions[i] = i + 1
	}
	res, err := s.selectAds(q, requestID, positions, assign["ads"], allowed)
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, err.Error())
		return
	}
	resp.Degraded = q.degraded
	resp.Ads = &res
	if len(res.Slots) > 0 {
		list := make([]mixer.Fused, len(res.Slots))
		for i, sl := range res.Slots {
			list[i] = mixer.Fused{ItemID: sl.ItemID, Primary: SourceAds}
		}
		rendered, served := s.render(requestID, SurfaceChat, -1, assign, configVersion, q.degraded, list, res.Slots, 0, len(list), false, uc.HV)
		resp.Items = rendered.Items
		served.Ads = &res
		s.logServed(q.uid, requestID, served)
	}
	writeJSON(w, http.StatusOK, resp)
}

// sponsoredItem decorates a rendered item at a sponsored position.
func sponsoredItem(it *RecItem, sv *servedItem, sl cache.Slot) {
	it.Slot, it.Label, it.Source = SlotSponsored, LabelSponsored, SourceAds
	it.CampaignID, it.AdToken, it.Sources, it.Score = sl.CampaignID, sl.AdToken, []mixer.Hit{}, 0
	sv.Slot, sv.Source, sv.Score, sv.Ranks = SlotSponsored, SourceAds, 0, map[string]int{}
	sv.CampaignID, sv.Price, sv.PCTRPPM, sv.AdToken = sl.CampaignID, sl.PriceCents, sl.PCTRPPM, sl.AdToken
}
