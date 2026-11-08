package ads

import (
	"context"
	"log/slog"
	"slices"
	"sync/atomic"
	"time"
)

// Ad serving inside a recommendation request (docs/design.md §9.2, §9.4): the in-memory set of
// active campaigns, refreshed from PostgreSQL every few seconds, and the gates a campaign must
// pass on every request (flight window, budget left above a safety margin, frequency cap,
// optional pacing). Spend and frequency come from Redis on every request; the caller reads them
// and hands them in, so everything here is in-process and deterministic given its inputs.

// RefreshEvery is how often ad serving reloads the campaigns (§9.2 step 1): a pause, a new bid
// or a budget change takes effect within this time.
const RefreshEvery = 5 * time.Second

// MaxStaleness: a snapshot older than this (PostgreSQL unreachable for a while) serves no ads.
// When in doubt, no ads (§5.1): a paused campaign must not keep serving.
const MaxStaleness = time.Minute

// Slots are the sponsored positions of the home feed (§9.2 step 6): the auction's first place
// goes to position 3, the second to position 9.
var Slots = []int{3, 9}

// Live is an active campaign as ad serving sees it.
type Live struct {
	CampaignID   int64
	AdvertiserID int32
	BidCents     int32
	FreqCap      int32
	DailyBudget  int32
	// RowDay / RowBudget: the campaign's budget for the UTC day the snapshot was loaded on. It
	// differs from DailyBudget when the budget changed during the day and was capped at what
	// had been spent (§9.5).
	RowDay         time.Time
	RowBudget      int32
	StartAt, EndAt *time.Time
	Items          []int32 // sorted
}

// BudgetOn is the campaign's budget on a UTC day.
func (l *Live) BudgetOn(day time.Time) int32 {
	if l.RowDay.Equal(day) {
		return l.RowBudget
	}
	return l.DailyBudget
}

// InFlight reports whether the campaign may serve at t: start_at <= t < end_at, either bound
// open when unset.
func (l *Live) InFlight(t time.Time) bool {
	return (l.StartAt == nil || !t.Before(*l.StartAt)) && (l.EndAt == nil || t.Before(*l.EndAt))
}

// Window is the part of a UTC day the campaign is in flight: [max(start_at, day), min(end_at,
// day + 24 h)). Pacing spreads the day's budget over it.
func (l *Live) Window(day time.Time) (from, to time.Time) {
	from, to = day, day.Add(24*time.Hour)
	if l.StartAt != nil && l.StartAt.After(from) {
		from = *l.StartAt
	}
	if l.EndAt != nil && l.EndAt.Before(to) {
		to = *l.EndAt
	}
	return from, to
}

// Snapshot is the set of active campaigns at one refresh.
type Snapshot struct {
	Campaigns []*Live
	Day       time.Time // the UTC day whose budget rows were loaded
	LoadedAt  time.Time
}

// NewSnapshot keeps the active campaigns with at least one item. Today is the budget row of
// `day` that the store returned with each campaign.
func NewSnapshot(cs []Campaign, day, loadedAt time.Time) *Snapshot {
	s := &Snapshot{Day: day, LoadedAt: loadedAt}
	for _, c := range cs {
		if c.Status != StatusActive || len(c.ItemIDs) == 0 {
			continue
		}
		items := slices.Clone(c.ItemIDs)
		slices.Sort(items)
		s.Campaigns = append(s.Campaigns, &Live{
			CampaignID: c.ID, AdvertiserID: c.AdvertiserID, BidCents: c.BidCents,
			FreqCap: c.FreqCapPerDay, DailyBudget: c.DailyBudgetCents,
			RowDay: day, RowBudget: c.Today.BudgetCents,
			StartAt: c.StartAt, EndAt: c.EndAt, Items: items,
		})
	}
	return s
}

// Fresh reports whether the snapshot may still be used at now.
func (s *Snapshot) Fresh(now time.Time) bool {
	return s != nil && now.Sub(s.LoadedAt) <= MaxStaleness
}

// LoadFunc returns every campaign with its budget row of `today` (store.Store.Campaigns).
type LoadFunc func(ctx context.Context, today time.Time) ([]Campaign, error)

// Index keeps the current Snapshot.
type Index struct {
	load LoadFunc
	now  func() time.Time
	log  *slog.Logger
	cur  atomic.Pointer[Snapshot]
}

// NewIndex returns an empty index; call Refresh (or Run) to fill it.
func NewIndex(load LoadFunc, now func() time.Time, log *slog.Logger) *Index {
	if now == nil {
		now = time.Now
	}
	return &Index{load: load, now: now, log: log}
}

// Refresh loads a new snapshot; on error the previous one stays (until it is stale).
func (x *Index) Refresh(ctx context.Context) error {
	now := x.now().UTC()
	day := UTCDay(now)
	cs, err := x.load(ctx, day)
	if err != nil {
		return err
	}
	x.cur.Store(NewSnapshot(cs, day, now))
	return nil
}

// Current is the latest snapshot (nil before the first successful refresh).
func (x *Index) Current() *Snapshot { return x.cur.Load() }

// Set replaces the snapshot (tests).
func (x *Index) Set(s *Snapshot) { x.cur.Store(s) }

// Run refreshes every `every` until ctx is done.
func (x *Index) Run(ctx context.Context, every time.Duration) {
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			if err := x.Refresh(ctx); err != nil && ctx.Err() == nil {
				x.log.Warn("campaign refresh failed; keeping the previous snapshot", "err", err)
			}
		}
	}
}

// UTCDay truncates to the UTC day.
func UTCDay(t time.Time) time.Time {
	t = t.UTC()
	return time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, time.UTC)
}

// Policy is how ad serving keeps campaigns inside their budgets (§9.4). Billing never charges
// past a budget whatever serving does; the policy decides how much overdelivery (clicks the
// platform absorbs because the budget ran out before they arrived) it risks.
type Policy struct {
	// SafetyMarginClicks: a campaign stops serving once its remaining budget is at most this
	// many clicks at its bid. Slots served before still bring clicks in for a while.
	SafetyMarginClicks int32
	// Pacing spreads the day's budget evenly over the campaign's flight window of the day.
	Pacing bool
	// MinPCTR is the auction's quality floor.
	MinPCTR float64
}

// Gates a campaign can fail on a request (the empty string: it passed).
const (
	GateBudget  = "budget"   // remaining budget <= margin × bid
	GateFreqCap = "freq_cap" // the user saw it freq_cap_per_day times today
	GatePacing  = "pacing"   // ahead of its even schedule (pacing on)
)

// Admit applies the budget, frequency and pacing gates to a campaign at `now`: spent is its
// spend today (Redis), seen the user's impressions of it today (Redis, from impression events),
// draw a uniform number in [0, 1) for pacing.
func (p Policy) Admit(l *Live, spent, seen int64, now time.Time, draw float64) string {
	day := UTCDay(now)
	budget := int64(l.BudgetOn(day))
	if budget-spent <= int64(max(p.SafetyMarginClicks, 0))*int64(l.BidCents) {
		return GateBudget
	}
	if seen >= int64(l.FreqCap) {
		return GateFreqCap
	}
	if p.Pacing {
		from, to := l.Window(day)
		if draw >= PacingRate(budget, spent, int64(l.BidCents), now, from, to) {
			return GatePacing
		}
	}
	return ""
}

// Stats of one ad request: how many campaigns each step left (written into the served event).
type Stats struct {
	Campaigns  int `json:"campaigns"`  // active campaigns in the snapshot
	InFlight   int `json:"in_flight"`  // ... inside their flight window
	NoItem     int `json:"no_item"`    // ... of which none of the items may be shown here
	Budget     int `json:"budget"`     // stopped by the budget gate
	FreqCap    int `json:"freq_cap"`   // stopped by the frequency cap
	Pacing     int `json:"pacing"`     // held back by pacing
	Candidates int `json:"candidates"` // (campaign, item) pairs in the auction
	Winners    int `json:"winners"`
}

// Request is one ad request. Allowed says whether an item may be advertised here: in stock,
// not bought by the user, not among the organic results, inside the caller's constraints.
type Request struct {
	Now     time.Time
	Allowed func(item int32) bool
}

// Pending are the campaigns of a request that need their spend and frequency read: in flight,
// with at least one allowed item.
type Pending struct {
	Campaigns []*Live
	Items     [][]int32 // the allowed items of each campaign
	Stats     Stats
}

// Prepare picks the campaigns worth a Redis read.
func (s *Snapshot) Prepare(req Request) Pending {
	var p Pending
	if s == nil {
		return p
	}
	p.Stats.Campaigns = len(s.Campaigns)
	for _, l := range s.Campaigns {
		if !l.InFlight(req.Now) {
			continue
		}
		p.Stats.InFlight++
		var items []int32
		for _, it := range l.Items {
			if req.Allowed == nil || req.Allowed(it) {
				items = append(items, it)
			}
		}
		if len(items) == 0 {
			p.Stats.NoItem++
			continue
		}
		p.Campaigns = append(p.Campaigns, l)
		p.Items = append(p.Items, items)
	}
	return p
}

// Usage of a campaign for one request, from Redis.
type Usage struct {
	SpentCents int64 // spend:{campaign}:{day}
	Seen       int64 // fc:{user}:{campaign}:{day}
}

// Candidates applies the gates (Admit) to the pending campaigns, usage[i] being campaign i's,
// and returns the auction's candidates: every allowed item of every admitted campaign, PCTR
// still to be filled in.
func (p Policy) Candidates(pend Pending, usage []Usage, now time.Time, draw func() float64) ([]Candidate, Stats) {
	st := pend.Stats
	var out []Candidate
	for i, l := range pend.Campaigns {
		d := 0.0
		if p.Pacing && draw != nil {
			d = draw()
		}
		switch p.Admit(l, usage[i].SpentCents, usage[i].Seen, now, d) {
		case GateBudget:
			st.Budget++
			continue
		case GateFreqCap:
			st.FreqCap++
			continue
		case GatePacing:
			st.Pacing++
			continue
		}
		for _, it := range pend.Items[i] {
			out = append(out, Candidate{CampaignID: l.CampaignID, AdvertiserID: l.AdvertiserID, ItemID: it, BidCents: l.BidCents})
		}
	}
	st.Candidates = len(out)
	return out, st
}

// PacingRate is the serving probability of a paced campaign. The target spend at now is the
// budget times the share of [from, to) that has passed; on or behind it the campaign serves
// (1), and the probability falls linearly to 0 as its spend gets one bid ahead of the target.
// Spend lags behind serving (clicks arrive later), so the band is kept to one click.
func PacingRate(budget, spent, bid int64, now, from, to time.Time) float64 {
	if !to.After(from) || bid <= 0 {
		return 1
	}
	elapsed := min(max(float64(now.Sub(from))/float64(to.Sub(from)), 0), 1)
	target := elapsed * float64(budget)
	return min(max(1+(target-float64(spent))/float64(bid), 0), 1)
}
