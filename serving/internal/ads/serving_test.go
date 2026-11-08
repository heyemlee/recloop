package ads_test

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"testing"
	"time"

	"pgregory.net/rapid"

	"github.com/heyemlee/recloop/serving/internal/ads"
)

var day = time.Date(2026, 11, 7, 0, 0, 0, 0, time.UTC)

func at(h, m int) time.Time {
	return day.Add(time.Duration(h)*time.Hour + time.Duration(m)*time.Minute)
}

func ptr(t time.Time) *time.Time { return &t }

func live(id int64, bid, budget int32) *ads.Live {
	return &ads.Live{CampaignID: id, AdvertiserID: int32(id), BidCents: bid, FreqCap: 3,
		DailyBudget: budget, RowDay: day, RowBudget: budget, Items: []int32{int32(id) * 10, int32(id)*10 + 1}}
}

func TestSnapshotKeepsActiveCampaignsWithItems(t *testing.T) {
	cs := []ads.Campaign{
		{ID: 1, Settings: ads.Settings{Status: ads.StatusActive, DailyBudgetCents: 500, ItemIDs: []int32{7, 3}}, Today: ads.BudgetDay{BudgetCents: 400}},
		{ID: 2, Settings: ads.Settings{Status: ads.StatusPaused, DailyBudgetCents: 500, ItemIDs: []int32{1}}},
		{ID: 3, Settings: ads.Settings{Status: ads.StatusEnded, DailyBudgetCents: 500, ItemIDs: []int32{1}}},
		{ID: 4, Settings: ads.Settings{Status: ads.StatusActive, DailyBudgetCents: 500}},
	}
	s := ads.NewSnapshot(cs, day, at(1, 0))
	if len(s.Campaigns) != 1 || s.Campaigns[0].CampaignID != 1 {
		t.Fatalf("campaigns %+v", s.Campaigns)
	}
	l := s.Campaigns[0]
	if l.Items[0] != 3 || l.Items[1] != 7 {
		t.Fatalf("items not sorted: %v", l.Items)
	}
	// The row of the loading day (budget changed and capped during the day), the daily budget
	// on any other day (the next day's row does not exist yet).
	if l.BudgetOn(day) != 400 || l.BudgetOn(day.Add(24*time.Hour)) != 500 {
		t.Fatalf("budgets %d %d", l.BudgetOn(day), l.BudgetOn(day.Add(24*time.Hour)))
	}
	if !s.Fresh(at(1, 1)) || s.Fresh(at(1, 0).Add(ads.MaxStaleness+time.Second)) {
		t.Fatal("freshness")
	}
	var nilSnap *ads.Snapshot
	if nilSnap.Fresh(at(1, 0)) {
		t.Fatal("a missing snapshot is not fresh")
	}
}

func TestIndexKeepsTheLastSnapshotOnError(t *testing.T) {
	fail := false
	load := func(context.Context, time.Time) ([]ads.Campaign, error) {
		if fail {
			return nil, errors.New("down")
		}
		return []ads.Campaign{{ID: 9, Settings: ads.Settings{Status: ads.StatusActive, DailyBudgetCents: 1, ItemIDs: []int32{1}}}}, nil
	}
	x := ads.NewIndex(load, func() time.Time { return at(5, 0) }, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if x.Current() != nil {
		t.Fatal("empty before the first refresh")
	}
	if err := x.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	fail = true
	if err := x.Refresh(context.Background()); err == nil {
		t.Fatal("want the load error")
	}
	if s := x.Current(); s == nil || len(s.Campaigns) != 1 || !s.Day.Equal(day) || !s.LoadedAt.Equal(at(5, 0)) {
		t.Fatalf("snapshot %+v", s)
	}
}

func TestFlightWindow(t *testing.T) {
	l := live(1, 10, 1000)
	if !l.InFlight(at(0, 0)) {
		t.Fatal("no bounds: always in flight")
	}
	from, to := l.Window(day)
	if !from.Equal(day) || !to.Equal(day.Add(24*time.Hour)) {
		t.Fatalf("window %v %v", from, to)
	}
	l.StartAt, l.EndAt = ptr(at(9, 0)), ptr(at(9, 30))
	for _, c := range []struct {
		t  time.Time
		in bool
	}{{at(8, 59), false}, {at(9, 0), true}, {at(9, 29), true}, {at(9, 30), false}} {
		if l.InFlight(c.t) != c.in {
			t.Fatalf("InFlight(%v) = %v", c.t, !c.in)
		}
	}
	from, to = l.Window(day)
	if !from.Equal(at(9, 0)) || !to.Equal(at(9, 30)) {
		t.Fatalf("window %v %v", from, to)
	}
	// A flight over midnight: today's part ends at midnight.
	l.StartAt, l.EndAt = ptr(at(23, 0)), ptr(at(25, 0))
	if from, to = l.Window(day); !from.Equal(at(23, 0)) || !to.Equal(at(24, 0)) {
		t.Fatalf("window %v %v", from, to)
	}
}

func TestBudgetGate(t *testing.T) {
	l := live(1, 10, 1000)
	cases := []struct {
		margin int32
		spent  int64
		ok     bool
	}{
		{0, 999, true}, {0, 1000, false}, {0, 1200, false}, // margin 0: anything left
		{5, 949, true}, {5, 950, false}, // margin 5 clicks of 10¢: more than 50¢ left
		{20, 799, true}, {20, 800, false},
	}
	for _, c := range cases {
		got := ads.Policy{SafetyMarginClicks: c.margin}.Admit(l, c.spent, 0, at(12, 0), 0)
		if (got == "") != c.ok || (!c.ok && got != ads.GateBudget) {
			t.Errorf("margin %d spent %d: %q", c.margin, c.spent, got)
		}
	}
}

// The budget row (refreshed every few seconds) is a floor under the Redis counter: a counter
// that Redis lost does not make a spent campaign look new.
func TestSpendFloorFromTheBudgetRow(t *testing.T) {
	l := live(1, 10, 1000)
	l.RowSpent = 995
	if l.Spent(day, 0) != 995 || l.Spent(day, 998) != 998 || l.Spent(day.Add(24*time.Hour), 3) != 3 {
		t.Fatalf("spent %d %d %d", l.Spent(day, 0), l.Spent(day, 998), l.Spent(day.Add(24*time.Hour), 3))
	}
	if g := (ads.Policy{}).Admit(l, 0, 0, at(12, 0), 0); g != "" {
		t.Fatalf("5 left: %q", g)
	}
	l.RowSpent = 1000 // the counter was evicted (0), the row says it is spent
	if g := (ads.Policy{}).Admit(l, 0, 0, at(12, 0), 0); g != ads.GateBudget {
		t.Fatalf("evicted counter: %q", g)
	}
	cs := []ads.Campaign{{ID: 1, Settings: ads.Settings{Status: ads.StatusActive, DailyBudgetCents: 500, ItemIDs: []int32{1}},
		Today: ads.BudgetDay{BudgetCents: 500, SpentCents: 120}}}
	if s := ads.NewSnapshot(cs, day, at(1, 0)); s.Campaigns[0].RowSpent != 120 {
		t.Fatalf("row spent %d", s.Campaigns[0].RowSpent)
	}
}

func TestFrequencyCap(t *testing.T) {
	l := live(1, 10, 1000)
	p := ads.Policy{}
	if g := p.Admit(l, 0, 2, at(12, 0), 0); g != "" {
		t.Fatalf("2 of 3: %q", g)
	}
	if g := p.Admit(l, 0, 3, at(12, 0), 0); g != ads.GateFreqCap {
		t.Fatalf("3 of 3: %q", g)
	}
	// The budget gate comes first.
	if g := p.Admit(l, 1000, 3, at(12, 0), 0); g != ads.GateBudget {
		t.Fatalf("both: %q", g)
	}
}

func TestPacingRate(t *testing.T) {
	from, to := at(9, 0), at(10, 0)
	cases := []struct {
		now   time.Time
		spent int64
		want  float64
	}{
		{at(9, 0), 0, 1},     // start: on schedule
		{at(9, 0), 10, 0},    // a whole bid ahead of a zero target
		{at(9, 0), 5, 0.5},   // half a bid ahead
		{at(9, 30), 500, 1},  // half the budget at half time
		{at(9, 30), 400, 1},  // behind
		{at(9, 30), 505, .5}, // half a bid ahead
		{at(9, 30), 600, 0},  // ahead
		{at(10, 0), 999, 1},  // the end: the whole budget is due
		{at(8, 0), 0, 1},     // before the window (not in flight anyway)
	}
	for _, c := range cases {
		if got := ads.PacingRate(1000, c.spent, 10, c.now, from, to); got != c.want {
			t.Errorf("PacingRate(%v, spent %d) = %v, want %v", c.now.Format("15:04"), c.spent, got, c.want)
		}
	}
	if ads.PacingRate(1000, 900, 10, at(9, 0), to, from) != 1 {
		t.Error("an empty window does not pace")
	}
}

// Pacing never serves a campaign one bid or more ahead of its schedule, and always serves one
// on or behind it.
func TestPacingFollowsTheSchedule(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		budget := rapid.Int64Range(1, 100_000).Draw(t, "budget")
		bid := rapid.Int64Range(5, 10_000).Draw(t, "bid")
		spent := rapid.Int64Range(0, budget).Draw(t, "spent")
		minutes := rapid.IntRange(1, 1440).Draw(t, "window")
		into := rapid.IntRange(0, minutes).Draw(t, "elapsed")
		from := day
		to := from.Add(time.Duration(minutes) * time.Minute)
		now := from.Add(time.Duration(into) * time.Minute)
		draw := rapid.Float64Range(0, 0.999999).Draw(t, "draw")
		l := &ads.Live{CampaignID: 1, BidCents: int32(bid), FreqCap: 3, DailyBudget: int32(budget),
			RowDay: day, RowBudget: int32(budget), StartAt: &from, EndAt: &to}
		got := ads.Policy{Pacing: true}.Admit(l, spent, 0, now, draw)
		target := float64(budget) * float64(into) / float64(minutes)
		switch {
		case got == ads.GateBudget:
			if spent < budget {
				t.Fatalf("budget gate with %d of %d spent", spent, budget)
			}
		case float64(spent) >= target+float64(bid) && got == "":
			t.Fatalf("served %d ahead of a target of %.1f (bid %d)", spent, target, bid)
		case float64(spent) <= target && got != "":
			t.Fatalf("held back on schedule: spent %d, target %.1f: %q", spent, target, got)
		}
	})
}

func TestPrepareAndCandidates(t *testing.T) {
	a, b, c := live(1, 30, 1000), live(2, 20, 1000), live(3, 10, 1000)
	c.StartAt = ptr(at(13, 0)) // not yet in flight
	s := &ads.Snapshot{Campaigns: []*ads.Live{a, b, c}, Day: day, LoadedAt: at(12, 0)}
	// Item 10 of campaign 1 is not allowed (e.g. in the organic results); campaign 2 has none left.
	pend := s.Prepare(ads.Request{Now: at(12, 0), Allowed: func(it int32) bool { return it != 10 && it/10 != 2 }})
	if len(pend.Campaigns) != 1 || pend.Campaigns[0] != a || len(pend.Items[0]) != 1 || pend.Items[0][0] != 11 {
		t.Fatalf("pending %+v %v", pend.Campaigns, pend.Items)
	}
	if pend.Stats.Campaigns != 3 || pend.Stats.InFlight != 2 || pend.Stats.NoItem != 1 {
		t.Fatalf("stats %+v", pend.Stats)
	}
	cands, st := ads.Policy{}.Candidates(pend, []ads.Usage{{SpentCents: 10, Seen: 1}}, at(12, 0), nil)
	if len(cands) != 1 || cands[0] != (ads.Candidate{CampaignID: 1, AdvertiserID: 1, ItemID: 11, BidCents: 30}) || st.Candidates != 1 {
		t.Fatalf("candidates %+v %+v", cands, st)
	}
	_, st = ads.Policy{}.Candidates(pend, []ads.Usage{{SpentCents: 1000}}, at(12, 0), nil)
	if st.Budget != 1 || st.Candidates != 0 {
		t.Fatalf("budget stats %+v", st)
	}
	_, st = ads.Policy{}.Candidates(pend, []ads.Usage{{Seen: 3}}, at(12, 0), nil)
	if st.FreqCap != 1 {
		t.Fatalf("cap stats %+v", st)
	}
	// Paced, at noon of a whole-day window: the target is 500, and 540 is more than one 30¢ bid
	// ahead of it, so the rate is 0.
	_, st = ads.Policy{Pacing: true}.Candidates(pend, []ads.Usage{{SpentCents: 540}}, at(12, 0), func() float64 { return 0.5 })
	if st.Pacing != 1 {
		t.Fatalf("pacing stats %+v", st)
	}
	var none *ads.Snapshot
	if p := none.Prepare(ads.Request{Now: at(12, 0)}); len(p.Campaigns) != 0 {
		t.Fatal("no snapshot, no campaigns")
	}
}
