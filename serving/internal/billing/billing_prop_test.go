package billing_test

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgtype"
	"pgregory.net/rapid"

	"github.com/heyemlee/recloop/serving/internal/ads"
	"github.com/heyemlee/recloop/serving/internal/adtoken"
	"github.com/heyemlee/recloop/serving/internal/billing"
	"github.com/heyemlee/recloop/serving/internal/store"
)

// What a click's token verifies as (ad_clicks.token_status).
const (
	tokValid    = "valid"
	tokExpired  = "expired"
	tokMismatch = "mismatch"
	tokInvalid  = "invalid"
)

// event is one click event of a scenario.
type event struct {
	click  billing.Click
	imp    int    // index into scenario.imps
	status string // what its token must verify as
}

type scenario struct {
	campaigns []int64
	daily     map[int64]int32 // daily budget at the start
	imps      []impression
	byID      map[string]adtoken.Claims // impression id -> claims
	events    []event
	order     []int // deliveries: indexes into events, technical duplicates included
}

// drawScenario: 1-3 campaigns with small budgets (they run out), up to 20 impressions served
// on day0 or day0+1, 0-3 click events per impression (repeat clicks), some late, forged or
// sent by another user, each event delivered 1-3 times, all in a random order.
func (e *env) drawScenario(t *rapid.T) scenario {
	sc := scenario{daily: map[int64]int32{}, byID: map[string]adtoken.Claims{}}
	for range rapid.IntRange(1, 3).Draw(t, "campaigns") {
		b := rapid.Int32Range(1, 300).Draw(t, "budget")
		id := e.campaign(t, b)
		sc.campaigns = append(sc.campaigns, id)
		sc.daily[id] = b
	}
	for range rapid.IntRange(1, 20).Draw(t, "impressions") {
		at := day0.AddDate(0, 0, rapid.IntRange(0, 1).Draw(t, "day")).
			Add(time.Duration(rapid.IntRange(0, 86399).Draw(t, "second")) * time.Second)
		c := adtoken.Claims{
			RequestID:  uuid.Must(uuid.NewV7()),
			Position:   rapid.IntRange(1, 20).Draw(t, "position"),
			CampaignID: rapid.SampledFrom(sc.campaigns).Draw(t, "campaign"),
			ItemID:     rapid.Int32Range(0, 30).Draw(t, "item"),
			PriceCents: rapid.Int32Range(1, 80).Draw(t, "price"),
			UserID:     rapid.Int64Range(0, 50).Draw(t, "user"),
		}
		tok, err := e.keys.Sign(c, at)
		if err != nil {
			t.Fatalf("sign: %v", err)
		}
		c.IssuedAt, c.ExpiresAt = at, at.Add(adtoken.TTL)
		sc.imps = append(sc.imps, impression{claims: c, token: tok})
		sc.byID[c.ImpressionID()] = c
	}
	for i, im := range sc.imps {
		for range rapid.IntRange(0, 3).Draw(t, "clicks") {
			delay := time.Duration(rapid.IntRange(0, 40*60).Draw(t, "delay seconds")) * time.Second
			c := im.click(delay)
			status := tokValid
			switch rapid.IntRange(0, 9).Draw(t, "kind") {
			case 0:
				c.AdToken, status = tamper(c.AdToken), tokInvalid
			case 1:
				c.UserID, status = c.UserID+1, tokMismatch
			}
			if status == tokValid && delay >= adtoken.TTL {
				status = tokExpired
			}
			sc.events = append(sc.events, event{click: c, imp: i, status: status})
		}
	}
	var order []int
	for i := range sc.events {
		for range rapid.IntRange(1, 3).Draw(t, "copies") {
			order = append(order, i)
		}
	}
	sc.order = rapid.Permutation(order).Draw(t, "order")
	return sc
}

type campaignDay struct {
	campaign int64
	day      string // YYYY-MM-DD
}

func keyOf(campaign int64, day time.Time) campaignDay {
	return campaignDay{campaign, day.Format(time.DateOnly)}
}

type budgetRow struct{ budget, spent int32 }

type charge struct {
	event         uuid.UUID
	campaign      int64
	day           string
	price, billed int32
}

// state is the ledger: the budget rows, charges and processed clicks of a scenario's
// campaigns, and the Redis spend counters.
type state struct {
	rows    map[campaignDay]budgetRow
	charges map[string]charge     // impression id -> charge
	claimed map[uuid.UUID]string  // event id -> token status
	redis   map[campaignDay]int64 // spend counters
}

func (e *env) snapshot(t fataler, sc scenario) state {
	t.Helper()
	ctx := context.Background()
	s := state{rows: map[campaignDay]budgetRow{}, charges: map[string]charge{},
		claimed: map[uuid.UUID]string{}, redis: map[campaignDay]int64{}}
	rows, err := e.st.Pool.Query(ctx,
		"SELECT campaign_id, day, budget_cents, spent_cents FROM campaign_budget_daily WHERE campaign_id = ANY($1)", sc.campaigns)
	if err != nil {
		t.Fatalf("budget rows: %v", err)
	}
	for rows.Next() {
		var cid int64
		var day time.Time
		var r budgetRow
		if err := rows.Scan(&cid, &day, &r.budget, &r.spent); err != nil {
			t.Fatalf("scan: %v", err)
		}
		k := keyOf(cid, day)
		s.rows[k] = r
		if s.redis[k], err = e.spend.Spent(ctx, cid, day); err != nil {
			t.Fatalf("redis: %v", err)
		}
	}
	rows, err = e.st.Pool.Query(ctx, `SELECT impression_id, click_event_id, campaign_id, day, price_cents, billed_cents
		FROM ad_charges WHERE campaign_id = ANY($1)`, sc.campaigns)
	if err != nil {
		t.Fatalf("charges: %v", err)
	}
	for rows.Next() {
		var id string
		var ev pgtype.UUID
		var day time.Time
		var c charge
		if err := rows.Scan(&id, &ev, &c.campaign, &day, &c.price, &c.billed); err != nil {
			t.Fatalf("scan: %v", err)
		}
		c.event, c.day = uuid.UUID(ev.Bytes), day.Format(time.DateOnly)
		s.charges[id] = c
	}
	ids := make([]uuid.UUID, len(sc.events))
	for i, ev := range sc.events {
		ids[i] = ev.click.EventID
	}
	rows, err = e.st.Pool.Query(ctx, "SELECT event_id, token_status FROM ad_clicks WHERE event_id = ANY($1)", ids)
	if err != nil {
		t.Fatalf("clicks: %v", err)
	}
	for rows.Next() {
		var ev pgtype.UUID
		var status string
		if err := rows.Scan(&ev, &status); err != nil {
			t.Fatalf("scan: %v", err)
		}
		s.claimed[uuid.UUID(ev.Bytes)] = status
	}
	if rows.Err() != nil {
		t.Fatalf("rows: %v", rows.Err())
	}
	return s
}

// model is billing done by hand, one delivery at a time: what the database must hold after
// the same deliveries were processed in the same order.
type model struct {
	daily map[int64]int32
	state
}

func newModel(sc scenario) *model {
	m := &model{daily: map[int64]int32{}, state: state{rows: map[campaignDay]budgetRow{},
		charges: map[string]charge{}, claimed: map[uuid.UUID]string{}, redis: map[campaignDay]int64{}}}
	for c, b := range sc.daily {
		m.daily[c] = b
	}
	return m
}

// deliver processes one delivery that committed; redisLost: the process died before the Redis
// update.
func (m *model) deliver(sc scenario, ev event, redisLost bool) billing.Outcome {
	if _, ok := m.claimed[ev.click.EventID]; ok {
		return billing.DuplicateEvent
	}
	m.claimed[ev.click.EventID] = ev.status
	switch ev.status {
	case tokInvalid:
		return billing.Invalid
	case tokMismatch:
		return billing.Mismatch
	case tokExpired:
		return billing.Expired
	}
	c := sc.imps[ev.imp].claims
	k := keyOf(c.CampaignID, c.Day())
	r, ok := m.rows[k]
	if !ok {
		r = budgetRow{budget: m.daily[c.CampaignID]}
	}
	if _, ok := m.charges[c.ImpressionID()]; ok {
		m.rows[k] = r
		return billing.DuplicateImpression
	}
	billed := min(c.PriceCents, r.budget-r.spent)
	m.charges[c.ImpressionID()] = charge{event: ev.click.EventID, campaign: c.CampaignID, day: k.day, price: c.PriceCents, billed: billed}
	r.spent += billed
	m.rows[k] = r
	if !redisLost {
		m.redis[k] += int64(billed)
	}
	return billing.Charged
}

// setBudget is a PATCH of the daily budget made on day today.
func (m *model) setBudget(cid int64, today time.Time, budget int32) {
	if m.daily[cid] == budget {
		return // unchanged: the store touches no budget row
	}
	m.daily[cid] = budget
	k := keyOf(cid, today)
	r := m.rows[k]
	r.budget = max(budget, r.spent)
	m.rows[k] = r
}

func (m *model) compare(t fataler, got state) {
	t.Helper()
	diff := func(what string, want, have any) { t.Fatalf("%s:\n model %v\n    db %v", what, want, have) }
	if len(got.rows) != len(m.rows) {
		diff("budget rows", m.rows, got.rows)
	}
	for k, r := range m.rows {
		if got.rows[k] != r {
			diff(fmt.Sprintf("budget row %v", k), r, got.rows[k])
		}
		if got.redis[k] != m.redis[k] {
			diff(fmt.Sprintf("redis spend %v", k), m.redis[k], got.redis[k])
		}
	}
	if len(got.charges) != len(m.charges) {
		diff("charges", m.charges, got.charges)
	}
	for id, c := range m.charges {
		if got.charges[id] != c {
			diff("charge "+id, c, got.charges[id])
		}
	}
	if len(got.claimed) != len(m.claimed) {
		diff("processed clicks", m.claimed, got.claimed)
	}
	for ev, s := range m.claimed {
		if got.claimed[ev] != s {
			diff("click "+ev.String(), s, got.claimed[ev])
		}
	}
}

// checkTruth checks what holds for any processing order, without the model: exactly one
// charge per impression that had a valid click; each charge carries its token's price,
// campaign and day and one of the impression's valid clicks; every event is recorded with its
// token status; no budget row is spent past its budget and spent is the sum of its charges.
// Where the budget never changed, a campaign-day billed exactly min(budget, sum of the prices
// of its charged impressions), whatever the order.
func checkTruth(t fataler, sc scenario, got state, changed map[int64]bool) {
	t.Helper()
	valid := map[string]map[uuid.UUID]bool{}
	for _, ev := range sc.events {
		if got.claimed[ev.click.EventID] != ev.status {
			t.Fatalf("event %s recorded as %q, its token is %q", ev.click.EventID, got.claimed[ev.click.EventID], ev.status)
		}
		if ev.status == tokValid {
			id := sc.imps[ev.imp].claims.ImpressionID()
			if valid[id] == nil {
				valid[id] = map[uuid.UUID]bool{}
			}
			valid[id][ev.click.EventID] = true
		}
	}
	if len(got.claimed) != len(sc.events) {
		t.Fatalf("%d processed clicks for %d events", len(got.claimed), len(sc.events))
	}
	if len(got.charges) != len(valid) {
		t.Fatalf("%d charges for %d impressions with a valid click", len(got.charges), len(valid))
	}
	prices, billed := map[campaignDay]int64{}, map[campaignDay]int64{}
	for id, ch := range got.charges {
		c := sc.byID[id]
		switch {
		case !valid[id][ch.event]:
			t.Fatalf("charge %s billed event %s, not one of its valid clicks", id, ch.event)
		case ch.price != c.PriceCents || ch.campaign != c.CampaignID || ch.day != c.Day().Format(time.DateOnly):
			t.Fatalf("charge %s = %+v, token %+v", id, ch, c)
		case ch.billed < 0 || ch.billed > ch.price:
			t.Fatalf("charge %s billed %d of %d", id, ch.billed, ch.price)
		}
		k := campaignDay{ch.campaign, ch.day}
		prices[k] += int64(ch.price)
		billed[k] += int64(ch.billed)
	}
	for k, r := range got.rows {
		switch {
		case r.spent > r.budget:
			t.Fatalf("%v: spent %d past the budget %d", k, r.spent, r.budget)
		case int64(r.spent) != billed[k]:
			t.Fatalf("%v: spent %d, charges add up to %d", k, r.spent, billed[k])
		case !changed[k.campaign] && (r.budget != sc.daily[k.campaign] || billed[k] != min(int64(r.budget), prices[k])):
			t.Fatalf("%v: billed %d with budget %d (daily %d) and prices %d", k, billed[k], r.budget, sc.daily[k.campaign], prices[k])
		}
	}
	for k := range billed {
		if _, ok := got.rows[k]; !ok {
			t.Fatalf("charges of %v without a budget row", k)
		}
	}
}

var crashSteps = []billing.Step{billing.StepClaimed, billing.StepLocked, billing.StepCharged, billing.StepSpent, billing.StepCommitted}

// drawCrash: no crash most of the time, else one of the steps.
func drawCrash(t *rapid.T, steps []billing.Step) billing.Step {
	if n := rapid.IntRange(0, 24).Draw(t, "crash"); n < len(steps) {
		return steps[n]
	}
	return ""
}

// Property: deliveries processed one at a time, in any order, with technical and business
// duplicates, late and forged clicks, budget changes, and crashes at every step (the crashed
// delivery comes back later, as after a consumer restart), leave exactly the ledger the model
// computes; and that ledger satisfies checkTruth.
func TestDeliveriesMatchTheModel(t *testing.T) {
	e := newEnv(t)
	rapid.Check(t, func(t *rapid.T) {
		sc := e.drawScenario(t)
		m := newModel(sc)
		changed := map[int64]bool{}
		crashes := map[int]int{}
		queue := slices.Clone(sc.order)
		for len(queue) > 0 {
			if rapid.IntRange(0, 9).Draw(t, "change budget") == 0 {
				cid := rapid.SampledFrom(sc.campaigns).Draw(t, "campaign")
				today := day0.AddDate(0, 0, rapid.IntRange(0, 1).Draw(t, "today"))
				budget := rapid.Int32Range(1, 300).Draw(t, "new budget")
				e.setBudget(t, cid, budget, today)
				m.setBudget(cid, today, budget)
				changed[cid] = true
			}
			i := queue[0]
			queue = queue[1:]
			ev := sc.events[i]
			var at billing.Step
			if crashes[i] < 3 {
				at = drawCrash(t, crashSteps)
			}
			ctx, plan := withPlan(at)
			res, err := e.svc.Handle(ctx, ev.click)
			if plan.fired {
				if err == nil {
					t.Fatalf("crash at %s did not fail the delivery", at)
				}
				crashes[i]++
				if at == billing.StepCommitted { // committed; Redis update lost
					m.deliver(sc, ev, true)
				}
				j := rapid.IntRange(0, len(queue)).Draw(t, "redelivered at")
				queue = slices.Insert(queue, j, i)
				continue
			}
			if err != nil {
				t.Fatalf("event %d: %v", i, err)
			}
			want := m.deliver(sc, ev, false)
			if res.Outcome != want {
				t.Fatalf("event %d (token %s): outcome %s, model %s", i, ev.status, res.Outcome, want)
			}
			if want == billing.Charged && res.BilledCents != m.charges[sc.imps[ev.imp].claims.ImpressionID()].billed {
				t.Fatalf("event %d billed %d, model %d", i, res.BilledCents, m.charges[sc.imps[ev.imp].claims.ImpressionID()].billed)
			}
		}
		got := e.snapshot(t, sc)
		m.compare(t, got)
		checkTruth(t, sc, got, changed)
	})
}

func (e *env) setBudget(t fataler, cid int64, budget int32, today time.Time) {
	t.Helper()
	ctx := context.Background()
	cur, err := e.st.Campaign(ctx, cid, today)
	if err != nil {
		t.Fatalf("campaign %d: %v", cid, err)
	}
	patch := ads.Patch{DailyBudgetCents: ads.Opt[int32]{Set: true, V: budget}}
	if _, err := e.st.UpdateCampaign(ctx, cid, cur.Version, patch, func(*ads.Settings) error { return nil }, today); err != nil {
		t.Fatalf("budget of %d: %v", cid, err)
	}
}

// Property: the same deliveries processed by several workers at once (copies of one event in
// flight together, crashes before COMMIT redelivered) end in a ledger that satisfies
// checkTruth, with every event's outcome counted exactly once: one Charged per impression with
// a valid click, one DuplicateImpression per further valid click, one DuplicateEvent per extra
// delivery. Redis matches the ledger (no crash after COMMIT here).
func TestConcurrentDeliveries(t *testing.T) {
	e := newEnv(t)
	preCommit := crashSteps[:4]
	rapid.Check(t, func(t *rapid.T) {
		sc := e.drawScenario(t)
		workers := rapid.IntRange(2, 8).Draw(t, "workers")
		type job struct {
			event int
			crash billing.Step
		}
		jobs := make(chan job, 2*len(sc.order)+1)
		for _, i := range sc.order {
			jobs <- job{i, drawCrash(t, preCommit)}
		}
		var (
			mu       sync.Mutex
			outcomes = map[billing.Outcome]int{}
			billed   = map[campaignDay]int64{}
			failures []string
			pending  = len(sc.order)
		)
		if pending == 0 { // no clicks at all
			close(jobs)
		}
		var wg sync.WaitGroup
		for range workers {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for j := range jobs {
					ctx, plan := withPlan(j.crash)
					res, err := e.svc.Handle(ctx, sc.events[j.event].click)
					mu.Lock()
					switch {
					case plan.fired:
						if err == nil {
							failures = append(failures, fmt.Sprintf("crash at %s did not fail", j.crash))
						}
						jobs <- job{j.event, ""} // redelivered
					case err != nil:
						failures = append(failures, err.Error())
						pending--
					default:
						outcomes[res.Outcome]++
						if res.Outcome == billing.Charged {
							billed[keyOf(res.CampaignID, res.Day)] += int64(res.BilledCents)
						}
						pending--
					}
					if pending == 0 {
						close(jobs)
						pending = -1
					}
					mu.Unlock()
				}
			}()
		}
		wg.Wait()
		if len(failures) > 0 {
			t.Fatalf("%d deliveries failed: %v", len(failures), failures)
		}

		got := e.snapshot(t, sc)
		checkTruth(t, sc, got, nil)
		want := map[billing.Outcome]int{billing.DuplicateEvent: len(sc.order) - len(sc.events)}
		imps := map[int]bool{}
		for _, ev := range sc.events {
			switch ev.status {
			case tokValid:
				if imps[ev.imp] {
					want[billing.DuplicateImpression]++
				} else {
					want[billing.Charged]++
				}
				imps[ev.imp] = true
			case tokExpired:
				want[billing.Expired]++
			case tokMismatch:
				want[billing.Mismatch]++
			case tokInvalid:
				want[billing.Invalid]++
			}
		}
		for _, o := range []billing.Outcome{billing.Charged, billing.DuplicateImpression, billing.DuplicateEvent,
			billing.Expired, billing.Mismatch, billing.Invalid} {
			if outcomes[o] != want[o] {
				t.Fatalf("outcomes %v, want %v", outcomes, want)
			}
		}
		for k, r := range got.rows {
			if billed[k] != int64(r.spent) || got.redis[k] != int64(r.spent) {
				t.Fatalf("%v: spent %d, results billed %d, redis %d", k, r.spent, billed[k], got.redis[k])
			}
		}
	})
}

// Property: budget changes made while clicks are being charged (the PATCH of design §9.5,
// racing billing on today's row) never let spend pass the budget, and today's row ends at
// max(last budget set, spent): a budget lowered below the spend is capped at the spend.
func TestBudgetChangesWhileCharging(t *testing.T) {
	e := newEnv(t)
	rapid.Check(t, func(t *rapid.T) {
		cid := e.campaign(t, rapid.Int32Range(1, 400).Draw(t, "budget"))
		n := rapid.IntRange(1, 30).Draw(t, "clicks")
		clicks := make([]billing.Click, n)
		for i := range clicks {
			im := e.serve(t, cid, rapid.Int32Range(1, 60).Draw(t, "price"), day0.Add(time.Duration(i)*time.Second))
			clicks[i] = im.click(time.Second)
		}
		changes := rapid.SliceOfN(rapid.Int32Range(1, 400), 1, 6).Draw(t, "budget changes")
		workers := rapid.IntRange(1, 6).Draw(t, "workers")

		next := make(chan billing.Click, n)
		for _, c := range clicks {
			next <- c
		}
		close(next)
		errs := make(chan error, n+len(changes))
		var wg sync.WaitGroup
		for range workers {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for c := range next {
					if r, err := e.svc.Handle(context.Background(), c); err != nil || r.Outcome != billing.Charged {
						errs <- fmt.Errorf("click: %+v %v", r, err)
					}
				}
			}()
		}
		wg.Add(1)
		go func() { // the advertiser, retrying when it loses the optimistic lock
			defer wg.Done()
			ctx := context.Background()
			for _, b := range changes {
				for {
					cur, err := e.st.Campaign(ctx, cid, day0)
					if err != nil {
						errs <- err
						return
					}
					patch := ads.Patch{DailyBudgetCents: ads.Opt[int32]{Set: true, V: b}}
					_, err = e.st.UpdateCampaign(ctx, cid, cur.Version, patch, func(*ads.Settings) error { return nil }, day0)
					if err == nil {
						break
					}
					if !errors.Is(err, store.ErrVersionMismatch) {
						errs <- err
						return
					}
				}
			}
		}()
		wg.Wait()
		close(errs)
		for err := range errs {
			t.Fatalf("%v", err)
		}

		budget, spent := e.row(t, cid, day0)
		var sum, charges int
		if err := e.st.Pool.QueryRow(context.Background(),
			"SELECT COALESCE(sum(billed_cents), 0), count(*) FROM ad_charges WHERE campaign_id = $1", cid).Scan(&sum, &charges); err != nil {
			t.Fatalf("%v", err)
		}
		last := changes[len(changes)-1]
		if charges != n || sum != int(spent) || spent > budget || budget != max(last, spent) {
			t.Fatalf("%d charges for %d clicks, billed %d, spent %d, budget %d, last budget set %d",
				charges, n, sum, spent, budget, last)
		}
	})
}

// The budget row of a campaign-day is shared by every click on it; this is the only lock.
func TestChargesOfOtherCampaignsDoNotWait(t *testing.T) {
	e := newEnv(t)
	a, b := e.campaign(t, 1000), e.campaign(t, 1000)
	ctx := context.Background()
	tx, err := e.st.Pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback(ctx)
	if _, err := tx.Exec(ctx, "INSERT INTO campaign_budget_daily (campaign_id, day, budget_cents) VALUES ($1, $2, 1000)", a, day0); err != nil {
		t.Fatal(err)
	}
	if _, err := tx.Exec(ctx, "SELECT 1 FROM campaign_budget_daily WHERE campaign_id = $1 AND day = $2 FOR UPDATE", a, day0); err != nil {
		t.Fatal(err)
	}
	// Campaign a's row is locked (uncommitted); campaign b's click goes through at once.
	cctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	if r, err := e.svc.Handle(cctx, e.serve(t, b, 10, day0.Add(time.Hour)).click(time.Second)); err != nil || r.Outcome != billing.Charged {
		t.Fatalf("%+v %v", r, err)
	}
}
