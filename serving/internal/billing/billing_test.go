package billing_test

import (
	"context"
	"errors"
	"flag"
	"io"
	"log/slog"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/redis/go-redis/v9"

	"github.com/heyemlee/recloop/serving/internal/adtoken"
	"github.com/heyemlee/recloop/serving/internal/billing"
	"github.com/heyemlee/recloop/serving/internal/metrics"
	"github.com/heyemlee/recloop/serving/internal/store"
	"github.com/heyemlee/recloop/serving/internal/testutil"
	promtest "github.com/prometheus/client_golang/prometheus/testutil"
)

// The properties run against the real database, so the default number of rapid checks is
// lower than rapid's 100; pass -rapid.checks=N for a long run (see CLAUDE.md).
const defaultChecks = "30"

func TestMain(m *testing.M) {
	flag.Parse()
	set := false
	flag.Visit(func(f *flag.Flag) { set = set || f.Name == "rapid.checks" })
	if !set {
		_ = flag.Set("rapid.checks", defaultChecks)
	}
	os.Exit(m.Run())
}

const testKeys = "t1:billing-test-key-0123456789abcdef0123456789"

// day0 and day0+1 are the days impressions are served on in these tests.
var day0 = time.Date(2026, 11, 6, 0, 0, 0, 0, time.UTC)

var errCrash = errors.New("simulated crash")

// crashPlan says where one Handle call should crash; fired reports whether it did (a click
// that stops early, e.g. as a duplicate, never reaches later steps).
type crashPlan struct {
	at    billing.Step
	fired bool
}

type planKey struct{}

func withPlan(at billing.Step) (context.Context, *crashPlan) {
	p := &crashPlan{at: at}
	return context.WithValue(context.Background(), planKey{}, p), p
}

// crashHook drops the transaction's connection at the planned step, as when the billing
// process is killed there: PostgreSQL sees the session end and rolls the transaction back,
// and Handle fails at its next statement. After COMMIT it makes Handle fail, as if the process
// died before acknowledging the event.
func crashHook(ctx context.Context, step billing.Step, tx pgx.Tx) error {
	p, _ := ctx.Value(planKey{}).(*crashPlan)
	if p == nil || p.at != step || p.fired {
		return nil
	}
	p.fired = true
	if tx == nil {
		return errCrash
	}
	return tx.Conn().Close(context.Background())
}

type env struct {
	st         *store.Store
	svc        *billing.Service
	spend      billing.RedisSpend
	keys       *adtoken.Keys
	advertiser int32
}

func newEnv(t *testing.T) *env {
	t.Helper()
	st := testutil.PG(t)
	mr := miniredis.RunT(t)
	rdb := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	t.Cleanup(func() { rdb.Close() })
	keys, err := adtoken.ParseKeys(testKeys)
	if err != nil {
		t.Fatal(err)
	}
	e := &env{st: st, spend: billing.RedisSpend{R: rdb}, keys: keys}
	e.svc = &billing.Service{DB: st.Pool, Keys: keys, Spend: e.spend, Log: slog.New(slog.NewTextHandler(io.Discard, nil))}
	billing.SetHook(e.svc, crashHook)
	if err := st.Pool.QueryRow(context.Background(),
		"INSERT INTO advertisers (name) VALUES ('test') RETURNING advertiser_id").Scan(&e.advertiser); err != nil {
		t.Fatal(err)
	}
	return e
}

type fataler interface {
	Helper()
	Fatalf(format string, args ...any)
}

func (e *env) campaign(t fataler, budget int32) int64 {
	t.Helper()
	var id int64
	if err := e.st.Pool.QueryRow(context.Background(), `INSERT INTO campaigns
		(advertiser_id, name, status, daily_budget_cents, bid_cpc_cents) VALUES ($1, 'test', 'active', $2, 100)
		RETURNING campaign_id`, e.advertiser, budget).Scan(&id); err != nil {
		t.Fatalf("campaign: %v", err)
	}
	return id
}

// impression is a served sponsored slot and its token.
type impression struct {
	claims adtoken.Claims // IssuedAt set
	token  string
}

func (e *env) serve(t fataler, campaign int64, price int32, at time.Time) impression {
	t.Helper()
	c := adtoken.Claims{
		RequestID: uuid.Must(uuid.NewV7()), Position: 3, CampaignID: campaign,
		ItemID: 7, PriceCents: price, UserID: 11,
	}
	tok, err := e.keys.Sign(c, at)
	if err != nil {
		t.Fatalf("sign: %v", err)
	}
	c.IssuedAt = at.Truncate(time.Second)
	c.ExpiresAt = c.IssuedAt.Add(adtoken.TTL)
	return impression{claims: c, token: tok}
}

// click on an impression, received after the delay; a new event id each time.
func (im impression) click(delay time.Duration) billing.Click {
	return billing.Click{
		EventID: uuid.Must(uuid.NewV7()), ImpressionID: im.claims.ImpressionID(),
		UserID: im.claims.UserID, ItemID: im.claims.ItemID, AdToken: im.token,
		ReceivedAt: im.claims.IssuedAt.Add(delay),
	}
}

// tamper changes one character in the middle of the token (inside the payload).
func tamper(tok string) string {
	i := len(tok) / 2
	r := byte('A')
	if tok[i] == 'A' {
		r = 'B'
	}
	return tok[:i] + string(r) + tok[i+1:]
}

func (e *env) row(t fataler, campaign int64, day time.Time) (budget, spent int32) {
	t.Helper()
	err := e.st.Pool.QueryRow(context.Background(),
		"SELECT budget_cents, spent_cents FROM campaign_budget_daily WHERE campaign_id = $1 AND day = $2",
		campaign, day).Scan(&budget, &spent)
	if err != nil {
		t.Fatalf("budget row %d %s: %v", campaign, day.Format(time.DateOnly), err)
	}
	return budget, spent
}

func (e *env) count(t fataler, sql string, args ...any) int {
	t.Helper()
	var n int
	if err := e.st.Pool.QueryRow(context.Background(), sql, args...).Scan(&n); err != nil {
		t.Fatalf("%s: %v", sql, err)
	}
	return n
}

// The cases of the table in design §9.3, one by one.
func TestHandleExamples(t *testing.T) {
	e := newEnv(t)
	ctx := context.Background()
	cid := e.campaign(t, 100)
	handle := func(c billing.Click) billing.Result {
		t.Helper()
		r, err := e.svc.Handle(ctx, c)
		if err != nil {
			t.Fatal(err)
		}
		return r
	}
	expect := func(r billing.Result, o billing.Outcome, billed int32) {
		t.Helper()
		if r.Outcome != o || r.BilledCents != billed {
			t.Fatalf("got %s billed %d, want %s billed %d", r.Outcome, r.BilledCents, o, billed)
		}
	}

	// Served at 23:59:30 on day0; the click arrives after midnight and still bills day0.
	a := e.serve(t, cid, 40, day0.Add(24*time.Hour-30*time.Second))
	first := a.click(time.Minute)
	expect(handle(first), billing.Charged, 40)
	// The same event again (client retry, redelivery): nothing changes.
	expect(handle(first), billing.DuplicateEvent, 0)
	// A second click on the same impression (double click): no second charge, however the
	// client spells the impression id.
	expect(handle(a.click(2*time.Minute)), billing.DuplicateImpression, 0)
	upper := a.click(3 * time.Minute)
	upper.ImpressionID = strings.ToUpper(upper.ImpressionID)
	expect(handle(upper), billing.DuplicateImpression, 0)
	if _, spent := e.row(t, cid, day0); spent != 40 {
		t.Fatalf("spent %d after duplicates", spent)
	}

	// 60 left: a 70-cent click bills 60, the next one 0 (overdelivery; the charge row stays).
	b := e.serve(t, cid, 70, day0.Add(time.Hour))
	expect(handle(b.click(time.Second)), billing.Charged, 60)
	c := e.serve(t, cid, 25, day0.Add(2*time.Hour))
	expect(handle(c.click(time.Second)), billing.Charged, 0)
	if budget, spent := e.row(t, cid, day0); budget != 100 || spent != 100 {
		t.Fatalf("row %d / %d", spent, budget)
	}

	// Not billable: expired, for another user, forged. Each is recorded once.
	d := e.serve(t, cid, 30, day0.Add(25*time.Hour))
	late := d.click(adtoken.TTL)
	expect(handle(late), billing.Expired, 0)
	expect(handle(late), billing.DuplicateEvent, 0)
	other := d.click(time.Minute)
	other.UserID++
	expect(handle(other), billing.Mismatch, 0)
	forged := d.click(time.Minute)
	forged.AdToken = tamper(forged.AdToken)
	expect(handle(forged), billing.Invalid, 0)
	if n := e.count(t, "SELECT count(*) FROM ad_clicks WHERE campaign_id IS NULL AND event_id = $1", forged.EventID); n != 1 {
		t.Fatal("forged click not recorded without a campaign")
	}
	if n := e.count(t, "SELECT count(*) FROM campaign_budget_daily WHERE campaign_id = $1 AND day = $2", cid, day0.AddDate(0, 0, 1)); n != 0 {
		t.Fatal("a non-billable click created a budget row")
	}

	// The token's campaign is not in the database: an error, and nothing is written.
	ghost := e.serve(t, 1<<40, 10, day0.Add(time.Hour)).click(time.Second)
	if _, err := e.svc.Handle(ctx, ghost); !errors.Is(err, billing.ErrUnknownCampaign) {
		t.Fatalf("unknown campaign: %v", err)
	}
	if n := e.count(t, "SELECT count(*) FROM ad_clicks WHERE event_id = $1", ghost.EventID); n != 0 {
		t.Fatal("claim of a failed click was kept")
	}

	// Ledger, budget row and Redis agree.
	if n := e.count(t, "SELECT count(*) FROM ad_charges WHERE campaign_id = $1", cid); n != 3 {
		t.Fatalf("%d charges", n)
	}
	if got, _ := e.spend.Spent(ctx, cid, day0); got != 100 {
		t.Fatalf("redis spend %d", got)
	}
	// The database refuses to bill past the budget even without this code.
	if _, err := e.st.Pool.Exec(ctx, "UPDATE campaign_budget_daily SET spent_cents = budget_cents + 1 WHERE campaign_id = $1", cid); err == nil {
		t.Fatal("CHECK (spent_cents <= budget_cents) did not fire")
	}
}

// A crash at any step before COMMIT leaves nothing behind; the retry bills once. A crash after
// COMMIT (before the event is acknowledged) leaves the charge, the retry is a duplicate, and
// only the Redis counter lags until reconciliation.
func TestCrashAndRetry(t *testing.T) {
	e := newEnv(t)
	cid := e.campaign(t, 1000)
	for i, at := range []billing.Step{billing.StepClaimed, billing.StepLocked, billing.StepCharged, billing.StepSpent} {
		im := e.serve(t, cid, 10, day0.Add(time.Duration(i)*time.Minute))
		c := im.click(time.Second)
		ctx, plan := withPlan(at)
		if _, err := e.svc.Handle(ctx, c); err == nil || !plan.fired {
			t.Fatalf("%s: err %v, fired %v", at, err, plan.fired)
		}
		if n := e.count(t, "SELECT count(*) FROM ad_clicks WHERE event_id = $1", c.EventID); n != 0 {
			t.Fatalf("%s: the claim survived the crash", at)
		}
		if r, err := e.svc.Handle(context.Background(), c); err != nil || r.Outcome != billing.Charged {
			t.Fatalf("%s: retry %+v %v", at, r, err)
		}
	}
	im := e.serve(t, cid, 10, day0.Add(time.Hour))
	c := im.click(time.Second)
	ctx, _ := withPlan(billing.StepCommitted)
	if _, err := e.svc.Handle(ctx, c); !errors.Is(err, errCrash) {
		t.Fatalf("after commit: %v", err)
	}
	if r, err := e.svc.Handle(context.Background(), c); err != nil || r.Outcome != billing.DuplicateEvent {
		t.Fatalf("retry after commit: %+v %v", r, err)
	}
	_, spent := e.row(t, cid, day0)
	redisSpent, _ := e.spend.Spent(context.Background(), cid, day0)
	if spent != 50 || redisSpent != 40 {
		t.Fatalf("spent %d, redis %d: want 50 and 40", spent, redisSpent)
	}
}

// oomSpend is a spend counter on a full state Redis: it refuses every write.
type oomSpend struct{}

func (oomSpend) Add(context.Context, uuid.UUID, int64, time.Time, int32) error {
	return errors.New("OOM command not allowed when used memory > 'maxmemory'. script: 8b42fd55, on @user_script:5.")
}

// A full state Redis does not stop billing: the charge is in the ledger, the counter is left to
// reconciliation, and the refused write is counted (recloop_redis_errors_total, kind oom).
func TestChargeWhenTheSpendCounterIsRefused(t *testing.T) {
	e := newEnv(t)
	e.svc.Spend = oomSpend{}
	cid := e.campaign(t, 100)
	refused := metrics.RedisErrors.WithLabelValues("billing", "spend", "oom")
	before := promtest.ToFloat64(refused)
	r, err := e.svc.Handle(context.Background(), e.serve(t, cid, 40, day0).click(time.Minute))
	if err != nil || r.Outcome != billing.Charged || r.BilledCents != 40 {
		t.Fatalf("got %+v, %v", r, err)
	}
	if _, spent := e.row(t, cid, day0); spent != 40 {
		t.Fatalf("budget row spent %d", spent)
	}
	if got := promtest.ToFloat64(refused) - before; got != 1 {
		t.Fatalf("%v refused writes counted", got)
	}
}
