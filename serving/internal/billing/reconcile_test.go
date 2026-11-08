package billing_test

import (
	"context"
	"io"
	"log/slog"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
	"github.com/twmb/franz-go/pkg/kgo"

	"github.com/heyemlee/recloop/serving/internal/billing"
	"github.com/heyemlee/recloop/serving/internal/events"
	"github.com/heyemlee/recloop/serving/internal/store"
)

// reconciler on the env's database and a Redis of its own, at noon of day0.
func (e *env) reconciler(t *testing.T) (*billing.Reconciler, *miniredis.Miniredis) {
	t.Helper()
	mr := miniredis.RunT(t)
	rdb := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	t.Cleanup(func() { rdb.Close() })
	e.spend = billing.RedisSpend{R: rdb}
	e.svc.Spend = e.spend
	return &billing.Reconciler{DB: e.st.Pool, Redis: rdb, Log: slog.New(slog.NewTextHandler(io.Discard, nil)),
		Now: func() time.Time { return day0.Add(12 * time.Hour) }}, mr
}

func (e *env) charge(t *testing.T, cid int64, price int32, at time.Time) billing.Click {
	t.Helper()
	c := e.serve(t, cid, price, at).click(time.Second)
	if r, err := e.svc.Handle(context.Background(), c); err != nil || r.Outcome != billing.Charged {
		t.Fatalf("charge: %+v %v", r, err)
	}
	return c
}

func pass(t *testing.T, r *billing.Reconciler) billing.Report {
	t.Helper()
	rep, err := r.Pass(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	return rep
}

// A Redis update lost between COMMIT and the counter (crash): the pass adds exactly the lost
// charge, with the charge's own dedup marker, so billing's redelivered update cannot add it
// again; the next pass finds everything in agreement.
func TestReconcileRepairsALostIncrement(t *testing.T) {
	e := newEnv(t)
	rec, _ := e.reconciler(t)
	cid := e.campaign(t, 1000)
	e.charge(t, cid, 10, day0.Add(time.Hour))
	e.charge(t, cid, 15, day0.Add(2*time.Hour))
	lost := e.serve(t, cid, 20, day0.Add(3*time.Hour)).click(time.Second)
	ctx, _ := withPlan(billing.StepCommitted)
	if _, err := e.svc.Handle(ctx, lost); err == nil {
		t.Fatal("want the crash after COMMIT")
	}
	if got, _ := e.spend.Spent(context.Background(), cid, day0); got != 25 {
		t.Fatalf("redis %d before the pass", got)
	}
	rep := pass(t, rec)
	if rep.RedisBehind != 1 || rep.RepairedCents != 20 || !rep.OK() || len(rep.Days) != 1 {
		t.Fatalf("report %+v", rep)
	}
	d := rep.Days[0]
	if d.LedgerCents != 45 || d.SpentCents != 45 || d.RedisBefore != 25 || d.RedisAfter != 45 || d.Charges != 3 {
		t.Fatalf("day %+v", d)
	}
	// Billing's own update, late (e.g. the redelivered event after a restart): no effect.
	if err := e.spend.Add(context.Background(), lost.EventID, cid, day0, 20); err != nil {
		t.Fatal(err)
	}
	if got, _ := e.spend.Spent(context.Background(), cid, day0); got != 45 {
		t.Fatalf("redis %d after a late update", got)
	}
	if rep := pass(t, rec); rep.Match != 1 || rep.RedisBehind != 0 || rep.RepairedCents != 0 || rep.LedgerToday != 45 || rep.RedisToday != 45 {
		t.Fatalf("second pass %+v", rep)
	}
}

// Redis restarted empty (no persistence): counters and markers are gone; the pass rebuilds
// today's and yesterday's counters from the ledger, and only those days.
func TestReconcileRebuildsAFlushedRedis(t *testing.T) {
	e := newEnv(t)
	rec, mr := e.reconciler(t)
	a, b := e.campaign(t, 1000), e.campaign(t, 30)
	e.charge(t, a, 10, day0.Add(time.Hour))
	e.charge(t, a, 12, day0.Add(-time.Hour)) // yesterday
	e.charge(t, b, 25, day0.Add(time.Hour))
	e.charge(t, b, 25, day0.Add(2*time.Hour))  // 5 of 25 billed: the budget ran out
	e.charge(t, b, 25, day0.Add(3*time.Hour))  // 0 billed: no marker, nothing to add
	e.charge(t, a, 7, day0.Add(-49*time.Hour)) // three days ago: outside the window
	mr.FlushAll()
	rep := pass(t, rec)
	if rep.RedisBehind != 3 || rep.RepairedCents != 10+12+30 || !rep.OK() || len(rep.Days) != 3 {
		t.Fatalf("report %+v", rep)
	}
	for _, c := range []struct {
		cid  int64
		day  time.Time
		want int64
	}{{a, day0, 10}, {a, day0.AddDate(0, 0, -1), 12}, {b, day0, 30}, {a, day0.AddDate(0, 0, -3), 0}} {
		if got, _ := e.spend.Spent(context.Background(), c.cid, c.day); got != c.want {
			t.Errorf("campaign %d %s: %d, want %d", c.cid, c.day.Format(time.DateOnly), got, c.want)
		}
	}
	if rep := pass(t, rec); rep.Match != 3 || rep.RepairedCents != 0 {
		t.Fatalf("second pass %+v", rep)
	}
}

// A counter evicted with its markers kept (volatile-ttl evicts the 2-day counters before the
// 7-day markers): re-applying finds every marker, so the pass raises the counter to the ledger.
func TestReconcileRaisesAnEvictedCounter(t *testing.T) {
	e := newEnv(t)
	rec, mr := e.reconciler(t)
	cid := e.campaign(t, 1000)
	e.charge(t, cid, 10, day0.Add(time.Hour))
	e.charge(t, cid, 11, day0.Add(2*time.Hour))
	mr.Del(billing.SpendKey(cid, day0))
	rep := pass(t, rec)
	if rep.RedisBehind != 1 || rep.RepairedCents != 21 || rep.Days[0].RedisAfter != 21 {
		t.Fatalf("report %+v", rep)
	}
	if ttl := mr.TTL(billing.SpendKey(cid, day0)); ttl <= 0 || ttl > billing.SpendTTL {
		t.Fatalf("ttl %v", ttl)
	}
}

// A counter ahead of the ledger means a bug: reported, left as it is.
func TestReconcileReportsRedisAhead(t *testing.T) {
	e := newEnv(t)
	rec, mr := e.reconciler(t)
	cid := e.campaign(t, 1000)
	e.charge(t, cid, 10, day0.Add(time.Hour))
	if _, err := mr.Incr(billing.SpendKey(cid, day0), 7); err != nil {
		t.Fatal(err)
	}
	rep := pass(t, rec)
	if rep.RedisAhead != 1 || rep.OK() || rep.Days[0].State != billing.StateRedisAhead {
		t.Fatalf("report %+v", rep)
	}
	if got, _ := e.spend.Spent(context.Background(), cid, day0); got != 17 {
		t.Fatalf("counter changed to %d", got)
	}
}

// The budget row and the ledger disagreeing is reported (a bug, or someone edited the table).
func TestReconcileReportsARowMismatch(t *testing.T) {
	e := newEnv(t)
	rec, _ := e.reconciler(t)
	cid := e.campaign(t, 1000)
	e.charge(t, cid, 10, day0.Add(time.Hour))
	if _, err := e.st.Pool.Exec(context.Background(), "UPDATE campaign_budget_daily SET spent_cents = 9 WHERE campaign_id = $1", cid); err != nil {
		t.Fatal(err)
	}
	if rep := pass(t, rec); rep.RowMismatch != 1 || rep.OK() || !rep.Days[0].RowMismatch {
		t.Fatalf("report %+v", rep)
	}
}

// record is the event-log record of an event, as the api writes it.
func record(t *testing.T, ev events.Event) *kgo.Record {
	t.Helper()
	r, err := events.KafkaRecord(events.Topic, ev)
	if err != nil {
		t.Fatal(err)
	}
	r.Timestamp = ev.Received
	return r
}

func clickEvent(c billing.Click) events.Event {
	rid, _, _ := events.ParseImpressionID(c.ImpressionID)
	at := events.V7Time(c.EventID)
	return events.Event{
		ID: c.EventID, Day: events.Day(at), Time: at, Type: events.TypeClick, UserID: c.UserID, ItemID: c.ItemID,
		RequestID: uuid.NullUUID{UUID: rid, Valid: true}, ImpressionID: c.ImpressionID, AdToken: c.AdToken,
		Received: c.ReceivedAt,
	}
}

// The consumer charges the sponsored clicks of a batch and skips everything else; the receive
// time comes from the record (the api's server_ts), so a click redelivered long after it was
// received is judged as it was then.
func TestConsumerChargesTheSponsoredClicksOfABatch(t *testing.T) {
	e := newEnv(t)
	cid := e.campaign(t, 1000)
	cons := &billing.Consumer{Service: e.svc, Workers: 4, Log: slog.New(slog.NewTextHandler(io.Discard, nil))}
	im := e.serve(t, cid, 30, day0.Add(time.Hour))
	first, double := im.click(time.Second), im.click(2*time.Second)
	late := e.serve(t, cid, 30, day0.Add(2*time.Hour)).click(31 * time.Minute)
	ghost := e.serve(t, 1<<40, 10, day0.Add(time.Hour)).click(time.Second)
	organic := clickEvent(first)
	organic.ID, organic.AdToken, organic.ImpressionID = uuid.Must(uuid.NewV7()), "", ""
	imp := clickEvent(first)
	imp.ID, imp.Type = uuid.Must(uuid.NewV7()), events.TypeImpression
	served := events.Event{ID: uuid.Must(uuid.NewV7()), Type: events.TypeServed, UserID: 11, Received: day0,
		RequestID: uuid.NullUUID{UUID: im.claims.RequestID, Valid: true}, Payload: []byte(`{}`)}
	served.Time = events.V7Time(served.ID)
	served.Day = events.Day(served.Time)
	broken := record(t, clickEvent(first))
	broken.Value = []byte("{not json")
	recs := []*kgo.Record{
		record(t, served), record(t, imp), record(t, organic), broken,
		record(t, clickEvent(first)), record(t, clickEvent(first)), record(t, clickEvent(double)),
		record(t, clickEvent(late)), record(t, clickEvent(ghost)),
	}
	n, err := cons.HandleRecords(context.Background(), recs)
	if err != nil {
		t.Fatal(err)
	}
	want := map[billing.Outcome]int{billing.Charged: 1, billing.DuplicateEvent: 1, billing.DuplicateImpression: 1, billing.Expired: 1}
	if n.Clicks != 5 || n.Ignored != 3 || n.Invalid != 1 || n.UnknownCampaign != 1 || len(n.Outcomes) != len(want) {
		t.Fatalf("counts %+v", n)
	}
	for o, k := range want {
		if n.Outcomes[o] != k {
			t.Fatalf("outcome %s: %d, want %d (%+v)", o, n.Outcomes[o], k, n.Outcomes)
		}
	}
	if _, spent := e.row(t, cid, day0); spent != 30 {
		t.Fatalf("spent %d", spent)
	}
	// The whole batch again (a crash before the offset commit): nothing changes.
	n, err = cons.HandleRecords(context.Background(), recs)
	if err != nil || n.Outcomes[billing.DuplicateEvent] != 4 {
		t.Fatalf("redelivery %+v %v", n, err)
	}
	if _, spent := e.row(t, cid, day0); spent != 30 {
		t.Fatalf("spent %d after a redelivery", spent)
	}
}

// A database failure fails the batch, so the loop retries it and does not commit its offsets.
func TestConsumerFailsTheBatchWhenTheDatabaseFails(t *testing.T) {
	e := newEnv(t)
	cid := e.campaign(t, 1000)
	c := e.serve(t, cid, 30, day0.Add(time.Hour)).click(time.Second)
	closed, err := store.Open(context.Background(), e.st.Pool.Config().ConnString())
	if err != nil {
		t.Fatal(err)
	}
	closed.Close()
	svc := &billing.Service{DB: closed.Pool, Keys: e.keys, Log: slog.New(slog.NewTextHandler(io.Discard, nil))}
	cons := &billing.Consumer{Service: svc, Workers: 2, Log: svc.Log}
	if _, err := cons.HandleRecords(context.Background(), []*kgo.Record{record(t, clickEvent(c))}); err == nil {
		t.Fatal("want the database error")
	}
	if n := e.count(t, "SELECT count(*) FROM ad_clicks WHERE event_id = $1", c.EventID); n != 0 {
		t.Fatal("a failed click left a trace")
	}
}
