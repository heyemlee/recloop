package billing

import (
	"context"
	_ "embed"
	"fmt"
	"log/slog"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/redis/go-redis/v9"

	"github.com/heyemlee/recloop/serving/internal/metrics"
	"github.com/heyemlee/recloop/serving/internal/store/db"
	"github.com/heyemlee/recloop/serving/internal/userstate"
)

// ReconcileEvery is the reconciliation period (design §9.4: every minute).
const ReconcileEvery = time.Minute

//go:embed raise.lua
var raiseLua string

var raiseScript = redis.NewScript(raiseLua)

// Reconciler recomputes every campaign's spend from the ledger (ad_charges) and compares it
// with the campaign's budget row and with the Redis counter that ad serving reads (§9.4).
//
//   - Budget row vs ledger: written by the same transaction, read in one statement; a difference
//     is a bug, reported (metric, error log) and left for a human.
//   - Redis behind the ledger: an increment was lost (the process died between COMMIT and the
//     Redis update, or Redis lost data). Repaired from the ledger: each charge's increment is
//     applied with billing's own script and dedup marker (dedup:billing:{click event_id}), so
//     a charge counts exactly once whether billing's own update or the repair gets there first.
//     Then the counter is raised to at least the ledger's sum, for a counter that lost its value
//     but kept its markers (eviction). Neither step can push it past the ledger.
//   - Redis ahead of the ledger: cannot happen (increments follow commits, Redis is read
//     before the ledger). Reported, not corrected: correcting it would hide the bug.
type Reconciler struct {
	DB    *pgxpool.Pool
	Redis redis.UniversalClient
	Log   *slog.Logger
	Now   func() time.Time
}

// Campaign-day states of a pass.
const (
	StateMatch       = "match"
	StateRedisBehind = "redis_behind" // repaired
	StateRedisAhead  = "redis_ahead"
)

// DayState is one campaign-day of a pass.
type DayState struct {
	CampaignID  int64  `json:"campaign_id"`
	Day         string `json:"day"`
	BudgetCents int32  `json:"budget_cents"`
	SpentCents  int32  `json:"spent_cents"`  // the budget row
	LedgerCents int64  `json:"ledger_cents"` // sum of billed_cents in ad_charges
	Charges     int32  `json:"charges"`
	RedisBefore int64  `json:"redis_before"`
	RedisAfter  int64  `json:"redis_after"`
	Repaired    int64  `json:"repaired_cents,omitempty"`
	State       string `json:"state"`
	RowMismatch bool   `json:"row_mismatch,omitempty"`
}

// Report of one pass.
type Report struct {
	At            time.Time  `json:"at"`
	Days          []DayState `json:"days"`
	Match         int        `json:"match"`
	RedisBehind   int        `json:"redis_behind"`
	RedisAhead    int        `json:"redis_ahead"`
	RowMismatch   int        `json:"row_mismatch"`
	RepairedCents int64      `json:"repaired_cents"`
	LedgerToday   int64      `json:"ledger_today_cents"`
	RedisToday    int64      `json:"redis_today_cents"` // after repairs
}

// OK reports whether the pass found the three sources in agreement (after repairs).
func (r Report) OK() bool { return r.RedisAhead == 0 && r.RowMismatch == 0 }

func (r *Reconciler) now() time.Time {
	if r.Now != nil {
		return r.Now()
	}
	return time.Now()
}

// Pass reconciles today and yesterday (UTC): a day's charges keep arriving until its last
// tokens expire, 30 minutes after midnight.
func (r *Reconciler) Pass(ctx context.Context) (Report, error) {
	now := r.now().UTC()
	today := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, time.UTC)
	days := []pgtype.Date{{Time: today, Valid: true}, {Time: today.AddDate(0, 0, -1), Valid: true}}
	rep := Report{At: now}
	q := db.New(r.DB)
	rows, err := q.ReconcileDays(ctx, days)
	if err != nil {
		return rep, err
	}
	if len(rows) == 0 {
		return rep, nil
	}
	keys := make([]string, len(rows))
	for i, row := range rows {
		keys[i] = SpendKey(row.CampaignID, row.Day.Time)
	}
	vals, err := r.Redis.MGet(ctx, keys...).Result()
	if err != nil {
		return rep, err
	}
	counters := userstate.Ints(vals)
	for i, row := range rows {
		d := DayState{
			CampaignID: row.CampaignID, Day: row.Day.Time.Format(time.DateOnly), BudgetCents: row.BudgetCents,
			SpentCents: row.SpentCents, LedgerCents: row.LedgerCents, Charges: row.Charges,
			RedisBefore: counters[i], RedisAfter: counters[i], State: StateMatch,
		}
		if int64(row.SpentCents) != row.LedgerCents {
			d.RowMismatch = true
			rep.RowMismatch++
			r.Log.Error("budget row and ledger disagree", "campaign_id", d.CampaignID, "day", d.Day,
				"spent_cents", d.SpentCents, "ledger_cents", d.LedgerCents)
		}
		switch {
		case d.RedisBefore < d.LedgerCents:
			added, after, err := r.repair(ctx, row.CampaignID, row.Day)
			if err != nil {
				return rep, err
			}
			d.State, d.Repaired, d.RedisAfter = StateRedisBehind, added, after
			rep.RedisBehind++
			rep.RepairedCents += added
		case d.RedisBefore > d.LedgerCents:
			// Charges committed after the ledger read may be counted already: read it again.
			ledger, _, err := r.charges(ctx, row.CampaignID, row.Day)
			if err != nil {
				return rep, err
			}
			if d.RedisBefore > ledger {
				d.State = StateRedisAhead
				rep.RedisAhead++
				r.Log.Error("redis spend is ahead of the ledger", "campaign_id", d.CampaignID, "day", d.Day,
					"redis_cents", d.RedisBefore, "ledger_cents", ledger)
			} else {
				rep.Match++
			}
		default:
			rep.Match++
		}
		if row.Day.Time.Equal(today) {
			rep.LedgerToday += d.LedgerCents
			rep.RedisToday += d.RedisAfter
		}
		rep.Days = append(rep.Days, d)
	}
	return rep, nil
}

// charges returns the ledger sum of a campaign-day and its money-moving charges.
func (r *Reconciler) charges(ctx context.Context, campaignID int64, day pgtype.Date) (int64, []db.ChargesOfDayRow, error) {
	rows, err := db.New(r.DB).ChargesOfDay(ctx, db.ChargesOfDayParams{CampaignID: campaignID, Day: day})
	if err != nil {
		return 0, nil, err
	}
	var sum int64
	for _, c := range rows {
		sum += int64(c.BilledCents)
	}
	return sum, rows, nil
}

// repair brings a counter that is behind up to the ledger; it returns the cents it added and
// the counter's value after.
func (r *Reconciler) repair(ctx context.Context, campaignID int64, day pgtype.Date) (int64, int64, error) {
	sum, rows, err := r.charges(ctx, campaignID, day)
	if err != nil {
		return 0, 0, err
	}
	key := SpendKey(campaignID, day.Time)
	if err := spendScript.Load(ctx, r.Redis).Err(); err != nil {
		return 0, 0, err
	}
	var added int64
	const chunk = 500
	for start := 0; start < len(rows); start += chunk {
		part := rows[start:min(start+chunk, len(rows))]
		pipe := r.Redis.Pipeline()
		cmds := make([]*redis.Cmd, len(part))
		for i, c := range part {
			cmds[i] = spendScript.EvalSha(ctx, pipe, []string{
				userstate.DedupKey(DedupGroup, uuid.UUID(c.ClickEventID.Bytes).String()), key,
			}, c.BilledCents, userstate.DedupTTL.Milliseconds(), SpendTTL.Milliseconds())
		}
		if _, err := pipe.Exec(ctx); err != nil {
			return added, 0, err
		}
		for i, cmd := range cmds {
			if n, _ := cmd.Int(); n == 1 {
				added += int64(part[i].BilledCents)
			}
		}
	}
	raised, err := raiseScript.Run(ctx, r.Redis, []string{key}, sum, SpendTTL.Milliseconds()).Int64()
	if err != nil {
		return added, 0, err
	}
	after, err := RedisSpend{R: r.Redis}.Spent(ctx, campaignID, day.Time)
	if err != nil {
		return added + raised, 0, err
	}
	return added + raised, after, nil
}

// Run reconciles at once and then every `every` until ctx is done.
func (r *Reconciler) Run(ctx context.Context, every time.Duration) {
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		rep, err := r.Pass(ctx)
		switch {
		case err != nil && ctx.Err() == nil:
			metrics.ReconcileRuns.WithLabelValues("error").Inc()
			r.Log.Warn("reconciliation failed", "err", err)
		case err == nil:
			metrics.ReconcileRuns.WithLabelValues("ok").Inc()
			metrics.ReconcileLastSuccess.Set(float64(rep.At.Unix()))
			metrics.ReconcileDays.WithLabelValues(StateMatch).Set(float64(rep.Match))
			metrics.ReconcileDays.WithLabelValues(StateRedisBehind).Set(float64(rep.RedisBehind))
			metrics.ReconcileDays.WithLabelValues(StateRedisAhead).Set(float64(rep.RedisAhead))
			metrics.ReconcileDays.WithLabelValues("row_mismatch").Set(float64(rep.RowMismatch))
			metrics.ReconcileRepairedCents.Add(float64(rep.RepairedCents))
			metrics.SpendCents.WithLabelValues("ledger").Set(float64(rep.LedgerToday))
			metrics.SpendCents.WithLabelValues("redis").Set(float64(rep.RedisToday))
			if rep.RedisBehind > 0 || !rep.OK() {
				r.Log.Info("reconciled", "campaign_days", len(rep.Days), "repaired", rep.RedisBehind,
					"repaired_cents", rep.RepairedCents, "redis_ahead", rep.RedisAhead, "row_mismatch", rep.RowMismatch)
			}
		}
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}

// String summarises a report in one line.
func (r Report) String() string {
	return fmt.Sprintf("%d campaign-days: %d match, %d repaired (+%d¢), %d redis ahead, %d row mismatches",
		len(r.Days), r.Match, r.RedisBehind, r.RepairedCents, r.RedisAhead, r.RowMismatch)
}
