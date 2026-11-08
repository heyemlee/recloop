// Package billing charges sponsored clicks (docs/design.md §9.3, §9.4). Two invariants hold
// whatever the delivery order, duplication or crashes: no campaign is billed past its daily
// budget, and an impression is billed at most once.
//
// Handle verifies the click's ad_token, then runs one PostgreSQL transaction:
//
//  1. Claim the event_id in ad_clicks (INSERT ... ON CONFLICT DO NOTHING). Nothing inserted:
//     the event was processed before (a redelivery or a retry), stop. A concurrent copy of the
//     event waits on the unique index until the first one commits or aborts.
//  2. Create the campaign's budget row for the day the impression was served if needed, and
//     lock it (SELECT ... FOR UPDATE): the charges of one campaign-day are serialised.
//  3. billed = min(token price, budget - spent). It is 0 once the budget is spent: the click
//     still gets its charge row, and the platform absorbs the overdelivery (§9.4).
//  4. Insert the charge (INSERT ... ON CONFLICT (impression_id) DO NOTHING). Nothing inserted:
//     the impression was charged already (a repeat click), stop; the claim stays.
//  5. spent += billed.
//
// A crash anywhere before COMMIT rolls everything back, the claim included, so the redelivered
// click is processed from scratch; after COMMIT a redelivery stops at step 1. The CHECK
// (spent_cents <= budget_cents) on the budget row backs step 3 even if this code had a bug.
//
// After the commit the charge is added to the Redis spend counter that ad serving reads, at
// most once per event. That step is lost if the process dies between the two; reconciliation
// recomputes the counters from the ledger.
package billing

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/heyemlee/recloop/serving/internal/adtoken"
	"github.com/heyemlee/recloop/serving/internal/store/db"
	"github.com/heyemlee/recloop/serving/internal/userstate"
)

// Click is a click event on a sponsored slot.
type Click struct {
	EventID      uuid.UUID
	ImpressionID string // {request_id}:{position}, as the client sent it
	UserID       int64
	ItemID       int32
	AdToken      string
	// ReceivedAt is when the api accepted the event, stamped by the api (the record timestamp
	// once events go through Redpanda). Token expiry is judged against it, not against the
	// time billing processes the event and not against the client-made UUIDv7 time.
	ReceivedAt time.Time
}

// Outcome of one Handle call.
type Outcome string

const (
	Charged             Outcome = "charged"              // a new charge (BilledCents may be below the price, or 0)
	DuplicateImpression Outcome = "duplicate_impression" // the impression was charged already: a repeat click
	DuplicateEvent      Outcome = "duplicate_event"      // this event_id was processed before
	Expired             Outcome = "expired"              // authentic token, too late: not billed
	Mismatch            Outcome = "mismatch"             // authentic token for another impression, user or item
	Invalid             Outcome = "invalid"              // the token did not verify
)

// Token status values of ad_clicks.token_status.
const (
	tokenValid    = "valid"
	tokenExpired  = "expired"
	tokenMismatch = "mismatch"
	tokenInvalid  = "invalid"
)

var statusOutcome = map[string]Outcome{tokenExpired: Expired, tokenMismatch: Mismatch, tokenInvalid: Invalid}

// Result of one Handle call. CampaignID, Day and PriceCents are those of the token (zero for
// Invalid); BilledCents is set for Charged.
type Result struct {
	Outcome     Outcome
	CampaignID  int64
	Day         time.Time
	PriceCents  int32
	BilledCents int32
}

// ErrUnknownCampaign: an authentic token names a campaign this database does not have. It is
// permanent; a consumer should log the event and move on rather than retry it.
var ErrUnknownCampaign = errors.New("billing: the token's campaign does not exist")

// Step is a point of Handle where tests inject crashes.
type Step string

const (
	StepClaimed   Step = "claimed"   // after the claim
	StepLocked    Step = "locked"    // after the budget row lock
	StepCharged   Step = "charged"   // after the charge insert
	StepSpent     Step = "spent"     // after the spend update, before COMMIT
	StepCommitted Step = "committed" // after COMMIT, before the Redis update
)

// Service charges clicks.
type Service struct {
	DB    *pgxpool.Pool
	Keys  *adtoken.Keys
	Spend SpendCounter // optional
	Log   *slog.Logger

	hook func(ctx context.Context, step Step, tx pgx.Tx) error // crash injection (tests)
}

func (s *Service) step(ctx context.Context, st Step, tx pgx.Tx) error {
	if s.hook == nil {
		return nil
	}
	return s.hook(ctx, st, tx)
}

// verify classifies the click's token: invalid, then mismatch, then expired, then valid.
func (s *Service) verify(c Click) (adtoken.Claims, string) {
	claims, err := s.Keys.Verify(c.AdToken, c.ReceivedAt)
	switch {
	case err != nil && !errors.Is(err, adtoken.ErrExpired):
		return adtoken.Claims{}, tokenInvalid
	case !claims.Matches(c.ImpressionID, c.UserID, c.ItemID):
		return claims, tokenMismatch
	case err != nil:
		return claims, tokenExpired
	}
	return claims, tokenValid
}

// Handle processes one delivery of a click. It returns an error only when the click could not
// be processed (the database failed, or ErrUnknownCampaign); the caller retries those with the
// same event, and only after Handle returned without an error may it commit the consumer
// offset.
func (s *Service) Handle(ctx context.Context, c Click) (Result, error) {
	if c.ReceivedAt.IsZero() {
		return Result{}, errors.New("billing: click without a receive time")
	}
	claims, status := s.verify(c)
	claim := db.ClaimClickParams{
		EventID: pgtype.UUID{Bytes: c.EventID, Valid: true}, ImpressionID: c.ImpressionID,
		UserID: c.UserID, ItemID: c.ItemID, TokenStatus: status,
		ReceivedAt: pgtype.Timestamptz{Time: c.ReceivedAt, Valid: true},
	}
	var res Result
	if status != tokenInvalid {
		claim.CampaignID = pgtype.Int8{Int64: claims.CampaignID, Valid: true}
		claim.Day = pgtype.Date{Time: claims.Day(), Valid: true}
		res = Result{CampaignID: claims.CampaignID, Day: claims.Day(), PriceCents: claims.PriceCents}
	}

	if status != tokenValid { // not billable: record the click once
		_, err := db.New(s.DB).ClaimClick(ctx, claim)
		switch {
		case errors.Is(err, pgx.ErrNoRows):
			res.Outcome = DuplicateEvent
		case err != nil:
			return Result{}, err
		default:
			res.Outcome = statusOutcome[status]
		}
		return res, nil
	}

	day := pgtype.Date{Time: claims.Day(), Valid: true}
	err := pgx.BeginFunc(ctx, s.DB, func(tx pgx.Tx) error {
		q := db.New(tx)
		// 1. Technical dedup.
		if _, err := q.ClaimClick(ctx, claim); err != nil {
			if errors.Is(err, pgx.ErrNoRows) {
				res.Outcome = DuplicateEvent
				return nil
			}
			return err
		}
		if err := s.step(ctx, StepClaimed, tx); err != nil {
			return err
		}
		// 2. The campaign-day's budget row, locked.
		if err := q.EnsureBudgetDay(ctx, db.EnsureBudgetDayParams{CampaignID: claims.CampaignID, Day: day}); err != nil {
			return err
		}
		b, err := q.LockBudgetDay(ctx, db.LockBudgetDayParams{CampaignID: claims.CampaignID, Day: day})
		if errors.Is(err, pgx.ErrNoRows) {
			return fmt.Errorf("%w: campaign %d", ErrUnknownCampaign, claims.CampaignID)
		}
		if err != nil {
			return err
		}
		if err := s.step(ctx, StepLocked, tx); err != nil {
			return err
		}
		// 3. Never past the budget.
		billed := min(claims.PriceCents, b.BudgetCents-b.SpentCents)
		// 4. Business dedup.
		if _, err := q.InsertCharge(ctx, db.InsertChargeParams{
			ImpressionID: claims.ImpressionID(), ClickEventID: claim.EventID,
			CampaignID: claims.CampaignID, Day: day, PriceCents: claims.PriceCents, BilledCents: billed,
		}); err != nil {
			if errors.Is(err, pgx.ErrNoRows) {
				res.Outcome = DuplicateImpression
				return nil
			}
			return err
		}
		if err := s.step(ctx, StepCharged, tx); err != nil {
			return err
		}
		// 5. Spend.
		if billed > 0 {
			if err := q.AddSpend(ctx, db.AddSpendParams{CampaignID: claims.CampaignID, Day: day, BilledCents: billed}); err != nil {
				return err
			}
		}
		if err := s.step(ctx, StepSpent, tx); err != nil {
			return err
		}
		res.Outcome, res.BilledCents = Charged, billed
		return nil
	})
	if err != nil {
		return Result{}, err
	}
	if err := s.step(ctx, StepCommitted, nil); err != nil {
		return Result{}, err
	}
	if res.Outcome == Charged && res.BilledCents > 0 && s.Spend != nil {
		if err := s.Spend.Add(ctx, c.EventID, res.CampaignID, res.Day, res.BilledCents); err != nil {
			userstate.CountError("billing", "spend", err)
			if s.Log != nil {
				s.Log.Warn("spend counter not updated; reconciliation repairs it",
					"event_id", c.EventID, "campaign_id", res.CampaignID, "cents", res.BilledCents, "err", err)
			}
		}
	}
	return res, nil
}
