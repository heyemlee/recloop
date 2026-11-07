package store

import (
	"context"
	"errors"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgtype"

	"github.com/heyemlee/recloop/serving/internal/ads"
	"github.com/heyemlee/recloop/serving/internal/store/db"
)

// Errors of the campaign methods.
var (
	ErrNotFound = errors.New("not found")
	// ErrVersionMismatch: the campaign changed since the version the caller read.
	ErrVersionMismatch = errors.New("the campaign changed since that version")
)

// utcDay truncates to the UTC day (events.Day; store cannot import events).
func utcDay(t time.Time) time.Time {
	t = t.UTC()
	return time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, time.UTC)
}

func date(t time.Time) pgtype.Date { return pgtype.Date{Time: utcDay(t), Valid: true} }

func ts(t *time.Time) pgtype.Timestamptz {
	if t == nil {
		return pgtype.Timestamptz{}
	}
	return pgtype.Timestamptz{Time: *t, Valid: true}
}

func timePtr(t pgtype.Timestamptz) *time.Time {
	if !t.Valid {
		return nil
	}
	v := t.Time.UTC()
	return &v
}

// foreignKey turns a violated foreign key into a validation error.
func foreignKey(err error) error {
	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) && pgErr.Code == "23503" {
		switch pgErr.ConstraintName {
		case "campaigns_advertiser_id_fkey":
			return &ads.ValidationError{Msg: "no such advertiser"}
		case "campaign_items_item_id_fkey":
			return &ads.ValidationError{Msg: "an item is not in the catalog"}
		}
	}
	return err
}

// ---------------------------------------------------------------- advertisers

func advertiser(r db.Advertiser) ads.Advertiser {
	return ads.Advertiser{ID: r.AdvertiserID, Name: r.Name, Brand: r.Brand.String, CreatedAt: r.CreatedAt.Time.UTC()}
}

// CreateAdvertiser adds an advertiser.
func (s *Store) CreateAdvertiser(ctx context.Context, name, brand string) (ads.Advertiser, error) {
	r, err := s.Q.CreateAdvertiser(ctx, db.CreateAdvertiserParams{
		Name: name, Brand: pgtype.Text{String: brand, Valid: brand != ""},
	})
	if err != nil {
		return ads.Advertiser{}, err
	}
	return advertiser(r), nil
}

// Advertisers lists all advertisers.
func (s *Store) Advertisers(ctx context.Context) ([]ads.Advertiser, error) {
	rows, err := s.Q.ListAdvertisers(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]ads.Advertiser, len(rows))
	for i, r := range rows {
		out[i] = advertiser(r)
	}
	return out, nil
}

// ---------------------------------------------------------------- campaigns

func campaign(r db.ListCampaignsRow, today time.Time) ads.Campaign {
	c := ads.Campaign{
		ID: r.CampaignID, AdvertiserID: r.AdvertiserID,
		Settings: ads.Settings{
			Name: r.Name, Status: r.Status, DailyBudgetCents: r.DailyBudgetCents,
			BidCents: r.BidCpcCents, FreqCapPerDay: r.FreqCapPerDay,
			StartAt: timePtr(r.StartAt), EndAt: timePtr(r.EndAt), Targeting: r.Targeting,
			ItemIDs: r.ItemIds,
		},
		Version: r.Version, CreatedAt: r.CreatedAt.Time.UTC(), UpdatedAt: r.UpdatedAt.Time.UTC(),
		Today: ads.BudgetDay{
			Day: utcDay(today).Format(time.DateOnly), BudgetCents: r.DailyBudgetCents,
			SpentCents: r.TodaySpentCents,
		},
	}
	if r.TodayBudgetCents.Valid {
		c.Today.BudgetCents = r.TodayBudgetCents.Int32
	}
	return c
}

// Campaigns lists campaigns, all or one advertiser's, with today's (UTC) budget row.
func (s *Store) Campaigns(ctx context.Context, advertiserID *int32, today time.Time) ([]ads.Campaign, error) {
	p := db.ListCampaignsParams{Today: date(today)}
	if advertiserID != nil {
		p.AdvertiserID = pgtype.Int4{Int32: *advertiserID, Valid: true}
	}
	rows, err := s.Q.ListCampaigns(ctx, p)
	if err != nil {
		return nil, err
	}
	out := make([]ads.Campaign, len(rows))
	for i, r := range rows {
		out[i] = campaign(r, today)
	}
	return out, nil
}

// Campaign reads one campaign with today's budget row.
func (s *Store) Campaign(ctx context.Context, id int64, today time.Time) (ads.Campaign, error) {
	return getCampaign(ctx, s.Q, id, today)
}

func getCampaign(ctx context.Context, q *db.Queries, id int64, today time.Time) (ads.Campaign, error) {
	rows, err := q.ListCampaigns(ctx, db.ListCampaignsParams{
		Today: date(today), CampaignID: pgtype.Int8{Int64: id, Valid: true},
	})
	if err != nil {
		return ads.Campaign{}, err
	}
	if len(rows) == 0 {
		return ads.Campaign{}, ErrNotFound
	}
	return campaign(rows[0], today), nil
}

// CreateCampaign stores a new campaign and its items (validated settings).
func (s *Store) CreateCampaign(ctx context.Context, advertiserID int32, set ads.Settings, today time.Time) (ads.Campaign, error) {
	var out ads.Campaign
	err := pgx.BeginFunc(ctx, s.Pool, func(tx pgx.Tx) error {
		q := s.Q.WithTx(tx)
		r, err := q.CreateCampaign(ctx, db.CreateCampaignParams{
			AdvertiserID: advertiserID, Name: set.Name, Status: set.Status,
			DailyBudgetCents: set.DailyBudgetCents, BidCpcCents: set.BidCents,
			FreqCapPerDay: set.FreqCapPerDay, StartAt: ts(set.StartAt), EndAt: ts(set.EndAt),
			Targeting: set.Targeting,
		})
		if err != nil {
			return foreignKey(err)
		}
		if err := q.InsertCampaignItems(ctx, db.InsertCampaignItemsParams{CampaignID: r.CampaignID, ItemIds: set.ItemIDs}); err != nil {
			return foreignKey(err)
		}
		out, err = getCampaign(ctx, q, r.CampaignID, today)
		return err
	})
	return out, err
}

// UpdateCampaign applies a patch if the campaign is still at version (optimistic lock, design
// §9.5). validate checks the patched settings. A new daily budget applies to today's budget
// row in the same transaction, capped below at what was already spent today.
//
// Errors: ErrNotFound, ErrVersionMismatch (with the current campaign), ads.ErrEnded, or a
// *ads.ValidationError.
func (s *Store) UpdateCampaign(ctx context.Context, id int64, version int32, patch ads.Patch,
	validate func(*ads.Settings) error, today time.Time) (ads.Campaign, error) {
	var out ads.Campaign
	err := pgx.BeginFunc(ctx, s.Pool, func(tx pgx.Tx) error {
		q := s.Q.WithTx(tx)
		cur, err := getCampaign(ctx, q, id, today)
		if err != nil {
			return err
		}
		if cur.Version != version {
			out = cur
			return ErrVersionMismatch
		}
		next, err := patch.Apply(cur.Settings)
		if err != nil {
			return err
		}
		if err := validate(&next); err != nil {
			return err
		}
		if _, err := q.UpdateCampaign(ctx, db.UpdateCampaignParams{
			CampaignID: id, Version: version,
			Name: next.Name, Status: next.Status, DailyBudgetCents: next.DailyBudgetCents,
			BidCpcCents: next.BidCents, FreqCapPerDay: next.FreqCapPerDay,
			StartAt: ts(next.StartAt), EndAt: ts(next.EndAt), Targeting: next.Targeting,
		}); err != nil {
			if errors.Is(err, pgx.ErrNoRows) { // changed between our read and our write
				return ErrVersionMismatch
			}
			return err
		}
		if next.DailyBudgetCents != cur.DailyBudgetCents {
			if err := q.SetBudgetDay(ctx, db.SetBudgetDayParams{
				CampaignID: id, Day: date(today), BudgetCents: next.DailyBudgetCents,
			}); err != nil {
				return err
			}
		}
		if patch.ItemIDs.Set {
			if err := q.DeleteCampaignItems(ctx, id); err != nil {
				return err
			}
			if err := q.InsertCampaignItems(ctx, db.InsertCampaignItemsParams{CampaignID: id, ItemIds: next.ItemIDs}); err != nil {
				return foreignKey(err)
			}
		}
		out, err = getCampaign(ctx, q, id, today)
		return err
	})
	if errors.Is(err, ErrVersionMismatch) && out.ID == 0 {
		// Lost the race inside the UPDATE: report the version that won.
		if cur, gerr := s.Campaign(ctx, id, today); gerr == nil {
			out = cur
		}
	}
	return out, err
}

// ---------------------------------------------------------------- report (§9.5)

// ReportDay is one UTC day of a campaign. Charges, spend and clicks are counted on the day the
// impression was served (the billing day).
type ReportDay struct {
	Day            string   `json:"day"`
	BudgetCents    int32    `json:"budget_cents"` // the day's budget row; the daily budget if none
	SpentCents     int32    `json:"spent_cents"`
	Impressions    int      `json:"impressions"`
	Clicks         int      `json:"clicks"` // charged clicks: at most one per impression
	BilledCents    int64    `json:"billed_cents"`
	UnbilledCents  int64    `json:"unbilled_cents"` // auction price not billed: budget ran out (overdelivery, §9.4)
	RepeatClicks   int      `json:"repeat_clicks"`  // further clicks on an impression already charged
	ExpiredClicks  int      `json:"expired_clicks"`
	MismatchClicks int      `json:"mismatch_clicks"`
	CTR            *float64 `json:"ctr"`           // clicks / impressions; null without impressions
	AvgCPCCents    *float64 `json:"avg_cpc_cents"` // billed / clicks; null without clicks
}

// Report is a campaign's days in [From, To] and their total.
type Report struct {
	CampaignID int64       `json:"campaign_id"`
	From       string      `json:"from"`
	To         string      `json:"to"`
	Days       []ReportDay `json:"days"`
	Total      ReportDay   `json:"total"`
}

func (d *ReportDay) rates() {
	d.CTR, d.AvgCPCCents = nil, nil
	if d.Impressions > 0 {
		v := float64(d.Clicks) / float64(d.Impressions)
		d.CTR = &v
	}
	if d.Clicks > 0 {
		v := float64(d.BilledCents) / float64(d.Clicks)
		d.AvgCPCCents = &v
	}
}

// CampaignReport builds the report of the UTC days from..to (inclusive).
func (s *Store) CampaignReport(ctx context.Context, id int64, from, to time.Time) (Report, error) {
	from, to = utcDay(from), utcDay(to)
	c, err := s.Q.GetCampaign(ctx, id)
	if errors.Is(err, pgx.ErrNoRows) {
		return Report{}, ErrNotFound
	}
	if err != nil {
		return Report{}, err
	}
	rep := Report{CampaignID: id, From: from.Format(time.DateOnly), To: to.Format(time.DateOnly)}
	byDay := map[string]*ReportDay{}
	for d := from; !d.After(to); d = d.AddDate(0, 0, 1) {
		k := d.Format(time.DateOnly)
		rep.Days = append(rep.Days, ReportDay{Day: k, BudgetCents: c.DailyBudgetCents})
	}
	for i := range rep.Days {
		byDay[rep.Days[i].Day] = &rep.Days[i]
	}
	day := func(d pgtype.Date) *ReportDay {
		if r := byDay[d.Time.Format(time.DateOnly)]; r != nil {
			return r
		}
		return &ReportDay{} // outside the range (impression events of the day after): ignored
	}

	budgets, err := s.Q.ReportBudgetDays(ctx, db.ReportBudgetDaysParams{CampaignID: id, FromDay: date(from), ToDay: date(to)})
	if err != nil {
		return Report{}, err
	}
	for _, b := range budgets {
		d := day(b.Day)
		d.BudgetCents, d.SpentCents = b.BudgetCents, b.SpentCents
	}
	charges, err := s.Q.ReportChargeDays(ctx, db.ReportChargeDaysParams{CampaignID: id, FromDay: date(from), ToDay: date(to)})
	if err != nil {
		return Report{}, err
	}
	for _, ch := range charges {
		d := day(ch.Day)
		d.Clicks, d.BilledCents, d.UnbilledCents = int(ch.Clicks), ch.BilledCents, ch.PriceCents-ch.BilledCents
	}
	clicks, err := s.Q.ReportClickDays(ctx, db.ReportClickDaysParams{
		CampaignID: pgtype.Int8{Int64: id, Valid: true}, FromDay: date(from), ToDay: date(to),
	})
	if err != nil {
		return Report{}, err
	}
	for _, cl := range clicks {
		d := day(cl.Day)
		switch cl.TokenStatus {
		case "valid": // charged clicks plus repeats
			d.RepeatClicks += int(cl.Clicks)
		case "expired":
			d.ExpiredClicks += int(cl.Clicks)
		case "mismatch":
			d.MismatchClicks += int(cl.Clicks)
		}
	}
	imps, err := s.Q.ReportImpressionDays(ctx, db.ReportImpressionDaysParams{CampaignID: id, FromDay: date(from), ToDay: date(to)})
	if err != nil {
		return Report{}, err
	}
	for _, im := range imps {
		day(im.Day).Impressions = int(im.Impressions)
	}

	rep.Total = ReportDay{Day: "total"}
	for i := range rep.Days {
		d := &rep.Days[i]
		d.RepeatClicks = max(d.RepeatClicks-d.Clicks, 0)
		d.rates()
		t := &rep.Total
		t.BudgetCents += d.BudgetCents
		t.SpentCents += d.SpentCents
		t.Impressions += d.Impressions
		t.Clicks += d.Clicks
		t.BilledCents += d.BilledCents
		t.UnbilledCents += d.UnbilledCents
		t.RepeatClicks += d.RepeatClicks
		t.ExpiredClicks += d.ExpiredClicks
		t.MismatchClicks += d.MismatchClicks
	}
	rep.Total.rates()
	return rep, nil
}
