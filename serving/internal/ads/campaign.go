package ads

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"
)

// Campaign statuses. Ended is terminal: the row stays because the ledger refers to it.
const (
	StatusActive = "active"
	StatusPaused = "paused"
	StatusEnded  = "ended"
)

// Limits on what an advertiser can set. ReserveCents is also the auction's reserve price, so
// every campaign that can be saved can take part.
const (
	ReserveCents   = 5
	MaxBidCents    = 10_000     // $100 per click
	MaxBudgetCents = 10_000_000 // $100,000 per day
	MaxFreqCap     = 100
	MaxItems       = 50
	MaxNameLen     = 200
)

// ErrEnded: an ended campaign cannot change any more.
var ErrEnded = errors.New("the campaign has ended and cannot be changed")

// ValidationError is an advertiser input the api rejects with 400.
type ValidationError struct{ Msg string }

func (e *ValidationError) Error() string { return e.Msg }

func invalid(format string, args ...any) error {
	return &ValidationError{Msg: fmt.Sprintf(format, args...)}
}

// Advertiser owns campaigns.
type Advertiser struct {
	ID        int32     `json:"advertiser_id"`
	Name      string    `json:"name"`
	Brand     string    `json:"brand,omitempty"`
	CreatedAt time.Time `json:"created_at"`
}

// Settings are the fields an advertiser controls.
type Settings struct {
	Name             string          `json:"name"`
	Status           string          `json:"status"`
	DailyBudgetCents int32           `json:"daily_budget_cents"`
	BidCents         int32           `json:"bid_cpc_cents"`
	FreqCapPerDay    int32           `json:"freq_cap_per_day"`
	StartAt          *time.Time      `json:"start_at"`
	EndAt            *time.Time      `json:"end_at"`
	Targeting        json.RawMessage `json:"targeting"`
	ItemIDs          []int32         `json:"item_ids"`
}

// BudgetDay is a campaign's budget row for one UTC day.
type BudgetDay struct {
	Day         string `json:"day"` // YYYY-MM-DD
	BudgetCents int32  `json:"budget_cents"`
	SpentCents  int32  `json:"spent_cents"`
}

// Campaign is a stored campaign with today's budget row (the daily budget and 0 spent when
// nothing happened today yet).
type Campaign struct {
	ID           int64 `json:"campaign_id"`
	AdvertiserID int32 `json:"advertiser_id"`
	Settings
	Version   int32     `json:"version"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
	Today     BudgetDay `json:"today"`
}

// Validate checks the settings of a new or changed campaign; knownItem says whether an item
// is in the catalog. It sorts and de-duplicates ItemIDs.
func (s *Settings) Validate(knownItem func(int32) bool) error {
	s.Name = strings.TrimSpace(s.Name)
	switch {
	case s.Name == "" || len(s.Name) > MaxNameLen:
		return invalid("name must be 1 to %d characters", MaxNameLen)
	case s.Status != StatusActive && s.Status != StatusPaused && s.Status != StatusEnded:
		return invalid("status must be active, paused or ended")
	case s.DailyBudgetCents < 1 || s.DailyBudgetCents > MaxBudgetCents:
		return invalid("daily_budget_cents must be between 1 and %d", MaxBudgetCents)
	case s.BidCents < ReserveCents || s.BidCents > MaxBidCents:
		return invalid("bid_cpc_cents must be between %d (the reserve price) and %d", ReserveCents, MaxBidCents)
	case s.FreqCapPerDay < 1 || s.FreqCapPerDay > MaxFreqCap:
		return invalid("freq_cap_per_day must be between 1 and %d", MaxFreqCap)
	case s.StartAt != nil && s.EndAt != nil && !s.StartAt.Before(*s.EndAt):
		return invalid("start_at must be before end_at")
	}
	if len(s.Targeting) > 0 {
		t := bytes.TrimSpace(s.Targeting)
		switch {
		case len(t) == 0 || bytes.Equal(t, []byte("null")):
			s.Targeting = nil
		case len(t) > 2048 || t[0] != '{' || !json.Valid(t):
			return invalid("targeting must be a JSON object of at most 2 KB")
		}
	}
	slices.Sort(s.ItemIDs)
	s.ItemIDs = slices.Compact(s.ItemIDs)
	if len(s.ItemIDs) == 0 || len(s.ItemIDs) > MaxItems {
		return invalid("item_ids must list 1 to %d items", MaxItems)
	}
	for _, id := range s.ItemIDs {
		if !knownItem(id) {
			return invalid("item %d is not in the catalog", id)
		}
	}
	return nil
}

// Opt is a PATCH field: absent (Set false), null (Set and Null) or a value.
type Opt[T any] struct {
	Set, Null bool
	V         T
}

// UnmarshalJSON records that the field was present.
func (o *Opt[T]) UnmarshalJSON(b []byte) error {
	o.Set = true
	if bytes.Equal(bytes.TrimSpace(b), []byte("null")) {
		o.Null = true
		return nil
	}
	return json.Unmarshal(b, &o.V)
}

// Patch is the body of PATCH /v1/campaigns/{id}: only the fields present change.
type Patch struct {
	Name             Opt[string]          `json:"name"`
	Status           Opt[string]          `json:"status"`
	DailyBudgetCents Opt[int32]           `json:"daily_budget_cents"`
	BidCents         Opt[int32]           `json:"bid_cpc_cents"`
	FreqCapPerDay    Opt[int32]           `json:"freq_cap_per_day"`
	StartAt          Opt[time.Time]       `json:"start_at"`
	EndAt            Opt[time.Time]       `json:"end_at"`
	Targeting        Opt[json.RawMessage] `json:"targeting"`
	ItemIDs          Opt[[]int32]         `json:"item_ids"`
}

func required[T any](name string, o Opt[T], cur T) (T, error) {
	if !o.Set {
		return cur, nil
	}
	if o.Null {
		return cur, invalid("%s cannot be null", name)
	}
	return o.V, nil
}

func nullable[T any](o Opt[T], cur *T) *T {
	switch {
	case !o.Set:
		return cur
	case o.Null:
		return nil
	}
	v := o.V
	return &v
}

// Apply returns the settings after the patch (not yet validated). An ended campaign cannot be
// patched; any other campaign can move between active and paused, or end.
func (p Patch) Apply(cur Settings) (Settings, error) {
	if cur.Status == StatusEnded {
		return cur, ErrEnded
	}
	next := cur
	var err error
	if next.Name, err = required("name", p.Name, cur.Name); err != nil {
		return cur, err
	}
	if next.Status, err = required("status", p.Status, cur.Status); err != nil {
		return cur, err
	}
	if next.DailyBudgetCents, err = required("daily_budget_cents", p.DailyBudgetCents, cur.DailyBudgetCents); err != nil {
		return cur, err
	}
	if next.BidCents, err = required("bid_cpc_cents", p.BidCents, cur.BidCents); err != nil {
		return cur, err
	}
	if next.FreqCapPerDay, err = required("freq_cap_per_day", p.FreqCapPerDay, cur.FreqCapPerDay); err != nil {
		return cur, err
	}
	if next.ItemIDs, err = required("item_ids", p.ItemIDs, cur.ItemIDs); err != nil {
		return cur, err
	}
	next.StartAt = nullable(p.StartAt, cur.StartAt)
	next.EndAt = nullable(p.EndAt, cur.EndAt)
	if p.Targeting.Set {
		next.Targeting = p.Targeting.V // nil when null
	}
	return next, nil
}
