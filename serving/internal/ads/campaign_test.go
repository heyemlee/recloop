package ads_test

import (
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/heyemlee/recloop/serving/internal/ads"
)

func known(id int32) bool { return id >= 0 && id < 100 }

func valid() ads.Settings {
	return ads.Settings{Name: " Pens ", Status: ads.StatusActive, DailyBudgetCents: 2000, BidCents: 40,
		FreqCapPerDay: 3, ItemIDs: []int32{9, 3, 9}}
}

func TestValidate(t *testing.T) {
	s := valid()
	if err := s.Validate(known); err != nil {
		t.Fatal(err)
	}
	if s.Name != "Pens" || len(s.ItemIDs) != 2 || s.ItemIDs[0] != 3 {
		t.Fatalf("normalised %+v", s)
	}
	for _, empty := range []string{" ", "null"} {
		s := valid()
		s.Targeting = json.RawMessage(empty)
		if err := s.Validate(known); err != nil || s.Targeting != nil {
			t.Fatalf("targeting %q: %v %s", empty, err, s.Targeting)
		}
	}
	start := time.Date(2026, 11, 6, 0, 0, 0, 0, time.UTC)
	end := start.Add(-time.Hour)
	for name, edit := range map[string]func(*ads.Settings){
		"empty name":     func(s *ads.Settings) { s.Name = "  " },
		"status":         func(s *ads.Settings) { s.Status = "deleted" },
		"zero budget":    func(s *ads.Settings) { s.DailyBudgetCents = 0 },
		"huge budget":    func(s *ads.Settings) { s.DailyBudgetCents = ads.MaxBudgetCents + 1 },
		"below reserve":  func(s *ads.Settings) { s.BidCents = ads.ReserveCents - 1 },
		"huge bid":       func(s *ads.Settings) { s.BidCents = ads.MaxBidCents + 1 },
		"freq cap":       func(s *ads.Settings) { s.FreqCapPerDay = 0 },
		"no items":       func(s *ads.Settings) { s.ItemIDs = nil },
		"unknown item":   func(s *ads.Settings) { s.ItemIDs = []int32{100} },
		"window":         func(s *ads.Settings) { s.StartAt, s.EndAt = &start, &end },
		"targeting list": func(s *ads.Settings) { s.Targeting = json.RawMessage(`[1]`) },
		"targeting junk": func(s *ads.Settings) { s.Targeting = json.RawMessage(`{"a":`) },
	} {
		s := valid()
		edit(&s)
		var ve *ads.ValidationError
		if err := s.Validate(known); !errors.As(err, &ve) {
			t.Errorf("%s: %v", name, err)
		}
	}
}

func patch(t *testing.T, body string) ads.Patch {
	t.Helper()
	var p ads.Patch
	if err := json.Unmarshal([]byte(body), &p); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestPatchApply(t *testing.T) {
	cur := valid()
	start := time.Date(2026, 11, 6, 0, 0, 0, 0, time.UTC)
	cur.StartAt = &start
	cur.Targeting = json.RawMessage(`{"categories":["Pens"]}`)

	// Absent fields stay, present ones change, null clears a nullable field.
	next, err := patch(t, `{"bid_cpc_cents": 55, "start_at": null, "targeting": null}`).Apply(cur)
	if err != nil {
		t.Fatal(err)
	}
	if next.BidCents != 55 || next.DailyBudgetCents != 2000 || next.Name != cur.Name || next.StartAt != nil || next.Targeting != nil {
		t.Fatalf("patched %+v", next)
	}
	if cur.StartAt == nil || cur.BidCents != 40 {
		t.Fatal("Apply changed its input")
	}
	next, _ = patch(t, `{"end_at": "2026-12-01T00:00:00Z", "item_ids": [4]}`).Apply(cur)
	if next.EndAt == nil || !next.EndAt.Equal(time.Date(2026, 12, 1, 0, 0, 0, 0, time.UTC)) || len(next.ItemIDs) != 1 {
		t.Fatalf("patched %+v", next)
	}
	// Required fields cannot be null; an ended campaign cannot change.
	if _, err := patch(t, `{"daily_budget_cents": null}`).Apply(cur); err == nil {
		t.Fatal("null budget accepted")
	}
	cur.Status = ads.StatusEnded
	if _, err := patch(t, `{"status": "active"}`).Apply(cur); !errors.Is(err, ads.ErrEnded) {
		t.Fatalf("ended: %v", err)
	}
}
