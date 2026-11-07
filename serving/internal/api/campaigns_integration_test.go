package api_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/heyemlee/recloop/serving/internal/ads"
	"github.com/heyemlee/recloop/serving/internal/adtoken"
	"github.com/heyemlee/recloop/serving/internal/billing"
	"github.com/heyemlee/recloop/serving/internal/events"
	"github.com/heyemlee/recloop/serving/internal/store"
	"github.com/heyemlee/recloop/serving/internal/store/db"
)

// call sends a JSON request and decodes the JSON answer into out (if not nil).
func (f *fixture) call(method, path string, body any, header map[string]string, out any) (int, http.Header) {
	f.t.Helper()
	var rd io.Reader
	if body != nil {
		raw, err := json.Marshal(body)
		if err != nil {
			f.t.Fatal(err)
		}
		rd = bytes.NewReader(raw)
	}
	req, err := http.NewRequest(method, f.srv.URL+path, rd)
	if err != nil {
		f.t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range header {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		f.t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if out != nil {
		if err := json.Unmarshal(raw, out); err != nil {
			f.t.Fatalf("%s %s: %d %s: %v", method, path, resp.StatusCode, raw, err)
		}
	}
	return resp.StatusCode, resp.Header
}

func (f *fixture) newCampaign(budget, bid int32, items ...int32) ads.Campaign {
	f.t.Helper()
	var adv ads.Advertiser
	if code, _ := f.call("POST", "/v1/advertisers", map[string]any{"name": "Acme", "brand": "B1"}, nil, &adv); code != 201 {
		f.t.Fatalf("advertiser: %d", code)
	}
	var c ads.Campaign
	code, h := f.call("POST", "/v1/campaigns", map[string]any{
		"advertiser_id": adv.ID, "name": "Pens week", "daily_budget_cents": budget,
		"bid_cpc_cents": bid, "item_ids": items,
	}, nil, &c)
	if code != 201 || h.Get("ETag") != `"1"` || h.Get("Location") != fmt.Sprintf("/v1/campaigns/%d", c.ID) {
		f.t.Fatalf("create: %d %v %+v", code, h, c)
	}
	return c
}

func TestCampaignCRUD(t *testing.T) {
	f := newFixture(t)
	c := f.newCampaign(2000, 40, 5, 1, 5)
	today := time.Now().UTC().Format(time.DateOnly)
	if c.Status != "active" || c.FreqCapPerDay != 3 || len(c.ItemIDs) != 2 || c.ItemIDs[0] != 1 ||
		c.Today != (ads.BudgetDay{Day: today, BudgetCents: 2000}) {
		t.Fatalf("created %+v", c)
	}
	var list []ads.Campaign
	f.call("GET", fmt.Sprintf("/v1/campaigns?advertiser_id=%d", c.AdvertiserID), nil, nil, &list)
	if len(list) != 1 || list[0].ID != c.ID {
		t.Fatalf("list %+v", list)
	}
	path := fmt.Sprintf("/v1/campaigns/%d", c.ID)

	// Optimistic locking: no If-Match, then a good one, then a stale one.
	if code, _ := f.call("PATCH", path, map[string]any{"bid_cpc_cents": 50}, nil, nil); code != http.StatusPreconditionRequired {
		t.Fatalf("PATCH without If-Match: %d", code)
	}
	var got ads.Campaign
	code, h := f.call("PATCH", path, map[string]any{"bid_cpc_cents": 50, "status": "paused"}, map[string]string{"If-Match": `"1"`}, &got)
	if code != 200 || h.Get("ETag") != `"2"` || got.BidCents != 50 || got.Status != "paused" || got.Name != "Pens week" {
		t.Fatalf("PATCH: %d %v %+v", code, h, got)
	}
	var conflict struct {
		Error    string       `json:"error"`
		Campaign ads.Campaign `json:"campaign"`
	}
	code, h = f.call("PATCH", path, map[string]any{"bid_cpc_cents": 60}, map[string]string{"If-Match": `"1"`}, &conflict)
	if code != http.StatusPreconditionFailed || h.Get("ETag") != `"2"` || conflict.Campaign.BidCents != 50 {
		t.Fatalf("stale PATCH: %d %v %+v", code, h, conflict)
	}

	// Rejected input leaves the campaign alone.
	for _, bad := range []map[string]any{
		{"bid_cpc_cents": ads.ReserveCents - 1},
		{"daily_budget_cents": 0},
		{"item_ids": []int{nItems + 7}},
		{"item_ids": []int{}},
		{"name": nil},
		{"status": "deleted"},
		{"color": "red"},
	} {
		if code, _ := f.call("PATCH", path, bad, map[string]string{"If-Match": `"2"`}, nil); code != 400 {
			t.Fatalf("PATCH %v: %d", bad, code)
		}
	}

	// A budget lowered below today's spend is capped at the spend; raising it works at once.
	ctx := context.Background()
	if _, err := f.st.Pool.Exec(ctx, "INSERT INTO campaign_budget_daily (campaign_id, day, budget_cents, spent_cents) VALUES ($1, $2, 2000, 300)",
		c.ID, today); err != nil {
		t.Fatal(err)
	}
	f.call("PATCH", path, map[string]any{"daily_budget_cents": 100}, map[string]string{"If-Match": `"2"`}, &got)
	if got.DailyBudgetCents != 100 || got.Today.BudgetCents != 300 || got.Today.SpentCents != 300 {
		t.Fatalf("lowered below spend: %+v", got)
	}
	f.call("PATCH", path, map[string]any{"daily_budget_cents": 5000, "item_ids": []int{7}}, map[string]string{"If-Match": `"3"`}, &got)
	if got.Today.BudgetCents != 5000 || len(got.ItemIDs) != 1 || got.ItemIDs[0] != 7 || got.Version != 4 {
		t.Fatalf("raised: %+v", got)
	}

	// DELETE ends it (idempotent); an ended campaign cannot change.
	if code, _ := f.call("DELETE", path, nil, map[string]string{"If-Match": `"3"`}, nil); code != http.StatusPreconditionFailed {
		t.Fatalf("stale DELETE: %d", code)
	}
	for range 2 {
		if code, _ := f.call("DELETE", path, nil, nil, &got); code != 200 || got.Status != "ended" {
			t.Fatalf("DELETE: %d %+v", code, got)
		}
	}
	if code, _ := f.call("PATCH", path, map[string]any{"status": "active"}, map[string]string{"If-Match": `"5"`}, nil); code != http.StatusConflict {
		t.Fatalf("PATCH of an ended campaign: %d", code)
	}
	if code, _ := f.call("GET", "/v1/campaigns/999999", nil, nil, nil); code != 404 {
		t.Fatalf("missing campaign: %d", code)
	}
	if code, _ := f.call("POST", "/v1/campaigns", map[string]any{
		"advertiser_id": 999999, "name": "x", "daily_budget_cents": 100, "bid_cpc_cents": 10, "item_ids": []int{1},
	}, nil, nil); code != 400 {
		t.Fatalf("unknown advertiser: %d", code)
	}
}

// The report counts sponsored impressions from the served log and clicks and spend from the
// ledger, all on the day the slot was served.
func TestCampaignReport(t *testing.T) {
	f := newFixture(t)
	c := f.newCampaign(1000, 40, 5)
	ctx := context.Background()
	keys, err := adtoken.ParseKeys("r1:report-test-key-0123456789abcdef0123456789")
	if err != nil {
		t.Fatal(err)
	}
	bill := &billing.Service{DB: f.st.Pool, Keys: keys}

	// One request with the campaign's item in sponsored slot 3, shown and clicked; a repeat
	// click; a click that comes too late on a second slot.
	rid := uuid.Must(uuid.NewV7())
	served := time.Now().UTC()
	payload, _ := json.Marshal(map[string]any{
		"model_ver": "test", "exp": map[string]any{"recall": map[string]any{"variant": "control"}}, "config_version": 1,
		"items": []map[string]any{
			{"item_id": 2, "pos": 1, "slot": "organic"},
			{"item_id": 5, "pos": 3, "slot": "sponsored", "campaign_id": c.ID, "price": 31},
			{"item_id": 5, "pos": 9, "slot": "sponsored", "campaign_id": c.ID, "price": 25},
		},
	})
	p, err := events.Served(1, rid, payload)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.st.InsertServed(ctx, []db.InsertEventParams{events.Params(p)}); err != nil {
		t.Fatal(err)
	}
	f.post(event("impression", 1, 5, events.ImpressionID(rid, 3)), event("impression", 1, 2, events.ImpressionID(rid, 1)))
	click := func(pos int, price int32, delay time.Duration) billing.Result {
		t.Helper()
		claims := adtoken.Claims{RequestID: rid, Position: pos, CampaignID: c.ID, ItemID: 5, PriceCents: price, UserID: 1}
		tok, err := keys.Sign(claims, served)
		if err != nil {
			t.Fatal(err)
		}
		r, err := bill.Handle(ctx, billing.Click{
			EventID: uuid.Must(uuid.NewV7()), ImpressionID: claims.ImpressionID(), UserID: 1, ItemID: 5,
			AdToken: tok, ReceivedAt: served.Add(delay),
		})
		if err != nil {
			t.Fatal(err)
		}
		return r
	}
	if r := click(3, 31, 10*time.Second); r.Outcome != billing.Charged || r.BilledCents != 31 {
		t.Fatalf("click: %+v", r)
	}
	click(3, 31, 20*time.Second)
	if r := click(9, 25, adtoken.TTL+time.Second); r.Outcome != billing.Expired {
		t.Fatalf("late click: %+v", r)
	}

	var rep store.Report
	day := served.Format(time.DateOnly)
	if code, _ := f.call("GET", fmt.Sprintf("/v1/campaigns/%d/report?from=%s&to=%s", c.ID, day, day), nil, nil, &rep); code != 200 {
		t.Fatalf("report: %d", code)
	}
	d := rep.Total
	if len(rep.Days) != 1 || d.Impressions != 1 || d.Clicks != 1 || d.BilledCents != 31 || d.SpentCents != 31 ||
		d.RepeatClicks != 1 || d.ExpiredClicks != 1 || d.CTR == nil || *d.CTR != 1 || *d.AvgCPCCents != 31 {
		t.Fatalf("report %+v", d)
	}
	if code, _ := f.call("GET", fmt.Sprintf("/v1/campaigns/%d/report?from=2026-01-01&to=2026-12-31", c.ID), nil, nil, nil); code != 400 {
		t.Fatalf("a year of report: %d", code)
	}
}
