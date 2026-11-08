package api_test

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"slices"
	"testing"
	"time"

	"github.com/heyemlee/recloop/serving/internal/ads"
	"github.com/heyemlee/recloop/serving/internal/adtoken"
	"github.com/heyemlee/recloop/serving/internal/api"
	"github.com/heyemlee/recloop/serving/internal/billing"
	"github.com/heyemlee/recloop/serving/internal/store"
	"github.com/heyemlee/recloop/serving/internal/userstate"
)

const testAdKeys = "t1:api-test-ad-token-key-0123456789abcdef0123"

// adItems: the ads tests use a catalog larger than the 200-item result list, so the items of
// the campaigns (250 and up) are never organic results.
const adItems = 300

type campaignSpec struct {
	bid, budget, cap int32
	items            []int32
}

// campaigns creates one advertiser per campaign (the auction gives an advertiser one slot) and
// loads them into the api's campaign index.
func (f *fixture) campaigns(specs ...campaignSpec) []int64 {
	f.t.Helper()
	ctx := context.Background()
	today := time.Now()
	var ids []int64
	for i, c := range specs {
		adv, err := f.st.CreateAdvertiser(ctx, "advertiser "+string(rune('A'+i)), "")
		if err != nil {
			f.t.Fatal(err)
		}
		set := ads.Settings{Name: "c", Status: ads.StatusActive, DailyBudgetCents: c.budget, BidCents: c.bid,
			FreqCapPerDay: max(c.cap, 3), ItemIDs: c.items}
		if err := set.Validate(f.cat.Has); err != nil {
			f.t.Fatal(err)
		}
		cmp, err := f.st.CreateCampaign(ctx, adv.ID, set, today)
		if err != nil {
			f.t.Fatal(err)
		}
		ids = append(ids, cmp.ID)
	}
	if err := f.ads.Index.Refresh(ctx); err != nil {
		f.t.Fatal(err)
	}
	return ids
}

func (f *fixture) refreshExperiments() {
	f.t.Helper()
	if err := f.exps.Refresh(context.Background()); err != nil {
		f.t.Fatal(err)
	}
}

func (f *fixture) claims(it api.RecItem) adtoken.Claims {
	f.t.Helper()
	c, err := f.ads.Keys.Verify(it.AdToken, time.Now())
	if err != nil {
		f.t.Fatalf("ad_token of position %d: %v", it.Position, err)
	}
	return c
}

type servedRow struct {
	Surface string `json:"surface"`
	Items   []struct {
		ItemID     int32  `json:"item_id"`
		Pos        int    `json:"pos"`
		Slot       string `json:"slot"`
		CampaignID int64  `json:"campaign_id"`
		Price      int32  `json:"price"`
		AdToken    string `json:"ad_token"`
	} `json:"items"`
	Ads *struct {
		Outcome string `json:"outcome"`
		Stats   struct {
			Winners int `json:"winners"`
		} `json:"stats"`
	} `json:"ads"`
}

// served waits for the served event of a request (written asynchronously) and decodes it.
func (f *fixture) servedRows(requestID string) []servedRow {
	f.t.Helper()
	var out []servedRow
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		rows, err := f.st.Pool.Query(context.Background(),
			"SELECT payload FROM events_raw WHERE type = 'served' AND request_id = $1 ORDER BY server_ts", requestID)
		if err != nil {
			f.t.Fatal(err)
		}
		out = out[:0]
		for rows.Next() {
			var raw []byte
			if err := rows.Scan(&raw); err != nil {
				f.t.Fatal(err)
			}
			var r servedRow
			if err := json.Unmarshal(raw, &r); err != nil {
				f.t.Fatal(err)
			}
			out = append(out, r)
		}
		rows.Close()
		if len(out) > 0 {
			return out
		}
		time.Sleep(20 * time.Millisecond)
	}
	f.t.Fatalf("no served event for %s", requestID)
	return nil
}

func sponsored(items []api.RecItem) []api.RecItem {
	var out []api.RecItem
	for _, it := range items {
		if it.Slot == api.SlotSponsored {
			out = append(out, it)
		}
	}
	return out
}

// Positions 3 and 9 of the home feed are the auction's two winners, labelled, with tokens that
// name the request, the position, the campaign, the item, the GSP price and the user; the
// served event records the same; a later page keeps the slots of the first.
func TestSponsoredSlotsOnTheHomeFeed(t *testing.T) {
	f := newFixtureWith(t, fixtureOpts{items: adItems, ads: true})
	ids := f.campaigns(
		campaignSpec{bid: 50, budget: 1000, items: []int32{251, 250}},
		campaignSpec{bid: 30, budget: 1000, items: []int32{260}},
		campaignSpec{bid: 20, budget: 1000, items: []int32{270}},
	)
	var page recResp
	if code := f.get("/v1/recommendations?user_id=5&limit=20&debug=1", &page); code != 200 {
		t.Fatalf("status %d", code)
	}
	if len(page.Items) != 20 || len(page.Degraded) != 0 {
		t.Fatalf("%d items, degraded %v", len(page.Items), page.Degraded)
	}
	ads := sponsored(page.Items)
	if len(ads) != 2 || ads[0].Position != 3 || ads[1].Position != 9 {
		t.Fatalf("sponsored %+v", ads)
	}
	// Equal pCTR: ranked by bid. GSP: the first pays the second's bid + 1¢, the second the
	// third's + 1¢; the third (a loser) only sets a price.
	want := []struct {
		cid   int64
		item  int32
		price int32
	}{{ids[0], 250, 31}, {ids[1], 260, 21}}
	for i, it := range ads {
		c := f.claims(it)
		if it.Label != "Sponsored" || it.Source != "ads" || it.CampaignID != want[i].cid || it.ItemID != want[i].item ||
			c.CampaignID != want[i].cid || c.ItemID != want[i].item || c.PriceCents != want[i].price ||
			c.UserID != 5 || c.Position != it.Position || c.RequestID.String() != page.RequestID ||
			c.ImpressionID() != it.ImpressionID {
			t.Fatalf("slot %d: %+v, claims %+v", it.Position, it, c)
		}
	}
	for i, it := range page.Items {
		if it.Position != i+1 {
			t.Fatalf("positions %d at index %d", it.Position, i)
		}
		if it.Slot == api.SlotOrganic && it.ItemID >= 250 {
			t.Fatalf("a campaign item among the organic results: %+v", it)
		}
	}
	if d := page.Debug.Ads; d == nil || d.Outcome != "filled" || d.Stats.Campaigns != 3 || d.Stats.Candidates != 4 || len(d.Slots) != 2 {
		t.Fatalf("debug %+v", page.Debug.Ads)
	}

	rows := f.servedRows(page.RequestID)
	if len(rows) != 1 || rows[0].Ads == nil || rows[0].Ads.Outcome != "filled" || rows[0].Ads.Stats.Winners != 2 {
		t.Fatalf("served %+v", rows)
	}
	for _, it := range rows[0].Items {
		isAd := it.Pos == 3 || it.Pos == 9
		if isAd != (it.Slot == "sponsored") || isAd != (it.CampaignID != 0) || isAd != (it.AdToken != "") {
			t.Fatalf("served item %+v", it)
		}
	}
	if it := rows[0].Items[2]; it.CampaignID != ids[0] || it.Price != 31 || it.AdToken != ads[0].AdToken {
		t.Fatalf("served slot 3 %+v", it)
	}

	// limit 5: slot 9 comes with the second page, from the page store, token unchanged.
	var p1, p2 recResp
	f.get("/v1/recommendations?user_id=6&limit=5", &p1)
	f.get("/v1/recommendations?user_id=6&limit=5&cursor="+p1.NextCursor, &p2)
	a1, a2 := sponsored(p1.Items), sponsored(p2.Items)
	if len(a1) != 1 || a1[0].Position != 3 || len(a2) != 1 || a2[0].Position != 9 || a2[0].ItemID != 260 {
		t.Fatalf("pages %+v / %+v", a1, a2)
	}
	if c := f.claims(a2[0]); c.RequestID.String() != p1.RequestID || c.Position != 9 {
		t.Fatalf("second page's token %+v", c)
	}
	var p3 recResp
	f.get("/v1/recommendations?user_id=6&limit=5&cursor="+p2.NextCursor, &p3)
	if len(sponsored(p3.Items)) != 0 {
		t.Fatal("a third page with ads")
	}
	// The item surface has no sponsored slots.
	var item recResp
	f.get("/v1/recommendations?user_id=6&surface=item&item_id=3&limit=20", &item)
	if len(sponsored(item.Items)) != 0 {
		t.Fatal("ads on the item surface")
	}
}

// Two advertisers promoting the same item never fill both slots with it.
func TestOneSponsoredSlotPerItem(t *testing.T) {
	f := newFixtureWith(t, fixtureOpts{items: adItems, ads: true})
	ids := f.campaigns(
		campaignSpec{bid: 50, budget: 1000, items: []int32{250}},
		campaignSpec{bid: 40, budget: 1000, items: []int32{250}},
		campaignSpec{bid: 30, budget: 1000, items: []int32{270}},
	)
	var r recResp
	f.get("/v1/recommendations?user_id=7&limit=20", &r)
	got := sponsored(r.Items)
	if len(got) != 1 || got[0].Position != 3 || got[0].CampaignID != ids[0] || got[0].ItemID != 250 {
		t.Fatalf("sponsored %+v", got)
	}
	if len(r.Items) != 20 || r.Items[8].Slot != api.SlotOrganic {
		t.Fatalf("position 9: %+v", r.Items[8])
	}
}

// Frequency caps count impression events: a sponsored impression reported with its token adds
// one to fc:{uid}:{cid}:{day}; once the user saw the campaign freq_cap_per_day times today,
// the next one wins. A bought or sold-out item is never advertised.
func TestFrequencyCapAndExclusions(t *testing.T) {
	f := newFixtureWith(t, fixtureOpts{items: adItems, ads: true})
	ids := f.campaigns(
		campaignSpec{bid: 50, budget: 1000, cap: 1, items: []int32{250}},
		campaignSpec{bid: 30, budget: 1000, items: []int32{260, adItems - 1}}, // the last one is sold out
		campaignSpec{bid: 20, budget: 1000, items: []int32{270}},
	)
	// cap 1 is below the default of 3 that campaigns() keeps: set it directly.
	if _, err := f.st.Pool.Exec(context.Background(), "UPDATE campaigns SET freq_cap_per_day = 1 WHERE campaign_id = $1", ids[0]); err != nil {
		t.Fatal(err)
	}
	if err := f.ads.Index.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	var first recResp
	f.get("/v1/recommendations?user_id=8&limit=20", &first)
	got := sponsored(first.Items)
	if len(got) != 2 || got[0].CampaignID != ids[0] || got[1].ItemID != 260 {
		t.Fatalf("first: %+v", got)
	}
	// The impression of slot 3, with its token; an organic impression and one with a forged
	// token do not count.
	imp := event("impression", 8, 250, got[0].ImpressionID)
	imp["ad_token"] = got[0].AdToken
	forged := event("impression", 8, 260, got[1].ImpressionID)
	forged["ad_token"] = got[0].AdToken // another slot's token
	if r := f.post(imp, imp, forged, event("impression", 8, int(first.Items[0].ItemID), first.Items[0].ImpressionID)); r.Accepted != 3 {
		t.Fatalf("events %+v", r)
	}
	n, err := userstate.Store{R: f.rdb}.Frequency(context.Background(), []string{
		userstate.FreqKey(8, ids[0], time.Now()), userstate.FreqKey(8, ids[1], time.Now()),
	})
	if err != nil || n[0] != 1 || n[1] != 0 {
		t.Fatalf("frequency %v %v", n, err)
	}
	var second recResp
	f.get("/v1/recommendations?user_id=8&limit=20", &second)
	got = sponsored(second.Items)
	if len(got) != 2 || got[0].CampaignID != ids[1] || got[1].CampaignID != ids[2] {
		t.Fatalf("after the cap: %+v", got)
	}
	// Item 260 bought: campaign 2 has no item left to show.
	f.post(event("purchase", 8, 260, ""))
	var third recResp
	f.get("/v1/recommendations?user_id=8&limit=20&debug=1", &third)
	got = sponsored(third.Items)
	if len(got) != 1 || got[0].CampaignID != ids[2] || third.Debug.Ads.Stats.NoItem != 1 || third.Debug.Ads.Stats.FreqCap != 1 {
		t.Fatalf("after the purchase: %+v %+v", got, third.Debug.Ads)
	}
}

// The budget gate reads billing's Redis counter: a campaign whose budget left is at most the
// safety margin (here 0 clicks: nothing left) stops; an ads-layer experiment can move the margin.
func TestBudgetGateAndPolicyParams(t *testing.T) {
	f := newFixtureWith(t, fixtureOpts{items: adItems, ads: true})
	ids := f.campaigns(
		campaignSpec{bid: 50, budget: 1000, items: []int32{250}},
		campaignSpec{bid: 30, budget: 1000, items: []int32{260}},
	)
	spent := func(cid int64, cents int64) {
		if err := f.rdb.Set(context.Background(), billing.SpendKey(cid, time.Now()), cents, time.Hour).Err(); err != nil {
			t.Fatal(err)
		}
	}
	first := func(path string) []api.RecItem {
		var r recResp
		f.get(path, &r)
		return sponsored(r.Items)
	}
	spent(ids[0], 999)
	if got := first("/v1/recommendations?user_id=9"); len(got) != 2 || got[0].CampaignID != ids[0] {
		t.Fatalf("1¢ left, margin 0: %+v", got)
	}
	spent(ids[0], 1000)
	if got := first("/v1/recommendations?user_id=9"); len(got) != 1 || got[0].CampaignID != ids[1] {
		t.Fatalf("spent: %+v", got)
	}
	// An ads-layer experiment with a 20-click margin (20 × 50¢ = 1000): campaign 1 stops at
	// 1¢ spent already, campaign 2 (20 × 30¢ = 600) still serves with 1000 left.
	spent(ids[0], 1)
	// testutil.PG keeps the experiments table (migration 00003's row): remove ours afterwards.
	t.Cleanup(func() {
		_, _ = f.st.Pool.Exec(context.Background(), "DELETE FROM experiments WHERE exp_id = 'ads-margin-test'")
	})
	if _, err := f.st.Pool.Exec(context.Background(), `INSERT INTO experiments
		(exp_id, layer, status, bucket_from, bucket_to, variants, config_version)
		VALUES ('ads-margin-test', 'ads', 'running', 0, 1000,
		'[{"name": "m20", "share": 100, "params": {"safety_margin_clicks": 20}}]', 2)`); err != nil {
		t.Fatal(err)
	}
	f.refreshExperiments()
	if got := first("/v1/recommendations?user_id=9"); len(got) != 1 || got[0].CampaignID != ids[1] {
		t.Fatalf("margin 20: %+v", got)
	}
}

// When the ads stage fails, the feed is organic, the reason is in `degraded`, and it is never
// an error: a Redis read that times out, or a campaign snapshot that was never loaded.
func TestAdsFailureLeavesTheOrganicFeed(t *testing.T) {
	plan := testPlan
	plan.Ads = time.Nanosecond
	f := newFixtureWith(t, fixtureOpts{items: adItems, ads: true, plan: &plan})
	f.campaigns(campaignSpec{bid: 50, budget: 1000, items: []int32{250}})
	var r recResp
	if code := f.get("/v1/recommendations?user_id=4&limit=20", &r); code != 200 {
		t.Fatalf("status %d", code)
	}
	if len(r.Items) != 20 || len(sponsored(r.Items)) != 0 || !slices.Contains(r.Degraded, "ads_timeout") {
		t.Fatalf("%d items, %d ads, degraded %v", len(r.Items), len(sponsored(r.Items)), r.Degraded)
	}

	g := newFixtureWith(t, fixtureOpts{items: adItems, ads: true})
	g.ads.Index.Set(nil) // never refreshed (PostgreSQL down since the start)
	var s recResp
	g.get("/v1/recommendations?user_id=4&limit=20", &s)
	if len(s.Items) != 20 || len(sponsored(s.Items)) != 0 || !slices.Contains(s.Degraded, "ads_stale") {
		t.Fatalf("stale: %d items, degraded %v", len(s.Items), s.Degraded)
	}
	stale := ads.NewSnapshot(nil, ads.UTCDay(time.Now()), time.Now().Add(-ads.MaxStaleness-time.Second))
	g.ads.Index.Set(stale)
	g.get("/v1/recommendations?user_id=4&limit=20", &s)
	if !slices.Contains(s.Degraded, "ads_stale") {
		t.Fatalf("an old snapshot: degraded %v", s.Degraded)
	}
}

func (f *fixture) selectAds(body map[string]any) (int, api.AdSelectResponse) {
	f.t.Helper()
	raw, _ := json.Marshal(body)
	resp, err := http.Post(f.srv.URL+"/internal/ads/select", "application/json", bytes.NewReader(raw))
	if err != nil {
		f.t.Fatal(err)
	}
	defer resp.Body.Close()
	var out api.AdSelectResponse
	if resp.StatusCode == 200 {
		if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
			f.t.Fatal(err)
		}
	}
	return resp.StatusCode, out
}

// /internal/ads/select (design §10.3 (5)): the chat reply's sponsored item comes from the
// assistant's candidates (which meet the request's hard constraints), never one of the reply's
// organic items, within max_price_cents (unknown prices excluded), at most one; it has a token
// for position 1 of its own request and a served event (surface chat).
func TestInternalAdSelect(t *testing.T) {
	f := newFixtureWith(t, fixtureOpts{items: adItems, ads: true, prep: func(st *store.Store) {
		if _, err := st.Pool.Exec(context.Background(), "UPDATE items SET price_cents = NULL WHERE item_id = 280"); err != nil {
			t.Fatal(err)
		}
	}})
	ids := f.campaigns(
		campaignSpec{bid: 50, budget: 1000, items: []int32{250}}, // price 350¢ (SeedItems: 100 + id)
		campaignSpec{bid: 30, budget: 1000, items: []int32{260}}, // 360¢
		campaignSpec{bid: 20, budget: 1000, items: []int32{270}}, // not a candidate
		campaignSpec{bid: 10, budget: 1000, items: []int32{280}}, // price unknown
	)
	code, r := f.selectAds(map[string]any{"user_id": 12, "candidate_item_ids": []int{250, 260, 3}, "max_price_cents": 400})
	if code != 200 || len(r.Items) != 1 || r.Surface != "chat" || len(r.Degraded) != 0 {
		t.Fatalf("%d %+v", code, r)
	}
	it := r.Items[0]
	c := f.claims(it)
	// Campaign 1 (50¢) beats campaign 2 (30¢) and pays 31¢.
	if it.ItemID != 250 || it.CampaignID != ids[0] || it.Label != "Sponsored" || it.Position != 1 ||
		c.Position != 1 || c.PriceCents != 31 || c.RequestID.String() != r.RequestID || it.Item == nil {
		t.Fatalf("item %+v claims %+v", it, c)
	}
	rows := f.servedRows(r.RequestID)
	if rows[0].Surface != "chat" || len(rows[0].Items) != 1 || rows[0].Items[0].CampaignID != ids[0] || rows[0].Items[0].Price != 31 {
		t.Fatalf("served %+v", rows[0])
	}

	// The reply shows 250 organically: the ad is 260, at the reserve price (nobody below it).
	_, r = f.selectAds(map[string]any{"user_id": 12, "candidate_item_ids": []int{250, 260}, "exclude_item_ids": []int{250}})
	if len(r.Items) != 1 || r.Items[0].ItemID != 260 || f.claims(r.Items[0]).PriceCents != ads.ReserveCents {
		t.Fatalf("excluded: %+v", r.Items)
	}
	// The price constraint is checked here too.
	_, r = f.selectAds(map[string]any{"user_id": 12, "candidate_item_ids": []int{250, 260}, "max_price_cents": 355})
	if len(r.Items) != 1 || r.Items[0].ItemID != 250 {
		t.Fatalf("max price 355: %+v", r.Items)
	}
	_, r = f.selectAds(map[string]any{"user_id": 12, "candidate_item_ids": []int{250, 260}, "max_price_cents": 300})
	if len(r.Items) != 0 || r.Ads == nil || r.Ads.Outcome != "empty" {
		t.Fatalf("max price 300: %+v", r)
	}
	// An unknown price never meets a budget; without one, the item can be advertised.
	_, r = f.selectAds(map[string]any{"user_id": 12, "candidate_item_ids": []int{280}, "max_price_cents": 10000})
	if len(r.Items) != 0 {
		t.Fatalf("unknown price: %+v", r.Items)
	}
	_, r = f.selectAds(map[string]any{"user_id": 12, "candidate_item_ids": []int{280}})
	if len(r.Items) != 1 || r.Items[0].CampaignID != ids[3] {
		t.Fatalf("no budget: %+v", r.Items)
	}
	// Bought: never advertised to that user.
	f.post(event("purchase", 13, 260, ""))
	_, r = f.selectAds(map[string]any{"user_id": 13, "candidate_item_ids": []int{260}})
	if len(r.Items) != 0 {
		t.Fatalf("bought: %+v", r.Items)
	}
	for _, bad := range []map[string]any{
		{"candidate_item_ids": []int{250}},
		{"user_id": 1, "candidate_item_ids": []int{250}, "slots": 2},
		{"user_id": 1, "candidate_item_ids": []int{250}, "unknown": true},
	} {
		if code, _ := f.selectAds(bad); code != 400 {
			t.Fatalf("%v: %d", bad, code)
		}
	}
}
