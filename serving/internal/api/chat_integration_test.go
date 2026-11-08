package api_test

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"github.com/google/uuid"

	"github.com/heyemlee/recloop/serving/internal/api"
)

func ids(items []api.RecItem) []int32 {
	out := make([]int32, len(items))
	for i, it := range items {
		out[i] = it.ItemID
	}
	return out
}

// The chat surface (design §10.2) is the shopping assistant's personalisation signal: the home
// feed's sources over the user's history followed by the assistant's seed items (the first seed
// as the most recent item). It has no sponsored slots, skips the candidate cache, is not paged,
// keeps items the user saw today, and is logged as a served event with its seeds.
func TestChatSurface(t *testing.T) {
	f := newFixtureWith(t, fixtureOpts{items: adItems, ads: true})
	f.campaigns(campaignSpec{bid: 50, budget: 1000, items: []int32{250}}, campaignSpec{bid: 40, budget: 1000, items: []int32{260}})
	// User 21 saw item 5 and clicked item 100: both are "seen", 100 is the history.
	f.post(event("impression", 21, 5, uuid.Must(uuid.NewV7()).String()+":1"), event("click", 21, 100, ""))

	var home recResp
	if code := f.get("/v1/recommendations?user_id=21&limit=50", &home); code != 200 {
		t.Fatalf("home: %d", code)
	}
	if len(sponsored(home.Items)) != 2 || slices.Contains(ids(home.Items), 5) {
		t.Fatalf("home feed: sponsored %d, items %v", len(sponsored(home.Items)), ids(home.Items))
	}

	var chat recResp
	if code := f.get("/v1/recommendations?user_id=21&surface=chat&seed_items=200,30&limit=50&debug=1", &chat); code != 200 {
		t.Fatalf("chat: %d", code)
	}
	got := ids(chat.Items)
	if len(sponsored(chat.Items)) != 0 || chat.NextCursor != "" || len(chat.Degraded) != 0 {
		t.Fatalf("chat: sponsored %d, cursor %q, degraded %v", len(sponsored(chat.Items)), chat.NextCursor, chat.Degraded)
	}
	// The ItemCF neighbours of the seeds (200 -> 201.., 30 -> 31..) and of the history (100 ->
	// 101..) come first, ahead of the popular items; the item seen today stays.
	if i := slices.Index(got, 201); i < 0 || i >= 15 || slices.Index(got, 31) >= 15 || slices.Index(got, 101) >= 15 ||
		!slices.Contains(got, 5) {
		t.Fatalf("chat items %v", got)
	}
	var window []int32
	for _, h := range chat.Debug.History {
		window = append(window, h.ItemID)
	}
	if !slices.Equal(window, []int32{200, 30, 100}) {
		t.Fatalf("history window seen by the sources: %v", window)
	}
	// Without seeds: the history alone (101 leads; 31 is only a popular item, 201 not there).
	var plain recResp
	f.get("/v1/recommendations?user_id=21&surface=chat&limit=50", &plain)
	if p := ids(plain.Items); p[0] != 101 || slices.Contains(p, 201) || slices.Index(p, 31) < 15 {
		t.Fatalf("chat without seeds: %v", p)
	}

	// Served with its seeds; no page and no candidate cache entry in Redis.
	rows := f.servedRows(chat.RequestID)
	if rows[0].Surface != "chat" || len(rows[0].Items) != 50 {
		t.Fatalf("served %+v", rows[0])
	}
	var seeds []int32
	var raw []byte
	if err := f.st.Pool.QueryRow(context.Background(),
		"SELECT payload->'seed_items' FROM events_raw WHERE type = 'served' AND request_id = $1", chat.RequestID).Scan(&raw); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &seeds); err != nil || !slices.Equal(seeds, []int32{200, 30}) {
		t.Fatalf("served seed_items %s (%v)", raw, err)
	}
	ctx := context.Background()
	if n, _ := f.rdb.Exists(ctx, "page:"+chat.RequestID).Result(); n != 0 {
		t.Fatal("a chat result was stored for paging")
	}
	keys, _ := f.rdb.Keys(ctx, "cand:*").Result()
	for _, k := range keys {
		if strings.Contains(k, ":chat:") {
			t.Fatalf("candidate cache entry %s for a chat request", k)
		}
	}

	for _, bad := range []string{"seed_items=99999", "seed_items=abc", "seed_items=1,2,3,4,5,6,7,8,9,10,11"} {
		if code := f.get("/v1/recommendations?user_id=21&surface=chat&"+bad, nil); code != 400 {
			t.Fatalf("%s: %d", bad, code)
		}
	}
}
