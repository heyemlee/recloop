// Package userstate is the per-user state in Redis (design §12.2): recent history, history
// version, purchases and recent impressions. The api reads it on every request; in M1 the api
// also applies events to it directly (the aggregator takes this over in M2 with the same
// script).
package userstate

import (
	"context"
	_ "embed"
	"fmt"
	"strconv"
	"time"

	"github.com/redis/go-redis/v9"
)

const (
	// HistoryMax is how many recent items u:{uid}:hist keeps.
	HistoryMax = 50
	// SeenWindow is how long an impression excludes an item from recommendations.
	SeenWindow = 24 * time.Hour
	// DedupTTL keeps the "already applied" marker of an event (design §7.3: 7 days).
	DedupTTL = 7 * 24 * time.Hour
	// DedupGroup is this writer's consumer group in dedup:{group}:{event_id}.
	DedupGroup = "api"
)

// Keys of one user.
func HistKey(uid int64) string   { return fmt.Sprintf("u:%d:hist", uid) }
func HVKey(uid int64) string     { return fmt.Sprintf("u:%d:hv", uid) }
func BoughtKey(uid int64) string { return fmt.Sprintf("u:%d:bought", uid) }
func SeenKey(uid int64) string   { return fmt.Sprintf("u:%d:seen", uid) }
func DedupKey(group, eventID string) string {
	return fmt.Sprintf("dedup:%s:%s", group, eventID)
}

// DemoUsersKey holds the demo user list written by `make seed`.
const DemoUsersKey = "demo:users"

// Context is what a recommendation request needs to know about the user.
type Context struct {
	History []int32 // most recent HistoryMax items, oldest first
	HV      int64   // history version: bumped by every history change
	Bought  map[int32]bool
	Seen    map[int32]bool // impressions in the last SeenWindow
}

// Store reads and updates user state.
type Store struct{ R redis.UniversalClient }

// Read fetches the user's context in one pipeline round trip.
func (s Store) Read(ctx context.Context, uid int64, now time.Time) (Context, error) {
	pipe := s.R.Pipeline()
	hist := pipe.ZRangeArgs(ctx, redis.ZRangeArgs{Key: HistKey(uid), Start: 0, Stop: -1})
	hv := pipe.Get(ctx, HVKey(uid))
	bought := pipe.SMembers(ctx, BoughtKey(uid))
	seen := pipe.ZRangeByScore(ctx, SeenKey(uid), &redis.ZRangeBy{
		Min: strconv.FormatInt(now.Add(-SeenWindow).UnixMilli(), 10), Max: "+inf",
	})
	if _, err := pipe.Exec(ctx); err != nil && err != redis.Nil {
		return Context{}, err
	}
	c := Context{Bought: map[int32]bool{}, Seen: map[int32]bool{}}
	for _, m := range hist.Val() {
		if id, err := strconv.ParseInt(m, 10, 32); err == nil {
			c.History = append(c.History, int32(id))
		}
	}
	if v, err := hv.Int64(); err == nil {
		c.HV = v
	}
	for _, m := range bought.Val() {
		if id, err := strconv.ParseInt(m, 10, 32); err == nil {
			c.Bought[int32(id)] = true
		}
	}
	for _, m := range seen.Val() {
		if id, err := strconv.ParseInt(m, 10, 32); err == nil {
			c.Seen[int32(id)] = true
		}
	}
	return c, nil
}

// Kinds of state change an event causes.
const (
	KindHistory    = "history"    // click, add_to_cart: history (and seen)
	KindPurchase   = "purchase"   // history + bought (and seen)
	KindImpression = "impression" // seen
)

//go:embed apply.lua
var applyLua string

var applyScript = redis.NewScript(applyLua)

// Change is one event's effect on user state.
type Change struct {
	EventID string
	UserID  int64
	ItemID  int32
	Kind    string
	At      time.Time // event time (from the UUIDv7 event_id)
}

// Apply applies the change exactly once: the dedup marker and the update run in one Lua script,
// so a retried or redelivered event changes nothing (design §7.3). Reports whether it applied.
func (s Store) Apply(ctx context.Context, c Change, now time.Time) (bool, error) {
	keys := []string{
		DedupKey(DedupGroup, c.EventID), HistKey(c.UserID), HVKey(c.UserID),
		BoughtKey(c.UserID), SeenKey(c.UserID),
	}
	args := []any{
		c.Kind, c.ItemID, c.At.UnixMilli(), now.UnixMilli(),
		HistoryMax, SeenWindow.Milliseconds(), DedupTTL.Milliseconds(),
	}
	n, err := applyScript.Run(ctx, s.R, keys, args...).Int()
	return n == 1, err
}
