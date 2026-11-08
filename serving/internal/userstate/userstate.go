// Package userstate is the per-user state in Redis (design §12.2): recent history, history
// version, purchases and recent impressions. The api reads it on every request; the aggregator
// applies the events from the event log to it (the api itself only on the synchronous path,
// EVENTS_MODE=direct). Both use the same script and the same dedup group.
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
	// DedupGroup is the consumer group in dedup:{group}:{event_id} of everything that applies
	// events to user state: the aggregator, a rebuild from events_raw and the synchronous path
	// share it, so an event applied by one is not applied again by another.
	DedupGroup = "aggregator"
	// FreqTTL keeps a frequency-cap counter (design §12.2: two days).
	FreqTTL = 48 * time.Hour
)

// Keys of one user.
func HistKey(uid int64) string   { return fmt.Sprintf("u:%d:hist", uid) }
func HVKey(uid int64) string     { return fmt.Sprintf("u:%d:hv", uid) }
func BoughtKey(uid int64) string { return fmt.Sprintf("u:%d:bought", uid) }
func SeenKey(uid int64) string   { return fmt.Sprintf("u:%d:seen", uid) }
func DedupKey(group, eventID string) string {
	return fmt.Sprintf("dedup:%s:%s", group, eventID)
}

// FreqKey is fc:{uid}:{campaign_id}:{yyyymmdd}: the user's impressions of a campaign's sponsored
// slots served on that UTC day (design §9.2 step 2), counted from impression events. The date
// format is the one of spend:{campaign_id}:{yyyymmdd} and of the events_raw partitions.
func FreqKey(uid, campaignID int64, day time.Time) string {
	return fmt.Sprintf("fc:%d:%d:%s", uid, campaignID, day.UTC().Format("20060102"))
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

// Store reads and updates user state. Group is the dedup group (DedupGroup when empty).
type Store struct {
	R     redis.UniversalClient
	Group string
}

func (s Store) group() string {
	if s.Group == "" {
		return DedupGroup
	}
	return s.Group
}

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
	// FreqKey, on the impression of a sponsored slot, is the frequency-cap counter it adds to
	// (FreqKey(...)); empty for every other event.
	FreqKey string
}

func (s Store) scriptArgs(c Change, now time.Time) ([]string, []any) {
	keys := []string{
		DedupKey(s.group(), c.EventID), HistKey(c.UserID), HVKey(c.UserID),
		BoughtKey(c.UserID), SeenKey(c.UserID),
	}
	if c.FreqKey != "" {
		keys = append(keys, c.FreqKey)
	}
	args := []any{
		c.Kind, c.ItemID, c.At.UnixMilli(), now.UnixMilli(),
		HistoryMax, SeenWindow.Milliseconds(), DedupTTL.Milliseconds(), FreqTTL.Milliseconds(),
	}
	return keys, args
}

// Frequency reads the frequency-cap counters (0 for a missing key) of keys in one round trip.
func (s Store) Frequency(ctx context.Context, keys []string) ([]int64, error) {
	if len(keys) == 0 {
		return nil, nil
	}
	vals, err := s.R.MGet(ctx, keys...).Result()
	if err != nil {
		return nil, err
	}
	return Ints(vals), nil
}

// Ints converts an MGET reply to integers (missing or unparsable values are 0).
func Ints(vals []any) []int64 {
	out := make([]int64, len(vals))
	for i, v := range vals {
		if str, ok := v.(string); ok {
			out[i], _ = strconv.ParseInt(str, 10, 64)
		}
	}
	return out
}

// Apply applies the change exactly once: the dedup marker and the update run in one Lua script,
// so a retried or redelivered event changes nothing (design §7.3). Reports whether it applied.
func (s Store) Apply(ctx context.Context, c Change, now time.Time) (bool, error) {
	keys, args := s.scriptArgs(c, now)
	n, err := applyScript.Run(ctx, s.R, keys, args...).Int()
	return n == 1, err
}

// ApplyBatch applies changes in order in one pipeline round trip (each change is still its own
// atomic script call). Reports per change whether it applied (false: a duplicate). On error some
// changes may have been applied; retrying the batch is safe, the dedup markers skip them.
func (s Store) ApplyBatch(ctx context.Context, cs []Change, now time.Time) ([]bool, error) {
	if len(cs) == 0 {
		return nil, nil
	}
	for attempt := 0; ; attempt++ {
		pipe := s.R.Pipeline()
		cmds := make([]*redis.Cmd, len(cs))
		for i, c := range cs {
			keys, args := s.scriptArgs(c, now)
			cmds[i] = applyScript.EvalSha(ctx, pipe, keys, args...)
		}
		_, err := pipe.Exec(ctx)
		if err != nil && attempt == 0 && redis.HasErrorPrefix(err, "NOSCRIPT") {
			// First use, or Redis restarted: load the script and run the batch again.
			if err := applyScript.Load(ctx, s.R).Err(); err != nil {
				return nil, err
			}
			continue
		}
		if err != nil {
			return nil, err
		}
		out := make([]bool, len(cs))
		for i, cmd := range cmds {
			n, err := cmd.Int()
			if err != nil {
				return nil, err
			}
			out[i] = n == 1
		}
		return out, nil
	}
}
