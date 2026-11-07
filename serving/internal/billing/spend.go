package billing

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"

	"github.com/heyemlee/recloop/serving/internal/userstate"
)

// SpendCounter is the real-time spend per campaign and day that ad serving reads to stop a
// campaign before its budget runs out (§9.4). The ledger in PostgreSQL is the truth; the
// counter may lag behind it, never run ahead.
type SpendCounter interface {
	Add(ctx context.Context, eventID uuid.UUID, campaignID int64, day time.Time, cents int32) error
}

const (
	// SpendTTL keeps a day's counter (§12.2: two days).
	SpendTTL = 48 * time.Hour
	// DedupGroup is billing's consumer group in dedup:{group}:{event_id}.
	DedupGroup = "billing"
)

// SpendKey is spend:{campaign_id}:{yyyymmdd} for a UTC day.
func SpendKey(campaignID int64, day time.Time) string {
	return fmt.Sprintf("spend:%d:%s", campaignID, day.UTC().Format("20060102"))
}

//go:embed spend.lua
var spendLua string

var spendScript = redis.NewScript(spendLua)

// RedisSpend is the SpendCounter in Redis. The increment and its dedup marker are applied by
// one Lua script, so a retried call (a client retry after a timeout) never counts twice.
type RedisSpend struct{ R redis.UniversalClient }

// Add implements SpendCounter.
func (s RedisSpend) Add(ctx context.Context, eventID uuid.UUID, campaignID int64, day time.Time, cents int32) error {
	keys := []string{userstate.DedupKey(DedupGroup, eventID.String()), SpendKey(campaignID, day)}
	return spendScript.Run(ctx, s.R, keys, cents,
		userstate.DedupTTL.Milliseconds(), SpendTTL.Milliseconds()).Err()
}

// Spent reads a counter (0 if it does not exist).
func (s RedisSpend) Spent(ctx context.Context, campaignID int64, day time.Time) (int64, error) {
	v, err := s.R.Get(ctx, SpendKey(campaignID, day)).Int64()
	if errors.Is(err, redis.Nil) {
		return 0, nil
	}
	return v, err
}
