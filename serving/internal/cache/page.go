// Package cache holds what the api keeps in Redis between requests. M1: result pages
// (page:{request_id}, design §4.5); the candidate cache comes in M2.
package cache

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"

	"github.com/heyemlee/recloop/serving/internal/experiment"
	"github.com/heyemlee/recloop/serving/internal/mixer"
)

// PageTTL is how long a result list can be paged through.
const PageTTL = 10 * time.Minute

// PageKey of a request.
func PageKey(requestID uuid.UUID) string { return "page:" + requestID.String() }

// Page is the full ranked list of one request; later pages are slices of it, so paging never
// repeats or skips an item even when the user's state changes in between.
type Page struct {
	RequestID     uuid.UUID                        `json:"request_id"`
	UserID        int64                            `json:"user_id"`
	Surface       string                           `json:"surface"`
	Seed          int32                            `json:"seed"`
	ModelVer      string                           `json:"model_ver"`
	Exp           map[string]experiment.Assignment `json:"exp"`
	ConfigVersion int                              `json:"config_version"`
	Degraded      []string                         `json:"degraded"`
	HV            int64                            `json:"hv"`
	Items         []mixer.Fused                    `json:"items"`
	CreatedAt     time.Time                        `json:"created_at"`
}

// ErrNoPage means the cursor's page expired or never existed.
var ErrNoPage = errors.New("page expired or unknown")

// Pages stores pages in Redis.
type Pages struct{ R redis.UniversalClient }

func (p Pages) Put(ctx context.Context, pg *Page) error {
	raw, err := json.Marshal(pg)
	if err != nil {
		return err
	}
	return p.R.Set(ctx, PageKey(pg.RequestID), raw, PageTTL).Err()
}

func (p Pages) Get(ctx context.Context, requestID uuid.UUID) (*Page, error) {
	raw, err := p.R.Get(ctx, PageKey(requestID)).Bytes()
	if errors.Is(err, redis.Nil) {
		return nil, ErrNoPage
	}
	if err != nil {
		return nil, err
	}
	var pg Page
	if err := json.Unmarshal(raw, &pg); err != nil {
		return nil, err
	}
	return &pg, nil
}

// Cursor points into a page: the request and the offset of the next item.
type Cursor struct {
	RequestID uuid.UUID
	Offset    int
}

// Encode returns an opaque URL-safe token.
func (c Cursor) Encode() string {
	return base64.RawURLEncoding.EncodeToString([]byte(c.RequestID.String() + ":" + strconv.Itoa(c.Offset)))
}

// DecodeCursor parses a token from Encode.
func DecodeCursor(s string) (Cursor, error) {
	raw, err := base64.RawURLEncoding.DecodeString(s)
	if err != nil {
		return Cursor{}, fmt.Errorf("bad cursor: %w", err)
	}
	id, off, ok := strings.Cut(string(raw), ":")
	if !ok {
		return Cursor{}, errors.New("bad cursor")
	}
	rid, err := uuid.Parse(id)
	if err != nil {
		return Cursor{}, fmt.Errorf("bad cursor: %w", err)
	}
	n, err := strconv.Atoi(off)
	if err != nil || n <= 0 {
		return Cursor{}, errors.New("bad cursor offset")
	}
	return Cursor{RequestID: rid, Offset: n}, nil
}
