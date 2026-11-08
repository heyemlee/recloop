// Package cache holds what the api keeps in Redis between requests (design §4.5): result pages
// (page:{request_id}, 10 minutes) and the candidate cache (cand:..., 5 minutes).
package cache

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/klauspost/compress/s2"
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
	Items         []mixer.Fused                    `json:"items,omitempty"`
	// Sponsored are the ad slots among Items (by position): a later page shows them as the
	// first one did, with the tokens signed when the request was served.
	Sponsored []Slot    `json:"sponsored,omitempty"`
	CreatedAt time.Time `json:"created_at"`
}

// Slot is a sponsored position of a result list (design §9.2): the auction's winner and the
// signed ad_token its impression and click events carry.
type Slot struct {
	Position   int    `json:"pos"`
	ItemID     int32  `json:"item_id"`
	CampaignID int64  `json:"campaign_id"`
	PriceCents int32  `json:"price"`
	PCTRPPM    int64  `json:"pctr_ppm"`
	AdToken    string `json:"ad_token"`
}

const pageFormat = 'P'

// EncodePage: 'P', the request metadata as JSON, the items in the binary list format;
// s2-compressed.
func EncodePage(pg *Page) ([]byte, error) {
	meta := *pg
	meta.Items = nil
	head, err := json.Marshal(&meta)
	if err != nil {
		return nil, err
	}
	b := make([]byte, 0, 16+len(head)+36*len(pg.Items))
	b = append(b, pageFormat)
	b = binary.AppendUvarint(b, uint64(len(head)))
	b = append(b, head...)
	b = appendFused(b, pg.Items)
	return s2.Encode(nil, b), nil
}

// DecodePage reverses EncodePage; it also reads the JSON pages M1 stored.
func DecodePage(raw []byte) (*Page, error) {
	var pg Page
	if len(raw) > 0 && raw[0] == '{' {
		if err := json.Unmarshal(raw, &pg); err != nil {
			return nil, err
		}
		return &pg, nil
	}
	b, err := s2.Decode(nil, raw)
	if err != nil {
		return nil, fmt.Errorf("page: %w", err)
	}
	r := &reader{b: b}
	if r.byte() != pageFormat {
		return nil, errCorrupt
	}
	head := r.bytes(r.uvarint())
	if r.err != nil {
		return nil, r.err
	}
	if err := json.Unmarshal(head, &pg); err != nil {
		return nil, err
	}
	pg.Items = readFused(r)
	if r.err != nil {
		return nil, r.err
	}
	return &pg, nil
}

// ErrNoPage means the cursor's page expired or never existed.
var ErrNoPage = errors.New("page expired or unknown")

// Pages stores pages in Redis.
type Pages struct{ R redis.UniversalClient }

func (p Pages) Put(ctx context.Context, pg *Page) error {
	raw, err := EncodePage(pg)
	if err != nil {
		return err
	}
	return p.R.Set(ctx, PageKey(pg.RequestID), raw, PageTTL).Err()
}

// PutCmd queues the page on a pipeline.
func (p Pages) PutCmd(ctx context.Context, pipe redis.Pipeliner, pg *Page) error {
	raw, err := EncodePage(pg)
	if err != nil {
		return err
	}
	pipe.Set(ctx, PageKey(pg.RequestID), raw, PageTTL)
	return nil
}

func (p Pages) Get(ctx context.Context, requestID uuid.UUID) (*Page, error) {
	raw, err := p.R.Get(ctx, PageKey(requestID)).Bytes()
	if errors.Is(err, redis.Nil) {
		return nil, ErrNoPage
	}
	if err != nil {
		return nil, err
	}
	return DecodePage(raw)
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
