package cache

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/klauspost/compress/s2"
	"github.com/redis/go-redis/v9"

	"github.com/heyemlee/recloop/serving/internal/mixer"
	"github.com/heyemlee/recloop/serving/internal/recall"
)

// Candidate cache (design §4.5): the output of recall + ranking, the expensive part that
// changes slowly, kept for CandidateTTL. Filters (seen, bought, stock), ads and budgets change
// quickly and are applied on every request, cached or not.
const (
	CandidateTTL   = 5 * time.Minute
	CandidateDepth = 300
)

// CandidateKey names one cached candidate list.
//
//	home: cand:{uid}:home:{variant}:{model_ver}:{hv}
//	item: cand:item:{seed}:{variant}:{model_ver}
//
// hv is the user's history version: a new click, add-to-cart or purchase bumps it, so the key
// changes by itself and nothing has to be invalidated. model_ver keeps a new bundle's results
// apart from the old one's. The item page's candidates (ItemCF + text neighbours of the current
// item) do not depend on the user, so they are cached per item and shared by every user.
func CandidateKey(userID int64, surface string, seed int32, variant, modelVer string, hv int64) string {
	if surface == recall.SurfaceItem {
		return "cand:item:" + strconv.FormatInt(int64(seed), 10) + ":" + variant + ":" + modelVer
	}
	var b strings.Builder
	b.Grow(64)
	b.WriteString("cand:")
	b.WriteString(strconv.FormatInt(userID, 10))
	b.WriteByte(':')
	b.WriteString(surface)
	b.WriteByte(':')
	b.WriteString(variant)
	b.WriteByte(':')
	b.WriteString(modelVer)
	b.WriteByte(':')
	b.WriteString(strconv.FormatInt(hv, 10))
	return b.String()
}

// Candidates is one cache entry: the fused list, best first, and the version each source
// reported.
type Candidates struct {
	Items    []mixer.Fused
	Versions map[string]string
}

// ErrMiss: no entry under the key.
var ErrMiss = errors.New("cache miss")

const candidateFormat = 1

// EncodeCandidates: format byte, versions, fused list; s2-compressed.
func EncodeCandidates(c *Candidates) []byte {
	b := make([]byte, 0, 64+36*len(c.Items))
	b = append(b, candidateFormat)
	b = binary.AppendUvarint(b, uint64(len(c.Versions)))
	for k, v := range c.Versions {
		b = binary.AppendUvarint(b, uint64(len(k)))
		b = append(b, k...)
		b = binary.AppendUvarint(b, uint64(len(v)))
		b = append(b, v...)
	}
	b = appendFused(b, c.Items)
	return s2.Encode(nil, b)
}

// DecodeCandidates reverses EncodeCandidates.
func DecodeCandidates(raw []byte) (*Candidates, error) {
	b, err := s2.Decode(nil, raw)
	if err != nil {
		return nil, fmt.Errorf("cache: %w", err)
	}
	r := &reader{b: b}
	if r.byte() != candidateFormat {
		return nil, errCorrupt
	}
	c := &Candidates{Versions: map[string]string{}}
	for n := r.count(2); n > 0 && r.err == nil; n-- {
		k := string(r.bytes(r.uvarint()))
		c.Versions[k] = string(r.bytes(r.uvarint()))
	}
	c.Items = readFused(r)
	if r.err != nil {
		return nil, r.err
	}
	return c, nil
}

// CandidateCache keeps candidate lists in Redis.
type CandidateCache struct {
	R   redis.UniversalClient
	TTL time.Duration
}

func (c CandidateCache) ttl() time.Duration {
	if c.TTL <= 0 {
		return CandidateTTL
	}
	return c.TTL
}

// Get returns the entry under key, or ErrMiss.
func (c CandidateCache) Get(ctx context.Context, key string) (*Candidates, error) {
	raw, err := c.R.Get(ctx, key).Bytes()
	if errors.Is(err, redis.Nil) {
		return nil, ErrMiss
	}
	if err != nil {
		return nil, err
	}
	return DecodeCandidates(raw)
}

// SetCmd queues the entry on a pipeline (it is written together with the page).
func (c CandidateCache) SetCmd(ctx context.Context, p redis.Pipeliner, key string, cands *Candidates) {
	p.Set(ctx, key, EncodeCandidates(cands), c.ttl())
}
