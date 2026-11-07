// Package adtoken signs and verifies the ad_token of a sponsored slot (docs/design.md §7.1,
// §9.3): an HS256 JWT (golang-jwt) that names the impression, campaign, item and user and
// carries the auction price. Billing charges only the price inside a token that verifies, so the
// client cannot change it, and a click that arrives after the token expired is not billed (§7.4).
//
// Keys come from AD_TOKEN_KEYS ("kid:secret[,kid:secret...]"). The first key signs; every listed
// key verifies, chosen by the token's kid header. To rotate, put the new key first and keep the
// old one listed for one TTL, then drop it.
package adtoken

import (
	"errors"
	"fmt"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"

	"github.com/heyemlee/recloop/serving/internal/events"
)

// TTL is how long after serving a click on a sponsored slot can be billed.
const TTL = 30 * time.Minute

// MinKeyBytes is the shortest accepted key: an HS256 key should be at least as long as the hash.
const MinKeyBytes = 32

// EnvKeys is the environment variable that holds the keys.
const EnvKeys = "AD_TOKEN_KEYS"

// Errors of Verify. Expired tokens are authentic: their claims are returned with the error.
var (
	ErrInvalid = errors.New("ad_token invalid")
	ErrExpired = errors.New("ad_token expired")
)

// Claims is what a token says about one sponsored slot.
type Claims struct {
	RequestID  uuid.UUID
	Position   int // 1-based position of the slot in the response
	CampaignID int64
	ItemID     int32
	PriceCents int32 // per-click price set by the auction: billing charges at most this
	UserID     int64
	IssuedAt   time.Time // when the slot was served (whole seconds)
	ExpiresAt  time.Time // IssuedAt + TTL
}

// ImpressionID is the impression the token belongs to: "{request_id}:{position}".
func (c Claims) ImpressionID() string { return events.ImpressionID(c.RequestID, c.Position) }

// Day is the billing day: the UTC day the slot was served. Every click on one impression is
// billed against the same day's budget, whenever it arrives.
func (c Claims) Day() time.Time { return events.Day(c.IssuedAt) }

// Matches reports whether a click event is for this token's impression, user and item. The
// impression id is compared parsed, so the spelling of the UUID does not matter; billing keys
// its ledger by the token's canonical ImpressionID, never by the client's string.
func (c Claims) Matches(impressionID string, userID int64, itemID int32) bool {
	rid, pos, err := events.ParseImpressionID(impressionID)
	return err == nil && rid == c.RequestID && pos == c.Position && userID == c.UserID && itemID == c.ItemID
}

// wire is the JWT payload; short names keep the token small.
type wire struct {
	RequestID  string `json:"rid"`
	Position   int    `json:"pos"`
	CampaignID int64  `json:"cid"`
	ItemID     int32  `json:"item"`
	PriceCents int32  `json:"price"`
	UserID     int64  `json:"uid"`
	jwt.RegisteredClaims
}

func (c Claims) check() error {
	switch {
	case c.RequestID == uuid.Nil:
		return errors.New("request id missing")
	case c.Position < 1:
		return errors.New("position must be >= 1")
	case c.CampaignID <= 0:
		return errors.New("campaign id must be > 0")
	case c.ItemID < 0:
		return errors.New("item id must be >= 0")
	case c.PriceCents <= 0:
		return errors.New("price must be > 0")
	case c.UserID < 0:
		return errors.New("user id must be >= 0")
	}
	return nil
}

func (w *wire) claims() (Claims, error) {
	rid, err := uuid.Parse(w.RequestID)
	if err != nil {
		return Claims{}, fmt.Errorf("rid: %w", err)
	}
	if w.IssuedAt == nil || w.ExpiresAt == nil || !w.ExpiresAt.After(w.IssuedAt.Time) {
		return Claims{}, errors.New("iat and exp are required, exp after iat")
	}
	c := Claims{
		RequestID: rid, Position: w.Position, CampaignID: w.CampaignID, ItemID: w.ItemID,
		PriceCents: w.PriceCents, UserID: w.UserID,
		IssuedAt: w.IssuedAt.UTC(), ExpiresAt: w.ExpiresAt.UTC(),
	}
	return c, c.check()
}

// Keys signs and verifies tokens.
type Keys struct {
	signKID string
	keys    map[string][]byte
}

var kidPattern = regexp.MustCompile(`^[A-Za-z0-9._-]{1,32}$`)

// ParseKeys reads "kid:secret[,kid:secret...]"; the first key signs. Secrets are used as given
// (e.g. the output of `openssl rand -base64 48`) and must be at least MinKeyBytes long.
func ParseKeys(spec string) (*Keys, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return nil, fmt.Errorf("%s is empty: set it to kid:secret", EnvKeys)
	}
	k := &Keys{keys: map[string][]byte{}}
	for i, part := range strings.Split(spec, ",") {
		kid, secret, ok := strings.Cut(strings.TrimSpace(part), ":")
		switch {
		case !ok:
			return nil, fmt.Errorf("%s entry %d is not kid:secret", EnvKeys, i+1)
		case !kidPattern.MatchString(kid):
			return nil, fmt.Errorf("%s entry %d: key id must match %s", EnvKeys, i+1, kidPattern)
		case len(secret) < MinKeyBytes:
			return nil, fmt.Errorf("%s key %q is %d bytes, want at least %d", EnvKeys, kid, len(secret), MinKeyBytes)
		case k.keys[kid] != nil:
			return nil, fmt.Errorf("%s lists key id %q twice", EnvKeys, kid)
		}
		k.keys[kid] = []byte(secret)
		if i == 0 {
			k.signKID = kid
		}
	}
	return k, nil
}

// KeysFromEnv reads AD_TOKEN_KEYS.
func KeysFromEnv() (*Keys, error) { return ParseKeys(os.Getenv(EnvKeys)) }

// SigningKeyID is the kid new tokens get.
func (k *Keys) SigningKeyID() string { return k.signKID }

// Sign issues the token of a slot served at now; IssuedAt and ExpiresAt in c are ignored.
func (k *Keys) Sign(c Claims, now time.Time) (string, error) {
	if err := c.check(); err != nil {
		return "", fmt.Errorf("ad_token claims: %w", err)
	}
	w := wire{
		RequestID: c.RequestID.String(), Position: c.Position, CampaignID: c.CampaignID,
		ItemID: c.ItemID, PriceCents: c.PriceCents, UserID: c.UserID,
		RegisteredClaims: jwt.RegisteredClaims{
			IssuedAt:  jwt.NewNumericDate(now),
			ExpiresAt: jwt.NewNumericDate(now.Add(TTL)),
		},
	}
	tok := jwt.NewWithClaims(jwt.SigningMethodHS256, w)
	tok.Header["kid"] = k.signKID
	return tok.SignedString(k.keys[k.signKID])
}

// Verify checks a token at time at: when the click reached the api, not when billing gets to
// it, so a click redelivered after a consumer outage gets the same answer as the first time.
// It returns ErrInvalid (with the reason) for anything that does not verify, and ErrExpired with
// the authentic claims for a token used at or after its expiry.
func (k *Keys) Verify(token string, at time.Time) (Claims, error) {
	var w wire
	p := jwt.NewParser(
		jwt.WithValidMethods([]string{jwt.SigningMethodHS256.Alg()}), // no "none", no other algs
		jwt.WithExpirationRequired(),
		jwt.WithStrictDecoding(), // canonical base64 only: no two strings for one signature
		jwt.WithTimeFunc(func() time.Time { return at }),
	)
	_, err := p.ParseWithClaims(token, &w, func(t *jwt.Token) (any, error) {
		kid, _ := t.Header["kid"].(string)
		key, ok := k.keys[kid]
		if !ok {
			return nil, fmt.Errorf("unknown key id %q", kid)
		}
		return key, nil
	})
	// Claims are validated only after the signature verified, so ErrTokenExpired implies an
	// authentic token.
	expired := err != nil && errors.Is(err, jwt.ErrTokenExpired)
	if err != nil && !expired {
		return Claims{}, fmt.Errorf("%w: %v", ErrInvalid, err)
	}
	c, cerr := w.claims()
	if cerr != nil {
		return Claims{}, fmt.Errorf("%w: %v", ErrInvalid, cerr)
	}
	if expired {
		return c, ErrExpired
	}
	return c, nil
}
