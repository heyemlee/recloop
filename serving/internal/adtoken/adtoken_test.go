package adtoken_test

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
	"pgregory.net/rapid"

	"github.com/heyemlee/recloop/serving/internal/adtoken"
)

const (
	key1 = "k1:0123456789abcdef0123456789abcdef-first"
	key2 = "k2:fedcba9876543210fedcba9876543210-second"
)

var served = time.Date(2026, 11, 6, 23, 50, 7, 123_000_000, time.UTC)

func keys(t testing.TB, spec string) *adtoken.Keys {
	t.Helper()
	k, err := adtoken.ParseKeys(spec)
	if err != nil {
		t.Fatal(err)
	}
	return k
}

func sample() adtoken.Claims {
	return adtoken.Claims{
		RequestID: uuid.MustParse("01932c4e-7d1a-7b3e-9f20-5c8e2a1b4d6f"), Position: 3,
		CampaignID: 42, ItemID: 812, PriceCents: 37, UserID: 1213,
	}
}

func sign(t testing.TB, k *adtoken.Keys, c adtoken.Claims, at time.Time) string {
	t.Helper()
	tok, err := k.Sign(c, at)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func TestRoundTrip(t *testing.T) {
	k := keys(t, key1)
	tok := sign(t, k, sample(), served)
	got, err := k.Verify(tok, served.Add(time.Minute))
	if err != nil {
		t.Fatal(err)
	}
	want := sample()
	want.IssuedAt = served.Truncate(time.Second)
	want.ExpiresAt = want.IssuedAt.Add(adtoken.TTL)
	if got != want {
		t.Fatalf("claims\n got %+v\nwant %+v", got, want)
	}
	if got.ImpressionID() != "01932c4e-7d1a-7b3e-9f20-5c8e2a1b4d6f:3" {
		t.Fatalf("impression id %q", got.ImpressionID())
	}
	// Served at 23:50 UTC: the billing day stays the 6th even for a click after midnight.
	if day := got.Day(); !day.Equal(time.Date(2026, 11, 6, 0, 0, 0, 0, time.UTC)) {
		t.Fatalf("day %v", day)
	}
	if !got.Matches(got.ImpressionID(), 1213, 812) || got.Matches(got.ImpressionID(), 1214, 812) ||
		got.Matches(got.ImpressionID(), 1213, 813) || got.Matches(got.RequestID.String()+":4", 1213, 812) ||
		got.Matches("garbage", 1213, 812) {
		t.Fatal("Matches")
	}
	if !got.Matches("01932C4E-7D1A-7B3E-9F20-5C8E2A1B4D6F:3", 1213, 812) {
		t.Fatal("an upper-case request id is the same impression")
	}
}

func TestExpiry(t *testing.T) {
	k := keys(t, key1)
	tok := sign(t, k, sample(), served)
	exp := served.Truncate(time.Second).Add(adtoken.TTL)
	for _, at := range []time.Time{served.Truncate(time.Second), exp.Add(-time.Nanosecond)} {
		if _, err := k.Verify(tok, at); err != nil {
			t.Fatalf("at %v: %v", at, err)
		}
	}
	for _, at := range []time.Time{exp, exp.Add(time.Second), exp.Add(48 * time.Hour)} {
		c, err := k.Verify(tok, at)
		if !errors.Is(err, adtoken.ErrExpired) {
			t.Fatalf("at %v: err %v, want ErrExpired", at, err)
		}
		if c.PriceCents != 37 || c.CampaignID != 42 { // an expired token is still authentic
			t.Fatalf("expired claims %+v", c)
		}
	}
}

// segments splits a token and decodes its payload into a map.
func payload(t *testing.T, tok string) (parts []string, claims map[string]any) {
	t.Helper()
	parts = strings.Split(tok, ".")
	raw, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &claims); err != nil {
		t.Fatal(err)
	}
	return parts, claims
}

func encode(t *testing.T, v any) string {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return base64.RawURLEncoding.EncodeToString(raw)
}

// Changing any claim invalidates the signature: the price, the expiry, who and what it is for.
func TestEditedClaimsAreRejected(t *testing.T) {
	k := keys(t, key1)
	tok := sign(t, k, sample(), served)
	edits := map[string]any{
		"price": 1, "cid": 43, "item": 813, "uid": 1, "pos": 9,
		"rid":   "01932c4e-7d1a-7b3e-9f20-5c8e2a1b4d70",
		"exp":   served.Add(24 * time.Hour).Unix(),
		"iat":   served.Add(time.Hour).Unix(),
		"extra": true,
	}
	for field, v := range edits {
		parts, claims := payload(t, tok)
		claims[field] = v
		forged := parts[0] + "." + encode(t, claims) + "." + parts[2]
		if _, err := k.Verify(forged, served.Add(time.Minute)); !errors.Is(err, adtoken.ErrInvalid) {
			t.Errorf("edited %s: err %v, want ErrInvalid", field, err)
		}
	}
	// The original still verifies (the edits above did not touch it).
	if _, err := k.Verify(tok, served.Add(time.Minute)); err != nil {
		t.Fatal(err)
	}
}

func TestForgedHeadersAndKeys(t *testing.T) {
	k := keys(t, key1+","+key2)
	at := served.Add(time.Minute)
	tok := sign(t, k, sample(), served)
	parts, claims := payload(t, tok)

	header := func(h map[string]any) string { return encode(t, h) }
	cases := map[string]string{
		// alg none with an empty signature
		"alg none": header(map[string]any{"alg": "none", "typ": "JWT", "kid": "k1"}) + "." + parts[1] + ".",
		// the right signature under another listed key id
		"kid swapped": header(map[string]any{"alg": "HS256", "typ": "JWT", "kid": "k2"}) + "." + parts[1] + "." + parts[2],
		"kid unknown": header(map[string]any{"alg": "HS256", "typ": "JWT", "kid": "k9"}) + "." + parts[1] + "." + parts[2],
		"kid missing": header(map[string]any{"alg": "HS256", "typ": "JWT"}) + "." + parts[1] + "." + parts[2],
		"empty":       "",
		"garbage":     "not-a-token",
		"two parts":   parts[0] + "." + parts[1],
		"truncated":   tok[:len(tok)-1],
	}
	// Signed by someone else under our key id, and with a stronger alg we do not accept.
	other := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims(claims))
	other.Header["kid"] = "k1"
	cases["attacker key"], _ = other.SignedString([]byte("attacker-secret-attacker-secret-attacker"))
	hs512 := jwt.NewWithClaims(jwt.SigningMethodHS512, jwt.MapClaims(claims))
	hs512.Header["kid"] = "k1"
	cases["HS512"], _ = hs512.SignedString([]byte(strings.TrimPrefix(key1, "k1:")))

	for name, forged := range cases {
		if _, err := k.Verify(forged, at); !errors.Is(err, adtoken.ErrInvalid) {
			t.Errorf("%s: err %v, want ErrInvalid", name, err)
		}
	}
}

// Rotation: a token signed with the old key verifies while the old key is still listed.
func TestKeyRotation(t *testing.T) {
	old := keys(t, key1)
	tok := sign(t, old, sample(), served)
	rotated := keys(t, key2+","+key1)
	if rotated.SigningKeyID() != "k2" {
		t.Fatalf("signing kid %q", rotated.SigningKeyID())
	}
	if _, err := rotated.Verify(tok, served.Add(time.Minute)); err != nil {
		t.Fatalf("old token after rotation: %v", err)
	}
	if _, err := old.Verify(sign(t, rotated, sample(), served), served.Add(time.Minute)); !errors.Is(err, adtoken.ErrInvalid) {
		t.Fatalf("new-key token on the old keyring: %v", err)
	}
	if _, err := keys(t, key2).Verify(tok, served.Add(time.Minute)); !errors.Is(err, adtoken.ErrInvalid) {
		t.Fatalf("old key dropped: %v", err)
	}
}

func TestParseKeys(t *testing.T) {
	for _, spec := range []string{
		"", "   ", "k1", "k1:short", ":0123456789abcdef0123456789abcdef",
		"bad kid:0123456789abcdef0123456789abcdef",
		key1 + "," + key1,
		key1 + ",",
	} {
		if _, err := adtoken.ParseKeys(spec); err == nil {
			t.Errorf("ParseKeys(%q) accepted", spec)
		}
	}
	if _, err := adtoken.ParseKeys(" " + key1 + " , " + key2 + " "); err != nil {
		t.Fatal(err)
	}
}

func TestSignRejectsNonsense(t *testing.T) {
	k := keys(t, key1)
	for name, edit := range map[string]func(*adtoken.Claims){
		"zero price":    func(c *adtoken.Claims) { c.PriceCents = 0 },
		"position 0":    func(c *adtoken.Claims) { c.Position = 0 },
		"no campaign":   func(c *adtoken.Claims) { c.CampaignID = 0 },
		"no request id": func(c *adtoken.Claims) { c.RequestID = uuid.Nil },
		"negative user": func(c *adtoken.Claims) { c.UserID = -1 },
	} {
		c := sample()
		edit(&c)
		if _, err := k.Sign(c, served); err == nil {
			t.Errorf("%s: signed", name)
		}
	}
}

const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_."

// Property: a token with any one character replaced, or cut short, never verifies. Strict
// base64 decoding matters here: without it the last character of the signature has two bits
// that can change without changing the decoded signature.
func TestAnyEditIsRejected(t *testing.T) {
	k := keys(t, key1)
	rapid.Check(t, func(t *rapid.T) {
		c := adtoken.Claims{
			RequestID:  uuid.Must(uuid.NewV7()),
			Position:   rapid.IntRange(1, 200).Draw(t, "pos"),
			CampaignID: rapid.Int64Range(1, 1<<40).Draw(t, "cid"),
			ItemID:     rapid.Int32Range(0, 3458).Draw(t, "item"),
			PriceCents: rapid.Int32Range(1, 10_000).Draw(t, "price"),
			UserID:     rapid.Int64Range(0, 1<<40).Draw(t, "uid"),
		}
		tok, err := k.Sign(c, served)
		if err != nil {
			t.Fatal(err)
		}
		// Any time inside the window: iat and exp are whole seconds.
		at := served.Truncate(time.Second).Add(time.Duration(rapid.Int64Range(0, int64(adtoken.TTL)-1).Draw(t, "delay")))
		if _, err := k.Verify(tok, at); err != nil {
			t.Fatalf("untouched token: %v", err)
		}
		i := rapid.IntRange(0, len(tok)-1).Draw(t, "index")
		r := rapid.SampledFrom([]byte(alphabet)).Filter(func(b byte) bool { return b != tok[i] }).Draw(t, "char")
		edited := tok[:i] + string(r) + tok[i+1:]
		if _, err := k.Verify(edited, at); !errors.Is(err, adtoken.ErrInvalid) {
			t.Fatalf("token with char %d replaced by %q: err %v", i, r, err)
		}
		if _, err := k.Verify(tok[:i], at); !errors.Is(err, adtoken.ErrInvalid) {
			t.Fatalf("token cut at %d: err %v", i, err)
		}
	})
}
