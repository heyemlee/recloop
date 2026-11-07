package cache

import (
	"encoding/json"
	"reflect"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/heyemlee/recloop/serving/internal/experiment"
	"github.com/heyemlee/recloop/serving/internal/mixer"
)

func TestCursorRoundTrip(t *testing.T) {
	c := Cursor{RequestID: uuid.Must(uuid.NewV7()), Offset: 40}
	got, err := DecodeCursor(c.Encode())
	if err != nil || got != c {
		t.Fatalf("got %+v, %v", got, err)
	}
	for _, bad := range []string{"", "!!", "bm90LWEtdXVpZDoz", Cursor{RequestID: uuid.New(), Offset: 0}.Encode()} {
		if _, err := DecodeCursor(bad); err == nil {
			t.Errorf("cursor %q should be rejected", bad)
		}
	}
}

func sampleList(n int) []mixer.Fused {
	out := make([]mixer.Fused, n)
	for i := range out {
		out[i] = mixer.Fused{
			ItemID: int32(3458 - i), Score: 1.0 / float64(61+i), Primary: []string{"sasrec", "itemcf", "pop"}[i%3],
			Hits: []mixer.Hit{{Source: "sasrec", Rank: i + 1, Score: 0.5 - float64(i)/1000}, {Source: "itemcf", Rank: 2*i + 1, Score: 3.25, Via: int32(i)}},
		}
	}
	out[1].Hits = nil
	out[2].Primary = "backfill"
	return out
}

func TestCandidatesRoundTrip(t *testing.T) {
	in := &Candidates{Items: sampleList(CandidateDepth), Versions: map[string]string{"sasrec": "v1", "itemcf": "v1"}}
	raw := EncodeCandidates(in)
	out, err := DecodeCandidates(raw)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(in, out) {
		t.Fatal("candidates changed in the round trip")
	}
	if len(raw) > 16<<10 {
		t.Fatalf("%d bytes for %d candidates", len(raw), CandidateDepth)
	}
	for i := 0; i < len(raw); i += 7 {
		if _, err := DecodeCandidates(raw[:i]); err == nil {
			t.Fatalf("truncated entry (%d bytes) decoded", i)
		}
	}
}

func TestPageRoundTripAndM1Pages(t *testing.T) {
	pg := &Page{RequestID: uuid.Must(uuid.NewV7()), UserID: 7, Surface: "home", Seed: -1, ModelVer: "v",
		Exp:           map[string]experiment.Assignment{"recall": {Bucket: 3, ExpID: "e", Variant: "control"}},
		ConfigVersion: 2, Degraded: []string{"sasrec_timeout"}, HV: 4, Items: sampleList(200),
		CreatedAt: time.Now().UTC().Truncate(time.Millisecond)}
	raw, err := EncodePage(pg)
	if err != nil {
		t.Fatal(err)
	}
	back, err := DecodePage(raw)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(pg, back) {
		t.Fatalf("page changed in the round trip:\n%+v\n%+v", pg, back)
	}
	legacy, _ := json.Marshal(pg)
	if old, err := DecodePage(legacy); err != nil || len(old.Items) != 200 || old.RequestID != pg.RequestID {
		t.Fatalf("M1 JSON page: %v", err)
	}
}

func TestCandidateKeys(t *testing.T) {
	if k := CandidateKey(42, "home", -1, "control.default", "v9", 17); k != "cand:42:home:control.default:v9:17" {
		t.Fatal(k)
	}
	// The item page's candidates do not depend on the user: one entry per item.
	if a, b := CandidateKey(1, "item", 812, "c", "v9", 3), CandidateKey(2, "item", 812, "c", "v9", 8); a != b || a != "cand:item:812:c:v9" {
		t.Fatal(a, b)
	}
}
