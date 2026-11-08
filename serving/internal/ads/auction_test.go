package ads_test

import (
	"context"
	"math"
	"testing"

	"pgregory.net/rapid"

	"github.com/heyemlee/recloop/serving/internal/ads"
)

type win struct {
	campaign int64
	price    int32
}

func run(cands []ads.Candidate, cfg ads.Config) []win {
	var out []win
	for _, w := range ads.Run(cands, cfg) {
		out = append(out, win{w.CampaignID, w.PriceCents})
	}
	return out
}

func ad(campaign int64, advertiser int32, bid int32, pctr float64) ads.Candidate {
	return ads.Candidate{CampaignID: campaign, AdvertiserID: advertiser, ItemID: int32(campaign), BidCents: bid, PCTR: pctr}
}

// Hand-computed prices: floor(eCPM(k+1) / pCTR(k)) + 1, within [reserve, bid].
func TestGSPExamples(t *testing.T) {
	cases := []struct {
		name  string
		cands []ads.Candidate
		cfg   ads.Config
		want  []win
	}{
		{
			// eCPM: A 100 × 0.02 = 2.0, B 80 × 0.03 = 2.4 → B first. B pays 2.0/0.03 = 66.7 → 67.
			name:  "higher pCTR beats higher bid",
			cands: []ads.Candidate{ad(1, 1, 100, 0.02), ad(2, 2, 80, 0.03)},
			cfg:   ads.Config{Slots: 2, ReserveCents: 5, MinPCTR: 0.001},
			want:  []win{{2, 67}, {1, 5}}, // the last slot has nobody below it: reserve
		},
		{
			// A 50 × 0.04 = 2.0, B 100 × 0.01 = 1.0, C 30 × 0.02 = 0.6. A pays 1.0/0.04 = 25 → 26;
			// B pays 0.6/0.01 = 60 → 61, set by C, which wins nothing.
			name:  "a loser prices the last winner",
			cands: []ads.Candidate{ad(3, 3, 30, 0.02), ad(2, 2, 100, 0.01), ad(1, 1, 50, 0.04)},
			cfg:   ads.Config{Slots: 2, ReserveCents: 5, MinPCTR: 0.001},
			want:  []win{{1, 26}, {2, 61}},
		},
		{
			// Exact ratio 20: one cent above it.
			name:  "integer ratio plus one cent",
			cands: []ads.Candidate{ad(1, 1, 40, 0.025), ad(2, 2, 20, 0.025)},
			cfg:   ads.Config{Slots: 1, ReserveCents: 5, MinPCTR: 0.001},
			want:  []win{{1, 21}},
		},
		{
			// Reserve 30: B's bid of 20 cannot take part, so A has nobody below it.
			name:  "reserve filters and floors",
			cands: []ads.Candidate{ad(1, 1, 40, 0.025), ad(2, 2, 20, 0.025)},
			cfg:   ads.Config{Slots: 2, ReserveCents: 30, MinPCTR: 0.001},
			want:  []win{{1, 30}},
		},
		{
			// Equal eCPM and pCTR: the lower campaign id ranks first and pays its own bid (cap).
			name:  "tie pays own bid",
			cands: []ads.Candidate{ad(2, 2, 40, 0.025), ad(1, 1, 40, 0.025)},
			cfg:   ads.Config{Slots: 1, ReserveCents: 5, MinPCTR: 0.001},
			want:  []win{{1, 40}},
		},
		{
			// 1000 × 0.001 = 1.0 would rank first, but 0.001 is under the quality floor.
			name:  "quality floor",
			cands: []ads.Candidate{ad(1, 1, 1000, 0.001), ad(2, 2, 30, 0.02)},
			cfg:   ads.Config{Slots: 2, ReserveCents: 5, MinPCTR: 0.005},
			want:  []win{{2, 5}},
		},
		{
			// Advertiser 1 has two campaigns: only its best runs, so it does not raise its own
			// price (51 from advertiser 2, not 91 from its second campaign).
			name:  "one slot per advertiser",
			cands: []ads.Candidate{ad(1, 1, 100, 0.02), ad(2, 1, 90, 0.02), ad(3, 2, 50, 0.02)},
			cfg:   ads.Config{Slots: 2, ReserveCents: 5, MinPCTR: 0.001},
			want:  []win{{1, 51}, {3, 5}},
		},
		{
			name:  "no slots",
			cands: []ads.Candidate{ad(1, 1, 100, 0.02)},
			cfg:   ads.Config{Slots: 0, ReserveCents: 5},
		},
		{
			name:  "NaN and zero pCTR do not run",
			cands: []ads.Candidate{ad(1, 1, 100, math.NaN()), ad(2, 2, 100, 0)},
			cfg:   ads.Config{Slots: 2, ReserveCents: 5},
		},
	}
	for _, tc := range cases {
		got := run(tc.cands, tc.cfg)
		if len(got) != len(tc.want) {
			t.Errorf("%s: got %v, want %v", tc.name, got, tc.want)
			continue
		}
		for i := range got {
			if got[i] != tc.want[i] {
				t.Errorf("%s: got %v, want %v", tc.name, got, tc.want)
				break
			}
		}
	}
}

func TestQuantizePCTR(t *testing.T) {
	for p, want := range map[float64]int64{
		-0.1: 0, 0: 0, 0.0000004: 0, 0.0000005: 1, 0.0123456: 12346, 0.5: 500_000, 1: ads.PPM, 7: ads.PPM,
	} {
		if got := ads.QuantizePCTR(p); got != want {
			t.Errorf("QuantizePCTR(%v) = %d, want %d", p, got, want)
		}
	}
	if ads.QuantizePCTR(math.NaN()) != 0 {
		t.Error("NaN")
	}
}

func TestFixedPCTR(t *testing.T) {
	got, err := ads.FixedPCTR(0.02).PredictCTR(context.Background(), ads.PCTRRequest{ItemIDs: []int32{1, 2, 3}})
	if err != nil || len(got.CTR) != 3 || got.CTR[2] != 0.02 || got.Model != ads.PCTRFixed || got.Features != nil {
		t.Fatalf("%v %v", got, err)
	}
	if _, err := ads.FixedPCTR(1.5).PredictCTR(context.Background(), ads.PCTRRequest{}); err == nil {
		t.Fatal("pCTR 1.5 accepted")
	}
}

type auction struct {
	cands []ads.Candidate
	cfg   ads.Config
}

func drawAuction(t *rapid.T) auction {
	n := rapid.IntRange(0, 12).Draw(t, "n")
	cands := make([]ads.Candidate, n)
	for i := range cands {
		cands[i] = ads.Candidate{
			CampaignID:   int64(i + 1),
			AdvertiserID: rapid.Int32Range(1, 6).Draw(t, "advertiser"),
			ItemID:       rapid.Int32Range(0, 50).Draw(t, "item"),
			BidCents:     rapid.Int32Range(1, 500).Draw(t, "bid"),
			PCTR:         rapid.Float64Range(0, 0.2).Draw(t, "pctr"),
		}
	}
	return auction{cands, ads.Config{
		Slots:        rapid.IntRange(1, 4).Draw(t, "slots"),
		ReserveCents: rapid.Int32Range(1, 60).Draw(t, "reserve"),
		MinPCTR:      rapid.Float64Range(0, 0.02).Draw(t, "minpctr"),
	}}
}

func eligible(c ads.Candidate, cfg ads.Config) bool {
	return c.BidCents >= cfg.ReserveCents && ads.QuantizePCTR(c.PCTR) >= max(ads.QuantizePCTR(cfg.MinPCTR), 1)
}

// Properties of every auction outcome.
func TestAuctionProperties(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		a := drawAuction(t)
		ws := ads.Run(a.cands, a.cfg)

		advertisers := map[int32]bool{}
		for _, c := range a.cands {
			if eligible(c, a.cfg) {
				advertisers[c.AdvertiserID] = true
			}
		}
		if len(ws) != min(a.cfg.Slots, len(advertisers)) {
			t.Fatalf("%d winners, %d slots, %d eligible advertisers", len(ws), a.cfg.Slots, len(advertisers))
		}
		won := map[int32]bool{}
		for k, w := range ws {
			if w.Rank != k+1 || won[w.AdvertiserID] || !eligible(w.Candidate, a.cfg) {
				t.Fatalf("winner %d: %+v (advertiser seen: %v)", k, w, won[w.AdvertiserID])
			}
			won[w.AdvertiserID] = true
			if w.PriceCents < a.cfg.ReserveCents || w.PriceCents > w.BidCents {
				t.Fatalf("price %d outside [reserve %d, bid %d]", w.PriceCents, a.cfg.ReserveCents, w.BidCents)
			}
			if k > 0 && ws[k-1].ECPM < w.ECPM {
				t.Fatalf("not sorted by eCPM: %d before %d", ws[k-1].ECPM, w.ECPM)
			}
		}
		// The eCPM below each winner: the next winner, or for the last one the best ad of an
		// advertiser that won nothing (-1 if there is none).
		bestLoser := int64(-1)
		for _, c := range a.cands {
			if eligible(c, a.cfg) && !won[c.AdvertiserID] {
				bestLoser = max(bestLoser, int64(c.BidCents)*ads.QuantizePCTR(c.PCTR))
			}
		}
		for k, w := range ws {
			if bestLoser > w.ECPM {
				t.Fatalf("a loser (eCPM %d) beats winner %d (eCPM %d)", bestLoser, k, w.ECPM)
			}
			next := bestLoser
			if k+1 < len(ws) {
				next = ws[k+1].ECPM
			}
			if next < 0 {
				if w.PriceCents != a.cfg.ReserveCents && w.PriceCents != w.BidCents {
					t.Fatalf("nobody below winner %d, price %d is neither reserve nor bid", k, w.PriceCents)
				}
				continue
			}
			p := int64(w.PriceCents)
			// The price keeps the slot (strictly, unless capped by the bid) and a cent less
			// would not (unless the reserve set it).
			if w.PriceCents < w.BidCents && p*w.PCTRPPM <= next {
				t.Fatalf("winner %d: price %d × %d does not beat %d", k, p, w.PCTRPPM, next)
			}
			if w.PriceCents > a.cfg.ReserveCents && (p-1)*w.PCTRPPM > next {
				t.Fatalf("winner %d: price %d is not the smallest (next eCPM %d)", k, p, next)
			}
		}
	})
}

// GSP: what a winner pays does not depend on its own bid, as long as it keeps its slot.
func TestPriceDoesNotDependOnOwnBid(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		a := drawAuction(t)
		ws := ads.Run(a.cands, a.cfg)
		if len(ws) == 0 {
			t.Skip("no winner")
		}
		w := ws[rapid.IntRange(0, len(ws)-1).Draw(t, "winner")]
		if w.PriceCents == w.BidCents {
			t.Skip("price capped by the bid")
		}
		raised := append([]ads.Candidate(nil), a.cands...)
		for i := range raised {
			if raised[i].CampaignID == w.CampaignID {
				raised[i].BidCents += rapid.Int32Range(1, 1000).Draw(t, "raise")
			}
		}
		for _, v := range ads.Run(raised, a.cfg) {
			if v.CampaignID == w.CampaignID && v.Rank == w.Rank && v.PriceCents != w.PriceCents {
				t.Fatalf("same slot %d, price %d -> %d after raising the bid", w.Rank, w.PriceCents, v.PriceCents)
			}
		}
	})
}

// The outcome does not depend on the order of the candidates.
func TestAuctionIsOrderIndependent(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		a := drawAuction(t)
		shuffled := rapid.Permutation(a.cands).Draw(t, "order")
		want, got := run(a.cands, a.cfg), run(shuffled, a.cfg)
		if len(want) != len(got) {
			t.Fatalf("%v vs %v", want, got)
		}
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("%v vs %v", want, got)
			}
		}
	})
}
