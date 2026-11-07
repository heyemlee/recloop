// Package ads is the sponsored-slot logic of a recommendation request (docs/design.md §9.2):
// the auction that ranks eligible ads by expected revenue and prices each winner with a
// generalised second price, and the pCTR interface it needs. Eligibility (schedule, budget
// margin, frequency cap), the slots of the feed and the ad_token are wired in by the mixer.
package ads

import (
	"cmp"
	"math"
	"slices"
)

// PPM is the resolution of a pCTR inside the auction (parts per million). Quantising first
// makes ranking and pricing exact integer arithmetic: no float rounding can flip two ads or
// move a price by a cent.
const PPM = 1_000_000

// Candidate is one eligible ad: an item of a campaign, the campaign's bid and its predicted CTR.
type Candidate struct {
	CampaignID   int64
	AdvertiserID int32
	ItemID       int32
	BidCents     int32   // max CPC
	PCTR         float64 // predicted click-through rate at the reference position, in [0, 1]
}

// Config of one auction.
type Config struct {
	Slots        int     // sponsored positions to fill
	ReserveCents int32   // reserve price per click: lower bids do not take part, winners pay at least this
	MinPCTR      float64 // quality floor (§9.2 step 4): a high bid cannot buy an irrelevant slot
}

// Winner is an ad that won a slot.
type Winner struct {
	Candidate
	Rank       int   // 1 = the best slot
	PCTRPPM    int64 // pCTR as ranked, in parts per million
	ECPM       int64 // bid × pCTR per thousand impressions, in thousandths of a cent (bid × PCTRPPM)
	PriceCents int32 // charged per click: ReserveCents <= PriceCents <= BidCents
}

// QuantizePCTR maps a probability to parts per million (rounded, clamped to [0, PPM]).
func QuantizePCTR(p float64) int64 {
	if math.IsNaN(p) || p <= 0 {
		return 0
	}
	if p >= 1 {
		return PPM
	}
	return int64(math.Round(p * PPM))
}

type entry struct {
	Candidate
	q, ecpm int64
}

// better orders entries: higher eCPM, then higher pCTR (the more relevant ad), then lower
// campaign and item id, so equal inputs always give the same ranking.
func better(a, b entry) int {
	if c := cmp.Compare(b.ecpm, a.ecpm); c != 0 {
		return c
	}
	if c := cmp.Compare(b.q, a.q); c != 0 {
		return c
	}
	if c := cmp.Compare(a.CampaignID, b.CampaignID); c != 0 {
		return c
	}
	return cmp.Compare(a.ItemID, b.ItemID)
}

// Run ranks the candidates by eCPM = bid × pCTR and fills up to cfg.Slots slots. An advertiser
// takes at most one slot (its best ad), so it never bids against itself.
//
// Price (GSP): the winner of slot k pays the smallest whole-cent bid that would still rank it
// above the ad at k+1, i.e. floor(eCPM(k+1) / pCTR(k)) + 1¢, the design's eCPM(k+1) / pCTR(k)
// + 1¢ rounded down to a cent. It is never above its own bid nor below the reserve; the last
// ad, with nobody below it, pays the reserve. The ad at k+1 may be one that did not win a slot.
func Run(cands []Candidate, cfg Config) []Winner {
	if cfg.Slots <= 0 {
		return nil
	}
	minQ := max(QuantizePCTR(cfg.MinPCTR), 1)
	best := map[int32]entry{} // advertiser -> its best eligible ad
	for _, c := range cands {
		q := QuantizePCTR(c.PCTR)
		if c.BidCents <= 0 || c.BidCents < cfg.ReserveCents || q < minQ {
			continue
		}
		e := entry{Candidate: c, q: q, ecpm: int64(c.BidCents) * q}
		if cur, ok := best[c.AdvertiserID]; !ok || better(e, cur) < 0 {
			best[c.AdvertiserID] = e
		}
	}
	ranked := make([]entry, 0, len(best))
	for _, e := range best {
		ranked = append(ranked, e)
	}
	slices.SortFunc(ranked, better)

	n := min(cfg.Slots, len(ranked))
	out := make([]Winner, n)
	for k := range n {
		e := ranked[k]
		price := cfg.ReserveCents
		if k+1 < len(ranked) {
			price = max(price, int32(ranked[k+1].ecpm/e.q+1))
		}
		price = min(price, e.BidCents)
		out[k] = Winner{Candidate: e.Candidate, Rank: k + 1, PCTRPPM: e.q, ECPM: e.ecpm, PriceCents: price}
	}
	return out
}
