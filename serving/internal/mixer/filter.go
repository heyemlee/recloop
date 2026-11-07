package mixer

// Exclusions are the per-request filters (design §4.4).
type Exclusions struct {
	Bought  map[int32]bool   // already purchased
	Seen    map[int32]bool   // shown in the last 24 hours
	Seed    int32            // the current item on the item surface (-1: none)
	Known   func(int32) bool // item exists in the catalog
	InStock func(int32) bool
}

// FilterStats counts what each filter removed.
type FilterStats struct {
	Bought  int `json:"bought"`
	Seen    int `json:"seen"`
	SoldOut int `json:"sold_out"`
	Seed    int `json:"seed"`
	Unknown int `json:"unknown"`
}

// reason returns why the item is excluded ("" if it is not).
func (ex Exclusions) reason(id int32) string {
	switch {
	case ex.Known != nil && !ex.Known(id):
		return "unknown"
	case id == ex.Seed:
		return "seed"
	case ex.Bought[id]:
		return "bought"
	case ex.Seen[id]:
		return "seen"
	case ex.InStock != nil && !ex.InStock(id):
		return "sold_out"
	}
	return ""
}

func (s *FilterStats) add(reason string) {
	switch reason {
	case "unknown":
		s.Unknown++
	case "seed":
		s.Seed++
	case "bought":
		s.Bought++
	case "seen":
		s.Seen++
	case "sold_out":
		s.SoldOut++
	}
}

// Filter drops excluded items, keeping the order.
func Filter(in []Fused, ex Exclusions) ([]Fused, FilterStats) {
	var st FilterStats
	out := make([]Fused, 0, len(in))
	for _, f := range in {
		if r := ex.reason(f.ItemID); r != "" {
			st.add(r)
			continue
		}
		out = append(out, f)
	}
	return out, st
}

// Diversify enforces "at most maxPerBrand items of one brand in the first window positions"
// (design §4.4: 2 in the top 10). Positions are filled greedily with the best remaining item
// whose brand still has room; items passed over keep their order after the window. Items with
// an unknown brand ("") are not limited. Returns the new order and how many positions the cap
// changed (a lower-ranked item promoted over a capped one).
func Diversify(in []Fused, brandOf func(int32) string, window, maxPerBrand int) ([]Fused, int) {
	out := make([]Fused, 0, len(in))
	used := make([]bool, len(in))
	count := map[string]int{}
	promoted := 0
	for len(out) < window && len(out) < len(in) {
		first, pick := -1, -1
		for i, f := range in {
			if used[i] {
				continue
			}
			if first < 0 {
				first = i
			}
			if b := brandOf(f.ItemID); b == "" || count[b] < maxPerBrand {
				pick = i
				break
			}
		}
		if pick < 0 { // every remaining item is from a full brand: keep the best one
			pick = first
		}
		if pick != first {
			promoted++
		}
		used[pick] = true
		if b := brandOf(in[pick].ItemID); b != "" {
			count[b]++
		}
		out = append(out, in[pick])
	}
	for i, f := range in {
		if !used[i] {
			out = append(out, f)
		}
	}
	return out, promoted
}

// Backfill appends popular items (best first) that pass the exclusions and are not in the
// list yet, until the list has `want` items. Returns the list and the number added.
func Backfill(list []Fused, popular []int32, want int, ex Exclusions) ([]Fused, int) {
	if len(list) >= want {
		return list, 0
	}
	present := make(map[int32]bool, len(list))
	for _, f := range list {
		present[f.ItemID] = true
	}
	added := 0
	for rank, id := range popular {
		if len(list) >= want {
			break
		}
		if present[id] || ex.reason(id) != "" {
			continue
		}
		list = append(list, Fused{
			ItemID: id, Primary: "backfill",
			Hits: []Hit{{Source: "backfill", Rank: rank + 1}},
		})
		present[id] = true
		added++
	}
	return list, added
}
