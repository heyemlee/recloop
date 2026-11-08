package api

import (
	"testing"

	"github.com/heyemlee/recloop/serving/internal/cache"
	"github.com/heyemlee/recloop/serving/internal/mixer"
)

func ids(list []mixer.Fused) []int32 {
	out := make([]int32, len(list))
	for i, f := range list {
		out[i] = f.ItemID
	}
	return out
}

func TestInsertSponsored(t *testing.T) {
	organic := func(n int) []mixer.Fused {
		out := make([]mixer.Fused, n)
		for i := range out {
			out[i] = mixer.Fused{ItemID: int32(i + 1), Primary: "pop"}
		}
		return out
	}
	slots := []cache.Slot{{Position: 3, ItemID: 100}, {Position: 9, ItemID: 200}}
	got := insertSponsored(organic(10), slots, 200)
	want := []int32{1, 2, 100, 3, 4, 5, 6, 7, 200, 8, 9, 10}
	if len(got) != len(want) {
		t.Fatalf("got %v", ids(got))
	}
	for i := range want {
		if got[i].ItemID != want[i] {
			t.Fatalf("got %v, want %v", ids(got), want)
		}
	}
	if got[2].Primary != SourceAds || got[8].Primary != SourceAds {
		t.Fatal("sponsored items are not marked")
	}
	// The page keeps `depth` items: the last organic ones drop.
	if got := insertSponsored(organic(10), slots, 10); len(got) != 10 || got[9].ItemID != 8 {
		t.Fatalf("depth: %v", ids(got))
	}
	if got := insertSponsored(organic(4), slots[:1], 200); len(got) != 5 || got[2].ItemID != 100 {
		t.Fatalf("one slot: %v", ids(got))
	}
	if got := insertSponsored(organic(3), nil, 200); len(got) != 3 {
		t.Fatalf("no slots: %v", ids(got))
	}
}
