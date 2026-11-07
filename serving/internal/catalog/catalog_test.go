package catalog

import (
	"context"
	"io"
	"log/slog"
	"testing"
)

type fakeSource struct {
	items []Item
	stock []int32
}

func (f fakeSource) Items(context.Context) ([]Item, []int32, error) { return f.items, f.stock, nil }

func (f fakeSource) Stock(context.Context) (map[int32]int32, error) {
	out := map[int32]int32{}
	for i, s := range f.stock {
		out[int32(i)] = s
	}
	return out, nil
}

func TestLoadAndStock(t *testing.T) {
	src := fakeSource{
		items: []Item{{ID: 0, Title: "a", Brand: "X"}, {ID: 1, Title: "b"}, {ID: 2, Title: "c", Brand: "Y"}},
		stock: []int32{3, 0, 1},
	}
	c, err := Load(context.Background(), src, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatal(err)
	}
	if c.Len() != 3 || !c.Has(2) || c.Has(3) || c.Has(-1) {
		t.Fatal("ids")
	}
	if c.InStock(1) || !c.InStock(0) || c.Brand(2) != "Y" || c.Brand(1) != "" || c.Stock(9) != 0 {
		t.Fatal("stock / brand")
	}
	c.SetStock(2, 0)
	if c.InStock(2) {
		t.Fatal("SetStock")
	}
	if it, ok := c.Get(0); !ok || it.Title != "a" {
		t.Fatalf("Get: %+v", it)
	}
}

func TestLoadRequiresDenseIDs(t *testing.T) {
	src := fakeSource{items: []Item{{ID: 0}, {ID: 2}}, stock: []int32{1, 1}}
	if _, err := Load(context.Background(), src, slog.New(slog.NewTextHandler(io.Discard, nil))); err == nil {
		t.Fatal("non-dense ids accepted")
	}
}
