// Package catalog keeps the whole item catalog in process (design §4.5: 3.5K items is small).
// Item metadata is an immutable snapshot swapped on refresh; stock is a separate array of
// atomics so purchases are visible immediately and the periodic refresh corrects drift.
package catalog

import (
	"context"
	"fmt"
	"log/slog"
	"sync/atomic"
	"time"
)

// Item is what the api shows and filters on.
type Item struct {
	ID          int32    `json:"item_id"`
	Title       string   `json:"title"`
	Brand       string   `json:"brand,omitempty"`
	Categories  []string `json:"categories,omitempty"`
	PriceCents  *int32   `json:"price_cents"` // nil = price unknown
	ImageURL    string   `json:"image_url,omitempty"`
	Description string   `json:"description,omitempty"`
	Features    []string `json:"features,omitempty"`
}

// Source loads the catalog (PostgreSQL in production).
type Source interface {
	Items(ctx context.Context) ([]Item, []int32, error) // items and their stock, both indexed by item id
	Stock(ctx context.Context) (map[int32]int32, error)
}

// Catalog is safe for concurrent use.
type Catalog struct {
	src   Source
	items atomic.Pointer[[]Item]
	stock []atomic.Int32
	log   *slog.Logger
}

// Load reads the catalog once; item ids must be dense (0..n-1).
func Load(ctx context.Context, src Source, log *slog.Logger) (*Catalog, error) {
	items, stock, err := src.Items(ctx)
	if err != nil {
		return nil, err
	}
	for i, it := range items {
		if it.ID != int32(i) {
			return nil, fmt.Errorf("catalog: item ids are not dense: position %d has id %d", i, it.ID)
		}
	}
	c := &Catalog{src: src, stock: make([]atomic.Int32, len(items)), log: log}
	c.items.Store(&items)
	for i, s := range stock {
		c.stock[i].Store(s)
	}
	return c, nil
}

// New builds a catalog from memory (tests).
func New(items []Item, stock []int32) *Catalog {
	c := &Catalog{stock: make([]atomic.Int32, len(items))}
	c.items.Store(&items)
	for i, s := range stock {
		c.stock[i].Store(s)
	}
	return c
}

// Len is the number of items.
func (c *Catalog) Len() int { return len(*c.items.Load()) }

// Get returns the item with this id.
func (c *Catalog) Get(id int32) (Item, bool) {
	items := *c.items.Load()
	if id < 0 || int(id) >= len(items) {
		return Item{}, false
	}
	return items[id], true
}

// Has reports whether id is a known item.
func (c *Catalog) Has(id int32) bool { return id >= 0 && int(id) < c.Len() }

// Brand of an item ("" if unknown).
func (c *Catalog) Brand(id int32) string {
	it, _ := c.Get(id)
	return it.Brand
}

// Stock of an item (0 for unknown ids).
func (c *Catalog) Stock(id int32) int32 {
	if !c.Has(id) {
		return 0
	}
	return c.stock[id].Load()
}

// InStock reports whether the item can be recommended.
func (c *Catalog) InStock(id int32) bool { return c.Stock(id) > 0 }

// SetStock records a stock level reported by the database (after a purchase).
func (c *Catalog) SetStock(id int32, stock int32) {
	if c.Has(id) {
		c.stock[id].Store(stock)
	}
}

// Run refreshes stock every `every` and metadata every `metaEvery` until ctx is done.
func (c *Catalog) Run(ctx context.Context, every, metaEvery time.Duration) {
	t := time.NewTicker(every)
	defer t.Stop()
	lastMeta := time.Now()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
		if time.Since(lastMeta) >= metaEvery {
			items, stock, err := c.src.Items(ctx)
			if err == nil && len(items) == c.Len() {
				c.items.Store(&items)
				for i, s := range stock {
					c.stock[i].Store(s)
				}
				lastMeta = time.Now()
				continue
			}
			c.log.Warn("catalog refresh failed", "err", err, "items", len(items))
		}
		stock, err := c.src.Stock(ctx)
		if err != nil {
			c.log.Warn("stock refresh failed", "err", err)
			continue
		}
		for id, s := range stock {
			c.SetStock(id, s)
		}
	}
}
