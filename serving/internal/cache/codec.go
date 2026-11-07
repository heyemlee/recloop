package cache

import (
	"encoding/binary"
	"errors"
	"math"

	"github.com/heyemlee/recloop/serving/internal/mixer"
)

// Binary encoding of a fused list, shared by the candidate cache and the pages. JSON of 300
// items with their per-source hits is ~60 KB and takes a few hundred microseconds to marshal;
// this is ~12 KB before compression and a few microseconds.
//
//	source table:  uvarint count, then per source uvarint length + bytes
//	items:         uvarint count, then per item
//	                 uvarint item_id, float64 score, byte primary source (0xff: none),
//	                 uvarint hit count, then per hit
//	                   byte source, uvarint rank, float64 score, varint via

var errCorrupt = errors.New("cache: corrupt entry")

const noSource = 0xff

func appendFused(b []byte, items []mixer.Fused) []byte {
	index := map[string]byte{}
	var names []string
	ref := func(s string) {
		if _, ok := index[s]; !ok && s != "" {
			index[s] = byte(len(names))
			names = append(names, s)
		}
	}
	for _, f := range items {
		ref(f.Primary)
		for _, h := range f.Hits {
			ref(h.Source)
		}
	}
	b = binary.AppendUvarint(b, uint64(len(names)))
	for _, n := range names {
		b = binary.AppendUvarint(b, uint64(len(n)))
		b = append(b, n...)
	}
	b = binary.AppendUvarint(b, uint64(len(items)))
	for _, f := range items {
		b = binary.AppendUvarint(b, uint64(uint32(f.ItemID)))
		b = binary.LittleEndian.AppendUint64(b, math.Float64bits(f.Score))
		if i, ok := index[f.Primary]; ok {
			b = append(b, i)
		} else {
			b = append(b, noSource)
		}
		b = binary.AppendUvarint(b, uint64(len(f.Hits)))
		for _, h := range f.Hits {
			b = append(b, index[h.Source])
			b = binary.AppendUvarint(b, uint64(h.Rank))
			b = binary.LittleEndian.AppendUint64(b, math.Float64bits(h.Score))
			b = binary.AppendVarint(b, int64(h.Via))
		}
	}
	return b
}

type reader struct {
	b   []byte
	err error
}

func (r *reader) uvarint() uint64 {
	if r.err != nil {
		return 0
	}
	v, n := binary.Uvarint(r.b)
	if n <= 0 {
		r.err = errCorrupt
		return 0
	}
	r.b = r.b[n:]
	return v
}

func (r *reader) varint() int64 {
	if r.err != nil {
		return 0
	}
	v, n := binary.Varint(r.b)
	if n <= 0 {
		r.err = errCorrupt
		return 0
	}
	r.b = r.b[n:]
	return v
}

func (r *reader) byte() byte {
	if r.err != nil || len(r.b) < 1 {
		r.err = errCorrupt
		return 0
	}
	v := r.b[0]
	r.b = r.b[1:]
	return v
}

func (r *reader) float() float64 {
	if r.err != nil || len(r.b) < 8 {
		r.err = errCorrupt
		return 0
	}
	v := math.Float64frombits(binary.LittleEndian.Uint64(r.b))
	r.b = r.b[8:]
	return v
}

func (r *reader) bytes(n uint64) []byte {
	if r.err != nil || uint64(len(r.b)) < n {
		r.err = errCorrupt
		return nil
	}
	v := r.b[:n]
	r.b = r.b[n:]
	return v
}

// count reads a length and checks it against what is left (each element takes >= min bytes).
func (r *reader) count(min int) int {
	n := r.uvarint()
	if r.err == nil && n > uint64(len(r.b)/min+1) {
		r.err = errCorrupt
	}
	return int(n)
}

func readFused(r *reader) []mixer.Fused {
	names := make([]string, r.count(1))
	for i := range names {
		names[i] = string(r.bytes(r.uvarint()))
	}
	name := func(i byte) string {
		if int(i) < len(names) {
			return names[i]
		}
		return ""
	}
	items := make([]mixer.Fused, r.count(11))
	for i := range items {
		f := mixer.Fused{ItemID: int32(uint32(r.uvarint())), Score: r.float()}
		if p := r.byte(); p != noSource {
			f.Primary = name(p)
		}
		hits := r.count(11)
		if hits > 0 {
			f.Hits = make([]mixer.Hit, hits)
		}
		for j := range f.Hits {
			f.Hits[j] = mixer.Hit{Source: name(r.byte()), Rank: int(r.uvarint()), Score: r.float(), Via: int32(r.varint())}
		}
		items[i] = f
		if r.err != nil {
			return nil
		}
	}
	return items
}
