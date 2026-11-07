// Package bundle loads the parts of a model bundle the api serves in process (design §6): the
// popularity table and the ItemCF neighbour table. The SASRec encoder and the FAISS index of
// the same bundle are served by the model server; every response carries the bundle version.
package bundle

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// Format is the bundle layout version this code reads.
const Format = 1

type fileEntry struct {
	SHA256 string `json:"sha256"`
	Bytes  int64  `json:"bytes"`
}

type manifest struct {
	Format   int                  `json:"format"`
	Version  string               `json:"version"`
	NumItems int                  `json:"num_items"`
	Files    map[string]fileEntry `json:"files"`
}

// ItemCF is the neighbour table: Neighbors[i] are item i's top neighbours, best first.
type ItemCF struct {
	Neighbors    [][]int32
	Scores       [][]float32
	HistoryDecay float64 // gamma: weight of the k-th most recent history item is gamma^k
}

// Bundle is the in-process part of one bundle version.
type Bundle struct {
	Version  string
	Dir      string
	NumItems int
	// Pop lists every item by training interactions, most popular first.
	Pop       []int32
	PopCounts []int32
	ItemCF    ItemCF
	Manifest  json.RawMessage
}

// Load reads dir (a symlink such as artifacts/bundles/current is resolved first), verifies the
// checksums of the files it uses against the manifest and parses them.
func Load(dir string) (*Bundle, error) {
	real, err := filepath.EvalSymlinks(dir)
	if err != nil {
		return nil, fmt.Errorf("bundle %s: %w", dir, err)
	}
	raw, err := os.ReadFile(filepath.Join(real, "manifest.json"))
	if err != nil {
		return nil, fmt.Errorf("bundle %s: %w", real, err)
	}
	var m manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("bundle %s: manifest: %w", real, err)
	}
	if m.Format != Format {
		return nil, fmt.Errorf("bundle %s: format %d, expected %d", real, m.Format, Format)
	}
	b := &Bundle{Version: m.Version, Dir: real, NumItems: m.NumItems, Manifest: raw}

	var pop struct {
		Items  []int32 `json:"items"`
		Counts []int32 `json:"counts"`
	}
	if err := readVerified(real, "pop.json", m.Files, &pop); err != nil {
		return nil, err
	}
	if len(pop.Items) != m.NumItems || len(pop.Counts) != m.NumItems {
		return nil, fmt.Errorf("bundle %s: pop.json has %d items, manifest %d", real, len(pop.Items), m.NumItems)
	}
	b.Pop, b.PopCounts = pop.Items, pop.Counts

	var cf struct {
		Params struct {
			HistoryDecay float64 `json:"history_decay"`
		} `json:"params"`
		Neighbors [][]int32   `json:"neighbors"`
		Scores    [][]float32 `json:"scores"`
	}
	if err := readVerified(real, "itemcf.json", m.Files, &cf); err != nil {
		return nil, err
	}
	if len(cf.Neighbors) != m.NumItems || len(cf.Scores) != m.NumItems {
		return nil, fmt.Errorf("bundle %s: itemcf.json has %d rows, manifest %d", real, len(cf.Neighbors), m.NumItems)
	}
	for i := range cf.Neighbors {
		if len(cf.Neighbors[i]) != len(cf.Scores[i]) {
			return nil, fmt.Errorf("bundle %s: itemcf row %d is ragged", real, i)
		}
		for _, j := range cf.Neighbors[i] {
			if j < 0 || int(j) >= m.NumItems {
				return nil, fmt.Errorf("bundle %s: itemcf row %d has item %d", real, i, j)
			}
		}
	}
	b.ItemCF = ItemCF{Neighbors: cf.Neighbors, Scores: cf.Scores, HistoryDecay: cf.Params.HistoryDecay}
	return b, nil
}

func readVerified(dir, name string, files map[string]fileEntry, into any) error {
	want, ok := files[name]
	if !ok {
		return fmt.Errorf("bundle %s: %s is not in the manifest", dir, name)
	}
	f, err := os.Open(filepath.Join(dir, name))
	if err != nil {
		return fmt.Errorf("bundle %s: %w", dir, err)
	}
	defer f.Close()
	h := sha256.New()
	raw, err := io.ReadAll(io.TeeReader(f, h))
	if err != nil {
		return fmt.Errorf("bundle %s: %s: %w", dir, name, err)
	}
	if int64(len(raw)) != want.Bytes || hex.EncodeToString(h.Sum(nil)) != want.SHA256 {
		return fmt.Errorf("bundle %s: %s does not match its checksum in the manifest", dir, name)
	}
	if err := json.Unmarshal(raw, into); err != nil {
		return fmt.Errorf("bundle %s: %s: %w", dir, name, err)
	}
	return nil
}
