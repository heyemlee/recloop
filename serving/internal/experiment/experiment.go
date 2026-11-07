// Package experiment implements layered experiment assignment (docs/design.md §8).
//
// Each layer (recall, rank, ads, agent) has its own salt; a unit (user id) falls into
//
//	bucket = xxhash64(salt + ":" + unit) % 1000
//
// in every layer independently, so layers are orthogonal and experiments inside one layer are
// mutually exclusive (their bucket ranges may not overlap). xxhash64 is unseeded and identical
// in Go and Python (ml/experiment/bucketing.py); testdata/bucket_vectors.json pins both.
package experiment

import (
	"encoding/json"
	"fmt"
	"sort"
	"strconv"

	"github.com/cespare/xxhash/v2"
)

// Buckets per layer.
const Buckets = 1000

// Layers in a fixed order.
var Layers = []string{"recall", "rank", "ads", "agent"}

// Salts per layer. Changing a salt reshuffles every unit of that layer.
var Salts = map[string]string{
	"recall": "recall-v1",
	"rank":   "rank-v1",
	"ads":    "ads-v1",
	"agent":  "agent-v1",
}

// DefaultVariant is the group of a unit no running experiment covers.
const DefaultVariant = "default"

// Bucket returns xxhash64(salt + ":" + unit) mod 1000.
func Bucket(salt, unit string) int {
	return int(xxhash.Sum64String(salt+":"+unit) % Buckets)
}

// UserUnit is the unit id of a signed-in user.
func UserUnit(userID int64) string { return strconv.FormatInt(userID, 10) }

// Variant of an experiment; Share is a percentage of the experiment's bucket range.
type Variant struct {
	Name   string          `json:"name"`
	Share  int             `json:"share"`
	Params json.RawMessage `json:"params,omitempty"`
}

// Experiment covers buckets [From, To) of one layer.
type Experiment struct {
	ID            string
	Layer         string
	From, To      int
	Variants      []Variant
	ConfigVersion int
}

// Assignment of one unit in one layer.
type Assignment struct {
	Bucket  int             `json:"bucket"`
	ExpID   string          `json:"exp_id,omitempty"`
	Variant string          `json:"variant"`
	Params  json.RawMessage `json:"-"`
}

// Config is an immutable snapshot of the running experiments.
type Config struct {
	Version int
	byLayer map[string][]Experiment
}

// NewConfig validates the experiments and indexes them by layer. The version is the largest
// config_version among them (0 when there is none).
func NewConfig(exps []Experiment) (*Config, error) {
	c := &Config{byLayer: map[string][]Experiment{}}
	for _, e := range exps {
		if _, ok := Salts[e.Layer]; !ok {
			return nil, fmt.Errorf("experiment %s: unknown layer %q", e.ID, e.Layer)
		}
		if e.From < 0 || e.To > Buckets || e.From >= e.To {
			return nil, fmt.Errorf("experiment %s: bad bucket range [%d, %d)", e.ID, e.From, e.To)
		}
		total := 0
		for _, v := range e.Variants {
			if v.Share <= 0 || v.Name == "" {
				return nil, fmt.Errorf("experiment %s: bad variant %+v", e.ID, v)
			}
			total += v.Share
		}
		if total != 100 {
			return nil, fmt.Errorf("experiment %s: variant shares add up to %d, not 100", e.ID, total)
		}
		for _, other := range c.byLayer[e.Layer] {
			if e.From < other.To && other.From < e.To {
				return nil, fmt.Errorf("experiments %s and %s overlap in layer %s", e.ID, other.ID, e.Layer)
			}
		}
		c.byLayer[e.Layer] = append(c.byLayer[e.Layer], e)
		if e.ConfigVersion > c.Version {
			c.Version = e.ConfigVersion
		}
	}
	for _, list := range c.byLayer {
		sort.Slice(list, func(i, j int) bool { return list[i].From < list[j].From })
	}
	return c, nil
}

// Assign returns the unit's assignment in every layer.
func (c *Config) Assign(unit string) map[string]Assignment {
	out := make(map[string]Assignment, len(Layers))
	for _, layer := range Layers {
		b := Bucket(Salts[layer], unit)
		a := Assignment{Bucket: b, Variant: DefaultVariant}
		if c != nil {
			for _, e := range c.byLayer[layer] {
				if b < e.From || b >= e.To {
					continue
				}
				v := pickVariant(e, b)
				a.ExpID, a.Variant, a.Params = e.ID, v.Name, v.Params
				break
			}
		}
		out[layer] = a
	}
	return out
}

// pickVariant splits [From, To) into consecutive sub-ranges proportional to the shares.
func pickVariant(e Experiment, bucket int) Variant {
	width := e.To - e.From
	offset := bucket - e.From
	cum := 0
	for _, v := range e.Variants {
		cum += v.Share
		if offset < width*cum/100 {
			return v
		}
	}
	return e.Variants[len(e.Variants)-1]
}
