package release_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/heyemlee/recloop/serving/internal/bundle"
	"github.com/heyemlee/recloop/serving/internal/release"
	"github.com/heyemlee/recloop/serving/internal/store"
)

// memStore keeps model_versions, model_version_events and experiments in memory, with the
// transition rules of store.Store (one transaction: the old active retires at the moment the
// new one activates).
type memStore struct {
	mu       sync.Mutex
	now      func() time.Time
	versions map[string]*store.ModelVersion
	events   []store.ModelEvent
	exps     map[string]store.ExperimentRow
	cfg      int32
}

func newMem(now func() time.Time, active string) *memStore {
	t := now()
	m := &memStore{now: now, versions: map[string]*store.ModelVersion{}, exps: map[string]store.ExperimentRow{}}
	m.versions[active] = &store.ModelVersion{Version: active, Status: "active", CreatedAt: t, ActivatedAt: &t}
	return m
}

func (m *memStore) ModelVersions(context.Context) ([]store.ModelVersion, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var out []store.ModelVersion
	for _, v := range m.versions {
		out = append(out, *v)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Version < out[j].Version })
	return out, nil
}

func (m *memStore) ModelEvents(context.Context, int32) ([]store.ModelEvent, error) {
	return slices.Clone(m.events), nil
}

func (m *memStore) RegisterBuilt(_ context.Context, version, uri string, _ []byte, actor string) (bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.versions[version] != nil {
		return false, nil
	}
	m.versions[version] = &store.ModelVersion{Version: version, Status: "built", ArtifactURI: uri, CreatedAt: m.now()}
	m.events = append(m.events, store.ModelEvent{Version: version, To: "built", Actor: actor})
	return true, nil
}

func (m *memStore) Transition(_ context.Context, version string, from []string, to, actor, note string) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	t := m.now()
	v := m.versions[version]
	if v == nil {
		return "", store.ErrNoVersion
	}
	if !slices.Contains(from, v.Status) {
		return "", fmt.Errorf("%w: %s is %s", store.ErrTransition, version, v.Status)
	}
	previous := ""
	if to == "active" {
		for _, o := range m.versions {
			if o.Status == "active" && o.Version != version {
				previous = o.Version
				o.Status, o.RetiredAt = "retired", &t
				m.events = append(m.events, store.ModelEvent{Version: o.Version, From: "active", To: "retired", Actor: actor})
			}
		}
		v.ActivatedAt, v.RetiredAt = &t, nil
	}
	switch to {
	case "validated":
		v.ValidatedAt = &t
	case "canary":
		v.CanaryAt = &t
	case "retired":
		v.RetiredAt = &t
	}
	m.events = append(m.events, store.ModelEvent{Version: version, From: v.Status, To: to, Actor: actor, Note: note})
	v.Status = to
	return previous, nil
}

func (m *memStore) Experiments(context.Context) ([]store.ExperimentRow, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var out []store.ExperimentRow
	for _, e := range m.exps {
		out = append(out, e)
	}
	return out, nil
}

func (m *memStore) PutExperiment(_ context.Context, e store.ExperimentRow) (int32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.cfg++
	e.ConfigVersion = m.cfg
	m.exps[e.ID] = e
	return m.cfg, nil
}

func (m *memStore) StopExperiment(_ context.Context, id string) (bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	e, ok := m.exps[id]
	if !ok || e.Status != "running" {
		return false, nil
	}
	e.Status = "stopped"
	m.exps[id] = e
	return true, nil
}

// fakeMS is the model server's admin port.
type fakeMS struct {
	mu      sync.Mutex
	loaded  map[string]bool
	def     string
	failing map[string]bool
}

func (f *fakeMS) Status(context.Context) (release.MSStatus, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	st := release.MSStatus{Default: f.def}
	for v := range f.loaded {
		st.Loaded = append(st.Loaded, struct {
			Version string `json:"version"`
		}{v})
	}
	return st, nil
}

func (f *fakeMS) Load(_ context.Context, v string, _ bool) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.failing[v] {
		return "failed: golden queries disagree", nil
	}
	f.loaded[v] = true
	return "ready", nil
}

func (f *fakeMS) SetDefault(_ context.Context, v string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if !f.loaded[v] {
		return errors.New("not loaded")
	}
	f.def = v
	return nil
}

func (f *fakeMS) Unload(_ context.Context, v string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	delete(f.loaded, v)
	return nil
}

type clock struct{ t time.Time }

func (c *clock) now() time.Time { return c.t }

func setup(t *testing.T) (*release.Controller, *memStore, *fakeMS, *clock) {
	t.Helper()
	root := t.TempDir()
	for _, v := range []string{"v1", "v2"} {
		if err := os.MkdirAll(filepath.Join(root, v), 0o755); err != nil {
			t.Fatal(err)
		}
		raw, _ := json.Marshal(map[string]string{"version": v})
		if err := os.WriteFile(filepath.Join(root, v, "manifest.json"), raw, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	clk := &clock{t: time.Date(2026, 11, 8, 12, 0, 0, 0, time.UTC)}
	reg := bundle.NewRegistry(root, &bundle.Bundle{Version: "v0"}).WithLoader(func(dir string) (*bundle.Bundle, error) {
		return &bundle.Bundle{Version: filepath.Base(dir), Dir: dir}, nil
	})
	st := newMem(clk.now, "v0")
	ms := &fakeMS{loaded: map[string]bool{"v0": true}, def: "v0", failing: map[string]bool{}}
	c := &release.Controller{Store: st, Bundles: reg, MS: ms, Log: slog.New(slog.NewTextHandler(io.Discard, nil)), Now: clk.now}
	return c, st, ms, clk
}

func status(st *memStore, v string) string {
	st.mu.Lock()
	defer st.mu.Unlock()
	return st.versions[v].Status
}

func TestCanaryThenFullRolloutThenRollback(t *testing.T) {
	c, st, ms, clk := setup(t)
	ctx := context.Background()
	if err := c.Register(ctx, "v1", "test"); err != nil {
		t.Fatal(err)
	}
	if err := c.Canary(ctx, "v1", 5, "test"); !errors.Is(err, release.ErrState) {
		t.Fatalf("canary before validation: %v", err)
	}
	if err := c.Validate(ctx, "v1", "test"); err != nil || status(st, "v1") != "validated" {
		t.Fatal(err, status(st, "v1"))
	}
	if c.Bundles.Active().Version != "v0" || !ms.loaded["v1"] {
		t.Fatal("validation must load v1 without serving it")
	}
	if err := c.Canary(ctx, "v1", 5, "test"); err != nil || status(st, "v1") != "canary" {
		t.Fatal(err)
	}
	exp := st.exps[release.CanaryExperiment]
	if exp.Layer != "rank" || exp.BucketFrom != 0 || exp.BucketTo != 50 || exp.Status != "running" {
		t.Fatalf("canary experiment %+v", exp)
	}
	clk.t = clk.t.Add(10 * time.Minute)
	prev, err := c.Activate(ctx, "v1", "test", "canary passed")
	if err != nil || prev != "v0" {
		t.Fatal(prev, err)
	}
	if status(st, "v1") != "active" || status(st, "v0") != "retired" || c.Bundles.Active().Version != "v1" || ms.def != "v1" {
		t.Fatal("activation did not switch everything")
	}
	if st.exps[release.CanaryExperiment].Status != "stopped" {
		t.Fatal("canary experiment still running")
	}
	clk.t = clk.t.Add(5 * time.Minute)
	to, err := c.Rollback(ctx, "test", "")
	if err != nil || to != "v0" || c.Bundles.Active().Version != "v0" || status(st, "v1") != "retired" || ms.def != "v0" {
		t.Fatal(to, err)
	}
	// The rollback of the rollback goes forward again: v1 was retired when v0 came back.
	clk.t = clk.t.Add(time.Minute)
	if to, err := c.Rollback(ctx, "test", ""); err != nil || to != "v1" {
		t.Fatal(to, err)
	}
	// Too late: the previous version has been retired for more than 30 minutes.
	clk.t = clk.t.Add(31 * time.Minute)
	if _, err := c.Rollback(ctx, "test", ""); !errors.Is(err, release.ErrTooLate) {
		t.Fatalf("rollback after 31 min: %v", err)
	}
}

func TestValidationFailureLeavesTheVersionBuilt(t *testing.T) {
	c, st, ms, _ := setup(t)
	ctx := context.Background()
	_ = c.Register(ctx, "v1", "test")
	_ = c.Register(ctx, "v2", "test")
	c.Check = func(b *bundle.Bundle) error {
		if b.Version == "v1" {
			return errors.New("itemcf golden query 3 differs")
		}
		return nil
	}
	if err := c.Validate(ctx, "v1", "test"); err == nil || status(st, "v1") != "built" || c.Bundles.Get("v1") != nil {
		t.Fatalf("a failed check validated v1: %v", err)
	}
	ms.failing["v2"] = true
	if err := c.Validate(ctx, "v2", "test"); err == nil || status(st, "v2") != "built" {
		t.Fatalf("a failed model-server load validated v2: %v", err)
	}
	if err := c.Register(ctx, "missing", "test"); !errors.Is(err, release.ErrNotFound) {
		t.Fatal(err)
	}
}

func TestAbortCanaryRetiresIt(t *testing.T) {
	c, st, _, _ := setup(t)
	ctx := context.Background()
	_ = c.Register(ctx, "v1", "test")
	_ = c.Validate(ctx, "v1", "test")
	_ = c.Canary(ctx, "v1", 5, "test")
	if v, err := c.AbortCanary(ctx, "test", "guardrails failed"); err != nil || v != "v1" || status(st, "v1") != "retired" {
		t.Fatal(v, err)
	}
	if st.exps[release.CanaryExperiment].Status != "stopped" || c.Bundles.Active().Version != "v0" {
		t.Fatal("abort left the canary running")
	}
	// The aborted canary never was active: it is no rollback target.
	if _, err := c.Rollback(ctx, "test", ""); !errors.Is(err, release.ErrNoTarget) {
		t.Fatal(err)
	}
}

func TestReconcileLoadsWhatServesAndUnloadsExpiredVersions(t *testing.T) {
	c, _, ms, clk := setup(t)
	ctx := context.Background()
	_ = c.Register(ctx, "v1", "test")
	_ = c.Validate(ctx, "v1", "test")
	if _, err := c.Activate(ctx, "v1", "test", ""); err != nil {
		t.Fatal(err)
	}
	// Another api instance: it starts with v0 and learns about v1 from model_versions.
	other := &release.Controller{Store: c.Store, MS: c.MS, Log: c.Log, Now: c.Now,
		Bundles: bundle.NewRegistry(c.Bundles.Root(), &bundle.Bundle{Version: "v0"}).WithLoader(func(dir string) (*bundle.Bundle, error) {
			return &bundle.Bundle{Version: filepath.Base(dir)}, nil
		})}
	if err := other.Reconcile(ctx); err != nil {
		t.Fatal(err)
	}
	if other.Bundles.Active().Version != "v1" || other.Bundles.Get("v0") == nil {
		t.Fatal("reconcile: v1 active, v0 kept for a rollback")
	}
	clk.t = clk.t.Add(31 * time.Minute)
	if err := other.Reconcile(ctx); err != nil {
		t.Fatal(err)
	}
	if other.Bundles.Get("v0") != nil || ms.loaded["v0"] {
		t.Fatal("v0 still loaded 31 minutes after its retirement")
	}
}

// stallStore hands out a ModelVersions snapshot and then holds that caller until released: a
// Reconcile that read the release state just before an activation.
type stallStore struct {
	*memStore
	armed   atomic.Bool
	read    chan struct{} // closed once the stalled call has its snapshot
	release chan struct{}
}

func (s *stallStore) ModelVersions(ctx context.Context) ([]store.ModelVersion, error) {
	out, err := s.memStore.ModelVersions(ctx)
	if s.armed.CompareAndSwap(true, false) {
		close(s.read)
		<-s.release
	}
	return out, err
}

func TestAStaleReconcileDoesNotUndoAnActivation(t *testing.T) {
	c, st, ms, _ := setup(t)
	ctx := context.Background()
	if err := c.Register(ctx, "v1", "test"); err != nil {
		t.Fatal(err)
	}
	if err := c.Validate(ctx, "v1", "test"); err != nil {
		t.Fatal(err)
	}
	ss := &stallStore{memStore: st, read: make(chan struct{}), release: make(chan struct{})}
	c.Store = ss
	ss.armed.Store(true)
	rec := make(chan error, 1)
	go func() { rec <- c.Reconcile(ctx) }() // reads "v0 is active", then stalls
	<-ss.read
	act := make(chan error, 1)
	go func() {
		_, err := c.Activate(ctx, "v1", "test", "go")
		act <- err
	}()
	time.Sleep(50 * time.Millisecond) // time for the activation to finish, were it not blocked
	close(ss.release)
	if err := <-rec; err != nil {
		t.Fatal(err)
	}
	if err := <-act; err != nil {
		t.Fatal(err)
	}
	if got := c.Bundles.Active().Version; got != "v1" {
		t.Fatalf("the api serves %s after the activation of v1 and a stale reconcile", got)
	}
	if ms.def != "v1" {
		t.Fatalf("model server default %s", ms.def)
	}
}
