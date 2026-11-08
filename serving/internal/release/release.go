// Package release moves bundle versions through the release steps of design §6:
//
//		built -> validated -> canary -> active -> retired
//
//	  - register: a bundle directory under the bundles root becomes a built version;
//	  - validate: the api loads it (checksums, item count, ItemCF's golden queries) and the model
//	    server loads it (checksums, dimensions, item count, the encoder's and ranker's golden
//	    queries); both keep serving the active version meanwhile;
//	  - canary: a rank-layer experiment pins a share of the users to it (pin_version);
//	  - activate: one transaction makes it the active version and retires the previous one; the
//	    api switches its pointer at once, the model server's default follows; the canary
//	    experiment ends;
//	  - rollback: within KeepPrevious of a switch, the previous version is still loaded on both
//	    sides, so rolling back is the same pointer switch back.
//
// PostgreSQL's model_versions is the source of truth and model_version_events the record of
// every step. Reconcile, run every few seconds by every api instance, makes the process match
// it: the active version and any canary loaded and pointed at, the previous active version kept
// for KeepPrevious after its retirement and unloaded afterwards.
package release

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/heyemlee/recloop/serving/internal/bundle"
	"github.com/heyemlee/recloop/serving/internal/store"
)

// KeepPrevious: how long a replaced version stays loaded (design §6: 30 minutes).
const KeepPrevious = 30 * time.Minute

// CanaryExperiment is the id of the rank-layer experiment that pins the canary's users.
const CanaryExperiment = "release-canary"

// Store is what the controller needs from PostgreSQL (store.Store).
type Store interface {
	ModelVersions(ctx context.Context) ([]store.ModelVersion, error)
	ModelEvents(ctx context.Context, limit int32) ([]store.ModelEvent, error)
	RegisterBuilt(ctx context.Context, version, uri string, manifest []byte, actor string) (bool, error)
	Transition(ctx context.Context, version string, from []string, to, actor, note string) (string, error)
	Experiments(ctx context.Context) ([]store.ExperimentRow, error)
	PutExperiment(ctx context.Context, e store.ExperimentRow) (int32, error)
	StopExperiment(ctx context.Context, id string) (bool, error)
}

// ModelServer is the model server's admin port (model_server/admin.py).
type ModelServer interface {
	Status(ctx context.Context) (MSStatus, error)
	Load(ctx context.Context, version string, wait bool) (string, error)
	SetDefault(ctx context.Context, version string) error
	Unload(ctx context.Context, version string) error
}

// MSStatus is the model server's GET /bundles.
type MSStatus struct {
	Default string `json:"default"`
	Loaded  []struct {
		Version string `json:"version"`
	} `json:"loaded"`
	Loading map[string]string `json:"loading"`
}

// Has reports whether a version is loaded there.
func (s MSStatus) Has(version string) bool {
	for _, l := range s.Loaded {
		if l.Version == version {
			return true
		}
	}
	return false
}

// Controller runs the release steps. Its methods are safe for concurrent use.
type Controller struct {
	Store   Store
	Bundles *bundle.Registry
	MS      ModelServer
	// Check validates a bundle loaded by the api before it may serve (item count against the
	// catalog, ItemCF's golden queries).
	Check func(*bundle.Bundle) error
	Log   *slog.Logger
	Now   func() time.Time
	Keep  time.Duration // KeepPrevious if zero

	mu sync.Mutex // one release step at a time in this process
}

func (c *Controller) keep() time.Duration {
	if c.Keep > 0 {
		return c.Keep
	}
	return KeepPrevious
}

// Errors the admin api maps to 4xx.
var (
	ErrNotFound = errors.New("no such bundle")
	ErrState    = store.ErrTransition
	ErrTooLate  = errors.New("the previous version was retired too long ago: activate it again instead")
	ErrNoTarget = errors.New("no previous version to roll back to")
)

func (c *Controller) find(ctx context.Context, version string) (store.ModelVersion, error) {
	all, err := c.Store.ModelVersions(ctx)
	if err != nil {
		return store.ModelVersion{}, err
	}
	for _, v := range all {
		if v.Version == version {
			return v, nil
		}
	}
	return store.ModelVersion{}, fmt.Errorf("%w: %s", ErrNotFound, version)
}

// Register records <bundles root>/<version> as a built version.
func (c *Controller) Register(ctx context.Context, version, actor string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	dir := filepath.Join(c.Bundles.Root(), version)
	raw, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return fmt.Errorf("%w: %v", ErrNotFound, err)
	}
	var m struct {
		Version string `json:"version"`
	}
	if err := json.Unmarshal(raw, &m); err != nil || m.Version != version {
		return fmt.Errorf("%w: %s/manifest.json names version %q", ErrNotFound, version, m.Version)
	}
	_, err = c.Store.RegisterBuilt(ctx, version, dir, raw, actor)
	return err
}

// Validate loads a built version on both sides and checks it; nothing serves it yet.
func (c *Controller) Validate(ctx context.Context, version, actor string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	v, err := c.find(ctx, version)
	if err != nil {
		return err
	}
	if v.Status != "built" && v.Status != "retired" {
		return fmt.Errorf("%w: %s is %s", ErrState, version, v.Status)
	}
	t0 := time.Now()
	b, err := c.Bundles.Ensure(version)
	if err == nil && c.Check != nil {
		err = c.Check(b)
	}
	if err != nil {
		_ = c.Bundles.Unload(version)
		return fmt.Errorf("api side: %w", err)
	}
	state, err := c.MS.Load(ctx, version, true)
	if err != nil {
		return fmt.Errorf("model server: %w", err)
	}
	if state != "ready" {
		return fmt.Errorf("model server: %s", state)
	}
	note := fmt.Sprintf("api and model server loaded and checked it in %s", time.Since(t0).Round(time.Millisecond))
	_, err = c.Store.Transition(ctx, version, []string{"built", "retired"}, "validated", actor, note)
	return err
}

// Canary pins `percent` of the users (rank-layer buckets [0, percent * 10)) to a validated
// version through the experiment CanaryExperiment.
func (c *Controller) Canary(ctx context.Context, version string, percent int, actor string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if percent < 1 || percent > 50 {
		return fmt.Errorf("%w: a canary takes 1 to 50 percent of the users, not %d", ErrState, percent)
	}
	v, err := c.find(ctx, version)
	if err != nil {
		return err
	}
	if v.Status != "validated" && v.Status != "canary" {
		return fmt.Errorf("%w: %s is %s; validate it first", ErrState, version, v.Status)
	}
	if _, err := c.Bundles.Ensure(version); err != nil {
		return fmt.Errorf("api side: %w", err)
	}
	if state, err := c.MS.Load(ctx, version, true); err != nil || state != "ready" {
		return fmt.Errorf("model server: %v %s", err, state)
	}
	params, _ := json.Marshal(map[string]string{"pin_version": version})
	variants, _ := json.Marshal([]map[string]any{{"name": "canary", "share": 100, "params": json.RawMessage(params)}})
	if _, err := c.Store.PutExperiment(ctx, store.ExperimentRow{
		ID: CanaryExperiment, Layer: "rank", Status: "running", BucketFrom: 0, BucketTo: int32(percent * 10), Variants: variants,
	}); err != nil {
		return fmt.Errorf("canary experiment: %w", err)
	}
	if v.Status == "canary" {
		return nil // a new share for the running canary
	}
	_, err = c.Store.Transition(ctx, version, []string{"validated"}, "canary", actor, fmt.Sprintf("%d%% of users (rank buckets [0, %d))", percent, percent*10))
	return err
}

// Activate makes a validated (or canary, or recently retired) version the active one.
func (c *Controller) Activate(ctx context.Context, version, actor, note string) (string, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.activate(ctx, version, []string{"validated", "canary", "retired"}, actor, note)
}

func (c *Controller) activate(ctx context.Context, version string, from []string, actor, note string) (string, error) {
	if _, err := c.Bundles.Ensure(version); err != nil {
		return "", fmt.Errorf("api side: %w", err)
	}
	if state, err := c.MS.Load(ctx, version, true); err != nil || state != "ready" {
		return "", fmt.Errorf("model server: %v %s", err, state)
	}
	previous, err := c.Store.Transition(ctx, version, from, "active", actor, note)
	if err != nil {
		return "", err
	}
	// This process switches now; the other api instances at their next Reconcile.
	if err := c.Bundles.SetActive(version); err != nil {
		return previous, err
	}
	if err := c.MS.SetDefault(ctx, version); err != nil {
		c.Log.Warn("model server default not switched yet", "version", version, "err", err)
	}
	if _, err := c.Store.StopExperiment(ctx, CanaryExperiment); err != nil {
		c.Log.Warn("canary experiment not stopped", "err", err)
	}
	c.Log.Warn("bundle activated", "version", version, "previous", previous, "actor", actor)
	return previous, nil
}

// AbortCanary ends the canary: its experiment stops and the version is retired without
// having been active (the rollback of a release that did not pass its canary).
func (c *Controller) AbortCanary(ctx context.Context, actor, note string) (string, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	all, err := c.Store.ModelVersions(ctx)
	if err != nil {
		return "", err
	}
	for _, v := range all {
		if v.Status != "canary" {
			continue
		}
		if _, err := c.Store.StopExperiment(ctx, CanaryExperiment); err != nil {
			return "", err
		}
		if note == "" {
			note = "canary aborted"
		}
		_, err := c.Store.Transition(ctx, v.Version, []string{"canary"}, "retired", actor, note)
		return v.Version, err
	}
	return "", fmt.Errorf("%w: no canary is running", ErrState)
}

// Rollback re-activates the version the active one replaced, if that was less than
// KeepPrevious ago (it is still loaded: the switch back is a pointer switch).
func (c *Controller) Rollback(ctx context.Context, actor, note string) (string, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	all, err := c.Store.ModelVersions(ctx)
	if err != nil {
		return "", err
	}
	var active, target *store.ModelVersion
	for i := range all {
		if all[i].Status == "active" {
			active = &all[i]
		}
	}
	if active == nil || active.ActivatedAt == nil {
		return "", ErrNoTarget
	}
	// The target is the version the activation retired: the same transaction set its
	// retired_at and the active version's activated_at (PostgreSQL's now() is per transaction).
	for i := range all {
		v := &all[i]
		// ActivatedAt: it served as the active version once (an aborted canary never did).
		if v.Status == "retired" && v.RetiredAt != nil && v.ActivatedAt != nil && v.RetiredAt.Equal(*active.ActivatedAt) {
			target = v
		}
	}
	if target == nil {
		return "", ErrNoTarget
	}
	if c.Now().Sub(*target.RetiredAt) > c.keep() {
		return "", fmt.Errorf("%w (%s, retired %s ago)", ErrTooLate, target.Version, c.Now().Sub(*target.RetiredAt).Round(time.Second))
	}
	if note == "" {
		note = "rollback from " + active.Version
	}
	if _, err := c.activate(ctx, target.Version, []string{"retired"}, actor, note); err != nil {
		return "", err
	}
	return target.Version, nil
}

// Reconcile makes this process (and the model server) match model_versions: the active
// version and the canary loaded and active / default, a retired version kept loaded for
// KeepPrevious and then unloaded.
func (c *Controller) Reconcile(ctx context.Context) error {
	all, err := c.Store.ModelVersions(ctx)
	if err != nil {
		return err
	}
	want := map[string]bool{}
	var active string
	for _, v := range all {
		switch {
		case v.Status == "active":
			active = v.Version
			want[v.Version] = true
		case v.Status == "canary":
			want[v.Version] = true
		case v.Status == "retired" && v.RetiredAt != nil && c.Now().Sub(*v.RetiredAt) <= c.keep():
			want[v.Version] = true
		}
	}
	var errs []error
	for v := range want {
		if _, err := c.Bundles.Ensure(v); err != nil {
			errs = append(errs, fmt.Errorf("load %s: %w", v, err))
		}
	}
	if active != "" {
		if cur := c.Bundles.Active(); cur == nil || cur.Version != active {
			if err := c.Bundles.SetActive(active); err != nil {
				errs = append(errs, err)
			} else {
				c.Log.Warn("active bundle switched (reconcile)", "version", active)
			}
		}
	}
	for _, v := range c.Bundles.Versions() {
		if a := c.Bundles.Active(); want[v] || (a != nil && a.Version == v) {
			continue
		}
		_ = c.Bundles.Unload(v) // retired more than KeepPrevious ago
	}
	if c.MS != nil {
		st, err := c.MS.Status(ctx)
		if err != nil {
			return errors.Join(append(errs, fmt.Errorf("model server: %w", err))...)
		}
		for v := range want {
			if !st.Has(v) {
				if _, err := c.MS.Load(ctx, v, false); err != nil {
					errs = append(errs, err)
				}
			}
		}
		if active != "" && st.Default != active && st.Has(active) {
			if err := c.MS.SetDefault(ctx, active); err != nil {
				errs = append(errs, err)
			}
		}
		for _, l := range st.Loaded {
			if !want[l.Version] && l.Version != st.Default && l.Version != active {
				if err := c.MS.Unload(ctx, l.Version); err != nil {
					errs = append(errs, err)
				}
			}
		}
	}
	return errors.Join(errs...)
}

// Run reconciles every `every` until ctx is done.
func (c *Controller) Run(ctx context.Context, every time.Duration) {
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		if err := c.Reconcile(ctx); err != nil && ctx.Err() == nil {
			c.Log.Warn("release reconcile", "err", err)
		}
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}

// Overview is the admin page's view of the release state.
type Overview struct {
	Active   string               `json:"active"`
	Loaded   []string             `json:"loaded"`
	Versions []store.ModelVersion `json:"versions"`
	Events   []store.ModelEvent   `json:"events"`
	Server   *MSStatus            `json:"model_server,omitempty"`
	Canary   *store.ExperimentRow `json:"canary,omitempty"`
	Keep     string               `json:"keep_previous"`
}

// Overview lists the versions (without manifests), the latest events and what is loaded.
func (c *Controller) Overview(ctx context.Context) (Overview, error) {
	all, err := c.Store.ModelVersions(ctx)
	if err != nil {
		return Overview{}, err
	}
	events, err := c.Store.ModelEvents(ctx, 50)
	if err != nil {
		return Overview{}, err
	}
	out := Overview{Loaded: c.Bundles.Versions(), Events: events, Keep: c.keep().String()}
	if a := c.Bundles.Active(); a != nil {
		out.Active = a.Version
	}
	for _, v := range all {
		v.Manifest = nil
		out.Versions = append(out.Versions, v)
	}
	if exps, err := c.Store.Experiments(ctx); err == nil {
		for _, e := range exps {
			if e.ID == CanaryExperiment && e.Status == "running" {
				e := e
				out.Canary = &e
			}
		}
	}
	if c.MS != nil {
		if st, err := c.MS.Status(ctx); err == nil {
			out.Server = &st
		}
	}
	return out, nil
}

// Experiments lists every experiment (the admin page's table).
func (c *Controller) Experiments(ctx context.Context) ([]store.ExperimentRow, error) {
	return c.Store.Experiments(ctx)
}

// PutExperiment creates or replaces an experiment.
func (c *Controller) PutExperiment(ctx context.Context, e store.ExperimentRow) (int32, error) {
	return c.Store.PutExperiment(ctx, e)
}

// StopExperiment stops a running experiment.
func (c *Controller) StopExperiment(ctx context.Context, id string) (bool, error) {
	return c.Store.StopExperiment(ctx, id)
}

// Fault reads (body nil) or sets the model server's fault switch.
func (c *Controller) Fault(ctx context.Context, body map[string]any) (map[string]any, error) {
	f, ok := c.MS.(interface {
		Fault(ctx context.Context, body map[string]any) (map[string]any, error)
	})
	if !ok {
		return nil, errors.New("the model server's admin port is not configured")
	}
	return f.Fault(ctx, body)
}
