package bundle

import (
	"errors"
	"fmt"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
)

// Registry holds the bundle versions the api serves at once (design §6): the active one, a
// canary pinned for part of the traffic, and the previous active version for 30 minutes after a
// switch, so that a rollback is a pointer switch back. Versions are found by name under one
// root directory (artifacts/bundles on the host, /bundles in a container).
type Registry struct {
	root   string
	load   func(dir string) (*Bundle, error)
	mu     sync.Mutex
	byName map[string]*Bundle
	active atomic.Pointer[Bundle]
}

// NewRegistry starts with one loaded bundle, which is also the active one.
func NewRegistry(root string, initial *Bundle) *Registry {
	r := &Registry{root: root, load: Load, byName: map[string]*Bundle{initial.Version: initial}}
	r.active.Store(initial)
	return r
}

// Root is the directory the versions live in.
func (r *Registry) Root() string { return r.root }

// Active is the version unpinned requests get. One atomic load: a request reads it once and
// uses that bundle for every stage, so a switch never mixes two versions in one response.
func (r *Registry) Active() *Bundle { return r.active.Load() }

// Get returns a loaded version, or nil.
func (r *Registry) Get(version string) *Bundle {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.byName[version]
}

// Ensure loads <root>/<version> unless it is loaded; the manifest must name that version.
func (r *Registry) Ensure(version string) (*Bundle, error) {
	if b := r.Get(version); b != nil {
		return b, nil
	}
	if version == "" || strings.ContainsAny(version, `/\`) || strings.HasPrefix(version, ".") {
		return nil, fmt.Errorf("bundle version %q is not a directory name", version)
	}
	b, err := r.load(filepath.Join(r.root, version))
	if err != nil {
		return nil, err
	}
	if b.Version != version {
		return nil, fmt.Errorf("bundle directory %s holds version %s", version, b.Version)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if old := r.byName[version]; old != nil {
		return old, nil
	}
	r.byName[version] = b
	return b, nil
}

// SetActive switches the active version to a loaded one.
func (r *Registry) SetActive(version string) error {
	b := r.Get(version)
	if b == nil {
		return fmt.Errorf("bundle %s is not loaded", version)
	}
	r.active.Store(b)
	return nil
}

// ErrActive: the active version cannot be unloaded.
var ErrActive = errors.New("the active bundle cannot be unloaded")

// Unload drops a version that is not active.
func (r *Registry) Unload(version string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if a := r.active.Load(); a != nil && a.Version == version {
		return ErrActive
	}
	delete(r.byName, version)
	return nil
}

// Versions lists the loaded versions.
func (r *Registry) Versions() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]string, 0, len(r.byName))
	for v := range r.byName {
		out = append(out, v)
	}
	slices.Sort(out)
	return out
}

// WithLoader replaces how versions are read (tests).
func (r *Registry) WithLoader(load func(dir string) (*Bundle, error)) *Registry {
	r.load = load
	return r
}
