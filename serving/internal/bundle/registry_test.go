package bundle_test

import (
	"errors"
	"path/filepath"
	"testing"

	"github.com/heyemlee/recloop/serving/internal/bundle"
)

func TestRegistryLoadsSwitchesAndKeepsTheActiveOne(t *testing.T) {
	v1 := &bundle.Bundle{Version: "v1"}
	loads := 0
	r := bundle.NewRegistry("/bundles", v1).WithLoader(func(dir string) (*bundle.Bundle, error) {
		loads++
		switch filepath.Base(dir) {
		case "v2":
			return &bundle.Bundle{Version: "v2"}, nil
		case "wrong":
			return &bundle.Bundle{Version: "v3"}, nil
		}
		return nil, errors.New("no such bundle")
	})
	if r.Active() != v1 || r.Get("v2") != nil {
		t.Fatal("initial state")
	}
	b, err := r.Ensure("v2")
	if err != nil || b.Version != "v2" {
		t.Fatal(b, err)
	}
	if _, err := r.Ensure("v2"); err != nil || loads != 1 {
		t.Fatalf("loaded twice (%d)", loads)
	}
	for _, bad := range []string{"wrong", "../etc", "", ".hidden", "nope"} {
		if _, err := r.Ensure(bad); err == nil {
			t.Fatalf("%q loaded", bad)
		}
	}
	if err := r.SetActive("v2"); err != nil || r.Active().Version != "v2" {
		t.Fatal(err)
	}
	if err := r.Unload("v2"); !errors.Is(err, bundle.ErrActive) {
		t.Fatalf("unloaded the active bundle: %v", err)
	}
	if err := r.Unload("v1"); err != nil || len(r.Versions()) != 1 {
		t.Fatal(r.Versions(), err)
	}
	if err := r.SetActive("v1"); err == nil {
		t.Fatal("switched to an unloaded version")
	}
}
