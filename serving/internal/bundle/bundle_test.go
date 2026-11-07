package bundle

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeBundle(t *testing.T, files map[string]any, tamper func(dir string)) string {
	t.Helper()
	dir := t.TempDir()
	entries := map[string]fileEntry{}
	for name, v := range files {
		raw, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name), raw, 0o644); err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(raw)
		entries[name] = fileEntry{SHA256: hex.EncodeToString(sum[:]), Bytes: int64(len(raw))}
	}
	m, _ := json.Marshal(manifest{Format: Format, Version: "v-test", NumItems: 3, Files: entries})
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), m, 0o644); err != nil {
		t.Fatal(err)
	}
	if tamper != nil {
		tamper(dir)
	}
	link := filepath.Join(t.TempDir(), "current")
	if err := os.Symlink(dir, link); err != nil {
		t.Fatal(err)
	}
	return link
}

func goodFiles() map[string]any {
	return map[string]any{
		"pop.json": map[string]any{"items": []int{2, 0, 1}, "counts": []int{9, 5, 1}},
		"itemcf.json": map[string]any{
			"params":    map[string]any{"history_decay": 0.8},
			"neighbors": [][]int{{1, 2}, {0}, {}},
			"scores":    [][]float64{{0.5, 0.25}, {0.75}, {}},
		},
	}
}

func TestLoadResolvesTheSymlinkAndParses(t *testing.T) {
	b, err := Load(writeBundle(t, goodFiles(), nil))
	if err != nil {
		t.Fatal(err)
	}
	if b.Version != "v-test" || b.NumItems != 3 || b.Pop[0] != 2 || b.ItemCF.HistoryDecay != 0.8 {
		t.Fatalf("bundle %+v", b)
	}
	if len(b.ItemCF.Neighbors[0]) != 2 || b.ItemCF.Scores[1][0] != 0.75 {
		t.Fatalf("itemcf %+v", b.ItemCF)
	}
	if !strings.HasPrefix(b.Dir, "/") || filepath.Base(b.Dir) == "current" {
		t.Fatalf("Dir should be the resolved directory, got %s", b.Dir)
	}
}

func TestLoadRejectsBadBundles(t *testing.T) {
	cases := map[string]func(dir string){
		"tampered file": func(dir string) {
			_ = os.WriteFile(filepath.Join(dir, "pop.json"), []byte(`{"items":[0,1,2],"counts":[9,5,1]}`), 0o644)
		},
		"missing file": func(dir string) { _ = os.Remove(filepath.Join(dir, "itemcf.json")) },
		"bad format": func(dir string) {
			raw, _ := os.ReadFile(filepath.Join(dir, "manifest.json"))
			var m map[string]any
			_ = json.Unmarshal(raw, &m)
			m["format"] = 99
			raw, _ = json.Marshal(m)
			_ = os.WriteFile(filepath.Join(dir, "manifest.json"), raw, 0o644)
		},
	}
	for name, tamper := range cases {
		if _, err := Load(writeBundle(t, goodFiles(), tamper)); err == nil {
			t.Errorf("%s: loaded", name)
		}
	}
	bad := goodFiles()
	bad["itemcf.json"].(map[string]any)["neighbors"] = [][]int{{1, 7}, {0}, {}} // item 7 does not exist
	if _, err := Load(writeBundle(t, bad, nil)); err == nil {
		t.Error("out-of-range neighbour: loaded")
	}
}
