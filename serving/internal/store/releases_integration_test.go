package store_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/heyemlee/recloop/serving/internal/store"
	"github.com/heyemlee/recloop/serving/internal/testutil"
)

func status(t *testing.T, st *store.Store, version string) store.ModelVersion {
	t.Helper()
	all, err := st.ModelVersions(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range all {
		if v.Version == version {
			return v
		}
	}
	t.Fatalf("no version %s", version)
	return store.ModelVersion{}
}

// The release steps against PostgreSQL: one active version at a time, the previous one retired
// in the same transaction (its retired_at equals the new activated_at), every step recorded.
func TestModelReleaseTransitions(t *testing.T) {
	st := testutil.PG(t)
	ctx := context.Background()
	if _, err := st.Pool.Exec(ctx, "TRUNCATE model_version_events, model_versions"); err != nil {
		t.Fatal(err)
	}
	if err := st.RegisterActiveModel(ctx, "v0", "/bundles/v0", []byte(`{}`)); err != nil {
		t.Fatal(err)
	}
	if added, err := st.RegisterBuilt(ctx, "v1", "/bundles/v1", []byte(`{"version":"v1"}`), "test"); err != nil || !added {
		t.Fatal(added, err)
	}
	if added, _ := st.RegisterBuilt(ctx, "v1", "/bundles/v1", []byte(`{}`), "test"); added {
		t.Fatal("registered twice")
	}
	if _, err := st.Transition(ctx, "v1", []string{"validated"}, "canary", "test", ""); !errors.Is(err, store.ErrTransition) {
		t.Fatalf("canary of a built version: %v", err)
	}
	for _, step := range [][2]string{{"built", "validated"}, {"validated", "canary"}} {
		if _, err := st.Transition(ctx, "v1", []string{step[0]}, step[1], "test", ""); err != nil {
			t.Fatal(step, err)
		}
	}
	prev, err := st.Transition(ctx, "v1", []string{"canary"}, "active", "test", "full rollout")
	if err != nil || prev != "v0" {
		t.Fatal(prev, err)
	}
	v0, v1 := status(t, st, "v0"), status(t, st, "v1")
	if v0.Status != "retired" || v1.Status != "active" || v0.RetiredAt == nil || v1.ActivatedAt == nil || !v0.RetiredAt.Equal(*v1.ActivatedAt) {
		t.Fatalf("v0 %+v v1 %+v", v0, v1)
	}
	if v1.ValidatedAt == nil || v1.CanaryAt == nil {
		t.Fatal("step times not recorded")
	}
	// Rollback: v0 active again, its retired_at cleared; still one active version.
	if prev, err := st.Transition(ctx, "v0", []string{"retired"}, "active", "test", "rollback"); err != nil || prev != "v1" {
		t.Fatal(prev, err)
	}
	if status(t, st, "v0").RetiredAt != nil || status(t, st, "v1").Status != "retired" {
		t.Fatal("rollback state")
	}
	var active int
	if err := st.Pool.QueryRow(ctx, "SELECT count(*) FROM model_versions WHERE status = 'active'").Scan(&active); err != nil || active != 1 {
		t.Fatal(active, err)
	}
	// The unique index refuses a second active row however it is written.
	if _, err := st.Pool.Exec(ctx, "UPDATE model_versions SET status = 'active' WHERE version = 'v1'"); err == nil {
		t.Fatal("two active versions")
	}
	evs, err := st.ModelEvents(ctx, 100)
	if err != nil {
		t.Fatal(err)
	}
	// startup, built, validated, canary, v0 retired + v1 active, v1 retired + v0 active
	if len(evs) != 8 || evs[0].To != "active" || evs[0].Version != "v0" || evs[0].Note != "rollback" {
		t.Fatalf("%d events, latest %+v", len(evs), evs[0])
	}
}

func TestPutExperimentRespectsTheLayerExclusion(t *testing.T) {
	st := testutil.PG(t)
	ctx := context.Background()
	if _, err := st.Pool.Exec(ctx, "DELETE FROM experiments WHERE layer = 'rank'"); err != nil {
		t.Fatal(err)
	}
	variants, _ := json.Marshal([]map[string]any{{"name": "a", "share": 50}, {"name": "b", "share": 50}})
	v1, err := st.PutExperiment(ctx, store.ExperimentRow{ID: "rank-aa", Layer: "rank", Status: "running", BucketFrom: 0, BucketTo: 1000, Variants: variants})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := st.PutExperiment(ctx, store.ExperimentRow{ID: "rank-canary", Layer: "rank", Status: "running", BucketFrom: 0, BucketTo: 50, Variants: variants}); err == nil {
		t.Fatal("overlapping running experiments in one layer")
	}
	if ok, err := st.StopExperiment(ctx, "rank-aa"); err != nil || !ok {
		t.Fatal(ok, err)
	}
	v2, err := st.PutExperiment(ctx, store.ExperimentRow{ID: "rank-canary", Layer: "rank", Status: "running", BucketFrom: 0, BucketTo: 50, Variants: variants})
	if err != nil || v2 <= v1 {
		t.Fatal(v2, err)
	}
	exps, err := st.Experiments(ctx)
	if err != nil {
		t.Fatal(err)
	}
	running := 0
	for _, e := range exps {
		if e.Layer == "rank" && e.Status == "running" {
			running++
		}
	}
	if running != 1 {
		t.Fatalf("%d running rank experiments", running)
	}
}
