package api

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"slices"
	"strings"

	"github.com/heyemlee/recloop/serving/internal/experiment"
	"github.com/heyemlee/recloop/serving/internal/release"
	"github.com/heyemlee/recloop/serving/internal/store"
)

// Releaser is the release controller (release.Controller) as the admin api uses it.
type Releaser interface {
	Overview(ctx context.Context) (release.Overview, error)
	Register(ctx context.Context, version, actor string) error
	Validate(ctx context.Context, version, actor string) error
	Canary(ctx context.Context, version string, percent int, actor string) error
	AbortCanary(ctx context.Context, actor, note string) (string, error)
	Activate(ctx context.Context, version, actor, note string) (string, error)
	Rollback(ctx context.Context, actor, note string) (string, error)
	Experiments(ctx context.Context) ([]store.ExperimentRow, error)
	PutExperiment(ctx context.Context, e store.ExperimentRow) (int32, error)
	StopExperiment(ctx context.Context, id string) (bool, error)
	Fault(ctx context.Context, body map[string]any) (map[string]any, error)
}

// adminRoutes: the internal admin api behind the admin page (design §13.1, §22.1). Under
// /internal, which the web app's proxy never forwards (it forwards /api/v1/* only): the admin
// page reaches it through its own server-side route, and only when it is turned on.
//
//	GET  /internal/admin/models                     versions, events, what is loaded
//	POST /internal/admin/models                     {"version"}: register <bundles>/<version> as built
//	POST /internal/admin/models/{version}/validate  load and check it on both sides
//	POST /internal/admin/models/{version}/canary    {"percent": 5}: pin that share of users to it
//	POST /internal/admin/models/{version}/activate  make it the active version
//	POST /internal/admin/canary/abort               end the canary, retire its version
//	POST /internal/admin/rollback                   back to the version the active one replaced
//	GET  /internal/admin/experiments                every experiment
//	PUT  /internal/admin/experiments/{id}           create or replace one (running or stopped)
//	POST /internal/admin/experiments/{id}/stop
//	GET  /internal/admin/fault, PUT                 the model server's fault switch
//
// Every step takes "actor" and "note" in its body; they go into model_version_events.
func (s *Server) adminRoutes(mux *http.ServeMux) {
	if s.Release == nil {
		return
	}
	mux.HandleFunc("GET /internal/admin/models", s.adminOverview)
	mux.HandleFunc("POST /internal/admin/models", s.adminStep(func(ctx context.Context, b adminBody, _ string) (any, error) {
		return nil, s.Release.Register(ctx, b.Version, b.actor())
	}))
	mux.HandleFunc("POST /internal/admin/models/{version}/validate", s.adminStep(func(ctx context.Context, b adminBody, v string) (any, error) {
		return nil, s.Release.Validate(ctx, v, b.actor())
	}))
	mux.HandleFunc("POST /internal/admin/models/{version}/canary", s.adminStep(func(ctx context.Context, b adminBody, v string) (any, error) {
		p := b.Percent
		if p == 0 {
			p = 5
		}
		return nil, s.Release.Canary(ctx, v, p, b.actor())
	}))
	mux.HandleFunc("POST /internal/admin/models/{version}/activate", s.adminStep(func(ctx context.Context, b adminBody, v string) (any, error) {
		prev, err := s.Release.Activate(ctx, v, b.actor(), b.Note)
		return map[string]string{"active": v, "previous": prev}, err
	}))
	mux.HandleFunc("POST /internal/admin/canary/abort", s.adminStep(func(ctx context.Context, b adminBody, _ string) (any, error) {
		v, err := s.Release.AbortCanary(ctx, b.actor(), b.Note)
		return map[string]string{"retired": v}, err
	}))
	mux.HandleFunc("POST /internal/admin/rollback", s.adminStep(func(ctx context.Context, b adminBody, _ string) (any, error) {
		v, err := s.Release.Rollback(ctx, b.actor(), b.Note)
		return map[string]string{"active": v}, err
	}))
	mux.HandleFunc("GET /internal/admin/experiments", func(w http.ResponseWriter, r *http.Request) {
		exps, err := s.Release.Experiments(r.Context())
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		cfg := s.Exps.Current()
		version := 0
		if cfg != nil {
			version = cfg.Version
		}
		writeJSON(w, http.StatusOK, map[string]any{"experiments": exps, "loaded_config_version": version, "layers": experiment.Layers})
	})
	mux.HandleFunc("PUT /internal/admin/experiments/{id}", s.adminPutExperiment)
	mux.HandleFunc("POST /internal/admin/experiments/{id}/stop", func(w http.ResponseWriter, r *http.Request) {
		ok, err := s.Release.StopExperiment(r.Context(), r.PathValue("id"))
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]bool{"stopped": ok})
	})
	mux.HandleFunc("GET /internal/admin/fault", s.adminFault)
	mux.HandleFunc("PUT /internal/admin/fault", s.adminFault)
}

type adminBody struct {
	Version string `json:"version"`
	Percent int    `json:"percent"`
	Actor   string `json:"actor"`
	Note    string `json:"note"`
}

func (b adminBody) actor() string {
	if b.Actor == "" {
		return "admin-api"
	}
	return b.Actor
}

func (s *Server) adminOverview(w http.ResponseWriter, r *http.Request) {
	o, err := s.Release.Overview(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, o)
}

func (s *Server) adminStep(step func(ctx context.Context, b adminBody, version string) (any, error)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var b adminBody
		if r.ContentLength != 0 {
			if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10)).Decode(&b); err != nil {
				writeError(w, http.StatusBadRequest, "body: "+err.Error())
				return
			}
		}
		out, err := step(r.Context(), b, r.PathValue("version"))
		switch {
		case err == nil:
			if out == nil {
				out = map[string]string{"status": "ok"}
			}
			writeJSON(w, http.StatusOK, out)
		case errors.Is(err, release.ErrNotFound), errors.Is(err, store.ErrNoVersion):
			writeError(w, http.StatusNotFound, err.Error())
		case errors.Is(err, release.ErrState), errors.Is(err, release.ErrTooLate), errors.Is(err, release.ErrNoTarget):
			writeError(w, http.StatusConflict, err.Error())
		default:
			s.Log.Warn("admin step failed", "path", r.URL.Path, "err", err)
			writeError(w, http.StatusUnprocessableEntity, err.Error())
		}
	}
}

func (s *Server) adminPutExperiment(w http.ResponseWriter, r *http.Request) {
	var e store.ExperimentRow
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10)).Decode(&e); err != nil {
		writeError(w, http.StatusBadRequest, "body: "+err.Error())
		return
	}
	e.ID = r.PathValue("id")
	if e.Status == "" {
		e.Status = "running"
	}
	if err := validExperiment(e); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	v, err := s.Release.PutExperiment(r.Context(), e)
	if err != nil {
		if strings.Contains(err.Error(), "exclusion") || strings.Contains(err.Error(), "23P01") {
			writeError(w, http.StatusConflict, "overlaps a running experiment of the "+e.Layer+" layer: stop it first")
			return
		}
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"exp_id": e.ID, "config_version": v})
}

// validExperiment checks what PostgreSQL does not: the layer, the variants and their shares.
func validExperiment(e store.ExperimentRow) error {
	if e.ID == "" || len(e.ID) > 64 {
		return errors.New("exp_id: 1 to 64 characters")
	}
	if !slices.Contains(experiment.Layers, e.Layer) {
		return errors.New("layer must be one of " + strings.Join(experiment.Layers, ", "))
	}
	if e.Status != "running" && e.Status != "stopped" && e.Status != "draft" {
		return errors.New("status must be running, stopped or draft")
	}
	if e.BucketFrom < 0 || e.BucketFrom >= e.BucketTo || e.BucketTo > experiment.Buckets {
		return errors.New("buckets: 0 <= bucket_from < bucket_to <= 1000")
	}
	var vs []experiment.Variant
	if err := json.Unmarshal(e.Variants, &vs); err != nil || len(vs) == 0 {
		return errors.New("variants: a non-empty list of {name, share, params}")
	}
	total := 0
	names := map[string]bool{}
	for _, v := range vs {
		if v.Name == "" || names[v.Name] || v.Share <= 0 {
			return errors.New("variants: distinct non-empty names and positive shares")
		}
		names[v.Name] = true
		total += v.Share
		if len(v.Params) > 0 && !json.Valid(v.Params) {
			return errors.New("variants: params must be JSON")
		}
	}
	if total != 100 {
		return errors.New("variants: the shares must add up to 100")
	}
	return nil
}

func (s *Server) adminFault(w http.ResponseWriter, r *http.Request) {
	var body map[string]any
	if r.Method == http.MethodPut {
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4<<10)).Decode(&body); err != nil {
			writeError(w, http.StatusBadRequest, "body: "+err.Error())
			return
		}
	}
	out, err := s.Release.Fault(r.Context(), body)
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, out)
}
