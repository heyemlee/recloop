package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/heyemlee/recloop/serving/internal/ads"
	"github.com/heyemlee/recloop/serving/internal/store"
)

// CampaignStore is the advertiser side of the database (store.Store).
type CampaignStore interface {
	CreateAdvertiser(ctx context.Context, name, brand string) (ads.Advertiser, error)
	Advertisers(ctx context.Context) ([]ads.Advertiser, error)
	CreateCampaign(ctx context.Context, advertiserID int32, set ads.Settings, today time.Time) (ads.Campaign, error)
	Campaigns(ctx context.Context, advertiserID *int32, today time.Time) ([]ads.Campaign, error)
	Campaign(ctx context.Context, id int64, today time.Time) (ads.Campaign, error)
	UpdateCampaign(ctx context.Context, id int64, version int32, patch ads.Patch,
		validate func(*ads.Settings) error, today time.Time) (ads.Campaign, error)
	CampaignReport(ctx context.Context, id int64, from, to time.Time) (store.Report, error)
}

// MaxReportDays bounds GET /v1/campaigns/{id}/report.
const MaxReportDays = 92

// campaignRoutes: the advertiser api (design §9.5, §13.1). Budget and bid changes use
// optimistic locking: PATCH must send If-Match with the ETag (the version) it read.
func (s *Server) campaignRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /v1/advertisers", s.listAdvertisers)
	mux.HandleFunc("POST /v1/advertisers", s.createAdvertiser)
	mux.HandleFunc("GET /v1/campaigns", s.listCampaigns)
	mux.HandleFunc("POST /v1/campaigns", s.createCampaign)
	mux.HandleFunc("GET /v1/campaigns/{id}", s.getCampaign)
	mux.HandleFunc("PATCH /v1/campaigns/{id}", s.patchCampaign)
	mux.HandleFunc("DELETE /v1/campaigns/{id}", s.endCampaign)
	mux.HandleFunc("GET /v1/campaigns/{id}/report", s.campaignReport)
}

func utcDay(t time.Time) time.Time {
	t = t.UTC()
	return time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, time.UTC)
}

func etag(version int32) string { return `"` + strconv.Itoa(int(version)) + `"` }

// ifMatch reads the version in If-Match: "3" (or 3). ok is false when the header is absent.
func ifMatch(r *http.Request) (version int32, ok bool, err error) {
	h := strings.TrimSpace(r.Header.Get("If-Match"))
	if h == "" {
		return 0, false, nil
	}
	v, err := strconv.ParseInt(strings.Trim(h, `"`), 10, 32)
	if err != nil || v < 1 {
		return 0, true, fmt.Errorf("If-Match must be the campaign's ETag, e.g. \"3\"")
	}
	return int32(v), true, nil
}

func decodeStrict(w http.ResponseWriter, r *http.Request, v any) error {
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10))
	dec.DisallowUnknownFields()
	return dec.Decode(v)
}

func (s *Server) writeCampaign(w http.ResponseWriter, status int, c ads.Campaign) {
	w.Header().Set("ETag", etag(c.Version))
	writeJSON(w, status, c)
}

// campaignError maps store and validation errors to HTTP statuses.
func (s *Server) campaignError(w http.ResponseWriter, r *http.Request, err error) {
	var ve *ads.ValidationError
	switch {
	case errors.As(err, &ve):
		writeError(w, http.StatusBadRequest, ve.Msg)
	case errors.Is(err, store.ErrNotFound):
		writeError(w, http.StatusNotFound, "no such campaign")
	case errors.Is(err, ads.ErrEnded):
		writeError(w, http.StatusConflict, err.Error())
	default:
		s.Log.Error("campaign request failed", "method", r.Method, "path", r.URL.Path, "err", err)
		writeError(w, http.StatusServiceUnavailable, "campaign store unavailable")
	}
}

func campaignID(w http.ResponseWriter, r *http.Request) (int64, bool) {
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil || id < 1 {
		writeError(w, http.StatusBadRequest, "campaign id must be a positive integer")
		return 0, false
	}
	return id, true
}

func (s *Server) validator() func(*ads.Settings) error {
	return func(set *ads.Settings) error { return set.Validate(s.Catalog.Has) }
}

// ---------------------------------------------------------------- advertisers

func (s *Server) listAdvertisers(w http.ResponseWriter, r *http.Request) {
	list, err := s.Campaigns.Advertisers(r.Context())
	if err != nil {
		s.campaignError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, list)
}

func (s *Server) createAdvertiser(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Name  string `json:"name"`
		Brand string `json:"brand"`
	}
	if err := decodeStrict(w, r, &body); err != nil {
		writeError(w, http.StatusBadRequest, "body must be {\"name\": ..., \"brand\": ...}: "+err.Error())
		return
	}
	body.Name, body.Brand = strings.TrimSpace(body.Name), strings.TrimSpace(body.Brand)
	if body.Name == "" || len(body.Name) > ads.MaxNameLen || len(body.Brand) > ads.MaxNameLen {
		writeError(w, http.StatusBadRequest, fmt.Sprintf("name is required; name and brand are at most %d characters", ads.MaxNameLen))
		return
	}
	a, err := s.Campaigns.CreateAdvertiser(r.Context(), body.Name, body.Brand)
	if err != nil {
		s.campaignError(w, r, err)
		return
	}
	writeJSON(w, http.StatusCreated, a)
}

// ---------------------------------------------------------------- campaigns

func (s *Server) listCampaigns(w http.ResponseWriter, r *http.Request) {
	var adv *int32
	if v := r.URL.Query().Get("advertiser_id"); v != "" {
		id, err := strconv.ParseInt(v, 10, 32)
		if err != nil {
			writeError(w, http.StatusBadRequest, "advertiser_id must be an integer")
			return
		}
		a := int32(id)
		adv = &a
	}
	list, err := s.Campaigns.Campaigns(r.Context(), adv, s.Now())
	if err != nil {
		s.campaignError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, list)
}

func (s *Server) createCampaign(w http.ResponseWriter, r *http.Request) {
	var body struct {
		AdvertiserID int32 `json:"advertiser_id"`
		ads.Settings
	}
	if err := decodeStrict(w, r, &body); err != nil {
		writeError(w, http.StatusBadRequest, "bad campaign: "+err.Error())
		return
	}
	set := body.Settings
	if set.Status == "" {
		set.Status = ads.StatusActive
	}
	if set.FreqCapPerDay == 0 {
		set.FreqCapPerDay = 3
	}
	if set.Status == ads.StatusEnded {
		writeError(w, http.StatusBadRequest, "a new campaign is active or paused")
		return
	}
	if err := set.Validate(s.Catalog.Has); err != nil {
		s.campaignError(w, r, err)
		return
	}
	c, err := s.Campaigns.CreateCampaign(r.Context(), body.AdvertiserID, set, s.Now())
	if err != nil {
		s.campaignError(w, r, err)
		return
	}
	w.Header().Set("Location", fmt.Sprintf("/v1/campaigns/%d", c.ID))
	s.writeCampaign(w, http.StatusCreated, c)
}

func (s *Server) getCampaign(w http.ResponseWriter, r *http.Request) {
	id, ok := campaignID(w, r)
	if !ok {
		return
	}
	c, err := s.Campaigns.Campaign(r.Context(), id, s.Now())
	if err != nil {
		s.campaignError(w, r, err)
		return
	}
	s.writeCampaign(w, http.StatusOK, c)
}

// patchCampaign: PATCH /v1/campaigns/{id} with If-Match. 412 when the campaign changed since
// that version (the body is the current campaign, its ETag the new version to retry with);
// 428 without If-Match.
func (s *Server) patchCampaign(w http.ResponseWriter, r *http.Request) {
	id, ok := campaignID(w, r)
	if !ok {
		return
	}
	version, present, err := ifMatch(r)
	switch {
	case err != nil:
		writeError(w, http.StatusBadRequest, err.Error())
		return
	case !present:
		writeError(w, http.StatusPreconditionRequired, "send If-Match with the ETag of the campaign you read")
		return
	}
	var patch ads.Patch
	if err := decodeStrict(w, r, &patch); err != nil {
		writeError(w, http.StatusBadRequest, "bad patch: "+err.Error())
		return
	}
	c, err := s.Campaigns.UpdateCampaign(r.Context(), id, version, patch, s.validator(), s.Now())
	if errors.Is(err, store.ErrVersionMismatch) {
		w.Header().Set("ETag", etag(c.Version))
		writeJSON(w, http.StatusPreconditionFailed, map[string]any{
			"error":    fmt.Sprintf("the campaign changed since version %d; it is at version %d now", version, c.Version),
			"campaign": c,
		})
		return
	}
	if err != nil {
		s.campaignError(w, r, err)
		return
	}
	s.writeCampaign(w, http.StatusOK, c)
}

// endCampaign: DELETE /v1/campaigns/{id} ends the campaign (status ended, terminal). The row
// stays because the ledger refers to it. Idempotent; If-Match is optional.
func (s *Server) endCampaign(w http.ResponseWriter, r *http.Request) {
	id, ok := campaignID(w, r)
	if !ok {
		return
	}
	version, present, err := ifMatch(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	end := ads.Patch{Status: ads.Opt[string]{Set: true, V: ads.StatusEnded}}
	noCheck := func(*ads.Settings) error { return nil } // ending needs no valid settings
	for attempt := 0; ; attempt++ {
		cur, err := s.Campaigns.Campaign(r.Context(), id, s.Now())
		if err != nil {
			s.campaignError(w, r, err)
			return
		}
		if present && cur.Version != version {
			w.Header().Set("ETag", etag(cur.Version))
			writeJSON(w, http.StatusPreconditionFailed, map[string]any{
				"error": "the campaign changed since that version", "campaign": cur,
			})
			return
		}
		if cur.Status == ads.StatusEnded {
			s.writeCampaign(w, http.StatusOK, cur)
			return
		}
		c, err := s.Campaigns.UpdateCampaign(r.Context(), id, cur.Version, end, noCheck, s.Now())
		if errors.Is(err, store.ErrVersionMismatch) && !present && attempt < 3 {
			continue // changed under us: read it again
		}
		if err != nil {
			s.campaignError(w, r, err)
			return
		}
		s.writeCampaign(w, http.StatusOK, c)
		return
	}
}

// campaignReport: GET /v1/campaigns/{id}/report?from=YYYY-MM-DD&to=YYYY-MM-DD (UTC days,
// default the last 7 days).
func (s *Server) campaignReport(w http.ResponseWriter, r *http.Request) {
	id, ok := campaignID(w, r)
	if !ok {
		return
	}
	q := r.URL.Query()
	to := s.Now().UTC()
	if v := q.Get("to"); v != "" {
		t, err := time.Parse(time.DateOnly, v)
		if err != nil {
			writeError(w, http.StatusBadRequest, "to must be YYYY-MM-DD")
			return
		}
		to = t
	}
	from := to.AddDate(0, 0, -6)
	if v := q.Get("from"); v != "" {
		t, err := time.Parse(time.DateOnly, v)
		if err != nil {
			writeError(w, http.StatusBadRequest, "from must be YYYY-MM-DD")
			return
		}
		from = t
	}
	from, to = utcDay(from), utcDay(to)
	days := int(to.Sub(from).Hours()/24) + 1
	if from.After(to) || days > MaxReportDays {
		writeError(w, http.StatusBadRequest, fmt.Sprintf("from must not be after to, and at most %d days", MaxReportDays))
		return
	}
	rep, err := s.Campaigns.CampaignReport(r.Context(), id, from, to)
	if err != nil {
		s.campaignError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, rep)
}
