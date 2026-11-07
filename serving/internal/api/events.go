package api

import (
	"encoding/json"
	"net/http"

	"github.com/heyemlee/recloop/serving/internal/events"
)

type rejected struct {
	Index int    `json:"index"`
	Error string `json:"error"`
}

// postEvents handles POST /v1/events: {"events": [...]} with at most 100 events. Invalid events
// are rejected individually (do not retry them); if storage fails the whole request returns 503
// and the client retries it with the same event_ids.
func (s *Server) postEvents(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 512<<10)
	var body struct {
		Events []events.ClientEvent `json:"events"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, "body must be {\"events\": [...]}: "+err.Error())
		return
	}
	if len(body.Events) == 0 || len(body.Events) > events.MaxBatch {
		writeError(w, http.StatusBadRequest, "send between 1 and 100 events")
		return
	}
	now := s.Now()
	valid := make([]events.Event, 0, len(body.Events))
	seen := map[string]bool{}
	rej := []rejected{}
	dupInBatch := 0
	for i, ce := range body.Events {
		ev, err := events.Validate(ce, now, s.Catalog.Has)
		if err != nil {
			rej = append(rej, rejected{Index: i, Error: err.Error()})
			continue
		}
		if seen[ev.ID.String()] {
			dupInBatch++
			continue
		}
		seen[ev.ID.String()] = true
		valid = append(valid, ev)
	}
	res := events.Result{}
	if len(valid) > 0 {
		var err error
		res, err = s.Events.Ingest(r.Context(), valid)
		if err != nil {
			s.Log.Error("event ingest failed", "events", len(valid), "err", err)
			writeError(w, http.StatusServiceUnavailable, "events not stored; retry with the same event_ids")
			return
		}
	}
	writeJSON(w, http.StatusAccepted, map[string]any{
		"accepted":   res.Accepted,
		"duplicates": res.Duplicates + dupInBatch,
		"rejected":   rej,
	})
}
