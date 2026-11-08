package release

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"
)

// HTTPModelServer talks to the model server's admin port (model_server/admin.py).
type HTTPModelServer struct {
	Base   string // e.g. http://127.0.0.1:18082
	Client *http.Client
}

func (m HTTPModelServer) client() *http.Client {
	if m.Client != nil {
		return m.Client
	}
	// Loading a bundle with wait=true includes its golden checks: a few seconds.
	return &http.Client{Timeout: 2 * time.Minute}
}

func (m HTTPModelServer) do(ctx context.Context, method, path string, body any, out any) (int, error) {
	var rd io.Reader
	if body != nil {
		raw, err := json.Marshal(body)
		if err != nil {
			return 0, err
		}
		rd = bytes.NewReader(raw)
	}
	req, err := http.NewRequestWithContext(ctx, method, m.Base+path, rd)
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := m.client().Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return resp.StatusCode, err
	}
	if out != nil && len(raw) > 0 {
		if err := json.Unmarshal(raw, out); err != nil {
			return resp.StatusCode, fmt.Errorf("model server %s %s: %w", method, path, err)
		}
	}
	if resp.StatusCode >= 400 {
		return resp.StatusCode, fmt.Errorf("model server %s %s: %d %s", method, path, resp.StatusCode, bytes.TrimSpace(raw))
	}
	return resp.StatusCode, nil
}

// Status implements ModelServer.
func (m HTTPModelServer) Status(ctx context.Context) (MSStatus, error) {
	var st MSStatus
	_, err := m.do(ctx, http.MethodGet, "/bundles", nil, &st)
	return st, err
}

// Load implements ModelServer; with wait, it returns once the load finished or failed.
func (m HTTPModelServer) Load(ctx context.Context, version string, wait bool) (string, error) {
	var out struct {
		State string `json:"state"`
	}
	code, err := m.do(ctx, http.MethodPost, "/bundles", map[string]any{"version": version, "wait": wait}, &out)
	if code == http.StatusUnprocessableEntity {
		return out.State, nil // failed validation: the state says why
	}
	return out.State, err
}

// SetDefault implements ModelServer.
func (m HTTPModelServer) SetDefault(ctx context.Context, version string) error {
	_, err := m.do(ctx, http.MethodPut, "/bundles/default", map[string]string{"version": version}, nil)
	return err
}

// Unload implements ModelServer.
func (m HTTPModelServer) Unload(ctx context.Context, version string) error {
	_, err := m.do(ctx, http.MethodDelete, "/bundles/"+url.PathEscape(version), nil, nil)
	return err
}

// Fault reads (body nil) or sets the model server's fault switch; it returns its settings.
func (m HTTPModelServer) Fault(ctx context.Context, body map[string]any) (map[string]any, error) {
	out := map[string]any{}
	method := http.MethodGet
	if body != nil {
		method = http.MethodPut
	}
	_, err := m.do(ctx, method, "/fault", body, &out)
	return out, err
}
