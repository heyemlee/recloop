package api

import "testing"

func TestAdsFollowTheDegradationChain(t *testing.T) {
	cases := map[string]struct {
		degraded []string
		want     bool
	}{
		"healthy":                  {nil, true},
		"text recall timed out":    {[]string{"text_timeout"}, true},
		"page store failed":        {[]string{"pages_timeout"}, true},
		"no user state":            {[]string{"user_state_timeout"}, false},
		"SASRec timed out":         {[]string{"sasrec_timeout"}, false},
		"model server breaker":     {[]string{"sasrec_circuit_open"}, false},
		"ranker timed out (M3/M4)": {[]string{"rank_timeout"}, false},
		"everything failed":        {[]string{"pop_skipped", "static_popular"}, false},
		"bundle mismatch":          {[]string{"model_version_mismatch"}, false},
	}
	for name, c := range cases {
		if got := adsAllowed(c.degraded); got != c.want {
			t.Errorf("%s: adsAllowed(%v) = %v", name, c.degraded, got)
		}
	}
}
