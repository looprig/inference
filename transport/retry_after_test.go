package transport

import (
	"net/http"
	"testing"
	"time"
)

func TestParseRetryAfter(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		val  string
		want time.Duration
	}{
		{"absent", "", 0},
		{"integer seconds", "30", 30 * time.Second},
		{"zero", "0", 0},
		{"negative rejected", "-5", 0},
		{"garbage rejected", "soon", 0},
		{"http-date rejected (unsupported by design)", "Fri, 08 Aug 2026 12:00:00 GMT", 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			h := http.Header{}
			if tc.val != "" {
				h.Set("Retry-After", tc.val)
			}
			if got := parseRetryAfter(h); got != tc.want {
				t.Fatalf("parseRetryAfter(%q) = %v, want %v", tc.val, got, tc.want)
			}
		})
	}
}
