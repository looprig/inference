package route_test

import (
	"net/http"
	"testing"

	"github.com/looprig/inference"
	codec "github.com/looprig/inference/codec"
	route "github.com/looprig/inference/route"
)

// fakeRouter is a minimal Router used to prove the interface is satisfiable and that a caller
// can supply its own routing without importing any concrete builder. The concrete builders
// live in package route (Part B).
type fakeRouter struct{}

func (fakeRouter) BuildRoute(baseURL string, _ inference.Request, mode codec.RequestMode) (route.Route, error) {
	path := "/chat/completions"
	if mode == codec.RequestModeStream {
		path = "/chat/completions?stream=true"
	}
	h := http.Header{}
	h.Set("Content-Type", "application/json")
	return route.Route{Method: http.MethodPost, URL: baseURL + path, Header: h}, nil
}

// Compile-time assertion that fakeRouter satisfies Router.
var _ route.Router = fakeRouter{}

func TestRouter_Satisfiable(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name     string
		mode     codec.RequestMode
		wantURL  string
		wantMeth string
	}{
		{
			name:     "invoke mode builds static chat route",
			mode:     codec.RequestModeInvoke,
			wantURL:  "https://api.example.test/v1/chat/completions",
			wantMeth: http.MethodPost,
		},
		{
			name:     "stream mode may vary the route",
			mode:     codec.RequestModeStream,
			wantURL:  "https://api.example.test/v1/chat/completions?stream=true",
			wantMeth: http.MethodPost,
		},
	}

	var r route.Router = fakeRouter{}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			route, err := r.BuildRoute("https://api.example.test/v1", inference.Request{}, tc.mode)
			if err != nil {
				t.Fatalf("BuildRoute() error = %v", err)
			}
			if route.URL != tc.wantURL {
				t.Errorf("Route.URL = %q, want %q", route.URL, tc.wantURL)
			}
			if route.Method != tc.wantMeth {
				t.Errorf("Route.Method = %q, want %q", route.Method, tc.wantMeth)
			}
			if route.Header.Get("Content-Type") != "application/json" {
				t.Errorf("Route.Header Content-Type = %q, want application/json", route.Header.Get("Content-Type"))
			}
		})
	}
}
