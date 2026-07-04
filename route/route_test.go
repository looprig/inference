package route_test

import (
	"errors"
	"net/http"
	"testing"

	"github.com/looprig/inference"
	"github.com/looprig/inference/route"
)

// Compile-time proof the builders satisfy the Router contract.
var (
	_ inference.Router = route.StaticChat("/chat/completions")
	_ inference.Router = route.GeminiGenerateContent()
)

func TestStaticChat(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		path    string
		base    string
		mode    inference.RequestMode
		wantURL string
	}{
		{name: "openai invoke", path: "/chat/completions", base: "https://api.openai.test/v1", mode: inference.RequestModeInvoke, wantURL: "https://api.openai.test/v1/chat/completions"},
		{name: "openai stream same url", path: "/chat/completions", base: "https://api.openai.test/v1", mode: inference.RequestModeStream, wantURL: "https://api.openai.test/v1/chat/completions"},
		{name: "anthropic invoke", path: "/messages", base: "https://api.anthropic.test/v1", mode: inference.RequestModeInvoke, wantURL: "https://api.anthropic.test/v1/messages"},
		{name: "trailing slash on base trimmed", path: "/messages", base: "https://api.anthropic.test/v1/", mode: inference.RequestModeStream, wantURL: "https://api.anthropic.test/v1/messages"},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			r := route.StaticChat(tc.path)
			got, err := r.BuildRoute(tc.base, inference.Request{}, tc.mode)
			if err != nil {
				t.Fatalf("BuildRoute() error = %v", err)
			}
			if got.Method != http.MethodPost {
				t.Errorf("Method = %q, want POST", got.Method)
			}
			if got.URL != tc.wantURL {
				t.Errorf("URL = %q, want %q", got.URL, tc.wantURL)
			}
		})
	}
}

func TestGeminiGenerateContent(t *testing.T) {
	t.Parallel()

	model := func(name string) inference.Request {
		return inference.Request{Model: inference.Model{Name: name}}
	}

	cases := []struct {
		name    string
		req     inference.Request
		base    string
		mode    inference.RequestMode
		wantURL string
		wantErr bool
	}{
		{
			name:    "invoke uses generateContent with model in path",
			req:     model("gemini-2.5-pro"),
			base:    "https://gen.googleapis.test/v1beta",
			mode:    inference.RequestModeInvoke,
			wantURL: "https://gen.googleapis.test/v1beta/models/gemini-2.5-pro:generateContent",
		},
		{
			name:    "stream uses streamGenerateContent with alt=sse",
			req:     model("gemini-2.5-flash"),
			base:    "https://gen.googleapis.test/v1beta",
			mode:    inference.RequestModeStream,
			wantURL: "https://gen.googleapis.test/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse",
		},
		{
			name:    "trailing slash trimmed",
			req:     model("m"),
			base:    "https://gen.googleapis.test/v1beta/",
			mode:    inference.RequestModeInvoke,
			wantURL: "https://gen.googleapis.test/v1beta/models/m:generateContent",
		},
		{
			name:    "empty model name is an error",
			req:     model(""),
			base:    "https://gen.googleapis.test/v1beta",
			mode:    inference.RequestModeInvoke,
			wantErr: true,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			r := route.GeminiGenerateContent()
			got, err := r.BuildRoute(tc.base, tc.req, tc.mode)
			if (err != nil) != tc.wantErr {
				t.Fatalf("BuildRoute() error = %v, wantErr %v", err, tc.wantErr)
			}
			if tc.wantErr {
				var mme *route.MissingModelError
				if !errors.As(err, &mme) {
					t.Fatalf("error = %T, want *route.MissingModelError", err)
				}
				return
			}
			if got.Method != http.MethodPost {
				t.Errorf("Method = %q, want POST", got.Method)
			}
			if got.URL != tc.wantURL {
				t.Errorf("URL = %q, want %q", got.URL, tc.wantURL)
			}
		})
	}
}
