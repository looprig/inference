package transport

import (
	"errors"
	"strings"
	"testing"

	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	"github.com/looprig/inference/codec/openaiapi"
	"github.com/looprig/inference/route"
)

// TestCheckBinding locks the wildcard binding rule: an empty request field means "use
// the bound endpoint" (a wildcard), a matching value is fine, and a non-empty value that
// disagrees on provider, endpoint, or API format fails closed with a
// *inference.ModelMismatchError — all before any I/O.
func TestCheckBinding(t *testing.T) {
	t.Parallel()

	ep := inference.Endpoint{
		Provider:  inference.ProviderName("openrouter"),
		BaseURL:   "https://openrouter.ai/api/v1",
		APIFormat: inference.APIFormatOpenAI,
	}
	c := New(ep, route.StaticChat("/chat/completions"), openaiapi.Codec{}, auth.None())

	tests := []struct {
		name    string
		model   inference.Model
		wantErr bool
	}{
		{name: "all empty binds to endpoint", model: inference.Model{Name: "m"}, wantErr: false},
		{name: "matching provider+base+format ok", model: inference.Model{Name: "m", Provider: "openrouter", BaseURL: "https://openrouter.ai/api/v1", APIFormat: inference.APIFormatOpenAI}, wantErr: false},
		{name: "empty provider is a wildcard", model: inference.Model{Name: "m", Provider: "", BaseURL: "https://openrouter.ai/api/v1"}, wantErr: false},
		{name: "conflicting base fails", model: inference.Model{Name: "m", BaseURL: "https://evil.example"}, wantErr: true},
		{name: "conflicting provider fails", model: inference.Model{Name: "m", Provider: "chutes"}, wantErr: true},
		{name: "conflicting api format fails", model: inference.Model{Name: "m", APIFormat: inference.APIFormatAnthropic}, wantErr: true},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			err := c.checkBinding(tt.model)
			if (err != nil) != tt.wantErr {
				t.Fatalf("checkBinding() err=%v wantErr=%v", err, tt.wantErr)
			}
			if tt.wantErr {
				var mme *inference.ModelMismatchError
				if !errors.As(err, &mme) {
					t.Fatalf("checkBinding() err=%T, want *inference.ModelMismatchError", err)
				}
			}
		})
	}
}

// TestModelMismatchError_APIFormatFieldsPopulated verifies an API-format-only conflict
// (provider and BaseURL are wildcards) names the format dimension on the returned error,
// not just fails closed. The error previously carried no APIFormat fields, so a
// format-only mismatch reported empty provider/endpoint and hid the real cause.
func TestModelMismatchError_APIFormatFieldsPopulated(t *testing.T) {
	t.Parallel()
	ep := inference.Endpoint{
		Provider:  inference.ProviderName("openrouter"),
		BaseURL:   "https://openrouter.ai/api/v1",
		APIFormat: inference.APIFormatOpenAI,
	}
	c := New(ep, route.StaticChat("/chat/completions"), openaiapi.Codec{}, auth.None())

	err := c.checkBinding(inference.Model{Name: "m", APIFormat: inference.APIFormatAnthropic})
	var mme *inference.ModelMismatchError
	if !errors.As(err, &mme) {
		t.Fatalf("checkBinding() err=%T, want *inference.ModelMismatchError", err)
	}
	if mme.RequestAPIFormat != inference.APIFormatAnthropic || mme.BoundAPIFormat != inference.APIFormatOpenAI {
		t.Errorf("APIFormat fields = req %q/bound %q, want %q/%q",
			mme.RequestAPIFormat, mme.BoundAPIFormat, inference.APIFormatAnthropic, inference.APIFormatOpenAI)
	}
	if !strings.Contains(mme.Error(), string(inference.APIFormatAnthropic)) {
		t.Errorf("Error()=%q, want it to name the conflicting format %q", mme.Error(), inference.APIFormatAnthropic)
	}
}
