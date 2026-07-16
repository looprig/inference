package transport

import (
	"errors"
	"strings"
	"testing"

	"github.com/looprig/inference/auth"
	"github.com/looprig/inference/failure"

	"github.com/looprig/inference/codec/openaiapi"
	model "github.com/looprig/inference/model"
	"github.com/looprig/inference/route"
)

// TestCheckBinding locks the wildcard binding rule: an empty request field means "use
// the bound endpoint" (a wildcard), a matching value is fine, and a non-empty value that
// disagrees on provider, endpoint, or API format fails closed with a
// *failure.ModelMismatchError — all before any I/O.
func TestCheckBinding(t *testing.T) {
	t.Parallel()

	ep := Endpoint{
		Provider:  model.ProviderName("openrouter"),
		BaseURL:   "https://openrouter.ai/api/v1",
		APIFormat: model.APIFormatOpenAI,
	}
	c := New(ep, route.StaticChat("/chat/completions"), openaiapi.Codec{}, auth.None())

	tests := []struct {
		name    string
		model   model.Model
		wantErr bool
	}{
		{name: "all empty binds to endpoint", model: model.Model{Name: "m"}, wantErr: false},
		{name: "matching provider+base+format ok", model: model.Model{Name: "m", Provider: "openrouter", BaseURL: "https://openrouter.ai/api/v1", APIFormat: model.APIFormatOpenAI}, wantErr: false},
		{name: "empty provider is a wildcard", model: model.Model{Name: "m", Provider: "", BaseURL: "https://openrouter.ai/api/v1"}, wantErr: false},
		{name: "conflicting base fails", model: model.Model{Name: "m", BaseURL: "https://evil.example"}, wantErr: true},
		{name: "conflicting provider fails", model: model.Model{Name: "m", Provider: "chutes"}, wantErr: true},
		{name: "conflicting api format fails", model: model.Model{Name: "m", APIFormat: model.APIFormatAnthropic}, wantErr: true},
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
				var mme *failure.ModelMismatchError
				if !errors.As(err, &mme) {
					t.Fatalf("checkBinding() err=%T, want *failure.ModelMismatchError", err)
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
	ep := Endpoint{
		Provider:  model.ProviderName("openrouter"),
		BaseURL:   "https://openrouter.ai/api/v1",
		APIFormat: model.APIFormatOpenAI,
	}
	c := New(ep, route.StaticChat("/chat/completions"), openaiapi.Codec{}, auth.None())

	err := c.checkBinding(model.Model{Name: "m", APIFormat: model.APIFormatAnthropic})
	var mme *failure.ModelMismatchError
	if !errors.As(err, &mme) {
		t.Fatalf("checkBinding() err=%T, want *failure.ModelMismatchError", err)
	}
	if mme.RequestAPIFormat != model.APIFormatAnthropic || mme.BoundAPIFormat != model.APIFormatOpenAI {
		t.Errorf("APIFormat fields = req %q/bound %q, want %q/%q",
			mme.RequestAPIFormat, mme.BoundAPIFormat, model.APIFormatAnthropic, model.APIFormatOpenAI)
	}
	if !strings.Contains(mme.Error(), string(model.APIFormatAnthropic)) {
		t.Errorf("Error()=%q, want it to name the conflicting format %q", mme.Error(), model.APIFormatAnthropic)
	}
}
