package inference_test

import (
	"errors"
	"testing"

	"github.com/looprig/inference"
)

func TestModelKey_Validate(t *testing.T) {
	t.Parallel()
	provider := inference.ProviderName("bedrock")
	tests := []struct {
		name       string
		key        inference.ModelKey
		wantField  inference.ModelKeyField
		wantReason inference.ModelKeyValidationReason
		wantErr    bool
	}{
		{
			name: "provider namespace and provider model id",
			key:  inference.ModelKey{Provider: provider, Model: "us.anthropic.claude-sonnet-4-20250514-v1:0"},
		},
		{
			name: "boundary one-character components",
			key:  inference.ModelKey{Provider: inference.ProviderName("p"), Model: "m"},
		},
		{
			name:       "empty provider namespace",
			key:        inference.ModelKey{Model: "m"},
			wantField:  inference.ModelKeyFieldProvider,
			wantReason: inference.ModelKeyValidationReasonEmpty,
			wantErr:    true,
		},
		{
			name:       "empty provider model id",
			key:        inference.ModelKey{Provider: inference.ProviderName("gemini")},
			wantField:  inference.ModelKeyFieldModel,
			wantReason: inference.ModelKeyValidationReasonEmpty,
			wantErr:    true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			err := tt.key.Validate()
			if (err != nil) != tt.wantErr {
				t.Fatalf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
			if !tt.wantErr {
				return
			}
			var validationErr *inference.ModelKeyValidationError
			if !errors.As(err, &validationErr) {
				t.Fatalf("Validate() error = %T, want *inference.ModelKeyValidationError", err)
			}
			if validationErr.Field != tt.wantField || validationErr.Reason != tt.wantReason {
				t.Errorf("Validate() error = %+v, want field %q reason %q", validationErr, tt.wantField, tt.wantReason)
			}
		})
	}
}

func TestModel_Key(t *testing.T) {
	t.Parallel()
	provider := inference.ProviderName("openrouter")
	tests := []struct {
		name  string
		model inference.Model
		want  inference.ModelKey
	}{
		{
			name: "projects provider namespace and provider model id",
			model: inference.Model{
				Provider:  provider,
				Name:      "anthropic/claude-sonnet-4",
				APIFormat: inference.APIFormatOpenAI,
				BaseURL:   "https://openrouter.ai/api/v1",
				Origin:    inference.OriginCatalog,
				Caps:      inference.Capabilities{Tools: true},
			},
			want: inference.ModelKey{Provider: provider, Model: "anthropic/claude-sonnet-4"},
		},
		{
			name:  "zero model projects invalid but deterministic zero key",
			model: inference.Model{},
			want:  inference.ModelKey{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := tt.model.Key(); got != tt.want {
				t.Errorf("Key() = %+v, want %+v", got, tt.want)
			}
		})
	}
}
