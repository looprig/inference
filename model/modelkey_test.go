package model_test

import (
	"errors"
	"testing"

	model "github.com/looprig/inference/model"
)

func TestModelKey_Validate(t *testing.T) {
	t.Parallel()
	provider := model.ProviderName("bedrock")
	tests := []struct {
		name       string
		key        model.ModelKey
		wantField  model.ModelKeyField
		wantReason model.ModelKeyValidationReason
		wantErr    bool
	}{
		{
			name: "provider namespace and provider model id",
			key:  model.ModelKey{Provider: provider, Model: "us.anthropic.claude-sonnet-4-20250514-v1:0"},
		},
		{
			name: "boundary one-character components",
			key:  model.ModelKey{Provider: model.ProviderName("p"), Model: "m"},
		},
		{
			name:       "empty provider namespace",
			key:        model.ModelKey{Model: "m"},
			wantField:  model.ModelKeyFieldProvider,
			wantReason: model.ModelKeyValidationReasonEmpty,
			wantErr:    true,
		},
		{
			name:       "empty provider model id",
			key:        model.ModelKey{Provider: model.ProviderName("gemini")},
			wantField:  model.ModelKeyFieldModel,
			wantReason: model.ModelKeyValidationReasonEmpty,
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
			var validationErr *model.ModelKeyValidationError
			if !errors.As(err, &validationErr) {
				t.Fatalf("Validate() error = %T, want *model.ModelKeyValidationError", err)
			}
			if validationErr.Field != tt.wantField || validationErr.Reason != tt.wantReason {
				t.Errorf("Validate() error = %+v, want field %q reason %q", validationErr, tt.wantField, tt.wantReason)
			}
		})
	}
}

func TestModel_Key(t *testing.T) {
	t.Parallel()
	provider := model.ProviderName("openrouter")
	tests := []struct {
		name  string
		model model.Model
		want  model.ModelKey
	}{
		{
			name: "projects provider namespace and provider model id",
			model: model.Model{
				Provider:  provider,
				Name:      "anthropic/claude-sonnet-4",
				APIFormat: model.APIFormatOpenAI,
				BaseURL:   "https://openrouter.ai/api/v1",
				Origin:    model.OriginCatalog,
				Caps:      model.Capabilities{Tools: true},
			},
			want: model.ModelKey{Provider: provider, Model: "anthropic/claude-sonnet-4"},
		},
		{
			name:  "zero model projects invalid but deterministic zero key",
			model: model.Model{},
			want:  model.ModelKey{},
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
