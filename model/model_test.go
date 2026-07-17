package model_test

import (
	"errors"
	"reflect"
	"testing"

	model "github.com/looprig/inference/model"
)

func f64ptr(v float64) *float64 { return &v }
func intptr(v int) *int         { return &v }

// TestModel_Validate exercises STRUCTURAL validation only: unknown provider/API-format
// labels are accepted, an empty BaseURL is a wildcard (accepted), a non-empty BaseURL is
// checked for syntactic safety (https-or-loopback-http, no userinfo, host present), and an
// empty Name is rejected.
func TestModel_Validate(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name    string
		model   model.Model
		wantErr bool
	}{
		{
			name:    "valid https accepts",
			model:   model.Model{Provider: model.ProviderName("chutes"), APIFormat: model.APIFormatOpenAI, BaseURL: "https://api.chutes.ai", Name: "moonshotai/Kimi-K2.6-TEE"},
			wantErr: false,
		},
		{
			name:    "unknown provider label accepts",
			model:   model.Model{Provider: model.ProviderName("totally-made-up"), APIFormat: model.APIFormatOpenAI, BaseURL: "https://x.example.test", Name: "m"},
			wantErr: false,
		},
		{
			name:    "empty provider label accepts (wildcard)",
			model:   model.Model{Provider: model.ProviderName(""), APIFormat: model.APIFormatOpenAI, BaseURL: "https://x.example.test", Name: "m"},
			wantErr: false,
		},
		{
			name:    "unknown APIFormat label accepts",
			model:   model.Model{Provider: model.ProviderName("custom"), APIFormat: model.APIFormat("some-custom-wire"), BaseURL: "https://x.example.test", Name: "m"},
			wantErr: false,
		},
		{
			name:    "empty APIFormat accepts",
			model:   model.Model{Provider: model.ProviderName("custom"), APIFormat: model.APIFormat(""), BaseURL: "https://x.example.test", Name: "m"},
			wantErr: false,
		},
		{
			name:    "provider-shaped-but-unbundled APIFormat accepts (no bedrock gate)",
			model:   model.Model{Provider: model.ProviderName("bedrock"), APIFormat: model.APIFormat("bedrock-converse"), BaseURL: "https://bedrock-runtime.us-east-1.amazonaws.com", Name: "m"},
			wantErr: false,
		},
		{
			name:    "empty BaseURL accepts (wildcard bound by client)",
			model:   model.Model{Provider: model.ProviderName("custom"), APIFormat: model.APIFormatOpenAI, BaseURL: "", Name: "m"},
			wantErr: false,
		},
		{
			name:    "unknown provider with empty BaseURL accepts (no provider gate)",
			model:   model.Model{Provider: model.ProviderName("totally-made-up"), APIFormat: model.APIFormatOpenAI, BaseURL: "", Name: "m"},
			wantErr: false,
		},
		{
			name:    "boundary http localhost accepts",
			model:   model.Model{Provider: model.ProviderName("lmstudio"), APIFormat: model.APIFormatOpenAI, BaseURL: "http://localhost:1234", Name: "qwen"},
			wantErr: false,
		},
		{
			name:    "boundary http 127.0.0.1 loopback accepts",
			model:   model.Model{Provider: model.ProviderName("lmstudio"), APIFormat: model.APIFormatOpenAI, BaseURL: "http://127.0.0.1:1234", Name: "qwen"},
			wantErr: false,
		},
		{
			name:    "boundary http uppercase LOCALHOST accepts",
			model:   model.Model{Provider: model.ProviderName("lmstudio"), APIFormat: model.APIFormatOpenAI, BaseURL: "http://LOCALHOST:1234", Name: "qwen"},
			wantErr: false,
		},
		{
			name:    "boundary http ipv6 loopback ::1 accepts",
			model:   model.Model{Provider: model.ProviderName("lmstudio"), APIFormat: model.APIFormatOpenAI, BaseURL: "http://[::1]:1234", Name: "qwen"},
			wantErr: false,
		},
		{
			name:    "error empty name",
			model:   model.Model{Provider: model.ProviderName("chutes"), APIFormat: model.APIFormatOpenAI, BaseURL: "https://api.chutes.ai", Name: ""},
			wantErr: true,
		},
		{
			name:    "error http to remote host",
			model:   model.Model{Provider: model.ProviderName("chutes"), APIFormat: model.APIFormatOpenAI, BaseURL: "http://api.chutes.ai", Name: "m"},
			wantErr: true,
		},
		{
			name:    "error http to non-loopback ip",
			model:   model.Model{Provider: model.ProviderName("lmstudio"), APIFormat: model.APIFormatOpenAI, BaseURL: "http://127.0.0.2:1234", Name: "m"},
			wantErr: true,
		},
		{
			name:    "error base url with userinfo credentials",
			model:   model.Model{Provider: model.ProviderName("phala"), APIFormat: model.APIFormatOpenAI, BaseURL: "https://user:pass@evil.example.com/v1", Name: "m"},
			wantErr: true,
		},
		{
			name:    "error base url not a url",
			model:   model.Model{Provider: model.ProviderName("chutes"), APIFormat: model.APIFormatOpenAI, BaseURL: "://not-a-url", Name: "m"},
			wantErr: true,
		},
		{
			name:    "error base url no scheme",
			model:   model.Model{Provider: model.ProviderName("chutes"), APIFormat: model.APIFormatOpenAI, BaseURL: "api.chutes.ai", Name: "m"},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			err := tt.model.Validate()
			if (err != nil) != tt.wantErr {
				t.Fatalf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				var ve *model.ValidationError
				if !errors.As(err, &ve) {
					t.Errorf("Validate() error is %T, want *model.ValidationError", err)
				}
			}
		})
	}
}

// TestCustomModel_Defaults confirms CustomModel sets only the four wire fields
// and leaves everything else at its fail-safe zero value: OriginCustom, all
// capabilities false, all context limits unknown, and an empty Sampling.
func TestCustomModel_Defaults(t *testing.T) {
	t.Parallel()
	m := model.CustomModel(model.ProviderName("chutes"), model.APIFormatOpenAI, "https://api.chutes.ai", "moonshotai/Kimi")

	if m.Provider != model.ProviderName("chutes") {
		t.Errorf("Provider = %q, want %q", m.Provider, "chutes")
	}
	if m.APIFormat != model.APIFormatOpenAI {
		t.Errorf("APIFormat = %q, want %q", m.APIFormat, model.APIFormatOpenAI)
	}
	if m.BaseURL != "https://api.chutes.ai" {
		t.Errorf("BaseURL = %q, want https://api.chutes.ai", m.BaseURL)
	}
	if m.Name != "moonshotai/Kimi" {
		t.Errorf("Name = %q, want moonshotai/Kimi", m.Name)
	}
	if m.Origin != model.OriginCustom {
		t.Errorf("Origin = %v, want OriginCustom (fail-safe default)", m.Origin)
	}
	if m.Caps.AcceptsImages || m.Caps.Tools || m.Caps.Thinking ||
		m.Caps.StructuredOutput || m.Caps.StructuredOutputWithTools {
		t.Errorf("Caps bool flags = %+v, want all false by default", m.Caps)
	}
	if m.Limits != (model.ContextLimits{}) {
		t.Errorf("Limits = %+v, want all limits unknown by default", m.Limits)
	}
	if m.Sampling.Temperature != nil || m.Sampling.TopP != nil || m.Sampling.MaxTokens != nil ||
		m.Sampling.Stop != nil || m.Sampling.Effort != model.EffortNone {
		t.Errorf("Sampling = %+v, want zero value by default", m.Sampling)
	}
}

// TestCustomModel_Options confirms each ModelOption opts a capability in.
func TestCustomModel_Options(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name  string
		opts  []model.ModelOption
		check func(t *testing.T, m model.Model)
	}{
		{
			name: "with context limits",
			opts: []model.ModelOption{model.WithContextLimits(model.ContextLimits{WindowTokens: 200_000, MaxOutputTokens: 16_384})},
			check: func(t *testing.T, m model.Model) {
				want := model.ContextLimits{WindowTokens: 200_000, MaxOutputTokens: 16_384}
				if m.Limits != want {
					t.Errorf("Limits = %+v, want %+v", m.Limits, want)
				}
			},
		},
		{
			name: "with tools",
			opts: []model.ModelOption{model.WithTools()},
			check: func(t *testing.T, m model.Model) {
				if !m.Caps.Tools {
					t.Error("Caps.Tools = false, want true")
				}
			},
		},
		{
			name: "with images",
			opts: []model.ModelOption{model.WithImages()},
			check: func(t *testing.T, m model.Model) {
				if !m.Caps.AcceptsImages {
					t.Error("Caps.AcceptsImages = false, want true")
				}
			},
		},
		{
			name: "with thinking",
			opts: []model.ModelOption{model.WithThinking()},
			check: func(t *testing.T, m model.Model) {
				if !m.Caps.Thinking {
					t.Error("Caps.Thinking = false, want true")
				}
			},
		},
		{
			name: "with structured output sets only structured output",
			opts: []model.ModelOption{model.WithStructuredOutput()},
			check: func(t *testing.T, m model.Model) {
				if !m.Caps.StructuredOutput {
					t.Error("Caps.StructuredOutput = false, want true")
				}
				if m.Caps.AcceptsImages || m.Caps.Tools || m.Caps.Thinking || m.Caps.StructuredOutputWithTools {
					t.Errorf("Caps = %+v, want only StructuredOutput enabled", m.Caps)
				}
			},
		},
		{
			name: "with structured output and tools sets required capabilities",
			opts: []model.ModelOption{model.WithStructuredOutputWithTools()},
			check: func(t *testing.T, m model.Model) {
				if !m.Caps.Tools || !m.Caps.StructuredOutput || !m.Caps.StructuredOutputWithTools {
					t.Errorf("Caps = %+v, want Tools, StructuredOutput, and StructuredOutputWithTools enabled", m.Caps)
				}
				if m.Caps.AcceptsImages || m.Caps.Thinking {
					t.Errorf("Caps = %+v, want unrelated capabilities disabled", m.Caps)
				}
			},
		},
		{
			name: "with sampling",
			opts: []model.ModelOption{model.WithSampling(model.Sampling{Temperature: f64ptr(0.5), MaxTokens: intptr(128), Effort: model.EffortHigh})},
			check: func(t *testing.T, m model.Model) {
				if m.Sampling.Temperature == nil || *m.Sampling.Temperature != 0.5 {
					t.Errorf("Sampling.Temperature = %v, want 0.5", m.Sampling.Temperature)
				}
				if m.Sampling.MaxTokens == nil || *m.Sampling.MaxTokens != 128 {
					t.Errorf("Sampling.MaxTokens = %v, want 128", m.Sampling.MaxTokens)
				}
				if m.Sampling.Effort != model.EffortHigh {
					t.Errorf("Sampling.Effort = %q, want high", m.Sampling.Effort)
				}
			},
		},
		{
			name: "combined options",
			opts: []model.ModelOption{model.WithTools(), model.WithImages(), model.WithThinking(), model.WithContextLimits(model.ContextLimits{WindowTokens: 64_000})},
			check: func(t *testing.T, m model.Model) {
				if !m.Caps.Tools || !m.Caps.AcceptsImages || !m.Caps.Thinking || m.Limits.WindowTokens != 64_000 {
					t.Errorf("combined model = %+v, want all capabilities and limits set", m)
				}
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			m := model.CustomModel(model.ProviderName("lmstudio"), model.APIFormatOpenAI, "http://localhost:1234", "qwen", tt.opts...)
			tt.check(t, m)
		})
	}
}

func TestModel_ValidateStructuredOutputCapabilities(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name      string
		caps      model.Capabilities
		wantField string
	}{
		{name: "zero value capabilities remain valid"},
		{name: "structured output alone is valid", caps: model.Capabilities{StructuredOutput: true}},
		{name: "tools alone is valid", caps: model.Capabilities{Tools: true}},
		{
			name: "independent tools and structured output are valid",
			caps: model.Capabilities{Tools: true, StructuredOutput: true},
		},
		{
			name: "structured output with tools is valid when prerequisites are set",
			caps: model.Capabilities{Tools: true, StructuredOutput: true, StructuredOutputWithTools: true},
		},
		{
			name:      "structured output with tools without tools is invalid",
			caps:      model.Capabilities{StructuredOutput: true, StructuredOutputWithTools: true},
			wantField: "Caps.StructuredOutputWithTools",
		},
		{
			name:      "structured output with tools without structured output is invalid",
			caps:      model.Capabilities{Tools: true, StructuredOutputWithTools: true},
			wantField: "Caps.StructuredOutputWithTools",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			err := (model.Model{Name: "m", Caps: tt.caps}).Validate()
			if tt.wantField == "" {
				if err != nil {
					t.Fatalf("Validate() error = %v, want nil", err)
				}
				return
			}
			var validationErr *model.ValidationError
			if !errors.As(err, &validationErr) {
				t.Fatalf("Validate() error = %T, want *model.ValidationError", err)
			}
			if validationErr.Field != tt.wantField {
				t.Errorf("ValidationError.Field = %q, want %q", validationErr.Field, tt.wantField)
			}
		})
	}
}

func TestModel_ValidateLimits(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name    string
		limits  model.ContextLimits
		wantErr bool
	}{
		{name: "unknown limits", limits: model.ContextLimits{}},
		{name: "valid known limits", limits: model.ContextLimits{WindowTokens: 100, MaxInputTokens: 90, MaxOutputTokens: 10}},
		{name: "invalid limits", limits: model.ContextLimits{WindowTokens: 100, MaxInputTokens: 101}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			candidate := model.Model{Name: "m", Limits: tt.limits}
			err := candidate.Validate()
			if (err != nil) != tt.wantErr {
				t.Fatalf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				var validationErr *model.ContextLimitsValidationError
				if !errors.As(err, &validationErr) {
					t.Errorf("Validate() error = %T, want *model.ContextLimitsValidationError", err)
				}
			}
		})
	}
}

func TestModel_Clone(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name     string
		model    model.Model
		wantZero bool
	}{
		{name: "zero model", model: model.Model{}, wantZero: true},
		{
			name: "all model and sampling state",
			model: model.Model{
				Provider:  model.ProviderName("p"),
				APIFormat: model.APIFormatOpenAI,
				BaseURL:   "https://api.example.test/v1",
				Name:      "provider/model",
				Origin:    model.OriginCatalog,
				Caps: model.Capabilities{
					AcceptsImages:             true,
					Tools:                     true,
					Thinking:                  true,
					StructuredOutput:          true,
					StructuredOutputWithTools: true,
				},
				Limits: model.ContextLimits{WindowTokens: 100, MaxInputTokens: 80, MaxOutputTokens: 20},
				Sampling: model.Sampling{
					Temperature: f64ptr(0.5),
					TopP:        f64ptr(0.75),
					MaxTokens:   intptr(16),
					Stop:        []string{"STOP", "END"},
					Effort:      model.EffortHigh,
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := tt.model.Clone()
			if tt.wantZero {
				if !reflect.DeepEqual(got, model.Model{}) {
					t.Fatalf("Clone() = %+v, want zero Model", got)
				}
				return
			}

			gotValues := got
			gotValues.Sampling = model.Sampling{}
			wantValues := tt.model
			wantValues.Sampling = model.Sampling{}
			if !reflect.DeepEqual(gotValues, wantValues) {
				t.Errorf("Clone() value fields = %+v, want %+v", gotValues, wantValues)
			}
			if !got.Caps.StructuredOutput || !got.Caps.StructuredOutputWithTools {
				t.Errorf("Clone().Caps = %+v, want structured output flags preserved", got.Caps)
			}
			if got.Sampling.Temperature == nil || *got.Sampling.Temperature != 0.5 || got.Sampling.Temperature == tt.model.Sampling.Temperature {
				t.Errorf("Clone().Sampling.Temperature = %v at %p, want independent 0.5", got.Sampling.Temperature, got.Sampling.Temperature)
			}
			if got.Sampling.TopP == nil || *got.Sampling.TopP != 0.75 || got.Sampling.TopP == tt.model.Sampling.TopP {
				t.Errorf("Clone().Sampling.TopP = %v at %p, want independent 0.75", got.Sampling.TopP, got.Sampling.TopP)
			}
			if got.Sampling.MaxTokens == nil || *got.Sampling.MaxTokens != 16 || got.Sampling.MaxTokens == tt.model.Sampling.MaxTokens {
				t.Errorf("Clone().Sampling.MaxTokens = %v at %p, want independent 16", got.Sampling.MaxTokens, got.Sampling.MaxTokens)
			}
			if !reflect.DeepEqual(got.Sampling.Stop, []string{"STOP", "END"}) {
				t.Errorf("Clone().Sampling.Stop = %v, want [STOP END]", got.Sampling.Stop)
			}
			if len(got.Sampling.Stop) == 0 || &got.Sampling.Stop[0] == &tt.model.Sampling.Stop[0] {
				t.Error("Clone().Sampling.Stop aliases source backing storage")
			}
			if got.Sampling.Effort != model.EffortHigh {
				t.Errorf("Clone().Sampling.Effort = %q, want %q", got.Sampling.Effort, model.EffortHigh)
			}

			*tt.model.Sampling.Temperature = 0.9
			*tt.model.Sampling.TopP = 0.1
			*tt.model.Sampling.MaxTokens = 99
			tt.model.Sampling.Stop[0] = "MUTATED"
			if *got.Sampling.Temperature != 0.5 || *got.Sampling.TopP != 0.75 || *got.Sampling.MaxTokens != 16 || got.Sampling.Stop[0] != "STOP" {
				t.Errorf("Clone().Sampling changed after source mutation: %+v", got.Sampling)
			}
		})
	}
}

// TestCustomModel_WithSamplingNoAlias guards that WithSampling deep-copies its
// argument so a later mutation of the caller's Sampling cannot reach the Model.
func TestCustomModel_WithSamplingNoAlias(t *testing.T) {
	t.Parallel()
	s := model.Sampling{Temperature: f64ptr(0.5), Stop: []string{"</s>"}}
	m := model.CustomModel(model.ProviderName("lmstudio"), model.APIFormatOpenAI, "http://localhost:1234", "qwen", model.WithSampling(s))

	*s.Temperature = 0.99
	s.Stop[0] = "MUTATED"

	if m.Sampling.Temperature == nil || *m.Sampling.Temperature != 0.5 {
		t.Errorf("Model.Sampling.Temperature aliased caller state: got %v, want 0.5", m.Sampling.Temperature)
	}
	if len(m.Sampling.Stop) != 1 || m.Sampling.Stop[0] != "</s>" {
		t.Errorf("Model.Sampling.Stop aliased caller state: got %v, want [</s>]", m.Sampling.Stop)
	}
}

// TestCustomModel_Validates confirms a custom model still passes through the same
// structural boundary rules: a well-formed one validates, an http-remote one is rejected.
func TestCustomModel_Validates(t *testing.T) {
	t.Parallel()
	ok := model.CustomModel(model.ProviderName("lmstudio"), model.APIFormatOpenAI, "http://localhost:1234", "qwen")
	if err := ok.Validate(); err != nil {
		t.Errorf("Validate() on valid custom model = %v, want nil", err)
	}

	bad := model.CustomModel(model.ProviderName("chutes"), model.APIFormatOpenAI, "http://api.chutes.ai", "m")
	if err := bad.Validate(); err == nil {
		t.Error("Validate() on http-remote custom model = nil, want error")
	}
}
