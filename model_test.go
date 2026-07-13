package inference_test

import (
	"errors"
	"reflect"
	"testing"

	"github.com/looprig/inference"
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
		model   inference.Model
		wantErr bool
	}{
		{
			name:    "valid https accepts",
			model:   inference.Model{Provider: inference.ProviderName("chutes"), APIFormat: inference.APIFormatOpenAI, BaseURL: "https://api.chutes.ai", Name: "moonshotai/Kimi-K2.6-TEE"},
			wantErr: false,
		},
		{
			name:    "unknown provider label accepts",
			model:   inference.Model{Provider: inference.ProviderName("totally-made-up"), APIFormat: inference.APIFormatOpenAI, BaseURL: "https://x.example.test", Name: "m"},
			wantErr: false,
		},
		{
			name:    "empty provider label accepts (wildcard)",
			model:   inference.Model{Provider: inference.ProviderName(""), APIFormat: inference.APIFormatOpenAI, BaseURL: "https://x.example.test", Name: "m"},
			wantErr: false,
		},
		{
			name:    "unknown APIFormat label accepts",
			model:   inference.Model{Provider: inference.ProviderName("custom"), APIFormat: inference.APIFormat("some-custom-wire"), BaseURL: "https://x.example.test", Name: "m"},
			wantErr: false,
		},
		{
			name:    "empty APIFormat accepts",
			model:   inference.Model{Provider: inference.ProviderName("custom"), APIFormat: inference.APIFormat(""), BaseURL: "https://x.example.test", Name: "m"},
			wantErr: false,
		},
		{
			name:    "provider-shaped-but-unbundled APIFormat accepts (no bedrock gate)",
			model:   inference.Model{Provider: inference.ProviderName("bedrock"), APIFormat: inference.APIFormat("bedrock-converse"), BaseURL: "https://bedrock-runtime.us-east-1.amazonaws.com", Name: "m"},
			wantErr: false,
		},
		{
			name:    "empty BaseURL accepts (wildcard bound by client)",
			model:   inference.Model{Provider: inference.ProviderName("custom"), APIFormat: inference.APIFormatOpenAI, BaseURL: "", Name: "m"},
			wantErr: false,
		},
		{
			name:    "unknown provider with empty BaseURL accepts (no provider gate)",
			model:   inference.Model{Provider: inference.ProviderName("totally-made-up"), APIFormat: inference.APIFormatOpenAI, BaseURL: "", Name: "m"},
			wantErr: false,
		},
		{
			name:    "boundary http localhost accepts",
			model:   inference.Model{Provider: inference.ProviderName("lmstudio"), APIFormat: inference.APIFormatOpenAI, BaseURL: "http://localhost:1234", Name: "qwen"},
			wantErr: false,
		},
		{
			name:    "boundary http 127.0.0.1 loopback accepts",
			model:   inference.Model{Provider: inference.ProviderName("lmstudio"), APIFormat: inference.APIFormatOpenAI, BaseURL: "http://127.0.0.1:1234", Name: "qwen"},
			wantErr: false,
		},
		{
			name:    "boundary http uppercase LOCALHOST accepts",
			model:   inference.Model{Provider: inference.ProviderName("lmstudio"), APIFormat: inference.APIFormatOpenAI, BaseURL: "http://LOCALHOST:1234", Name: "qwen"},
			wantErr: false,
		},
		{
			name:    "boundary http ipv6 loopback ::1 accepts",
			model:   inference.Model{Provider: inference.ProviderName("lmstudio"), APIFormat: inference.APIFormatOpenAI, BaseURL: "http://[::1]:1234", Name: "qwen"},
			wantErr: false,
		},
		{
			name:    "error empty name",
			model:   inference.Model{Provider: inference.ProviderName("chutes"), APIFormat: inference.APIFormatOpenAI, BaseURL: "https://api.chutes.ai", Name: ""},
			wantErr: true,
		},
		{
			name:    "error http to remote host",
			model:   inference.Model{Provider: inference.ProviderName("chutes"), APIFormat: inference.APIFormatOpenAI, BaseURL: "http://api.chutes.ai", Name: "m"},
			wantErr: true,
		},
		{
			name:    "error http to non-loopback ip",
			model:   inference.Model{Provider: inference.ProviderName("lmstudio"), APIFormat: inference.APIFormatOpenAI, BaseURL: "http://127.0.0.2:1234", Name: "m"},
			wantErr: true,
		},
		{
			name:    "error base url with userinfo credentials",
			model:   inference.Model{Provider: inference.ProviderName("phala"), APIFormat: inference.APIFormatOpenAI, BaseURL: "https://user:pass@evil.example.com/v1", Name: "m"},
			wantErr: true,
		},
		{
			name:    "error base url not a url",
			model:   inference.Model{Provider: inference.ProviderName("chutes"), APIFormat: inference.APIFormatOpenAI, BaseURL: "://not-a-url", Name: "m"},
			wantErr: true,
		},
		{
			name:    "error base url no scheme",
			model:   inference.Model{Provider: inference.ProviderName("chutes"), APIFormat: inference.APIFormatOpenAI, BaseURL: "api.chutes.ai", Name: "m"},
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
				var ve *inference.ValidationError
				if !errors.As(err, &ve) {
					t.Errorf("Validate() error is %T, want *inference.ValidationError", err)
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
	m := inference.CustomModel(inference.ProviderName("chutes"), inference.APIFormatOpenAI, "https://api.chutes.ai", "moonshotai/Kimi")

	if m.Provider != inference.ProviderName("chutes") {
		t.Errorf("Provider = %q, want %q", m.Provider, "chutes")
	}
	if m.APIFormat != inference.APIFormatOpenAI {
		t.Errorf("APIFormat = %q, want %q", m.APIFormat, inference.APIFormatOpenAI)
	}
	if m.BaseURL != "https://api.chutes.ai" {
		t.Errorf("BaseURL = %q, want https://api.chutes.ai", m.BaseURL)
	}
	if m.Name != "moonshotai/Kimi" {
		t.Errorf("Name = %q, want moonshotai/Kimi", m.Name)
	}
	if m.Origin != inference.OriginCustom {
		t.Errorf("Origin = %v, want OriginCustom (fail-safe default)", m.Origin)
	}
	if m.Caps.AcceptsImages || m.Caps.Tools || m.Caps.Thinking {
		t.Errorf("Caps bool flags = %+v, want all false by default", m.Caps)
	}
	if m.Limits != (inference.ContextLimits{}) {
		t.Errorf("Limits = %+v, want all limits unknown by default", m.Limits)
	}
	if m.Sampling.Temperature != nil || m.Sampling.TopP != nil || m.Sampling.MaxTokens != nil ||
		m.Sampling.Stop != nil || m.Sampling.Effort != inference.EffortNone {
		t.Errorf("Sampling = %+v, want zero value by default", m.Sampling)
	}
}

// TestCustomModel_Options confirms each ModelOption opts a capability in.
func TestCustomModel_Options(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name  string
		opts  []inference.ModelOption
		check func(t *testing.T, m inference.Model)
	}{
		{
			name: "with context limits",
			opts: []inference.ModelOption{inference.WithContextLimits(inference.ContextLimits{WindowTokens: 200_000, MaxOutputTokens: 16_384})},
			check: func(t *testing.T, m inference.Model) {
				want := inference.ContextLimits{WindowTokens: 200_000, MaxOutputTokens: 16_384}
				if m.Limits != want {
					t.Errorf("Limits = %+v, want %+v", m.Limits, want)
				}
			},
		},
		{
			name: "with tools",
			opts: []inference.ModelOption{inference.WithTools()},
			check: func(t *testing.T, m inference.Model) {
				if !m.Caps.Tools {
					t.Error("Caps.Tools = false, want true")
				}
			},
		},
		{
			name: "with images",
			opts: []inference.ModelOption{inference.WithImages()},
			check: func(t *testing.T, m inference.Model) {
				if !m.Caps.AcceptsImages {
					t.Error("Caps.AcceptsImages = false, want true")
				}
			},
		},
		{
			name: "with thinking",
			opts: []inference.ModelOption{inference.WithThinking()},
			check: func(t *testing.T, m inference.Model) {
				if !m.Caps.Thinking {
					t.Error("Caps.Thinking = false, want true")
				}
			},
		},
		{
			name: "with sampling",
			opts: []inference.ModelOption{inference.WithSampling(inference.Sampling{Temperature: f64ptr(0.5), MaxTokens: intptr(128), Effort: inference.EffortHigh})},
			check: func(t *testing.T, m inference.Model) {
				if m.Sampling.Temperature == nil || *m.Sampling.Temperature != 0.5 {
					t.Errorf("Sampling.Temperature = %v, want 0.5", m.Sampling.Temperature)
				}
				if m.Sampling.MaxTokens == nil || *m.Sampling.MaxTokens != 128 {
					t.Errorf("Sampling.MaxTokens = %v, want 128", m.Sampling.MaxTokens)
				}
				if m.Sampling.Effort != inference.EffortHigh {
					t.Errorf("Sampling.Effort = %q, want high", m.Sampling.Effort)
				}
			},
		},
		{
			name: "combined options",
			opts: []inference.ModelOption{inference.WithTools(), inference.WithImages(), inference.WithThinking(), inference.WithContextLimits(inference.ContextLimits{WindowTokens: 64_000})},
			check: func(t *testing.T, m inference.Model) {
				if !m.Caps.Tools || !m.Caps.AcceptsImages || !m.Caps.Thinking || m.Limits.WindowTokens != 64_000 {
					t.Errorf("combined model = %+v, want all capabilities and limits set", m)
				}
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			m := inference.CustomModel(inference.ProviderName("lmstudio"), inference.APIFormatOpenAI, "http://localhost:1234", "qwen", tt.opts...)
			tt.check(t, m)
		})
	}
}

func TestModel_ValidateLimits(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name    string
		limits  inference.ContextLimits
		wantErr bool
	}{
		{name: "unknown limits", limits: inference.ContextLimits{}},
		{name: "valid known limits", limits: inference.ContextLimits{WindowTokens: 100, MaxInputTokens: 90, MaxOutputTokens: 10}},
		{name: "invalid limits", limits: inference.ContextLimits{WindowTokens: 100, MaxInputTokens: 101}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			model := inference.Model{Name: "m", Limits: tt.limits}
			err := model.Validate()
			if (err != nil) != tt.wantErr {
				t.Fatalf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				var validationErr *inference.ContextLimitsValidationError
				if !errors.As(err, &validationErr) {
					t.Errorf("Validate() error = %T, want *inference.ContextLimitsValidationError", err)
				}
			}
		})
	}
}

func TestModel_Clone(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name     string
		model    inference.Model
		wantZero bool
	}{
		{name: "zero model", model: inference.Model{}, wantZero: true},
		{
			name: "all model and sampling state",
			model: inference.Model{
				Provider:  inference.ProviderName("p"),
				APIFormat: inference.APIFormatOpenAI,
				BaseURL:   "https://api.example.test/v1",
				Name:      "provider/model",
				Origin:    inference.OriginCatalog,
				Caps:      inference.Capabilities{AcceptsImages: true, Tools: true, Thinking: true},
				Limits:    inference.ContextLimits{WindowTokens: 100, MaxInputTokens: 80, MaxOutputTokens: 20},
				Sampling: inference.Sampling{
					Temperature: f64ptr(0.5),
					TopP:        f64ptr(0.75),
					MaxTokens:   intptr(16),
					Stop:        []string{"STOP", "END"},
					Effort:      inference.EffortHigh,
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := tt.model.Clone()
			if tt.wantZero {
				if !reflect.DeepEqual(got, inference.Model{}) {
					t.Fatalf("Clone() = %+v, want zero Model", got)
				}
				return
			}

			gotValues := got
			gotValues.Sampling = inference.Sampling{}
			wantValues := tt.model
			wantValues.Sampling = inference.Sampling{}
			if !reflect.DeepEqual(gotValues, wantValues) {
				t.Errorf("Clone() value fields = %+v, want %+v", gotValues, wantValues)
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
			if got.Sampling.Effort != inference.EffortHigh {
				t.Errorf("Clone().Sampling.Effort = %q, want %q", got.Sampling.Effort, inference.EffortHigh)
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
	s := inference.Sampling{Temperature: f64ptr(0.5), Stop: []string{"</s>"}}
	m := inference.CustomModel(inference.ProviderName("lmstudio"), inference.APIFormatOpenAI, "http://localhost:1234", "qwen", inference.WithSampling(s))

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
	ok := inference.CustomModel(inference.ProviderName("lmstudio"), inference.APIFormatOpenAI, "http://localhost:1234", "qwen")
	if err := ok.Validate(); err != nil {
		t.Errorf("Validate() on valid custom model = %v, want nil", err)
	}

	bad := inference.CustomModel(inference.ProviderName("chutes"), inference.APIFormatOpenAI, "http://api.chutes.ai", "m")
	if err := bad.Validate(); err == nil {
		t.Error("Validate() on http-remote custom model = nil, want error")
	}
}
