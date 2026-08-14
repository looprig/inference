package inference_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	model "github.com/looprig/inference/model"
	stream "github.com/looprig/inference/stream"
	"github.com/looprig/inference/transport"

	usage "github.com/looprig/inference/usage"
)

// fakeClient satisfies inference.Client for interface compliance testing.
type fakeClient struct{}

func f64ptr(v float64) *float64 { return &v }

func (f *fakeClient) Invoke(_ context.Context, _ inference.Request) (*inference.Response, error) {
	return nil, nil
}

func (f *fakeClient) Stream(_ context.Context, _ inference.Request) (*stream.StreamReader[content.Chunk], error) {
	return nil, nil
}

// compile-time interface check
var _ inference.Client = (*fakeClient)(nil)

// sampleModel is a small local model builder standing in for a catalogue
// constructor: it yields a valid Model with an opaque ProviderName, used purely
// as a Request.Model fixture in this file.
func sampleModel() model.Model {
	return model.CustomModel(model.ProviderName("chutes"), model.APIFormatOpenAI, "https://api.chutes.ai", "moonshotai/Kimi-K2.6-TEE", model.WithContextLimits(model.ContextLimits{WindowTokens: 128_000}), model.WithTools(), model.WithThinking())
}

func TestClient_InterfaceCompliance(t *testing.T) {
	t.Parallel()
	// compile-time assertion is at the top of the file via var _ inference.Client = (*fakeClient)(nil).
	// This runtime test confirms the type is instantiable and usable as the interface.
	var iface inference.Client = &fakeClient{}
	ctx := context.Background()

	resp, err := iface.Invoke(ctx, inference.Request{})
	if err != nil {
		t.Fatalf("fakeClient.Invoke returned unexpected error: %v", err)
	}
	if resp != nil {
		t.Errorf("fakeClient.Invoke returned non-nil response, want nil")
	}

	sr, err := iface.Stream(ctx, inference.Request{})
	if err != nil {
		t.Fatalf("fakeClient.Stream returned unexpected error: %v", err)
	}
	if sr != nil {
		t.Errorf("fakeClient.Stream returned non-nil StreamReader, want nil")
	}
}

// TestRequest_Fields verifies a Request carries a secret-free Model, a per-agent
// System prompt, messages, tools, and an optional per-call Sampling override.
func TestRequest_Fields(t *testing.T) {
	t.Parallel()

	override := &model.Sampling{Temperature: f64ptr(0.2)}
	req := inference.Request{
		Model:  sampleModel(),
		System: "you are helpful",
		Messages: content.AgenticMessages{&content.UserMessage{Message: content.Message{
			Role:   content.RoleUser,
			Blocks: []content.Block{&content.TextBlock{Text: "hello"}},
		}}},
		TransientMessages: 1,
		Tools:             []inference.Tool{{Name: "search"}},
		Output:            &inference.OutputSchema{Name: "answer"},
		ToolChoice:        inference.ToolRequired(),
		Override:          override,
	}

	if req.Model.Provider != model.ProviderName("chutes") {
		t.Errorf("Request.Model.Provider = %q, want chutes", req.Model.Provider)
	}
	if req.System != "you are helpful" {
		t.Errorf("Request.System = %q, want %q", req.System, "you are helpful")
	}
	if req.TransientMessages != 1 {
		t.Errorf("Request.TransientMessages = %d, want 1", req.TransientMessages)
	}
	if len(req.Tools) != 1 || req.Tools[0].Name != "search" {
		t.Errorf("Request.Tools = %+v, want one tool named search", req.Tools)
	}
	if req.Output == nil || req.Output.Name != "answer" {
		t.Errorf("Request.Output = %+v, want output named answer", req.Output)
	}
	if req.ToolChoice != inference.ToolRequired() {
		t.Errorf("Request.ToolChoice = %v, want ToolRequired()", req.ToolChoice)
	}
	if req.Override == nil || req.Override.Temperature == nil || *req.Override.Temperature != 0.2 {
		t.Errorf("Request.Override = %+v, want Temperature 0.2", req.Override)
	}

	// A nil Override is the documented "use Model.Sampling" default.
	def := inference.Request{Model: sampleModel()}
	if def.Override != nil {
		t.Errorf("zero-value Request.Override = %+v, want nil", def.Override)
	}
	if def.Output != nil {
		t.Errorf("zero-value Request.Output = %+v, want nil", def.Output)
	}
	if def.ToolChoice != inference.ToolAuto() {
		t.Errorf("zero-value Request.ToolChoice = %v, want ToolAuto()", def.ToolChoice)
	}
}

// TestToolChoiceVariants pins the constructed vocabulary: the zero value is
// auto, the forced tool name is reachable only through the named variant, and
// a ToolChoice is comparable so callers can keep using ==.
func TestToolChoiceVariants(t *testing.T) {
	t.Parallel()

	var zero inference.ToolChoice
	if zero != inference.ToolAuto() {
		t.Errorf("zero ToolChoice = %v, want ToolAuto()", zero)
	}
	if zero.Mode() != inference.ToolChoiceModeAuto {
		t.Errorf("zero ToolChoice.Mode() = %v, want ToolChoiceAuto", zero.Mode())
	}

	tests := []struct {
		name      string
		choice    inference.ToolChoice
		wantMode  inference.ToolChoiceMode
		wantName  string
		wantNamed bool
	}{
		{name: "auto", choice: inference.ToolAuto(), wantMode: inference.ToolChoiceModeAuto},
		{name: "required", choice: inference.ToolRequired(), wantMode: inference.ToolChoiceModeRequired},
		{
			name:      "named",
			choice:    inference.ToolNamed("search"),
			wantMode:  inference.ToolChoiceModeNamed,
			wantName:  "search",
			wantNamed: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			if got := tt.choice.Mode(); got != tt.wantMode {
				t.Errorf("Mode() = %v, want %v", got, tt.wantMode)
			}
			gotName, gotNamed := tt.choice.Named()
			if gotName != tt.wantName || gotNamed != tt.wantNamed {
				t.Errorf("Named() = %q, %t, want %q, %t", gotName, gotNamed, tt.wantName, tt.wantNamed)
			}
		})
	}

	// ToolChoice must stay comparable: Request literals and assertions across
	// the workspace use ==, and a future multi-name variant carrying a slice
	// would silently end that. Built through separate locals because
	// staticcheck reads a literal self-comparison as SA4000 and cannot see
	// that comparability itself is the property under test.
	firstNamed, secondNamed := inference.ToolNamed("a"), inference.ToolNamed("a")
	if firstNamed != secondNamed {
		t.Error("ToolNamed(a) != ToolNamed(a), want equal values")
	}
	if inference.ToolNamed("a") == inference.ToolNamed("b") {
		t.Error("ToolNamed(a) == ToolNamed(b), want distinct values")
	}
	if inference.ToolRequired() == inference.ToolAuto() {
		t.Error("ToolRequired() == ToolAuto(), want distinct values")
	}
}

func TestValidateRequestFeaturesTransientMessages(t *testing.T) {
	t.Parallel()

	oneMessage := content.AgenticMessages{&content.UserMessage{}}
	tests := []struct {
		name             string
		req              inference.Request
		wantErr          bool
		wantTransient    int
		wantMessageCount int
	}{
		{
			name: "zero is valid",
			req:  inference.Request{},
		},
		{
			name: "one trailing transient message is valid",
			req: inference.Request{
				Messages:          oneMessage,
				TransientMessages: 1,
			},
		},
		{
			name:             "negative is invalid",
			req:              inference.Request{TransientMessages: -1},
			wantErr:          true,
			wantTransient:    -1,
			wantMessageCount: 0,
		},
		{
			name: "count past messages is invalid",
			req: inference.Request{
				Messages:          oneMessage,
				TransientMessages: 2,
			},
			wantErr:          true,
			wantTransient:    2,
			wantMessageCount: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			err := inference.ValidateRequestFeatures(tt.req)
			if !tt.wantErr {
				if err != nil {
					t.Fatalf("ValidateRequestFeatures() error = %v, want nil", err)
				}
				return
			}

			var target *inference.InvalidTransientMessagesError
			if !errors.As(err, &target) {
				t.Fatalf("ValidateRequestFeatures() error = %T %v, want *InvalidTransientMessagesError", err, err)
			}
			if target.Transient != tt.wantTransient {
				t.Errorf("InvalidTransientMessagesError.Transient = %d, want %d", target.Transient, tt.wantTransient)
			}
			if target.Messages != tt.wantMessageCount {
				t.Errorf("InvalidTransientMessagesError.Messages = %d, want %d", target.Messages, tt.wantMessageCount)
			}
			if got := target.Error(); got != "inference: transient message count is outside request messages" {
				t.Errorf("InvalidTransientMessagesError.Error() = %q, want stable contract message", got)
			}
		})
	}
}

func TestValidateRequestFeatures(t *testing.T) {
	t.Parallel()

	output := validOutput(`{"type":"object","properties":{},"required":[],"additionalProperties":false}`)
	tool := inference.Tool{Name: "search"}
	outputCap := model.Capabilities{StructuredOutput: true}
	combinedCaps := model.Capabilities{
		Tools:                     true,
		StructuredOutput:          true,
		StructuredOutputWithTools: true,
	}

	tests := []struct {
		name                string
		req                 inference.Request
		wantUnsupported     bool
		wantCombined        bool
		wantConflict        bool
		wantSchemaError     bool
		wantConflictFeature string
	}{
		{name: "nil output and automatic tool choice", req: inference.Request{}},
		{
			name: "supported output only",
			req:  inference.Request{Model: model.Model{Name: "supported", Caps: outputCap}, Output: &output},
		},
		{
			name:            "missing base capability",
			req:             inference.Request{Model: model.Model{Name: "plain"}, Output: &output},
			wantUnsupported: true,
		},
		{
			name:         "output with tools missing combined capability",
			req:          inference.Request{Model: model.Model{Name: "base-only", Caps: outputCap}, Tools: []inference.Tool{tool}, Output: &output},
			wantCombined: true,
		},
		{
			name: "combined output with tools accepted",
			req:  inference.Request{Model: model.Model{Name: "combined", Caps: combinedCaps}, Tools: []inference.Tool{tool}, Output: &output},
		},
		{
			name:                "required tool choice without tools",
			req:                 inference.Request{ToolChoice: inference.ToolRequired()},
			wantConflict:        true,
			wantConflictFeature: "tool_choice_required_without_tools",
		},
		{
			name: "required tool choice with tools",
			req:  inference.Request{Tools: []inference.Tool{tool}, ToolChoice: inference.ToolRequired()},
		},
		{
			name: "named tool choice naming a declared tool",
			req:  inference.Request{Tools: []inference.Tool{tool}, ToolChoice: inference.ToolNamed("search")},
		},
		{
			name:                "named tool choice without tools",
			req:                 inference.Request{ToolChoice: inference.ToolNamed("search")},
			wantConflict:        true,
			wantConflictFeature: "tool_choice_tool_undeclared_name",
		},
		{
			name:                "named tool choice naming an undeclared tool",
			req:                 inference.Request{Tools: []inference.Tool{tool}, ToolChoice: inference.ToolNamed("other")},
			wantConflict:        true,
			wantConflictFeature: "tool_choice_tool_undeclared_name",
		},
		{
			// The empty name is the one defective named choice a constructor
			// can still build. It needs no error code of its own: no request
			// may declare an unnamed tool, so the declared-tool check is what
			// rejects it, and the diagnostic it produces is the true one.
			name:                "named tool choice with an empty name",
			req:                 inference.Request{Tools: []inference.Tool{tool}, ToolChoice: inference.ToolNamed("")},
			wantConflict:        true,
			wantConflictFeature: "tool_choice_tool_undeclared_name",
		},
		{
			name: "invalid schema propagates shared validation error",
			req: inference.Request{
				Model:  model.Model{Name: "supported", Caps: outputCap},
				Output: &inference.OutputSchema{Name: "answer", Schema: json.RawMessage(`{"type":"array"}`)},
			},
			wantSchemaError: true,
		},
		{
			name: "reserved terminal tool collision",
			req: inference.Request{
				Model:  model.Model{Name: "combined", Caps: combinedCaps},
				Tools:  []inference.Tool{{Name: inference.StructuredOutputToolName}},
				Output: &output,
			},
			wantConflict:        true,
			wantConflictFeature: "reserved_structured_output_tool",
		},
		{
			name: "duplicate tool names",
			req: inference.Request{
				Model:  model.Model{Name: "combined", Caps: combinedCaps},
				Tools:  []inference.Tool{tool, tool},
				Output: &output,
			},
			wantConflict:        true,
			wantConflictFeature: "duplicate_tool_name",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			err := inference.ValidateRequestFeatures(tt.req)
			if tt.wantUnsupported {
				var target *inference.StructuredOutputUnsupportedError
				if !errors.As(err, &target) {
					t.Fatalf("ValidateRequestFeatures() error = %T %v, want *StructuredOutputUnsupportedError", err, err)
				}
				if target.Model != tt.req.Model.Name {
					t.Errorf("StructuredOutputUnsupportedError.Model = %q, want %q", target.Model, tt.req.Model.Name)
				}
				return
			}
			if tt.wantCombined {
				var target *inference.StructuredOutputWithToolsUnsupportedError
				if !errors.As(err, &target) {
					t.Fatalf("ValidateRequestFeatures() error = %T %v, want *StructuredOutputWithToolsUnsupportedError", err, err)
				}
				if target.Model != tt.req.Model.Name {
					t.Errorf("StructuredOutputWithToolsUnsupportedError.Model = %q, want %q", target.Model, tt.req.Model.Name)
				}
				return
			}
			if tt.wantConflict {
				var target *inference.StructuredOutputConflictError
				if !errors.As(err, &target) {
					t.Fatalf("ValidateRequestFeatures() error = %T %v, want *StructuredOutputConflictError", err, err)
				}
				if target.Feature != tt.wantConflictFeature {
					t.Errorf("StructuredOutputConflictError.Feature = %q, want %q", target.Feature, tt.wantConflictFeature)
				}
				return
			}
			if tt.wantSchemaError {
				var target *inference.SchemaValidationError
				if !errors.As(err, &target) {
					t.Fatalf("ValidateRequestFeatures() error = %T %v, want propagated *SchemaValidationError", err, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("ValidateRequestFeatures() error = %v, want nil", err)
			}
		})
	}
}

func TestValidateRequestFeaturesEmptyModelDoesNotExposePayloads(t *testing.T) {
	t.Parallel()

	const secret = "schema-secret-do-not-log"
	output := validOutput(`{"type":"object","description":"` + secret + `","properties":{},"required":[],"additionalProperties":false}`)
	err := inference.ValidateRequestFeatures(inference.Request{Output: &output})

	var target *inference.StructuredOutputUnsupportedError
	if !errors.As(err, &target) {
		t.Fatalf("ValidateRequestFeatures() error = %T %v, want *StructuredOutputUnsupportedError", err, err)
	}
	if target.Model != "" {
		t.Errorf("StructuredOutputUnsupportedError.Model = %q, want empty", target.Model)
	}
	if strings.Contains(err.Error(), secret) || strings.Contains(err.Error(), string(output.Schema)) {
		t.Errorf("ValidateRequestFeatures() error exposed schema payload: %q", err)
	}
}

func TestValidateRequestFeaturesBoundsUnsupportedModelMetadata(t *testing.T) {
	t.Parallel()

	output := validOutput(`{"type":"object","properties":{},"required":[],"additionalProperties":false}`)
	oversized := strings.Repeat("世", inference.MaxStructuredOutputDiagnosticBytes)
	tests := []struct {
		name string
		req  inference.Request
		get  func(error) string
	}{
		{
			name: "base capability",
			req:  inference.Request{Model: model.Model{Name: oversized}, Output: &output},
			get: func(err error) string {
				var target *inference.StructuredOutputUnsupportedError
				if !errors.As(err, &target) {
					t.Fatalf("error = %T %v, want *StructuredOutputUnsupportedError", err, err)
				}
				return target.Model
			},
		},
		{
			name: "combined capability",
			req: inference.Request{
				Model:  model.Model{Name: oversized, Caps: model.Capabilities{StructuredOutput: true}},
				Tools:  []inference.Tool{{Name: "search"}},
				Output: &output,
			},
			get: func(err error) string {
				var target *inference.StructuredOutputWithToolsUnsupportedError
				if !errors.As(err, &target) {
					t.Fatalf("error = %T %v, want *StructuredOutputWithToolsUnsupportedError", err, err)
				}
				return target.Model
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.get(inference.ValidateRequestFeatures(tt.req))
			if len(got) > inference.MaxStructuredOutputDiagnosticBytes {
				t.Fatalf("diagnostic model length = %d, max %d", len(got), inference.MaxStructuredOutputDiagnosticBytes)
			}
			if !utf8.ValidString(got) {
				t.Fatalf("diagnostic model is invalid UTF-8: %q", got)
			}
			if got == oversized {
				t.Fatal("diagnostic model retained oversized input")
			}
		})
	}
}

func TestTool_Schema(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name   string
		schema json.RawMessage
	}{
		{
			name:   "object schema",
			schema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`),
		},
		{
			name:   "empty object",
			schema: json.RawMessage(`{}`),
		},
		{
			name:   "array schema",
			schema: json.RawMessage(`{"type":"array","items":{"type":"number"}}`),
		},
		{
			name:   "nil schema",
			schema: nil,
		},
		{
			name:   "string literal",
			schema: json.RawMessage(`"hello"`),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			tool := inference.Tool{
				Name:        "search",
				Description: "searches the web",
				Schema:      tc.schema,
			}
			if string(tool.Schema) != string(tc.schema) {
				t.Errorf("Tool.Schema = %q, want %q", tool.Schema, tc.schema)
			}
		})
	}
}

// TestProviderName_OpaqueLabel confirms ProviderName is a bare string label with
// no inference-level constants or policy: any value round-trips as itself.
func TestProviderName_OpaqueLabel(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		p    model.ProviderName
		want string
	}{
		{name: "empty is a wildcard", p: model.ProviderName(""), want: ""},
		{name: "arbitrary label", p: model.ProviderName("chutes"), want: "chutes"},
		{name: "unknown label accepted", p: model.ProviderName("totally-made-up"), want: "totally-made-up"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if string(tc.p) != tc.want {
				t.Errorf("ProviderName = %q, want %q", tc.p, tc.want)
			}
		})
	}
}

// TestEndpoint_Fields confirms Endpoint carries only base URL plus opaque
// provider/API-format labels, and no chat path.
func TestEndpoint_Fields(t *testing.T) {
	t.Parallel()

	ep := transport.Endpoint{
		BaseURL:   "https://api.example.test/v1",
		Provider:  model.ProviderName("custom"),
		APIFormat: model.APIFormat("weird-dialect"),
	}
	if ep.BaseURL != "https://api.example.test/v1" {
		t.Errorf("Endpoint.BaseURL = %q, want https://api.example.test/v1", ep.BaseURL)
	}
	if ep.Provider != model.ProviderName("custom") {
		t.Errorf("Endpoint.Provider = %q, want custom", ep.Provider)
	}
	if ep.APIFormat != model.APIFormat("weird-dialect") {
		t.Errorf("Endpoint.APIFormat = %q, want weird-dialect", ep.APIFormat)
	}
}

func TestUsage(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name         string
		inputTokens  content.TokenCount
		outputTokens content.TokenCount
	}{
		{name: "zero", inputTokens: 0, outputTokens: 0},
		{name: "positive", inputTokens: 100, outputTokens: 50},
		{name: "large", inputTokens: 1_000_000, outputTokens: 999_999},
		{name: "input only", inputTokens: 42, outputTokens: 0},
		{name: "output only", inputTokens: 0, outputTokens: 7},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			u := usage.Usage{
				InputTokens:  tc.inputTokens,
				OutputTokens: tc.outputTokens,
			}
			if u.InputTokens != tc.inputTokens {
				t.Errorf("Usage.InputTokens = %d, want %d", u.InputTokens, tc.inputTokens)
			}
			if u.OutputTokens != tc.outputTokens {
				t.Errorf("Usage.OutputTokens = %d, want %d", u.OutputTokens, tc.outputTokens)
			}
		})
	}
}
