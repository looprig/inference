package inference_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
)

// fakeClient satisfies inference.Client for interface compliance testing.
type fakeClient struct{}

func (f *fakeClient) Invoke(_ context.Context, _ inference.Request) (*inference.Response, error) {
	return nil, nil
}

func (f *fakeClient) Stream(_ context.Context, _ inference.Request) (*inference.StreamReader[content.Chunk], error) {
	return nil, nil
}

// compile-time interface check
var _ inference.Client = (*fakeClient)(nil)

// sampleModel is a small local model builder standing in for a catalogue
// constructor: it yields a valid Model with an opaque ProviderName, used purely
// as a Request.Model fixture in this file.
func sampleModel() inference.Model {
	return inference.CustomModel(inference.ProviderName("chutes"), inference.APIFormatOpenAI, "https://api.chutes.ai", "moonshotai/Kimi-K2.6-TEE", inference.WithMaxContext(128_000), inference.WithTools(), inference.WithThinking())
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

	override := &inference.Sampling{Temperature: f64ptr(0.2)}
	req := inference.Request{
		Model:    sampleModel(),
		System:   "you are helpful",
		Messages: content.AgenticMessages{},
		Tools:    []inference.Tool{{Name: "search"}},
		Override: override,
	}

	if req.Model.Provider != inference.ProviderName("chutes") {
		t.Errorf("Request.Model.Provider = %q, want chutes", req.Model.Provider)
	}
	if req.System != "you are helpful" {
		t.Errorf("Request.System = %q, want %q", req.System, "you are helpful")
	}
	if len(req.Tools) != 1 || req.Tools[0].Name != "search" {
		t.Errorf("Request.Tools = %+v, want one tool named search", req.Tools)
	}
	if req.Override == nil || req.Override.Temperature == nil || *req.Override.Temperature != 0.2 {
		t.Errorf("Request.Override = %+v, want Temperature 0.2", req.Override)
	}

	// A nil Override is the documented "use Model.Sampling" default.
	def := inference.Request{Model: sampleModel()}
	if def.Override != nil {
		t.Errorf("zero-value Request.Override = %+v, want nil", def.Override)
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
		p    inference.ProviderName
		want string
	}{
		{name: "empty is a wildcard", p: inference.ProviderName(""), want: ""},
		{name: "arbitrary label", p: inference.ProviderName("chutes"), want: "chutes"},
		{name: "unknown label accepted", p: inference.ProviderName("totally-made-up"), want: "totally-made-up"},
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

	ep := inference.Endpoint{
		BaseURL:   "https://api.example.test/v1",
		Provider:  inference.ProviderName("custom"),
		APIFormat: inference.APIFormat("weird-dialect"),
	}
	if ep.BaseURL != "https://api.example.test/v1" {
		t.Errorf("Endpoint.BaseURL = %q, want https://api.example.test/v1", ep.BaseURL)
	}
	if ep.Provider != inference.ProviderName("custom") {
		t.Errorf("Endpoint.Provider = %q, want custom", ep.Provider)
	}
	if ep.APIFormat != inference.APIFormat("weird-dialect") {
		t.Errorf("Endpoint.APIFormat = %q, want weird-dialect", ep.APIFormat)
	}
}

func TestUsage(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name         string
		inputTokens  int
		outputTokens int
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
			u := inference.Usage{
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
