package openaiapi_test

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/conformance"
	"github.com/looprig/inference/codec/openaiapi"
	model "github.com/looprig/inference/model"
)

// repeat builds a name of exactly n legal characters.
func repeat(n int) string { return strings.Repeat("a", n) }

func samplingModel(s model.Sampling) model.Model {
	return model.Model{Name: "gpt-test", Sampling: s}
}

func float64Ptr(v float64) *float64 { return &v }

// --- Tool names ------------------------------------------------------------
//
// Spec: components.schemas.FunctionObject.name — "The name of the function to
// be called. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a
// maximum length of 64." (openai/openai-openapi, OpenAPI 3.1).
//
// The constraint is PROSE only: the derived request document types `name` as a
// bare string with no `pattern` and no `maxLength`, so conformance's gate
// accepts every value below (measured, see
// TestTheChatRequestGateHoldsSamplingButNotToolNames). This rule is therefore
// carried by the encoder alone.

func TestEncodeRequestRejectsIllegalToolNames(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		request inference.Request
		want    string
	}{
		{
			name: "declared tool with a dot",
			request: inference.Request{
				Model:    model.Model{Name: "gpt-test"},
				Messages: content.AgenticMessages{userMsg(textBlock("hi"))},
				Tools:    []inference.Tool{{Name: "mcp.search", Schema: json.RawMessage(`{"type":"object"}`)}},
			},
			want: "mcp.search",
		},
		{
			name: "declared tool with a slash",
			request: inference.Request{
				Model:    model.Model{Name: "gpt-test"},
				Messages: content.AgenticMessages{userMsg(textBlock("hi"))},
				Tools:    []inference.Tool{{Name: "server/tool", Schema: json.RawMessage(`{"type":"object"}`)}},
			},
			want: "server/tool",
		},
		{
			name: "declared tool with a space",
			request: inference.Request{
				Model:    model.Model{Name: "gpt-test"},
				Messages: content.AgenticMessages{userMsg(textBlock("hi"))},
				Tools:    []inference.Tool{{Name: "read file", Schema: json.RawMessage(`{"type":"object"}`)}},
			},
			want: "read file",
		},
		{
			name: "declared tool one character over the 64-character cap",
			request: inference.Request{
				Model:    model.Model{Name: "gpt-test"},
				Messages: content.AgenticMessages{userMsg(textBlock("hi"))},
				Tools:    []inference.Tool{{Name: repeat(65), Schema: json.RawMessage(`{"type":"object"}`)}},
			},
			want: repeat(65),
		},
		{
			name: "declared tool with an empty name",
			request: inference.Request{
				Model:    model.Model{Name: "gpt-test"},
				Messages: content.AgenticMessages{userMsg(textBlock("hi"))},
				Tools:    []inference.Tool{{Name: "", Schema: json.RawMessage(`{"type":"object"}`)}},
			},
			want: "",
		},
		{
			name: "replayed tool call naming an illegal function",
			request: inference.Request{
				Model: model.Model{Name: "gpt-test"},
				Messages: content.AgenticMessages{
					userMsg(textBlock("hi")),
					aiMsg(toolUseBlock("call-1", "mcp.search", json.RawMessage(`{}`))),
				},
			},
			want: "mcp.search",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			body, err := openaiapi.EncodeRequest(tt.request, false)
			if err == nil {
				t.Fatalf("EncodeRequest() accepted a tool name FunctionObject.name forbids; body = %s", body)
			}
			var invalid *openaiapi.InvalidToolNameError
			if !errors.As(err, &invalid) {
				t.Fatalf("EncodeRequest() error = %v (%T), want *openaiapi.InvalidToolNameError", err, err)
			}
			if invalid.Name != tt.want {
				t.Errorf("InvalidToolNameError.Name = %q, want %q", invalid.Name, tt.want)
			}
			if invalid.Reason == "" {
				t.Error("InvalidToolNameError.Reason is empty; a local error must name the violated constraint")
			}
		})
	}
}

// TestEncodeRequestAcceptsLegalToolNames is the positive control on the rule
// above: a rejection rule that also rejects valid input is worse than the bug
// it closes. Every name here is inside FunctionObject.name's published class,
// including the exact 64-character boundary.
func TestEncodeRequestAcceptsLegalToolNames(t *testing.T) {
	t.Parallel()

	for _, name := range []string{"read", "read_file", "read-file", "Read2", "a", repeat(64)} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			body, err := openaiapi.EncodeRequest(inference.Request{
				Model: model.Model{Name: "gpt-test"},
				Messages: content.AgenticMessages{
					userMsg(textBlock("hi")),
					aiMsg(toolUseBlock("call-1", name, json.RawMessage(`{}`))),
				},
				Tools:      []inference.Tool{{Name: name, Schema: json.RawMessage(`{"type":"object"}`)}},
				ToolChoice: inference.ToolNamed(name),
			}, false)
			if err != nil {
				t.Fatalf("EncodeRequest() rejected the legal tool name %q: %v", name, err)
			}
			conformance.MustValidateRequest(t, "openai", kindChatCompletionRequest, body)
		})
	}
}

// TestEncodeRequestNamedToolChoiceInheritsTheRule pins why the encoder does NOT
// re-check ChatCompletionNamedToolChoice.function.name. A forced choice must
// name a declared tool — ValidateRequestFeatures refuses it otherwise — and
// every declared tool is held to the class, so the choice cannot carry an
// illegal name that the tools array did not carry first. If either half of that
// pairing ever changes, this test says so.
func TestEncodeRequestNamedToolChoiceInheritsTheRule(t *testing.T) {
	t.Parallel()

	// Declared AND chosen: the tools loop is the thing that refuses.
	_, err := openaiapi.EncodeRequest(inference.Request{
		Model:      model.Model{Name: "gpt-test"},
		Messages:   content.AgenticMessages{userMsg(textBlock("hi"))},
		Tools:      []inference.Tool{{Name: "mcp.search", Schema: json.RawMessage(`{"type":"object"}`)}},
		ToolChoice: inference.ToolNamed("mcp.search"),
	}, false)
	var invalid *openaiapi.InvalidToolNameError
	if !errors.As(err, &invalid) {
		t.Fatalf("EncodeRequest() error = %v (%T), want *openaiapi.InvalidToolNameError", err, err)
	}

	// Chosen but NOT declared: refused a layer earlier, before this codec runs.
	_, err = openaiapi.EncodeRequest(inference.Request{
		Model:      model.Model{Name: "gpt-test"},
		Messages:   content.AgenticMessages{userMsg(textBlock("hi"))},
		Tools:      []inference.Tool{{Name: "ok", Schema: json.RawMessage(`{"type":"object"}`)}},
		ToolChoice: inference.ToolNamed("mcp.search"),
	}, false)
	var conflict *inference.StructuredOutputConflictError
	if !errors.As(err, &conflict) {
		t.Fatalf("EncodeRequest() error = %v (%T), want *inference.StructuredOutputConflictError; "+
			"an undeclared forced name is no longer refused upstream, so this codec must check it itself", err, err)
	}
}

// --- Sampling ranges -------------------------------------------------------
//
// Spec: components.schemas.ModelResponseProperties.temperature declares
// minimum 0 / maximum 2, and .top_p declares minimum 0 / maximum 1
// (CreateChatCompletionRequest reaches both through
// CreateModelResponseProperties). Unlike the tool-name rule these ARE carried
// by the derived request document, so the gate below independently rejects them
// — the encoder check exists so the failure names the field before the body is
// built rather than after it is sent.
//
// The bound differs per provider on purpose: Anthropic and Bedrock cap
// temperature at 1, OpenAI at 2. A session moved across providers carries the
// SOURCE's value into the DESTINATION's request, which is why the destination
// codec has to own the check.

func TestEncodeRequestRejectsOutOfRangeSampling(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		sampling  model.Sampling
		wantField string
		wantValue float64
	}{
		{name: "temperature above 2", sampling: model.Sampling{Temperature: float64Ptr(2.5)}, wantField: "temperature", wantValue: 2.5},
		{name: "temperature below 0", sampling: model.Sampling{Temperature: float64Ptr(-0.1)}, wantField: "temperature", wantValue: -0.1},
		{name: "top_p above 1", sampling: model.Sampling{TopP: float64Ptr(1.5)}, wantField: "top_p", wantValue: 1.5},
		{name: "top_p below 0", sampling: model.Sampling{TopP: float64Ptr(-1)}, wantField: "top_p", wantValue: -1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			body, err := openaiapi.EncodeRequest(inference.Request{
				Model:    samplingModel(tt.sampling),
				Messages: content.AgenticMessages{userMsg(textBlock("hi"))},
			}, false)
			if err == nil {
				t.Fatalf("EncodeRequest() accepted %s = %v, which the request schema forbids; body = %s",
					tt.wantField, tt.wantValue, body)
			}
			var rangeErr *openaiapi.SamplingRangeError
			if !errors.As(err, &rangeErr) {
				t.Fatalf("EncodeRequest() error = %v (%T), want *openaiapi.SamplingRangeError", err, err)
			}
			if rangeErr.Field != tt.wantField || rangeErr.Value != tt.wantValue {
				t.Errorf("SamplingRangeError = {%q, %v}, want {%q, %v}",
					rangeErr.Field, rangeErr.Value, tt.wantField, tt.wantValue)
			}
		})
	}
}

// TestEncodeRequestAcceptsInRangeSampling is the positive control: both
// endpoints of each declared interval, and the value that would be illegal for
// Anthropic but is legal here (temperature 2), must still encode.
func TestEncodeRequestAcceptsInRangeSampling(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		sampling model.Sampling
	}{
		{name: "unset", sampling: model.Sampling{}},
		{name: "temperature 0", sampling: model.Sampling{Temperature: float64Ptr(0)}},
		{name: "temperature 1", sampling: model.Sampling{Temperature: float64Ptr(1)}},
		{name: "temperature 2 (illegal for Anthropic, legal here)", sampling: model.Sampling{Temperature: float64Ptr(2)}},
		{name: "top_p 0", sampling: model.Sampling{TopP: float64Ptr(0)}},
		{name: "top_p 1", sampling: model.Sampling{TopP: float64Ptr(1)}},
		{name: "both at their maxima", sampling: model.Sampling{Temperature: float64Ptr(2), TopP: float64Ptr(1)}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			body, err := openaiapi.EncodeRequest(inference.Request{
				Model:    samplingModel(tt.sampling),
				Messages: content.AgenticMessages{userMsg(textBlock("hi"))},
			}, false)
			if err != nil {
				t.Fatalf("EncodeRequest() rejected in-range sampling: %v", err)
			}
			conformance.MustValidateRequest(t, "openai", kindChatCompletionRequest, body)
		})
	}
}

// TestEncodeRequestSamplingOverrideIsValidatedToo pins that the per-call
// Override, not just Model.Sampling, passes through the same check — the
// override is the field cross-provider switching actually populates.
func TestEncodeRequestSamplingOverrideIsValidatedToo(t *testing.T) {
	t.Parallel()

	override := model.Sampling{Temperature: float64Ptr(2.5)}
	_, err := openaiapi.EncodeRequest(inference.Request{
		Model:    samplingModel(model.Sampling{Temperature: float64Ptr(0.5)}),
		Messages: content.AgenticMessages{userMsg(textBlock("hi"))},
		Override: &override,
	}, false)
	var rangeErr *openaiapi.SamplingRangeError
	if !errors.As(err, &rangeErr) {
		t.Fatalf("EncodeRequest() error = %v (%T), want *openaiapi.SamplingRangeError", err, err)
	}
}

// --- Gate strength ---------------------------------------------------------

// TestTheChatRequestGateHoldsSamplingButNotToolNames measures what the
// conformance gate actually catches for this format rather than assuming it.
// Both halves matter: the sampling half proves the gate is asserting at all,
// and the tool-name half is the reason the encoder has to carry that rule
// itself.
func TestTheChatRequestGateHoldsSamplingButNotToolNames(t *testing.T) {
	t.Parallel()

	rejects := map[string]string{
		"temperature above the declared maximum": `{"model":"m","messages":[{"role":"user","content":"hi"}],"temperature":3}`,
		"top_p above the declared maximum":       `{"model":"m","messages":[{"role":"user","content":"hi"}],"top_p":2}`,
	}
	for name, body := range rejects {
		if err := conformance.Validate("openai", kindChatCompletionRequest, []byte(body)); err == nil {
			t.Errorf("gate accepted %s; the sampling bounds are supposed to be schema-backed", name)
		}
	}

	// Measured, not reasoned: FunctionObject.name carries its class and its
	// 64-character cap in prose only, so these bodies pass the gate. If a
	// future schema refresh starts declaring pattern/maxLength this test fails,
	// and the comment above it becomes wrong — which is the point.
	accepts := map[string]string{
		"a declared tool name with a space": `{"model":"m","messages":[{"role":"user","content":"hi"}],` +
			`"tools":[{"type":"function","function":{"name":"bad name!","parameters":{"type":"object"}}}]}`,
		"a declared tool name of 100 characters": `{"model":"m","messages":[{"role":"user","content":"hi"}],` +
			`"tools":[{"type":"function","function":{"name":"` + repeat(100) + `","parameters":{"type":"object"}}}]}`,
	}
	for name, body := range accepts {
		if err := conformance.Validate("openai", kindChatCompletionRequest, []byte(body)); err != nil {
			t.Errorf("gate rejected %s: %v\n"+
				"the schema has started enforcing tool names; update the encoder-only comments", name, err)
		}
	}
}
