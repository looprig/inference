package openairesponses_test

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/conformance"
	"github.com/looprig/inference/codec/openairesponses"
	model "github.com/looprig/inference/model"
)

// kindCreateResponseRequest is the conformance gate's entry point for a
// POST /v1/responses body.
const kindCreateResponseRequest = "create_response_request"

func samplingRequest(sampling model.Sampling) inference.Request {
	return inference.Request{
		Model: model.Model{Name: "gpt-test", Sampling: sampling},
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{
				Role:   content.RoleUser,
				Blocks: []content.Block{&content.TextBlock{Text: "hi"}},
			}},
		},
	}
}

func floatPtr(v float64) *float64 { return &v }

// --- Sampling ranges -------------------------------------------------------
//
// Spec: CreateResponse reaches CreateModelResponseProperties ->
// ModelResponseProperties, which declares temperature minimum 0 / maximum 2 and
// top_p minimum 0 / maximum 1 — the same intervals Chat Completions declares,
// because both dialects inherit the same schema. openaiapi already holds its
// encoder to them; this dialect forwarded both values unchecked.
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
		wantMin   float64
		wantMax   float64
	}{
		{name: "temperature above 2", sampling: model.Sampling{Temperature: floatPtr(2.5)}, wantField: "temperature", wantValue: 2.5, wantMin: 0, wantMax: 2},
		{name: "temperature below 0", sampling: model.Sampling{Temperature: floatPtr(-0.1)}, wantField: "temperature", wantValue: -0.1, wantMin: 0, wantMax: 2},
		{name: "top_p above 1", sampling: model.Sampling{TopP: floatPtr(1.5)}, wantField: "top_p", wantValue: 1.5, wantMin: 0, wantMax: 1},
		{name: "top_p below 0", sampling: model.Sampling{TopP: floatPtr(-1)}, wantField: "top_p", wantValue: -1, wantMin: 0, wantMax: 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			body, err := openairesponses.EncodeRequest(samplingRequest(tt.sampling), false)
			if err == nil {
				t.Fatalf("EncodeRequest() accepted %s = %v, which the request schema forbids; body = %s",
					tt.wantField, tt.wantValue, body)
			}
			var rangeErr *openairesponses.SamplingRangeError
			if !errors.As(err, &rangeErr) {
				t.Fatalf("EncodeRequest() error = %v (%T), want *openairesponses.SamplingRangeError", err, err)
			}
			if rangeErr.Field != tt.wantField || rangeErr.Value != tt.wantValue {
				t.Errorf("SamplingRangeError = {%q, %v}, want {%q, %v}",
					rangeErr.Field, rangeErr.Value, tt.wantField, tt.wantValue)
			}
			if rangeErr.Min != tt.wantMin || rangeErr.Max != tt.wantMax {
				t.Errorf("SamplingRangeError bounds = [%v, %v], want [%v, %v]",
					rangeErr.Min, rangeErr.Max, tt.wantMin, tt.wantMax)
			}
		})
	}
}

// TestEncodeRequestAcceptsInRangeSampling is the positive control: a rejection
// rule that also rejects valid input is worse than the bug it closes. Both
// endpoints of each declared interval must still encode, and the encoded body
// must still pass the request gate.
func TestEncodeRequestAcceptsInRangeSampling(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		sampling model.Sampling
	}{
		{name: "unset", sampling: model.Sampling{}},
		{name: "temperature 0", sampling: model.Sampling{Temperature: floatPtr(0)}},
		{name: "temperature 1", sampling: model.Sampling{Temperature: floatPtr(1)}},
		{name: "temperature 2 (illegal for Anthropic, legal here)", sampling: model.Sampling{Temperature: floatPtr(2)}},
		{name: "top_p 0", sampling: model.Sampling{TopP: floatPtr(0)}},
		{name: "top_p 1", sampling: model.Sampling{TopP: floatPtr(1)}},
		{name: "both at their maxima", sampling: model.Sampling{Temperature: floatPtr(2), TopP: floatPtr(1)}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			body, err := openairesponses.EncodeRequest(samplingRequest(tt.sampling), false)
			if err != nil {
				t.Fatalf("EncodeRequest() rejected in-range sampling: %v", err)
			}
			conformance.MustValidateRequest(t, "openai-responses", kindCreateResponseRequest, body)
		})
	}
}

// TestEncodeRequestSamplingOverrideIsValidatedToo pins that the per-call
// Override, not just Model.Sampling, passes through the same check — the
// override is the field cross-provider switching actually populates, and it
// REPLACES Model.Sampling wholesale in buildResponsesRequest, so a legal model
// default cannot rescue an illegal override.
func TestEncodeRequestSamplingOverrideIsValidatedToo(t *testing.T) {
	t.Parallel()

	override := model.Sampling{Temperature: floatPtr(2.5)}
	req := samplingRequest(model.Sampling{Temperature: floatPtr(0.5)})
	req.Override = &override

	_, err := openairesponses.EncodeRequest(req, false)
	var rangeErr *openairesponses.SamplingRangeError
	if !errors.As(err, &rangeErr) {
		t.Fatalf("EncodeRequest() error = %v (%T), want *openairesponses.SamplingRangeError", err, err)
	}

	// Positive control on the same path: an in-range override still encodes.
	legal := model.Sampling{Temperature: floatPtr(1.5), TopP: floatPtr(0.9)}
	req.Override = &legal
	body, err := openairesponses.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest() rejected an in-range override: %v", err)
	}
	conformance.MustValidateRequest(t, "openai-responses", kindCreateResponseRequest, body)
}

// --- Gate strength ---------------------------------------------------------

// TestTheResponsesRequestGateHoldsSamplingButNotToolNames measures what the
// conformance gate actually catches for this format rather than assuming it,
// and records why the sibling openaiapi tool-name rule is NOT transferred here.
//
// Chat Completions' FunctionObject.name carries a prose constraint — "Must be
// a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of
// 64" — which openaiapi's encoder enforces because the derived schema does not.
// The Responses dialect's FunctionTool (title "Function") declares `name` as a
// bare `{"type":"string"}` with NO pattern, NO maxLength, and NO prose
// constraint anywhere in its description. There is therefore nothing to
// transcribe: transplanting Chat's class would invent a rule this dialect never
// published, and would reject MCP-style dotted/slashed names that the Responses
// endpoint accepts. The accepts half below is the measurement, not the
// argument.
func TestTheResponsesRequestGateHoldsSamplingButNotToolNames(t *testing.T) {
	t.Parallel()

	rejects := map[string]string{
		"temperature above the declared maximum": `{"model":"m","input":[],"temperature":3}`,
		"top_p above the declared maximum":       `{"model":"m","input":[],"top_p":2}`,
	}
	for name, body := range rejects {
		if err := conformance.Validate("openai-responses", kindCreateResponseRequest, []byte(body)); err == nil {
			t.Errorf("gate accepted %s; the sampling bounds are supposed to be schema-backed", name)
		}
	}

	accepts := map[string]string{
		"a declared tool name with a dot": `{"model":"m","input":[],` +
			`"tools":[{"type":"function","name":"mcp.search","strict":false,"parameters":{"type":"object"}}]}`,
		"a declared tool name of 100 characters": `{"model":"m","input":[],` +
			`"tools":[{"type":"function","name":"` + strings.Repeat("a", 100) + `","strict":false,"parameters":{"type":"object"}}]}`,
	}
	for name, body := range accepts {
		if err := conformance.Validate("openai-responses", kindCreateResponseRequest, []byte(body)); err != nil {
			t.Errorf("gate rejected %s: %v\n"+
				"FunctionTool.name has started declaring a constraint; revisit the encoder-only comment above", name, err)
		}
	}
}

// TestEncodeRequestKeepsDottedToolNames is the behavioural half of the note
// above: this dialect must NOT acquire openaiapi's tool-name rule. An MCP-style
// name encodes, and the body passes the gate.
func TestEncodeRequestKeepsDottedToolNames(t *testing.T) {
	t.Parallel()

	req := samplingRequest(model.Sampling{})
	req.Tools = []inference.Tool{{Name: "mcp.search", Schema: json.RawMessage(`{"type":"object"}`)}}
	body, err := openairesponses.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest() rejected a tool name the Responses schema permits: %v", err)
	}
	if !strings.Contains(string(body), `"mcp.search"`) {
		t.Errorf("encoded body lost the tool name: %s", body)
	}
	conformance.MustValidateRequest(t, "openai-responses", kindCreateResponseRequest, body)
}
