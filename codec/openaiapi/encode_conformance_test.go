package openaiapi_test

import (
	"encoding/json"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/conformance"
	"github.com/looprig/inference/codec/openaiapi"
	model "github.com/looprig/inference/model"
)

// kindChatCompletionRequest is the gate's key for the Chat Completions request
// body.
const kindChatCompletionRequest = "chat_completion_request"

func gatedModel() model.Model {
	return model.Model{
		Provider:  model.ProviderName("openai"),
		APIFormat: model.APIFormatOpenAI,
		BaseURL:   "https://api.openai.com/v1",
		Name:      "gpt-4o",
	}
}

// TestEncodeRequestHoldsAgainstTheOfficialRequestSchema is this codec's half of
// the module rule: "every encode path must hold its encoded body against the
// format's official request schema in tests".
//
// The rule was unsatisfiable here until the gate moved into this module. The
// schemas lived in llm, one tier up and behind an internal/, so the only tests
// that could reach them belonged to the provider clients that compose this
// codec — a tier too late to say which encoder produced a rejected body. These
// are the exact bytes this package puts on the wire.
//
// The openai request document is the strict half of the specification: it
// declares required properties and closes object shapes, so a variant emitted
// with the wrong member set is rejected here rather than by a live 400. Both a
// streaming and a non-streaming body are gated, because the two differ in the
// members they emit and only one of them would otherwise be covered.
func TestEncodeRequestHoldsAgainstTheOfficialRequestSchema(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		request inference.Request
		stream  bool
	}{
		{
			name: "system, user and assistant turns",
			request: inference.Request{
				Model:  gatedModel(),
				System: "be brief",
				Messages: content.AgenticMessages{
					userMsg(&content.TextBlock{Text: "hello"}),
					aiMsg(&content.TextBlock{Text: "hi"}),
				},
			},
		},
		{
			name:   "streaming body",
			stream: true,
			request: inference.Request{
				Model:    gatedModel(),
				Messages: content.AgenticMessages{userMsg(&content.TextBlock{Text: "hello"})},
			},
		},
		{
			name: "tool call and its result",
			request: inference.Request{
				Model: gatedModel(),
				Tools: []inference.Tool{{
					Name:        "lookup",
					Description: "Look up a value",
					Schema:      json.RawMessage(`{"type":"object","properties":{"key":{"type":"string"}},"required":["key"],"additionalProperties":false}`),
				}},
				ToolChoice: inference.ToolNamed("lookup"),
				Messages: content.AgenticMessages{
					userMsg(&content.TextBlock{Text: "look it up"}),
					aiMsg(content.NewToolUseBlock("call-1", "lookup", json.RawMessage(`{"key":"a"}`), nil, "")),
					toolMsg("call-1", &content.TextBlock{Text: "found"}),
				},
			},
		},
		{
			// A refusal is its own member of the assistant message, not text.
			// The request document types it, so an encoder that folded it into
			// content would be caught here.
			name: "assistant refusal",
			request: inference.Request{
				Model: gatedModel(),
				Messages: content.AgenticMessages{
					userMsg(&content.TextBlock{Text: "do the thing"}),
					aiMsg(&content.RefusalBlock{Text: "I will not."}),
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			body, err := openaiapi.EncodeRequest(tt.request, tt.stream)
			if err != nil {
				t.Fatalf("EncodeRequest() error = %v", err)
			}
			conformance.MustValidateRequest(t, "openai", kindChatCompletionRequest, body)
		})
	}
}

// TestTheRequestGateActuallyRejects is the control on the gate above. A gate
// that is never actually asserting proves nothing; the only way to know it is
// asserting is to hand it something the specification forbids and watch it
// refuse.
func TestTheRequestGateActuallyRejects(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		body string
	}{
		{
			name: "a role the enum does not declare",
			body: `{"model":"gpt-4o","messages":[{"role":"oracle","content":"hi"}]}`,
		},
		{
			name: "no model, which the request document requires",
			body: `{"messages":[{"role":"user","content":"hi"}]}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			if err := conformance.Validate("openai", kindChatCompletionRequest, []byte(tt.body)); err == nil {
				t.Fatalf("Validate(%s) = nil, want a violation; the gate is not asserting", tt.body)
			}
		})
	}
}
