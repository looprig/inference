package anthropicapi_test

import (
	"encoding/json"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/anthropicapi"
	"github.com/looprig/inference/codec/conformance"
	model "github.com/looprig/inference/model"
)

// kindCreateMessageRequest is the gate's key for the CreateMessageParams body.
const kindCreateMessageRequest = "create_message_request"

func conformanceModel() model.Model {
	return model.Model{
		Provider:  "anthropic",
		Name:      "claude-sonnet-5",
		APIFormat: model.APIFormatAnthropic,
		Caps: model.Capabilities{
			Tools: true, AcceptsImages: true,
			// claude-sonnet-5 is an adaptive-thinking model; the origin API
			// rejects the budget spelling on it by name.
			Thinking: true, ThinkingDialect: model.ThinkingDialectAdaptive,
		},
	}
}

func conformanceUser(blocks ...content.Block) *content.UserMessage {
	return &content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: blocks}}
}

func conformanceAssistant(blocks ...content.Block) *content.AIMessage {
	return &content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: blocks}}
}

func conformanceToolResult(id string, blocks ...content.Block) *content.ToolResultMessage {
	return &content.ToolResultMessage{
		Message:   content.Message{Role: content.RoleTool, Blocks: blocks},
		ToolUseID: id,
	}
}

// TestEncodeRequestHoldsAgainstTheOfficialRequestSchema is this codec's half of
// the module rule: "every encode path must hold its encoded body against the
// format's official request schema in tests". It was the last bundled codec
// without the gate wired in.
//
// Anthropic's derived request document is the STRONGEST of the five. Measured,
// not reasoned — every one of these was fed to the gate and rejected:
// a missing max_tokens, a missing messages array, an unknown top-level
// property, an unknown property inside a content block, a role outside the
// enum, a non-string signature, and a temperature outside [0, 1]. That is a
// real gate on nearly every mistake this encoder could make.
//
// The most load-bearing rejection for the change this file accompanies is that
// a `thinking` block WITHOUT a `signature` is refused by the schema itself.
// That is the specification agreeing with the live probe: an unsigned thinking
// block is not a legal request, so "drop the signature we cannot claim" was
// never an available degrade — the encoder has to fail instead.
func TestEncodeRequestHoldsAgainstTheOfficialRequestSchema(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		request inference.Request
	}{
		{
			name: "system and a plain exchange",
			request: inference.Request{
				Model:  conformanceModel(),
				System: "base instruction",
				Messages: content.AgenticMessages{
					conformanceUser(&content.TextBlock{Text: "hello"}),
					conformanceAssistant(&content.TextBlock{Text: "hi"}),
					conformanceUser(&content.TextBlock{Text: "again"}),
				},
			},
		},
		{
			name: "tool call and its result",
			request: inference.Request{
				Model: conformanceModel(),
				Tools: []inference.Tool{{
					Name:        "lookup",
					Description: "Look up a value",
					Schema:      json.RawMessage(`{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}`),
				}},
				Messages: content.AgenticMessages{
					conformanceUser(&content.TextBlock{Text: "look it up"}),
					conformanceAssistant(content.NewToolUseBlock(
						"toolu_1", "lookup", json.RawMessage(`{"key":"a"}`), nil, "")),
					conformanceToolResult("toolu_1", &content.TextBlock{Text: "found"}),
				},
			},
		},
		{
			name: "reasoning replayed with the signature this dialect minted",
			request: inference.Request{
				Model: conformanceModel(),
				Messages: content.AgenticMessages{
					conformanceUser(&content.TextBlock{Text: "think"}),
					conformanceAssistant(content.NewSignedThinkingBlock(
						"because", "EosnCkYIBxgCKkBd", signatureFormatAnthropic, nil, "")),
					conformanceUser(&content.TextBlock{Text: "go on"}),
				},
			},
		},
		{
			name: "redacted reasoning replayed from its opaque provider state",
			request: inference.Request{
				Model: conformanceModel(),
				Messages: content.AgenticMessages{
					conformanceUser(&content.TextBlock{Text: "think"}),
					conformanceAssistant(content.NewThinkingBlock(
						"", "", json.RawMessage(`"`+redactedOpaque+`"`), providerStateFormatRedacted)),
					conformanceUser(&content.TextBlock{Text: "go on"}),
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			body, err := anthropicapi.EncodeRequest(tt.request, false)
			if err != nil {
				t.Fatalf("EncodeRequest() error = %v", err)
			}
			conformance.MustValidateRequest(t, "anthropic", kindCreateMessageRequest, body)
		})
	}
}

// TestTheRequestGateActuallyRejects is the control on the gate above. A gate
// that is never actually asserting proves nothing, and the only way to know it
// is asserting is to hand it something the specification forbids and watch it
// refuse.
//
// The chosen violation is the one this codec's signature handling turns on: a
// `thinking` block with no `signature`. Anthropic's CreateMessageParams marks
// signature required on RequestThinkingBlock, so the shape the encoder would
// have produced had it "degraded" by stripping a foreign signature is provably
// illegal before any request is sent.
func TestTheRequestGateActuallyRejects(t *testing.T) {
	t.Parallel()

	illegal := []byte(`{"model":"claude-sonnet-5","max_tokens":8,"messages":[` +
		`{"role":"assistant","content":[{"type":"thinking","thinking":"because"}]}]}`)
	if err := conformance.Validate("anthropic", kindCreateMessageRequest, illegal); err == nil {
		t.Fatal("Validate() accepted a thinking block with no signature, which CreateMessageParams requires; the gate is not asserting")
	}
}
