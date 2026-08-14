package bedrockconverse_test

import (
	"encoding/json"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/bedrockconverse"
	"github.com/looprig/inference/codec/conformance"
)

// kindConverseRequest is the gate's key for the ConverseRequest body shape.
const kindConverseRequest = "converse_request"

// signatureFormatBedrockConverse mirrors the package-private label this dialect
// stamps on a reasoning signature. A fixture that replays a signature has to
// carry it, because an unlabelled signature is refused by the encoder — the
// same refusal that stops an Anthropic-minted signature from reaching Bedrock.
const signatureFormatBedrockConverse = "bedrock-converse"

// TestEncodeRequestHoldsAgainstTheOfficialRequestSchema is this codec's half of
// the module rule: "every encode path must hold its encoded body against the
// format's official request schema in tests".
//
// The rule was unsatisfiable here until the gate moved into this module. The
// schemas lived in llm, one tier up and behind an internal/, so the only tests
// that could reach them were the provider clients' — a tier too late to say
// which encoder produced a rejected body. What is validated below is the exact
// bytes this package puts on the wire.
//
// Know what this gate can and cannot do for bedrock-converse, and do not reason
// about it: AWS marks only modelId @required on ConverseRequest and modelId
// travels in the URI path, so the derived request document requires NOTHING at
// the top level. Its strength here is types, enums, nesting, and the toolUseId
// pattern and length caps — which are real, and which are what
// llm/providers/bedrock's negative fixtures already exercise. A missing
// messages array would pass; that constraint is carried by the encoder, not by
// this gate.
func TestEncodeRequestHoldsAgainstTheOfficialRequestSchema(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		request inference.Request
	}{
		{
			name: "system and a plain exchange",
			request: inference.Request{
				Model:  baseModel(),
				System: "base instruction",
				Messages: content.AgenticMessages{
					userMessage(&content.TextBlock{Text: "hello"}),
					assistantMessage(&content.TextBlock{Text: "hi"}),
				},
			},
		},
		{
			name: "tool call and its result",
			request: inference.Request{
				Model: baseModel(),
				Tools: []inference.Tool{{
					Name:        "lookup",
					Description: "Look up a value",
					Schema:      json.RawMessage(`{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}`),
				}},
				Messages: content.AgenticMessages{
					userMessage(&content.TextBlock{Text: "look it up"}),
					assistantMessage(content.NewToolUseBlock(
						"call-1", "lookup", json.RawMessage(`{"key":"a"}`), nil, "")),
					toolResultMessage("call-1", false, &content.TextBlock{Text: "found"}),
				},
			},
		},
		{
			name: "reasoning replayed with its provider state",
			request: inference.Request{
				Model: baseModel(),
				Messages: content.AgenticMessages{
					userMessage(&content.TextBlock{Text: "think"}),
					assistantMessage(content.NewSignedThinkingBlock("because", "sig", signatureFormatBedrockConverse, nil, "")),
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			body, err := bedrockconverse.EncodeRequest(tt.request)
			if err != nil {
				t.Fatalf("EncodeRequest() error = %v", err)
			}
			conformance.MustValidateRequest(t, "bedrock-converse", kindConverseRequest, body)
		})
	}
}

// TestTheRequestGateActuallyRejects is the control on the gate above. A gate
// that is never actually asserting proves nothing, and the only way to know it
// is asserting is to hand it something the specification forbids and watch it
// refuse. The toolUseId pattern is chosen because it is one of the few
// constraints AWS's model really does declare on this body.
func TestTheRequestGateActuallyRejects(t *testing.T) {
	t.Parallel()

	illegal := []byte(`{"messages":[{"role":"user","content":[` +
		`{"toolResult":{"toolUseId":"has spaces","content":[{"text":"x"}]}}]}]}`)
	if err := conformance.Validate("bedrock-converse", kindConverseRequest, illegal); err == nil {
		t.Fatal("Validate() accepted a toolUseId the ConverseRequest pattern forbids; the gate is not asserting")
	}
}
