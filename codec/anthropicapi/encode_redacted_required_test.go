package anthropicapi

import (
	"encoding/json"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/model"
)

// Anthropic declares RequestRedactedThinkingBlock with
// required = [data, type] and additionalProperties = false, so `data` must be
// on the wire even when the opaque payload decoded to an empty string. The
// shared DTO's omitempty drops it and the replayed block is rejected. This is
// the redacted sibling of the thinking-block required-field guarantee.
func TestEncodeRequest_RedactedThinkingAlwaysEmitsRequiredFields(t *testing.T) {
	for _, tc := range []struct {
		name  string
		state string
	}{
		{name: "populated data", state: "AQID/w=="},
		{name: "empty data", state: ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			state, err := json.Marshal(tc.state)
			if err != nil {
				t.Fatalf("marshal provider state: %v", err)
			}
			req := inference.Request{
				Model: model.Model{Name: "claude-sonnet-5", APIFormat: model.APIFormatAnthropic},
				Messages: content.AgenticMessages{&content.AIMessage{Message: content.Message{
					Role: content.RoleAssistant,
					Blocks: []content.Block{
						content.NewThinkingBlock("", "", state, providerStateFormatAnthropicRedacted),
					},
				}}},
			}

			body, err := EncodeRequest(req, false)
			if err != nil {
				t.Fatalf("EncodeRequest() error = %v", err)
			}

			var decoded struct {
				Messages []struct {
					Content []map[string]json.RawMessage `json:"content"`
				} `json:"messages"`
			}
			if err := json.Unmarshal(body, &decoded); err != nil {
				t.Fatalf("unmarshal body: %v", err)
			}
			if len(decoded.Messages) != 1 || len(decoded.Messages[0].Content) != 1 {
				t.Fatalf("body = %s, want one message with one block", body)
			}
			block := decoded.Messages[0].Content[0]
			if _, ok := block["data"]; !ok {
				t.Errorf("redacted_thinking block missing required key %q: %v", "data", block)
			}
			for key := range block {
				if key != "type" && key != "data" {
					t.Errorf("redacted_thinking block carries %q, outside additionalProperties=false schema", key)
				}
			}
		})
	}
}
