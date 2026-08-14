package geminiapi_test

import (
	"errors"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/geminiapi"
	model "github.com/looprig/inference/model"
)

// --- TestEncodeRequest_ToolResultUnsupportedBlock ---

// A block in a tool result that this dialect cannot represent anywhere must fail
// secure with a *geminiapi.UnsupportedBlockError rather than being silently
// dropped — the model must never receive less than the caller sent. Mirrors the
// user/model-turn fail-secure behavior covered by
// TestEncodeRequest_UnsupportedBlock.
//
// Media blocks are no longer among them: an image, audio or document result now
// travels as inlineData parts of the same user turn as the functionResponse
// (toolResultContent, encode.go), covered by TestEncodeRequest_ToolResultMedia.
// The mime contract still fails closed there — see
// TestEncodeRequest_ToolResultUnsupportedMedia.
func TestEncodeRequest_ToolResultUnsupportedBlock(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name  string
		block content.Block
	}{
		{
			name:  "thinking block in a tool result is unsupported",
			block: &content.ThinkingBlock{Thinking: "not a tool's to send"},
		},
		{
			name:  "nested tool result block in a tool result is unsupported",
			block: &content.ToolResultBlock{ToolUseID: "call_1"},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			req := inference.Request{
				Model:    model.Model{Name: "m", Caps: model.Capabilities{AcceptsImages: true}},
				Messages: content.AgenticMessages{toolMsg("call_1", tc.block)},
			}
			_, err := geminiapi.EncodeRequest(req)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			var ube *geminiapi.UnsupportedBlockError
			if !errors.As(err, &ube) {
				t.Errorf("error = %v (%T), want *geminiapi.UnsupportedBlockError", err, err)
			}
		})
	}
}

// A tool result whose blocks are all text keeps encoding as a functionResponse
// — the fail-secure path must not disturb the supported shape.
func TestEncodeRequest_ToolResultTextStillEncodes(t *testing.T) {
	t.Parallel()

	req := inference.Request{
		Model:    model.Model{Name: "m"},
		Messages: content.AgenticMessages{toolMsg("call_1", &content.TextBlock{Text: "ok"})},
	}
	if _, err := geminiapi.EncodeRequest(req); err != nil {
		t.Fatalf("EncodeRequest error: %v", err)
	}
}
