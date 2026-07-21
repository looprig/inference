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

// The classic Gemini functionResponse carries text only, so a non-text block in
// a tool result must fail secure with a *geminiapi.UnsupportedBlockError rather
// than being silently dropped — the model must never receive less than the
// caller sent. Mirrors the user/model-turn fail-secure behavior covered by
// TestEncodeRequest_UnsupportedBlock.
func TestEncodeRequest_ToolResultUnsupportedBlock(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name  string
		block content.Block
	}{
		{
			name:  "image block in a tool result is unsupported",
			block: &content.ImageBlock{MediaType: content.MediaTypeImagePNG, Source: content.ImageSource{Data: []byte{0x89}}},
		},
		{
			name:  "audio block in a tool result is unsupported",
			block: &content.AudioBlock{MediaType: content.MediaTypeAudioMPEG, Data: []byte{1}},
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
