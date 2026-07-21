package openaiapi_test

import (
	"errors"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/openaiapi"
	model "github.com/looprig/inference/model"
)

// --- TestEncodeRequest_UnsupportedBlock ---

// A block the OpenAI chat dialect does not model must fail secure with a
// *openaiapi.UnsupportedBlockError rather than being silently dropped — the
// model must never receive less than the caller sent. This mirrors the sibling
// anthropicapi and geminiapi codecs. The tool message is text-only on this
// wire, so a non-text block in a tool result is likewise refused, not dropped.
func TestEncodeRequest_UnsupportedBlock(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		msgs content.AgenticMessages
	}{
		{
			name: "audio block in a user turn is unsupported",
			msgs: content.AgenticMessages{userMsg(&content.AudioBlock{MediaType: content.MediaTypeAudioMPEG, Data: []byte{1}})},
		},
		{
			name: "document block in a user turn is unsupported",
			msgs: content.AgenticMessages{userMsg(&content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Data: []byte{1}})},
		},
		{
			name: "image block in a tool result is unsupported",
			msgs: content.AgenticMessages{toolMsg("call_1", imageDataBlock(content.MediaTypeImagePNG, []byte{0x89}))},
		},
		{
			name: "audio block in a tool result is unsupported",
			msgs: content.AgenticMessages{toolMsg("call_1", &content.AudioBlock{MediaType: content.MediaTypeAudioMPEG, Data: []byte{1}})},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			req := inference.Request{
				Model:    model.Model{Name: "m", Caps: model.Capabilities{AcceptsImages: true}},
				Messages: tc.msgs,
			}
			_, err := openaiapi.EncodeRequest(req, false)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			var ube *openaiapi.UnsupportedBlockError
			if !errors.As(err, &ube) {
				t.Errorf("error = %v (%T), want *openaiapi.UnsupportedBlockError", err, err)
			}
		})
	}
}

// A tool result whose blocks are all text keeps encoding as a plain-string
// tool message — the fail-secure path must not disturb the supported shape.
func TestEncodeRequest_ToolResultTextStillEncodes(t *testing.T) {
	t.Parallel()

	req := inference.Request{
		Model:    model.Model{Name: "m"},
		Messages: content.AgenticMessages{toolMsg("call_1", &content.TextBlock{Text: "ok"})},
	}
	if _, err := openaiapi.EncodeRequest(req, false); err != nil {
		t.Fatalf("EncodeRequest error: %v", err)
	}
}
