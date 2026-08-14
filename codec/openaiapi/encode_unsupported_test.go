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

// A block this dialect cannot place in the position it occupies must fail
// secure with a *openaiapi.UnsupportedBlockError rather than being silently
// dropped — the model must never receive less than the caller sent. This
// mirrors the sibling anthropicapi and geminiapi codecs.
//
// The tool message is the position with no richer shape available:
// ChatCompletionRequestToolMessageContentPart is a one-member union over the
// text part ("For tool messages, only type `text` is supported"), so an image,
// audio or document block in a tool result is refused however well the user
// turn models it. A user turn itself now carries all four content-part
// members — see encode_media_test.go.
func TestEncodeRequest_UnsupportedBlock(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		msgs content.AgenticMessages
	}{
		{
			name: "image block in a tool result is unsupported",
			msgs: content.AgenticMessages{toolMsg("call_1", imageDataBlock(content.MediaTypeImagePNG, []byte{0x89}))},
		},
		{
			name: "audio block in a tool result is unsupported",
			msgs: content.AgenticMessages{toolMsg("call_1", &content.AudioBlock{MediaType: content.MediaTypeAudioMPEG, Data: []byte{1}})},
		},
		{
			name: "document block in a tool result is unsupported",
			msgs: content.AgenticMessages{toolMsg("call_1", &content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Name: "report.pdf", Data: []byte{1}})},
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
