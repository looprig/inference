package openaiapi_test

import (
	"errors"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference/codec/openaiapi"
)

// Server-decode coverage for the file and input_audio user-content parts, so
// the neutral -> wire mapping added in encode.go closes as a round trip rather
// than being a one-way street.

func decodeUserBlocks(t *testing.T, body string) []content.Block {
	t.Helper()
	c, req := decodeReq(t, body)
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	um, ok := decoded.Request.Messages[0].(*content.UserMessage)
	if !ok {
		t.Fatalf("message 0 = %T, want *content.UserMessage", decoded.Request.Messages[0])
	}
	return um.Blocks
}

func TestServerDecode_FilePartBecomesDocumentBlock(t *testing.T) {
	t.Parallel()

	blocks := decodeUserBlocks(t, `{
		"model": "gpt-test",
		"messages": [{"role":"user","content":[
			{"type":"text","text":"look"},
			{"type":"file","file":{"filename":"report.pdf","file_data":"data:application/pdf;base64,JVBERi0="}},
			{"type":"file","file":{"filename":"bare.pdf","file_data":"JVBERi0="}}
		]}]
	}`)
	if len(blocks) != 3 {
		t.Fatalf("Blocks len = %d, want 3", len(blocks))
	}

	typed, ok := blocks[1].(*content.DocumentBlock)
	if !ok {
		t.Fatalf("block 1 = %T, want *content.DocumentBlock", blocks[1])
	}
	if typed.MediaType != content.MediaTypeDocumentPDF {
		t.Errorf("MediaType = %q, want application/pdf", typed.MediaType)
	}
	if typed.Name != "report.pdf" || string(typed.Data) != "%PDF-" {
		t.Errorf("document = %#v", typed)
	}

	// A bare base64 file_data is equally legal (the schema types it as a plain
	// string); it simply carries no media type, and re-encoding it produces the
	// same bare form.
	bare, ok := blocks[2].(*content.DocumentBlock)
	if !ok {
		t.Fatalf("block 2 = %T, want *content.DocumentBlock", blocks[2])
	}
	if bare.MediaType != "" || string(bare.Data) != "%PDF-" {
		t.Errorf("bare document = %#v", bare)
	}
}

func TestServerDecode_InputAudioPartBecomesAudioBlock(t *testing.T) {
	t.Parallel()

	blocks := decodeUserBlocks(t, `{
		"model": "gpt-test",
		"messages": [{"role":"user","content":[
			{"type":"input_audio","input_audio":{"data":"UklGRg==","format":"wav"}},
			{"type":"input_audio","input_audio":{"data":"SUQz","format":"mp3"}}
		]}]
	}`)
	if len(blocks) != 2 {
		t.Fatalf("Blocks len = %d, want 2", len(blocks))
	}
	wav, ok := blocks[0].(*content.AudioBlock)
	if !ok {
		t.Fatalf("block 0 = %T, want *content.AudioBlock", blocks[0])
	}
	if wav.MediaType != content.MediaTypeAudioWAV || string(wav.Data) != "RIFF" {
		t.Errorf("wav audio = %#v", wav)
	}
	mp3, ok := blocks[1].(*content.AudioBlock)
	if !ok {
		t.Fatalf("block 1 = %T, want *content.AudioBlock", blocks[1])
	}
	if mp3.MediaType != content.MediaTypeAudioMPEG || string(mp3.Data) != "ID3" {
		t.Errorf("mp3 audio = %#v", mp3)
	}
}

// TestServerDecode_MediaPartsFailClosed pins the untrusted-input rejections.
// A file_id references a file in OpenAI's Files API that no neutral block can
// hold, an unknown audio format is outside the spec's two-member enum, and a
// missing or undecodable payload cannot become block bytes.
func TestServerDecode_MediaPartsFailClosed(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		part string
	}{
		{"file_id reference", `{"type":"file","file":{"file_id":"file-abc123"}}`},
		{"file with no object", `{"type":"file"}`},
		{"file with no data", `{"type":"file","file":{"filename":"x.pdf"}}`},
		{"file with invalid base64", `{"type":"file","file":{"filename":"x.pdf","file_data":"!!!"}}`},
		{"audio with no object", `{"type":"input_audio"}`},
		{"audio in an unlisted format", `{"type":"input_audio","input_audio":{"data":"AA==","format":"flac"}}`},
		{"audio with invalid base64", `{"type":"input_audio","input_audio":{"data":"!!!","format":"wav"}}`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			c, req := decodeReq(t, `{"model":"gpt-test","messages":[{"role":"user","content":[`+tc.part+`]}]}`)
			_, err := c.DecodeRequest(req)
			var sde *openaiapi.ServerDecodeError
			if !errors.As(err, &sde) {
				t.Fatalf("error = %v (%T), want *openaiapi.ServerDecodeError", err, err)
			}
		})
	}
}
