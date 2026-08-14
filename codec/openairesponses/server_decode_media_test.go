package openairesponses_test

import (
	"errors"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference/codec/openairesponses"
)

// Server-decode coverage for the input_file content part, so the neutral ->
// wire mapping added in encode.go closes as a round trip.

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

func TestServerDecode_InputFileBecomesDocumentBlock(t *testing.T) {
	t.Parallel()

	blocks := decodeUserBlocks(t, `{
		"model": "gpt-test",
		"input": [{"type":"message","role":"user","content":[
			{"type":"input_text","text":"look"},
			{"type":"input_file","filename":"report.pdf","file_data":"data:application/pdf;base64,JVBERi0="},
			{"type":"input_file","filename":"bare.pdf","file_data":"JVBERi0="}
		]}]
	}`)
	if len(blocks) != 3 {
		t.Fatalf("Blocks len = %d, want 3", len(blocks))
	}

	typed, ok := blocks[1].(*content.DocumentBlock)
	if !ok {
		t.Fatalf("block 1 = %T, want *content.DocumentBlock", blocks[1])
	}
	if typed.MediaType != content.MediaTypeDocumentPDF || typed.Name != "report.pdf" || string(typed.Data) != "%PDF-" {
		t.Errorf("document = %#v", typed)
	}

	bare, ok := blocks[2].(*content.DocumentBlock)
	if !ok {
		t.Fatalf("block 2 = %T, want *content.DocumentBlock", blocks[2])
	}
	if bare.MediaType != "" || string(bare.Data) != "%PDF-" {
		t.Errorf("bare document = %#v", bare)
	}
}

// TestServerDecode_InputFileFailsClosed pins the untrusted-input rejections:
// file_id and file_url both name resources outside the neutral transcript, and
// an undecodable payload cannot become block bytes.
func TestServerDecode_InputFileFailsClosed(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		part string
	}{
		{"file_id reference", `{"type":"input_file","file_id":"file-abc123"}`},
		{"no file data", `{"type":"input_file","filename":"x.pdf"}`},
		{"invalid base64", `{"type":"input_file","filename":"x.pdf","file_data":"!!!"}`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			c, req := decodeReq(t, `{"model":"gpt-test","input":[{"type":"message","role":"user","content":[`+tc.part+`]}]}`)
			_, err := c.DecodeRequest(req)
			var sde *openairesponses.ServerDecodeError
			if !errors.As(err, &sde) {
				t.Fatalf("error = %v (%T), want *openairesponses.ServerDecodeError", err, err)
			}
		})
	}
}
