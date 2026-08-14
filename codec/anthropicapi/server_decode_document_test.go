package anthropicapi_test

import (
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference/codec/anthropicapi"
)

// The server decoder is the only direction in which a document block can be
// READ: documents are request-only in the Anthropic document, so a response
// never carries one. Closing the round trip therefore means accepting the two
// source members the encoder emits, and refusing by name the two it cannot
// represent — a client that sends one gets a decode error naming the member
// rather than a silently emptied document.

func TestDecodeRequest_Documents(t *testing.T) {
	t.Parallel()

	t.Run("base64 pdf", func(t *testing.T) {
		t.Parallel()
		body := `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":[
			{"type":"document","title":"filing","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0="}}
		]}]}`
		req := mustDecode(t, body)
		um := req.Messages[0].(*content.UserMessage)
		doc, ok := um.Blocks[0].(*content.DocumentBlock)
		if !ok {
			t.Fatalf("Blocks[0] type = %T, want *content.DocumentBlock", um.Blocks[0])
		}
		if doc.MediaType != content.MediaTypeDocumentPDF {
			t.Errorf("MediaType = %q, want %q", doc.MediaType, content.MediaTypeDocumentPDF)
		}
		if doc.Name != "filing" {
			t.Errorf("Name = %q, want %q", doc.Name, "filing")
		}
		if string(doc.Data) != "%PDF-" {
			t.Errorf("Data = %q, want the decoded payload", doc.Data)
		}
	})

	t.Run("plain text", func(t *testing.T) {
		t.Parallel()
		body := `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":[
			{"type":"document","source":{"type":"text","media_type":"text/plain","data":"line one"}}
		]}]}`
		req := mustDecode(t, body)
		doc := req.Messages[0].(*content.UserMessage).Blocks[0].(*content.DocumentBlock)
		if doc.MediaType != content.MediaTypeDocumentText {
			t.Errorf("MediaType = %q, want %q", doc.MediaType, content.MediaTypeDocumentText)
		}
		if doc.Text != "line one" {
			t.Errorf("Text = %q, want %q", doc.Text, "line one")
		}
		if len(doc.Data) != 0 {
			t.Errorf("Data = %q, want empty on a text source", doc.Data)
		}
	})

	t.Run("inside a tool result", func(t *testing.T) {
		t.Parallel()
		body := `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":[
			{"type":"tool_result","tool_use_id":"toolu_1","content":[
				{"type":"document","source":{"type":"text","media_type":"text/plain","data":"body"}}
			]}
		]}]}`
		req := mustDecode(t, body)
		result := req.Messages[0].(*content.ToolResultMessage)
		if _, ok := result.Blocks[0].(*content.DocumentBlock); !ok {
			t.Fatalf("tool result Blocks[0] type = %T, want *content.DocumentBlock", result.Blocks[0])
		}
	})

	// URLPDFSource and ContentBlockSource are legal Anthropic sources with no
	// neutral counterpart: content.DocumentBlock has neither a URL field nor a
	// nested block list. Both are refused rather than decoded into an empty
	// document, which would report success while losing the entire payload.
	for _, tc := range []struct {
		name   string
		source string
		want   string
	}{
		{
			name:   "url source",
			source: `{"type":"url","url":"https://example.test/report.pdf"}`,
			want:   "url",
		},
		{
			name:   "content block source",
			source: `{"type":"content","content":[{"type":"text","text":"chunk"}]}`,
			want:   "content",
		},
		{
			name:   "unknown source type",
			source: `{"type":"file_id"}`,
			want:   "file_id",
		},
	} {
		tc := tc
		t.Run(tc.name+" is refused", func(t *testing.T) {
			t.Parallel()
			body := `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":[
				{"type":"document","source":` + tc.source + `}
			]}]}`
			_, err := anthropicapi.Codec{}.DecodeRequest(newDecodeRequest(body))
			if err == nil {
				t.Fatalf("DecodeRequest() error = nil, want an error naming %q", tc.want)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error = %q, want it to name %q", err, tc.want)
			}
		})
	}

	t.Run("not allowed in assistant message", func(t *testing.T) {
		t.Parallel()
		body := `{"model":"m","max_tokens":8,"messages":[{"role":"assistant","content":[
			{"type":"document","source":{"type":"text","media_type":"text/plain","data":"x"}}
		]}]}`
		_, err := anthropicapi.Codec{}.DecodeRequest(newDecodeRequest(body))
		if err == nil {
			t.Error("DecodeRequest() error = nil, want error (document in assistant message)")
		}
	})
}
