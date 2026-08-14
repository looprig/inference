package anthropicapi_test

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/anthropicapi"
)

// This file covers the two content blocks the shared vocabulary declares and
// this dialect had never mapped: content.DocumentBlock, which Anthropic models
// as RequestDocumentBlock, and content.AudioBlock, which Anthropic's Messages
// API does not model at all.
//
// RequestDocumentBlock declares required = [source, type] with
// additionalProperties = false, and its `source` is a four-member tagged union
// discriminated by `type`:
//
//	base64  -> Base64PDFSource   required [data, media_type, type], media_type const "application/pdf"
//	text    -> PlainTextSource   required [data, media_type, type], media_type const "text/plain"
//	url     -> URLPDFSource      required [type, url]
//	content -> ContentBlockSource required [content, type]
//
// The neutral content.DocumentBlock carries MediaType/Name/Data/Text, so it can
// express the first two members and neither of the last two. Those are refused
// with a typed error rather than approximated, because a document silently
// re-sourced is a document the caller never asked for.

// pdfBytes is a minimal PDF magic-number payload; the codec never parses it.
var pdfBytes = []byte{0x25, 0x50, 0x44, 0x46, 0x2d}

// TestEncodeRequest_DocumentBlock proves each representable source member
// reaches the wire in its own required-complete shape.
func TestEncodeRequest_DocumentBlock(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name   string
		block  content.Block
		assert func(t *testing.T, block map[string]json.RawMessage)
	}{
		{
			name:  "pdf bytes become a base64 source",
			block: &content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Name: "quarterly report", Data: pdfBytes},
			assert: func(t *testing.T, block map[string]json.RawMessage) {
				t.Helper()
				if got := asString(t, block["type"]); got != "document" {
					t.Errorf("type = %q, want %q", got, "document")
				}
				if got := asString(t, block["title"]); got != "quarterly report" {
					t.Errorf("title = %q, want %q", got, "quarterly report")
				}
				source := decodeObj(t, block["source"])
				if got := asString(t, source["type"]); got != "base64" {
					t.Errorf("source.type = %q, want %q", got, "base64")
				}
				if got := asString(t, source["media_type"]); got != "application/pdf" {
					t.Errorf("source.media_type = %q, want %q", got, "application/pdf")
				}
				if got := asString(t, source["data"]); got != base64.StdEncoding.EncodeToString(pdfBytes) {
					t.Errorf("source.data = %q, want the base64 payload", got)
				}
				if _, present := source["url"]; present {
					t.Errorf("base64 source carries url, which Base64PDFSource does not declare: %s", block["source"])
				}
				if _, present := source["content"]; present {
					t.Errorf("base64 source carries content, which Base64PDFSource does not declare: %s", block["source"])
				}
			},
		},
		{
			name:  "extracted text becomes a plain-text source",
			block: &content.DocumentBlock{MediaType: content.MediaTypeDocumentText, Name: "notes", Text: "line one"},
			assert: func(t *testing.T, block map[string]json.RawMessage) {
				t.Helper()
				source := decodeObj(t, block["source"])
				if got := asString(t, source["type"]); got != "text" {
					t.Errorf("source.type = %q, want %q", got, "text")
				}
				if got := asString(t, source["media_type"]); got != "text/plain" {
					t.Errorf("source.media_type = %q, want %q", got, "text/plain")
				}
				if got := asString(t, source["data"]); got != "line one" {
					t.Errorf("source.data = %q, want the extracted text", got)
				}
			},
		},
		{
			// title is optional (minLength 1), so a nameless document must not
			// travel with an empty one.
			name:  "a nameless document omits title rather than sending an empty one",
			block: &content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Data: pdfBytes},
			assert: func(t *testing.T, block map[string]json.RawMessage) {
				t.Helper()
				if _, present := block["title"]; present {
					t.Errorf("title is present on a nameless document: %v", block)
				}
			},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			data, err := anthropicapi.EncodeRequest(inference.Request{
				Model:    baseModel(),
				Messages: content.AgenticMessages{userMsg(tc.block, &content.TextBlock{Text: "Summarize it."})},
			}, false)
			if err != nil {
				t.Fatalf("EncodeRequest: %v", err)
			}
			blocks := blocksOf(t, messagesOf(t, decodeObj(t, data))[0])
			tc.assert(t, blocks[0])
		})
	}
}

// TestEncodeRequest_DocumentBlockInToolResult proves a document survives inside
// a tool_result, whose content union lists RequestDocumentBlock alongside text
// and image.
func TestEncodeRequest_DocumentBlockInToolResult(t *testing.T) {
	t.Parallel()

	data, err := anthropicapi.EncodeRequest(inference.Request{
		Model: baseModel(),
		Messages: content.AgenticMessages{
			userMsg(&content.TextBlock{Text: "Fetch the filing."}),
			aiMsg(content.NewToolUseBlock("toolu_fetch", "fetch", json.RawMessage(`{}`), nil, "")),
			toolResultMsg("toolu_fetch", false, &content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Name: "filing", Data: pdfBytes}),
		},
	}, false)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}
	messages := messagesOf(t, decodeObj(t, data))
	inner := asObjs(t, blocksOf(t, messages[2])[0]["content"])
	if got := asString(t, inner[0]["type"]); got != "document" {
		t.Fatalf("tool_result content[0].type = %q, want %q", got, "document")
	}
	if got := asString(t, decodeObj(t, inner[0]["source"])["media_type"]); got != "application/pdf" {
		t.Errorf("tool_result document media_type = %q, want %q", got, "application/pdf")
	}
}

// TestEncodeRequest_DocumentBlockRejections covers every neutral document shape
// with no legal RequestDocumentBlock form. Each one would otherwise travel as a
// body Anthropic answers with an HTTP 400 that names no field.
func TestEncodeRequest_DocumentBlockRejections(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name  string
		block content.Block
		want  string
	}{
		{
			// Base64PDFSource.media_type is const "application/pdf": no other
			// binary document type has a wire form at all.
			name:  "non-pdf binary document",
			block: &content.DocumentBlock{MediaType: content.MediaTypeDocumentDOCX, Name: "spec", Data: []byte{0x50, 0x4b}},
			want:  "application/pdf",
		},
		{
			// PlainTextSource.media_type is const "text/plain", so markdown
			// cannot be declared as itself; relabelling it would be a silent
			// rewrite of the caller's media type.
			name:  "markdown text document",
			block: &content.DocumentBlock{MediaType: content.MediaTypeDocumentMarkdown, Name: "readme", Text: "# Title"},
			want:  "text/plain",
		},
		{
			name:  "document with neither data nor text",
			block: &content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Name: "empty"},
			want:  "source",
		},
		{
			// title is maxLength 500.
			name:  "title longer than the declared maximum",
			block: &content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Name: strings.Repeat("n", 501), Data: pdfBytes},
			want:  "500",
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := anthropicapi.EncodeRequest(inference.Request{
				Model:    baseModel(),
				Messages: content.AgenticMessages{userMsg(tc.block)},
			}, false)
			if err == nil {
				t.Fatalf("expected an error, got nil")
			}
			var unsupported *anthropicapi.UnsupportedDocumentError
			if !errors.As(err, &unsupported) {
				t.Fatalf("error = %v (%T), want *UnsupportedDocumentError", err, err)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error = %q, want it to name %q", err, tc.want)
			}
		})
	}
}

// TestEncodeRequest_AudioBlockFailsClosed pins the audio limitation to a typed
// error of its own. Anthropic's document contains no audio content block in any
// form — the string "audio" does not appear in it — so an AudioBlock has no
// wire shape here, not merely an unmapped one. mcp/pkg/harness constructs one
// from an MCP audio tool result, and the harness persists it, so this block
// really does reach the encoder; naming the limitation is the difference
// between a diagnosable session and an opaque provider rejection.
func TestEncodeRequest_AudioBlockFailsClosed(t *testing.T) {
	t.Parallel()

	for _, placement := range []struct {
		name     string
		messages content.AgenticMessages
	}{
		{
			name:     "user message",
			messages: content.AgenticMessages{userMsg(&content.AudioBlock{MediaType: content.MediaTypeAudioMPEG, Data: []byte{0x49, 0x44, 0x33}})},
		},
		{
			name: "tool result",
			messages: content.AgenticMessages{
				userMsg(&content.TextBlock{Text: "listen"}),
				aiMsg(content.NewToolUseBlock("toolu_listen", "listen", json.RawMessage(`{}`), nil, "")),
				toolResultMsg("toolu_listen", false, &content.AudioBlock{MediaType: content.MediaTypeAudioWAV, Data: []byte{0x52, 0x49, 0x46, 0x46}}),
			},
		},
	} {
		placement := placement
		t.Run(placement.name, func(t *testing.T) {
			t.Parallel()
			_, err := anthropicapi.EncodeRequest(inference.Request{Model: baseModel(), Messages: placement.messages}, false)
			if err == nil {
				t.Fatalf("expected an error, got nil")
			}
			var audio *anthropicapi.UnsupportedAudioError
			if !errors.As(err, &audio) {
				t.Fatalf("error = %v (%T), want *UnsupportedAudioError", err, err)
			}
			if !strings.Contains(err.Error(), "audio") {
				t.Errorf("error = %q, want it to name the audio limitation", err)
			}
		})
	}
}
