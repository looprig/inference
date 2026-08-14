package openaiapi_test

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/openaiapi"
	model "github.com/looprig/inference/model"
)

// Document and audio user-content coverage.
//
// ChatCompletionRequestUserMessageContentPart is a four-member union: text,
// image_url, input_audio and file. This file pins the two members the codec
// gained for content.DocumentBlock and content.AudioBlock, including the
// spec's `input_audio.format` enum transcribed as a fail-closed allowlist.

// mediaParts encodes a single user message and returns its content parts.
func mediaParts(t *testing.T, blocks ...content.Block) []map[string]json.RawMessage {
	t.Helper()
	req := inference.Request{
		Model:    model.Model{Name: "m", Caps: model.Capabilities{AcceptsImages: true}},
		Messages: content.AgenticMessages{userMsg(blocks...)},
	}
	body, err := openaiapi.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	msgs := messagesFromRaw(t, mustDecode(t, body))
	if len(msgs) != 1 {
		t.Fatalf("messages = %d, want 1", len(msgs))
	}
	var parts []map[string]json.RawMessage
	if err := json.Unmarshal(msgs[0]["content"], &parts); err != nil {
		t.Fatalf("unmarshal content parts: %v\nbody: %s", err, body)
	}
	return parts
}

// partField unmarshals one member of a content part as a string.
func partField(t *testing.T, part map[string]json.RawMessage, key string) string {
	t.Helper()
	raw, ok := part[key]
	if !ok {
		t.Fatalf("content part %v has no %q", part, key)
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatalf("unmarshal %q: %v", key, err)
	}
	return s
}

// partObject unmarshals one member of a content part as an object.
func partObject(t *testing.T, part map[string]json.RawMessage, key string) map[string]string {
	t.Helper()
	raw, ok := part[key]
	if !ok {
		t.Fatalf("content part %v has no %q", part, key)
	}
	var obj map[string]string
	if err := json.Unmarshal(raw, &obj); err != nil {
		t.Fatalf("unmarshal %q: %v", key, err)
	}
	return obj
}

// TestEncodeRequest_DocumentBlockBecomesFilePart pins the file content part:
// ChatCompletionRequestMessageContentPartFile.required is ["type","file"], and
// the inline form carries filename plus file_data.
func TestEncodeRequest_DocumentBlockBecomesFilePart(t *testing.T) {
	t.Parallel()

	parts := mediaParts(t,
		textBlock("Summarize the attached report."),
		&content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Name: "report.pdf", Data: []byte("%PDF-1.4\n")},
	)
	if len(parts) != 2 {
		t.Fatalf("content parts = %d, want text then file", len(parts))
	}
	if got := partField(t, parts[1], "type"); got != "file" {
		t.Fatalf("part 1 type = %q, want file", got)
	}
	file := partObject(t, parts[1], "file")
	if file["filename"] != "report.pdf" {
		t.Errorf("file.filename = %q, want report.pdf", file["filename"])
	}
	want := "data:application/pdf;base64," + base64.StdEncoding.EncodeToString([]byte("%PDF-1.4\n"))
	if file["file_data"] != want {
		t.Errorf("file.file_data = %q, want %q", file["file_data"], want)
	}
	if _, ok := file["file_id"]; ok {
		t.Error("file.file_id emitted; the neutral vocabulary has no server-side file handle")
	}
}

// TestEncodeRequest_DocumentBlockTextBecomesFileData pins that a document
// provided as extracted text (DocumentBlock.Text, the shape an MCP embedded
// text resource takes) still travels as file data rather than being dropped or
// silently flattened into prose.
func TestEncodeRequest_DocumentBlockTextBecomesFileData(t *testing.T) {
	t.Parallel()

	parts := mediaParts(t, &content.DocumentBlock{
		MediaType: content.MediaTypeDocumentMarkdown, Name: "notes.md", Text: "# Title",
	})
	file := partObject(t, parts[0], "file")
	want := "data:text/markdown;base64," + base64.StdEncoding.EncodeToString([]byte("# Title"))
	if file["file_data"] != want {
		t.Errorf("file.file_data = %q, want %q", file["file_data"], want)
	}
}

// TestEncodeRequest_AudioBlockBecomesInputAudioPart pins the audio content
// part: ChatCompletionRequestMessageContentPartAudio.required is
// ["type","input_audio"], and input_audio.required is ["data","format"].
func TestEncodeRequest_AudioBlockBecomesInputAudioPart(t *testing.T) {
	t.Parallel()

	cases := []struct {
		mediaType  content.MediaType
		wantFormat string
	}{
		{content.MediaTypeAudioWAV, "wav"},
		{content.MediaTypeAudioMPEG, "mp3"},
	}
	for _, tc := range cases {
		t.Run(string(tc.mediaType), func(t *testing.T) {
			t.Parallel()

			parts := mediaParts(t, &content.AudioBlock{MediaType: tc.mediaType, Data: []byte("RIFF")})
			if got := partField(t, parts[0], "type"); got != "input_audio" {
				t.Fatalf("part type = %q, want input_audio", got)
			}
			audio := partObject(t, parts[0], "input_audio")
			if audio["format"] != tc.wantFormat {
				t.Errorf("input_audio.format = %q, want %q", audio["format"], tc.wantFormat)
			}
			if want := base64.StdEncoding.EncodeToString([]byte("RIFF")); audio["data"] != want {
				t.Errorf("input_audio.data = %q, want %q", audio["data"], want)
			}
		})
	}
}

// TestEncodeRequest_AudioFormatIsAnAllowlist pins the fail-closed direction of
// the enum. `input_audio.format` is exactly ["wav","mp3"], so every other audio
// media type — including ones content declares constants for — must be refused
// rather than sent as a value the provider will reject.
func TestEncodeRequest_AudioFormatIsAnAllowlist(t *testing.T) {
	t.Parallel()

	for _, mediaType := range []content.MediaType{
		content.MediaTypeAudioOGG,
		content.MediaTypeAudioFLAC,
		content.MediaTypeAudioAAC,
		content.MediaTypeAudioMP4,
		content.MediaTypeAudioWebM,
		content.MediaType("audio/future"),
		content.MediaType(""),
	} {
		t.Run(string(mediaType), func(t *testing.T) {
			t.Parallel()

			req := inference.Request{
				Model:    model.Model{Name: "m"},
				Messages: content.AgenticMessages{userMsg(&content.AudioBlock{MediaType: mediaType, Data: []byte("x")})},
			}
			_, err := openaiapi.EncodeRequest(req, false)
			var ube *openaiapi.UnsupportedBlockError
			if !errors.As(err, &ube) {
				t.Fatalf("error = %v (%T), want *openaiapi.UnsupportedBlockError", err, err)
			}
			if ube.Reason == "" {
				t.Error("UnsupportedBlockError.Reason is empty; it must name the limitation")
			}
		})
	}
}

// TestEncodeRequest_MediaBlocksFailClosedOnMissingMembers pins the two local
// validations: a file part with inline data needs a filename (OpenAI's file
// input guide requires it alongside file_data), and neither part can carry an
// empty payload.
func TestEncodeRequest_MediaBlocksFailClosedOnMissingMembers(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name  string
		block content.Block
	}{
		{"document with no name", &content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Data: []byte("%PDF")}},
		{"document with no content", &content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Name: "report.pdf"}},
		{"audio with no data", &content.AudioBlock{MediaType: content.MediaTypeAudioWAV}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			req := inference.Request{
				Model:    model.Model{Name: "m"},
				Messages: content.AgenticMessages{userMsg(tc.block)},
			}
			_, err := openaiapi.EncodeRequest(req, false)
			var ube *openaiapi.UnsupportedBlockError
			if !errors.As(err, &ube) {
				t.Fatalf("error = %v (%T), want *openaiapi.UnsupportedBlockError", err, err)
			}
			if ube.Reason == "" {
				t.Error("UnsupportedBlockError.Reason is empty; it must name the limitation")
			}
		})
	}
}
