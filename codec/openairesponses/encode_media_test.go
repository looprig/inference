package openairesponses_test

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/openairesponses"
	model "github.com/looprig/inference/model"
)

// Document and audio user-content coverage.
//
// The Responses input content union (InputContent) has exactly three members —
// input_text, input_image and input_file — so a DocumentBlock has a wire home
// and an AudioBlock does not. This file pins both halves of that.

// mediaInputParts encodes a single user message and returns its content parts.
func mediaInputParts(t *testing.T, blocks ...content.Block) []map[string]json.RawMessage {
	t.Helper()
	req := inference.Request{
		Model:    model.Model{Name: "gpt-test", Caps: model.Capabilities{AcceptsImages: true}},
		Messages: content.AgenticMessages{&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: blocks}}},
	}
	body, err := openairesponses.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	var decoded struct {
		Input []struct {
			Content []map[string]json.RawMessage `json:"content"`
		} `json:"input"`
	}
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("unmarshal encoded request: %v (body=%s)", err, body)
	}
	if len(decoded.Input) != 1 {
		t.Fatalf("input items = %d, want 1", len(decoded.Input))
	}
	return decoded.Input[0].Content
}

func inputPartField(t *testing.T, part map[string]json.RawMessage, key string) string {
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

// TestEncodeRequest_DocumentBlockBecomesInputFile pins the input_file part.
// InputFileContent.required is ["type"] alone; filename and file_data are the
// two members the inline form populates.
func TestEncodeRequest_DocumentBlockBecomesInputFile(t *testing.T) {
	t.Parallel()

	parts := mediaInputParts(t,
		&content.TextBlock{Text: "Summarize the attached report."},
		&content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Name: "report.pdf", Data: []byte("%PDF-1.4\n")},
	)
	if len(parts) != 2 {
		t.Fatalf("content parts = %d, want input_text then input_file", len(parts))
	}
	if got := inputPartField(t, parts[1], "type"); got != "input_file" {
		t.Fatalf("part 1 type = %q, want input_file", got)
	}
	if got := inputPartField(t, parts[1], "filename"); got != "report.pdf" {
		t.Errorf("filename = %q, want report.pdf", got)
	}
	want := "data:application/pdf;base64," + base64.StdEncoding.EncodeToString([]byte("%PDF-1.4\n"))
	if got := inputPartField(t, parts[1], "file_data"); got != want {
		t.Errorf("file_data = %q, want %q", got, want)
	}
	// `detail` belongs to input_image (where it is required) and to the
	// optional file-rendering control; the neutral vocabulary chooses neither.
	if _, ok := parts[1]["file_id"]; ok {
		t.Error("file_id emitted; the neutral vocabulary has no server-side file handle")
	}
}

// TestEncodeRequest_DocumentBlockTextBecomesFileData pins that a document
// carried as extracted text still travels as file data.
func TestEncodeRequest_DocumentBlockTextBecomesFileData(t *testing.T) {
	t.Parallel()

	parts := mediaInputParts(t, &content.DocumentBlock{
		MediaType: content.MediaTypeDocumentMarkdown, Name: "notes.md", Text: "# Title",
	})
	want := "data:text/markdown;base64," + base64.StdEncoding.EncodeToString([]byte("# Title"))
	if got := inputPartField(t, parts[0], "file_data"); got != want {
		t.Errorf("file_data = %q, want %q", got, want)
	}
}

// TestEncodeRequest_AudioBlockHasNoResponsesRepresentation pins the fail-closed
// half. The spec declares an InputAudio object, but nothing in the Responses
// request references it — InputContent is input_text|input_image|input_file,
// and InputAudio is reachable only from EvalItemContentItem, the Evals API. So
// audio must be refused with an error that names the limitation, never
// degraded into a text or file part.
func TestEncodeRequest_AudioBlockHasNoResponsesRepresentation(t *testing.T) {
	t.Parallel()

	req := inference.Request{
		Model: model.Model{Name: "gpt-test"},
		Messages: content.AgenticMessages{&content.UserMessage{Message: content.Message{
			Role:   content.RoleUser,
			Blocks: []content.Block{&content.AudioBlock{MediaType: content.MediaTypeAudioWAV, Data: []byte("RIFF")}},
		}}},
	}
	_, err := openairesponses.EncodeRequest(req, false)
	var ube *openairesponses.UnsupportedBlockError
	if !errors.As(err, &ube) {
		t.Fatalf("error = %v (%T), want *openairesponses.UnsupportedBlockError", err, err)
	}
	if ube.Reason == "" {
		t.Error("UnsupportedBlockError.Reason is empty; it must name the limitation")
	}
}

// TestEncodeRequest_DocumentFailsClosedOnMissingMembers pins the local
// validations, matching openaiapi's file part.
func TestEncodeRequest_DocumentFailsClosedOnMissingMembers(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name  string
		block content.Block
	}{
		{"document with no name", &content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Data: []byte("%PDF")}},
		{"document with no content", &content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Name: "report.pdf"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			req := inference.Request{
				Model: model.Model{Name: "gpt-test"},
				Messages: content.AgenticMessages{&content.UserMessage{Message: content.Message{
					Role: content.RoleUser, Blocks: []content.Block{tc.block},
				}}},
			}
			_, err := openairesponses.EncodeRequest(req, false)
			var ube *openairesponses.UnsupportedBlockError
			if !errors.As(err, &ube) {
				t.Fatalf("error = %v (%T), want *openairesponses.UnsupportedBlockError", err, err)
			}
			if ube.Reason == "" {
				t.Error("UnsupportedBlockError.Reason is empty; it must name the limitation")
			}
		})
	}
}
