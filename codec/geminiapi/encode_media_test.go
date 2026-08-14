package geminiapi_test

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/geminiapi"
	model "github.com/looprig/inference/model"
)

// partDataMembers is the `Part.data` union: the members the discovery document
// describes as "A `Part` can only contain one of the accepted types in
// `Part.data`". thought/thoughtSignature/partMetadata are deliberately absent —
// they are Part attributes that legally accompany a data member, not members of
// the union themselves.
var partDataMembers = []string{
	"text", "inlineData", "fileData", "functionCall", "functionResponse",
	"executableCode", "codeExecutionResult", "toolCall", "toolResponse",
}

// assertSingleDataMember pins the union arity the Gemini request schema cannot:
// the discovery document expresses Part as an ordinary object with every member
// optional, so a two-member part validates cleanly against the conformance gate
// (see TestGeminiPartWithTwoContentMembers in llm/providers/gemini). The
// encoder's own tests are therefore the only place this invariant is enforced.
func assertSingleDataMember(t *testing.T, part map[string]json.RawMessage) {
	t.Helper()
	var set []string
	for _, member := range partDataMembers {
		if _, ok := part[member]; ok {
			set = append(set, member)
		}
	}
	if len(set) != 1 {
		t.Errorf("part carries %v, want exactly one Part.data member (part = %v)", set, part)
	}
}

// assertInlineData checks a part is the inlineData (Blob) form carrying mime and
// the base64 of want.
func assertInlineData(t *testing.T, part map[string]json.RawMessage, mime string, want []byte) {
	t.Helper()
	raw, ok := part["inlineData"]
	if !ok {
		t.Fatalf("part = %v, want an inlineData part", part)
	}
	var blob struct {
		MimeType string `json:"mimeType"`
		Data     string `json:"data"`
	}
	if err := json.Unmarshal(raw, &blob); err != nil {
		t.Fatalf("unmarshal inlineData: %v", err)
	}
	if blob.MimeType != mime {
		t.Errorf("inlineData.mimeType = %q, want %q", blob.MimeType, mime)
	}
	if got, want := blob.Data, base64.StdEncoding.EncodeToString(want); got != want {
		t.Errorf("inlineData.data = %q, want %q", got, want)
	}
}

// --- TestEncodeRequest_DocumentAndAudioParts ---

// A DocumentBlock and an AudioBlock both travel in Gemini's inlineData (Blob)
// member: the Blob's documented mime list covers application/pdf and audio/*,
// so the two block types differ only in which mime they carry. A document that
// arrived as extracted text instead of bytes takes Part.text, which the Blob
// documentation explicitly directs text at ("Text should not be sent as raw
// bytes, use the 'text' field").
func TestEncodeRequest_DocumentAndAudioParts(t *testing.T) {
	t.Parallel()

	pdf := []byte("%PDF-1.7\n")
	wav := []byte("RIFF\x00\x00\x00\x00WAVE")

	req := inference.Request{
		Model: model.Model{Name: "m"},
		Messages: content.AgenticMessages{userMsg(
			textBlock("summarize these"),
			&content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Name: "report.pdf", Data: pdf},
			&content.AudioBlock{MediaType: content.MediaTypeAudioWAV, Data: wav},
			&content.DocumentBlock{MediaType: content.MediaTypeDocumentMarkdown, Name: "notes.md", Text: "# notes"},
		)},
	}

	got, err := geminiapi.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}

	parts := partsOf(t, contentsFromRaw(t, mustDecode(t, got))[0])
	if len(parts) != 4 {
		t.Fatalf("parts = %d, want 4 (block order preserved, nothing dropped)", len(parts))
	}
	for _, part := range parts {
		assertSingleDataMember(t, part)
	}
	if text := strField(t, parts[0], "text"); text != "summarize these" {
		t.Errorf("parts[0].text = %q, want the leading text block", text)
	}
	assertInlineData(t, parts[1], "application/pdf", pdf)
	assertInlineData(t, parts[2], "audio/wav", wav)
	if text := strField(t, parts[3], "text"); text != "# notes" {
		t.Errorf("parts[3].text = %q, want the document's extracted text", text)
	}
}

// --- TestEncodeRequest_DocumentWithBytesAndText ---

// A document carrying both bytes and extracted text yields both parts, in that
// order: dropping either half would be a silent loss of what the caller sent,
// and neither half can share a Part with the other.
func TestEncodeRequest_DocumentWithBytesAndText(t *testing.T) {
	t.Parallel()

	pdf := []byte("%PDF-1.7\n")
	req := inference.Request{
		Model: model.Model{Name: "m"},
		Messages: content.AgenticMessages{userMsg(&content.DocumentBlock{
			MediaType: content.MediaTypeDocumentPDF,
			Data:      pdf,
			Text:      "extracted",
		})},
	}

	got, err := geminiapi.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	parts := partsOf(t, contentsFromRaw(t, mustDecode(t, got))[0])
	if len(parts) != 2 {
		t.Fatalf("parts = %d, want 2 (bytes and text)", len(parts))
	}
	assertInlineData(t, parts[0], "application/pdf", pdf)
	if text := strField(t, parts[1], "text"); text != "extracted" {
		t.Errorf("parts[1].text = %q, want extracted", text)
	}
}

// --- TestEncodeRequest_MediaMIMEIsHeldToTheBlobContract ---

// Blob's mime list is part of the contract, so a media type Gemini does not
// accept fails closed here with a typed error naming the block, rather than
// being sent as a request we can prove will be refused. The .docx wire type is
// absent from the Blob documentation's Applications list; an AudioBlock that is
// not audio/* is equally refused, since the block type is not a licence to send
// an arbitrary mime.
func TestEncodeRequest_MediaMIMEIsHeldToTheBlobContract(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name  string
		block content.Block
	}{
		{
			name:  "docx bytes are not an accepted Blob mime type",
			block: &content.DocumentBlock{MediaType: content.MediaTypeDocumentDOCX, Data: []byte{0x50, 0x4b}},
		},
		{
			name:  "an audio block carrying a non-audio mime type",
			block: &content.AudioBlock{MediaType: content.MediaTypeImagePNG, Data: []byte{0x89}},
		},
		{
			name:  "a document with neither bytes nor text",
			block: &content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Name: "empty.pdf"},
		},
		{
			name:  "an audio block with no bytes",
			block: &content.AudioBlock{MediaType: content.MediaTypeAudioWAV},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			_, err := geminiapi.EncodeRequest(inference.Request{
				Model:    model.Model{Name: "m"},
				Messages: content.AgenticMessages{userMsg(tc.block)},
			})
			var unsupported *geminiapi.UnsupportedBlockError
			if !errors.As(err, &unsupported) {
				t.Fatalf("error = %v (%T), want *geminiapi.UnsupportedBlockError", err, err)
			}
			if unsupported.Reason == "" {
				t.Error("UnsupportedBlockError.Reason is empty, want a diagnosis naming the defect")
			}
		})
	}
}

// --- TestEncodeRequest_ToolResultMedia ---

// An MCP tool that returns audio or an embedded resource produces an
// AudioBlock/DocumentBlock inside a TOOL RESULT (mcp/pkg/harness/tools.go), and
// the harness persists it, so refusing it at encode time fails every later turn
// of that session, not just the one that fetched it. Gemini's classic
// functionResponse carries a JSON object and no media, so the bytes travel as
// inlineData parts of the same user turn, immediately after the functionResponse
// they belong to — the arrangement Google's own multimodal function-calling
// guidance used before FunctionResponse.parts existed. A document's extracted
// text is tool OUTPUT, so it folds into the functionResponse result rather than
// becoming a separate user-visible part.
func TestEncodeRequest_ToolResultMedia(t *testing.T) {
	t.Parallel()

	mp3 := []byte("ID3\x04\x00")
	req := inference.Request{
		Model: model.Model{Name: "m"},
		Messages: content.AgenticMessages{
			userMsg(textBlock("read it aloud")),
			aiMsg(content.NewToolUseBlock("call_1", "speak", json.RawMessage(`{}`), nil, "")),
			toolMsg("call_1",
				textBlock("spoken:"),
				&content.AudioBlock{MediaType: content.MediaTypeAudioMPEG, Data: mp3},
				&content.DocumentBlock{MediaType: content.MediaTypeDocumentText, Name: "transcript.txt", Text: " the transcript"},
			),
		},
	}

	got, err := geminiapi.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}

	parts := partsOf(t, contentsFromRaw(t, mustDecode(t, got))[2])
	if len(parts) != 2 {
		t.Fatalf("parts = %d, want 2 (the functionResponse and the audio blob)", len(parts))
	}
	for _, part := range parts {
		assertSingleDataMember(t, part)
	}

	var response struct {
		Name     string `json:"name"`
		Response struct {
			Result string `json:"result"`
		} `json:"response"`
	}
	raw, ok := parts[0]["functionResponse"]
	if !ok {
		t.Fatalf("parts[0] = %v, want the functionResponse", parts[0])
	}
	if err := json.Unmarshal(raw, &response); err != nil {
		t.Fatalf("unmarshal functionResponse: %v", err)
	}
	if response.Name != "speak" {
		t.Errorf("functionResponse.name = %q, want speak", response.Name)
	}
	if want := "spoken: the transcript"; response.Response.Result != want {
		t.Errorf("functionResponse.response.result = %q, want %q", response.Response.Result, want)
	}
	assertInlineData(t, parts[1], "audio/mpeg", mp3)
}

// --- TestEncodeRequest_ToolResultUnsupportedMedia ---

// The mime contract holds inside a tool result exactly as it does on a user
// turn: media the Blob cannot carry still fails closed rather than reaching the
// provider.
func TestEncodeRequest_ToolResultUnsupportedMedia(t *testing.T) {
	t.Parallel()

	req := inference.Request{
		Model: model.Model{Name: "m"},
		Messages: content.AgenticMessages{toolMsg("call_1",
			&content.DocumentBlock{MediaType: content.MediaTypeDocumentXLSX, Data: []byte{0x50, 0x4b}},
		)},
	}
	_, err := geminiapi.EncodeRequest(req)
	var unsupported *geminiapi.UnsupportedBlockError
	if !errors.As(err, &unsupported) {
		t.Fatalf("error = %v (%T), want *geminiapi.UnsupportedBlockError", err, err)
	}
}

// --- TestEncodeRequest_DocumentTextSurvivesAnUnsupportedMIME ---

// A document whose bytes Gemini could not accept still travels when it carries
// extracted text: text has no mime constraint on the wire, so the caller's
// content reaches the model instead of the whole turn failing.
func TestEncodeRequest_DocumentTextSurvivesAnUnsupportedMIME(t *testing.T) {
	t.Parallel()

	req := inference.Request{
		Model: model.Model{Name: "m"},
		Messages: content.AgenticMessages{userMsg(&content.DocumentBlock{
			MediaType: content.MediaTypeDocumentDOCX,
			Name:      "memo.docx",
			Text:      "the memo body",
		})},
	}
	got, err := geminiapi.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	parts := partsOf(t, contentsFromRaw(t, mustDecode(t, got))[0])
	if len(parts) != 1 {
		t.Fatalf("parts = %d, want 1", len(parts))
	}
	if text := strField(t, parts[0], "text"); text != "the memo body" {
		t.Errorf("parts[0].text = %q, want the memo body", text)
	}
}
