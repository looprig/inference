package geminiapi_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strconv"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/geminiapi"
	model "github.com/looprig/inference/model"
)

// newDecodeRequest builds an httptest request for the non-streaming
// generateContent route carrying body.
func newDecodeRequest(model, body string) *http.Request {
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/"+model+":generateContent", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")
	return req
}

func decodeReq(t *testing.T, model, body string) (*geminiapi.Codec, *http.Request) {
	t.Helper()
	c := &geminiapi.Codec{}
	return c, newDecodeRequest(model, body)
}

func TestServerDecode_MatchesBothRoutes(t *testing.T) {
	t.Parallel()
	c := geminiapi.Codec{}

	nonStreaming := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-test:generateContent", nil)
	if !c.MatchRequest(nonStreaming) {
		t.Error("MatchRequest(:generateContent) = false, want true")
	}

	streaming := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-test:streamGenerateContent?alt=sse", nil)
	if !c.MatchRequest(streaming) {
		t.Error("MatchRequest(:streamGenerateContent?alt=sse) = false, want true")
	}

	if c.MatchRequest(httptest.NewRequest(http.MethodGet, "/v1beta/models/gemini-test:generateContent", nil)) {
		t.Error("MatchRequest(GET) = true, want false")
	}
	if c.MatchRequest(httptest.NewRequest(http.MethodPost, "/v1beta/models", nil)) {
		t.Error("MatchRequest(no model/suffix) = true, want false")
	}
	if c.MatchRequest(httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)) {
		t.Error("MatchRequest(/v1/chat/completions) = true, want false")
	}
}

func TestServerDecode_ExtractsModelFromPath(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, "gemini-2.5-pro", `{"contents":[]}`)
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if decoded.RequestedModel != "gemini-2.5-pro" {
		t.Errorf("RequestedModel = %q, want gemini-2.5-pro", decoded.RequestedModel)
	}
	if decoded.Streaming {
		t.Error("Streaming = true, want false for :generateContent")
	}
}

func TestServerDecode_StreamingRouteSetsStreamingTrue(t *testing.T) {
	t.Parallel()
	c := geminiapi.Codec{}
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-test:streamGenerateContent?alt=sse", bytes.NewReader([]byte(`{"contents":[]}`)))
	req.Header.Set("Content-Type", "application/json")
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if decoded.RequestedModel != "gemini-test" {
		t.Errorf("RequestedModel = %q, want gemini-test", decoded.RequestedModel)
	}
	if !decoded.Streaming {
		t.Error("Streaming = false, want true for :streamGenerateContent")
	}
}

func TestServerDecode_StreamingRouteWithoutAltSSEStillMatches(t *testing.T) {
	t.Parallel()
	// alt=sse is a Google convention, not this codec's routing signal (the
	// path suffix alone is unambiguous) — see the package doc for the
	// rationale. A request missing it must still be served correctly.
	c := geminiapi.Codec{}
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-test:streamGenerateContent", bytes.NewReader([]byte(`{"contents":[]}`)))
	req.Header.Set("Content-Type", "application/json")
	if !c.MatchRequest(req) {
		t.Fatal("MatchRequest() = false, want true (alt=sse must not be required)")
	}
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if !decoded.Streaming {
		t.Error("Streaming = false, want true")
	}
}

func TestServerDecode_SystemInstruction(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, "m", `{
		"systemInstruction": {"parts": [{"text": "be terse"}]},
		"contents": [{"role":"user","parts":[{"text":"hi"}]}]
	}`)
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if decoded.Request.System != "be terse" {
		t.Errorf("System = %q", decoded.Request.System)
	}
	if len(decoded.Request.Messages) != 1 {
		t.Fatalf("Messages len = %d", len(decoded.Request.Messages))
	}
	um, ok := decoded.Request.Messages[0].(*content.UserMessage)
	if !ok || um.Blocks[0].(*content.TextBlock).Text != "hi" {
		t.Errorf("Messages[0] = %#v", decoded.Request.Messages[0])
	}
}

func TestServerDecode_ModelRoleBecomesAIMessage(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, "m", `{
		"contents": [
			{"role":"user","parts":[{"text":"go"}]},
			{"role":"model","parts":[{"text":"planning","thought":true},{"text":"answer"},{"functionCall":{"id":"call_1","name":"f","args":{"x":1}}}]}
		]
	}`)
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if len(decoded.Request.Messages) != 2 {
		t.Fatalf("Messages len = %d, want 2", len(decoded.Request.Messages))
	}
	ai, ok := decoded.Request.Messages[1].(*content.AIMessage)
	if !ok {
		t.Fatalf("Messages[1] type = %T", decoded.Request.Messages[1])
	}
	if len(ai.Blocks) != 3 {
		t.Fatalf("Blocks len = %d, want 3", len(ai.Blocks))
	}
	tb, ok := ai.Blocks[0].(*content.ThinkingBlock)
	if !ok || tb.Thinking != "planning" {
		t.Errorf("Blocks[0] = %#v", ai.Blocks[0])
	}
	txt, ok := ai.Blocks[1].(*content.TextBlock)
	if !ok || txt.Text != "answer" {
		t.Errorf("Blocks[1] = %#v", ai.Blocks[1])
	}
	tu, ok := ai.Blocks[2].(*content.ToolUseBlock)
	if !ok || tu.ID != "call_1" || tu.Name != "f" {
		t.Errorf("Blocks[2] = %#v", ai.Blocks[2])
	}
}

func TestServerDecode_ThoughtSignaturePreserved(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, "m", `{
		"contents": [{"role":"model","parts":[{"text":"planning","thought":true,"thoughtSignature":"opaque-sig"}]}]
	}`)
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	ai := decoded.Request.Messages[0].(*content.AIMessage)
	tb := ai.Blocks[0].(*content.ThinkingBlock)
	var sig string
	if err := json.Unmarshal(tb.ProviderState, &sig); err != nil {
		t.Fatalf("ProviderState not a JSON string: %v", err)
	}
	if sig != "opaque-sig" {
		t.Errorf("sig = %q, want opaque-sig", sig)
	}
}

func TestServerDecode_FunctionResponseBecomesToolResultMessage(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, "m", `{
		"contents": [
			{"role":"user","parts":[{"functionResponse":{"id":"call_1","name":"get_weather","response":{"result":"sunny"}}}]}
		]
	}`)
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	tr, ok := decoded.Request.Messages[0].(*content.ToolResultMessage)
	if !ok {
		t.Fatalf("Messages[0] type = %T", decoded.Request.Messages[0])
	}
	if tr.ToolUseID != "call_1" {
		t.Errorf("ToolUseID = %q", tr.ToolUseID)
	}
	if tr.Blocks[0].(*content.TextBlock).Text != "sunny" {
		t.Errorf("text = %q", tr.Blocks[0].(*content.TextBlock).Text)
	}
}

func TestServerDecode_FunctionResponseWithoutIDStillDecodes(t *testing.T) {
	t.Parallel()
	// Gemini matches functionResponse to functionCall by NAME, not id — id
	// must not be required.
	c, req := decodeReq(t, "m", `{
		"contents": [{"role":"user","parts":[{"functionResponse":{"name":"get_weather","response":{"result":"sunny"}}}]}]
	}`)
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v, want no id required", err)
	}
	tr := decoded.Request.Messages[0].(*content.ToolResultMessage)
	if tr.ToolUseID != "" {
		t.Errorf("ToolUseID = %q, want empty", tr.ToolUseID)
	}
}

func TestServerDecode_InlineImage(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, "m", `{
		"contents": [{"role":"user","parts":[
			{"text":"look"},
			{"inlineData":{"mimeType":"image/png","data":"aGVsbG8="}}
		]}]
	}`)
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	um := decoded.Request.Messages[0].(*content.UserMessage)
	if len(um.Blocks) != 2 {
		t.Fatalf("Blocks len = %d, want 2", len(um.Blocks))
	}
	img := um.Blocks[1].(*content.ImageBlock)
	if string(img.MediaType) != "image/png" || string(img.Source.Data) != "hello" {
		t.Errorf("img = %#v", img)
	}
}

func TestServerDecode_FileDataImage(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, "m", `{
		"contents": [{"role":"user","parts":[{"fileData":{"mimeType":"image/png","fileUri":"https://harness-supplied/whatever"}}]}]
	}`)
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	um := decoded.Request.Messages[0].(*content.UserMessage)
	img := um.Blocks[0].(*content.ImageBlock)
	if img.Source.URL != "https://harness-supplied/whatever" {
		t.Errorf("URL = %q", img.Source.URL)
	}
}

// An inlineData part is not necessarily an image: the same Blob carries audio
// and documents, distinguished only by its mimeType. Decoding all three as
// ImageBlocks (as this decoder once did) is a silent corruption of the block
// type, so the mime chooses the block — which is also what closes the round trip
// with the encoder, whose document and audio parts are the very same Blob.
func TestServerDecode_InlineDataMIMEChoosesTheBlockType(t *testing.T) {
	t.Parallel()

	c, req := decodeReq(t, "m", `{
		"contents": [{"role":"user","parts":[
			{"inlineData":{"mimeType":"audio/wav","data":"aGVsbG8="}},
			{"inlineData":{"mimeType":"application/pdf","data":"aGVsbG8="}},
			{"inlineData":{"mimeType":"text/markdown","data":"aGVsbG8="}},
			{"inlineData":{"mimeType":"image/png","data":"aGVsbG8="}}
		]}]
	}`)
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	blocks := decoded.Request.Messages[0].(*content.UserMessage).Blocks
	if len(blocks) != 4 {
		t.Fatalf("Blocks len = %d, want 4", len(blocks))
	}

	audio, ok := blocks[0].(*content.AudioBlock)
	if !ok {
		t.Fatalf("Blocks[0] = %T, want *content.AudioBlock", blocks[0])
	}
	if audio.MediaType != content.MediaTypeAudioWAV || string(audio.Data) != "hello" {
		t.Errorf("audio = %#v", audio)
	}
	for i, wantMediaType := range map[int]content.MediaType{1: content.MediaTypeDocumentPDF, 2: content.MediaTypeDocumentMarkdown} {
		doc, ok := blocks[i].(*content.DocumentBlock)
		if !ok {
			t.Fatalf("Blocks[%d] = %T, want *content.DocumentBlock", i, blocks[i])
		}
		if doc.MediaType != wantMediaType || string(doc.Data) != "hello" {
			t.Errorf("document = %#v, want media type %q", doc, wantMediaType)
		}
	}
	if _, ok := blocks[3].(*content.ImageBlock); !ok {
		t.Fatalf("Blocks[3] = %T, want *content.ImageBlock", blocks[3])
	}
}

// A fileData part is a URI, and only ImageBlock has a URL source in the neutral
// vocabulary. Audio and document blocks carry bytes and nothing else, so a
// fileUri with one of their mime types cannot be decoded without inventing a
// source — it fails closed instead of arriving as an ImageBlock whose media type
// says audio.
func TestServerDecode_FileDataRejectsMIMEItCannotRepresent(t *testing.T) {
	t.Parallel()

	c, req := decodeReq(t, "m", `{
		"contents": [{"role":"user","parts":[{"fileData":{"mimeType":"audio/mpeg","fileUri":"https://generativelanguage.googleapis.com/v1beta/files/abc"}}]}]
	}`)
	_, err := c.DecodeRequest(req)
	var decodeErr *geminiapi.ServerDecodeError
	if !errors.As(err, &decodeErr) {
		t.Fatalf("DecodeRequest() error = %v (%T), want *geminiapi.ServerDecodeError", err, err)
	}
}

// TestServerCodec_MediaRoundTrip closes the loop the two directions form: a
// request encoded from document and audio blocks decodes back into the same
// block types, media types and bytes. The one deliberate asymmetry is a
// document that arrived as extracted text, which travels as Part.text (per
// Blob's own instruction not to send text as raw bytes) and therefore returns
// as a TextBlock — the content survives, the document framing does not.
func TestServerCodec_MediaRoundTrip(t *testing.T) {
	t.Parallel()

	pdf := []byte("%PDF-1.7\n")
	wav := []byte("RIFF\x00\x00\x00\x00WAVE")
	original := inference.Request{
		Model: model.Model{Name: "gemini-test"},
		Messages: content.AgenticMessages{userMsg(
			&content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Data: pdf},
			&content.AudioBlock{MediaType: content.MediaTypeAudioWAV, Data: wav},
			&content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Text: "extracted"},
		)},
	}

	body, err := geminiapi.EncodeRequest(original)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	c, req := decodeReq(t, "gemini-test", string(body))
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v (body=%s)", err, body)
	}

	blocks := decoded.Request.Messages[0].(*content.UserMessage).Blocks
	if len(blocks) != 3 {
		t.Fatalf("Blocks len = %d, want 3", len(blocks))
	}
	doc, ok := blocks[0].(*content.DocumentBlock)
	if !ok {
		t.Fatalf("Blocks[0] = %T, want *content.DocumentBlock", blocks[0])
	}
	if doc.MediaType != content.MediaTypeDocumentPDF || !bytes.Equal(doc.Data, pdf) {
		t.Errorf("document = %#v, want the pdf bytes back verbatim", doc)
	}
	audio, ok := blocks[1].(*content.AudioBlock)
	if !ok {
		t.Fatalf("Blocks[1] = %T, want *content.AudioBlock", blocks[1])
	}
	if audio.MediaType != content.MediaTypeAudioWAV || !bytes.Equal(audio.Data, wav) {
		t.Errorf("audio = %#v, want the wav bytes back verbatim", audio)
	}
	text, ok := blocks[2].(*content.TextBlock)
	if !ok {
		t.Fatalf("Blocks[2] = %T, want *content.TextBlock", blocks[2])
	}
	if text.Text != "extracted" {
		t.Errorf("text = %q, want extracted", text.Text)
	}
}

func TestServerDecode_ToolsAndRequiredToolChoice(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, "m", `{
		"contents": [],
		"tools": [{"functionDeclarations":[{"name":"get_weather","description":"gets weather","parameters":{"type":"object"}}]}],
		"toolConfig": {"functionCallingConfig":{"mode":"ANY"}}
	}`)
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if len(decoded.Request.Tools) != 1 || decoded.Request.Tools[0].Name != "get_weather" {
		t.Errorf("Tools = %#v", decoded.Request.Tools)
	}
	if decoded.Request.ToolChoice != inference.ToolRequired() {
		t.Errorf("ToolChoice = %v, want ToolRequired()", decoded.Request.ToolChoice)
	}
}

func TestServerDecode_ThinkingBudget(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		budget int
		want   model.Effort
	}{
		{budget: 1024, want: model.EffortLow},
		{budget: 8192, want: model.EffortMedium},
		{budget: 24576, want: model.EffortHigh},
	} {
		tc := tc
		t.Run(strconv.Itoa(tc.budget), func(t *testing.T) {
			t.Parallel()
			c, req := decodeReq(t, "m", `{
				"contents": [],
				"generationConfig": {"thinkingConfig":{"thinkingBudget":`+strconv.Itoa(tc.budget)+`}}
			}`)
			decoded, err := c.DecodeRequest(req)
			if err != nil {
				t.Fatalf("DecodeRequest() error = %v", err)
			}
			if decoded.Request.Override == nil || decoded.Request.Override.Effort != tc.want {
				t.Fatalf("Override = %#v, want effort %q", decoded.Request.Override, tc.want)
			}
		})
	}

	t.Run("dynamic budget has no neutral Gemini effort", func(t *testing.T) {
		t.Parallel()
		c, req := decodeReq(t, "m", `{
			"contents": [],
			"generationConfig": {"thinkingConfig":{"thinkingBudget":-1}}
		}`)
		if _, err := c.DecodeRequest(req); err == nil {
			t.Fatal("DecodeRequest() error = nil, want unsupported_thinking_budget")
		}
	})
}

// TestServerDecode_AllowedFunctionNames covers the mode-ANY-plus-allowlist
// form. A single allowed name is exactly the neutral named choice; an
// allowlist with more than one member restricts the model in a way the neutral
// vocabulary cannot express, so it fails closed instead of being silently
// widened back to an unrestricted ToolRequired.
func TestServerDecode_AllowedFunctionNames(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name       string
		toolConfig string
		want       inference.ToolChoice
		wantError  bool
	}{
		{
			name:       "one allowed name",
			toolConfig: `{"functionCallingConfig":{"mode":"ANY","allowedFunctionNames":["get_weather"]}}`,
			want:       inference.ToolNamed("get_weather"),
		},
		{
			name:       "empty allowlist is unrestricted",
			toolConfig: `{"functionCallingConfig":{"mode":"ANY","allowedFunctionNames":[]}}`,
			want:       inference.ToolRequired(),
		},
		{
			name:       "two allowed names",
			toolConfig: `{"functionCallingConfig":{"mode":"ANY","allowedFunctionNames":["a","b"]}}`,
			wantError:  true,
		},
		{
			name:       "allowlist on AUTO",
			toolConfig: `{"functionCallingConfig":{"mode":"AUTO","allowedFunctionNames":["a"]}}`,
			wantError:  true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			c, req := decodeReq(t, "m", `{
				"contents": [],
				"tools": [{"functionDeclarations":[{"name":"get_weather","parameters":{"type":"object"}}]}],
				"toolConfig": `+tc.toolConfig+`
			}`)
			decoded, err := c.DecodeRequest(req)
			if tc.wantError {
				if err == nil {
					t.Fatalf("DecodeRequest() error = nil, want rejection of %s", tc.toolConfig)
				}
				return
			}
			if err != nil {
				t.Fatalf("DecodeRequest() error = %v", err)
			}
			if decoded.Request.ToolChoice != tc.want {
				t.Errorf("ToolChoice = %v, want %v", decoded.Request.ToolChoice, tc.want)
			}
		})
	}
}

// TestServerDecode_FunctionDeclarationParameterFields covers both of
// FunctionDeclaration's mutually exclusive parameter fields on the way IN.
//
// parametersJsonSchema became reachable when the encoder started using it, and
// a decoder that reads only `parameters` would drop such a tool's schema
// entirely. `parameters` is Gemini's dialect, so its uppercase type names are
// mapped back to JSON Schema's. A declaration setting both fields is illegal
// per Google and is refused rather than resolved by guesswork.
func TestServerDecode_FunctionDeclarationParameterFields(t *testing.T) {
	t.Parallel()

	t.Run("parametersJsonSchema is taken verbatim", func(t *testing.T) {
		t.Parallel()
		schema := `{"type":"object","properties":{"q":{"type":"string"}},"additionalProperties":false}`
		c, req := decodeReq(t, "m", `{
			"contents": [],
			"tools": [{"functionDeclarations":[{"name":"search","parametersJsonSchema":`+schema+`}]}]
		}`)
		decoded, err := c.DecodeRequest(req)
		if err != nil {
			t.Fatalf("DecodeRequest() error = %v", err)
		}
		if len(decoded.Request.Tools) != 1 {
			t.Fatalf("Tools = %#v, want one", decoded.Request.Tools)
		}
		assertSameJSON(t, decoded.Request.Tools[0].Schema, json.RawMessage(schema))
	})

	t.Run("Gemini's uppercase types are mapped back to JSON Schema", func(t *testing.T) {
		t.Parallel()
		c, req := decodeReq(t, "m", `{
			"contents": [],
			"tools": [{"functionDeclarations":[{"name":"get_weather","parameters":{"type":"OBJECT","properties":{"city":{"type":"STRING"},"days":{"type":"ARRAY","items":{"type":"INTEGER"}}},"required":["city"]}}]}]
		}`)
		decoded, err := c.DecodeRequest(req)
		if err != nil {
			t.Fatalf("DecodeRequest() error = %v", err)
		}
		if len(decoded.Request.Tools) != 1 {
			t.Fatalf("Tools = %#v, want one", decoded.Request.Tools)
		}
		want := json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"array","items":{"type":"integer"}}},"required":["city"]}`)
		assertSameJSON(t, decoded.Request.Tools[0].Schema, want)
	})

	t.Run("both fields at once is refused", func(t *testing.T) {
		t.Parallel()
		c, req := decodeReq(t, "m", `{
			"contents": [],
			"tools": [{"functionDeclarations":[{"name":"search","parameters":{"type":"OBJECT"},"parametersJsonSchema":{"type":"object"}}]}]
		}`)
		if _, err := c.DecodeRequest(req); err == nil {
			t.Fatal("DecodeRequest() error = nil, want the mutually exclusive fields refused")
		}
	})
}

func assertSameJSON(t *testing.T, got, want json.RawMessage) {
	t.Helper()
	var gotValue, wantValue any
	if err := json.Unmarshal(got, &gotValue); err != nil {
		t.Fatalf("unmarshal got %s: %v", got, err)
	}
	if err := json.Unmarshal(want, &wantValue); err != nil {
		t.Fatalf("unmarshal want %s: %v", want, err)
	}
	if !reflect.DeepEqual(gotValue, wantValue) {
		t.Errorf("schema = %s, want %s", got, want)
	}
}

func TestServerDecode_StructuredOutput(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, "m", `{
		"contents": [],
		"generationConfig": {"responseMimeType":"application/json","responseJsonSchema":{"type":"object"}}
	}`)
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if decoded.Request.Output == nil {
		t.Fatal("Output is nil")
	}
}

func TestServerDecode_MalformedBodyNeverPanics(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, "m", `{"contents":`)
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("DecodeRequest panicked: %v", r)
		}
	}()
	if _, err := c.DecodeRequest(req); err == nil {
		t.Fatal("DecodeRequest() error = nil, want rejection of malformed body")
	}
}

func TestServerDecode_DuplicateKey(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, "m", `{"contents":[],"contents":[]}`)
	if _, err := c.DecodeRequest(req); err == nil {
		t.Fatal("DecodeRequest() error = nil, want rejection of duplicate key")
	}
}

func TestServerDecode_UnknownField(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, "m", `{"contents":[],"candidateCount":2}`)
	if _, err := c.DecodeRequest(req); err == nil {
		t.Fatal("DecodeRequest() error = nil, want rejection of unmodeled candidateCount (multi-candidate)")
	}
}

func TestServerDecode_WrongContentType(t *testing.T) {
	t.Parallel()
	c := geminiapi.Codec{}
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/m:generateContent", bytes.NewReader([]byte(`{}`)))
	req.Header.Set("Content-Type", "text/plain")
	if _, err := c.DecodeRequest(req); err == nil {
		t.Fatal("DecodeRequest() error = nil, want rejection of wrong content type")
	}
}

func TestServerDecode_UnsupportedRole(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, "m", `{"contents":[{"role":"weird","parts":[{"text":"x"}]}]}`)
	if _, err := c.DecodeRequest(req); err == nil {
		t.Fatal("DecodeRequest() error = nil, want rejection of unsupported role")
	}
}
