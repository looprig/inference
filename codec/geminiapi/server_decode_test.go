package geminiapi_test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/geminiapi"
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
	if decoded.Request.ToolChoice != inference.ToolChoiceRequired {
		t.Errorf("ToolChoice = %v, want ToolChoiceRequired", decoded.Request.ToolChoice)
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
