package openairesponses_test

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/openairesponses"
)

func decodeReq(t *testing.T, body string) (*openairesponses.Codec, *http.Request) {
	t.Helper()
	c := &openairesponses.Codec{}
	return c, newDecodeRequest(body)
}

// newDecodeRequest builds an httptest request for POST /v1/responses
// carrying body.
func newDecodeRequest(body string) *http.Request {
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")
	return req
}

func TestServerDecode_MatchesRoute(t *testing.T) {
	t.Parallel()
	c := openairesponses.Codec{}
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	if !c.MatchRequest(req) {
		t.Error("MatchRequest() = false, want true")
	}
	if c.MatchRequest(httptest.NewRequest(http.MethodGet, "/v1/responses", nil)) {
		t.Error("MatchRequest(GET) = true, want false")
	}
	if c.MatchRequest(httptest.NewRequest(http.MethodPost, "/v1/messages", nil)) {
		t.Error("MatchRequest(/v1/messages) = true, want false")
	}
}

func TestServerDecode_InstructionsAndUserMessage(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{
		"model": "gpt-test",
		"instructions": "be terse",
		"input": [{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]
	}`)
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if decoded.RequestedModel != "gpt-test" {
		t.Errorf("RequestedModel = %q", decoded.RequestedModel)
	}
	if decoded.Request.System != "be terse" {
		t.Errorf("System = %q", decoded.Request.System)
	}
	if len(decoded.Request.Messages) != 1 {
		t.Fatalf("Messages len = %d", len(decoded.Request.Messages))
	}
	um, ok := decoded.Request.Messages[0].(*content.UserMessage)
	if !ok {
		t.Fatalf("Messages[0] type = %T", decoded.Request.Messages[0])
	}
	tb := um.Blocks[0].(*content.TextBlock)
	if tb.Text != "hi" {
		t.Errorf("text = %q", tb.Text)
	}
}

func TestServerDecode_SystemInputItemFoldsIntoInstructions(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{
		"model": "gpt-test",
		"input": [{"type":"message","role":"system","content":[{"type":"input_text","text":"sys text"}]}]
	}`)
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if len(decoded.Request.Messages) != 1 {
		t.Fatalf("Messages len = %d", len(decoded.Request.Messages))
	}
	sm, ok := decoded.Request.Messages[0].(*content.SystemMessage)
	if !ok {
		t.Fatalf("Messages[0] type = %T, want *content.SystemMessage", decoded.Request.Messages[0])
	}
	if sm.Blocks[0].(*content.TextBlock).Text != "sys text" {
		t.Errorf("text = %q", sm.Blocks[0].(*content.TextBlock).Text)
	}
}

func TestServerDecode_AssistantTextGroupsWithToolCallAndReasoning(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{
		"model": "gpt-test",
		"input": [
			{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"thinking"}],"encrypted_content":"opaque"},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]},
			{"type":"function_call","call_id":"call_1","name":"f","arguments":"{\"x\":1}"}
		]
	}`)
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if len(decoded.Request.Messages) != 1 {
		t.Fatalf("Messages len = %d, want 1 (grouped into a single AIMessage)", len(decoded.Request.Messages))
	}
	ai, ok := decoded.Request.Messages[0].(*content.AIMessage)
	if !ok {
		t.Fatalf("Messages[0] type = %T", decoded.Request.Messages[0])
	}
	if len(ai.Blocks) != 3 {
		t.Fatalf("Blocks len = %d, want 3", len(ai.Blocks))
	}
	if _, ok := ai.Blocks[0].(*content.ThinkingBlock); !ok {
		t.Errorf("Blocks[0] type = %T, want ThinkingBlock", ai.Blocks[0])
	}
	if _, ok := ai.Blocks[1].(*content.TextBlock); !ok {
		t.Errorf("Blocks[1] type = %T, want TextBlock", ai.Blocks[1])
	}
	tu, ok := ai.Blocks[2].(*content.ToolUseBlock)
	if !ok || tu.ID != "call_1" {
		t.Errorf("Blocks[2] = %#v", ai.Blocks[2])
	}
}

func TestServerDecode_ToolResultMessage(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{
		"model": "gpt-test",
		"input": [{"type":"function_call_output","call_id":"call_1","output":"sunny"}]
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

func TestServerDecode_ParallelToolCallsAndResults(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{
		"model": "gpt-test",
		"input": [
			{"type":"message","role":"user","content":[{"type":"input_text","text":"go"}]},
			{"type":"function_call","call_id":"call_1","name":"a","arguments":"{}"},
			{"type":"function_call","call_id":"call_2","name":"b","arguments":"{}"},
			{"type":"function_call_output","call_id":"call_1","output":"a-result"},
			{"type":"function_call_output","call_id":"call_2","output":"b-result"}
		]
	}`)
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if len(decoded.Request.Messages) != 4 {
		t.Fatalf("Messages len = %d, want 4 (user, AI[2 tool calls], 2 tool results)", len(decoded.Request.Messages))
	}
	ai, ok := decoded.Request.Messages[1].(*content.AIMessage)
	if !ok || len(ai.Blocks) != 2 {
		t.Fatalf("Messages[1] = %#v", decoded.Request.Messages[1])
	}
}

func TestServerDecode_Images(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{
		"model": "gpt-test",
		"input": [{"type":"message","role":"user","content":[
			{"type":"input_text","text":"look"},
			{"type":"input_image","image_url":"data:image/png;base64,aGVsbG8=","detail":"auto"},
			{"type":"input_image","image_url":"https://example.com/x.png","detail":"auto"}
		]}]
	}`)
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	um := decoded.Request.Messages[0].(*content.UserMessage)
	if len(um.Blocks) != 3 {
		t.Fatalf("Blocks len = %d, want 3", len(um.Blocks))
	}
	inline := um.Blocks[1].(*content.ImageBlock)
	if string(inline.MediaType) != "image/png" || string(inline.Source.Data) != "hello" {
		t.Errorf("inline image = %#v", inline)
	}
	urlImg := um.Blocks[2].(*content.ImageBlock)
	if urlImg.Source.URL != "https://example.com/x.png" {
		t.Errorf("url image = %#v", urlImg)
	}
}

func TestServerDecode_ToolsAndRequiredToolChoice(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{
		"model": "gpt-test",
		"tools": [{"type":"function","name":"get_weather","description":"gets weather","parameters":{"type":"object"}}],
		"tool_choice": "required",
		"input": []
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

func TestServerDecode_ReasoningEffort(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{
		"model": "gpt-test",
		"reasoning": {"effort": "high", "summary": "auto"},
		"input": []
	}`)
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if decoded.Request.Override == nil {
		t.Fatal("Override is nil")
	}
}

func TestServerDecode_StructuredOutput(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{
		"model": "gpt-test",
		"text": {"format": {"type":"json_schema","name":"answer","strict":true,"schema":{"type":"object","properties":{"x":{"type":"string"}},"required":["x"],"additionalProperties":false}}},
		"input": []
	}`)
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if decoded.Request.Output == nil || decoded.Request.Output.Name != "answer" || !decoded.Request.Output.Strict {
		t.Errorf("Output = %#v", decoded.Request.Output)
	}
}

func TestServerDecode_BenignFieldsAcceptedAndIgnored(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{
		"model": "gpt-test",
		"metadata": {"user_id": "x"},
		"parallel_tool_calls": true,
		"include": ["reasoning.encrypted_content"],
		"store": false,
		"input": []
	}`)
	if _, err := c.DecodeRequest(req); err != nil {
		t.Fatalf("DecodeRequest() error = %v, want benign fields accepted", err)
	}
}

func TestServerDecode_RejectsStoreTrue(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{"model":"gpt-test","store":true,"input":[]}`)
	if _, err := c.DecodeRequest(req); err == nil {
		t.Fatal("DecodeRequest() error = nil, want rejection of store:true")
	}
}

func TestServerDecode_RejectsPreviousResponseID(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{"model":"gpt-test","previous_response_id":"resp_abc","input":[]}`)
	if _, err := c.DecodeRequest(req); err == nil {
		t.Fatal("DecodeRequest() error = nil, want rejection of previous_response_id")
	}
}

func TestServerDecode_RejectsToolChoiceNone(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{"model":"gpt-test","tool_choice":"none","input":[]}`)
	if _, err := c.DecodeRequest(req); err == nil {
		t.Fatal("DecodeRequest() error = nil, want rejection of tool_choice:none")
	}
}

func TestServerDecode_RejectsToolChoiceObjectForm(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{"model":"gpt-test","tool_choice":{"type":"function","name":"f"},"input":[]}`)
	if _, err := c.DecodeRequest(req); err == nil {
		t.Fatal("DecodeRequest() error = nil, want rejection of named tool_choice object")
	}
}

func TestServerDecode_MissingModel(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{"input":[]}`)
	if _, err := c.DecodeRequest(req); err == nil {
		t.Fatal("DecodeRequest() error = nil, want rejection of missing model")
	}
}

func TestServerDecode_DuplicateKey(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{"model":"m","model":"m2","input":[]}`)
	if _, err := c.DecodeRequest(req); err == nil {
		t.Fatal("DecodeRequest() error = nil, want rejection of duplicate key")
	}
}

func TestServerDecode_UnknownField(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{"model":"m","unknown_field":true,"input":[]}`)
	if _, err := c.DecodeRequest(req); err == nil {
		t.Fatal("DecodeRequest() error = nil, want rejection of unknown field")
	}
}

func TestServerDecode_WrongContentType(t *testing.T) {
	t.Parallel()
	c := openairesponses.Codec{}
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader([]byte(`{}`)))
	req.Header.Set("Content-Type", "text/plain")
	if _, err := c.DecodeRequest(req); err == nil {
		t.Fatal("DecodeRequest() error = nil, want rejection of wrong content type")
	}
}

func TestServerDecode_StreamFlag(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{"model":"m","stream":true,"input":[]}`)
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if !decoded.Streaming {
		t.Error("Streaming = false, want true")
	}
}
