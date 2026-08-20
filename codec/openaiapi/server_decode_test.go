package openaiapi_test

import (
	"bytes"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/openaiapi"
	"github.com/looprig/inference/model"
)

// newDecodeRequest builds an httptest request for POST /v1/chat/completions
// carrying body.
func newDecodeRequest(body string) *http.Request {
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")
	return req
}

func decodeReq(t *testing.T, body string) (*openaiapi.Codec, *http.Request) {
	t.Helper()
	c := &openaiapi.Codec{}
	return c, newDecodeRequest(body)
}

func TestServerDecode_MatchesRoute(t *testing.T) {
	t.Parallel()
	c := openaiapi.Codec{}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	if !c.MatchRequest(req) {
		t.Error("MatchRequest() = false, want true")
	}
	if c.MatchRequest(httptest.NewRequest(http.MethodGet, "/v1/chat/completions", nil)) {
		t.Error("MatchRequest(GET) = true, want false")
	}
	if c.MatchRequest(httptest.NewRequest(http.MethodPost, "/v1/responses", nil)) {
		t.Error("MatchRequest(/v1/responses) = true, want false")
	}
}

func TestServerDecode_SystemFoldsIntoRequestSystem(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{
		"model": "gpt-test",
		"messages": [
			{"role":"system","content":"be terse"},
			{"role":"user","content":"hi"}
		]
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
		t.Fatalf("Messages len = %d, want 1 (system folded away)", len(decoded.Request.Messages))
	}
	um, ok := decoded.Request.Messages[0].(*content.UserMessage)
	if !ok {
		t.Fatalf("Messages[0] type = %T", decoded.Request.Messages[0])
	}
	if um.Blocks[0].(*content.TextBlock).Text != "hi" {
		t.Errorf("text = %q", um.Blocks[0].(*content.TextBlock).Text)
	}
}

func TestServerDecode_MultipleSystemMessagesJoin(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{
		"model": "gpt-test",
		"messages": [
			{"role":"system","content":"first"},
			{"role":"system","content":"second"}
		]
	}`)
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if decoded.Request.System != "first\n\nsecond" {
		t.Errorf("System = %q", decoded.Request.System)
	}
}

func TestServerDecode_AssistantTextReasoningAndToolCalls(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{
		"model": "gpt-test",
		"messages": [
			{"role":"assistant","reasoning_content":"thinking it through","content":"answer","tool_calls":[
				{"id":"call_1","type":"function","function":{"name":"f","arguments":"{\"x\":1}"}}
			]}
		]
	}`)
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if len(decoded.Request.Messages) != 1 {
		t.Fatalf("Messages len = %d", len(decoded.Request.Messages))
	}
	ai, ok := decoded.Request.Messages[0].(*content.AIMessage)
	if !ok {
		t.Fatalf("Messages[0] type = %T", decoded.Request.Messages[0])
	}
	if len(ai.Blocks) != 3 {
		t.Fatalf("Blocks len = %d, want 3 (reasoning, text, tool_use)", len(ai.Blocks))
	}
	tb, ok := ai.Blocks[0].(*content.ThinkingBlock)
	if !ok || tb.Thinking != "thinking it through" {
		t.Errorf("Blocks[0] = %#v", ai.Blocks[0])
	}
	if tb.ProviderState != nil {
		t.Errorf("ProviderState = %q, want nil (Chat Completions has no opaque reasoning state)", tb.ProviderState)
	}
	txt, ok := ai.Blocks[1].(*content.TextBlock)
	if !ok || txt.Text != "answer" {
		t.Errorf("Blocks[1] = %#v", ai.Blocks[1])
	}
	tu, ok := ai.Blocks[2].(*content.ToolUseBlock)
	if !ok || tu.ID != "call_1" || tu.Name != "f" || string(tu.Input) != `{"x":1}` {
		t.Errorf("Blocks[2] = %#v", ai.Blocks[2])
	}
}

func TestServerDecode_AssistantContentNullWithOnlyToolCalls(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{
		"model": "gpt-test",
		"messages": [
			{"role":"assistant","content":null,"tool_calls":[
				{"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}}
			]}
		]
	}`)
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	ai := decoded.Request.Messages[0].(*content.AIMessage)
	if len(ai.Blocks) != 1 {
		t.Fatalf("Blocks len = %d, want 1 (tool_use only)", len(ai.Blocks))
	}
}

func TestServerDecode_ToolResultMessage(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{
		"model": "gpt-test",
		"messages": [{"role":"tool","tool_call_id":"call_1","content":"sunny"}]
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

func TestServerDecode_ToolResultMissingCallID(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{"model":"m","messages":[{"role":"tool","content":"x"}]}`)
	if _, err := c.DecodeRequest(req); err == nil {
		t.Fatal("DecodeRequest() error = nil, want rejection of tool message with no tool_call_id")
	}
}

func TestServerDecode_ImageURLAndDataURI(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{
		"model": "gpt-test",
		"messages": [{"role":"user","content":[
			{"type":"text","text":"look"},
			{"type":"image_url","image_url":{"url":"data:image/png;base64,aGVsbG8="}},
			{"type":"image_url","image_url":{"url":"https://example.com/x.png"}}
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
		"tools": [{"type":"function","function":{"name":"get_weather","description":"gets weather","parameters":{"type":"object"}}}],
		"tool_choice": "required",
		"messages": []
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

func TestServerDecode_RejectsToolChoiceNone(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{"model":"m","tool_choice":"none","messages":[]}`)
	if _, err := c.DecodeRequest(req); err == nil {
		t.Fatal("DecodeRequest() error = nil, want rejection of tool_choice:none")
	}
}

// TestServerDecode_NamedToolChoice covers the ChatCompletionNamedToolChoice
// object form, which the neutral vocabulary now represents as a named choice
// plus a name.
func TestServerDecode_NamedToolChoice(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{"model":"m","tool_choice":{"type":"function","function":{"name":"f"}},"messages":[]}`)
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if decoded.Request.ToolChoice != inference.ToolNamed("f") {
		t.Errorf("ToolChoice = %v, want ToolNamed(f)", decoded.Request.ToolChoice)
	}
}

// TestServerDecode_RejectsUnrepresentableToolChoiceObjects keeps the object
// forms outside the neutral vocabulary failing closed. The custom-tool and
// allowed-tools members of ChatCompletionToolChoiceOption are real wire shapes
// with no neutral spelling, so they must not degrade into a function choice.
func TestServerDecode_RejectsUnrepresentableToolChoiceObjects(t *testing.T) {
	t.Parallel()
	cases := map[string]string{
		"custom tool":           `{"type":"custom","custom":{"name":"f"}}`,
		"allowed tools":         `{"type":"allowed_tools","mode":"required","tools":[]}`,
		"function with no name": `{"type":"function","function":{}}`,
	}
	for name, toolChoice := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			c, req := decodeReq(t, `{"model":"m","tool_choice":`+toolChoice+`,"messages":[]}`)
			if _, err := c.DecodeRequest(req); err == nil {
				t.Fatalf("DecodeRequest() error = nil, want rejection of %s", toolChoice)
			}
		})
	}
}

func TestServerDecode_ReasoningEffort(t *testing.T) {
	t.Parallel()
	for _, effort := range []model.Effort{
		model.EffortMinimal, model.EffortLow, model.EffortMedium, model.EffortHigh, model.EffortXHigh, model.EffortMax,
	} {
		effort := effort
		t.Run(string(effort), func(t *testing.T) {
			t.Parallel()
			c, req := decodeReq(t, `{"model":"m","reasoning_effort":"`+string(effort)+`","messages":[]}`)
			decoded, err := c.DecodeRequest(req)
			if err != nil {
				t.Fatalf("DecodeRequest() error = %v", err)
			}
			if decoded.Request.Override == nil || decoded.Request.Override.Effort != effort {
				t.Errorf("Override = %#v, want effort %q", decoded.Request.Override, effort)
			}
		})
	}
}

func TestServerDecode_StructuredOutput(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{
		"model": "m",
		"response_format": {"type":"json_schema","json_schema":{"name":"answer","strict":true,"schema":{"type":"object"}}},
		"messages": []
	}`)
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if decoded.Request.Output == nil || decoded.Request.Output.Name != "answer" || !decoded.Request.Output.Strict {
		t.Errorf("Output = %#v", decoded.Request.Output)
	}
}

func TestServerDecode_RejectsMultipleChoices(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{"model":"m","n":2,"messages":[]}`)
	_, err := c.DecodeRequest(req)
	if err == nil {
		t.Fatal("DecodeRequest() error = nil, want rejection of n>1")
	}
	var nErr *openaiapi.UnsupportedChoiceCountError
	if !errors.As(err, &nErr) {
		t.Errorf("error = %v, want *UnsupportedChoiceCountError", err)
	}
}

func TestServerDecode_AllowsNEqualsOne(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{"model":"m","n":1,"messages":[]}`)
	if _, err := c.DecodeRequest(req); err != nil {
		t.Fatalf("DecodeRequest() error = %v, want n=1 accepted", err)
	}
}

func TestServerDecode_BenignFieldsAcceptedAndIgnored(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{
		"model": "m",
		"parallel_tool_calls": true,
		"user": "harness-1",
		"seed": 42,
		"stream_options": {"include_usage": true},
		"messages": []
	}`)
	if _, err := c.DecodeRequest(req); err != nil {
		t.Fatalf("DecodeRequest() error = %v, want benign fields accepted", err)
	}
}

func TestServerDecode_MissingModel(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{"messages":[]}`)
	if _, err := c.DecodeRequest(req); err == nil {
		t.Fatal("DecodeRequest() error = nil, want rejection of missing model")
	}
}

func TestServerDecode_DuplicateKey(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{"model":"m","model":"m2","messages":[]}`)
	if _, err := c.DecodeRequest(req); err == nil {
		t.Fatal("DecodeRequest() error = nil, want rejection of duplicate key")
	}
}

func TestServerDecode_UnknownField(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{"model":"m","unknown_field":true,"messages":[]}`)
	if _, err := c.DecodeRequest(req); err == nil {
		t.Fatal("DecodeRequest() error = nil, want rejection of unknown field")
	}
}

func TestServerDecode_WrongContentType(t *testing.T) {
	t.Parallel()
	c := openaiapi.Codec{}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte(`{}`)))
	req.Header.Set("Content-Type", "text/plain")
	if _, err := c.DecodeRequest(req); err == nil {
		t.Fatal("DecodeRequest() error = nil, want rejection of wrong content type")
	}
}

func TestServerDecode_StreamFlag(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{"model":"m","stream":true,"messages":[]}`)
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if !decoded.Streaming {
		t.Error("Streaming = false, want true")
	}
}

func TestServerDecode_UnsupportedRole(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{"model":"m","messages":[{"role":"weird","content":"x"}]}`)
	if _, err := c.DecodeRequest(req); err == nil {
		t.Fatal("DecodeRequest() error = nil, want rejection of unsupported role")
	}
}

func TestServerDecode_MalformedBodyNeverPanics(t *testing.T) {
	t.Parallel()
	c := openaiapi.Codec{}
	req := newDecodeRequest(`{"model":`)
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("DecodeRequest panicked: %v", r)
		}
	}()
	if _, err := c.DecodeRequest(req); err == nil {
		t.Fatal("DecodeRequest() error = nil, want rejection of malformed body")
	}
}

// TestServerDecode_MaxCompletionTokens covers the ingress half of the
// deprecated-max_tokens migration. decodeChatCompletionsBody runs with
// DisallowUnknownFields, so a client sending the modern (and, for o-series,
// mandatory) max_completion_tokens spelling must be recognized rather than
// failed closed as an unknown field; it maps to the same neutral
// Sampling.MaxTokens as the legacy name.
func TestServerDecode_MaxCompletionTokens(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{
		"model": "gpt-test",
		"messages": [{"role":"user","content":"hi"}],
		"max_completion_tokens": 512
	}`)
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if decoded.Request.Override == nil {
		t.Fatal("Override = nil, want MaxTokens 512")
	}
	got := decoded.Request.Override.MaxTokens
	if got == nil {
		t.Fatal("Override.MaxTokens = nil, want 512")
	}
	if *got != 512 {
		t.Errorf("Override.MaxTokens = %d, want 512", *got)
	}
}

// TestServerDecode_MaxTokensConflict rejects a body carrying both token-limit
// spellings: they are mutually exclusive on the wire and picking one silently
// would let a client smuggle a different limit past review.
func TestServerDecode_MaxTokensConflict(t *testing.T) {
	t.Parallel()
	c, req := decodeReq(t, `{
		"model": "gpt-test",
		"messages": [{"role":"user","content":"hi"}],
		"max_tokens": 16,
		"max_completion_tokens": 512
	}`)
	if _, err := c.DecodeRequest(req); err == nil {
		t.Fatal("DecodeRequest() error = nil, want conflict error")
	}
}
