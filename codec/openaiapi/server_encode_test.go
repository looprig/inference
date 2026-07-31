package openaiapi_test

import (
	"encoding/json"
	"net/http/httptest"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/openaiapi"
	stream "github.com/looprig/inference/stream"
	usage "github.com/looprig/inference/usage"
)

func TestServerEncode_WriteResponse_TextOnly(t *testing.T) {
	t.Parallel()
	c := openaiapi.Codec{}
	rec := httptest.NewRecorder()

	resp := &inference.Response{
		Message: &content.AIMessage{
			Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{&content.TextBlock{Text: "hello"}}},
		},
		Model:        "gpt-test",
		Usage:        &usage.Usage{InputTokens: 10, OutputTokens: 5},
		FinishReason: stream.FinishReasonStop,
	}
	if err := c.WriteResponse(rec, resp); err != nil {
		t.Fatalf("WriteResponse() error = %v", err)
	}
	if rec.Code != 200 {
		t.Fatalf("status = %d", rec.Code)
	}

	var wire map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &wire); err != nil {
		t.Fatalf("response not valid JSON: %v", err)
	}
	choices := wire["choices"].([]any)
	msg := choices[0].(map[string]any)["message"].(map[string]any)
	if msg["role"] != "assistant" {
		t.Errorf("role = %v", msg["role"])
	}
	if msg["content"] != "hello" {
		t.Errorf("content = %v", msg["content"])
	}
	if choices[0].(map[string]any)["finish_reason"] != "stop" {
		t.Errorf("finish_reason = %v", choices[0].(map[string]any)["finish_reason"])
	}
	usageWire := wire["usage"].(map[string]any)
	if usageWire["prompt_tokens"].(float64) != 10 || usageWire["completion_tokens"].(float64) != 5 {
		t.Errorf("usage = %v", usageWire)
	}
}

func TestServerEncode_WriteResponse_ToolCallsHaveNullContent(t *testing.T) {
	t.Parallel()
	c := openaiapi.Codec{}
	rec := httptest.NewRecorder()

	resp := &inference.Response{
		Message: &content.AIMessage{
			Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{
				&content.ToolUseBlock{ID: "call_1", Name: "get_weather", Input: json.RawMessage(`{"city":"nyc"}`)},
			}},
		},
		FinishReason: stream.FinishReasonToolUse,
	}
	if err := c.WriteResponse(rec, resp); err != nil {
		t.Fatalf("WriteResponse() error = %v", err)
	}

	var wire map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &wire); err != nil {
		t.Fatalf("response not valid JSON: %v", err)
	}
	choices := wire["choices"].([]any)
	msg := choices[0].(map[string]any)["message"].(map[string]any)
	if msg["content"] != nil {
		t.Errorf("content = %v, want null", msg["content"])
	}
	calls := msg["tool_calls"].([]any)
	fn := calls[0].(map[string]any)["function"].(map[string]any)
	if fn["name"] != "get_weather" {
		t.Errorf("name = %v", fn["name"])
	}
	// arguments must be a JSON-encoded STRING, not a bare object.
	argsRaw, ok := fn["arguments"].(string)
	if !ok {
		t.Fatalf("arguments type = %T, want string", fn["arguments"])
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(argsRaw), &args); err != nil {
		t.Fatalf("arguments not valid JSON: %v", err)
	}
	if args["city"] != "nyc" {
		t.Errorf("args = %v", args)
	}
	if choices[0].(map[string]any)["finish_reason"] != "tool_calls" {
		t.Errorf("finish_reason = %v", choices[0].(map[string]any)["finish_reason"])
	}
}

func TestServerEncode_WriteResponse_ReasoningContent(t *testing.T) {
	t.Parallel()
	c := openaiapi.Codec{}
	rec := httptest.NewRecorder()
	resp := &inference.Response{
		Message: &content.AIMessage{
			Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{
				content.NewThinkingBlock("thinking...", "", nil, ""),
				&content.TextBlock{Text: "answer"},
			}},
		},
	}
	if err := c.WriteResponse(rec, resp); err != nil {
		t.Fatalf("WriteResponse() error = %v", err)
	}
	var wire map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &wire); err != nil {
		t.Fatalf("response not valid JSON: %v", err)
	}
	msg := wire["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	if msg["reasoning_content"] != "thinking..." {
		t.Errorf("reasoning_content = %v", msg["reasoning_content"])
	}
	if msg["content"] != "answer" {
		t.Errorf("content = %v", msg["content"])
	}
}

func TestServerEncode_WriteError_NeverPanics(t *testing.T) {
	t.Parallel()
	c := openaiapi.Codec{}
	rec := httptest.NewRecorder()
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("WriteError panicked: %v", r)
		}
	}()
	c.WriteError(rec, nil)
	if rec.Code < 400 {
		t.Errorf("status = %d, want 4xx/5xx", rec.Code)
	}

	rec2 := httptest.NewRecorder()
	c.WriteError(rec2, &openaiapi.ServerDecodeError{Reason: "missing_model"})
	if rec2.Code != 400 {
		t.Errorf("status = %d, want 400", rec2.Code)
	}
	var wire map[string]any
	if err := json.Unmarshal(rec2.Body.Bytes(), &wire); err != nil {
		t.Fatalf("error body not valid JSON: %v", err)
	}
	if _, ok := wire["error"]; !ok {
		t.Errorf("body = %v, want top-level error key", wire)
	}
}
