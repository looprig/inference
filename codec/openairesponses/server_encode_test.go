package openairesponses_test

import (
	"encoding/json"
	"net/http/httptest"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/openairesponses"
	stream "github.com/looprig/inference/stream"
	usage "github.com/looprig/inference/usage"
)

func TestServerEncode_WriteResponseText(t *testing.T) {
	t.Parallel()
	c := openairesponses.Codec{}
	rec := httptest.NewRecorder()
	resp := &inference.Response{
		Message: &content.AIMessage{
			Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{&content.TextBlock{Text: "hello"}}},
		},
		Model:        "gpt-test",
		Usage:        &usage.Usage{InputTokens: 10, OutputTokens: 5, CacheReadTokens: 2},
		FinishReason: stream.FinishReasonStop,
	}
	if err := c.WriteResponse(rec, resp); err != nil {
		t.Fatalf("WriteResponse() error = %v", err)
	}
	if rec.Code != 200 {
		t.Fatalf("status = %d", rec.Code)
	}
	var m map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if m["status"] != "completed" {
		t.Errorf("status = %v", m["status"])
	}
	if m["model"] != "gpt-test" {
		t.Errorf("model = %v", m["model"])
	}
	output := m["output"].([]any)
	item := output[0].(map[string]any)
	if item["type"] != "message" || item["role"] != "assistant" {
		t.Errorf("item = %#v", item)
	}
	parts := item["content"].([]any)
	if parts[0].(map[string]any)["text"] != "hello" {
		t.Errorf("parts = %#v", parts)
	}
	u := m["usage"].(map[string]any)
	// gross input = InputTokens(10) + CacheReadTokens(2) = 12
	if u["input_tokens"] != float64(12) {
		t.Errorf("input_tokens = %v, want 12", u["input_tokens"])
	}
}

func TestServerEncode_ToolCallGetsSynthesizedID(t *testing.T) {
	t.Parallel()
	c := openairesponses.Codec{}
	rec := httptest.NewRecorder()
	resp := &inference.Response{
		Message: &content.AIMessage{
			Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{
				&content.ToolUseBlock{Name: "get_weather", Input: json.RawMessage(`{"city":"nyc"}`)}, // no ID
			}},
		},
		Model:        "gpt-test",
		FinishReason: stream.FinishReasonToolUse,
	}
	if err := c.WriteResponse(rec, resp); err != nil {
		t.Fatalf("WriteResponse() error = %v", err)
	}
	var m map[string]any
	json.Unmarshal(rec.Body.Bytes(), &m)
	output := m["output"].([]any)
	item := output[0].(map[string]any)
	callID, _ := item["call_id"].(string)
	if callID == "" {
		t.Error("call_id was not synthesized")
	}
}

func TestServerEncode_IncompleteMaxOutputTokens(t *testing.T) {
	t.Parallel()
	c := openairesponses.Codec{}
	rec := httptest.NewRecorder()
	resp := &inference.Response{
		Message:      &content.AIMessage{Message: content.Message{Role: content.RoleAssistant}},
		FinishReason: stream.FinishReasonLength,
	}
	if err := c.WriteResponse(rec, resp); err != nil {
		t.Fatalf("WriteResponse() error = %v", err)
	}
	var m map[string]any
	json.Unmarshal(rec.Body.Bytes(), &m)
	if m["status"] != "incomplete" {
		t.Errorf("status = %v, want incomplete", m["status"])
	}
	details := m["incomplete_details"].(map[string]any)
	if details["reason"] != "max_output_tokens" {
		t.Errorf("reason = %v", details["reason"])
	}
}

func TestServerEncode_ThinkingBlockToReasoningItem(t *testing.T) {
	t.Parallel()
	c := openairesponses.Codec{}
	rec := httptest.NewRecorder()
	resp := &inference.Response{
		Message: &content.AIMessage{
			Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{
				content.NewThinkingBlock("step 1", "", json.RawMessage(`"opaque-xyz"`), "openai-responses"),
			}},
		},
		FinishReason: stream.FinishReasonStop,
	}
	if err := c.WriteResponse(rec, resp); err != nil {
		t.Fatalf("WriteResponse() error = %v", err)
	}
	var m map[string]any
	json.Unmarshal(rec.Body.Bytes(), &m)
	output := m["output"].([]any)
	item := output[0].(map[string]any)
	if item["type"] != "reasoning" {
		t.Fatalf("item type = %v", item["type"])
	}
	if item["encrypted_content"] != "opaque-xyz" {
		t.Errorf("encrypted_content = %v", item["encrypted_content"])
	}
}

func TestServerEncode_WriteErrorShape(t *testing.T) {
	t.Parallel()
	c := openairesponses.Codec{}
	rec := httptest.NewRecorder()
	c.WriteError(rec, &openairesponses.ServerDecodeError{Reason: "missing_model"})
	if rec.Code != 400 {
		t.Errorf("status = %d, want 400", rec.Code)
	}
	var m map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	errObj, ok := m["error"].(map[string]any)
	if !ok {
		t.Fatalf("error field missing or wrong type: %#v", m)
	}
	if errObj["message"] == "" {
		t.Error("error.message empty")
	}
}
