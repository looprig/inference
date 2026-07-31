package geminiapi_test

import (
	"encoding/json"
	"net/http/httptest"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/geminiapi"
	stream "github.com/looprig/inference/stream"
	usage "github.com/looprig/inference/usage"
)

func TestServerEncode_WriteResponse_TextOnly(t *testing.T) {
	t.Parallel()
	c := geminiapi.Codec{}
	rec := httptest.NewRecorder()

	resp := &inference.Response{
		Message: &content.AIMessage{
			Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{&content.TextBlock{Text: "hello"}}},
		},
		Model:        "gemini-test",
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
	candidates := wire["candidates"].([]any)
	cand := candidates[0].(map[string]any)
	if cand["finishReason"] != "STOP" {
		t.Errorf("finishReason = %v", cand["finishReason"])
	}
	parts := cand["content"].(map[string]any)["parts"].([]any)
	if parts[0].(map[string]any)["text"] != "hello" {
		t.Errorf("text = %v", parts[0])
	}
	if cand["content"].(map[string]any)["role"] != "model" {
		t.Errorf("role = %v", cand["content"])
	}
	usageWire := wire["usageMetadata"].(map[string]any)
	if usageWire["promptTokenCount"].(float64) != 10 || usageWire["candidatesTokenCount"].(float64) != 5 {
		t.Errorf("usage = %v", usageWire)
	}
}

func TestServerEncode_WriteResponse_ToolCall(t *testing.T) {
	t.Parallel()
	c := geminiapi.Codec{}
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
	cand := wire["candidates"].([]any)[0].(map[string]any)
	// FinishReasonToolUse has no dedicated Gemini wire value: real Gemini
	// reports STOP even with a functionCall present.
	if cand["finishReason"] != "STOP" {
		t.Errorf("finishReason = %v, want STOP", cand["finishReason"])
	}
	parts := cand["content"].(map[string]any)["parts"].([]any)
	fc := parts[0].(map[string]any)["functionCall"].(map[string]any)
	if fc["name"] != "get_weather" {
		t.Errorf("name = %v", fc["name"])
	}
	args := fc["args"].(map[string]any)
	if args["city"] != "nyc" {
		t.Errorf("args = %v", args)
	}
}

func TestServerEncode_WriteResponse_ThinkingWithSignature(t *testing.T) {
	t.Parallel()
	c := geminiapi.Codec{}
	rec := httptest.NewRecorder()
	resp := &inference.Response{
		Message: &content.AIMessage{
			Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{
				content.NewThinkingBlock("planning", "", json.RawMessage(`"opaque-sig"`), "gemini"),
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
	parts := wire["candidates"].([]any)[0].(map[string]any)["content"].(map[string]any)["parts"].([]any)
	thoughtPart := parts[0].(map[string]any)
	if thoughtPart["text"] != "planning" {
		t.Errorf("text = %v", thoughtPart["text"])
	}
	if thoughtPart["thought"] != true {
		t.Errorf("thought = %v, want true", thoughtPart["thought"])
	}
	if thoughtPart["thoughtSignature"] != "opaque-sig" {
		t.Errorf("thoughtSignature = %v, want opaque-sig", thoughtPart["thoughtSignature"])
	}
}

func TestServerEncode_WriteResponse_ThinkingWithoutSignatureStillWritten(t *testing.T) {
	t.Parallel()
	// Unlike the outbound REQUEST encoder (encode.go), the harness-facing
	// response encoder must not drop visible reasoning just because it lacks
	// a replay signature — the harness is not Gemini itself.
	c := geminiapi.Codec{}
	rec := httptest.NewRecorder()
	resp := &inference.Response{
		Message: &content.AIMessage{
			Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{
				content.NewThinkingBlock("planning", "", nil, ""),
			}},
		},
	}
	if err := c.WriteResponse(rec, resp); err != nil {
		t.Fatalf("WriteResponse() error = %v", err)
	}
	var wire map[string]any
	json.Unmarshal(rec.Body.Bytes(), &wire)
	parts := wire["candidates"].([]any)[0].(map[string]any)["content"].(map[string]any)["parts"].([]any)
	if len(parts) != 1 {
		t.Fatalf("parts len = %d, want 1 (thinking must be written)", len(parts))
	}
	if _, hasSig := parts[0].(map[string]any)["thoughtSignature"]; hasSig {
		t.Errorf("thoughtSignature present = %v, want omitted for nil ProviderState", parts[0])
	}
}

func TestServerEncode_WriteError_NeverPanics(t *testing.T) {
	t.Parallel()
	c := geminiapi.Codec{}
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
	c.WriteError(rec2, &geminiapi.ServerDecodeError{Reason: "missing_model"})
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
