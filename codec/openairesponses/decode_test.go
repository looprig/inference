package openairesponses_test

import (
	"encoding/json"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference/codec/openairesponses"
	stream "github.com/looprig/inference/stream"
)

func TestDecodeResponse_TextAndUsage(t *testing.T) {
	t.Parallel()
	body := []byte(`{
		"id": "resp_1",
		"status": "completed",
		"model": "gpt-test",
		"output": [
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello there","annotations":[]}]}
		],
		"usage": {"input_tokens": 55, "output_tokens": 11, "total_tokens": 66, "input_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 3}, "output_tokens_details": {"reasoning_tokens": 0}}
	}`)
	resp, err := openairesponses.DecodeResponse(body)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if resp.Model != "gpt-test" {
		t.Errorf("Model = %q", resp.Model)
	}
	if len(resp.Message.Blocks) != 1 {
		t.Fatalf("Blocks len = %d", len(resp.Message.Blocks))
	}
	tb, ok := resp.Message.Blocks[0].(*content.TextBlock)
	if !ok || tb.Text != "hello there" {
		t.Errorf("Blocks[0] = %#v", resp.Message.Blocks[0])
	}
	if resp.Usage == nil {
		t.Fatal("Usage is nil")
	}
	// gross input_tokens(55) - cached_tokens(2) - cache_write_tokens(3) = 50
	if resp.Usage.InputTokens != 50 {
		t.Errorf("InputTokens = %d, want 50", resp.Usage.InputTokens)
	}
	if resp.Usage.CacheReadTokens != 2 {
		t.Errorf("CacheReadTokens = %d, want 2", resp.Usage.CacheReadTokens)
	}
	if resp.Usage.CacheCreationTokens != 3 {
		t.Errorf("CacheCreationTokens = %d, want 3", resp.Usage.CacheCreationTokens)
	}
	if resp.Usage.OutputTokens != 11 {
		t.Errorf("OutputTokens = %d, want 11", resp.Usage.OutputTokens)
	}
	if resp.FinishReason != stream.FinishReasonStop {
		t.Errorf("FinishReason = %v, want Stop", resp.FinishReason)
	}
}

func TestDecodeResponse_ToolCallFinishReason(t *testing.T) {
	t.Parallel()
	body := []byte(`{
		"id": "resp_1", "status": "completed", "model": "gpt-test",
		"output": [
			{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"nyc\"}"}
		]
	}`)
	resp, err := openairesponses.DecodeResponse(body)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if resp.FinishReason != stream.FinishReasonToolUse {
		t.Errorf("FinishReason = %v, want ToolUse", resp.FinishReason)
	}
	tu, ok := resp.Message.Blocks[0].(*content.ToolUseBlock)
	if !ok {
		t.Fatalf("Blocks[0] = %#v", resp.Message.Blocks[0])
	}
	if tu.ID != "call_1" || tu.Name != "get_weather" {
		t.Errorf("ToolUseBlock = %#v", tu)
	}
	if string(tu.Input) != `{"city":"nyc"}` {
		t.Errorf("Input = %s", tu.Input)
	}
}

func TestDecodeResponse_ParallelToolCalls(t *testing.T) {
	t.Parallel()
	body := []byte(`{
		"id": "resp_1", "status": "completed", "model": "gpt-test",
		"output": [
			{"type":"function_call","call_id":"call_1","name":"a","arguments":"{}"},
			{"type":"function_call","call_id":"call_2","name":"b","arguments":"{}"}
		]
	}`)
	resp, err := openairesponses.DecodeResponse(body)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if len(resp.Message.Blocks) != 2 {
		t.Fatalf("Blocks len = %d, want 2", len(resp.Message.Blocks))
	}
	first := resp.Message.Blocks[0].(*content.ToolUseBlock)
	second := resp.Message.Blocks[1].(*content.ToolUseBlock)
	if first.ID != "call_1" || second.ID != "call_2" {
		t.Errorf("ids = %q, %q", first.ID, second.ID)
	}
}

func TestDecodeResponse_ReasoningSummaryAndEncryptedContent(t *testing.T) {
	t.Parallel()
	body := []byte(`{
		"id": "resp_1", "status": "completed", "model": "gpt-test",
		"output": [
			{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"step 1"}],"encrypted_content":"opaque-abc"}
		]
	}`)
	resp, err := openairesponses.DecodeResponse(body)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	tb, ok := resp.Message.Blocks[0].(*content.ThinkingBlock)
	if !ok {
		t.Fatalf("Blocks[0] = %#v", resp.Message.Blocks[0])
	}
	if tb.Thinking != "step 1" {
		t.Errorf("Thinking = %q", tb.Thinking)
	}
	if tb.Signature != "" {
		t.Errorf("Signature = %q, want empty (Responses uses ProviderState, not Signature)", tb.Signature)
	}
	// ProviderState carries the whole replayable reasoning item, not just the
	// encrypted blob: the item's id is required to replay it and has nowhere
	// else to live. (This assertion previously required a bare JSON string.)
	var state struct {
		EncryptedContent string `json:"encrypted_content"`
	}
	if err := json.Unmarshal(tb.ProviderState, &state); err != nil {
		t.Fatalf("ProviderState = %s: %v", tb.ProviderState, err)
	}
	if state.EncryptedContent != "opaque-abc" {
		t.Errorf("ProviderState.encrypted_content = %q, want opaque-abc", state.EncryptedContent)
	}
}

func TestDecodeResponse_IncompleteMaxOutputTokens(t *testing.T) {
	t.Parallel()
	body := []byte(`{
		"id": "resp_1", "status": "incomplete", "model": "gpt-test",
		"output": [{"type":"message","role":"assistant","content":[{"type":"output_text","text":"partial"}]}],
		"incomplete_details": {"reason": "max_output_tokens"}
	}`)
	resp, err := openairesponses.DecodeResponse(body)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if resp.FinishReason != stream.FinishReasonLength {
		t.Errorf("FinishReason = %v, want Length", resp.FinishReason)
	}
}

func TestDecodeResponse_IncompleteOtherReason(t *testing.T) {
	t.Parallel()
	body := []byte(`{
		"id": "resp_1", "status": "incomplete", "model": "gpt-test", "output": [],
		"incomplete_details": {"reason": "content_filter"}
	}`)
	resp, err := openairesponses.DecodeResponse(body)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if resp.FinishReason != stream.FinishReasonUnknown {
		t.Errorf("FinishReason = %v, want Unknown", resp.FinishReason)
	}
}

func TestDecodeResponse_Failed(t *testing.T) {
	t.Parallel()
	body := []byte(`{
		"id": "resp_1", "status": "failed", "model": "gpt-test", "output": [],
		"error": {"code": "server_error", "message": "boom"}
	}`)
	_, err := openairesponses.DecodeResponse(body)
	if err == nil {
		t.Fatal("DecodeResponse() error = nil, want error for status:failed")
	}
	if err.Error() == "" {
		t.Error("error message empty")
	}
}

func TestDecodeResponse_ToolCallOverridesIncompleteStatus(t *testing.T) {
	t.Parallel()
	// A function_call in output always means ToolUse, even if status happens
	// to be "incomplete" for some other reason.
	body := []byte(`{
		"id": "resp_1", "status": "incomplete", "model": "gpt-test",
		"output": [{"type":"function_call","call_id":"c1","name":"f","arguments":"{}"}],
		"incomplete_details": {"reason": "max_output_tokens"}
	}`)
	resp, err := openairesponses.DecodeResponse(body)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if resp.FinishReason != stream.FinishReasonToolUse {
		t.Errorf("FinishReason = %v, want ToolUse", resp.FinishReason)
	}
}

func TestDecodeResponse_EmptyOutputIsNotAnError(t *testing.T) {
	t.Parallel()
	body := []byte(`{"id":"resp_1","status":"completed","model":"gpt-test","output":[]}`)
	resp, err := openairesponses.DecodeResponse(body)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if len(resp.Message.Blocks) != 0 {
		t.Errorf("Blocks = %#v, want empty", resp.Message.Blocks)
	}
}

// TestDecodeResponse_ReasoningItemIDIsPreserved locks the other half of the
// reasoning round trip: the response's own reasoning item id must survive into
// ThinkingBlock.ProviderState, because ReasoningItem.required includes "id"
// and a replayed item cannot supply one any other way. The neutral block has
// no id field, so the dialect's opaque provider-state payload carries it.
func TestDecodeResponse_ReasoningItemIDIsPreserved(t *testing.T) {
	t.Parallel()
	body := []byte(`{
		"id": "resp_1", "status": "completed", "model": "gpt-test",
		"output": [
			{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"step 1"}],"encrypted_content":"opaque-abc"}
		]
	}`)
	resp, err := openairesponses.DecodeResponse(body)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	tb, ok := resp.Message.Blocks[0].(*content.ThinkingBlock)
	if !ok {
		t.Fatalf("Blocks[0] = %#v", resp.Message.Blocks[0])
	}
	var state struct {
		ID               string `json:"id"`
		EncryptedContent string `json:"encrypted_content"`
	}
	if err := json.Unmarshal(tb.ProviderState, &state); err != nil {
		t.Fatalf("ProviderState = %s: %v", tb.ProviderState, err)
	}
	if state.ID != "rs_1" {
		t.Errorf("ProviderState.id = %q, want rs_1", state.ID)
	}
	if state.EncryptedContent != "opaque-abc" {
		t.Errorf("ProviderState.encrypted_content = %q, want opaque-abc", state.EncryptedContent)
	}
}

// TestDecodeResponse_ReasoningItemIDWithoutEncryptedContent covers a reasoning
// item returned without encrypted content (the default when
// reasoning.encrypted_content was not requested): the id alone is still worth
// preserving, since it is what makes the item replayable at all.
func TestDecodeResponse_ReasoningItemIDWithoutEncryptedContent(t *testing.T) {
	t.Parallel()
	body := []byte(`{
		"id": "resp_1", "status": "completed", "model": "gpt-test",
		"output": [{"type":"reasoning","id":"rs_1","summary":[]}]
	}`)
	resp, err := openairesponses.DecodeResponse(body)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	tb, ok := resp.Message.Blocks[0].(*content.ThinkingBlock)
	if !ok {
		t.Fatalf("Blocks[0] = %#v", resp.Message.Blocks[0])
	}
	if !tb.ReplayableAs("openai-responses") {
		t.Fatalf("ThinkingBlock not replayable: %#v", tb)
	}
}
