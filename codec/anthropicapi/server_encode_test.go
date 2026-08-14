package anthropicapi_test

import (
	"encoding/json"
	"net/http/httptest"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/anthropicapi"
	stream "github.com/looprig/inference/stream"
	usage "github.com/looprig/inference/usage"
)

// decodeWireResponse parses a WriteResponse body's fields we care about,
// treating it as an opaque map (semantic assertions on parsed fields, not raw
// string equality).
type wireResponse struct {
	ID         string                     `json:"id"`
	Type       string                     `json:"type"`
	Role       string                     `json:"role"`
	Model      string                     `json:"model"`
	StopReason string                     `json:"stop_reason"`
	Content    []map[string]any           `json:"content"`
	Usage      map[string]json.RawMessage `json:"usage"`
}

func writeResponseAndDecode(t *testing.T, resp *inference.Response) (*httptest.ResponseRecorder, wireResponse) {
	t.Helper()
	rec := httptest.NewRecorder()
	if err := (anthropicapi.Codec{}).WriteResponse(rec, resp); err != nil {
		t.Fatalf("WriteResponse() error = %v", err)
	}
	var wire wireResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &wire); err != nil {
		t.Fatalf("unmarshal response body: %v (%s)", err, rec.Body.String())
	}
	return rec, wire
}

func TestWriteResponse_BasicShape(t *testing.T) {
	t.Parallel()
	resp := &inference.Response{
		Message: &content.AIMessage{
			Message: content.Message{
				Role:   content.RoleAssistant,
				Blocks: []content.Block{&content.TextBlock{Text: "hello"}},
			},
		},
		Model:        "claude-test",
		FinishReason: stream.FinishReasonStop,
	}
	rec, wire := writeResponseAndDecode(t, resp)

	if rec.Code != 200 {
		t.Errorf("status = %d, want 200", rec.Code)
	}
	if wire.Type != "message" {
		t.Errorf("type = %q, want message", wire.Type)
	}
	if wire.Role != "assistant" {
		t.Errorf("role = %q, want assistant", wire.Role)
	}
	if wire.Model != "claude-test" {
		t.Errorf("model = %q, want claude-test", wire.Model)
	}
	if wire.ID == "" {
		t.Error("id is empty, want a synthesized id")
	}
	if wire.StopReason != "end_turn" {
		t.Errorf("stop_reason = %q, want end_turn", wire.StopReason)
	}
	if len(wire.Content) != 1 || wire.Content[0]["type"] != "text" || wire.Content[0]["text"] != "hello" {
		t.Errorf("content = %v", wire.Content)
	}
}

func TestWriteResponse_FinishReasons(t *testing.T) {
	t.Parallel()
	cases := []struct {
		reason stream.FinishReason
		want   string
	}{
		{reason: stream.FinishReasonStop, want: "end_turn"},
		{reason: stream.FinishReasonLength, want: "max_tokens"},
		{reason: stream.FinishReasonToolUse, want: "tool_use"},
		{reason: stream.FinishReasonContentFilter, want: "refusal"},
		{reason: stream.FinishReasonUnknown, want: "end_turn"},
	}
	for _, tc := range cases {
		t.Run(string(tc.reason), func(t *testing.T) {
			t.Parallel()
			_, wire := writeResponseAndDecode(t, &inference.Response{FinishReason: tc.reason})
			if wire.StopReason != tc.want {
				t.Errorf("stop_reason = %q, want %q", wire.StopReason, tc.want)
			}
		})
	}
}

func TestWriteResponse_Usage(t *testing.T) {
	t.Parallel()
	resp := &inference.Response{
		Usage: &usage.Usage{InputTokens: 10, OutputTokens: 20, CacheReadTokens: 3, CacheCreationTokens: 4},
	}
	_, wire := writeResponseAndDecode(t, resp)
	if wire.Usage == nil {
		t.Fatal("usage is nil, want non-nil")
	}
	var input, output int
	_ = json.Unmarshal(wire.Usage["input_tokens"], &input)
	_ = json.Unmarshal(wire.Usage["output_tokens"], &output)
	if input != 10 || output != 20 {
		t.Errorf("input=%d output=%d, want 10/20", input, output)
	}
}

func TestWriteResponse_ToolUseIDPreservedWhenPresent(t *testing.T) {
	t.Parallel()
	resp := &inference.Response{
		Message: &content.AIMessage{
			Message: content.Message{
				Role: content.RoleAssistant,
				Blocks: []content.Block{
					&content.ToolUseBlock{ID: "toolu_given", Name: "f", Input: json.RawMessage(`{}`)},
				},
			},
		},
	}
	_, wire := writeResponseAndDecode(t, resp)
	if wire.Content[0]["id"] != "toolu_given" {
		t.Errorf("id = %v, want toolu_given", wire.Content[0]["id"])
	}
}

// TestWriteResponse_ToolUseIDSynthesizedWhenEmpty pins the defensive id
// synthesis path: a cross-dialect target might hand back a tool call with no
// id, and this codec must never forward an empty native id.
func TestWriteResponse_ToolUseIDSynthesizedWhenEmpty(t *testing.T) {
	t.Parallel()
	resp := &inference.Response{
		Message: &content.AIMessage{
			Message: content.Message{
				Role: content.RoleAssistant,
				Blocks: []content.Block{
					&content.ToolUseBlock{ID: "", Name: "f1", Input: json.RawMessage(`{}`)},
					&content.ToolUseBlock{ID: "", Name: "f2", Input: json.RawMessage(`{}`)},
				},
			},
		},
	}
	_, wire := writeResponseAndDecode(t, resp)
	id1, _ := wire.Content[0]["id"].(string)
	id2, _ := wire.Content[1]["id"].(string)
	if id1 == "" || id2 == "" {
		t.Fatalf("synthesized ids are empty: %q, %q", id1, id2)
	}
	if id1 == id2 {
		t.Errorf("synthesized ids collide: both %q", id1)
	}
}

func TestWriteResponse_SignedThinkingRoundTrip(t *testing.T) {
	t.Parallel()
	const sig = "sig-roundtrip-xyz"
	resp := &inference.Response{
		Message: &content.AIMessage{
			Message: content.Message{
				Role:   content.RoleAssistant,
				Blocks: []content.Block{content.NewSignedThinkingBlock("hmm", sig, signatureFormatAnthropic, nil, "")},
			},
		},
	}
	_, wire := writeResponseAndDecode(t, resp)
	if wire.Content[0]["type"] != "thinking" {
		t.Fatalf("content[0].type = %v, want thinking", wire.Content[0]["type"])
	}
	if wire.Content[0]["signature"] != sig {
		t.Errorf("signature = %v, want %q", wire.Content[0]["signature"], sig)
	}
}

// --- WriteError ---------------------------------------------------------

func TestWriteError_NativeEnvelope(t *testing.T) {
	t.Parallel()
	rec := httptest.NewRecorder()
	(anthropicapi.Codec{}).WriteError(rec, &anthropicapi.ServerDecodeError{Reason: "missing_model"})

	if rec.Code != 400 {
		t.Errorf("status = %d, want 400", rec.Code)
	}
	var envelope struct {
		Type  string `json:"type"`
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("unmarshal error body: %v (%s)", err, rec.Body.String())
	}
	if envelope.Type != "error" {
		t.Errorf("type = %q, want error", envelope.Type)
	}
	if envelope.Error.Type != "invalid_request_error" {
		t.Errorf("error.type = %q, want invalid_request_error", envelope.Error.Type)
	}
	if envelope.Error.Message == "" {
		t.Error("error.message is empty")
	}
}

func TestWriteError_NeverPanicsOnNil(t *testing.T) {
	t.Parallel()
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("WriteError panicked on nil error: %v", r)
		}
	}()
	rec := httptest.NewRecorder()
	(anthropicapi.Codec{}).WriteError(rec, nil)
	if rec.Code < 400 || rec.Code >= 600 {
		t.Errorf("status = %d, want 4xx/5xx", rec.Code)
	}
}

// --- count_tokens response ------------------------------------------------

func TestWriteCountTokensResponse(t *testing.T) {
	t.Parallel()
	rec := httptest.NewRecorder()
	if err := anthropicapi.WriteCountTokensResponse(rec, 42); err != nil {
		t.Fatalf("WriteCountTokensResponse() error = %v", err)
	}
	if rec.Code != 200 {
		t.Errorf("status = %d, want 200", rec.Code)
	}
	var body struct {
		InputTokens int `json:"input_tokens"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if body.InputTokens != 42 {
		t.Errorf("input_tokens = %d, want 42", body.InputTokens)
	}
}
