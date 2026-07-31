package openaiapi_test

import (
	"bufio"
	"bytes"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference/codec/openaiapi"
	stream "github.com/looprig/inference/stream"
	usage "github.com/looprig/inference/usage"
)

// dataFrames extracts the JSON payloads of every "data: " SSE line in body,
// as raw strings (including the literal "[DONE]" sentinel line).
func dataFrames(t *testing.T, body []byte) []string {
	t.Helper()
	var out []string
	sc := bufio.NewScanner(bytes.NewReader(body))
	sc.Buffer(make([]byte, 64*1024), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "data: ") {
			out = append(out, strings.TrimPrefix(line, "data: "))
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scanning SSE body: %v", err)
	}
	return out
}

func TestServerStream_TextDeltasCarryRoleOnce(t *testing.T) {
	t.Parallel()
	c := openaiapi.Codec{}
	rec := httptest.NewRecorder()
	enc, err := c.OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}
	if err := enc.WriteChunk(&content.TextChunk{Text: "hel"}); err != nil {
		t.Fatalf("WriteChunk: %v", err)
	}
	if err := enc.WriteChunk(&content.TextChunk{Text: "lo"}); err != nil {
		t.Fatalf("WriteChunk: %v", err)
	}
	if err := enc.Finish(stream.StreamResult{Model: "gpt-test", FinishReason: stream.FinishReasonStop}); err != nil {
		t.Fatalf("Finish: %v", err)
	}

	frames := dataFrames(t, rec.Body.Bytes())
	if len(frames) != 4 {
		t.Fatalf("frames = %d, want 4 (2 deltas + finish + [DONE], no usage chunk since Usage nil)", len(frames))
	}
	if frames[3] != "[DONE]" {
		t.Fatalf("last frame = %q, want [DONE]", frames[3])
	}

	var first map[string]any
	if err := json.Unmarshal([]byte(frames[0]), &first); err != nil {
		t.Fatalf("frame[0] not JSON: %v", err)
	}
	delta0 := first["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)
	if delta0["role"] != "assistant" {
		t.Errorf("delta0 role = %v, want assistant", delta0["role"])
	}
	if delta0["content"] != "hel" {
		t.Errorf("delta0 content = %v", delta0["content"])
	}

	var second map[string]any
	if err := json.Unmarshal([]byte(frames[1]), &second); err != nil {
		t.Fatalf("frame[1] not JSON: %v", err)
	}
	delta1 := second["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)
	if _, ok := delta1["role"]; ok {
		t.Errorf("delta1 carries role again: %v, want role omitted after first chunk", delta1["role"])
	}
	if delta1["content"] != "lo" {
		t.Errorf("delta1 content = %v", delta1["content"])
	}
}

func TestServerStream_ToolCallDeltasCarryIDNameOnlyOnce(t *testing.T) {
	t.Parallel()
	c := openaiapi.Codec{}
	rec := httptest.NewRecorder()
	enc, err := c.OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}
	if err := enc.WriteChunk(&content.ToolUseChunk{Index: 0, ID: "call_1", Name: "get_weather"}); err != nil {
		t.Fatalf("WriteChunk: %v", err)
	}
	if err := enc.WriteChunk(&content.ToolUseChunk{Index: 0, InputJSON: `{"city":`}); err != nil {
		t.Fatalf("WriteChunk: %v", err)
	}
	if err := enc.WriteChunk(&content.ToolUseChunk{Index: 0, InputJSON: `"nyc"}`}); err != nil {
		t.Fatalf("WriteChunk: %v", err)
	}
	if err := enc.Finish(stream.StreamResult{FinishReason: stream.FinishReasonToolUse}); err != nil {
		t.Fatalf("Finish: %v", err)
	}

	frames := dataFrames(t, rec.Body.Bytes())
	if len(frames) != 5 {
		t.Fatalf("frames = %d, want 5 (3 deltas + finish + [DONE])", len(frames))
	}

	var f0 map[string]any
	json.Unmarshal([]byte(frames[0]), &f0)
	tc0 := f0["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)["tool_calls"].([]any)[0].(map[string]any)
	if tc0["id"] != "call_1" || tc0["function"].(map[string]any)["name"] != "get_weather" {
		t.Errorf("tc0 = %v", tc0)
	}
	if int(tc0["index"].(float64)) != 0 {
		t.Errorf("tc0 index = %v", tc0["index"])
	}

	var f1 map[string]any
	json.Unmarshal([]byte(frames[1]), &f1)
	tc1 := f1["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)["tool_calls"].([]any)[0].(map[string]any)
	if _, hasID := tc1["id"]; hasID {
		t.Errorf("tc1 carries id again: %v, want omitted on later fragments", tc1["id"])
	}
	if tc1["function"].(map[string]any)["arguments"] != `{"city":` {
		t.Errorf("tc1 arguments = %v", tc1["function"])
	}
}

func TestServerStream_ParallelToolCallIndexes(t *testing.T) {
	t.Parallel()
	c := openaiapi.Codec{}
	rec := httptest.NewRecorder()
	enc, err := c.OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}
	if err := enc.WriteChunk(&content.ToolUseChunk{Index: 0, ID: "call_1", Name: "a"}); err != nil {
		t.Fatal(err)
	}
	if err := enc.WriteChunk(&content.ToolUseChunk{Index: 1, ID: "call_2", Name: "b"}); err != nil {
		t.Fatal(err)
	}
	if err := enc.Finish(stream.StreamResult{}); err != nil {
		t.Fatal(err)
	}
	frames := dataFrames(t, rec.Body.Bytes())
	var f0, f1 map[string]any
	json.Unmarshal([]byte(frames[0]), &f0)
	json.Unmarshal([]byte(frames[1]), &f1)
	idx0 := f0["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)["tool_calls"].([]any)[0].(map[string]any)["index"]
	idx1 := f1["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)["tool_calls"].([]any)[0].(map[string]any)["index"]
	if idx0 != float64(0) || idx1 != float64(1) {
		t.Errorf("indexes = %v, %v, want 0, 1", idx0, idx1)
	}
}

func TestServerStream_FinishEmitsUsageChunkWithEmptyChoicesThenDone(t *testing.T) {
	t.Parallel()
	c := openaiapi.Codec{}
	rec := httptest.NewRecorder()
	enc, err := c.OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}
	if err := enc.WriteChunk(&content.TextChunk{Text: "hi"}); err != nil {
		t.Fatal(err)
	}
	result := stream.StreamResult{
		Model:        "gpt-test",
		FinishReason: stream.FinishReasonStop,
		Usage:        &usage.Usage{InputTokens: 3, OutputTokens: 2},
	}
	if err := enc.Finish(result); err != nil {
		t.Fatalf("Finish: %v", err)
	}

	frames := dataFrames(t, rec.Body.Bytes())
	if len(frames) != 4 {
		t.Fatalf("frames = %d, want 4 (delta, finish, usage, [DONE])", len(frames))
	}
	if frames[3] != "[DONE]" {
		t.Fatalf("last frame = %q, want [DONE]", frames[3])
	}

	var finishFrame map[string]any
	json.Unmarshal([]byte(frames[1]), &finishFrame)
	if finishFrame["choices"].([]any)[0].(map[string]any)["finish_reason"] != "stop" {
		t.Errorf("finish frame finish_reason = %v", finishFrame["choices"])
	}

	var usageFrame map[string]any
	json.Unmarshal([]byte(frames[2]), &usageFrame)
	choices, ok := usageFrame["choices"].([]any)
	if !ok || len(choices) != 0 {
		t.Errorf("usage frame choices = %v, want empty array", usageFrame["choices"])
	}
	usageWire := usageFrame["usage"].(map[string]any)
	if usageWire["prompt_tokens"].(float64) != 3 || usageWire["completion_tokens"].(float64) != 2 {
		t.Errorf("usage = %v", usageWire)
	}
}

func TestServerStream_NoUsageChunkWhenResultHasNoUsage(t *testing.T) {
	t.Parallel()
	c := openaiapi.Codec{}
	rec := httptest.NewRecorder()
	enc, _ := c.OpenStream(rec)
	_ = enc.WriteChunk(&content.TextChunk{Text: "hi"})
	if err := enc.Finish(stream.StreamResult{FinishReason: stream.FinishReasonStop}); err != nil {
		t.Fatalf("Finish: %v", err)
	}
	frames := dataFrames(t, rec.Body.Bytes())
	if len(frames) != 3 {
		t.Fatalf("frames = %d, want 3 (delta, finish, [DONE]) with no usage chunk", len(frames))
	}
}

func TestServerStream_FailWritesBoundedGatewayEventThenDone(t *testing.T) {
	t.Parallel()
	c := openaiapi.Codec{}
	rec := httptest.NewRecorder()
	enc, err := c.OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}
	if err := enc.Fail(&openaiapi.ServerDecodeError{Reason: "boom"}); err != nil {
		t.Fatalf("Fail: %v", err)
	}
	frames := dataFrames(t, rec.Body.Bytes())
	if len(frames) != 2 {
		t.Fatalf("frames = %d, want 2 (gateway error + [DONE])", len(frames))
	}
	if frames[1] != "[DONE]" {
		t.Fatalf("last frame = %q, want [DONE]", frames[1])
	}
	var errFrame map[string]any
	if err := json.Unmarshal([]byte(frames[0]), &errFrame); err != nil {
		t.Fatalf("error frame not JSON: %v", err)
	}
	errObj, ok := errFrame["error"].(map[string]any)
	if !ok {
		t.Fatalf("frame = %v, want top-level error key", errFrame)
	}
	if errObj["type"] != "gateway_error" {
		t.Errorf("type = %v, want gateway_error", errObj["type"])
	}
}

func TestServerStream_UnsupportedChunkType(t *testing.T) {
	t.Parallel()
	c := openaiapi.Codec{}
	rec := httptest.NewRecorder()
	enc, _ := c.OpenStream(rec)
	if err := enc.WriteChunk(nil); err == nil {
		t.Fatal("WriteChunk(nil) error = nil, want *UnsupportedChunkError")
	}
}
