package geminiapi_test

import (
	"bufio"
	"bytes"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference/codec/geminiapi"
	stream "github.com/looprig/inference/stream"
	usage "github.com/looprig/inference/usage"
)

// dataFrames extracts the JSON payloads of every "data: " SSE line in body.
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

func TestServerStream_TextChunks(t *testing.T) {
	t.Parallel()
	c := geminiapi.Codec{}
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
	if err := enc.Finish(stream.StreamResult{Model: "gemini-test", FinishReason: stream.FinishReasonStop}); err != nil {
		t.Fatalf("Finish: %v", err)
	}

	frames := dataFrames(t, rec.Body.Bytes())
	// No [DONE]-style sentinel: exactly 2 deltas + 1 terminal candidate frame.
	if len(frames) != 3 {
		t.Fatalf("frames = %d, want 3 (no terminal sentinel for Gemini)", len(frames))
	}
	for _, f := range frames {
		if f == "[DONE]" {
			t.Fatalf("frame = [DONE], Gemini must never emit a terminal sentinel")
		}
	}

	var f0 map[string]any
	json.Unmarshal([]byte(frames[0]), &f0)
	parts := f0["candidates"].([]any)[0].(map[string]any)["content"].(map[string]any)["parts"].([]any)
	if parts[0].(map[string]any)["text"] != "hel" {
		t.Errorf("frame0 text = %v", parts[0])
	}

	var f2 map[string]any
	json.Unmarshal([]byte(frames[2]), &f2)
	cand := f2["candidates"].([]any)[0].(map[string]any)
	if cand["finishReason"] != "STOP" {
		t.Errorf("finishReason = %v", cand["finishReason"])
	}
	if f2["modelVersion"] != "gemini-test" {
		t.Errorf("modelVersion = %v", f2["modelVersion"])
	}
}

func TestServerStream_ThinkingChunk(t *testing.T) {
	t.Parallel()
	c := geminiapi.Codec{}
	rec := httptest.NewRecorder()
	enc, _ := c.OpenStream(rec)
	if err := enc.WriteChunk(&content.ThinkingChunk{Thinking: "planning"}); err != nil {
		t.Fatalf("WriteChunk: %v", err)
	}
	_ = enc.Finish(stream.StreamResult{})
	frames := dataFrames(t, rec.Body.Bytes())
	var f0 map[string]any
	json.Unmarshal([]byte(frames[0]), &f0)
	part := f0["candidates"].([]any)[0].(map[string]any)["content"].(map[string]any)["parts"].([]any)[0].(map[string]any)
	if part["thought"] != true || part["text"] != "planning" {
		t.Errorf("part = %v", part)
	}
}

func TestServerStream_ToolUseChunkWritesCompleteArgsPerChunk(t *testing.T) {
	t.Parallel()
	c := geminiapi.Codec{}
	rec := httptest.NewRecorder()
	enc, _ := c.OpenStream(rec)
	if err := enc.WriteChunk(&content.ToolUseChunk{Index: 0, ID: "call_1", Name: "get_weather", InputJSON: `{"city":"nyc"}`}); err != nil {
		t.Fatalf("WriteChunk: %v", err)
	}
	_ = enc.Finish(stream.StreamResult{})
	frames := dataFrames(t, rec.Body.Bytes())
	var f0 map[string]any
	json.Unmarshal([]byte(frames[0]), &f0)
	part := f0["candidates"].([]any)[0].(map[string]any)["content"].(map[string]any)["parts"].([]any)[0].(map[string]any)
	fc := part["functionCall"].(map[string]any)
	if fc["name"] != "get_weather" {
		t.Errorf("name = %v", fc["name"])
	}
	if fc["args"].(map[string]any)["city"] != "nyc" {
		t.Errorf("args = %v", fc["args"])
	}
}

func TestServerStream_FinishCarriesUsage(t *testing.T) {
	t.Parallel()
	c := geminiapi.Codec{}
	rec := httptest.NewRecorder()
	enc, _ := c.OpenStream(rec)
	_ = enc.WriteChunk(&content.TextChunk{Text: "hi"})
	result := stream.StreamResult{
		Model:        "gemini-test",
		FinishReason: stream.FinishReasonStop,
		Usage:        &usage.Usage{InputTokens: 3, OutputTokens: 2},
	}
	if err := enc.Finish(result); err != nil {
		t.Fatalf("Finish: %v", err)
	}
	frames := dataFrames(t, rec.Body.Bytes())
	var last map[string]any
	json.Unmarshal([]byte(frames[len(frames)-1]), &last)
	usageWire := last["usageMetadata"].(map[string]any)
	if usageWire["promptTokenCount"].(float64) != 3 || usageWire["candidatesTokenCount"].(float64) != 2 {
		t.Errorf("usage = %v", usageWire)
	}
}

func TestServerStream_FailWritesTerminalErrorRecordNoSentinel(t *testing.T) {
	t.Parallel()
	c := geminiapi.Codec{}
	rec := httptest.NewRecorder()
	enc, err := c.OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}
	if err := enc.Fail(&geminiapi.ServerDecodeError{Reason: "boom"}); err != nil {
		t.Fatalf("Fail: %v", err)
	}
	frames := dataFrames(t, rec.Body.Bytes())
	if len(frames) != 1 {
		t.Fatalf("frames = %d, want 1 (one terminal error record, no sentinel)", len(frames))
	}
	var errFrame map[string]any
	if err := json.Unmarshal([]byte(frames[0]), &errFrame); err != nil {
		t.Fatalf("error frame not JSON: %v", err)
	}
	if _, ok := errFrame["error"]; !ok {
		t.Fatalf("frame = %v, want top-level error key", errFrame)
	}
}

func TestServerStream_UnsupportedChunkType(t *testing.T) {
	t.Parallel()
	c := geminiapi.Codec{}
	rec := httptest.NewRecorder()
	enc, _ := c.OpenStream(rec)
	if err := enc.WriteChunk(nil); err == nil {
		t.Fatal("WriteChunk(nil) error = nil, want *UnsupportedChunkError")
	}
}

func TestServerStream_SingleTermination(t *testing.T) {
	t.Parallel()
	c := geminiapi.Codec{}
	rec := httptest.NewRecorder()
	enc, _ := c.OpenStream(rec)
	if err := enc.Finish(stream.StreamResult{}); err != nil {
		t.Fatalf("first Finish: %v", err)
	}
	if err := enc.Finish(stream.StreamResult{}); err == nil {
		t.Fatal("second Finish: error = nil, want rejection")
	}
	if err := enc.Fail(&geminiapi.ServerDecodeError{}); err == nil {
		t.Fatal("Fail after Finish: error = nil, want rejection")
	}
	if err := enc.WriteChunk(&content.TextChunk{Text: "x"}); err == nil {
		t.Fatal("WriteChunk after Finish: error = nil, want rejection")
	}
}
