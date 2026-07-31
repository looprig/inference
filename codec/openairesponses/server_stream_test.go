package openairesponses_test

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference/codec/openairesponses"
	stream "github.com/looprig/inference/stream"
)

// sseEventTypes extracts the ordered sequence of `event: <name>` lines from a
// raw SSE response body.
func sseEventTypes(t *testing.T, body string) []string {
	t.Helper()
	var types []string
	for _, line := range strings.Split(body, "\n") {
		if strings.HasPrefix(line, "event: ") {
			types = append(types, strings.TrimPrefix(line, "event: "))
		}
	}
	return types
}

func TestServerStream_NativeEventOrderForText(t *testing.T) {
	t.Parallel()
	c := openairesponses.Codec{}
	rec := httptest.NewRecorder()
	enc, err := c.OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}
	if err := enc.WriteChunk(&content.TextChunk{Text: "hello "}); err != nil {
		t.Fatalf("WriteChunk() error = %v", err)
	}
	if err := enc.WriteChunk(&content.TextChunk{Text: "world"}); err != nil {
		t.Fatalf("WriteChunk() error = %v", err)
	}
	if err := enc.Finish(stream.StreamResult{Model: "gpt-test", FinishReason: stream.FinishReasonStop}); err != nil {
		t.Fatalf("Finish() error = %v", err)
	}

	got := sseEventTypes(t, rec.Body.String())
	want := []string{
		"response.created",
		"response.output_item.added",
		"response.content_part.added",
		"response.output_text.delta",
		"response.output_text.delta",
		"response.content_part.done",
		"response.output_item.done",
		"response.completed",
	}
	if len(got) != len(want) {
		t.Fatalf("event sequence = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("event[%d] = %q, want %q (full: %v)", i, got[i], want[i], got)
		}
	}
}

func TestServerStream_ToolCallEvents(t *testing.T) {
	t.Parallel()
	c := openairesponses.Codec{}
	rec := httptest.NewRecorder()
	enc, err := c.OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}
	if err := enc.WriteChunk(&content.ToolUseChunk{Index: 0, ID: "call_1", Name: "get_weather"}); err != nil {
		t.Fatalf("WriteChunk() error = %v", err)
	}
	if err := enc.WriteChunk(&content.ToolUseChunk{Index: 0, InputJSON: `{"city":`}); err != nil {
		t.Fatalf("WriteChunk() error = %v", err)
	}
	if err := enc.WriteChunk(&content.ToolUseChunk{Index: 0, InputJSON: `"nyc"}`}); err != nil {
		t.Fatalf("WriteChunk() error = %v", err)
	}
	if err := enc.Finish(stream.StreamResult{FinishReason: stream.FinishReasonToolUse}); err != nil {
		t.Fatalf("Finish() error = %v", err)
	}

	got := sseEventTypes(t, rec.Body.String())
	want := []string{
		"response.created",
		"response.output_item.added",
		"response.function_call_arguments.delta",
		"response.function_call_arguments.delta",
		"response.function_call_arguments.done",
		"response.output_item.done",
		"response.completed",
	}
	if len(got) != len(want) {
		t.Fatalf("event sequence = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("event[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestServerStream_ParallelToolCallIndexes(t *testing.T) {
	t.Parallel()
	c := openairesponses.Codec{}
	rec := httptest.NewRecorder()
	enc, err := c.OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}
	// Each tool call's deltas arrive together (not interleaved): the wire
	// protocol only ever streams one open item's content at a time, so a
	// genuinely interleaved source is intentionally re-serialized as
	// repeated add/done pairs (see ensureItem's doc comment) — this test
	// exercises the ordinary non-interleaved case.
	if err := enc.WriteChunk(&content.ToolUseChunk{Index: 0, ID: "call_1", Name: "a"}); err != nil {
		t.Fatal(err)
	}
	if err := enc.WriteChunk(&content.ToolUseChunk{Index: 0, InputJSON: `{}`}); err != nil {
		t.Fatal(err)
	}
	if err := enc.WriteChunk(&content.ToolUseChunk{Index: 1, ID: "call_2", Name: "b"}); err != nil {
		t.Fatal(err)
	}
	if err := enc.WriteChunk(&content.ToolUseChunk{Index: 1, InputJSON: `{}`}); err != nil {
		t.Fatal(err)
	}
	if err := enc.Finish(stream.StreamResult{FinishReason: stream.FinishReasonToolUse}); err != nil {
		t.Fatalf("Finish() error = %v", err)
	}

	// Extract output_index from every response.output_item.added event.
	var addedIndexes []int
	for _, line := range strings.Split(rec.Body.String(), "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var ev struct {
			Type        string `json:"type"`
			OutputIndex int    `json:"output_index"`
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &ev); err != nil {
			continue
		}
		if ev.Type == "response.output_item.added" {
			addedIndexes = append(addedIndexes, ev.OutputIndex)
		}
	}
	if len(addedIndexes) != 2 || addedIndexes[0] != 0 || addedIndexes[1] != 1 {
		t.Errorf("added output_index sequence = %v, want [0 1]", addedIndexes)
	}
}

func TestServerStream_ReasoningEvents(t *testing.T) {
	t.Parallel()
	c := openairesponses.Codec{}
	rec := httptest.NewRecorder()
	enc, err := c.OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}
	if err := enc.WriteChunk(&content.ThinkingChunk{Thinking: "step "}); err != nil {
		t.Fatal(err)
	}
	if err := enc.WriteChunk(&content.ThinkingChunk{Thinking: "one"}); err != nil {
		t.Fatal(err)
	}
	if err := enc.Finish(stream.StreamResult{FinishReason: stream.FinishReasonStop}); err != nil {
		t.Fatalf("Finish() error = %v", err)
	}
	got := sseEventTypes(t, rec.Body.String())
	want := []string{
		"response.created",
		"response.output_item.added",
		"response.reasoning_summary_text.delta",
		"response.reasoning_summary_text.delta",
		"response.reasoning_summary_text.done",
		"response.output_item.done",
		"response.completed",
	}
	if len(got) != len(want) {
		t.Fatalf("event sequence = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("event[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestServerStream_FailAfterHeaderCommitment(t *testing.T) {
	t.Parallel()
	c := openairesponses.Codec{}
	rec := httptest.NewRecorder()
	enc, err := c.OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}
	if err := enc.WriteChunk(&content.TextChunk{Text: "partial"}); err != nil {
		t.Fatal(err)
	}
	if err := enc.Fail(&openairesponses.ServerDecodeError{Reason: "upstream_error"}); err != nil {
		t.Fatalf("Fail() error = %v", err)
	}
	got := sseEventTypes(t, rec.Body.String())
	if len(got) == 0 || got[len(got)-1] != "response.failed" {
		t.Fatalf("event sequence = %v, want to end with response.failed", got)
	}
	// Fail must not attempt to close the still-open item cleanly first.
	for _, ev := range got {
		if ev == "response.content_part.done" || ev == "response.output_item.done" {
			t.Errorf("Fail should not emit a clean-close event %q", ev)
		}
	}
}

func TestServerStream_NativeErrorBeforeHeaderCommitment(t *testing.T) {
	t.Parallel()
	c := openairesponses.Codec{}
	rec := httptest.NewRecorder()
	c.WriteError(rec, &openairesponses.ServerDecodeError{Reason: "malformed_body"})
	if rec.Code < 400 || rec.Code >= 500 {
		t.Errorf("status = %d, want 4xx", rec.Code)
	}
	var m map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &m); err != nil {
		t.Fatalf("WriteError body is not JSON: %v (body=%s)", err, rec.Body.String())
	}
	if _, ok := m["error"]; !ok {
		t.Errorf("body missing error field: %#v", m)
	}
}
