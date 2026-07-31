package anthropicapi_test

import (
	"encoding/json"
	"errors"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference/codec/anthropicapi"
	stream "github.com/looprig/inference/stream"
	usage "github.com/looprig/inference/usage"
)

// sseEvent is one de-framed "event: <name>\ndata: <json>\n\n" SSE frame.
type sseEvent struct {
	Name string
	Data map[string]any
}

// parseSSEEvents splits a full SSE body into its ordered events. It is a
// minimal test-only parser (real framing is wire/sse's job on the read side);
// it assumes each event was written as exactly "event: name\ndata: json\n\n",
// which is what server_stream.go's writeEvent produces.
func parseSSEEvents(t *testing.T, body string) []sseEvent {
	t.Helper()
	var events []sseEvent
	for _, raw := range strings.Split(strings.TrimSpace(body), "\n\n") {
		raw = strings.TrimSpace(raw)
		if raw == "" {
			continue
		}
		lines := strings.SplitN(raw, "\n", 2)
		if len(lines) != 2 {
			t.Fatalf("malformed SSE frame: %q", raw)
		}
		name := strings.TrimPrefix(lines[0], "event: ")
		dataLine := strings.TrimPrefix(lines[1], "data: ")
		var data map[string]any
		if err := json.Unmarshal([]byte(dataLine), &data); err != nil {
			t.Fatalf("unmarshal event data: %v (%s)", err, dataLine)
		}
		events = append(events, sseEvent{Name: name, Data: data})
	}
	return events
}

func eventNames(events []sseEvent) []string {
	names := make([]string, len(events))
	for i, e := range events {
		names[i] = e.Name
	}
	return names
}

// TestStream_NativeEventOrder pins the exact Anthropic-native SSE event order
// for a simple single-text-block stream: message_start, content_block_start,
// content_block_delta..., content_block_stop, message_delta, message_stop.
func TestStream_NativeEventOrder(t *testing.T) {
	t.Parallel()
	rec := httptest.NewRecorder()
	enc, err := (anthropicapi.Codec{}).OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}

	if err := enc.WriteChunk(&content.TextChunk{Text: "hel"}); err != nil {
		t.Fatalf("WriteChunk() error = %v", err)
	}
	if err := enc.WriteChunk(&content.TextChunk{Text: "lo"}); err != nil {
		t.Fatalf("WriteChunk() error = %v", err)
	}
	if err := enc.Finish(stream.StreamResult{FinishReason: stream.FinishReasonStop}); err != nil {
		t.Fatalf("Finish() error = %v", err)
	}

	events := parseSSEEvents(t, rec.Body.String())
	got := eventNames(events)
	want := []string{
		"message_start",
		"content_block_start",
		"content_block_delta",
		"content_block_delta",
		"content_block_stop",
		"message_delta",
		"message_stop",
	}
	if len(got) != len(want) {
		t.Fatalf("event names = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("event[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

// TestStream_FlushesAfterEachEvent asserts progressive delivery at a finer
// grain than the shared servertest suite: the response body must grow after
// EVERY WriteChunk call that produces a wire event, not merely be non-empty
// before Finish.
func TestStream_FlushesAfterEachEvent(t *testing.T) {
	t.Parallel()
	rec := httptest.NewRecorder()
	enc, err := (anthropicapi.Codec{}).OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}

	prevLen := rec.Body.Len() // message_start already written by OpenStream
	if prevLen == 0 {
		t.Fatal("body is empty immediately after OpenStream, want message_start already flushed")
	}

	chunks := []content.Chunk{
		&content.TextChunk{Text: "a"},
		&content.TextChunk{Text: "b"},
		&content.ThinkingChunk{Thinking: "reasoning"}, // switches block kind
		&content.ToolUseChunk{Index: 0, ID: "toolu_1", Name: "f"},
		&content.ToolUseChunk{Index: 0, InputJSON: `{"x":1}`},
	}
	for i, c := range chunks {
		if err := enc.WriteChunk(c); err != nil {
			t.Fatalf("WriteChunk[%d]: %v", i, err)
		}
		newLen := rec.Body.Len()
		if newLen <= prevLen {
			t.Errorf("WriteChunk[%d]: body length did not grow (%d -> %d)", i, prevLen, newLen)
		}
		prevLen = newLen
	}
}

// TestStream_ParallelToolCalls exercises two tool calls streamed with their own
// Index, pinning that each gets its own content_block_start/stop pair with the
// right id/name, in the order they were written.
func TestStream_ParallelToolCalls(t *testing.T) {
	t.Parallel()
	rec := httptest.NewRecorder()
	enc, err := (anthropicapi.Codec{}).OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}

	must := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatalf("WriteChunk: %v", err)
		}
	}
	must(enc.WriteChunk(&content.ToolUseChunk{Index: 0, ID: "toolu_1", Name: "get_weather"}))
	must(enc.WriteChunk(&content.ToolUseChunk{Index: 0, InputJSON: `{"city":`}))
	must(enc.WriteChunk(&content.ToolUseChunk{Index: 0, InputJSON: `"nyc"}`}))
	must(enc.WriteChunk(&content.ToolUseChunk{Index: 1, ID: "toolu_2", Name: "get_time"}))
	must(enc.WriteChunk(&content.ToolUseChunk{Index: 1, InputJSON: `{"tz":"utc"}`}))
	if err := enc.Finish(stream.StreamResult{FinishReason: stream.FinishReasonToolUse}); err != nil {
		t.Fatalf("Finish() error = %v", err)
	}

	events := parseSSEEvents(t, rec.Body.String())
	var starts []map[string]any
	var deltas []string
	for _, e := range events {
		switch e.Name {
		case "content_block_start":
			cb, _ := e.Data["content_block"].(map[string]any)
			starts = append(starts, cb)
		case "content_block_delta":
			delta, _ := e.Data["delta"].(map[string]any)
			if pj, ok := delta["partial_json"].(string); ok {
				deltas = append(deltas, pj)
			}
		}
	}
	if len(starts) != 2 {
		t.Fatalf("content_block_start count = %d, want 2", len(starts))
	}
	if starts[0]["id"] != "toolu_1" || starts[0]["name"] != "get_weather" {
		t.Errorf("starts[0] = %v", starts[0])
	}
	if starts[1]["id"] != "toolu_2" || starts[1]["name"] != "get_time" {
		t.Errorf("starts[1] = %v", starts[1])
	}
	if strings.Join(deltas, "") != `{"city":"nyc"}{"tz":"utc"}` {
		t.Errorf("deltas = %v", deltas)
	}

	// The stop_reason from Finish's FinishReasonToolUse must survive to
	// message_delta.
	var stopReason string
	for _, e := range events {
		if e.Name == "message_delta" {
			delta, _ := e.Data["delta"].(map[string]any)
			stopReason, _ = delta["stop_reason"].(string)
		}
	}
	if stopReason != "tool_use" {
		t.Errorf("message_delta.delta.stop_reason = %q, want tool_use", stopReason)
	}
}

func TestStream_UsageOnMessageDelta(t *testing.T) {
	t.Parallel()
	rec := httptest.NewRecorder()
	enc, err := (anthropicapi.Codec{}).OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}
	if err := enc.WriteChunk(&content.TextChunk{Text: "hi"}); err != nil {
		t.Fatalf("WriteChunk: %v", err)
	}
	result := stream.StreamResult{
		FinishReason: stream.FinishReasonStop,
		Usage:        &usage.Usage{InputTokens: 5, OutputTokens: 7},
	}
	if err := enc.Finish(result); err != nil {
		t.Fatalf("Finish() error = %v", err)
	}

	events := parseSSEEvents(t, rec.Body.String())
	for _, e := range events {
		if e.Name == "message_delta" {
			u, _ := e.Data["usage"].(map[string]any)
			outputTokens, _ := u["output_tokens"].(float64)
			if int(outputTokens) != 7 {
				t.Errorf("usage.output_tokens = %v, want 7", u["output_tokens"])
			}
			return
		}
	}
	t.Fatal("no message_delta event found")
}

// TestStream_SignedThinkingIsNotStreamed documents that ThinkingBlock.Signature
// is a complete-block-only field (content.ThinkingChunk has no Signature field
// per core/content/chunk.go), so signature round-trip is only meaningful on the
// non-streaming WriteResponse path (see server_encode_test.go), not here.
func TestStream_SignedThinkingIsNotStreamed(t *testing.T) {
	t.Parallel()
	rec := httptest.NewRecorder()
	enc, err := (anthropicapi.Codec{}).OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}
	if err := enc.WriteChunk(&content.ThinkingChunk{Thinking: "hmm"}); err != nil {
		t.Fatalf("WriteChunk: %v", err)
	}
	if err := enc.Finish(stream.StreamResult{}); err != nil {
		t.Fatalf("Finish: %v", err)
	}
	events := parseSSEEvents(t, rec.Body.String())
	for _, e := range events {
		if e.Name == "content_block_start" {
			cb, _ := e.Data["content_block"].(map[string]any)
			if cb["type"] != "thinking" {
				continue
			}
			if _, has := cb["signature"]; has {
				t.Errorf("content_block_start carries a signature field on an in-progress thinking block: %v", cb)
			}
		}
	}
}

// --- pre-header vs post-header errors ---------------------------------------

func TestStream_FailEmitsNativeErrorEvent(t *testing.T) {
	t.Parallel()
	rec := httptest.NewRecorder()
	enc, err := (anthropicapi.Codec{}).OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}
	if err := enc.WriteChunk(&content.TextChunk{Text: "partial"}); err != nil {
		t.Fatalf("WriteChunk: %v", err)
	}

	// OpenStream already committed a 200 status: Fail must not (cannot) change
	// the HTTP status. It communicates failure only via the native `error` SSE
	// event.
	if rec.Code != 200 {
		t.Fatalf("status = %d, want 200 (already committed by OpenStream)", rec.Code)
	}

	if err := enc.Fail(&anthropicapi.ServerDecodeError{Reason: "upstream_boom"}); err != nil {
		t.Fatalf("Fail() error = %v", err)
	}

	events := parseSSEEvents(t, rec.Body.String())
	last := events[len(events)-1]
	if last.Name != "error" {
		t.Fatalf("last event name = %q, want error", last.Name)
	}
	errObj, _ := last.Data["error"].(map[string]any)
	if errObj["type"] != "invalid_request_error" {
		t.Errorf("error.type = %v, want invalid_request_error", errObj["type"])
	}
	if errObj["message"] == "" {
		t.Error("error.message is empty")
	}
	if last.Data["type"] != "error" {
		t.Errorf("event payload type = %v, want error", last.Data["type"])
	}
}

func TestWriteError_PreHeaderMatchesFailEnvelopeShape(t *testing.T) {
	t.Parallel()
	sampleErr := &anthropicapi.ServerDecodeError{Reason: "missing_model"}

	rec := httptest.NewRecorder()
	(anthropicapi.Codec{}).WriteError(rec, sampleErr)
	var preHeader struct {
		Type  string `json:"type"`
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &preHeader); err != nil {
		t.Fatalf("unmarshal WriteError body: %v", err)
	}

	streamRec := httptest.NewRecorder()
	enc, err := (anthropicapi.Codec{}).OpenStream(streamRec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}
	if err := enc.Fail(sampleErr); err != nil {
		t.Fatalf("Fail() error = %v", err)
	}
	events := parseSSEEvents(t, streamRec.Body.String())
	last := events[len(events)-1]
	errObj, _ := last.Data["error"].(map[string]any)

	if preHeader.Type != "error" || last.Data["type"] != "error" {
		t.Errorf("envelope type mismatch: pre-header=%q, post-header=%v", preHeader.Type, last.Data["type"])
	}
	if preHeader.Error.Type != errObj["type"] {
		t.Errorf("error.type mismatch: pre-header=%q, post-header=%v", preHeader.Error.Type, errObj["type"])
	}
	if preHeader.Error.Message != errObj["message"] {
		t.Errorf("error.message mismatch: pre-header=%q, post-header=%v", preHeader.Error.Message, errObj["message"])
	}
}

// --- single stream termination ----------------------------------------------

func TestStream_TerminationErrorsAreTyped(t *testing.T) {
	t.Parallel()
	rec := httptest.NewRecorder()
	enc, err := (anthropicapi.Codec{}).OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}
	if err := enc.Finish(stream.StreamResult{}); err != nil {
		t.Fatalf("first Finish: %v", err)
	}
	err = enc.WriteChunk(&content.TextChunk{Text: "late"})
	if err == nil {
		t.Fatal("WriteChunk after Finish: got nil error")
	}
	var termErr *anthropicapi.StreamTerminatedError
	if !errors.As(err, &termErr) {
		t.Errorf("error = %T (%v), want *anthropicapi.StreamTerminatedError", err, err)
	}
}
