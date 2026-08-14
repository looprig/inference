package openairesponses_test

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/openairesponses"
	stream "github.com/looprig/inference/stream"
)

// upstreamReasoningID is the id the UPSTREAM target issued for its reasoning
// item. ReasoningItem.required lists id, so a replay without it is a 400 — and
// the encoder now drops such an item rather than fabricating one, which makes
// losing the id equivalent to losing the whole reasoning item.
const (
	upstreamReasoningID = "rs_upstream_68f0a1"
	upstreamEncrypted   = "gAAAAAB-opaque-encrypted-blob"
)

// sseFrames returns the parsed data payloads of an SSE body, keyed in order.
func sseFrames(t *testing.T, body string) []map[string]json.RawMessage {
	t.Helper()
	var out []map[string]json.RawMessage
	for _, line := range strings.Split(body, "\n") {
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		var frame map[string]json.RawMessage
		if err := json.Unmarshal([]byte(data), &frame); err != nil {
			t.Fatalf("unmarshal SSE frame %q: %v", data, err)
		}
		out = append(out, frame)
	}
	return out
}

// streamReasoningItem drives the server stream encoder the way the gateway does
// when proxying an upstream Responses reasoning item — summary deltas first,
// then the terminal chunk carrying the item's opaque state — and returns the
// raw SSE body plus the reasoning item from the response.completed snapshot.
func streamReasoningItem(t *testing.T) (string, json.RawMessage) {
	t.Helper()
	rec := httptest.NewRecorder()
	enc, err := (openairesponses.Codec{}).OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}
	if err := enc.WriteChunk(&content.ThinkingChunk{Thinking: "planning"}); err != nil {
		t.Fatalf("WriteChunk(summary) error = %v", err)
	}
	// What stream.go emits for the upstream response.output_item.done: the
	// item's id AND its encrypted content, both provider-opaque.
	if err := enc.WriteChunk(&content.ThinkingChunk{
		ProviderState:       json.RawMessage(`{"id":"` + upstreamReasoningID + `","encrypted_content":"` + upstreamEncrypted + `"}`),
		ProviderStateFormat: "openai-responses",
	}); err != nil {
		t.Fatalf("WriteChunk(state) error = %v", err)
	}
	if err := enc.Finish(stream.StreamResult{Model: "gpt-test", FinishReason: stream.FinishReasonStop}); err != nil {
		t.Fatalf("Finish() error = %v", err)
	}

	body := rec.Body.String()
	for _, frame := range sseFrames(t, body) {
		var typ string
		if err := json.Unmarshal(frame["type"], &typ); err != nil || typ != "response.completed" {
			continue
		}
		var resp struct {
			Output []json.RawMessage `json:"output"`
		}
		if err := json.Unmarshal(frame["response"], &resp); err != nil {
			t.Fatalf("unmarshal response.completed: %v", err)
		}
		for _, item := range resp.Output {
			var probe struct {
				Type string `json:"type"`
			}
			if err := json.Unmarshal(item, &probe); err != nil {
				t.Fatalf("unmarshal output item: %v", err)
			}
			if probe.Type == "reasoning" {
				return body, item
			}
		}
		t.Fatalf("response.completed carries no reasoning item: %s", body)
	}
	t.Fatalf("no response.completed frame in stream: %s", body)
	return "", nil
}

// TestServerStream_ReasoningItemKeepsUpstreamID is the decisive round trip for
// the streaming path: serve an upstream reasoning item, feed the served bytes
// straight back into the server decoder, and require the upstream id to survive
// so the item can be replayed at all.
func TestServerStream_ReasoningItemKeepsUpstreamID(t *testing.T) {
	t.Parallel()

	_, item := streamReasoningItem(t)

	var served struct {
		Type             string `json:"type"`
		ID               string `json:"id"`
		EncryptedContent string `json:"encrypted_content"`
	}
	if err := json.Unmarshal(item, &served); err != nil {
		t.Fatalf("unmarshal reasoning item: %v", err)
	}
	if served.ID != upstreamReasoningID {
		t.Fatalf("served reasoning item id = %q, want the upstream %q (item=%s)", served.ID, upstreamReasoningID, item)
	}
	if served.EncryptedContent != upstreamEncrypted {
		t.Errorf("served encrypted_content = %q, want %q", served.EncryptedContent, upstreamEncrypted)
	}

	// Replay: the client sends the served item back as input.
	replay := `{"model":"gpt-test","input":[` + string(item) + `]}`
	decoded, err := (openairesponses.Codec{}).DecodeRequest(newDecodeRequest(replay))
	if err != nil {
		t.Fatalf("DecodeRequest() of the gateway's own output: %v (body=%s)", err, replay)
	}
	ai, ok := decoded.Request.Messages[0].(*content.AIMessage)
	if !ok {
		t.Fatalf("Messages[0] type = %T, want *content.AIMessage", decoded.Request.Messages[0])
	}
	tb, ok := ai.Blocks[0].(*content.ThinkingBlock)
	if !ok {
		t.Fatalf("Blocks[0] type = %T, want *content.ThinkingBlock", ai.Blocks[0])
	}
	var state struct {
		ID               string `json:"id"`
		EncryptedContent string `json:"encrypted_content"`
	}
	if err := json.Unmarshal(tb.ProviderState, &state); err != nil {
		t.Fatalf("ProviderState = %s: %v", tb.ProviderState, err)
	}
	if state.ID != upstreamReasoningID || state.EncryptedContent != upstreamEncrypted {
		t.Errorf("replayed state = %s, want id %q + encrypted_content %q", tb.ProviderState, upstreamReasoningID, upstreamEncrypted)
	}

	// And the replayed block must still reach the upstream request encoder:
	// an item with no real id is DROPPED there (encode.go), so a lost id is a
	// silently vanished reasoning item, not a cosmetic difference.
	body, err := openairesponses.EncodeRequest(inference.Request{Messages: decoded.Request.Messages}, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	if !strings.Contains(string(body), upstreamReasoningID) {
		t.Errorf("re-encoded request lost the reasoning item entirely: %s", body)
	}
}

// TestServerStream_ReasoningIDOnlyStateStillServesItem covers the store:true
// shape: an upstream item with an id and NO encrypted content. The id alone is
// replayable state, so the item must still be served.
func TestServerStream_ReasoningIDOnlyStateStillServesItem(t *testing.T) {
	t.Parallel()

	rec := httptest.NewRecorder()
	enc, err := (openairesponses.Codec{}).OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}
	if err := enc.WriteChunk(&content.ThinkingChunk{
		ProviderState:       json.RawMessage(`{"id":"` + upstreamReasoningID + `"}`),
		ProviderStateFormat: "openai-responses",
	}); err != nil {
		t.Fatalf("WriteChunk() error = %v", err)
	}
	if err := enc.Finish(stream.StreamResult{Model: "gpt-test", FinishReason: stream.FinishReasonStop}); err != nil {
		t.Fatalf("Finish() error = %v", err)
	}

	if !strings.Contains(rec.Body.String(), upstreamReasoningID) {
		t.Fatalf("stream dropped an id-only reasoning item: %s", rec.Body.String())
	}
}

// TestServerStream_EmptyStreamCompletesWithAnEmptyOutputArray pins the
// array-typed required field rule on the terminal event: Response.output is an
// array, and json.Marshal writes null for a nil slice.
func TestServerStream_EmptyStreamCompletesWithAnEmptyOutputArray(t *testing.T) {
	t.Parallel()

	rec := httptest.NewRecorder()
	enc, err := (openairesponses.Codec{}).OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}
	if err := enc.Finish(stream.StreamResult{Model: "gpt-test", FinishReason: stream.FinishReasonStop}); err != nil {
		t.Fatalf("Finish() error = %v", err)
	}
	if strings.Contains(rec.Body.String(), `"output":null`) {
		t.Errorf("response.completed carries \"output\":null, want []: %s", rec.Body.String())
	}
}
