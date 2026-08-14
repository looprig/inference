package anthropicapi_test

import (
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/anthropicapi"
	stream "github.com/looprig/inference/stream"
)

// redactedOpaque is the provider-opaque payload under test. It contains the
// base64 alphabet's non-alphanumeric members so a byte-for-byte comparison is
// meaningful rather than accidentally passing on a trivial value.
const redactedOpaque = "opaque+/=payload"

// providerStateFormatRedacted is the wire tag this dialect stamps on redacted
// thinking state; a block carrying any other tag must never be replayed here.
const providerStateFormatRedacted = "anthropic-redacted-thinking"

// signatureFormatAnthropic mirrors the package-private label this dialect
// stamps on a reasoning SIGNATURE. It is the other, independent channel of
// provider-private state on a ThinkingBlock; see the package's
// ForeignThinkingSignatureError for why a signature carrying any other label
// (or none) is refused rather than forwarded or dropped.
const signatureFormatAnthropic = "anthropic"

func redactedResponse() *inference.Response {
	return &inference.Response{
		Message: &content.AIMessage{
			Message: content.Message{
				Role: content.RoleAssistant,
				Blocks: []content.Block{
					content.NewThinkingBlock("", "", json.RawMessage(`"`+redactedOpaque+`"`), providerStateFormatRedacted),
					&content.TextBlock{Text: "answer"},
				},
			},
		},
		Model:        "claude-test",
		FinishReason: stream.FinishReasonStop,
	}
}

// replayRequestBody wraps served assistant content blocks in the request body a
// client would send back on the next turn.
func replayRequestBody(blocks []json.RawMessage) string {
	parts := make([]string, 0, len(blocks))
	for _, b := range blocks {
		parts = append(parts, string(b))
	}
	return fmt.Sprintf(`{"model":"claude-test","max_tokens":8,"messages":[{"role":"assistant","content":[%s]}]}`,
		strings.Join(parts, ","))
}

// firstThinkingBlock returns the single ThinkingBlock of a decoded assistant
// turn.
func firstThinkingBlock(t *testing.T, req inference.Request) *content.ThinkingBlock {
	t.Helper()
	if len(req.Messages) != 1 {
		t.Fatalf("decoded %d messages, want 1", len(req.Messages))
	}
	ai, ok := req.Messages[0].(*content.AIMessage)
	if !ok {
		t.Fatalf("Messages[0] type = %T, want *content.AIMessage", req.Messages[0])
	}
	tb, ok := ai.Blocks[0].(*content.ThinkingBlock)
	if !ok {
		t.Fatalf("Blocks[0] type = %T, want *content.ThinkingBlock", ai.Blocks[0])
	}
	return tb
}

// TestServerRoundTrip_RedactedThinkingNonStreaming is the decisive round trip:
// serve a redacted thinking block through the server ENCODER, feed the emitted
// bytes straight back into the server DECODER, and require the opaque state to
// survive byte-for-byte. The gateway is the only holder of that payload — if
// the encoder drops it, the continuation is gone.
func TestServerRoundTrip_RedactedThinkingNonStreaming(t *testing.T) {
	t.Parallel()

	rec := httptest.NewRecorder()
	if err := (anthropicapi.Codec{}).WriteResponse(rec, redactedResponse()); err != nil {
		t.Fatalf("WriteResponse() error = %v", err)
	}

	var served struct {
		Content []json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &served); err != nil {
		t.Fatalf("unmarshal served body: %v (%s)", err, rec.Body.String())
	}
	if len(served.Content) != 2 {
		t.Fatalf("served %d content blocks, want 2: %s", len(served.Content), rec.Body.String())
	}

	// The served block must be the redacted variant with its required `data`,
	// not a thinking block with an emptied payload.
	var block map[string]any
	if err := json.Unmarshal(served.Content[0], &block); err != nil {
		t.Fatalf("unmarshal served block: %v", err)
	}
	if block["type"] != "redacted_thinking" {
		t.Fatalf("served block type = %v, want redacted_thinking (block=%s)", block["type"], served.Content[0])
	}
	if block["data"] != redactedOpaque {
		t.Fatalf("served block data = %v, want %q (block=%s)", block["data"], redactedOpaque, served.Content[0])
	}

	decoded, err := (anthropicapi.Codec{}).DecodeRequest(newDecodeRequest(replayRequestBody(served.Content)))
	if err != nil {
		t.Fatalf("DecodeRequest() of the gateway's own output: %v (body=%s)", err, replayRequestBody(served.Content))
	}

	tb := firstThinkingBlock(t, decoded.Request)
	if !tb.ReplayableAs(providerStateFormatRedacted) {
		t.Fatalf("ProviderState = %s / format %q, want the redacted anthropic tag", tb.ProviderState, tb.ProviderStateFormat)
	}
	if got := string(tb.ProviderState); got != `"`+redactedOpaque+`"` {
		t.Errorf("ProviderState = %s, want %q (byte-for-byte)", got, `"`+redactedOpaque+`"`)
	}
}

// TestServerRoundTrip_RedactedThinkingStreamingMatchesNonStreaming holds the
// two server paths against each other: the streamed content_block and the
// non-streamed content block must be the same wire object, and both must decode
// back to the same neutral state. A streaming/non-streaming divergence here is
// exactly what CLAUDE.md forbids.
func TestServerRoundTrip_RedactedThinkingStreamingMatchesNonStreaming(t *testing.T) {
	t.Parallel()

	// Non-streaming.
	rec := httptest.NewRecorder()
	if err := (anthropicapi.Codec{}).WriteResponse(rec, redactedResponse()); err != nil {
		t.Fatalf("WriteResponse() error = %v", err)
	}
	var served struct {
		Content []json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &served); err != nil {
		t.Fatalf("unmarshal served body: %v", err)
	}

	// Streaming.
	srec := httptest.NewRecorder()
	enc, err := (anthropicapi.Codec{}).OpenStream(srec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}
	if err := enc.WriteChunk(&content.ThinkingChunk{
		ProviderState:       json.RawMessage(`"` + redactedOpaque + `"`),
		ProviderStateFormat: providerStateFormatRedacted,
	}); err != nil {
		t.Fatalf("WriteChunk() error = %v", err)
	}
	if err := enc.Finish(stream.StreamResult{Model: "claude-test", FinishReason: stream.FinishReasonStop}); err != nil {
		t.Fatalf("Finish() error = %v", err)
	}

	streamed := contentBlockStartBlock(t, srec.Body.String())

	var fromStream, fromResponse map[string]any
	if err := json.Unmarshal(streamed, &fromStream); err != nil {
		t.Fatalf("unmarshal streamed block: %v", err)
	}
	if err := json.Unmarshal(served.Content[0], &fromResponse); err != nil {
		t.Fatalf("unmarshal served block: %v", err)
	}
	if fmt.Sprint(fromStream) != fmt.Sprint(fromResponse) {
		t.Fatalf("streaming block %v != non-streaming block %v", fromStream, fromResponse)
	}

	// Both must survive a replay through the server decoder.
	for name, raw := range map[string]json.RawMessage{"streaming": streamed, "non-streaming": served.Content[0]} {
		decoded, err := (anthropicapi.Codec{}).DecodeRequest(newDecodeRequest(replayRequestBody([]json.RawMessage{raw})))
		if err != nil {
			t.Fatalf("%s: DecodeRequest() of the gateway's own output: %v", name, err)
		}
		tb := firstThinkingBlock(t, decoded.Request)
		if got := string(tb.ProviderState); got != `"`+redactedOpaque+`"` {
			t.Errorf("%s: ProviderState = %s, want %q", name, got, `"`+redactedOpaque+`"`)
		}
	}
}

// contentBlockStartBlock extracts the `content_block` object of the first
// content_block_start frame in an SSE body.
func contentBlockStartBlock(t *testing.T, body string) json.RawMessage {
	t.Helper()
	for _, line := range strings.Split(body, "\n") {
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		var frame struct {
			Type         string          `json:"type"`
			ContentBlock json.RawMessage `json:"content_block"`
		}
		if err := json.Unmarshal([]byte(data), &frame); err != nil {
			t.Fatalf("unmarshal SSE frame %q: %v", data, err)
		}
		if frame.Type == "content_block_start" {
			return frame.ContentBlock
		}
	}
	t.Fatalf("no content_block_start frame in stream: %s", body)
	return nil
}
