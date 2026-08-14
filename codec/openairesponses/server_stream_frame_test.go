package openairesponses_test

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/inference/codec/openairesponses"
	stream "github.com/looprig/inference/stream"
	usage "github.com/looprig/inference/usage"
)

// sseNamedFrame is one served SSE frame with its event name attached, so a
// failure can say which event was wrong rather than only which index.
type sseNamedFrame struct {
	Event string
	Data  []byte
}

// sseNamedFrames splits a served SSE body into its (event name, data payload)
// pairs. It is deliberately separate from sseEventTypes and sseFrames: these
// tests are about what is INSIDE each frame AND which event carried it, which
// neither existing helper reports together.
func sseNamedFrames(body string) []sseNamedFrame {
	var (
		frames []sseNamedFrame
		event  string
	)
	for _, line := range strings.Split(body, "\n") {
		switch {
		case strings.HasPrefix(line, "event: "):
			event = strings.TrimPrefix(line, "event: ")
		case strings.HasPrefix(line, "data: "):
			frames = append(frames, sseNamedFrame{event, []byte(strings.TrimPrefix(line, "data: "))})
		}
	}
	return frames
}

// driveEveryEventKind runs one stream through every chunk kind the encoder can
// serve, so the assertions below see every SSE payload struct the package
// defines rather than the text-only subset.
func driveEveryEventKind(t *testing.T, rec *httptest.ResponseRecorder, pause time.Duration) {
	t.Helper()
	enc, err := (openairesponses.Codec{}).OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}
	chunks := []content.Chunk{
		&content.ThinkingChunk{Thinking: "step by step"},
		&content.TextChunk{Text: "reading "},
		&content.TextChunk{Text: "it now"},
		&content.RefusalChunk{Text: "I cannot"},
		&content.ToolUseChunk{Index: 0, ID: "call_1", Name: "read_file", InputJSON: `{"path":"/y"}`},
	}
	for i, c := range chunks {
		if err := enc.WriteChunk(c); err != nil {
			t.Fatalf("WriteChunk(%d) error = %v", i, err)
		}
		if i == 0 {
			// See TestServerStream_CreatedAtIsIdenticalOnEveryFrame.
			time.Sleep(pause)
		}
	}
	if err := enc.Finish(stream.StreamResult{Model: "gpt-5", FinishReason: stream.FinishReasonToolUse,
		Usage: &usage.Usage{InputTokens: 5, OutputTokens: 6}}); err != nil {
		t.Fatalf("Finish() error = %v", err)
	}
}

// TestServerStream_EveryFrameCarriesAMonotonicSequenceNumber pins the member
// ResponseStreamEvent declares required on all 53 of its members. A
// per-frame constant would satisfy "present" while being a lie about ordering,
// so the counter itself is asserted: 0, 1, 2, ... across the whole stream,
// terminal frame included.
func TestServerStream_EveryFrameCarriesAMonotonicSequenceNumber(t *testing.T) {
	t.Parallel()

	rec := httptest.NewRecorder()
	driveEveryEventKind(t, rec, 0)

	frames := sseNamedFrames(rec.Body.String())
	if len(frames) < 10 {
		t.Fatalf("drove %d frames, want the full event vocabulary: %s", len(frames), rec.Body.String())
	}
	for i, frame := range frames {
		var probe struct {
			Sequence *int `json:"sequence_number"`
		}
		if err := json.Unmarshal(frame.Data, &probe); err != nil {
			t.Fatalf("frame %d (%s) is not JSON: %v", i, frame.Event, err)
		}
		if probe.Sequence == nil {
			t.Fatalf("frame %d (%s) omits sequence_number, which every ResponseStreamEvent member requires: %s",
				i, frame.Event, frame.Data)
		}
		if *probe.Sequence != i {
			t.Errorf("frame %d (%s) sequence_number = %d, want %d (it must count the stream, not repeat a constant)",
				i, frame.Event, *probe.Sequence, i)
		}
	}
}

// TestServerStream_CreatedAtIsIdenticalOnEveryFrame pins one created_at for the
// life of a response. The streaming path builds several encodeWireResponse
// literals and MarshalJSON stamps an unset CreatedAt with the marshal time, so
// a client watching response.created and response.completed saw the response's
// creation time move.
//
// The sleep is what makes this a real test rather than a coincidence:
// created_at has one-second resolution, so without crossing a second boundary
// mid-stream the defective encoder produces equal values anyway and the
// assertion passes on nothing.
func TestServerStream_CreatedAtIsIdenticalOnEveryFrame(t *testing.T) {
	t.Parallel()

	rec := httptest.NewRecorder()
	driveEveryEventKind(t, rec, 1100*time.Millisecond)

	first := int64(-1)
	seen := 0
	for i, frame := range sseNamedFrames(rec.Body.String()) {
		var probe struct {
			Response *struct {
				CreatedAt int64 `json:"created_at"`
			} `json:"response"`
		}
		if err := json.Unmarshal(frame.Data, &probe); err != nil {
			t.Fatalf("frame %d (%s) is not JSON: %v", i, frame.Event, err)
		}
		if probe.Response == nil {
			continue
		}
		seen++
		if first < 0 {
			first = probe.Response.CreatedAt
			if first == 0 {
				t.Fatalf("frame %d (%s) carries created_at 0", i, frame.Event)
			}
			continue
		}
		if probe.Response.CreatedAt != first {
			t.Errorf("frame %d (%s) created_at = %d, want %d: one response has one creation time",
				i, frame.Event, probe.Response.CreatedAt, first)
		}
	}
	if seen < 2 {
		t.Fatalf("found %d frames carrying a response object, want at least 2 (created + terminal)", seen)
	}
}
