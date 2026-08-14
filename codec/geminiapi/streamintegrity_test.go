package geminiapi_test

import (
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference/codec/geminiapi"
	stream "github.com/looprig/inference/stream"
)

// decodeStreamOf runs a raw SSE body through the streaming codec and returns
// the chunks it yielded plus the error that terminated it (io.EOF for a clean
// stream) and whether a semantic result survived.
func decodeStreamOf(t *testing.T, body string) ([]content.Chunk, error, bool) {
	t.Helper()
	reader, err := (geminiapi.Codec{}).DecodeStream(&http.Response{Body: io.NopCloser(strings.NewReader(body))})
	if err != nil {
		t.Fatalf("DecodeStream() error = %v", err)
	}
	defer reader.Close()

	var chunks []content.Chunk
	for {
		chunk, err := reader.Next()
		if err != nil {
			_, ok := reader.Result()
			return chunks, err, ok
		}
		chunks = append(chunks, chunk)
	}
}

// terminalFrame is a well-formed final chunk: candidates[0].finishReason is
// what tells a client generation actually stopped. The v1beta discovery
// document's Candidate.finishReason is explicit — "If empty, the model has not
// stopped generating tokens."
const terminalFrame = "data: {\"candidates\":[{\"content\":{\"parts\":[],\"role\":\"model\"},\"finishReason\":\"STOP\"}],\"modelVersion\":\"gemini-2.5-flash\"}\n\n"

// --- Defect 2: malformed stream JSON must never be silently dropped ---------

// TestDecodeEvent_MalformedJSONIsAnError pins the invariant the sibling
// openaiapi/openairesponses codecs already enforce: invalid or truncated
// streaming JSON is an error, never a successful response with silently
// missing content. Skipping the frame turned a half-delivered answer into a
// clean, complete-looking one.
func TestDecodeEvent_MalformedJSONIsAnError(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		payload string
	}{
		{name: "not JSON at all", payload: `not-json`},
		{name: "truncated object", payload: `{"candidates":[{"content":{"parts":[{"text":"Hel`},
		{name: "trailing garbage after a valid chunk", payload: `{"candidates":[]}}}`},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := (geminiapi.Codec{}).DecodeEvent([]byte(tc.payload))
			var decodeErr *geminiapi.StreamEventDecodeError
			if !errors.As(err, &decodeErr) {
				t.Fatalf("DecodeEvent() error = %T %v, want *StreamEventDecodeError", err, err)
			}
			if got != nil {
				t.Errorf("DecodeEvent() chunks = %+v, want none alongside the error", got)
			}
		})
	}
}

// TestDecodeEvent_UnknownValidShapesStaySkips is the forward-compatibility half
// of the same rule: a well-formed chunk this codec has no mapping for is still
// a tolerant skip, so a new Gemini part type does not break the stream.
func TestDecodeEvent_UnknownValidShapesStaySkips(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		payload string
	}{
		{name: "no candidates", payload: `{"candidates":[]}`},
		{name: "unmodeled part type", payload: `{"candidates":[{"content":{"parts":[{"executableCode":{"language":"PYTHON","code":"print(1)"}}],"role":"model"}}]}`},
		// promptFeedback WITHOUT a blockReason is only classification detail:
		// blockReason is "Optional. If set, the prompt was blocked", so a frame
		// carrying ratings alone has nothing to report. The blocking form is
		// covered by TestDecodeEvent_BlockedPromptFailsTheStream.
		{name: "prompt feedback that reports no block", payload: `{"promptFeedback":{"safetyRatings":[{"category":"HARM_CATEGORY_HARASSMENT","probability":"NEGLIGIBLE"}]}}`},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := (geminiapi.Codec{}).DecodeEvent([]byte(tc.payload))
			if err != nil {
				t.Fatalf("DecodeEvent() error = %v, want a tolerant skip", err)
			}
			if got != nil {
				t.Errorf("DecodeEvent() chunks = %+v, want none", got)
			}
		})
	}
}

// TestDecodeEvent_BlockedPromptFailsTheStream is the streaming half of the
// diagnosis the non-streaming decoder gained (candidateLessError, decode.go). A
// prompt refused by the content filter produces a frame with promptFeedback and
// no candidates, and the stream then simply ends. Skipped as an uninteresting
// candidate-less chunk, it surfaced as "ended before a candidate reported a
// finishReason" — a truncation report for what was actually a policy refusal.
func TestDecodeEvent_BlockedPromptFailsTheStream(t *testing.T) {
	t.Parallel()

	got, err := (geminiapi.Codec{}).DecodeEvent([]byte(`{"promptFeedback":{"blockReason":"SAFETY","safetyRatings":[{"category":"HARM_CATEGORY_HARASSMENT","probability":"HIGH","blocked":true}]}}`))
	var blocked *geminiapi.PromptBlockedError
	if !errors.As(err, &blocked) {
		t.Fatalf("DecodeEvent() error = %v (%T), want *geminiapi.PromptBlockedError", err, err)
	}
	if blocked.BlockReason != "SAFETY" {
		t.Errorf("BlockReason = %q, want SAFETY", blocked.BlockReason)
	}
	if got != nil {
		t.Errorf("DecodeEvent() chunks = %+v, want none alongside the error", got)
	}
}

// TestDecodeStream_TruncatedFrameFailsTheStream drives the same defect through
// the real streaming path: a frame cut off mid-flight used to vanish, and the
// stream reported a clean EOF with an authoritative result over content that
// was demonstrably incomplete.
func TestDecodeStream_TruncatedFrameFailsTheStream(t *testing.T) {
	t.Parallel()

	body := "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello\"}],\"role\":\"model\"}}]}\n\n" +
		"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\" wor\n\n" +
		terminalFrame

	chunks, err, resultOK := decodeStreamOf(t, body)
	if errors.Is(err, io.EOF) {
		t.Fatal("stream ended with a clean EOF; a truncated frame must fail the stream")
	}
	var decodeErr *geminiapi.StreamEventDecodeError
	if !errors.As(err, &decodeErr) {
		t.Fatalf("Next() error = %T %v, want *StreamEventDecodeError", err, err)
	}
	if resultOK {
		t.Error("Result() available after a malformed frame")
	}
	if len(chunks) != 1 {
		t.Errorf("chunks before the failure = %d, want the 1 well-formed chunk", len(chunks))
	}
}

// --- Defect 3: mid-stream errors and missing terminals must not read clean ---

// TestDecodeStream_ErrorFrameIsATypedAPIError covers the `{"error":{...}}`
// envelope Google returns after the HTTP success boundary (google.rpc.Status —
// code/message/status). It parses as a perfectly valid JSON object with no
// candidates, so the tolerant "no candidates is a skip" rule swallowed it and
// the partial answer committed as complete.
func TestDecodeStream_ErrorFrameIsATypedAPIError(t *testing.T) {
	t.Parallel()

	body := "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello\"}],\"role\":\"model\"}}]}\n\n" +
		"data: {\"error\":{\"code\":500,\"message\":\"internal error\",\"status\":\"INTERNAL\"}}\n\n"

	chunks, err, resultOK := decodeStreamOf(t, body)
	if errors.Is(err, io.EOF) {
		t.Fatal("stream ended with a clean EOF; a mid-stream error frame must fail the stream")
	}
	var apiErr *geminiapi.StreamAPIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("Next() error = %T %v, want *StreamAPIError", err, err)
	}
	if apiErr.Code != 500 || apiErr.Status != "INTERNAL" || apiErr.Message != "internal error" {
		t.Errorf("StreamAPIError = %+v, want code 500 / INTERNAL / \"internal error\"", apiErr)
	}
	if resultOK {
		t.Error("Result() available after a mid-stream error frame")
	}
	if len(chunks) != 1 {
		t.Errorf("chunks before the error = %d, want 1", len(chunks))
	}
}

// TestDecodeStream_EOFWithoutTerminalFails locks the terminal gate. Gemini has
// no [DONE] sentinel, so the ONLY signal that generation finished is a
// candidate carrying a finishReason; the v1beta discovery document states
// plainly that an empty finishReason means "the model has not stopped
// generating tokens". A body that just stops is a truncated answer, and
// returning ok=true over it presents partial content as complete.
func TestDecodeStream_EOFWithoutTerminalFails(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		body string
	}{
		{
			name: "content frames but no finishReason",
			body: "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello\"}],\"role\":\"model\"}}]}\n\n",
		},
		{
			name: "usage trailer but no finishReason",
			body: "data: {\"usageMetadata\":{\"promptTokenCount\":1}}\n\n",
		},
		{
			name: "empty body",
			body: "",
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err, resultOK := decodeStreamOf(t, tc.body)
			if errors.Is(err, io.EOF) {
				t.Fatal("stream ended with a clean EOF; no terminal finishReason was ever seen")
			}
			var streamErr *geminiapi.StreamDecodeError
			if !errors.As(err, &streamErr) {
				t.Fatalf("Next() error = %T %v, want *StreamDecodeError", err, err)
			}
			if resultOK {
				t.Error("Result() available for a stream that never terminated")
			}
		})
	}
}

// TestDecodeStream_TerminalFrameCompletesCleanly is the positive control: a
// stream that does carry a finishReason still ends in io.EOF with an
// authoritative result, so the new gate rejects only genuinely truncated
// streams.
func TestDecodeStream_TerminalFrameCompletesCleanly(t *testing.T) {
	t.Parallel()

	body := "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello\"}],\"role\":\"model\"}}]}\n\n" + terminalFrame

	chunks, err, resultOK := decodeStreamOf(t, body)
	if !errors.Is(err, io.EOF) {
		t.Fatalf("Next() error = %T %v, want io.EOF", err, err)
	}
	if !resultOK {
		t.Fatal("Result() unavailable after a properly terminated stream")
	}
	if len(chunks) != 1 {
		t.Errorf("chunks = %d, want 1", len(chunks))
	}
	reader, err := (geminiapi.Codec{}).DecodeStream(&http.Response{Body: io.NopCloser(strings.NewReader(body))})
	if err != nil {
		t.Fatalf("DecodeStream() error = %v", err)
	}
	defer reader.Close()
	for {
		if _, err := reader.Next(); err != nil {
			break
		}
	}
	got, _ := reader.Result()
	if got.FinishReason != stream.FinishReasonStop || got.Model != "gemini-2.5-flash" {
		t.Errorf("Result() = %+v, want STOP / gemini-2.5-flash", got)
	}
}
