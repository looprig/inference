package anthropicapi_test

import (
	"errors"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/anthropicapi"
	stream "github.com/looprig/inference/stream"
	usage "github.com/looprig/inference/usage"
)

func TestEncodeRequest_StreamPreservesStructuredFields(t *testing.T) {
	t.Parallel()

	req := inference.Request{Model: structuredModel(false), Output: structuredOutput()}
	nonStream, err := anthropicapi.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest(non-stream) error = %v", err)
	}
	streaming, err := anthropicapi.EncodeRequest(req, true)
	if err != nil {
		t.Fatalf("EncodeRequest(stream) error = %v", err)
	}
	nonStreamBody := decodeObj(t, nonStream)
	streamBody := decodeObj(t, streaming)
	delete(streamBody, "stream")
	if !reflect.DeepEqual(streamBody, nonStreamBody) {
		t.Errorf("stream request fields differ from non-stream request:\nstream=%s\nnon-stream=%s", streaming, nonStream)
	}
}

func TestAnthropicStreamResult(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		body          string
		wantUsage     *content.Usage
		wantModel     string
		wantReason    stream.FinishReason
		wantErr       bool
		interrupt     bool
		wantStreamErr bool
		wantChunks    int
	}{
		{
			name: "start input and cache combine with cumulative delta output",
			body: "data: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-test\",\"usage\":{\"input_tokens\":8,\"output_tokens\":0,\"cache_read_input_tokens\":3,\"cache_creation_input_tokens\":2}}}\n\n" +
				"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":4}}\n\n" +
				"data: {\"type\":\"message_stop\"}\n\n",
			wantUsage:  &content.Usage{InputTokens: 8, OutputTokens: 4, CacheReadTokens: 3, CacheCreationTokens: 2},
			wantModel:  "claude-test",
			wantReason: stream.FinishReasonStop,
		},
		{
			name:       "latest cumulative output and tool finish win",
			body:       "data: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-test\",\"usage\":{\"input_tokens\":1}}}\n\ndata: {\"type\":\"message_delta\",\"delta\":{},\"usage\":{\"output_tokens\":2}}\n\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":5}}\n\ndata: {\"type\":\"message_stop\"}\n\n",
			wantUsage:  &content.Usage{InputTokens: 1, OutputTokens: 5},
			wantModel:  "claude-test",
			wantReason: stream.FinishReasonToolUse,
		},
		{name: "max tokens finish is provider-neutral", body: "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\"}}\n\ndata: {\"type\":\"message_stop\"}\n\n", wantReason: stream.FinishReasonLength},
		{name: "missing trailers remains clean", body: "data: {\"type\":\"message_stop\"}\n\n"},
		{name: "null start count fails", body: "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":null}}}\n\n", wantErr: true},
		{name: "malformed delta count type fails", body: "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":\"many\"}}\n\n", wantErr: true},
		{name: "fractional delta count fails", body: "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":1.2}}\n\n", wantErr: true},
		{name: "negative delta count fails", body: "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":-1}}\n\n", wantErr: true},
		{name: "out-of-range delta count fails", body: "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":18446744073709551616}}\n\n", wantErr: true},
		{name: "transport interruption rejects collected trailer", body: "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1}}}\n\n", interrupt: true},
		{
			name: "error event after partial content is terminal",
			body: "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3}}}\n\n" +
				"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}\n\n" +
				"data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"capacity unavailable\"}}\n\n" +
				"data: {\"type\":\"message_stop\"}\n\n",
			wantStreamErr: true,
			wantChunks:    1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			var body io.ReadCloser = io.NopCloser(strings.NewReader(tt.body))
			if tt.interrupt {
				body = &interruptedReadCloser{data: []byte(tt.body)}
			}
			resp := &http.Response{Body: body}
			stream, err := (anthropicapi.Codec{}).DecodeStream(resp)
			if err != nil {
				t.Fatalf("DecodeStream() error = %v", err)
			}
			defer stream.Close()
			chunks := 0
			for {
				_, err = stream.Next()
				if err == nil {
					chunks++
					continue
				}
				if tt.wantErr {
					if errors.Is(err, io.EOF) {
						t.Fatalf("Next() error = EOF, want terminal decode failure")
					}
					var normalizationErr *usage.UsageNormalizationError
					if !errors.As(err, &normalizationErr) {
						t.Fatalf("Next() error = %T %v, want UsageNormalizationError", err, err)
					}
					if _, ok := stream.Result(); ok {
						t.Fatal("Result() available after terminal decode failure")
					}
					return
				}
				if tt.wantStreamErr {
					if chunks != tt.wantChunks {
						t.Fatalf("emitted chunks before error = %d, want %d", chunks, tt.wantChunks)
					}
					var streamErr *anthropicapi.StreamAPIError
					if !errors.As(err, &streamErr) {
						t.Fatalf("Next() error = %T %v, want StreamAPIError", err, err)
					}
					if streamErr.Type != "overloaded_error" || streamErr.Message != "capacity unavailable" {
						t.Errorf("StreamAPIError = type %q message %q", streamErr.Type, streamErr.Message)
					}
					if _, ok := stream.Result(); ok {
						t.Fatal("Result() available after Anthropic error event")
					}
					if _, nextErr := stream.Next(); !errors.As(nextErr, &streamErr) {
						t.Fatalf("Next() after terminal error = %T %v, want stable StreamAPIError", nextErr, nextErr)
					}
					return
				}
				if tt.interrupt {
					var interrupted *streamInterruptedError
					if !errors.As(err, &interrupted) {
						t.Fatalf("Next() error = %T %v, want streamInterruptedError", err, err)
					}
					if _, ok := stream.Result(); ok {
						t.Fatal("Result() available after transport interruption")
					}
					return
				}
				if !errors.Is(err, io.EOF) {
					t.Fatalf("Next() error = %v, want EOF", err)
				}
				break
			}
			if chunks != tt.wantChunks {
				t.Fatalf("emitted chunks = %d, want %d", chunks, tt.wantChunks)
			}
			got, ok := stream.Result()
			if !ok {
				t.Fatal("Result() unavailable after clean EOF")
			}
			if got.Model != tt.wantModel || got.FinishReason != tt.wantReason {
				t.Errorf("Result metadata = model %q reason %q, want %q %q", got.Model, got.FinishReason, tt.wantModel, tt.wantReason)
			}
			if !anthropicUsageEqual(got.Usage, tt.wantUsage) {
				t.Errorf("Result usage = %+v, want %+v", got.Usage, tt.wantUsage)
			}
			assertAnthropicUsageSnapshot(t, stream, got.Usage)
		})
	}
}

// TestAnthropicStream_MalformedFrameIsTerminal is the stream-level counterpart to
// TestCodec_DecodeEvent_MalformedJSONIsAnError. A truncated SSE frame must abort
// the stream with a typed decode error and withhold the result trailer — the
// previous behavior silently dropped the frame, so a stream that lost content
// mid-flight still reported an authoritative clean success once message_stop
// arrived. The message_stop below deliberately follows the truncated frame: it
// must never be reached.
func TestAnthropicStream_MalformedFrameIsTerminal(t *testing.T) {
	t.Parallel()

	body := "data: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-opus-4-8\",\"usage\":{\"input_tokens\":8,\"output_tokens\":0}}}\n\n" +
		"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n" +
		// A real text delta frame cut off mid-transmission.
		"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hel\n\n" +
		"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":4}}\n\n" +
		"data: {\"type\":\"message_stop\"}\n\n"

	resp := &http.Response{Body: io.NopCloser(strings.NewReader(body))}
	reader, err := (anthropicapi.Codec{}).DecodeStream(resp)
	if err != nil {
		t.Fatalf("DecodeStream() error = %v", err)
	}
	defer reader.Close()

	chunks := 0
	for {
		_, err = reader.Next()
		if err == nil {
			chunks++
			continue
		}
		break
	}
	if chunks != 1 {
		t.Errorf("chunks emitted before the truncated frame = %d, want 1", chunks)
	}
	if errors.Is(err, io.EOF) {
		t.Fatal("Next() = EOF, want a terminal decode failure on the truncated frame")
	}
	var decodeErr *anthropicapi.StreamEventDecodeError
	if !errors.As(err, &decodeErr) {
		t.Fatalf("Next() error = %T %v, want *StreamEventDecodeError", err, err)
	}
	if got, ok := reader.Result(); ok {
		t.Fatalf("Result() = %+v, true after a truncated frame; a lossy stream must not report success", got)
	}
}

// anthropicEvent renders one reference-shaped Anthropic SSE frame: the Messages
// streaming transport names every event on an `event:` line and repeats the
// name inside the JSON payload's `type` member, which is the one this codec
// reads.
func anthropicEvent(name, data string) string {
	return "event: " + name + "\ndata: " + data + "\n\n"
}

// TestAnthropicStream_EOFWithoutMessageStopFails locks the terminal gate. The
// Anthropic Messages stream declares MessageStopEvent (`type` const
// "message_stop") as a member of the MessageStreamEvent discriminated union and
// emits it last; there is no other end-of-generation marker on the wire, since
// the transport's own EOF is exactly what a dropped connection also produces. A
// body that stops before message_stop is therefore a truncated answer, and
// reporting it as a completed turn presents partial content as complete.
func TestAnthropicStream_EOFWithoutMessageStopFails(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		body string
	}{
		{
			name: "content deltas cut off mid-message",
			body: anthropicEvent("message_start", `{"type":"message_start","message":{"id":"msg_014p7gG3wDgGV9EUtLvnow3U","type":"message","role":"assistant","model":"claude-sonnet-4-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":472,"output_tokens":2,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}}`) +
				anthropicEvent("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`) +
				anthropicEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`),
		},
		{
			name: "message_delta trailer but no message_stop",
			body: anthropicEvent("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":25,"output_tokens":1}}}`) +
				anthropicEvent("content_block_stop", `{"type":"content_block_stop","index":0}`) +
				anthropicEvent("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":15}}`),
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
			resp := &http.Response{Body: io.NopCloser(strings.NewReader(tc.body))}
			reader, err := (anthropicapi.Codec{}).DecodeStream(resp)
			if err != nil {
				t.Fatalf("DecodeStream() error = %v", err)
			}
			defer reader.Close()

			for err == nil {
				_, err = reader.Next()
			}
			if errors.Is(err, io.EOF) {
				t.Fatal("stream ended with a clean EOF; no message_stop was ever seen")
			}
			var streamErr *anthropicapi.StreamDecodeError
			if !errors.As(err, &streamErr) {
				t.Fatalf("Next() error = %T %v, want *StreamDecodeError", err, err)
			}
			if got, ok := reader.Result(); ok {
				t.Errorf("Result() = %+v, true for a stream that never terminated", got)
			}
		})
	}
}

func assertAnthropicUsageSnapshot(t *testing.T, stream *stream.StreamReader[content.Chunk], usage *content.Usage) {
	t.Helper()
	if usage == nil {
		return
	}
	want := *usage
	usage.InputTokens++
	again, ok := stream.Result()
	if !ok || again.Usage == nil || *again.Usage != want {
		t.Errorf("Result() after caller mutation = %+v, want defensive snapshot %+v", again.Usage, want)
	}
}

type streamInterruptedError struct{}

func (*streamInterruptedError) Error() string { return "stream interrupted" }

type interruptedReadCloser struct {
	data []byte
	done bool
}

func (r *interruptedReadCloser) Read(p []byte) (int, error) {
	if r.done {
		return 0, &streamInterruptedError{}
	}
	r.done = true
	return copy(p, r.data), nil
}

func (*interruptedReadCloser) Close() error { return nil }

func anthropicUsageEqual(got, want *content.Usage) bool {
	if got == nil || want == nil {
		return got == nil && want == nil
	}
	return *got == *want
}
