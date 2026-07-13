package anthropicapi_test

import (
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/anthropicapi"
)

func TestAnthropicStreamResult(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		body          string
		wantUsage     *content.Usage
		wantModel     string
		wantReason    inference.FinishReason
		wantErr       bool
		interrupt     bool
		wantNoResult  bool
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
			wantReason: inference.FinishReasonStop,
		},
		{
			name:       "latest cumulative output and tool finish win",
			body:       "data: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-test\",\"usage\":{\"input_tokens\":1}}}\n\ndata: {\"type\":\"message_delta\",\"delta\":{},\"usage\":{\"output_tokens\":2}}\n\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":5}}\n\ndata: {\"type\":\"message_stop\"}\n\n",
			wantUsage:  &content.Usage{InputTokens: 1, OutputTokens: 5},
			wantModel:  "claude-test",
			wantReason: inference.FinishReasonToolUse,
		},
		{name: "max tokens finish is provider-neutral", body: "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\"}}\n\ndata: {\"type\":\"message_stop\"}\n\n", wantReason: inference.FinishReasonLength},
		{name: "missing trailers remains clean", body: "data: {\"type\":\"message_stop\"}\n\n"},
		{name: "null start count fails", body: "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":null}}}\n\n", wantErr: true},
		{name: "malformed delta count type fails", body: "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":\"many\"}}\n\n", wantErr: true},
		{name: "fractional delta count fails", body: "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":1.2}}\n\n", wantErr: true},
		{name: "negative delta count fails", body: "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":-1}}\n\n", wantErr: true},
		{name: "out-of-range delta count fails", body: "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":18446744073709551616}}\n\n", wantErr: true},
		{name: "transport interruption rejects collected trailer", body: "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1}}}\n\n", interrupt: true},
		{name: "raw EOF without message_stop rejects collected trailer", body: "data: {\"type\":\"message_start\",\"message\":{\"model\":\"partial\",\"usage\":{\"input_tokens\":1}}}\n\n", wantNoResult: true},
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
					var normalizationErr *inference.UsageNormalizationError
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
			if tt.wantNoResult {
				if ok {
					t.Fatalf("Result() = %+v, true after raw EOF without message_stop", got)
				}
				return
			}
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

func assertAnthropicUsageSnapshot(t *testing.T, stream *inference.StreamReader[content.Chunk], usage *content.Usage) {
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
