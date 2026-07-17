package geminiapi_test

import (
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	codec "github.com/looprig/inference/codec"
	"github.com/looprig/inference/codec/geminiapi"
	model "github.com/looprig/inference/model"
	stream "github.com/looprig/inference/stream"
	usage "github.com/looprig/inference/usage"
)

func TestGeminiStructuredOutputStreamParity(t *testing.T) {
	t.Parallel()

	req := inference.Request{
		Model:  model.Model{Name: "gemini", Caps: model.Capabilities{StructuredOutput: true}},
		Output: &inference.OutputSchema{Name: "out", Schema: []byte(`{"type":"object","properties":{},"additionalProperties":false}`)},
	}
	invoke, err := (geminiapi.Codec{}).EncodeRequest(req, codec.RequestModeInvoke)
	if err != nil {
		t.Fatalf("invoke EncodeRequest() error = %v", err)
	}
	streamed, err := (geminiapi.Codec{}).EncodeRequest(req, codec.RequestModeStream)
	if err != nil {
		t.Fatalf("stream EncodeRequest() error = %v", err)
	}
	invokeBody, err := io.ReadAll(invoke.Body)
	if err != nil {
		t.Fatalf("read invoke body: %v", err)
	}
	streamBody, err := io.ReadAll(streamed.Body)
	if err != nil {
		t.Fatalf("read stream body: %v", err)
	}
	if string(invokeBody) != string(streamBody) {
		t.Fatalf("invoke body = %s, stream body = %s", invokeBody, streamBody)
	}
	if !strings.Contains(string(streamBody), `"responseMimeType":"application/json"`) ||
		!strings.Contains(string(streamBody), `"responseJsonSchema"`) {
		t.Errorf("stream body lacks structured fields: %s", streamBody)
	}
}

func TestGeminiStreamResult(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		body       string
		wantUsage  *content.Usage
		wantModel  string
		wantReason stream.FinishReason
		wantErr    bool
		interrupt  bool
		wantChunks int
	}{
		{
			name: "latest cumulative metadata model and finish reason",
			body: "data: {\"modelVersion\":\"gemini-test\",\"candidates\":[{\"finishReason\":\"MAX_TOKENS\"}],\"usageMetadata\":{\"promptTokenCount\":10,\"candidatesTokenCount\":2,\"cachedContentTokenCount\":4,\"thoughtsTokenCount\":1,\"totalTokenCount\":13}}\n\n" +
				"data: {\"modelVersion\":\"gemini-test-v2\",\"candidates\":[{\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":10,\"candidatesTokenCount\":5,\"cachedContentTokenCount\":4,\"thoughtsTokenCount\":2,\"totalTokenCount\":17}}\n\n",
			wantUsage:  &content.Usage{InputTokens: 6, OutputTokens: 7, CacheReadTokens: 4, ReasoningTokens: 2},
			wantModel:  "gemini-test-v2",
			wantReason: stream.FinishReasonStop,
		},
		{name: "function call stop maps to tool use", body: "data: {\"candidates\":[{\"finishReason\":\"STOP\",\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"lookup\",\"args\":{}}}]}}]}\n\n", wantReason: stream.FinishReasonToolUse, wantChunks: 1},
		{name: "split function call then STOP remains tool use", body: "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"lookup\",\"args\":{}}}]}}]}\n\ndata: {\"candidates\":[{\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":1}}\n\n", wantReason: stream.FinishReasonToolUse, wantUsage: &content.Usage{InputTokens: 1}, wantChunks: 1},
		{name: "split function call then length overrides tool use", body: "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"lookup\",\"args\":{}}}]}}]}\n\ndata: {\"candidates\":[{\"finishReason\":\"MAX_TOKENS\"}]}\n\n", wantReason: stream.FinishReasonLength, wantChunks: 1},
		{name: "split function call then safety overrides tool use", body: "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"lookup\",\"args\":{}}}]}}]}\n\ndata: {\"candidates\":[{\"finishReason\":\"SAFETY\"}]}\n\n", wantReason: stream.FinishReasonContentFilter, wantChunks: 1},
		{name: "safety finish is provider-neutral", body: "data: {\"candidates\":[{\"finishReason\":\"SAFETY\"}]}\n\n", wantReason: stream.FinishReasonContentFilter},
		{name: "missing trailers remains clean", body: "data: {\"candidates\":[]}\n\n"},
		{name: "null count fails", body: "data: {\"usageMetadata\":{\"promptTokenCount\":null}}\n\n", wantErr: true},
		{name: "malformed count type fails", body: "data: {\"usageMetadata\":{\"promptTokenCount\":\"many\"}}\n\n", wantErr: true},
		{name: "fractional count fails", body: "data: {\"usageMetadata\":{\"candidatesTokenCount\":1.2}}\n\n", wantErr: true},
		{name: "negative count fails", body: "data: {\"usageMetadata\":{\"thoughtsTokenCount\":-1}}\n\n", wantErr: true},
		{name: "out-of-range count fails", body: "data: {\"usageMetadata\":{\"promptTokenCount\":18446744073709551616}}\n\n", wantErr: true},
		{name: "inconsistent total fails", body: "data: {\"usageMetadata\":{\"promptTokenCount\":2,\"candidatesTokenCount\":1,\"totalTokenCount\":2}}\n\n", wantErr: true},
		{name: "transport interruption rejects collected trailer", body: "data: {\"usageMetadata\":{\"promptTokenCount\":1}}\n\n", interrupt: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			var body io.ReadCloser = io.NopCloser(strings.NewReader(tt.body))
			if tt.interrupt {
				body = &interruptedReadCloser{data: []byte(tt.body)}
			}
			resp := &http.Response{Body: body}
			stream, err := (geminiapi.Codec{}).DecodeStream(resp)
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
			if !geminiUsageEqual(got.Usage, tt.wantUsage) {
				t.Errorf("Result usage = %+v, want %+v", got.Usage, tt.wantUsage)
			}
			assertGeminiUsageSnapshot(t, stream, got.Usage)
		})
	}
}

func assertGeminiUsageSnapshot(t *testing.T, stream *stream.StreamReader[content.Chunk], usage *content.Usage) {
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

func geminiUsageEqual(got, want *content.Usage) bool {
	if got == nil || want == nil {
		return got == nil && want == nil
	}
	return *got == *want
}
