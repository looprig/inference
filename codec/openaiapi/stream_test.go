package openaiapi_test

import (
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/openaiapi"
)

// Compile-time proof the OpenAI codec is a full StreamingCodec.
var _ inference.StreamingCodec = openaiapi.Codec{}

// closerSpy wraps an io.Reader and records whether Close was called.
type closerSpy struct {
	io.Reader
	closed bool
}

func (c *closerSpy) Close() error {
	c.closed = true
	return nil
}

func TestNewStream_TextChunks(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		body      string
		wantTexts []string
		wantEOF   bool
	}{
		{
			name:      "single text chunk then DONE",
			body:      "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\ndata: [DONE]\n\n",
			wantTexts: []string{"hello"},
			wantEOF:   true,
		},
		{
			name:      "multiple text chunks",
			body:      "data: {\"choices\":[{\"delta\":{\"content\":\"foo\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"bar\"}}]}\n\ndata: [DONE]\n\n",
			wantTexts: []string{"foo", "bar"},
			wantEOF:   true,
		},
		{
			name:      "role-only delta skipped",
			body:      "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n",
			wantTexts: []string{"hi"},
			wantEOF:   true,
		},
		{
			name:      "after DONE returns EOF",
			body:      "data: [DONE]\n\n",
			wantTexts: []string{},
			wantEOF:   true,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			stream := openaiapi.NewStream(io.NopCloser(strings.NewReader(tc.body)))
			defer stream.Close()

			var got []string
			for {
				chunk, err := stream.Next()
				if errors.Is(err, io.EOF) {
					break
				}
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				tc, ok := chunk.(*content.TextChunk)
				if !ok {
					t.Fatalf("expected *content.TextChunk, got %T", chunk)
				}
				got = append(got, tc.Text)
			}

			if len(got) != len(tc.wantTexts) {
				t.Fatalf("got %d chunks, want %d: %v", len(got), len(tc.wantTexts), got)
			}
			for i, want := range tc.wantTexts {
				if got[i] != want {
					t.Errorf("chunk[%d]: got %q, want %q", i, got[i], want)
				}
			}

			if tc.wantEOF {
				_, err := stream.Next()
				if !errors.Is(err, io.EOF) {
					t.Errorf("expected io.EOF after stream end, got %v", err)
				}
			}
		})
	}
}

func TestNewStream_ThinkingChunks(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name       string
		body       string
		wantTypes  []string
		wantValues []string
	}{
		{
			name:       "reasoning content yields thinking chunk",
			body:       "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"let me think\"}}]}\n\ndata: [DONE]\n\n",
			wantTypes:  []string{"thinking"},
			wantValues: []string{"let me think"},
		},
		{
			name:       "thinking then text in sequence",
			body:       "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"plan\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"result\"}}]}\n\ndata: [DONE]\n\n",
			wantTypes:  []string{"thinking", "text"},
			wantValues: []string{"plan", "result"},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			stream := openaiapi.NewStream(io.NopCloser(strings.NewReader(tc.body)))
			defer stream.Close()

			var gotTypes []string
			var gotValues []string

			for {
				chunk, err := stream.Next()
				if errors.Is(err, io.EOF) {
					break
				}
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				switch c := chunk.(type) {
				case *content.ThinkingChunk:
					gotTypes = append(gotTypes, "thinking")
					gotValues = append(gotValues, c.Thinking)
				case *content.TextChunk:
					gotTypes = append(gotTypes, "text")
					gotValues = append(gotValues, c.Text)
				default:
					t.Fatalf("unexpected chunk type: %T", chunk)
				}
			}

			if len(gotTypes) != len(tc.wantTypes) {
				t.Fatalf("got %d chunks, want %d", len(gotTypes), len(tc.wantTypes))
			}
			for i := range tc.wantTypes {
				if gotTypes[i] != tc.wantTypes[i] {
					t.Errorf("chunk[%d] type: got %q, want %q", i, gotTypes[i], tc.wantTypes[i])
				}
				if gotValues[i] != tc.wantValues[i] {
					t.Errorf("chunk[%d] value: got %q, want %q", i, gotValues[i], tc.wantValues[i])
				}
			}
		})
	}
}

func TestNewStream_ToolCallChunks(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		body string
		want []content.ToolUseChunk
	}{
		{
			name: "single tool call delta sequence: id+name first, arg fragments after",
			body: "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"read\",\"arguments\":\"{\\\"p\"}}]}}]}\n\n" +
				"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"ath\\\":\"}}]}}]}\n\n" +
				"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"x\\\"}\"}}]}}]}\n\n" +
				"data: [DONE]\n\n",
			want: []content.ToolUseChunk{
				{Index: 0, ID: "call_1", Name: "read", InputJSON: `{"p`},
				{Index: 0, InputJSON: `ath":`},
				{Index: 0, InputJSON: `"x"}`},
			},
		},
		{
			name: "first delta with id+name and empty arguments fragment",
			body: "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_2\",\"function\":{\"name\":\"ls\",\"arguments\":\"\"}}]}}]}\n\n" +
				"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{}\"}}]}}]}\n\n" +
				"data: [DONE]\n\n",
			want: []content.ToolUseChunk{
				{Index: 0, ID: "call_2", Name: "ls", InputJSON: ""},
				{Index: 0, InputJSON: "{}"},
			},
		},
		{
			name: "second tool call has its own index",
			body: "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"a\",\"function\":{\"name\":\"f\",\"arguments\":\"{}\"}}]}}]}\n\n" +
				"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"b\",\"function\":{\"name\":\"g\",\"arguments\":\"{}\"}}]}}]}\n\n" +
				"data: [DONE]\n\n",
			want: []content.ToolUseChunk{
				{Index: 0, ID: "a", Name: "f", InputJSON: "{}"},
				{Index: 1, ID: "b", Name: "g", InputJSON: "{}"},
			},
		},
		{
			name: "multiple tool-call entries in a single delta line are all emitted",
			body: "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"a\",\"function\":{\"name\":\"f\",\"arguments\":\"{}\"}},{\"index\":1,\"id\":\"b\",\"function\":{\"name\":\"g\",\"arguments\":\"{}\"}}]}}]}\n\n" +
				"data: [DONE]\n\n",
			want: []content.ToolUseChunk{
				{Index: 0, ID: "a", Name: "f", InputJSON: "{}"},
				{Index: 1, ID: "b", Name: "g", InputJSON: "{}"},
			},
		},
		{
			name: "entry with empty id, name and arguments is skipped",
			body: "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0}]}}]}\n\n" +
				"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{}\"}}]}}]}\n\n" +
				"data: [DONE]\n\n",
			want: []content.ToolUseChunk{
				{Index: 0, InputJSON: "{}"},
			},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			stream := openaiapi.NewStream(io.NopCloser(strings.NewReader(tc.body)))
			defer stream.Close()

			var got []content.ToolUseChunk
			for {
				chunk, err := stream.Next()
				if errors.Is(err, io.EOF) {
					break
				}
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				tu, ok := chunk.(*content.ToolUseChunk)
				if !ok {
					t.Fatalf("expected *content.ToolUseChunk, got %T", chunk)
				}
				got = append(got, *tu)
			}

			if len(got) != len(tc.want) {
				t.Fatalf("got %d chunks, want %d: %+v", len(got), len(tc.want), got)
			}
			for i, want := range tc.want {
				if got[i] != want {
					t.Errorf("chunk[%d] = %+v, want %+v", i, got[i], want)
				}
			}
		})
	}
}

func TestNewStream_TextAndToolCallInterleaving(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name       string
		body       string
		wantTypes  []string // "text", "thinking", "tool"
		wantValues []string // text/thinking string, or tool name|fragment
	}{
		{
			name: "text then tool call then text",
			body: "data: {\"choices\":[{\"delta\":{\"content\":\"before\"}}]}\n\n" +
				"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c\",\"function\":{\"name\":\"read\",\"arguments\":\"{}\"}}]}}]}\n\n" +
				"data: {\"choices\":[{\"delta\":{\"content\":\"after\"}}]}\n\n" +
				"data: [DONE]\n\n",
			wantTypes:  []string{"text", "tool", "text"},
			wantValues: []string{"before", "read|{}", "after"},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			stream := openaiapi.NewStream(io.NopCloser(strings.NewReader(tc.body)))
			defer stream.Close()

			var gotTypes, gotValues []string
			for {
				chunk, err := stream.Next()
				if errors.Is(err, io.EOF) {
					break
				}
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				switch c := chunk.(type) {
				case *content.TextChunk:
					gotTypes = append(gotTypes, "text")
					gotValues = append(gotValues, c.Text)
				case *content.ThinkingChunk:
					gotTypes = append(gotTypes, "thinking")
					gotValues = append(gotValues, c.Thinking)
				case *content.ToolUseChunk:
					gotTypes = append(gotTypes, "tool")
					gotValues = append(gotValues, c.Name+"|"+c.InputJSON)
				default:
					t.Fatalf("unexpected chunk type: %T", chunk)
				}
			}

			if len(gotTypes) != len(tc.wantTypes) {
				t.Fatalf("got %d chunks, want %d", len(gotTypes), len(tc.wantTypes))
			}
			for i := range tc.wantTypes {
				if gotTypes[i] != tc.wantTypes[i] {
					t.Errorf("chunk[%d] type: got %q, want %q", i, gotTypes[i], tc.wantTypes[i])
				}
				if gotValues[i] != tc.wantValues[i] {
					t.Errorf("chunk[%d] value: got %q, want %q", i, gotValues[i], tc.wantValues[i])
				}
			}
		})
	}
}

// TestCodec_DecodeStream drives the StreamingCodec path end to end through a fake
// *http.Response, proving DecodeStream frames with SSE and honors the [DONE] sentinel.
func TestCodec_DecodeStream(t *testing.T) {
	t.Parallel()
	body := "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n"
	resp := &http.Response{Body: io.NopCloser(strings.NewReader(body))}
	stream, err := openaiapi.Codec{}.DecodeStream(resp)
	if err != nil {
		t.Fatalf("DecodeStream error: %v", err)
	}
	defer stream.Close()

	chunk, err := stream.Next()
	if err != nil {
		t.Fatalf("Next error: %v", err)
	}
	tx, ok := chunk.(*content.TextChunk)
	if !ok || tx.Text != "hi" {
		t.Fatalf("chunk = %#v, want TextChunk{hi}", chunk)
	}
	if _, err := stream.Next(); !errors.Is(err, io.EOF) {
		t.Errorf("after [DONE] want io.EOF, got %v", err)
	}
}

func TestNewStream_BodyClosed(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
	}{
		{name: "Close sets closed flag"},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			spy := &closerSpy{Reader: strings.NewReader("data: [DONE]\n\n")}
			stream := openaiapi.NewStream(spy)

			if err := stream.Close(); err != nil {
				t.Fatalf("Close returned error: %v", err)
			}
			if !spy.closed {
				t.Error("expected underlying body to be closed after Close()")
			}
		})
	}
}

func TestNewStream_MalformedJSON(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name     string
		body     string
		wantText string
	}{
		{
			name:     "malformed line skipped, valid line yielded",
			body:     "data: not-json\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n",
			wantText: "ok",
		},
		{
			name:     "multiple malformed lines then valid",
			body:     "data: {\n\ndata: }\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"good\"}}]}\n\ndata: [DONE]\n\n",
			wantText: "good",
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			stream := openaiapi.NewStream(io.NopCloser(strings.NewReader(tc.body)))
			defer stream.Close()

			chunk, err := stream.Next()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			c, ok := chunk.(*content.TextChunk)
			if !ok {
				t.Fatalf("expected *content.TextChunk, got %T", chunk)
			}
			if c.Text != tc.wantText {
				t.Errorf("got text %q, want %q", c.Text, tc.wantText)
			}
		})
	}
}

func TestNewStream_EmptyChoices(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name     string
		body     string
		wantText string
	}{
		{
			name:     "empty choices skipped",
			body:     "data: {\"choices\":[]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"yes\"}}]}\n\ndata: [DONE]\n\n",
			wantText: "yes",
		},
		{
			name:     "missing choices field skipped",
			body:     "data: {\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2}}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"pass\"}}]}\n\ndata: [DONE]\n\n",
			wantText: "pass",
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			stream := openaiapi.NewStream(io.NopCloser(strings.NewReader(tc.body)))
			defer stream.Close()

			chunk, err := stream.Next()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			c, ok := chunk.(*content.TextChunk)
			if !ok {
				t.Fatalf("expected *content.TextChunk, got %T", chunk)
			}
			if c.Text != tc.wantText {
				t.Errorf("got text %q, want %q", c.Text, tc.wantText)
			}
		})
	}
}

func TestOpenAIStreamResult(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		body         string
		wantUsage    *content.Usage
		wantModel    string
		wantReason   inference.FinishReason
		wantErr      bool
		interrupt    bool
		wantNoResult bool
	}{
		{
			name: "usage-only terminal frame is metadata not content",
			body: "data: {\"model\":\"gpt-test\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
				"data: {\"model\":\"gpt-test\",\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":4,\"prompt_tokens_details\":{\"cached_tokens\":2},\"completion_tokens_details\":{\"reasoning_tokens\":1}}}\n\n" +
				"data: [DONE]\n\n",
			wantUsage:  &content.Usage{InputTokens: 7, OutputTokens: 4, CacheReadTokens: 2, ReasoningTokens: 1},
			wantModel:  "gpt-test",
			wantReason: inference.FinishReasonStop,
		},
		{
			name:       "missing usage trailer remains a clean result",
			body:       "data: {\"model\":\"gpt-test\",\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\ndata: [DONE]\n\n",
			wantModel:  "gpt-test",
			wantReason: inference.FinishReasonLength,
		},
		{name: "present empty usage remains known zero", body: "data: {\"choices\":[],\"usage\":{}}\n\ndata: [DONE]\n\n", wantUsage: &content.Usage{}},
		{name: "content filter finish is provider-neutral", body: "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"content_filter\"}]}\n\ndata: [DONE]\n\n", wantReason: inference.FinishReasonContentFilter},
		{name: "explicit null count fails", body: "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":null}}\n\ndata: [DONE]\n\n", wantErr: true},
		{name: "malformed count type fails", body: "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":\"many\"}}\n\ndata: [DONE]\n\n", wantErr: true},
		{name: "fractional count fails", body: "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1.5}}\n\ndata: [DONE]\n\n", wantErr: true},
		{name: "negative count fails", body: "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":-1}}\n\ndata: [DONE]\n\n", wantErr: true},
		{name: "out-of-range count fails", body: "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":18446744073709551616}}\n\ndata: [DONE]\n\n", wantErr: true},
		{name: "inconsistent prompt details fail", body: "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"prompt_tokens_details\":{\"cached_tokens\":2}}}\n\ndata: [DONE]\n\n", wantErr: true},
		{name: "transport interruption rejects collected trailer", body: "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1}}\n\n", interrupt: true},
		{name: "raw EOF without DONE rejects collected trailer", body: "data: {\"model\":\"partial\",\"choices\":[],\"usage\":{\"prompt_tokens\":1}}\n\n", wantNoResult: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			var body io.ReadCloser = io.NopCloser(strings.NewReader(tt.body))
			if tt.interrupt {
				body = &interruptedReadCloser{data: []byte(tt.body)}
			}
			stream := openaiapi.NewStream(body)
			defer stream.Close()
			chunks := 0
			for {
				_, err := stream.Next()
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
			if chunks != 0 {
				t.Fatalf("emitted chunks = %d, want 0", chunks)
			}
			got, ok := stream.Result()
			if tt.wantNoResult {
				if ok {
					t.Fatalf("Result() = %+v, true after raw EOF without DONE", got)
				}
				return
			}
			if !ok {
				t.Fatal("Result() unavailable after clean EOF")
			}
			if got.Model != tt.wantModel || got.FinishReason != tt.wantReason {
				t.Errorf("Result metadata = model %q reason %q, want %q %q", got.Model, got.FinishReason, tt.wantModel, tt.wantReason)
			}
			if !usageEqual(got.Usage, tt.wantUsage) {
				t.Errorf("Result usage = %+v, want %+v", got.Usage, tt.wantUsage)
			}
			assertUsageSnapshot(t, stream, got.Usage)
		})
	}
}

func assertUsageSnapshot(t *testing.T, stream *inference.StreamReader[content.Chunk], usage *content.Usage) {
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

func usageEqual(got, want *content.Usage) bool {
	if got == nil || want == nil {
		return got == nil && want == nil
	}
	return *got == *want
}
