package openaiapi_test

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	codec "github.com/looprig/inference/codec"
	"github.com/looprig/inference/codec/openaiapi"
	model "github.com/looprig/inference/model"
	stream "github.com/looprig/inference/stream"
	usage "github.com/looprig/inference/usage"
)

// Compile-time proof the OpenAI codec is a full StreamingCodec.
var _ codec.StreamingCodec = openaiapi.Codec{}

func TestStreamingRequestPreservesStructuredOutput(t *testing.T) {
	t.Parallel()

	output := &inference.OutputSchema{
		Name:   "answer",
		Strict: true,
		Schema: json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}`),
	}
	req := inference.Request{
		Model:  model.Model{Name: "gpt-4o", Caps: model.Capabilities{StructuredOutput: true}},
		Output: output,
	}
	body, err := openaiapi.EncodeRequest(req, true)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	var wire struct {
		Stream         bool            `json:"stream"`
		ResponseFormat json.RawMessage `json:"response_format"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if !wire.Stream || wire.ResponseFormat == nil {
		t.Errorf("stream request = stream %v response_format %s", wire.Stream, wire.ResponseFormat)
	}
}

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
				if !reflect.DeepEqual(got[i], want) {
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
				if !reflect.DeepEqual(got[i], want) {
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

func TestNewStream_MalformedJSONIsError(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		body string
	}{
		{
			name: "malformed line before valid line",
			body: "data: not-json\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n",
		},
		{
			name: "multiple malformed lines",
			body: "data: {\n\ndata: }\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"good\"}}]}\n\ndata: [DONE]\n\n",
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			stream := openaiapi.NewStream(io.NopCloser(strings.NewReader(tc.body)))
			defer stream.Close()

			_, err := stream.Next()
			if err == nil || errors.Is(err, io.EOF) {
				t.Fatalf("Next() error = %v, want malformed-event error", err)
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

func TestStreamUsageResult(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		body       string
		wantUsage  *content.Usage
		wantModel  string
		wantReason stream.FinishReason
		wantErr    bool
		interrupt  bool
	}{
		{
			name: "usage-only terminal frame is metadata not content",
			body: "data: {\"model\":\"gpt-test\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
				"data: {\"model\":\"gpt-test\",\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":4,\"prompt_tokens_details\":{\"cached_tokens\":2},\"completion_tokens_details\":{\"reasoning_tokens\":1}}}\n\n" +
				"data: [DONE]\n\n",
			wantUsage:  &content.Usage{InputTokens: 7, OutputTokens: 4, CacheReadTokens: 2, ReasoningTokens: 1},
			wantModel:  "gpt-test",
			wantReason: stream.FinishReasonStop,
		},
		{
			name: "null cache-write token in terminal usage is unreported",
			body: "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":6,\"prompt_tokens_details\":{\"cached_tokens\":3,\"cache_write_tokens\":null},\"completion_tokens_details\":{\"reasoning_tokens\":2}}}\n\n" +
				"data: [DONE]\n\n",
			wantUsage: &content.Usage{InputTokens: 7, OutputTokens: 6, CacheReadTokens: 3, ReasoningTokens: 2},
		},
		{
			name:       "missing usage trailer remains a clean result",
			body:       "data: {\"model\":\"gpt-test\",\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\ndata: [DONE]\n\n",
			wantModel:  "gpt-test",
			wantReason: stream.FinishReasonLength,
		},
		{name: "present empty usage remains known zero", body: "data: {\"choices\":[],\"usage\":{}}\n\ndata: [DONE]\n\n", wantUsage: &content.Usage{}},
		{name: "content filter finish is provider-neutral", body: "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"content_filter\"}]}\n\ndata: [DONE]\n\n", wantReason: stream.FinishReasonContentFilter},
		{name: "explicit null count fails", body: "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":null}}\n\ndata: [DONE]\n\n", wantErr: true},
		{name: "malformed count type fails", body: "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":\"many\"}}\n\ndata: [DONE]\n\n", wantErr: true},
		{name: "fractional count fails", body: "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1.5}}\n\ndata: [DONE]\n\n", wantErr: true},
		{name: "negative count fails", body: "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":-1}}\n\ndata: [DONE]\n\n", wantErr: true},
		{name: "out-of-range count fails", body: "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":18446744073709551616}}\n\ndata: [DONE]\n\n", wantErr: true},
		{name: "inconsistent prompt details fail", body: "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"prompt_tokens_details\":{\"cached_tokens\":2}}}\n\ndata: [DONE]\n\n", wantErr: true},
		{name: "transport interruption rejects collected trailer", body: "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1}}\n\n", interrupt: true},
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
			if chunks != 0 {
				t.Fatalf("emitted chunks = %d, want 0", chunks)
			}
			got, ok := stream.Result()
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

// TestNewStream_EOFWithoutTerminalFails locks the terminal gate. Chat
// Completions carries two end-of-generation signals, and a truncated stream has
// neither: CreateChatCompletionRequest.stream documents the stream as
// "terminated by a `data: [DONE]` message", and every choice in
// CreateChatCompletionStreamResponse carries a required, nullable finish_reason
// that goes non-null on the chunk where that choice stops. A body that just
// ends carries no evidence the model finished, so reporting it as a completed
// turn presents partial content as complete.
func TestNewStream_EOFWithoutTerminalFails(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		body string
	}{
		{
			name: "content chunks cut off mid-message",
			body: "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1749000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n" +
				"data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1749000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n",
		},
		{
			name: "usage trailer but no finish_reason and no DONE",
			body: "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1749000000,\"model\":\"gpt-4o\",\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":4,\"total_tokens\":13}}\n\n",
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
			reader := openaiapi.NewStream(io.NopCloser(strings.NewReader(tc.body)))
			defer reader.Close()

			var err error
			for err == nil {
				_, err = reader.Next()
			}
			if errors.Is(err, io.EOF) {
				t.Fatal("stream ended with a clean EOF; neither a finish_reason nor [DONE] was ever seen")
			}
			var streamErr *openaiapi.StreamDecodeError
			if !errors.As(err, &streamErr) {
				t.Fatalf("Next() error = %T %v, want *StreamDecodeError", err, err)
			}
			if got, ok := reader.Result(); ok {
				t.Errorf("Result() = %+v, true for a stream that never terminated", got)
			}
		})
	}
}

// TestNewStream_FinishReasonWithoutDoneIsTerminal is the other half of the
// gate. [DONE] is a transport convention OpenAI documents in prose but never
// models in its schema, while finish_reason is a required member of every
// streamed choice; OpenAI-compatible gateways routinely send the second and
// omit the first. A choice that reported why it stopped is authoritative
// evidence the generation completed, so the stream ends cleanly and keeps its
// finish reason.
func TestNewStream_FinishReasonWithoutDoneIsTerminal(t *testing.T) {
	t.Parallel()

	body := "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1749000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1749000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"

	reader := openaiapi.NewStream(io.NopCloser(strings.NewReader(body)))
	defer reader.Close()

	for {
		_, err := reader.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Next() error = %T %v, want a clean stream", err, err)
		}
	}
	result, ok := reader.Result()
	if !ok {
		t.Fatal("Result() ok = false, want the trailer of a stream whose choice reported finish_reason")
	}
	if result.FinishReason != stream.FinishReasonStop || result.Model != "gpt-4o" {
		t.Errorf("Result() = %+v, want stop/gpt-4o", result)
	}
}

func assertUsageSnapshot(t *testing.T, stream *stream.StreamReader[content.Chunk], usage *content.Usage) {
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

// TestNewStream_ErrorObjectIsError covers a well-formed provider error object
// arriving over an HTTP 200 stream. ErrorResponse ({"error": Error}) is the
// spec's error envelope, and OpenAI-compatible gateways (OpenRouter documents
// this explicitly) deliver it inside an otherwise successful streaming
// response rather than on a non-2xx status, where transport's
// APIErrorFromResponse would have caught it. Swallowing it turns an upstream
// rate limit into a clean, empty, successful turn. This is distinct from the
// malformed-JSON case above: the frame parses perfectly, it just says "error".
func TestNewStream_ErrorObjectIsError(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		body string
	}{
		{
			name: "error object with a numeric code and a finish_reason choice",
			body: "data: {\"error\":{\"code\":429,\"message\":\"upstream rate limited\"},\"choices\":[{\"delta\":{},\"finish_reason\":\"error\"}]}\n\ndata: [DONE]\n\n",
		},
		{
			name: "spec-shaped error envelope with no choices",
			body: "data: {\"error\":{\"type\":\"invalid_request_error\",\"code\":\"context_length_exceeded\",\"message\":\"too long\",\"param\":null}}\n\ndata: [DONE]\n\n",
		},
		{
			name: "error arrives after content has already streamed",
			body: "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\ndata: {\"error\":{\"type\":\"server_error\",\"code\":null,\"message\":\"boom\",\"param\":null}}\n\ndata: [DONE]\n\n",
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			s := openaiapi.NewStream(io.NopCloser(strings.NewReader(tc.body)))
			defer s.Close()

			var err error
			for err == nil {
				_, err = s.Next()
			}
			if errors.Is(err, io.EOF) {
				t.Fatal("stream ended cleanly, want a surfaced provider error")
			}
			var apiErr *openaiapi.StreamAPIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("Next() error = %v (%T), want *openaiapi.StreamAPIError", err, err)
			}
		})
	}
}

// TestNewStream_ErrorObjectCarriesCodeAndMessage locks the diagnostic payload
// retained from the provider's Error object.
func TestNewStream_ErrorObjectCarriesCodeAndMessage(t *testing.T) {
	t.Parallel()

	body := "data: {\"error\":{\"type\":\"invalid_request_error\",\"code\":\"context_length_exceeded\",\"message\":\"too long\",\"param\":null}}\n\ndata: [DONE]\n\n"
	s := openaiapi.NewStream(io.NopCloser(strings.NewReader(body)))
	defer s.Close()

	var err error
	for err == nil {
		_, err = s.Next()
	}
	var apiErr *openaiapi.StreamAPIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("Next() error = %v (%T), want *openaiapi.StreamAPIError", err, err)
	}
	if apiErr.Code != "context_length_exceeded" {
		t.Errorf("Code = %q, want context_length_exceeded", apiErr.Code)
	}
	if apiErr.Message != "too long" {
		t.Errorf("Message = %q, want %q", apiErr.Message, "too long")
	}
}

// TestNewStream_NullErrorMemberIsNotAnError guards the opposite failure: some
// gateways send an explicit "error":null alongside real content, and that must
// stay a clean stream.
func TestNewStream_NullErrorMemberIsNotAnError(t *testing.T) {
	t.Parallel()

	body := "data: {\"error\":null,\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n"
	s := openaiapi.NewStream(io.NopCloser(strings.NewReader(body)))
	defer s.Close()

	chunk, err := s.Next()
	if err != nil {
		t.Fatalf("Next() error = %v, want the text chunk", err)
	}
	text, ok := chunk.(*content.TextChunk)
	if !ok || text.Text != "hi" {
		t.Fatalf("Next() = %#v, want TextChunk{hi}", chunk)
	}
	if _, err := s.Next(); !errors.Is(err, io.EOF) {
		t.Fatalf("Next() error = %v, want io.EOF", err)
	}
}
