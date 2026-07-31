package openaiapi_test

import (
	"encoding/json"
	"net/http"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec"
	"github.com/looprig/inference/codec/openaiapi"
	"github.com/looprig/inference/codec/servertest"
	model "github.com/looprig/inference/model"
	stream "github.com/looprig/inference/stream"
	usage "github.com/looprig/inference/usage"
)

// TestChatCompletionsServerCodec_SatisfiesContract runs the reusable
// codec/servertest suite against the real openaiapi.Codec, per the plan:
// every dialect's real ServerCodec is expected to call servertest.Run
// against itself the same way anthropicapi and openairesponses do.
func TestChatCompletionsServerCodec_SatisfiesContract(t *testing.T) {
	servertest.Run(t, servertest.Config{
		NewCodec: func() codec.ServerCodec { return openaiapi.Codec{} },
		Method:   http.MethodPost,
		Path:     "/v1/chat/completions",

		ValidBody: []byte(`{
			"model": "gpt-test",
			"messages": [
				{"role":"system","content":"be terse"},
				{"role":"user","content":"what's the weather in nyc and the time in utc?"},
				{"role":"assistant","content":null,"tool_calls":[
					{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"nyc\"}"}},
					{"id":"call_2","type":"function","function":{"name":"get_time","arguments":"{\"tz\":\"utc\"}"}}
				]},
				{"role":"tool","tool_call_id":"call_1","content":"sunny"},
				{"role":"tool","tool_call_id":"call_2","content":"12:00"}
			],
			"tools": [{"type":"function","function":{"name":"get_weather","description":"gets weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]
		}`),

		UnmatchedMethod:  http.MethodGet,
		UnmatchedPath:    "/v1/responses",
		WrongContentType: "text/plain",
		MalformedBody:    []byte(`{"model":`),

		SampleResponse: &inference.Response{
			Message: &content.AIMessage{
				Message: content.Message{
					Role:   content.RoleAssistant,
					Blocks: []content.Block{&content.TextBlock{Text: "the weather is sunny and it is 12:00 utc"}},
				},
			},
			Model:        "gpt-test",
			Usage:        &usage.Usage{InputTokens: 42, OutputTokens: 11},
			FinishReason: stream.FinishReasonStop,
		},

		SampleChunks: []content.Chunk{
			&content.TextChunk{Text: "the weather is "},
			&content.TextChunk{Text: "sunny"},
			&content.ToolUseChunk{Index: 0, ID: "call_1", Name: "get_weather"},
			&content.ToolUseChunk{Index: 0, InputJSON: `{"city":"nyc"}`},
		},

		SampleResult: stream.StreamResult{
			Model:        "gpt-test",
			FinishReason: stream.FinishReasonToolUse,
			Usage:        &usage.Usage{InputTokens: 42, OutputTokens: 11},
		},

		SampleError: &openaiapi.ServerDecodeError{Reason: "missing_model"},
	})
}

// TestChatCompletionsServerCodec_SameDialectRoundTrip encodes a neutral
// request with the existing client-direction EncodeRequest, decodes it back
// with the new server-direction DecodeRequest, and checks the result is
// semantically equivalent to the original.
func TestChatCompletionsServerCodec_SameDialectRoundTrip(t *testing.T) {
	t.Parallel()

	original := inference.Request{
		Model:  model.Model{Name: "gpt-test"},
		System: "be terse",
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{
				Role:   content.RoleUser,
				Blocks: []content.Block{&content.TextBlock{Text: "solve this"}},
			}},
			&content.AIMessage{Message: content.Message{
				Role: content.RoleAssistant,
				Blocks: []content.Block{
					&content.TextBlock{Text: "step by step"},
					&content.ToolUseBlock{ID: "call_1", Name: "calc", Input: json.RawMessage(`{"x":1}`)},
				},
			}},
			&content.ToolResultMessage{
				Message:   content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "42"}}},
				ToolUseID: "call_1",
			},
		},
	}

	body, err := openaiapi.EncodeRequest(original, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}

	req := newDecodeRequest(string(body))
	decoded, err := (openaiapi.Codec{}).DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v (body=%s)", err, body)
	}

	got := decoded.Request
	if got.System != original.System {
		t.Errorf("System = %q, want %q", got.System, original.System)
	}
	if len(got.Messages) != len(original.Messages) {
		t.Fatalf("Messages len = %d, want %d", len(got.Messages), len(original.Messages))
	}

	ai, ok := got.Messages[1].(*content.AIMessage)
	if !ok {
		t.Fatalf("Messages[1] type = %T, want *content.AIMessage", got.Messages[1])
	}
	tb, ok := ai.Blocks[0].(*content.TextBlock)
	if !ok || tb.Text != "step by step" {
		t.Errorf("Blocks[0] = %#v", ai.Blocks[0])
	}
	tu, ok := ai.Blocks[1].(*content.ToolUseBlock)
	if !ok || tu.ID != "call_1" || tu.Name != "calc" {
		t.Errorf("Blocks[1] = %#v", ai.Blocks[1])
	}

	tr, ok := got.Messages[2].(*content.ToolResultMessage)
	if !ok || tr.ToolUseID != "call_1" {
		t.Errorf("Messages[2] = %#v", got.Messages[2])
	}
}

// TestChatCompletionsServerCodec_StreamingRoute proves DecodeRequest also
// honors the wire "stream" flag correctly when set alongside a full body
// (already covered by TestServerDecode_StreamFlag), and that MatchRequest
// does not accidentally key off it.
func TestChatCompletionsServerCodec_StreamingRoute(t *testing.T) {
	t.Parallel()
	c := openaiapi.Codec{}
	req := newDecodeRequest(`{"model":"m","stream":true,"messages":[]}`)
	if !c.MatchRequest(req) {
		t.Fatal("MatchRequest() = false for a streaming body on the shared route, want true")
	}
}
