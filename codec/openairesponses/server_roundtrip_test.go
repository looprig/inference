package openairesponses_test

import (
	"encoding/json"
	"net/http"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec"
	"github.com/looprig/inference/codec/openairesponses"
	"github.com/looprig/inference/codec/servertest"
	model "github.com/looprig/inference/model"
	stream "github.com/looprig/inference/stream"
	usage "github.com/looprig/inference/usage"
)

// TestResponsesServerCodec_SatisfiesContract runs the reusable codec/servertest
// suite against the real openairesponses.Codec, per the plan: every dialect's
// real ServerCodec is expected to call servertest.Run against itself the same
// way anthropicapi does.
func TestResponsesServerCodec_SatisfiesContract(t *testing.T) {
	servertest.Run(t, servertest.Config{
		NewCodec: func() codec.ServerCodec { return openairesponses.Codec{} },
		Method:   http.MethodPost,
		Path:     "/v1/responses",

		ValidBody: []byte(`{
			"model": "gpt-test",
			"instructions": "be terse",
			"tools": [{"type":"function","name":"get_weather","description":"gets weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}],
			"input": [
				{"type":"message","role":"user","content":[{"type":"input_text","text":"what's the weather in nyc and the time in utc?"}]},
				{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"nyc\"}"},
				{"type":"function_call","call_id":"call_2","name":"get_time","arguments":"{\"tz\":\"utc\"}"},
				{"type":"function_call_output","call_id":"call_1","output":"sunny"},
				{"type":"function_call_output","call_id":"call_2","output":"12:00"}
			]
		}`),

		UnmatchedMethod:  http.MethodGet,
		UnmatchedPath:    "/v1/other",
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

		SampleError: &openairesponses.ServerDecodeError{Reason: "missing_model"},
	})
}

// TestResponsesServerCodec_SameDialectRoundTrip encodes a neutral request
// with the existing client-direction EncodeRequest, decodes it back with the
// new server-direction DecodeRequest, and checks the result is semantically
// equivalent to the original — including the encrypted reasoning content
// round trip through ThinkingBlock.ProviderState, the concrete scenario the
// design calls out: "OpenAI encrypted reasoning content is only meaningful
// to a compatible Responses target ... same-dialect round trips MUST
// preserve it."
func TestResponsesServerCodec_SameDialectRoundTrip(t *testing.T) {
	t.Parallel()

	providerState := json.RawMessage(`"opaque-encrypted-blob"`)
	original := inference.Request{
		Model:  model.Model{Name: "gpt-test", Caps: model.Capabilities{Thinking: true}},
		System: "be terse",
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{
				Role:   content.RoleUser,
				Blocks: []content.Block{&content.TextBlock{Text: "solve this"}},
			}},
			&content.AIMessage{Message: content.Message{
				Role: content.RoleAssistant,
				Blocks: []content.Block{
					content.NewThinkingBlock("step by step", "", providerState, "openai-responses"),
					&content.ToolUseBlock{ID: "call_1", Name: "calc", Input: json.RawMessage(`{"x":1}`)},
				},
			}},
			&content.ToolResultMessage{
				Message:   content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "42"}}},
				ToolUseID: "call_1",
			},
		},
	}

	body, err := openairesponses.EncodeRequest(original, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}

	req := newDecodeRequest(string(body))
	decoded, err := (openairesponses.Codec{}).DecodeRequest(req)
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
	tb, ok := ai.Blocks[0].(*content.ThinkingBlock)
	if !ok {
		t.Fatalf("Blocks[0] type = %T, want *content.ThinkingBlock", ai.Blocks[0])
	}
	if tb.Thinking != "step by step" {
		t.Errorf("Thinking = %q", tb.Thinking)
	}
	var opaque string
	if err := json.Unmarshal(tb.ProviderState, &opaque); err != nil {
		t.Fatalf("ProviderState not a JSON string: %v", err)
	}
	if opaque != "opaque-encrypted-blob" {
		t.Errorf("ProviderState = %q, want %q (byte-for-byte replay)", opaque, "opaque-encrypted-blob")
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
