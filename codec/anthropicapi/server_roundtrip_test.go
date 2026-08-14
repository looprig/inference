package anthropicapi_test

import (
	"encoding/json"
	"net/http"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec"
	"github.com/looprig/inference/codec/anthropicapi"
	"github.com/looprig/inference/codec/servertest"
	model "github.com/looprig/inference/model"
	stream "github.com/looprig/inference/stream"
	usage "github.com/looprig/inference/usage"
)

// TestAnthropicServerCodec_SatisfiesContract runs the reusable
// codec/servertest suite against the real anthropicapi.Codec, per the plan:
// every dialect's real ServerCodec is expected to call servertest.Run against
// itself the same way codec/servertest's own contract_test.go does against its
// fake.
func TestAnthropicServerCodec_SatisfiesContract(t *testing.T) {
	servertest.Run(t, servertest.Config{
		NewCodec: func() codec.ServerCodec { return anthropicapi.Codec{} },
		Method:   http.MethodPost,
		Path:     "/v1/messages",

		ValidBody: []byte(`{
			"model": "claude-test",
			"max_tokens": 256,
			"system": "be terse",
			"tools": [{"name":"get_weather","description":"gets weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}}}}],
			"messages": [
				{"role": "user", "content": [{"type": "text", "text": "what's the weather in nyc and the time in utc?"}]},
				{"role": "assistant", "content": [
					{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city":"nyc"}},
					{"type": "tool_use", "id": "toolu_2", "name": "get_time", "input": {"tz":"utc"}}
				]},
				{"role": "user", "content": [
					{"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type":"text","text":"sunny"}]},
					{"type": "tool_result", "tool_use_id": "toolu_2", "content": [{"type":"text","text":"12:00"}]}
				]}
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
			Model:        "claude-test",
			Usage:        &usage.Usage{InputTokens: 42, OutputTokens: 11},
			FinishReason: stream.FinishReasonStop,
		},

		SampleChunks: []content.Chunk{
			&content.TextChunk{Text: "the weather is "},
			&content.TextChunk{Text: "sunny"},
			&content.ToolUseChunk{Index: 0, ID: "toolu_1", Name: "get_weather"},
			&content.ToolUseChunk{Index: 0, InputJSON: `{"city":"nyc"}`},
		},

		SampleResult: stream.StreamResult{
			Model:        "claude-test",
			FinishReason: stream.FinishReasonToolUse,
			Usage:        &usage.Usage{InputTokens: 42, OutputTokens: 11},
		},

		SampleError: &anthropicapi.ServerDecodeError{Reason: "missing_model"},
	})
}

// TestAnthropicServerCodec_SameDialectRoundTrip encodes a neutral request with
// the existing client-direction EncodeRequest, decodes it back with the new
// server-direction DecodeRequest, and checks the result is semantically
// equivalent to the original — the concrete scenario the design calls out:
// "a same-dialect round trip through this gateway doesn't drop/corrupt
// Anthropic's required thinking-signature replay for thinking-plus-tool-use
// turns."
func TestAnthropicServerCodec_SameDialectRoundTrip(t *testing.T) {
	t.Parallel()

	const signature = "sig-roundtrip-abc"
	original := inference.Request{
		Model:  model.Model{Name: "claude-test", Caps: model.Capabilities{Thinking: true}},
		System: "be terse",
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{
				Role:   content.RoleUser,
				Blocks: []content.Block{&content.TextBlock{Text: "solve this"}},
			}},
			&content.AIMessage{Message: content.Message{
				Role: content.RoleAssistant,
				Blocks: []content.Block{
					content.NewSignedThinkingBlock("step by step", signature, signatureFormatAnthropic, nil, ""),
					&content.ToolUseBlock{ID: "toolu_1", Name: "calc", Input: json.RawMessage(`{"x":1}`)},
				},
			}},
			&content.ToolResultMessage{
				Message:   content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "42"}}},
				ToolUseID: "toolu_1",
			},
		},
	}

	body, err := anthropicapi.EncodeRequest(original, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}

	req := newDecodeRequest(string(body))
	decoded, err := (anthropicapi.Codec{}).DecodeRequest(req)
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
	if tb.Signature != signature {
		t.Errorf("Signature = %q, want %q (byte-for-byte replay)", tb.Signature, signature)
	}
	if tb.Thinking != "step by step" {
		t.Errorf("Thinking = %q", tb.Thinking)
	}

	tu, ok := ai.Blocks[1].(*content.ToolUseBlock)
	if !ok || tu.ID != "toolu_1" || tu.Name != "calc" {
		t.Errorf("Blocks[1] = %#v", ai.Blocks[1])
	}

	tr, ok := got.Messages[2].(*content.ToolResultMessage)
	if !ok || tr.ToolUseID != "toolu_1" {
		t.Errorf("Messages[2] = %#v", got.Messages[2])
	}
}
