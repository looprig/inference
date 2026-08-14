package openaiapi_test

import (
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/core/content/streamaccumulator"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/openaiapi"
	model "github.com/looprig/inference/model"
)

// The Chat Completions wire shape for one assistant tool call. `arguments` is
// spec-typed `string` (ChatCompletionMessageToolCall.function.arguments,
// openai-openapi), so the value below is a JSON string whose CONTENTS are the
// arguments object — exactly what every real gateway sends.
func toolCallResponse(arguments string) []byte {
	body := `{"id":"chatcmpl-args","model":"gpt-4.1","choices":[{"index":0,"finish_reason":"tool_calls",` +
		`"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function",` +
		`"function":{"name":"get_weather","arguments":` + arguments + `}}]}}]}`
	return []byte(body)
}

func firstToolUse(t *testing.T, blocks []content.Block) *content.ToolUseBlock {
	t.Helper()
	for _, b := range blocks {
		if use, ok := b.(*content.ToolUseBlock); ok {
			return use
		}
	}
	t.Fatalf("no tool-use block in %#v", blocks)
	return nil
}

// TestDecodeResponseUnwrapsToolCallArguments pins the neutral meaning of
// ToolUseBlock.Input: the arguments OBJECT, never the wire's transport
// encoding of it. openaiapi's own server_decode (decodeToolCallArguments),
// openairesponses and anthropicapi all put the object there, and this codec's
// encoder quotes Input to rebuild `arguments` — so a decoder that left the
// JSON string literal in place double-encoded every replayed tool call, which
// strict gateways reject with
// "Assistant tool call function.arguments must be a JSON object".
func TestDecodeResponseUnwrapsToolCallArguments(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		arguments string // raw JSON text of the `arguments` member
		want      string
	}{
		{
			name:      "spec string form is unwrapped to the object",
			arguments: `"{\"city\": \"Paris\"}"`,
			want:      `{"city": "Paris"}`,
		},
		{
			name:      "bare object form is tolerated verbatim",
			arguments: `{"city":"Paris"}`,
			want:      `{"city":"Paris"}`,
		},
		{
			name:      "empty string defaults to the empty object",
			arguments: `""`,
			want:      `{}`,
		},
		{
			name:      "empty object string stays the empty object",
			arguments: `"{}"`,
			want:      `{}`,
		},
		{
			// OpenAI's own schema warns "the model does not always generate
			// valid JSON"; the bytes are preserved so the harness can fail the
			// single call (loopruntime checks json.Valid) and let the model
			// retry, instead of losing the whole completion.
			name:      "invalid JSON inside the string is preserved verbatim",
			arguments: `"{\"city\": "`,
			want:      `{"city": `,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			resp, err := openaiapi.DecodeResponse(toolCallResponse(tc.arguments))
			if err != nil {
				t.Fatalf("DecodeResponse() error = %v", err)
			}
			use := firstToolUse(t, resp.Message.Blocks)
			if string(use.Input) != tc.want {
				t.Errorf("Input = %s, want %s", use.Input, tc.want)
			}
		})
	}
}

// TestDecodeResponseToolArgumentsAbsent covers a tool call whose `arguments`
// member is missing entirely. The schema marks it required, but a gateway that
// omits it must not produce an Input the tool layer cannot unmarshal.
func TestDecodeResponseToolArgumentsAbsent(t *testing.T) {
	t.Parallel()

	body := []byte(`{"id":"chatcmpl-noargs","model":"gpt-4.1","choices":[{"index":0,"finish_reason":"tool_calls",` +
		`"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function",` +
		`"function":{"name":"list_files"}}]}}]}`)

	resp, err := openaiapi.DecodeResponse(body)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if got := string(firstToolUse(t, resp.Message.Blocks).Input); got != `{}` {
		t.Errorf("Input = %s, want {}", got)
	}
}

// TestToolCallArgumentsRoundTripToTheSameBytes is the property whose absence
// let the double-encode ship: what we REPLAY must be what the server SENT.
// Decode a response, replay the decoded assistant message through
// EncodeRequest, and compare the `arguments` member byte for byte.
func TestToolCallArgumentsRoundTripToTheSameBytes(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		arguments string
		// reEncoded marks a payload whose ESCAPING encoding/json normalizes on
		// the way out: it HTML-escapes <, > and & inside every string it
		// writes, and it emits a \uXXXX escape the server sent as the literal
		// UTF-8 rune. The bytes then differ while the JSON TEXT the model
		// produced — what the provider parses back out of the member — is
		// identical, which is the level at which tool arguments are the
		// contract. Every other case is byte-exact.
		reEncoded bool
	}{
		{name: "object", arguments: `"{\"city\": \"Paris\"}"`},
		{name: "empty object", arguments: `"{}"`},
		{name: "nested", arguments: `"{\"a\":{\"b\":[1,2]},\"c\":\"x\"}"`},
		{name: "embedded quotes and newline", arguments: `"{\"q\":\"say \\\"hi\\\"\\n\"}"`},
		{name: "unicode escape", arguments: `"{\"city\":\"S\u00e3o Paulo\"}"`, reEncoded: true},
		// Truncated model output round-trips byte-exactly too: preserving it
		// is what keeps this property total, and repairing it to "{}" would
		// break it while showing the model a call it never made.
		{name: "invalid model JSON", arguments: `"{\"city\": "`},
		{name: "html-escapable characters", arguments: `"{\"q\":\"a&b <c>\"}"`, reEncoded: true},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			resp, err := openaiapi.DecodeResponse(toolCallResponse(tc.arguments))
			if err != nil {
				t.Fatalf("DecodeResponse() error = %v", err)
			}

			body, err := openaiapi.EncodeRequest(inference.Request{
				Model:    model.Model{Name: "gpt-4.1"},
				Messages: content.AgenticMessages{resp.Message},
			}, false)
			if err != nil {
				t.Fatalf("EncodeRequest() error = %v", err)
			}

			got := encodedToolArguments(t, body)
			if !tc.reEncoded && string(got) != tc.arguments {
				t.Errorf("replayed arguments = %s, want the bytes the server sent %s", got, tc.arguments)
			}
			// The value is a JSON string (the spec's type) whose contents are
			// the object (the semantic every strict gateway enforces), and
			// those contents are what the server sent, character for
			// character.
			var replayed, sent string
			if err := json.Unmarshal(got, &replayed); err != nil {
				t.Fatalf("arguments must be a JSON string, got %s: %v", got, err)
			}
			if err := json.Unmarshal([]byte(tc.arguments), &sent); err != nil {
				t.Fatalf("fixture arguments are not a JSON string: %v", err)
			}
			if replayed != sent {
				t.Errorf("replayed arguments text = %q, want %q", replayed, sent)
			}
			if trimmed := strings.TrimSpace(replayed); !strings.HasPrefix(trimmed, "{") {
				t.Errorf("arguments string = %q, want it to contain a JSON object", replayed)
			}
		})
	}
}

// encodedToolArguments extracts messages[0].tool_calls[0].function.arguments
// from an encoded request body, without interpreting it.
func encodedToolArguments(t *testing.T, body []byte) json.RawMessage {
	t.Helper()
	var probe struct {
		Messages []struct {
			ToolCalls []struct {
				Function struct {
					Arguments json.RawMessage `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &probe); err != nil {
		t.Fatalf("unmarshal encoded body: %v", err)
	}
	if len(probe.Messages) == 0 || len(probe.Messages[0].ToolCalls) == 0 {
		t.Fatalf("encoded body has no tool call: %s", body)
	}
	return probe.Messages[0].ToolCalls[0].Function.Arguments
}

// TestStreamingAndNonStreamingToolUseAgree enforces inference/CLAUDE.md's rule
// that streaming must reconstruct the same continuation state as the
// non-streaming decoder. The divergence is what hid this bug: the streaming
// path reads `arguments` into a Go string (already unescaped) and accumulates
// the object, while the non-streaming path kept the quoted literal.
func TestStreamingAndNonStreamingToolUseAgree(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name       string
		arguments  string   // non-streaming wire value (a JSON string)
		fragments  []string // the same value split as streamed fragments
		wantObject string
	}{
		{
			name:       "single object",
			arguments:  `"{\"city\": \"Paris\"}"`,
			fragments:  []string{`{\"city\": `, `\"Paris\"}`},
			wantObject: `{"city": "Paris"}`,
		},
		{
			name:       "empty object",
			arguments:  `"{}"`,
			fragments:  []string{`{}`},
			wantObject: `{}`,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			resp, err := openaiapi.DecodeResponse(toolCallResponse(tc.arguments))
			if err != nil {
				t.Fatalf("DecodeResponse() error = %v", err)
			}
			nonStreaming := firstToolUse(t, resp.Message.Blocks)

			var sse strings.Builder
			sse.WriteString(`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"` + tc.fragments[0] + `"}}]}}]}` + "\n\n")
			for _, frag := range tc.fragments[1:] {
				sse.WriteString(`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"` + frag + `"}}]}}]}` + "\n\n")
			}
			sse.WriteString("data: [DONE]\n\n")

			streamed := accumulateToolUses(t, sse.String())
			if len(streamed) != 1 {
				t.Fatalf("streamed tool uses = %d, want 1", len(streamed))
			}

			if string(streamed[0].Input) != tc.wantObject {
				t.Errorf("streamed Input = %s, want %s", streamed[0].Input, tc.wantObject)
			}
			if string(nonStreaming.Input) != tc.wantObject {
				t.Errorf("non-streamed Input = %s, want %s", nonStreaming.Input, tc.wantObject)
			}
			if streamed[0].ID != nonStreaming.ID || streamed[0].Name != nonStreaming.Name || string(streamed[0].Input) != string(nonStreaming.Input) {
				t.Errorf("streaming/non-streaming disagree:\n streaming = %+v\n non-streaming = %+v", streamed[0], *nonStreaming)
			}
		})
	}
}

// TestEmptyArgumentsConvergeOnReplay records the one place the two paths still
// differ, and pins the boundary at which they converge.
//
// A no-argument tool call arrives as `"arguments":""`. The non-streaming
// decoder normalizes that to the empty OBJECT, because a neutral
// ToolUseBlock.Input is an arguments object and "" is not one. The streaming
// path cannot: streamaccumulator concatenates fragments and has no terminal
// event at which to distinguish "no fragment yet" from "the model sent
// nothing", so its Input stays empty. That accumulator lives in core and is
// shared by every codec. The encoder closes the gap — empty Input becomes
// "{}" — so both paths REPLAY identical bytes, which is the continuation state
// that reaches the provider.
func TestEmptyArgumentsConvergeOnReplay(t *testing.T) {
	t.Parallel()

	resp, err := openaiapi.DecodeResponse(toolCallResponse(`""`))
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	nonStreaming := firstToolUse(t, resp.Message.Blocks)
	if string(nonStreaming.Input) != `{}` {
		t.Errorf("non-streamed Input = %s, want {}", nonStreaming.Input)
	}

	streamed := accumulateToolUses(t,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}`+"\n\n"+
			"data: [DONE]\n\n")
	if len(streamed) != 1 {
		t.Fatalf("streamed tool uses = %d, want 1", len(streamed))
	}
	if len(streamed[0].Input) != 0 {
		t.Errorf("streamed Input = %s, want it empty (accumulator cannot normalize)", streamed[0].Input)
	}

	for _, block := range []content.ToolUseBlock{*nonStreaming, streamed[0]} {
		block := block
		body, err := openaiapi.EncodeRequest(inference.Request{
			Model: model.Model{Name: "gpt-4.1"},
			Messages: content.AgenticMessages{&content.AIMessage{Message: content.Message{
				Role:   content.RoleAssistant,
				Blocks: []content.Block{&block},
			}}},
		}, false)
		if err != nil {
			t.Fatalf("EncodeRequest() error = %v", err)
		}
		if got := string(encodedToolArguments(t, body)); got != `"{}"` {
			t.Errorf("replayed arguments = %s, want \"{}\"", got)
		}
	}
}

// accumulateToolUses drives an SSE body through the streaming decoder and
// folds the chunks exactly as a consumer does.
func accumulateToolUses(t *testing.T, body string) []content.ToolUseBlock {
	t.Helper()
	reader := openaiapi.NewStream(io.NopCloser(strings.NewReader(body)))
	defer reader.Close()

	var acc streamaccumulator.ToolUses
	for {
		chunk, err := reader.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("stream Next() error = %v", err)
		}
		if use, ok := chunk.(*content.ToolUseChunk); ok {
			acc.Add(use)
		}
	}
	return acc.Blocks()
}
