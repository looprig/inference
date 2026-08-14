package anthropicapi_test

import (
	"errors"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/anthropicapi"
)

// omittedThinkingResponse is a reference-shaped Anthropic `POST /v1/messages`
// response from a current-generation model: adaptive thinking defaults to
// display:"omitted", so the thinking block carries a verifiable signature and an
// EMPTY thinking string. This is the ordinary shape of turn one in every
// thinking + tool-use loop on Opus 4.7/4.8/5, Sonnet 5, Fable 5, and Mythos 5.
const omittedThinkingResponse = `{
  "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
  "type": "message",
  "role": "assistant",
  "model": "claude-opus-4-8",
  "content": [
    {"type": "thinking", "thinking": "", "signature": "EosnCkYIBxgCKkBd0nR2xJ8mQ1lPq7YsW5vT"},
    {"type": "tool_use", "id": "toolu_01A09q90qw90lq917835lq9", "name": "get_weather", "input": {"location": "Paris"}}
  ],
  "stop_reason": "tool_use",
  "usage": {"input_tokens": 25, "output_tokens": 40}
}`

// TestEncodeRequest_ThinkingBlockAlwaysEmitsRequiredFields pins the wire shape of
// a thinking block against the Anthropic OpenAPI schema, where
// RequestThinkingBlock declares required = [signature, thinking, type] and
// additionalProperties = false. An omittedempty `thinking` key silently drops out
// of an empty-text thinking block, which is exactly what a current-generation
// model returns — the resulting replay is rejected with HTTP 400 on the second
// turn of every thinking + tool-use loop.
func TestEncodeRequest_ThinkingBlockAlwaysEmitsRequiredFields(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name          string
		block         content.Block
		wantThinking  string
		wantSignature string
	}{
		{
			name:          "display omitted leaves thinking empty but still required",
			block:         content.NewSignedThinkingBlock("", "EosnCkYIBxgCKkBd", signatureFormatAnthropic, nil, ""),
			wantThinking:  "",
			wantSignature: "EosnCkYIBxgCKkBd",
		},
		{
			name:          "display summarized carries both fields",
			block:         content.NewSignedThinkingBlock("weighing the options", "EosnCkYIBxgCKkBd", signatureFormatAnthropic, nil, ""),
			wantThinking:  "weighing the options",
			wantSignature: "EosnCkYIBxgCKkBd",
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			req := inference.Request{
				Model:    baseModel(),
				Messages: content.AgenticMessages{aiMsg(tc.block)},
			}
			data, err := anthropicapi.EncodeRequest(req, false)
			if err != nil {
				t.Fatalf("EncodeRequest: %v", err)
			}
			block := blocksOf(t, messagesOf(t, decodeObj(t, data))[0])[0]

			for _, key := range []string{"type", "thinking", "signature"} {
				if _, ok := block[key]; !ok {
					t.Errorf("thinking block missing required key %q: %v", key, block)
				}
			}
			if got := asString(t, block["type"]); got != "thinking" {
				t.Errorf("type = %q, want thinking", got)
			}
			if got := asString(t, block["thinking"]); got != tc.wantThinking {
				t.Errorf("thinking = %q, want %q", got, tc.wantThinking)
			}
			if got := asString(t, block["signature"]); got != tc.wantSignature {
				t.Errorf("signature = %q, want %q", got, tc.wantSignature)
			}
			// additionalProperties: false — the thinking variant carries exactly
			// its three schema properties and nothing borrowed from the shared DTO.
			if len(block) != 3 {
				t.Errorf("thinking block keys = %v, want exactly type/thinking/signature", block)
			}
		})
	}
}

// TestEncodeRequest_OmittedThinkingRoundTrip is the end-to-end regression: decode a
// real display:"omitted" response, then replay the assistant turn as the codec
// would on the next request. The replayed thinking block must still satisfy
// RequestThinkingBlock, or Anthropic rejects the follow-up with a 400.
func TestEncodeRequest_OmittedThinkingRoundTrip(t *testing.T) {
	t.Parallel()

	resp, err := anthropicapi.DecodeResponse([]byte(omittedThinkingResponse))
	if err != nil {
		t.Fatalf("DecodeResponse: %v", err)
	}
	req := inference.Request{
		Model: baseModel(),
		Messages: content.AgenticMessages{
			userMsg(textBlock("weather in Paris?")),
			resp.Message,
			toolResultMsg("toolu_01A09q90qw90lq917835lq9", false, textBlock("18C, clear")),
		},
	}
	data, err := anthropicapi.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}
	assistant := messagesOf(t, decodeObj(t, data))[1]
	thinking := blocksOf(t, assistant)[0]
	if _, ok := thinking["thinking"]; !ok {
		t.Fatalf("replayed thinking block dropped the required thinking key: %v", thinking)
	}
	if got := asString(t, thinking["signature"]); got != "EosnCkYIBxgCKkBd0nR2xJ8mQ1lPq7YsW5vT" {
		t.Errorf("signature = %q, want the signature returned by the model", got)
	}
}

// TestEncodeRequest_EmptyResponseTextRoundTrip covers the asymmetric Anthropic
// contract: response text may be empty, while request text must contain at least
// one character. A legal response must not poison every later request when its
// assistant turn is replayed.
func TestEncodeRequest_EmptyResponseTextRoundTrip(t *testing.T) {
	t.Parallel()

	resp, err := anthropicapi.DecodeResponse([]byte(`{
  "id": "msg_empty_text",
  "type": "message",
  "role": "assistant",
  "model": "claude-sonnet-5",
  "content": [
    {"type": "text", "text": "", "citations": null},
    {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "weather"}}
  ],
  "stop_reason": "tool_use",
  "usage": {"input_tokens": 10, "output_tokens": 5}
}`))
	if err != nil {
		t.Fatalf("DecodeResponse: %v", err)
	}

	req := inference.Request{
		Model: baseModel(),
		Messages: content.AgenticMessages{
			userMsg(textBlock("check the weather")),
			resp.Message,
			toolResultMsg("toolu_1", false, textBlock("sunny")),
		},
	}
	data, err := anthropicapi.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest replayed a legal response as an invalid request: %v", err)
	}
	assistant := messagesOf(t, decodeObj(t, data))[1]
	blocks := blocksOf(t, assistant)
	if len(blocks) != 1 || asString(t, blocks[0]["type"]) != "tool_use" {
		t.Fatalf("assistant blocks = %v, want only the semantic tool_use block", blocks)
	}
}

// TestEncodeRequest_EmptyTextBlockRejected covers the sibling required-field
// defect: RequestTextBlock declares required = [text, type] with text.minLength =
// 1, so an empty text block encodes to the invalid `{"type":"text"}`. The codec
// refuses it at encode time rather than shipping a body the API will reject.
func TestEncodeRequest_EmptyTextBlockRejected(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name     string
		messages content.AgenticMessages
	}{
		{
			name:     "empty user text block",
			messages: content.AgenticMessages{userMsg(textBlock(""))},
		},
		{
			name:     "empty assistant text block",
			messages: content.AgenticMessages{aiMsg(textBlock(""))},
		},
		{
			name:     "empty text block inside a tool_result",
			messages: content.AgenticMessages{toolResultMsg("toolu_1", false, textBlock(""))},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := anthropicapi.EncodeRequest(inference.Request{Model: baseModel(), Messages: tc.messages}, false)
			if err == nil {
				t.Fatal("EncodeRequest accepted an empty text block, want a typed error")
			}
			var empty *anthropicapi.EmptyTextBlockError
			if !errors.As(err, &empty) {
				t.Fatalf("error = %T %v, want *EmptyTextBlockError", err, err)
			}
		})
	}
}

// TestEncodeRequest_NonEmptyTextStillEncodes guards the fix's blast radius: the
// empty-text rejection must not disturb ordinary text blocks, whose wire shape
// keeps its existing keys.
func TestEncodeRequest_NonEmptyTextStillEncodes(t *testing.T) {
	t.Parallel()

	req := inference.Request{Model: baseModel(), Messages: content.AgenticMessages{userMsg(textBlock("hello"))}}
	data, err := anthropicapi.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}
	block := blocksOf(t, messagesOf(t, decodeObj(t, data))[0])[0]
	if got := asString(t, block["text"]); got != "hello" {
		t.Errorf("text = %q, want hello", got)
	}
	if _, ok := block["thinking"]; ok {
		t.Errorf("text block leaked a thinking key: %v", block)
	}
}
