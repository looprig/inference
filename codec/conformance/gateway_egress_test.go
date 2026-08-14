package conformance

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	codec "github.com/looprig/inference/codec"
	"github.com/looprig/inference/codec/anthropicapi"
	"github.com/looprig/inference/codec/openairesponses"
	stream "github.com/looprig/inference/stream"
	usage "github.com/looprig/inference/usage"
)

// Gateway EGRESS validation.
//
// Every other call site of this gate points INWARD: a fixture, or a request we
// are about to send. Nothing held the bytes Looprig itself SERVES against the
// format's response schema, and that is why a cluster of required-field defects
// survived — `content: null`, `output: null`, an omitted `annotations`, an
// omitted `usage`. A gateway is a provider from its client's point of view, so
// its output is bound by the response schema exactly as a real provider's is.
//
// WHAT THIS GATE COVERS TODAY, precisely, because a gate whose strength is
// assumed rather than measured is worthless (see inference/CLAUDE.md):
//
//   - anthropic / message: the whole non-streaming response envelope — every
//     required root property, the complete `usage` object, `content` as an
//     array rather than null — over every content block the dialect can serve:
//     text (with its required `citations`), tool_use (with its required
//     `caller`), thinking and redacted_thinking.
//   - anthropic / message ROUND TRIP: the served bytes are fed back into the
//     server decoder, because a response can be schema-legal and still 400 on
//     our own ingress. This is the pairing that was broken.
//   - anthropic / stream_event: every SSE frame the streaming encoder emits,
//     whole — message_start's complete Message, the content_block_start
//     response blocks, message_delta's complete MessageDelta and
//     MessageDeltaUsage, message_stop.
//   - openai-responses / response: the whole non-streaming body, over text,
//     tool calls, refusals, reasoning, the no-message case that produced
//     `"output": null`, and every terminal status.
//   - openai-responses / stream_event: every SSE frame the streaming encoder
//     emits, whole — each event's own required members (including the
//     `sequence_number` every one of the 53 members declares) and, on the
//     envelope-bearing frames, the complete embedded Response.
//
// WHAT IT DOES NOT COVER, and why:
//
//   - openai / chat completions, gemini and bedrock-converse egress. Those
//     dialects have no server encoder in this repository yet.
//
// Each uncovered path is one encoder away from being gated: add its scenario
// here the moment its encoder emits a legal body.

// egressResponse builds an Anthropic-dialect gateway response around blocks.
func egressResponse(blocks []content.Block, u *usage.Usage) *inference.Response {
	return &inference.Response{
		Message: &content.AIMessage{Message: content.Message{
			Role:   content.RoleAssistant,
			Blocks: blocks,
		}},
		Model:        "claude-sonnet-5",
		Usage:        u,
		FinishReason: stream.FinishReasonStop,
	}
}

func TestGatewayEgress_AnthropicMessageIsALegalResponse(t *testing.T) {
	t.Parallel()

	redacted := content.NewThinkingBlock("", "", json.RawMessage(`"opaque+/="`), "anthropic-redacted-thinking")

	cases := []struct {
		name string
		resp *inference.Response
	}{
		{
			// The empty case is the one that produced `"content": null`.
			name: "no message at all",
			resp: &inference.Response{Model: "claude-sonnet-5"},
		},
		{
			name: "redacted thinking block",
			resp: egressResponse([]content.Block{redacted}, &usage.Usage{InputTokens: 12, OutputTokens: 3}),
		},
		{
			name: "signed thinking block",
			resp: egressResponse([]content.Block{content.NewSignedThinkingBlock("step by step", "sig", "anthropic", nil, "")},
				&usage.Usage{InputTokens: 1, OutputTokens: 2, ReasoningTokens: 1}),
		},
		{
			// `usage` is required; a response that carries no neutral usage
			// must still emit the object rather than dropping the key.
			name: "no usage",
			resp: egressResponse([]content.Block{redacted}, nil),
		},
		{
			name: "cache counts",
			resp: egressResponse([]content.Block{redacted},
				&usage.Usage{InputTokens: 10, OutputTokens: 20, CacheReadTokens: 3, CacheCreationTokens: 4}),
		},
		{
			// ResponseTextBlock.required is [citations, text, type].
			name: "text block",
			resp: egressResponse([]content.Block{&content.TextBlock{Text: "hello"}},
				&usage.Usage{InputTokens: 1, OutputTokens: 1}),
		},
		{
			// ResponseToolUseBlock.required is [caller, id, input, name, type].
			name: "tool_use block",
			resp: egressResponse([]content.Block{
				&content.ToolUseBlock{ID: "toolu_1", Name: "read_file", Input: json.RawMessage(`{"path":"/x"}`)},
			}, &usage.Usage{InputTokens: 1, OutputTokens: 1}),
		},
		{
			name: "tool_use block with no id of its own",
			resp: egressResponse([]content.Block{&content.ToolUseBlock{Name: "read_file"}},
				&usage.Usage{InputTokens: 1, OutputTokens: 1}),
		},
		{
			name: "text and tool_use together",
			resp: egressResponse([]content.Block{
				&content.TextBlock{Text: "reading it now"},
				&content.ToolUseBlock{ID: "toolu_2", Name: "read_file", Input: json.RawMessage(`{"path":"/y"}`)},
			}, &usage.Usage{InputTokens: 5, OutputTokens: 6}),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			rec := httptest.NewRecorder()
			if err := (anthropicapi.Codec{}).WriteResponse(rec, tc.resp); err != nil {
				t.Fatalf("WriteResponse() error = %v", err)
			}
			MustValidateResponse(t, "anthropic", "message", rec.Body.Bytes())
		})
	}
}

// TestGatewayEgress_AnthropicOwnOutputReplaysToOwnIngress is the decisive test
// for the live 400. A client that receives an assistant turn from this gateway
// and replays it verbatim on the next request must be ACCEPTED. Two halves have
// to hold at once, which is why they are asserted in one test:
//
//   - the served block carries the members the RESPONSE schema requires
//     (ResponseTextBlock.citations, ResponseToolUseBlock.caller), so a real
//     Anthropic client sees a legal block; and
//   - the server decoder accepts those same bytes back, with the content
//     intact, instead of rejecting them with malformed_body/unknown field.
//
// Fixing only the first half converts a schema violation into a 400 on our own
// ingress; fixing only the second half leaves the illegal body on the wire.
func TestGatewayEgress_AnthropicOwnOutputReplaysToOwnIngress(t *testing.T) {
	t.Parallel()

	served := egressResponse([]content.Block{
		&content.TextBlock{Text: "reading it now"},
		&content.ToolUseBlock{ID: "toolu_replay", Name: "read_file", Input: json.RawMessage(`{"path":"/y"}`)},
	}, &usage.Usage{InputTokens: 5, OutputTokens: 6})

	rec := httptest.NewRecorder()
	if err := (anthropicapi.Codec{}).WriteResponse(rec, served); err != nil {
		t.Fatalf("WriteResponse() error = %v", err)
	}
	body := rec.Body.Bytes()
	MustValidateResponse(t, "anthropic", "message", body)

	var envelope struct {
		Content []map[string]json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		t.Fatalf("unmarshal served body: %v", err)
	}
	if len(envelope.Content) != 2 {
		t.Fatalf("served %d content blocks, want 2: %s", len(envelope.Content), body)
	}
	// Without these the replay below proves nothing: it would be replaying a
	// block real Anthropic never emits.
	if got, ok := envelope.Content[0]["citations"]; !ok || string(got) != "null" {
		t.Errorf("served text block citations = %s (present=%v), want explicit null", got, ok)
	}
	if got, ok := envelope.Content[1]["caller"]; !ok || string(got) != `{"type":"direct"}` {
		t.Errorf("served tool_use block caller = %s (present=%v), want {\"type\":\"direct\"}", got, ok)
	}

	// The verbatim replay: the served assistant content, unmodified, as the
	// assistant turn of the next request.
	contentJSON, err := json.Marshal(envelope.Content)
	if err != nil {
		t.Fatalf("re-marshal served content: %v", err)
	}
	replay := []byte(`{"model":"claude-sonnet-5","max_tokens":64,"messages":[` +
		`{"role":"user","content":[{"type":"text","text":"read /y"}]},` +
		`{"role":"assistant","content":` + string(contentJSON) + `}]}`)

	// A replay is a REQUEST, so it is held against the request schema too:
	// accepting a body Anthropic itself would reject is not the goal.
	MustValidateRequest(t, "anthropic", "create_message_request", replay)

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(replay))
	req.Header.Set("Content-Type", "application/json")
	decoded, err := (anthropicapi.Codec{}).DecodeRequest(req)
	if err != nil {
		t.Fatalf("gateway rejected a verbatim replay of its own assistant turn: %v\nbody: %s", err, replay)
	}

	msgs := decoded.Request.Messages
	if len(msgs) != 2 {
		t.Fatalf("decoded %d turns, want 2", len(msgs))
	}
	ai, ok := msgs[1].(*content.AIMessage)
	if !ok {
		t.Fatalf("second turn is %T, want *content.AIMessage", msgs[1])
	}
	if len(ai.Blocks) != 2 {
		t.Fatalf("decoded %d assistant blocks, want 2", len(ai.Blocks))
	}
	text, ok := ai.Blocks[0].(*content.TextBlock)
	if !ok || text.Text != "reading it now" {
		t.Errorf("text block survived as %#v", ai.Blocks[0])
	}
	tool, ok := ai.Blocks[1].(*content.ToolUseBlock)
	if !ok {
		t.Fatalf("tool block is %T, want *content.ToolUseBlock", ai.Blocks[1])
	}
	if tool.ID != "toolu_replay" || tool.Name != "read_file" || string(tool.Input) != `{"path":"/y"}` {
		t.Errorf("tool_use block survived as %#v", tool)
	}
}

// TestGatewayEgress_AnthropicIngressAcceptsRealAnthropicBlocks is the same
// defect approached from the other side: not our own output replayed, but the
// shape REAL Anthropic emits. Both members are declared by Anthropic's own
// request schema (asserted here, so the test cannot drift into demanding
// something the provider would reject), and the gateway must decode them.
//
// Before the fix this failed with, verbatim:
//
//	anthropicapi: invalid request: malformed_body: json: unknown field "citations"
func TestGatewayEgress_AnthropicIngressAcceptsRealAnthropicBlocks(t *testing.T) {
	t.Parallel()

	body := []byte(`{"model":"claude-sonnet-5","max_tokens":64,"messages":[{"role":"assistant","content":[` +
		`{"type":"text","text":"hi","citations":null},` +
		`{"type":"tool_use","id":"toolu_1","name":"read_file","input":{},"caller":{"type":"direct"}}]}]}`)
	MustValidateRequest(t, "anthropic", "create_message_request", body)

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if _, err := (anthropicapi.Codec{}).DecodeRequest(req); err != nil {
		t.Fatalf("gateway ingress rejected a legal Anthropic-shaped replay: %v", err)
	}
}

// TestGatewayEgress_AnthropicImageBlockFailsClosed covers the one output shape
// that has NO legal Anthropic response form, so the gate above can never hold
// it: the response ContentBlock union has no image member (an image is an input
// to a model, not something a model emits), while the request union does. The
// encoder used to map *content.ImageBlock to an `image` block anyway. There is
// nothing honest to emit, so the requirement is that it refuses.
func TestGatewayEgress_AnthropicImageBlockFailsClosed(t *testing.T) {
	t.Parallel()

	resp := egressResponse([]content.Block{
		&content.ImageBlock{MediaType: "image/png", Source: content.ImageSource{Data: []byte{0x89, 'P', 'N', 'G'}}},
	}, &usage.Usage{InputTokens: 1, OutputTokens: 1})

	rec := httptest.NewRecorder()
	err := (anthropicapi.Codec{}).WriteResponse(rec, resp)
	if err == nil {
		t.Fatalf("WriteResponse() served an image block the response schema cannot admit: %s", rec.Body.Bytes())
	}
	var unsupported *anthropicapi.UnsupportedBlockError
	if !errors.As(err, &unsupported) {
		t.Fatalf("WriteResponse() error = %v, want *anthropicapi.UnsupportedBlockError", err)
	}
}

// TestGatewayEgress_AnthropicStreamEventsAreLegal holds every SSE frame the
// streaming server encoder emits against the stream_event union — message_start
// with its complete Message, the content_block_start blocks (which are RESPONSE
// blocks, so text needs citations and tool_use needs caller), message_delta with
// its complete MessageDelta and MessageDeltaUsage, and message_stop.
func TestGatewayEgress_AnthropicStreamEventsAreLegal(t *testing.T) {
	t.Parallel()

	rec := httptest.NewRecorder()
	enc, err := (anthropicapi.Codec{}).OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}
	chunks := []content.Chunk{
		&content.ThinkingChunk{Thinking: "step by step"},
		&content.ThinkingChunk{Signature: "sig", SignatureFormat: "anthropic"},
		&content.ThinkingChunk{ProviderState: json.RawMessage(`"opaque+/="`), ProviderStateFormat: "anthropic-redacted-thinking"},
		&content.TextChunk{Text: "reading it now"},
		&content.ToolUseChunk{Index: 0, ID: "toolu_1", Name: "read_file", InputJSON: `{"path":`},
		&content.ToolUseChunk{Index: 0, InputJSON: `"/y"}`},
	}
	for i, c := range chunks {
		if err := enc.WriteChunk(c); err != nil {
			t.Fatalf("WriteChunk(%d) error = %v", i, err)
		}
	}
	if err := enc.Finish(stream.StreamResult{
		Model:        "claude-sonnet-5",
		FinishReason: stream.FinishReasonToolUse,
		Usage:        &usage.Usage{InputTokens: 5, OutputTokens: 6, ReasoningTokens: 2},
	}); err != nil {
		t.Fatalf("Finish() error = %v", err)
	}

	if n := MustValidateStream(t, "anthropic", "stream_event", rec.Body.Bytes()); n < 12 {
		t.Errorf("validated only %d frames; the scenario should produce more", n)
	}
}

// responsesEgress builds a gateway response for the Responses dialect.
func responsesEgress(blocks []content.Block, u *usage.Usage, reason stream.FinishReason) *inference.Response {
	return &inference.Response{
		Message: &content.AIMessage{Message: content.Message{
			Role:   content.RoleAssistant,
			Blocks: blocks,
		}},
		Model:        "gpt-5",
		Usage:        u,
		FinishReason: reason,
	}
}

// TestGatewayEgress_OpenAIResponsesIsALegalResponse holds the Responses
// non-streaming body against the Response schema. Response.required is
// [id, object, created_at, error, incomplete_details, instructions, model,
// tools, output, parallel_tool_calls, metadata, tool_choice, temperature,
// top_p] and the encoder emitted four of those fourteen; `output` was
// additionally null whenever the response carried no message, and
// usage.input_tokens_details omitted the required cache_write_tokens.
func TestGatewayEgress_OpenAIResponsesIsALegalResponse(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		resp *inference.Response
	}{
		{
			// The case that produced `"output": null`.
			name: "no message at all",
			resp: &inference.Response{Model: "gpt-5"},
		},
		{
			name: "text block",
			resp: responsesEgress([]content.Block{&content.TextBlock{Text: "hello"}},
				&usage.Usage{InputTokens: 3, OutputTokens: 4}, stream.FinishReasonStop),
		},
		{
			name: "text and tool call",
			resp: responsesEgress([]content.Block{
				&content.TextBlock{Text: "reading it now"},
				&content.ToolUseBlock{ID: "call_1", Name: "read_file", Input: json.RawMessage(`{"path":"/y"}`)},
			}, &usage.Usage{InputTokens: 5, OutputTokens: 6}, stream.FinishReasonToolUse),
		},
		{
			name: "refusal",
			resp: responsesEgress([]content.Block{&content.RefusalBlock{Text: "no"}},
				&usage.Usage{InputTokens: 1, OutputTokens: 1}, stream.FinishReasonContentFilter),
		},
		{
			name: "reasoning with cache counts",
			resp: responsesEgress([]content.Block{
				&content.ThinkingBlock{Thinking: "step by step"},
				&content.TextBlock{Text: "done"},
			}, &usage.Usage{InputTokens: 10, OutputTokens: 20, CacheReadTokens: 3, CacheCreationTokens: 4, ReasoningTokens: 7},
				stream.FinishReasonStop),
		},
		{
			name: "truncated by max output tokens",
			resp: responsesEgress([]content.Block{&content.TextBlock{Text: "half a s"}},
				&usage.Usage{InputTokens: 1, OutputTokens: 8}, stream.FinishReasonLength),
		},
		{
			name: "no usage",
			resp: responsesEgress([]content.Block{&content.TextBlock{Text: "hello"}}, nil, stream.FinishReasonStop),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			rec := httptest.NewRecorder()
			if err := (openairesponses.Codec{}).WriteResponse(rec, tc.resp); err != nil {
				t.Fatalf("WriteResponse() error = %v", err)
			}
			MustValidateResponse(t, "openai-responses", "response", rec.Body.Bytes())
		})
	}
}

// TestGatewayEgress_OpenAIResponsesOwnOutputReplaysToOwnIngress is the
// Responses counterpart of the Anthropic round trip. OutputTextContent.required
// is ["type","text","annotations","logprobs"] in BOTH directions, so satisfying
// the response schema puts `logprobs` on the wire — and the question the
// Anthropic defect makes mandatory is whether our own ingress then accepts it.
//
// Here it does, and for a reason worth recording because it is NOT the reason
// one would assume: the Responses server decode is strict
// (DisallowUnknownFields) at the request and item levels, but a content PART is
// decoded by wireItemContent.UnmarshalJSON with a plain json.Unmarshal, so
// unknown part members were never rejected in the first place. Measured by
// renaming the decode-side struct tag and watching this test still pass. The
// round trip is asserted anyway: the tolerance is a property of one
// hand-written UnmarshalJSON, and this is what pins it.
func TestGatewayEgress_OpenAIResponsesOwnOutputReplaysToOwnIngress(t *testing.T) {
	t.Parallel()

	served := responsesEgress([]content.Block{&content.TextBlock{Text: "reading it now"}},
		&usage.Usage{InputTokens: 3, OutputTokens: 4}, stream.FinishReasonStop)

	rec := httptest.NewRecorder()
	if err := (openairesponses.Codec{}).WriteResponse(rec, served); err != nil {
		t.Fatalf("WriteResponse() error = %v", err)
	}
	body := rec.Body.Bytes()
	MustValidateResponse(t, "openai-responses", "response", body)

	var envelope struct {
		Output []json.RawMessage `json:"output"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		t.Fatalf("unmarshal served body: %v", err)
	}
	if len(envelope.Output) != 1 {
		t.Fatalf("served %d output items, want 1: %s", len(envelope.Output), body)
	}
	// The replay would prove nothing if the part did not actually carry the
	// two members the schema requires.
	if !bytes.Contains(envelope.Output[0], []byte(`"annotations":[]`)) ||
		!bytes.Contains(envelope.Output[0], []byte(`"logprobs":[]`)) {
		t.Errorf("served output_text part lacks annotations/logprobs: %s", envelope.Output[0])
	}

	replay := []byte(`{"model":"gpt-5","input":[` +
		`{"type":"message","role":"user","content":[{"type":"input_text","text":"read /y"}]},` +
		string(envelope.Output[0]) + `]}`)
	MustValidateRequest(t, "openai-responses", "create_response_request", replay)

	req := httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(replay))
	req.Header.Set("Content-Type", "application/json")
	decoded, err := (openairesponses.Codec{}).DecodeRequest(req)
	if err != nil {
		t.Fatalf("gateway rejected a verbatim replay of its own assistant turn: %v\nbody: %s", err, replay)
	}
	msgs := decoded.Request.Messages
	if len(msgs) != 2 {
		t.Fatalf("decoded %d turns, want 2", len(msgs))
	}
	ai, ok := msgs[1].(*content.AIMessage)
	if !ok {
		t.Fatalf("second turn is %T, want *content.AIMessage", msgs[1])
	}
	if len(ai.Blocks) != 1 {
		t.Fatalf("decoded %d assistant blocks, want 1", len(ai.Blocks))
	}
	if text, ok := ai.Blocks[0].(*content.TextBlock); !ok || text.Text != "reading it now" {
		t.Errorf("text block survived as %#v", ai.Blocks[0])
	}
}

// TestGatewayEgress_OpenAIResponsesStreamFramesAreLegal holds every SSE frame
// the Responses stream encoder emits against ResponseStreamEvent, whole. The
// envelope-bearing frames (response.created / .completed / .incomplete /
// .failed) each carry a complete Response, so every required member the
// non-streaming body owes is owed once per frame — and the streaming path
// builds its own encodeWireResponse literals, so it is a genuinely separate
// way to get it wrong.
//
// This used to validate only the embedded Response, because whole-frame
// validation failed on `sequence_number` — required on all 53 members of
// ResponseStreamEvent, emitted on none. That defect is fixed, so the gate is
// now the whole frame, and nothing is lost by the change: the two documents'
// Response subtrees are byte-identical (measured), so stream_event's $ref to
// Response enforces exactly what the response document did, plus every
// per-event required member the embedded-object check could never see —
// output_text.delta/.done's `logprobs` and
// function_call_arguments.done's `name` were both missing and were both
// found this way.
func TestGatewayEgress_OpenAIResponsesStreamFramesAreLegal(t *testing.T) {
	t.Parallel()

	drive := func(t *testing.T, finish func(codec.StreamEncoder) error) []byte {
		t.Helper()
		rec := httptest.NewRecorder()
		enc, err := (openairesponses.Codec{}).OpenStream(rec)
		if err != nil {
			t.Fatalf("OpenStream() error = %v", err)
		}
		chunks := []content.Chunk{
			&content.ThinkingChunk{Thinking: "step by step"},
			&content.TextChunk{Text: "reading it now"},
			&content.ToolUseChunk{Index: 0, ID: "call_1", Name: "read_file", InputJSON: `{"path":"/y"}`},
		}
		for i, c := range chunks {
			if err := enc.WriteChunk(c); err != nil {
				t.Fatalf("WriteChunk(%d) error = %v", i, err)
			}
		}
		if err := finish(enc); err != nil {
			t.Fatalf("terminal event error = %v", err)
		}
		return rec.Body.Bytes()
	}

	terminals := []struct {
		name   string
		finish func(codec.StreamEncoder) error
	}{
		{
			name: "completed",
			finish: func(e codec.StreamEncoder) error {
				return e.Finish(stream.StreamResult{Model: "gpt-5", FinishReason: stream.FinishReasonToolUse,
					Usage: &usage.Usage{InputTokens: 5, OutputTokens: 6, CacheReadTokens: 2, ReasoningTokens: 1}})
			},
		},
		{
			name: "incomplete",
			finish: func(e codec.StreamEncoder) error {
				return e.Finish(stream.StreamResult{Model: "gpt-5", FinishReason: stream.FinishReasonLength})
			},
		},
		{
			name:   "failed",
			finish: func(e codec.StreamEncoder) error { return e.Fail(errors.New("upstream exploded")) },
		},
	}

	for _, tc := range terminals {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			body := drive(t, tc.finish)
			validated := MustValidateStream(t, "openai-responses", "stream_event", body)

			// The three chunk kinds above open and close three items, so a
			// stream that stopped emitting mid-way would still satisfy the
			// loop inside MustValidateStream. Counting the frames — and
			// insisting the envelope-bearing ones are there — is what keeps
			// this from passing vacuously.
			if validated < 12 {
				t.Errorf("validated %d frames, want at least 12 for three items: %s", validated, body)
			}
			frames, err := ParseSSE(body)
			if err != nil {
				t.Fatalf("ParseSSE() error = %v", err)
			}
			envelopes := 0
			for _, frame := range frames {
				var probe struct {
					Response json.RawMessage `json:"response"`
				}
				if err := json.Unmarshal(frame.Data, &probe); err == nil && len(probe.Response) > 0 {
					envelopes++
				}
			}
			if envelopes < 2 {
				t.Errorf("found %d Response envelopes, want at least 2 (created + terminal)", envelopes)
			}
		})
	}
}

// TestGatewayEgressGateIsLive inverts the gate: each payload below is a shape
// the Anthropic server encoder actually produced before this gate existed, and
// the gate must reject every one of them. Without this, a change that silently
// stopped exercising the encoder would leave the test above passing on nothing.
func TestGatewayEgressGateIsLive(t *testing.T) {
	t.Parallel()

	const legalTail = `"stop_reason":"end_turn","stop_sequence":null,"stop_details":null,"container":null,` +
		`"usage":{"input_tokens":1,"output_tokens":1,"cache_read_input_tokens":0,"cache_creation_input_tokens":0,` +
		`"cache_creation":null,"inference_geo":null,"output_tokens_details":{"thinking_tokens":0},` +
		`"server_tool_use":null,"service_tier":null}`

	cases := []struct {
		name    string
		payload string
		wantIn  string
	}{
		{
			name:    "content is null rather than an empty array",
			payload: `{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-5","content":null,` + legalTail + `}`,
			wantIn:  "content",
		},
		{
			name: "usage object omitted entirely",
			payload: `{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-5","content":[],` +
				`"stop_reason":"end_turn","stop_sequence":null,"stop_details":null,"container":null}`,
			wantIn: "usage",
		},
		{
			name: "usage present but missing its required members",
			payload: `{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-5","content":[],` +
				`"stop_reason":"end_turn","stop_sequence":null,"stop_details":null,"container":null,` +
				`"usage":{"input_tokens":1,"output_tokens":1}}`,
			wantIn: "cache_creation",
		},
		{
			name: "nullable root properties omitted",
			payload: `{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-5","content":[],` +
				`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1,"cache_read_input_tokens":0,` +
				`"cache_creation_input_tokens":0,"cache_creation":null,"inference_geo":null,` +
				`"output_tokens_details":{"thinking_tokens":0},"server_tool_use":null,"service_tier":null}}`,
			wantIn: "stop_sequence",
		},
		{
			// The defect this wave started from, in its wire form: the opaque
			// payload gone and the block served as an empty thinking block. The
			// schema cannot see the LOSS (both shapes are legal blocks) — a
			// round trip is what catches that, in
			// codec/anthropicapi/server_redacted_roundtrip_test.go. What the
			// schema does catch is the emptied block missing `signature`.
			payload: `{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-5",` +
				`"content":[{"type":"thinking","thinking":""}],` + legalTail + `}`,
			name:   "thinking block without its required signature",
			wantIn: "signature",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			err := Validate("anthropic", "message", []byte(tc.payload))
			if err == nil {
				t.Fatalf("gate accepted an illegal gateway response: %s", tc.payload)
			}
			if !strings.Contains(err.Error(), tc.wantIn) {
				t.Errorf("diagnostic does not name %q:\n%v", tc.wantIn, err)
			}
		})
	}
}
