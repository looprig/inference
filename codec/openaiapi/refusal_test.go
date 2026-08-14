package openaiapi_test

import (
	"encoding/json"
	"io"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/openaiapi"
	model "github.com/looprig/inference/model"
	stream "github.com/looprig/inference/stream"
)

// Refusal coverage.
//
// ChatCompletionResponseMessage.required is ["role","content","refusal"], so a
// structured-output refusal always arrives with content:null and the refusal
// text in its own member.
//
// These tests replaced an earlier set that asserted the INTERIM mapping — the
// refusal decoding to a *content.TextBlock and the turn's finish reason being
// overridden to content_filter — which predated content.RefusalBlock. Both
// halves of that mapping were defects the block type exists to remove, so the
// tests asserting them had to be rewritten rather than preserved.

func TestDecodeResponse_RefusalBecomesRefusalBlock(t *testing.T) {
	t.Parallel()

	resp, err := openaiapi.DecodeResponse([]byte(`{
		"id": "chatcmpl-1",
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {"role":"assistant","content":null,"refusal":"I cannot help with that."},
			"finish_reason": "stop"
		}]
	}`))
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}

	if len(resp.Message.Blocks) != 1 {
		t.Fatalf("Blocks = %#v, want one block carrying the refusal", resp.Message.Blocks)
	}
	refusal, ok := resp.Message.Blocks[0].(*content.RefusalBlock)
	if !ok {
		t.Fatalf("block 0 = %T, want *content.RefusalBlock", resp.Message.Blocks[0])
	}
	if refusal.Text != "I cannot help with that." {
		t.Errorf("refusal text = %q, want it surfaced verbatim", refusal.Text)
	}
	// The finish reason is now reported exactly as OpenAI sent it. The block
	// carries the "declined" signal per block; manufacturing content_filter
	// here would report a content-filter intervention the provider did not
	// report and would make a real content_filter finish indistinguishable
	// from a model refusal.
	if resp.FinishReason != stream.FinishReasonStop {
		t.Errorf("FinishReason = %q, want %q (the wire value)", resp.FinishReason, stream.FinishReasonStop)
	}
}

// TestDecodeResponse_RefusalIsNeverAnEmptySuccess is the regression the block
// type exists for: a refusal must never decode as a zero-block reply.
func TestDecodeResponse_RefusalIsNeverAnEmptySuccess(t *testing.T) {
	t.Parallel()

	for _, body := range []string{
		`{"choices":[{"message":{"role":"assistant","content":null,"refusal":"no"},"finish_reason":"stop"}]}`,
		`{"choices":[{"message":{"role":"assistant","content":"","refusal":"no"},"finish_reason":"stop"}]}`,
		`{"choices":[{"message":{"role":"assistant","content":null,"refusal":"no"},"finish_reason":""}]}`,
		// A refusal with no explanation: the member is present but empty. The
		// PRESENCE of the block is the signal, not its contents, so this must
		// still yield a block rather than a clean empty reply.
		`{"choices":[{"message":{"role":"assistant","content":null,"refusal":""},"finish_reason":"stop"}]}`,
	} {
		resp, err := openaiapi.DecodeResponse([]byte(body))
		if err != nil {
			t.Fatalf("DecodeResponse(%s) error = %v", body, err)
		}
		found := false
		for _, b := range resp.Message.Blocks {
			if _, ok := b.(*content.RefusalBlock); ok {
				found = true
			}
		}
		if !found {
			t.Errorf("DecodeResponse(%s) produced no *content.RefusalBlock: %#v", body, resp.Message.Blocks)
		}
	}
}

// TestDecodeResponse_NullRefusalIsNotABlock pins the other side: the member is
// present on every response, so its null form must not manufacture a block.
func TestDecodeResponse_NullRefusalIsNotABlock(t *testing.T) {
	t.Parallel()

	resp, err := openaiapi.DecodeResponse([]byte(
		`{"choices":[{"message":{"role":"assistant","content":"hello","refusal":null},"finish_reason":"stop"}]}`))
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if len(resp.Message.Blocks) != 1 {
		t.Fatalf("Blocks = %#v, want just the text block", resp.Message.Blocks)
	}
	if _, ok := resp.Message.Blocks[0].(*content.TextBlock); !ok {
		t.Errorf("block 0 = %T, want *content.TextBlock", resp.Message.Blocks[0])
	}
	if resp.FinishReason != stream.FinishReasonStop {
		t.Errorf("FinishReason = %q, want stop", resp.FinishReason)
	}
}

// TestDecodeResponse_RealContentFilterStillMaps pins that dropping the refusal
// override did not weaken the genuine content_filter mapping: OpenAI's own
// finish_reason enum member still maps to the neutral value, and it is now the
// only thing that produces it — which is what makes the two distinguishable.
func TestDecodeResponse_RealContentFilterStillMaps(t *testing.T) {
	t.Parallel()

	resp, err := openaiapi.DecodeResponse([]byte(
		`{"choices":[{"message":{"role":"assistant","content":"","refusal":null},"finish_reason":"content_filter"}]}`))
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if resp.FinishReason != stream.FinishReasonContentFilter {
		t.Errorf("FinishReason = %q, want %q", resp.FinishReason, stream.FinishReasonContentFilter)
	}
	for _, b := range resp.Message.Blocks {
		if _, ok := b.(*content.RefusalBlock); ok {
			t.Errorf("a content_filter turn produced a refusal block: %#v", resp.Message.Blocks)
		}
	}
}

// TestDecodeEvent_RefusalDeltaBecomesRefusalChunk pins the streaming half:
// ChatCompletionStreamResponseDelta carries its own `refusal` delta channel.
func TestDecodeEvent_RefusalDeltaBecomesRefusalChunk(t *testing.T) {
	t.Parallel()

	chunks, err := openaiapi.Codec{}.DecodeEvent([]byte(
		`{"choices":[{"index":0,"delta":{"role":"assistant","refusal":"I cannot"}}]}`))
	if err != nil {
		t.Fatalf("DecodeEvent() error = %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("chunks = %#v, want one refusal chunk", chunks)
	}
	refusal, ok := chunks[0].(*content.RefusalChunk)
	if !ok {
		t.Fatalf("chunk 0 = %T, want *content.RefusalChunk", chunks[0])
	}
	if refusal.Text != "I cannot" {
		t.Errorf("chunk text = %q, want the refusal fragment", refusal.Text)
	}
}

// TestStream_RefusalDeltasFoldIntoARefusalBlock pins that the streaming path
// reconstructs the same block the non-streaming decoder produces, and that the
// terminal result reports the wire's own finish reason.
func TestStream_RefusalDeltasFoldIntoARefusalBlock(t *testing.T) {
	t.Parallel()

	body := strings.Join([]string{
		`data: {"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","refusal":"I cannot "}}]}`,
		`data: {"model":"gpt-4o","choices":[{"index":0,"delta":{"refusal":"help with that."}}]}`,
		`data: {"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		`data: [DONE]`,
		"",
	}, "\n\n")

	reader := openaiapi.NewStream(io.NopCloser(strings.NewReader(body)))
	t.Cleanup(func() { _ = reader.Close() })

	var acc streamaccumulatorRefusal
	for {
		chunk, err := reader.Next()
		if err != nil {
			break
		}
		if refusal, ok := chunk.(*content.RefusalChunk); ok {
			acc.add(refusal)
		}
	}
	block := acc.block()
	if block == nil || block.Text != "I cannot help with that." {
		t.Errorf("streamed refusal block = %#v, want the whole text", block)
	}
	result, ok := reader.Result()
	if !ok {
		t.Fatal("Result() reported no terminal result")
	}
	if result.FinishReason != stream.FinishReasonStop {
		t.Errorf("FinishReason = %q, want %q (the wire value)", result.FinishReason, stream.FinishReasonStop)
	}
}

// streamaccumulatorRefusal is a local stand-in for
// core/content/streamaccumulator.Refusal, kept here so this codec's tests do
// not depend on the accumulator package for what is a two-line fold.
type streamaccumulatorRefusal struct {
	sb       strings.Builder
	received bool
}

func (a *streamaccumulatorRefusal) add(c *content.RefusalChunk) {
	a.received = true
	a.sb.WriteString(c.Text)
}

func (a *streamaccumulatorRefusal) block() *content.RefusalBlock {
	if !a.received {
		return nil
	}
	return &content.RefusalBlock{Text: a.sb.String()}
}

// TestEncodeRequest_RefusalBlockReplaysAsTheRefusalMember is the point of the
// block type: replayed assistant history must go back onto the wire as the
// `refusal` member ChatCompletionRequestAssistantMessage declares, NOT as
// assistant text. Re-sending it as text shows the model its own refusal quoted
// back as something it said.
func TestEncodeRequest_RefusalBlockReplaysAsTheRefusalMember(t *testing.T) {
	t.Parallel()

	raw, err := openaiapi.EncodeRequest(inference.Request{
		Model: model.Model{Name: "gpt-4o"},
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{
				&content.TextBlock{Text: "do the thing"},
			}}},
			&content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{
				&content.RefusalBlock{Text: "I cannot help with that."},
			}}},
		},
	}, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}

	var decoded struct {
		Messages []map[string]json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("unmarshal request: %v", err)
	}
	if len(decoded.Messages) != 2 {
		t.Fatalf("messages = %d, want 2", len(decoded.Messages))
	}
	assistant := decoded.Messages[1]
	got, ok := assistant["refusal"]
	if !ok {
		t.Fatalf("assistant message %v carries no `refusal` member", assistant)
	}
	if string(got) != `"I cannot help with that."` {
		t.Errorf("refusal = %s, want the refusal text verbatim", got)
	}
	if content, ok := assistant["content"]; ok && string(content) != `""` && string(content) != "null" {
		t.Errorf("content = %s, want the refusal NOT to leak into assistant text", content)
	}
}

// TestEncodeRequest_NoRefusalOmitsTheMember pins that an ordinary assistant
// turn does not start carrying an empty request-side refusal.
func TestEncodeRequest_NoRefusalOmitsTheMember(t *testing.T) {
	t.Parallel()

	raw, err := openaiapi.EncodeRequest(inference.Request{
		Model: model.Model{Name: "gpt-4o"},
		Messages: content.AgenticMessages{
			&content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{
				&content.TextBlock{Text: "here you go"},
			}}},
		},
	}, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	var decoded struct {
		Messages []map[string]json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("unmarshal request: %v", err)
	}
	if _, ok := decoded.Messages[0]["refusal"]; ok {
		t.Errorf("assistant message = %v, want no `refusal` member", decoded.Messages[0])
	}
}

// TestServerEncode_EmitsRequiredRefusalMember pins the encode direction of the
// same schema rule: ChatCompletionResponseMessage.required lists `refusal`, so
// a response this codec serves must carry the member — null when the turn was
// not a refusal — rather than omitting a key the schema demands.
func TestServerEncode_EmitsRequiredRefusalMember(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		blocks []content.Block
		want   string
	}{
		{name: "ordinary turn", blocks: []content.Block{&content.TextBlock{Text: "hello"}}, want: "null"},
		{name: "refused turn", blocks: []content.Block{&content.RefusalBlock{Text: "I cannot."}}, want: `"I cannot."`},
		{name: "empty refusal", blocks: []content.Block{&content.RefusalBlock{Text: ""}}, want: `""`},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			rec := httptest.NewRecorder()
			resp := &inference.Response{
				Message: &content.AIMessage{
					Message: content.Message{Role: content.RoleAssistant, Blocks: tt.blocks},
				},
				Model:        "gpt-test",
				FinishReason: stream.FinishReasonStop,
			}
			if err := (openaiapi.Codec{}).WriteResponse(rec, resp); err != nil {
				t.Fatalf("WriteResponse() error = %v", err)
			}

			var body struct {
				Choices []struct {
					Message map[string]json.RawMessage `json:"message"`
				} `json:"choices"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
				t.Fatalf("unmarshal encoded response: %v", err)
			}
			if len(body.Choices) != 1 {
				t.Fatalf("choices = %d, want 1", len(body.Choices))
			}
			raw, ok := body.Choices[0].Message["refusal"]
			if !ok {
				t.Fatal("message has no `refusal`; the schema marks it required")
			}
			if string(raw) != tt.want {
				t.Errorf("refusal = %s, want %s", raw, tt.want)
			}
		})
	}
}

// TestServerDecode_AssistantRefusalBecomesARefusalBlock pins the ingress side of
// the same member: a harness replaying a turn OpenAI refused sends a legal body,
// and it must come back as the refusal it is.
func TestServerDecode_AssistantRefusalBecomesARefusalBlock(t *testing.T) {
	t.Parallel()

	c, req := decodeReq(t, `{
		"model": "gpt-test",
		"messages": [
			{"role":"user","content":"do the thing"},
			{"role":"assistant","content":null,"refusal":"I'm sorry, I can't help with that."}
		]
	}`)
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	ai, ok := decoded.Request.Messages[1].(*content.AIMessage)
	if !ok {
		t.Fatalf("message 1 = %T, want *content.AIMessage", decoded.Request.Messages[1])
	}
	if len(ai.Blocks) != 1 {
		t.Fatalf("Blocks = %#v, want the refusal", ai.Blocks)
	}
	refusal, ok := ai.Blocks[0].(*content.RefusalBlock)
	if !ok || refusal.Text != "I'm sorry, I can't help with that." {
		t.Errorf("block 0 = %#v, want the refusal as a *content.RefusalBlock", ai.Blocks[0])
	}
}

// TestServerStream_RefusalChunkBecomesARefusalDelta pins the gateway's own
// streaming egress: the neutral RefusalChunk has a native home in
// ChatCompletionStreamResponseDelta's `refusal` member, so serving it as text
// (or failing the stream) would corrupt or kill a proxied refusal.
func TestServerStream_RefusalChunkBecomesARefusalDelta(t *testing.T) {
	t.Parallel()

	rec := httptest.NewRecorder()
	enc, err := (openaiapi.Codec{}).OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}
	if err := enc.WriteChunk(&content.RefusalChunk{Text: "I cannot"}); err != nil {
		t.Fatalf("WriteChunk() error = %v", err)
	}
	if err := enc.Finish(stream.StreamResult{Model: "gpt-test", FinishReason: stream.FinishReasonStop}); err != nil {
		t.Fatalf("Finish() error = %v", err)
	}

	if !strings.Contains(rec.Body.String(), `"refusal":"I cannot"`) {
		t.Errorf("stream body = %s, want a refusal delta", rec.Body.String())
	}
	if strings.Contains(rec.Body.String(), `"content":"I cannot"`) {
		t.Errorf("stream body = %s, want the refusal NOT served as assistant text", rec.Body.String())
	}
}
