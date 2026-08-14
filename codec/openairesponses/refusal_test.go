package openairesponses_test

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/openairesponses"
	model "github.com/looprig/inference/model"
	stream "github.com/looprig/inference/stream"
)

// Refusal coverage.
//
// The Responses output content union (OutputContent) declares a `refusal`
// part alongside output_text, and the stream declares response.refusal.delta /
// response.refusal.done.
//
// These tests replaced an earlier set that asserted the INTERIM mapping — the
// refusal part decoding to a *content.TextBlock and the turn's finish reason
// being overridden to content_filter — which predated content.RefusalBlock.

func TestDecodeResponse_RefusalPartBecomesRefusalBlock(t *testing.T) {
	t.Parallel()

	resp, err := openairesponses.DecodeResponse([]byte(`{
		"id": "resp_1",
		"status": "completed",
		"model": "gpt-5",
		"output": [{
			"id": "msg_1",
			"type": "message",
			"role": "assistant",
			"status": "completed",
			"content": [{"type":"refusal","refusal":"I cannot help with that."}]
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
	// status is "completed" — a refusal is not a truncation and not a content
	// filter — and the block now carries the "declined" signal per block, so
	// the finish reason reports the wire's own outcome.
	if resp.FinishReason != stream.FinishReasonStop {
		t.Errorf("FinishReason = %q, want %q (status:\"completed\")", resp.FinishReason, stream.FinishReasonStop)
	}
}

// TestDecodeResponse_EmptyRefusalPartStillYieldsABlock pins that presence, not
// content, is the signal. RefusalContent.required is ["type","refusal"], so a
// structurally present part with empty text is a refusal with no explanation —
// not the absence of one.
func TestDecodeResponse_EmptyRefusalPartStillYieldsABlock(t *testing.T) {
	t.Parallel()

	resp, err := openairesponses.DecodeResponse([]byte(
		`{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"refusal","refusal":""}]}]}`))
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if len(resp.Message.Blocks) != 1 {
		t.Fatalf("Blocks = %#v, want one refusal block", resp.Message.Blocks)
	}
	if _, ok := resp.Message.Blocks[0].(*content.RefusalBlock); !ok {
		t.Errorf("block 0 = %T, want *content.RefusalBlock", resp.Message.Blocks[0])
	}
}

// TestDecodeResponse_RefusalBesideAToolCallKeepsBoth pins that the refusal no
// longer suppresses the tool-use finish reason. The refusal travels as its own
// block, so the turn-level reason is free to report the pending tool call the
// caller must actually act on.
func TestDecodeResponse_RefusalBesideAToolCallKeepsBoth(t *testing.T) {
	t.Parallel()

	resp, err := openairesponses.DecodeResponse([]byte(`{
		"status": "completed",
		"output": [
			{"type":"message","role":"assistant","content":[{"type":"refusal","refusal":"no"}]},
			{"type":"function_call","call_id":"call_1","name":"f","arguments":"{}"}
		]
	}`))
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if resp.FinishReason != stream.FinishReasonToolUse {
		t.Errorf("FinishReason = %q, want %q", resp.FinishReason, stream.FinishReasonToolUse)
	}
	var sawRefusal, sawTool bool
	for _, b := range resp.Message.Blocks {
		switch b.(type) {
		case *content.RefusalBlock:
			sawRefusal = true
		case *content.ToolUseBlock:
			sawTool = true
		}
	}
	if !sawRefusal || !sawTool {
		t.Errorf("Blocks = %#v, want both the refusal and the tool call", resp.Message.Blocks)
	}
}

func TestDecodeEvent_RefusalDeltaBecomesRefusalChunk(t *testing.T) {
	t.Parallel()

	chunks, err := openairesponses.Codec{}.DecodeEvent([]byte(
		`{"type":"response.refusal.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"I cannot","sequence_number":1}`))
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

// TestDecodeEvent_RefusalDoneEmitsNoChunk pins that response.refusal.done is a
// finalization echo of text already delivered by the deltas, so emitting a
// chunk for it would duplicate the refusal — the same reason
// response.output_text.done yields nothing.
func TestDecodeEvent_RefusalDoneEmitsNoChunk(t *testing.T) {
	t.Parallel()

	chunks, err := openairesponses.Codec{}.DecodeEvent([]byte(
		`{"type":"response.refusal.done","item_id":"msg_1","output_index":0,"content_index":0,"refusal":"I cannot","sequence_number":2}`))
	if err != nil {
		t.Fatalf("DecodeEvent() error = %v", err)
	}
	if len(chunks) != 0 {
		t.Errorf("chunks = %#v, want none", chunks)
	}
}

func TestStream_RefusalDeltasFoldIntoARefusalBlock(t *testing.T) {
	t.Parallel()

	body := sseBody(
		sseEvent("response.refusal.delta", `{"type":"response.refusal.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"I cannot ","sequence_number":1}`),
		sseEvent("response.refusal.delta", `{"type":"response.refusal.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"help with that.","sequence_number":2}`),
		sseEvent("response.refusal.done", `{"type":"response.refusal.done","item_id":"msg_1","output_index":0,"content_index":0,"refusal":"I cannot help with that.","sequence_number":3}`),
		sseEvent("response.completed", `{"type":"response.completed","response":{"id":"resp_1","status":"completed","model":"gpt-5","output":[]}}`),
	)
	reader, err := (openairesponses.Codec{}).DecodeStream(&http.Response{Body: body})
	if err != nil {
		t.Fatalf("DecodeStream() error = %v", err)
	}
	t.Cleanup(func() { _ = reader.Close() })

	var refusal strings.Builder
	var received bool
	for {
		chunk, err := reader.Next()
		if err != nil {
			break
		}
		if c, ok := chunk.(*content.RefusalChunk); ok {
			received = true
			refusal.WriteString(c.Text)
		}
	}
	if !received || refusal.String() != "I cannot help with that." {
		t.Errorf("streamed refusal = %q (received=%v), want the whole text exactly once", refusal.String(), received)
	}
	result, ok := reader.Result()
	if !ok {
		t.Fatal("Result() reported no terminal result")
	}
	if result.FinishReason != stream.FinishReasonStop {
		t.Errorf("FinishReason = %q, want %q (status:\"completed\")", result.FinishReason, stream.FinishReasonStop)
	}
}

// TestEncodeRequest_RefusalBlockFailsClosedOnReplay records a MEASURED
// limitation of the Responses request schema rather than a design preference.
//
// A `refusal` part is only legal inside an OutputMessage, whose required set is
// ["id","type","role","content","status"]. Replayed assistant history carries
// no message id — no neutral block holds one — and EasyInputMessage, the only
// id-free assistant form, takes InputContent
// (input_text|input_image|input_file), which has no refusal member. Feeding
// both shapes to the provider's own request schema confirms it: see
// TestResponsesRefusalReplayHasNoLegalIDFreeWireForm in
// llm/providers/openai/refusal_request_test.go.
//
// So the choices are: emit a body the schema rejects, drop the refusal
// silently, or re-send it as assistant text — the very defect the block type
// removes. It fails closed instead, naming the limitation.
func TestEncodeRequest_RefusalBlockFailsClosedOnReplay(t *testing.T) {
	t.Parallel()

	_, err := openairesponses.EncodeRequest(inference.Request{
		Model: model.Model{Name: "gpt-5"},
		Messages: content.AgenticMessages{
			&content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{
				&content.RefusalBlock{Text: "I cannot help with that."},
			}}},
		},
	}, false)

	var unsupported *openairesponses.UnsupportedBlockError
	if !errors.As(err, &unsupported) {
		t.Fatalf("EncodeRequest() error = %v, want *openairesponses.UnsupportedBlockError", err)
	}
	if !strings.Contains(unsupported.Reason, "id") {
		t.Errorf("Reason = %q, want it to name the missing message id", unsupported.Reason)
	}
}

// TestServerEncode_RefusalBlockBecomesARefusalPart is the direction that CAN
// carry it: when this process is the authority producing the response it
// synthesizes the OutputMessage id, so the refusal goes out as the native
// `refusal` content part rather than as output_text.
func TestServerEncode_RefusalBlockBecomesARefusalPart(t *testing.T) {
	t.Parallel()

	rec := httptest.NewRecorder()
	resp := &inference.Response{
		Message: &content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{
			&content.RefusalBlock{Text: "I cannot help with that."},
		}}},
		Model:        "gpt-5",
		FinishReason: stream.FinishReasonStop,
	}
	if err := (openairesponses.Codec{}).WriteResponse(rec, resp); err != nil {
		t.Fatalf("WriteResponse() error = %v", err)
	}

	var body struct {
		Output []struct {
			Type    string `json:"type"`
			ID      string `json:"id"`
			Status  string `json:"status"`
			Content []struct {
				Type    string `json:"type"`
				Text    string `json:"text"`
				Refusal string `json:"refusal"`
			} `json:"content"`
		} `json:"output"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("unmarshal encoded response: %v", err)
	}
	if len(body.Output) != 1 || len(body.Output[0].Content) != 1 {
		t.Fatalf("output = %s, want one message item with one part", rec.Body.String())
	}
	part := body.Output[0].Content[0]
	if part.Type != "refusal" || part.Refusal != "I cannot help with that." {
		t.Errorf("content part = %+v, want a refusal part carrying the text", part)
	}
	if body.Output[0].ID == "" || body.Output[0].Status == "" {
		t.Errorf("output item = %+v, want the OutputMessage id and status the schema requires", body.Output[0])
	}
}

// TestServerDecode_AssistantRefusalPartBecomesARefusalBlock pins the ingress
// side: a harness replaying a turn OpenAI refused sends the assistant message
// back with its refusal content part, and it must come back as the refusal it
// is rather than as assistant prose.
func TestServerDecode_AssistantRefusalPartBecomesARefusalBlock(t *testing.T) {
	t.Parallel()

	c, req := decodeReq(t, `{
		"model": "gpt-test",
		"input": [
			{"type":"message","role":"user","content":[{"type":"input_text","text":"do the thing"}]},
			{"type":"message","role":"assistant","content":[{"type":"refusal","refusal":"I'm sorry, I can't help with that."}]}
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

// TestServerStream_RefusalChunkBecomesRefusalEvents pins the gateway's own
// streaming egress: the refusal channel has native events, so a proxied refusal
// keeps them instead of being re-served as output_text or killing the stream.
func TestServerStream_RefusalChunkBecomesRefusalEvents(t *testing.T) {
	t.Parallel()

	rec := httptest.NewRecorder()
	enc, err := (openairesponses.Codec{}).OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}
	if err := enc.WriteChunk(&content.RefusalChunk{Text: "I cannot "}); err != nil {
		t.Fatalf("WriteChunk() error = %v", err)
	}
	if err := enc.WriteChunk(&content.RefusalChunk{Text: "help."}); err != nil {
		t.Fatalf("WriteChunk() error = %v", err)
	}
	if err := enc.Finish(stream.StreamResult{Model: "gpt-5", FinishReason: stream.FinishReasonStop}); err != nil {
		t.Fatalf("Finish() error = %v", err)
	}

	got := rec.Body.String()
	for _, want := range []string{
		"event: response.refusal.delta",
		`"delta":"I cannot "`,
		"event: response.refusal.done",
		`"refusal":"I cannot help."`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("stream body missing %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "response.output_text.delta") {
		t.Errorf("stream body served the refusal as output_text:\n%s", got)
	}
}
