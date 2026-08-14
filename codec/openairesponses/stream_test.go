package openairesponses_test

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/core/content/streamaccumulator"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/openairesponses"
	model "github.com/looprig/inference/model"
	stream "github.com/looprig/inference/stream"
)

func sseBody(events ...string) io.ReadCloser {
	return io.NopCloser(strings.NewReader(strings.Join(events, "")))
}

func sseEvent(eventType, data string) string {
	return "event: " + eventType + "\ndata: " + data + "\n\n"
}

func TestDecodeStream_TextDeltasAndCompletion(t *testing.T) {
	t.Parallel()
	body := sseBody(
		sseEvent("response.created", `{"type":"response.created","response":{"id":"resp_1","status":"in_progress"}}`),
		sseEvent("response.output_item.added", `{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant"}}`),
		sseEvent("response.content_part.added", `{"type":"response.content_part.added","item_id":"msg_1","output_index":0,"content_index":0,"part":{"type":"output_text","text":""}}`),
		sseEvent("response.output_text.delta", `{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"hello "}`),
		sseEvent("response.output_text.delta", `{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"world"}`),
		sseEvent("response.content_part.done", `{"type":"response.content_part.done","item_id":"msg_1","output_index":0,"content_index":0,"part":{"type":"output_text","text":"hello world"}}`),
		sseEvent("response.output_item.done", `{"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant"}}`),
		sseEvent("response.completed", `{"type":"response.completed","response":{"id":"resp_1","status":"completed","model":"gpt-test","output":[],"usage":{"input_tokens":5,"output_tokens":2,"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":0}}}}`),
	)
	resp := &http.Response{Body: body}
	reader, err := (openairesponses.Codec{}).DecodeStream(resp)
	if err != nil {
		t.Fatalf("DecodeStream() error = %v", err)
	}
	defer reader.Close()

	var texts []string
	for {
		chunk, err := reader.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Next() error = %v", err)
		}
		if tc, ok := chunk.(*content.TextChunk); ok {
			texts = append(texts, tc.Text)
		}
	}
	if got := strings.Join(texts, ""); got != "hello world" {
		t.Errorf("accumulated text = %q, want %q", got, "hello world")
	}

	result, ok := reader.Result()
	if !ok {
		t.Fatal("Result() ok = false, want true")
	}
	if result.Model != "gpt-test" {
		t.Errorf("Result.Model = %q", result.Model)
	}
	if result.FinishReason != stream.FinishReasonStop {
		t.Errorf("Result.FinishReason = %v, want Stop", result.FinishReason)
	}
	if result.Usage == nil || result.Usage.InputTokens != 5 || result.Usage.OutputTokens != 2 {
		t.Errorf("Result.Usage = %#v", result.Usage)
	}
}

func TestDecodeStream_ToolCallArgumentIndexes(t *testing.T) {
	t.Parallel()
	body := sseBody(
		sseEvent("response.output_item.added", `{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather"}}`),
		sseEvent("response.function_call_arguments.delta", `{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,"delta":"{\"city\":"}`),
		sseEvent("response.function_call_arguments.delta", `{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,"delta":"\"nyc\"}"}`),
		sseEvent("response.output_item.added", `{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc_2","call_id":"call_2","name":"get_time"}}`),
		sseEvent("response.function_call_arguments.delta", `{"type":"response.function_call_arguments.delta","item_id":"fc_2","output_index":1,"delta":"{}"}`),
		sseEvent("response.completed", `{"type":"response.completed","response":{"id":"resp_1","status":"completed","model":"gpt-test","output":[{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"nyc\"}"},{"type":"function_call","call_id":"call_2","name":"get_time","arguments":"{}"}]}}`),
	)
	resp := &http.Response{Body: body}
	reader, err := (openairesponses.Codec{}).DecodeStream(resp)
	if err != nil {
		t.Fatalf("DecodeStream() error = %v", err)
	}
	defer reader.Close()

	var chunks []*content.ToolUseChunk
	for {
		chunk, err := reader.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Next() error = %v", err)
		}
		if tc, ok := chunk.(*content.ToolUseChunk); ok {
			chunks = append(chunks, tc)
		}
	}
	if len(chunks) < 4 {
		t.Fatalf("got %d tool-use chunks, want at least 4", len(chunks))
	}
	// First seed chunk for index 0 carries id/name.
	if chunks[0].Index != 0 || chunks[0].ID != "call_1" || chunks[0].Name != "get_weather" {
		t.Errorf("chunks[0] = %#v", chunks[0])
	}
	// Seed chunk for index 1.
	var seenIndex1Seed bool
	for _, c := range chunks {
		if c.Index == 1 && c.ID == "call_2" {
			seenIndex1Seed = true
		}
	}
	if !seenIndex1Seed {
		t.Errorf("no seed chunk observed for index 1: %#v", chunks)
	}
	result, ok := reader.Result()
	if !ok || result.FinishReason != stream.FinishReasonToolUse {
		t.Errorf("Result = %#v, ok=%v, want FinishReasonToolUse", result, ok)
	}
}

func TestDecodeStream_ReasoningSummaryDeltas(t *testing.T) {
	t.Parallel()
	body := sseBody(
		sseEvent("response.output_item.added", `{"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1"}}`),
		sseEvent("response.reasoning_summary_text.delta", `{"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"delta":"step "}`),
		sseEvent("response.reasoning_summary_text.delta", `{"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"delta":"one"}`),
		sseEvent("response.completed", `{"type":"response.completed","response":{"id":"resp_1","status":"completed","model":"gpt-test","output":[]}}`),
	)
	resp := &http.Response{Body: body}
	reader, err := (openairesponses.Codec{}).DecodeStream(resp)
	if err != nil {
		t.Fatalf("DecodeStream() error = %v", err)
	}
	defer reader.Close()

	var thinking []string
	for {
		chunk, err := reader.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Next() error = %v", err)
		}
		if tc, ok := chunk.(*content.ThinkingChunk); ok {
			thinking = append(thinking, tc.Thinking)
		}
	}
	if got := strings.Join(thinking, ""); got != "step one" {
		t.Errorf("accumulated thinking = %q, want %q", got, "step one")
	}
}

func TestDecodeStream_ReasoningEncryptedContentIsReplayable(t *testing.T) {
	t.Parallel()
	body := sseBody(
		sseEvent("response.reasoning_summary_text.delta", `{"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"summary"}`),
		sseEvent("response.output_item.done", `{"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1","encrypted_content":"opaque-blob"}}`),
		sseEvent("response.completed", `{"type":"response.completed","response":{"status":"completed"}}`),
	)
	reader, err := (openairesponses.Codec{}).DecodeStream(&http.Response{Body: body})
	if err != nil {
		t.Fatalf("DecodeStream() error = %v", err)
	}
	defer reader.Close()
	var stateChunk *content.ThinkingChunk
	var thinking streamaccumulator.Thinking
	for {
		chunk, err := reader.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Next() error = %v", err)
		}
		if thinking, ok := chunk.(*content.ThinkingChunk); ok && len(thinking.ProviderState) > 0 {
			stateChunk = thinking
		}
		if delta, ok := chunk.(*content.ThinkingChunk); ok {
			thinking.Add(delta)
		}
	}
	// The state chunk carries the whole replayable reasoning item — the id
	// ReasoningItem requires as well as the encrypted content. (This
	// assertion previously required the bare `"opaque-blob"` JSON string.)
	if stateChunk == nil || stateChunk.ProviderStateFormat != "openai-responses" {
		t.Fatalf("state chunk = %#v, want tagged reasoning state", stateChunk)
	}
	if got, want := string(stateChunk.ProviderState), `{"id":"rs_1","encrypted_content":"opaque-blob"}`; got != want {
		t.Fatalf("ProviderState = %s, want %s", got, want)
	}
	blocks := thinking.Blocks()
	if len(blocks) != 1 {
		t.Fatalf("Blocks() = %#v, want exactly one reasoning block", blocks)
	}
	block := &blocks[0]
	raw, err := openairesponses.EncodeRequest(inference.Request{
		Model: model.Model{Name: "gpt-test", Caps: model.Capabilities{Thinking: true}},
		Messages: content.AgenticMessages{&content.AIMessage{Message: content.Message{
			Role: content.RoleAssistant, Blocks: []content.Block{block},
		}}},
	}, false)
	if err != nil {
		t.Fatalf("EncodeRequest(replay) error = %v", err)
	}
	var request struct {
		Input []map[string]json.RawMessage `json:"input"`
	}
	if err := json.Unmarshal(raw, &request); err != nil {
		t.Fatalf("replay json: %v", err)
	}
	var replayed string
	if err := json.Unmarshal(request.Input[0]["encrypted_content"], &replayed); err != nil || replayed != "opaque-blob" {
		t.Fatalf("replayed encrypted_content = %q, err=%v; body=%s", replayed, err, raw)
	}
	var replayedID string
	if err := json.Unmarshal(request.Input[0]["id"], &replayedID); err != nil || replayedID != "rs_1" {
		t.Fatalf("replayed id = %q, err=%v; body=%s", replayedID, err, raw)
	}
}

func TestDecodeStream_IncompleteIsTerminal(t *testing.T) {
	t.Parallel()
	body := sseBody(sseEvent("response.incomplete", `{"type":"response.incomplete","response":{"id":"resp_1","status":"incomplete","model":"gpt-test","incomplete_details":{"reason":"max_output_tokens"},"usage":{"input_tokens":3,"output_tokens":4}}}`))
	reader, err := (openairesponses.Codec{}).DecodeStream(&http.Response{Body: body})
	if err != nil {
		t.Fatalf("DecodeStream() error = %v", err)
	}
	defer reader.Close()
	for {
		_, err := reader.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Next() error = %v", err)
		}
	}
	result, ok := reader.Result()
	if !ok {
		t.Fatal("Result() ok = false, want terminal incomplete metadata")
	}
	if result.FinishReason != stream.FinishReasonLength || result.Model != "gpt-test" {
		t.Fatalf("Result() = %#v, want length/gpt-test", result)
	}
}

// TestDecodeStream_EOFWithoutTerminalFails locks the terminal gate. The
// Responses stream has no [DONE] sentinel; the ResponseStreamEvent union's
// terminal members are response.completed ("Emitted when the model response is
// complete"), response.incomplete ("emitted when a response finishes as
// incomplete") and the two failure events, response.failed and the top-level
// `error` event — the last two already abort the stream with a *StreamAPIError.
// A body that reaches EOF without any of them is a truncated answer, so
// reporting it as a completed turn presents partial content as complete.
func TestDecodeStream_EOFWithoutTerminalFails(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		body io.ReadCloser
	}{
		{
			name: "text deltas cut off before response.completed",
			body: sseBody(
				sseEvent("response.created", `{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response","status":"in_progress","model":"gpt-test","output":[]}}`),
				sseEvent("response.in_progress", `{"type":"response.in_progress","sequence_number":1,"response":{"id":"resp_1","object":"response","status":"in_progress","model":"gpt-test","output":[]}}`),
				sseEvent("response.output_item.added", `{"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"type":"message","id":"msg_1","status":"in_progress","role":"assistant","content":[]}}`),
				sseEvent("response.content_part.added", `{"type":"response.content_part.added","sequence_number":3,"item_id":"msg_1","output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}`),
				sseEvent("response.output_text.delta", `{"type":"response.output_text.delta","sequence_number":4,"item_id":"msg_1","output_index":0,"content_index":0,"delta":"Hello"}`),
			),
		},
		{
			name: "item completed but response never did",
			body: sseBody(
				sseEvent("response.output_text.done", `{"type":"response.output_text.done","sequence_number":5,"item_id":"msg_1","output_index":0,"content_index":0,"text":"Hello"}`),
				sseEvent("response.output_item.done", `{"type":"response.output_item.done","sequence_number":6,"output_index":0,"item":{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hello","annotations":[]}]}}`),
			),
		},
		{
			name: "empty body",
			body: sseBody(),
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			reader, err := (openairesponses.Codec{}).DecodeStream(&http.Response{Body: tc.body})
			if err != nil {
				t.Fatalf("DecodeStream() error = %v", err)
			}
			defer reader.Close()

			for err == nil {
				_, err = reader.Next()
			}
			if errors.Is(err, io.EOF) {
				t.Fatal("stream ended with a clean EOF; no terminal response event was ever seen")
			}
			var streamErr *openairesponses.StreamDecodeError
			if !errors.As(err, &streamErr) {
				t.Fatalf("Next() error = %T %v, want *StreamDecodeError", err, err)
			}
			if got, ok := reader.Result(); ok {
				t.Errorf("Result() = %+v, true for a stream that never terminated", got)
			}
		})
	}
}

func TestDecodeStream_MalformedJSONIsErrorButUnknownValidEventIsIgnored(t *testing.T) {
	t.Parallel()
	body := sseBody(
		sseEvent("response.future", `{"type":"response.future","new_field":true}`),
		sseEvent("response.output_text.delta", `{"type":"response.output_text.delta","delta":`),
	)
	reader, err := (openairesponses.Codec{}).DecodeStream(&http.Response{Body: body})
	if err != nil {
		t.Fatalf("DecodeStream() error = %v", err)
	}
	defer reader.Close()
	_, err = reader.Next()
	if err == nil || errors.Is(err, io.EOF) {
		t.Fatalf("Next() error = %v, want malformed-event error", err)
	}
}

func TestDecodeStream_ResponseFailed(t *testing.T) {
	t.Parallel()
	body := sseBody(
		sseEvent("response.created", `{"type":"response.created","response":{"id":"resp_1","status":"in_progress"}}`),
		sseEvent("response.failed", `{"type":"response.failed","response":{"id":"resp_1","status":"failed","error":{"code":"server_error","message":"boom"}}}`),
	)
	resp := &http.Response{Body: body}
	reader, err := (openairesponses.Codec{}).DecodeStream(resp)
	if err != nil {
		t.Fatalf("DecodeStream() error = %v", err)
	}
	defer reader.Close()

	var lastErr error
	for {
		_, err := reader.Next()
		if err != nil {
			lastErr = err
			break
		}
	}
	if lastErr == nil {
		t.Fatal("expected an error from response.failed")
	}
	var apiErr *openairesponses.StreamAPIError
	if !asStreamAPIError(lastErr, &apiErr) {
		t.Fatalf("error = %v (%T), want *StreamAPIError", lastErr, lastErr)
	}
	if apiErr.Message != "boom" {
		t.Errorf("Message = %q", apiErr.Message)
	}
}

func asStreamAPIError(err error, target **openairesponses.StreamAPIError) bool {
	for err != nil {
		if e, ok := err.(*openairesponses.StreamAPIError); ok {
			*target = e
			return true
		}
		u, ok := err.(interface{ Unwrap() error })
		if !ok {
			return false
		}
		err = u.Unwrap()
	}
	return false
}

// TestDecodeStream_ResponseErrorEvent covers the spec's ResponseErrorEvent —
// a first-class `type:"error"` stream event (required: type, code, message,
// param, sequence_number) that is NOT wrapped in a `response` object and so is
// not response.failed. Left unhandled it is skipped as an unknown event type,
// turning a mid-stream provider failure into a silently truncated success.
func TestDecodeStream_ResponseErrorEvent(t *testing.T) {
	t.Parallel()
	body := sseBody(
		sseEvent("response.created", `{"type":"response.created","response":{"id":"resp_1","status":"in_progress"}}`),
		sseEvent("response.output_text.delta", `{"type":"response.output_text.delta","output_index":0,"delta":"par"}`),
		sseEvent("error", `{"type":"error","code":"ERR_SOMETHING","message":"Something went wrong","param":null,"sequence_number":1}`),
	)
	resp := &http.Response{Body: body}
	reader, err := (openairesponses.Codec{}).DecodeStream(resp)
	if err != nil {
		t.Fatalf("DecodeStream() error = %v", err)
	}
	defer reader.Close()

	var lastErr error
	for {
		_, err := reader.Next()
		if err != nil {
			lastErr = err
			break
		}
	}
	if errors.Is(lastErr, io.EOF) {
		t.Fatal("stream ended cleanly, want the error event surfaced")
	}
	var apiErr *openairesponses.StreamAPIError
	if !asStreamAPIError(lastErr, &apiErr) {
		t.Fatalf("error = %v (%T), want *StreamAPIError", lastErr, lastErr)
	}
	if apiErr.Code != "ERR_SOMETHING" {
		t.Errorf("Code = %q, want ERR_SOMETHING", apiErr.Code)
	}
	if apiErr.Message != "Something went wrong" {
		t.Errorf("Message = %q, want %q", apiErr.Message, "Something went wrong")
	}
}

// TestDecodeStream_ResponseErrorEventNullCode confirms the spec's nullable
// `code` still produces the error, carrying only the message.
func TestDecodeStream_ResponseErrorEventNullCode(t *testing.T) {
	t.Parallel()
	body := sseBody(
		sseEvent("error", `{"type":"error","code":null,"message":"Something went wrong","param":null,"sequence_number":1}`),
	)
	resp := &http.Response{Body: body}
	reader, err := (openairesponses.Codec{}).DecodeStream(resp)
	if err != nil {
		t.Fatalf("DecodeStream() error = %v", err)
	}
	defer reader.Close()

	_, err = reader.Next()
	var apiErr *openairesponses.StreamAPIError
	if !asStreamAPIError(err, &apiErr) {
		t.Fatalf("error = %v (%T), want *StreamAPIError", err, err)
	}
	if apiErr.Code != "" {
		t.Errorf("Code = %q, want empty", apiErr.Code)
	}
	if apiErr.Message != "Something went wrong" {
		t.Errorf("Message = %q", apiErr.Message)
	}
}
