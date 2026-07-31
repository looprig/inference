package openairesponses_test

import (
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference/codec/openairesponses"
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
		if err == io.EOF {
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
		if err == io.EOF {
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
		if err == io.EOF {
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
