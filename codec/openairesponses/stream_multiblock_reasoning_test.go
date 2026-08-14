package openairesponses_test

import (
	"errors"
	"io"
	"net/http"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/core/content/streamaccumulator"
	"github.com/looprig/inference/codec/openairesponses"
)

// multiBlockReasoningStream is a real Responses stream for a reasoning model
// that thinks, answers, and thinks again: two `reasoning` output items at
// output_index 0 and 2 with a `message` item between them at output_index 1.
// Each reasoning item carries its own id and encrypted_content — the
// continuation state a follow-up request must replay item-for-item.
func multiBlockReasoningStream() io.ReadCloser {
	return sseBody(
		sseEvent("response.created", `{"type":"response.created","response":{"id":"resp_1","status":"in_progress"}}`),
		sseEvent("response.output_item.added", `{"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1"}}`),
		sseEvent("response.reasoning_summary_text.delta", `{"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"summary_index":0,"delta":"first"}`),
		sseEvent("response.output_item.done", `{"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1","encrypted_content":"BLOB-ONE","summary":[{"type":"summary_text","text":"first"}]}}`),
		sseEvent("response.output_item.added", `{"type":"response.output_item.added","output_index":1,"item":{"type":"message","id":"msg_1","role":"assistant"}}`),
		sseEvent("response.output_text.delta", `{"type":"response.output_text.delta","item_id":"msg_1","output_index":1,"content_index":0,"delta":"answer"}`),
		sseEvent("response.output_item.done", `{"type":"response.output_item.done","output_index":1,"item":{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"answer"}]}}`),
		sseEvent("response.output_item.added", `{"type":"response.output_item.added","output_index":2,"item":{"type":"reasoning","id":"rs_2"}}`),
		sseEvent("response.reasoning_summary_text.delta", `{"type":"response.reasoning_summary_text.delta","item_id":"rs_2","output_index":2,"summary_index":0,"delta":"second"}`),
		sseEvent("response.output_item.done", `{"type":"response.output_item.done","output_index":2,"item":{"type":"reasoning","id":"rs_2","encrypted_content":"BLOB-TWO","summary":[{"type":"summary_text","text":"second"}]}}`),
		sseEvent("response.completed", `{"type":"response.completed","response":{"id":"resp_1","model":"o-test","status":"completed"}}`),
	)
}

// multiBlockReasoningBody is the SAME response delivered non-streaming.
const multiBlockReasoningBody = `{"id":"resp_1","model":"o-test","status":"completed","output":[` +
	`{"type":"reasoning","id":"rs_1","encrypted_content":"BLOB-ONE","summary":[{"type":"summary_text","text":"first"}]},` +
	`{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"answer"}]},` +
	`{"type":"reasoning","id":"rs_2","encrypted_content":"BLOB-TWO","summary":[{"type":"summary_text","text":"second"}]}]}`

func accumulateThinking(t *testing.T, body io.ReadCloser) []content.ThinkingBlock {
	t.Helper()
	reader, err := (openairesponses.Codec{}).DecodeStream(&http.Response{Body: body})
	if err != nil {
		t.Fatalf("DecodeStream() error = %v", err)
	}
	defer reader.Close()
	var acc streamaccumulator.Thinking
	for {
		chunk, err := reader.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Next() error = %v", err)
		}
		if tc, ok := chunk.(*content.ThinkingChunk); ok {
			acc.Add(tc)
		}
	}
	return acc.Blocks()
}

// TestDecodeStream_MultipleReasoningItemsKeepTheirOwnState drives a real
// multi-item SSE fixture through the codec into the accumulator. Two reasoning
// items must reconstruct as two blocks, each holding the encrypted state OpenAI
// minted for it; folding them binds the last blob to both summaries and the
// replayed conversation no longer matches what the model produced.
func TestDecodeStream_MultipleReasoningItemsKeepTheirOwnState(t *testing.T) {
	t.Parallel()

	blocks := accumulateThinking(t, multiBlockReasoningStream())
	if len(blocks) != 2 {
		t.Fatalf("Blocks() = %d block(s) %#v, want 2", len(blocks), blocks)
	}
	if blocks[0].Thinking != "first" {
		t.Errorf("block 0 thinking = %q, want %q", blocks[0].Thinking, "first")
	}
	if got, want := string(blocks[0].ProviderState), `{"id":"rs_1","encrypted_content":"BLOB-ONE"}`; got != want {
		t.Errorf("block 0 provider state = %s, want %s", got, want)
	}
	if blocks[0].ProviderStateFormat != "openai-responses" {
		t.Errorf("block 0 provider state format = %q, want openai-responses", blocks[0].ProviderStateFormat)
	}
	if blocks[1].Thinking != "second" {
		t.Errorf("block 1 thinking = %q, want %q", blocks[1].Thinking, "second")
	}
	if got, want := string(blocks[1].ProviderState), `{"id":"rs_2","encrypted_content":"BLOB-TWO"}`; got != want {
		t.Errorf("block 1 provider state = %s, want %s", got, want)
	}
	if blocks[1].ProviderStateFormat != "openai-responses" {
		t.Errorf("block 1 provider state format = %q, want openai-responses", blocks[1].ProviderStateFormat)
	}
}

// TestStreamingMatchesNonStreamingReasoning holds the two decode paths against
// each other for the same response, as inference/CLAUDE.md requires.
func TestStreamingMatchesNonStreamingReasoning(t *testing.T) {
	t.Parallel()

	streamed := accumulateThinking(t, multiBlockReasoningStream())

	resp, err := openairesponses.DecodeResponse([]byte(multiBlockReasoningBody))
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	var direct []content.ThinkingBlock
	for _, b := range resp.Message.Blocks {
		if tb, ok := b.(*content.ThinkingBlock); ok {
			direct = append(direct, *tb)
		}
	}

	if len(streamed) != len(direct) {
		t.Fatalf("streamed %d reasoning block(s), non-streaming %d: %#v vs %#v", len(streamed), len(direct), streamed, direct)
	}
	for i := range direct {
		if streamed[i].Thinking != direct[i].Thinking {
			t.Errorf("block %d thinking: streamed %q, non-streaming %q", i, streamed[i].Thinking, direct[i].Thinking)
		}
		if string(streamed[i].ProviderState) != string(direct[i].ProviderState) ||
			streamed[i].ProviderStateFormat != direct[i].ProviderStateFormat {
			t.Errorf("block %d provider state: streamed %s/%q, non-streaming %s/%q", i,
				streamed[i].ProviderState, streamed[i].ProviderStateFormat,
				direct[i].ProviderState, direct[i].ProviderStateFormat)
		}
	}
}
