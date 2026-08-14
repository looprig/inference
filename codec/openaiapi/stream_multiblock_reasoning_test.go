package openaiapi_test

import (
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/core/content/streamaccumulator"
	"github.com/looprig/inference/codec/openaiapi"
)

// singleReasoningStream is a real Chat Completions SSE body from a reasoning
// model. The dialect has exactly ONE reasoning channel — `delta.reasoning_content`
// is a bare string on the choice's delta, with no block index and no per-block
// signature — so every fragment belongs to the same reasoning block. That is why
// this codec pins ThinkingChunk.Index at 0: the format cannot express a second
// reasoning block, and inventing indexes for it would split one block in two.
const singleReasoningStream = "" +
	"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n" +
	"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"first \"}}]}\n\n" +
	"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"and second\"}}]}\n\n" +
	"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"}}]}\n\n" +
	"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
	"data: [DONE]\n\n"

// singleReasoningBody is the SAME completion delivered non-streaming.
const singleReasoningBody = `{"id":"c1","model":"deepseek-reasoner","choices":[{"index":0,"finish_reason":"stop",` +
	`"message":{"role":"assistant","reasoning_content":"first and second","content":"answer"}}]}`

func accumulateThinking(t *testing.T, body string) []content.ThinkingBlock {
	t.Helper()
	reader, err := (openaiapi.Codec{}).DecodeStream(&http.Response{Body: io.NopCloser(strings.NewReader(body))})
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

// TestDecodeStream_ReasoningDeltasFoldIntoOneBlock proves the single-stream
// contract holds through the real decode path: N reasoning deltas, one block.
// For this dialect a second block would be WRONG, so this is the assertion that
// keeps a future "index everything" change from splitting one reasoning stream.
func TestDecodeStream_ReasoningDeltasFoldIntoOneBlock(t *testing.T) {
	t.Parallel()

	blocks := accumulateThinking(t, singleReasoningStream)
	if len(blocks) != 1 {
		t.Fatalf("Blocks() = %d block(s) %#v, want exactly 1", len(blocks), blocks)
	}
	if blocks[0].Thinking != "first and second" {
		t.Errorf("block 0 thinking = %q, want %q", blocks[0].Thinking, "first and second")
	}
}

// TestStreamingMatchesNonStreamingReasoning holds both decode paths against each
// other for the same completion.
func TestStreamingMatchesNonStreamingReasoning(t *testing.T) {
	t.Parallel()

	streamed := accumulateThinking(t, singleReasoningStream)

	resp, err := openaiapi.DecodeResponse([]byte(singleReasoningBody))
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
		if streamed[i].Thinking != direct[i].Thinking || streamed[i].Signature != direct[i].Signature {
			t.Errorf("block %d: streamed %q/%q, non-streaming %q/%q", i,
				streamed[i].Thinking, streamed[i].Signature, direct[i].Thinking, direct[i].Signature)
		}
	}
}
