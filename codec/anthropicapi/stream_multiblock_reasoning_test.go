package anthropicapi_test

import (
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/core/content/streamaccumulator"
	"github.com/looprig/inference/codec/anthropicapi"
)

// multiBlockReasoningStream is a reference-shaped Anthropic interleaved-thinking
// stream: two signed thinking blocks with a redacted_thinking block between
// them, at wire content-block indexes 0, 1 and 2. It is the real SSE the
// Messages API emits when extended thinking is enabled and the model reasons
// again after a tool result.
const multiBlockReasoningStream = "" +
	"data: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-test\",\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n" +
	"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}\n\n" +
	"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"first\"}}\n\n" +
	"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"SIG-ONE\"}}\n\n" +
	"data: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
	"data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"redacted_thinking\",\"data\":\"REDACTED\"}}\n\n" +
	"data: {\"type\":\"content_block_stop\",\"index\":1}\n\n" +
	"data: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}\n\n" +
	"data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"second\"}}\n\n" +
	"data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"SIG-TWO\"}}\n\n" +
	"data: {\"type\":\"content_block_stop\",\"index\":2}\n\n" +
	"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":4}}\n\n" +
	"data: {\"type\":\"message_stop\"}\n\n"

// multiBlockReasoningBody is the SAME response delivered non-streaming. The two
// paths must reconstruct identical reasoning blocks.
const multiBlockReasoningBody = `{"type":"message","model":"claude-test","stop_reason":"end_turn",` +
	`"usage":{"input_tokens":1,"output_tokens":4},"content":[` +
	`{"type":"thinking","thinking":"first","signature":"SIG-ONE"},` +
	`{"type":"redacted_thinking","data":"REDACTED"},` +
	`{"type":"thinking","thinking":"second","signature":"SIG-TWO"}]}`

// accumulateThinking drives a real SSE body through the codec's stream decoder
// into the shared accumulator, exactly as the loop runtime does.
func accumulateThinking(t *testing.T, body string) []content.ThinkingBlock {
	t.Helper()
	reader, err := (anthropicapi.Codec{}).DecodeStream(&http.Response{Body: io.NopCloser(strings.NewReader(body))})
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

// TestDecodeStream_MultipleReasoningBlocksKeepTheirOwnSignatures pins the
// interleaved-thinking invariant against a real wire fixture: three reasoning
// blocks in, three reasoning blocks out, each signature bound to the block that
// produced it. Folding them together rebinds the LAST signature to the
// concatenation of every block's text, which Anthropic rejects on the next turn.
func TestDecodeStream_MultipleReasoningBlocksKeepTheirOwnSignatures(t *testing.T) {
	t.Parallel()

	blocks := accumulateThinking(t, multiBlockReasoningStream)
	if len(blocks) != 3 {
		t.Fatalf("Blocks() = %d block(s) %#v, want 3", len(blocks), blocks)
	}
	if blocks[0].Thinking != "first" || blocks[0].Signature != "SIG-ONE" || blocks[0].SignatureFormat != "anthropic" {
		t.Errorf("block 0 = %q/%q/%q, want \"first\"/\"SIG-ONE\"/\"anthropic\"",
			blocks[0].Thinking, blocks[0].Signature, blocks[0].SignatureFormat)
	}
	if blocks[1].Thinking != "" || blocks[1].Signature != "" {
		t.Errorf("block 1 = %q/%q, want the redacted block's empty text and signature", blocks[1].Thinking, blocks[1].Signature)
	}
	if !blocks[1].ReplayableAs("anthropic-redacted-thinking") {
		t.Errorf("block 1 provider state = %s/%q, want redacted anthropic state", blocks[1].ProviderState, blocks[1].ProviderStateFormat)
	}
	if blocks[2].Thinking != "second" || blocks[2].Signature != "SIG-TWO" || blocks[2].SignatureFormat != "anthropic" {
		t.Errorf("block 2 = %q/%q/%q, want \"second\"/\"SIG-TWO\"/\"anthropic\"",
			blocks[2].Thinking, blocks[2].Signature, blocks[2].SignatureFormat)
	}
}

// TestStreamingMatchesNonStreamingReasoning is the invariant inference/CLAUDE.md
// demands: for the same response, the streaming decoder must reconstruct the
// same continuation state as the non-streaming decoder, per-block signatures
// included.
func TestStreamingMatchesNonStreamingReasoning(t *testing.T) {
	t.Parallel()

	streamed := accumulateThinking(t, multiBlockReasoningStream)

	resp, err := anthropicapi.DecodeResponse([]byte(multiBlockReasoningBody))
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
		if streamed[i].Thinking != direct[i].Thinking || streamed[i].Signature != direct[i].Signature ||
			streamed[i].SignatureFormat != direct[i].SignatureFormat {
			t.Errorf("block %d: streamed %q/%q/%q, non-streaming %q/%q/%q", i,
				streamed[i].Thinking, streamed[i].Signature, streamed[i].SignatureFormat,
				direct[i].Thinking, direct[i].Signature, direct[i].SignatureFormat)
		}
		if string(streamed[i].ProviderState) != string(direct[i].ProviderState) ||
			streamed[i].ProviderStateFormat != direct[i].ProviderStateFormat {
			t.Errorf("block %d provider state: streamed %s/%q, non-streaming %s/%q", i,
				streamed[i].ProviderState, streamed[i].ProviderStateFormat,
				direct[i].ProviderState, direct[i].ProviderStateFormat)
		}
	}
}
