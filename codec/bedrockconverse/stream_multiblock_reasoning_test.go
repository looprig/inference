package bedrockconverse_test

import (
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/core/content/streamaccumulator"
	"github.com/looprig/inference/codec/bedrockconverse"
)

// multiBlockReasoningFrames is a real ConverseStream event sequence for an
// interleaved-thinking turn: two signed reasoning blocks with a redacted
// reasoning block between them, at contentBlockIndex 0, 1 and 2. AWS puts the
// block index on every contentBlockDelta, so each reasoning block's signature is
// addressable on the wire.
func multiBlockReasoningFrames() []byte {
	return appendFrames(
		eventFrame("messageStart", `{"role":"assistant"}`),
		eventFrame("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"reasoningContent":{"text":"first"}}}`),
		eventFrame("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"reasoningContent":{"signature":"SIG-ONE"}}}`),
		eventFrame("contentBlockStop", `{"contentBlockIndex":0}`),
		eventFrame("contentBlockDelta", `{"contentBlockIndex":1,"delta":{"reasoningContent":{"redactedContent":"AQID/w=="}}}`),
		eventFrame("contentBlockStop", `{"contentBlockIndex":1}`),
		eventFrame("contentBlockDelta", `{"contentBlockIndex":2,"delta":{"reasoningContent":{"text":"second"}}}`),
		eventFrame("contentBlockDelta", `{"contentBlockIndex":2,"delta":{"reasoningContent":{"signature":"SIG-TWO"}}}`),
		eventFrame("contentBlockStop", `{"contentBlockIndex":2}`),
		eventFrame("messageStop", `{"stopReason":"end_turn"}`),
	)
}

// multiBlockReasoningBody is the SAME message delivered by non-streaming
// Converse.
const multiBlockReasoningBody = `{"output":{"message":{"role":"assistant","content":[` +
	`{"reasoningContent":{"reasoningText":{"text":"first","signature":"SIG-ONE"}}},` +
	`{"reasoningContent":{"redactedContent":"AQID/w=="}},` +
	`{"reasoningContent":{"reasoningText":{"text":"second","signature":"SIG-TWO"}}}]}},"stopReason":"end_turn"}`

func accumulateThinking(t *testing.T, body []byte) []content.ThinkingBlock {
	t.Helper()
	chunks, _, _, err := drainStream(t, body)
	if err != nil && err.Error() != "EOF" {
		t.Fatalf("terminal error = %v, want EOF", err)
	}
	var acc streamaccumulator.Thinking
	for _, chunk := range chunks {
		if tc, ok := chunk.(*content.ThinkingChunk); ok {
			acc.Add(tc)
		}
	}
	return acc.Blocks()
}

// TestDecodeStream_MultipleReasoningBlocksKeepTheirOwnSignatures pins the
// per-block signature invariant against a real event-stream fixture.
func TestDecodeStream_MultipleReasoningBlocksKeepTheirOwnSignatures(t *testing.T) {
	t.Parallel()

	blocks := accumulateThinking(t, multiBlockReasoningFrames())
	if len(blocks) != 3 {
		t.Fatalf("Blocks() = %d block(s) %#v, want 3", len(blocks), blocks)
	}
	if blocks[0].Thinking != "first" || blocks[0].Signature != "SIG-ONE" {
		t.Errorf("block 0 = %q/%q, want \"first\"/\"SIG-ONE\"", blocks[0].Thinking, blocks[0].Signature)
	}
	if blocks[1].Thinking != "" || blocks[1].Signature != "" {
		t.Errorf("block 1 = %q/%q, want the redacted block's empty text and signature", blocks[1].Thinking, blocks[1].Signature)
	}
	if !blocks[1].ReplayableAs("bedrock-converse-redacted-thinking") {
		t.Errorf("block 1 provider state = %s/%q, want redacted bedrock state", blocks[1].ProviderState, blocks[1].ProviderStateFormat)
	}
	if blocks[2].Thinking != "second" || blocks[2].Signature != "SIG-TWO" {
		t.Errorf("block 2 = %q/%q, want \"second\"/\"SIG-TWO\"", blocks[2].Thinking, blocks[2].Signature)
	}
}

// TestStreamingMatchesNonStreamingReasoning holds both decode paths against each
// other for the same message.
func TestStreamingMatchesNonStreamingReasoning(t *testing.T) {
	t.Parallel()

	streamed := accumulateThinking(t, multiBlockReasoningFrames())

	resp, err := bedrockconverse.DecodeResponse([]byte(multiBlockReasoningBody))
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
		if string(streamed[i].ProviderState) != string(direct[i].ProviderState) ||
			streamed[i].ProviderStateFormat != direct[i].ProviderStateFormat {
			t.Errorf("block %d provider state: streamed %s/%q, non-streaming %s/%q", i,
				streamed[i].ProviderState, streamed[i].ProviderStateFormat,
				direct[i].ProviderState, direct[i].ProviderStateFormat)
		}
	}
}
