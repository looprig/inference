package geminiapi_test

import (
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/core/content/streamaccumulator"
	"github.com/looprig/inference/codec/geminiapi"
)

// multiBlockReasoningStream is a real streamGenerateContent SSE body for a
// thinking model that reasons, calls a tool, and reasons again. Gemini seals a
// thought block by attaching its `thoughtSignature` to the block's LAST part, so
// the two signatures below belong to two distinct reasoning blocks that arrive
// in separate chunks — the shape ordinary Gemini streaming produces.
const multiBlockReasoningStream = "" +
	"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"first \",\"thought\":true}],\"role\":\"model\"}}]}\n\n" +
	"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"thought\",\"thought\":true,\"thoughtSignature\":\"SIG-ONE\"}],\"role\":\"model\"}}]}\n\n" +
	"data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"lookup\",\"args\":{}}}],\"role\":\"model\"}}]}\n\n" +
	"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"second thought\",\"thought\":true,\"thoughtSignature\":\"SIG-TWO\"}],\"role\":\"model\"}}]}\n\n" +
	"data: {\"candidates\":[{\"content\":{\"parts\":[],\"role\":\"model\"},\"finishReason\":\"STOP\"}]}\n\n"

// multiBlockReasoningBody is the SAME generation delivered non-streaming: the
// thought fragments joined into their two sealed parts.
const multiBlockReasoningBody = `{"candidates":[{"content":{"role":"model","parts":[` +
	`{"text":"first thought","thought":true,"thoughtSignature":"SIG-ONE"},` +
	`{"functionCall":{"name":"lookup","args":{}}},` +
	`{"text":"second thought","thought":true,"thoughtSignature":"SIG-TWO"}]},"finishReason":"STOP"}]}`

func accumulateThinking(t *testing.T, body string) []content.ThinkingBlock {
	t.Helper()
	reader, err := (geminiapi.Codec{}).DecodeStream(&http.Response{Body: io.NopCloser(strings.NewReader(body))})
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

// TestDecodeStream_MultipleThoughtBlocksKeepTheirOwnSignatures drives the real
// fixture through DecodeStream into the accumulator. A thoughtSignature is the
// continuation state Gemini requires back verbatim, per thought block; folding
// two blocks together rebinds the second signature onto both blocks' text.
func TestDecodeStream_MultipleThoughtBlocksKeepTheirOwnSignatures(t *testing.T) {
	t.Parallel()

	blocks := accumulateThinking(t, multiBlockReasoningStream)
	if len(blocks) != 2 {
		t.Fatalf("Blocks() = %d block(s) %#v, want 2", len(blocks), blocks)
	}
	if blocks[0].Thinking != "first thought" {
		t.Errorf("block 0 thinking = %q, want %q", blocks[0].Thinking, "first thought")
	}
	if got, want := string(blocks[0].ProviderState), `"SIG-ONE"`; got != want {
		t.Errorf("block 0 provider state = %s, want %s", got, want)
	}
	if blocks[1].Thinking != "second thought" {
		t.Errorf("block 1 thinking = %q, want %q", blocks[1].Thinking, "second thought")
	}
	if got, want := string(blocks[1].ProviderState), `"SIG-TWO"`; got != want {
		t.Errorf("block 1 provider state = %s, want %s", got, want)
	}
}

// TestStreamingMatchesNonStreamingReasoning holds both decode paths against each
// other for the same generation.
func TestStreamingMatchesNonStreamingReasoning(t *testing.T) {
	t.Parallel()

	streamed := accumulateThinking(t, multiBlockReasoningStream)

	resp, err := geminiapi.DecodeResponse([]byte(multiBlockReasoningBody))
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

func TestDecodeStream_SplitFunctionCallsKeepDistinctIndexesAndSignatures(t *testing.T) {
	t.Parallel()

	body := "" +
		"data: {\"candidates\":[{\"content\":{\"parts\":[{\"thoughtSignature\":\"SIG-A\",\"functionCall\":{\"name\":\"first\",\"args\":{\"x\":1}}}]}}]}\n\n" +
		"data: {\"candidates\":[{\"content\":{\"parts\":[{\"thoughtSignature\":\"SIG-B\",\"functionCall\":{\"name\":\"second\",\"args\":{\"x\":2}}}]}}]}\n\n" +
		"data: {\"candidates\":[{\"finishReason\":\"STOP\"}]}\n\n"
	reader, err := (geminiapi.Codec{}).DecodeStream(&http.Response{Body: io.NopCloser(strings.NewReader(body))})
	if err != nil {
		t.Fatalf("DecodeStream() error = %v", err)
	}
	defer reader.Close()

	var acc streamaccumulator.ToolUses
	for {
		chunk, err := reader.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Next() error = %v", err)
		}
		if tool, ok := chunk.(*content.ToolUseChunk); ok {
			acc.Add(tool)
		}
	}
	blocks := acc.Blocks()
	if len(blocks) != 2 {
		t.Fatalf("Blocks() = %d %#v, want 2 distinct calls", len(blocks), blocks)
	}
	if blocks[0].Name != "first" || blocks[1].Name != "second" {
		t.Fatalf("call names = %q, %q, want first, second", blocks[0].Name, blocks[1].Name)
	}
	if got := string(blocks[0].ProviderState); got != `"SIG-A"` {
		t.Errorf("first signature = %s, want SIG-A", got)
	}
	if blocks[0].ProviderStateFormat != "gemini" {
		t.Errorf("first signature format = %q, want gemini", blocks[0].ProviderStateFormat)
	}
	if got := string(blocks[1].ProviderState); got != `"SIG-B"` {
		t.Errorf("second signature = %s, want SIG-B", got)
	}
	if blocks[1].ProviderStateFormat != "gemini" {
		t.Errorf("second signature format = %q, want gemini", blocks[1].ProviderStateFormat)
	}
}
