package anthropicapi_test

import (
	"encoding/json"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/anthropicapi"
	model "github.com/looprig/inference/model"
)

// cachingModel is baseModel with the PromptCaching capability set.
func cachingModel() model.Model {
	m := baseModel()
	m.Caps.PromptCaching = true
	return m
}

// cacheControlOf returns the block's cache_control object, or nil if absent.
func cacheControlOf(t *testing.T, block map[string]json.RawMessage) map[string]string {
	t.Helper()
	raw, ok := block["cache_control"]
	if !ok {
		return nil
	}
	var cc map[string]string
	if err := json.Unmarshal(raw, &cc); err != nil {
		t.Fatalf("unmarshal cache_control: %v", err)
	}
	return cc
}

// requireEphemeral asserts the block carries {"type":"ephemeral"}.
func requireEphemeral(t *testing.T, block map[string]json.RawMessage) {
	t.Helper()
	cc := cacheControlOf(t, block)
	if cc == nil {
		t.Fatal("expected cache_control on block, got none")
	}
	if cc["type"] != "ephemeral" {
		t.Errorf(`cache_control type = %q, want "ephemeral"`, cc["type"])
	}
}

// --- TestEncodeRequest_PromptCachingBreakpoints ---

// With Caps.PromptCaching set, the codec emits at most two ephemeral
// cache_control breakpoints: one on the (single) system block — which caches
// tools + system, since tools render first — and one on the last cacheable
// block of the last message, so multi-turn hits accrue incrementally.
func TestEncodeRequest_PromptCachingBreakpoints(t *testing.T) {
	t.Parallel()

	req := inference.Request{
		Model:  cachingModel(),
		System: "be helpful",
		Messages: content.AgenticMessages{
			userMsg(textBlock("first")),
			aiMsg(textBlock("reply")),
			userMsg(textBlock("second")),
		},
		Tools: []inference.Tool{{Name: "t", Schema: json.RawMessage(`{"type":"object"}`)}},
	}
	data, err := anthropicapi.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}
	body := decodeObj(t, data)

	// System becomes the array-of-blocks form with a breakpoint on its block.
	sysBlocks := asObjs(t, body["system"])
	if len(sysBlocks) != 1 {
		t.Fatalf("system block count = %d, want 1", len(sysBlocks))
	}
	if got := asString(t, sysBlocks[0]["text"]); got != "be helpful" {
		t.Errorf("system text = %q, want %q", got, "be helpful")
	}
	requireEphemeral(t, sysBlocks[0])

	// The last block of the last message carries the second breakpoint; no
	// other message block carries one.
	msgs := messagesOf(t, body)
	for i, msg := range msgs {
		blocks := blocksOf(t, msg)
		for j, block := range blocks {
			last := i == len(msgs)-1 && j == len(blocks)-1
			if last {
				requireEphemeral(t, block)
			} else if cacheControlOf(t, block) != nil {
				t.Errorf("unexpected cache_control on message %d block %d", i, j)
			}
		}
	}
}

// A tool-result tail (the common agentic-loop shape) carries the message
// breakpoint on the tool_result block itself.
func TestEncodeRequest_PromptCachingToolResultTail(t *testing.T) {
	t.Parallel()

	req := inference.Request{
		Model: cachingModel(),
		Messages: content.AgenticMessages{
			userMsg(textBlock("do it")),
			aiMsg(&content.ToolUseBlock{ID: "call_1", Name: "t", Input: json.RawMessage(`{}`)}),
			toolResultMsg("call_1", false, textBlock("done")),
		},
	}
	data, err := anthropicapi.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}
	body := decodeObj(t, data)
	msgs := messagesOf(t, body)
	lastBlocks := blocksOf(t, msgs[len(msgs)-1])
	lastBlock := lastBlocks[len(lastBlocks)-1]
	if got := asString(t, lastBlock["type"]); got != "tool_result" {
		t.Fatalf("last block type = %q, want tool_result", got)
	}
	requireEphemeral(t, lastBlock)
}

// A trailing thinking block is not a valid breakpoint carrier: the breakpoint
// walks back to the nearest cacheable block.
func TestEncodeRequest_PromptCachingSkipsThinkingTail(t *testing.T) {
	t.Parallel()

	req := inference.Request{
		Model: cachingModel(),
		Messages: content.AgenticMessages{
			userMsg(textBlock("q")),
			aiMsg(textBlock("answer"), &content.ThinkingBlock{Thinking: "hmm", Signature: "sig"}),
		},
	}
	data, err := anthropicapi.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}
	body := decodeObj(t, data)
	msgs := messagesOf(t, body)
	lastBlocks := blocksOf(t, msgs[len(msgs)-1])
	if cc := cacheControlOf(t, lastBlocks[len(lastBlocks)-1]); cc != nil {
		t.Error("thinking block must not carry cache_control")
	}
	requireEphemeral(t, lastBlocks[len(lastBlocks)-2])
}

// Without the capability the wire shape is byte-identical to before: system
// stays a plain string and no cache_control appears anywhere.
func TestEncodeRequest_PromptCachingOffKeepsWireShape(t *testing.T) {
	t.Parallel()

	req := inference.Request{
		Model:    baseModel(),
		System:   "be helpful",
		Messages: content.AgenticMessages{userMsg(textBlock("hi"))},
	}
	data, err := anthropicapi.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}
	body := decodeObj(t, data)
	if got := asString(t, body["system"]); got != "be helpful" {
		t.Errorf("system = %q, want plain string %q", got, "be helpful")
	}
	for _, msg := range messagesOf(t, body) {
		for _, block := range blocksOf(t, msg) {
			if cacheControlOf(t, block) != nil {
				t.Error("unexpected cache_control with capability off")
			}
		}
	}
}

// With no system prompt, only the message breakpoint is emitted and the
// system field stays absent.
func TestEncodeRequest_PromptCachingNoSystem(t *testing.T) {
	t.Parallel()

	req := inference.Request{
		Model:    cachingModel(),
		Messages: content.AgenticMessages{userMsg(textBlock("hi"))},
	}
	data, err := anthropicapi.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}
	body := decodeObj(t, data)
	if _, ok := body["system"]; ok {
		t.Error("system field should be omitted when empty")
	}
	msgs := messagesOf(t, body)
	blocks := blocksOf(t, msgs[len(msgs)-1])
	requireEphemeral(t, blocks[len(blocks)-1])
}
