package anthropicapi

import (
	"encoding/json"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
)

// DecodeResponse parses a non-streaming Anthropic Messages API response body into
// a provider-neutral *inference.Response. An `error`-type envelope (a 200 body carrying
// {"type":"error",...}) is surfaced as an *inference.APIError. An empty content array
// is a valid response (e.g. a refusal or a pure stop), not an error.
func DecodeResponse(body []byte) (*inference.Response, error) {
	var wire messageResponse
	if err := json.Unmarshal(body, &wire); err != nil {
		return nil, err
	}

	if wire.Type == responseTypeError {
		msg := "anthropicapi: error response"
		if wire.Error != nil && wire.Error.Message != "" {
			msg = wire.Error.Message
		}
		return nil, &inference.APIError{Status: 0, Message: msg, Body: body}
	}

	var usage *inference.Usage
	if wire.Usage != nil {
		usage = &inference.Usage{
			InputTokens:  wire.Usage.InputTokens,
			OutputTokens: wire.Usage.OutputTokens,
		}
	}

	return &inference.Response{
		Message: &content.AIMessage{
			Message: content.Message{
				Role:   content.RoleAssistant,
				Blocks: decodeBlocks(wire.Content),
			},
		},
		Model: wire.Model,
		Usage: usage,
	}, nil
}

// decodeBlocks maps Anthropic response content blocks to provider-neutral blocks,
// preserving order. Unknown block types (redacted_thinking, server-tool blocks,
// etc.) are skipped tolerantly rather than erroring.
func decodeBlocks(blocks []anthropicBlock) []content.Block {
	var out []content.Block
	for _, b := range blocks {
		switch b.Type {
		case blockTypeText:
			out = append(out, &content.TextBlock{Text: b.Text})
		case blockTypeThinking:
			out = append(out, &content.ThinkingBlock{Thinking: b.Thinking, Signature: b.Signature})
		case blockTypeToolUse:
			out = append(out, &content.ToolUseBlock{ID: b.ID, Name: b.Name, Input: b.Input})
		default:
			// Skip block types the neutral vocabulary does not model.
		}
	}
	return out
}
