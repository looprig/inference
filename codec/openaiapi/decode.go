package openaiapi

import (
	"encoding/json"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
)

// DecodeResponse parses an OpenAI chat completions JSON response body into
// a provider-neutral *inference.Response.
func DecodeResponse(body []byte) (*inference.Response, error) {
	var wire chatResponse
	if err := json.Unmarshal(body, &wire); err != nil {
		return nil, err
	}

	if len(wire.Choices) == 0 {
		return nil, &inference.APIError{Status: 0, Message: "response contains no choices", Body: body}
	}

	msg := wire.Choices[0].Message
	blocks := buildBlocks(msg)

	var usage *inference.Usage
	if wire.Usage != nil {
		usage = &inference.Usage{
			InputTokens:  wire.Usage.PromptTokens,
			OutputTokens: wire.Usage.CompletionTokens,
		}
	}

	return &inference.Response{
		Message: &content.AIMessage{
			Message: content.Message{
				Role:   content.RoleAssistant,
				Blocks: blocks,
			},
		},
		Model: wire.Model,
		Usage: usage,
	}, nil
}

// buildBlocks constructs an ordered slice of content blocks from a decoded
// chatMessage. Reasoning comes first, then text, then tool calls.
func buildBlocks(msg chatMessage) []content.Block {
	var blocks []content.Block

	if msg.ReasoningContent != "" {
		blocks = append(blocks, &content.ThinkingBlock{Thinking: msg.ReasoningContent})
	}

	if s, ok := msg.Content.(string); ok && s != "" {
		blocks = append(blocks, &content.TextBlock{Text: s})
	}

	for _, tc := range msg.ToolCalls {
		blocks = append(blocks, &content.ToolUseBlock{
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: tc.Function.Arguments,
		})
	}

	return blocks
}
