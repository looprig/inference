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

	usage, err := normalizeUsage(wire.Usage)
	if err != nil {
		return nil, err
	}
	var messageUsage *content.Usage
	if usage != nil {
		cloned := *usage
		messageUsage = &cloned
	}

	return &inference.Response{
		Message: &content.AIMessage{
			Message: content.Message{
				Role:   content.RoleAssistant,
				Blocks: blocks,
			},
			Usage: messageUsage,
		},
		Model: wire.Model,
		Usage: usage,
	}, nil
}

func normalizeUsage(wire *chatUsage) (*inference.Usage, error) {
	if wire == nil {
		return nil, nil
	}
	input, cacheRead, cacheCreation, err := normalizePromptUsage(*wire)
	if err != nil {
		return nil, err
	}
	output, reasoning, err := normalizeCompletionUsage(*wire)
	if err != nil {
		return nil, err
	}
	usage := inference.Usage{InputTokens: input, OutputTokens: output, CacheReadTokens: cacheRead, CacheCreationTokens: cacheCreation, ReasoningTokens: reasoning}
	if err := inference.ValidateNormalizedUsage(usage); err != nil {
		return nil, err
	}
	return &usage, nil
}

func normalizePromptUsage(wire chatUsage) (content.TokenCount, content.TokenCount, content.TokenCount, error) {
	prompt, err := inference.NormalizeTokenCount(inference.UsageNormalizationFieldInputTokens, wire.PromptTokens)
	if err != nil {
		return 0, 0, 0, err
	}
	cacheRead, err := inference.NormalizeTokenCount(inference.UsageNormalizationFieldCacheReadTokens, wire.PromptTokensDetails.CachedTokens)
	if err != nil {
		return 0, 0, 0, err
	}
	cacheCreation, err := inference.NormalizeTokenCount(inference.UsageNormalizationFieldCacheCreationTokens, wire.PromptTokensDetails.CacheWriteTokens)
	if err != nil {
		return 0, 0, 0, err
	}
	input, err := inference.SubtractTokenCounts(inference.UsageNormalizationFieldInputTokens, prompt, cacheRead, cacheCreation)
	if err != nil {
		return 0, 0, 0, err
	}
	return input, cacheRead, cacheCreation, nil
}

func normalizeCompletionUsage(wire chatUsage) (content.TokenCount, content.TokenCount, error) {
	output, err := inference.NormalizeTokenCount(inference.UsageNormalizationFieldOutputTokens, wire.CompletionTokens)
	if err != nil {
		return 0, 0, err
	}
	reasoning, err := inference.NormalizeTokenCount(inference.UsageNormalizationFieldReasoningTokens, wire.CompletionTokensDetails.ReasoningTokens)
	return output, reasoning, err
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
