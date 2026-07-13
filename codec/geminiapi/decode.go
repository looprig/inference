package geminiapi

import (
	"encoding/json"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
)

// DecodeResponse parses a Gemini generateContent JSON response body into a
// provider-neutral *inference.Response. It reads candidates[0]; a body with no
// candidates is an *inference.APIError (matching the sibling OpenAI codec), and
// malformed JSON is a *DecodeError.
func DecodeResponse(body []byte) (*inference.Response, error) {
	var wire GenerateContentResponse
	if err := json.Unmarshal(body, &wire); err != nil {
		return nil, &DecodeError{Reason: "unmarshal response body", Err: err}
	}

	if len(wire.Candidates) == 0 {
		return nil, &inference.APIError{Status: 0, Message: "response contains no candidates", Body: body}
	}

	blocks := buildBlocks(wire.Candidates[0].Content.Parts)

	usage, err := normalizeUsage(wire.UsageMetadata)
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
		Model: wire.ModelVersion,
		Usage: usage,
	}, nil
}

func normalizeUsage(wire *usageMetadata) (*inference.Usage, error) {
	if wire == nil {
		return nil, nil
	}
	input, cacheRead, err := normalizeInputUsage(*wire)
	if err != nil {
		return nil, err
	}
	output, reasoning, err := normalizeOutputUsage(*wire)
	if err != nil {
		return nil, err
	}
	usage := inference.Usage{InputTokens: input, OutputTokens: output, CacheReadTokens: cacheRead, ReasoningTokens: reasoning}
	if err := inference.ValidateNormalizedUsage(usage); err != nil {
		return nil, err
	}
	return &usage, nil
}

func normalizeInputUsage(wire usageMetadata) (content.TokenCount, content.TokenCount, error) {
	prompt, err := inference.NormalizeTokenCount(inference.UsageNormalizationFieldInputTokens, wire.PromptTokenCount)
	if err != nil {
		return 0, 0, err
	}
	cacheRead, err := inference.NormalizeTokenCount(inference.UsageNormalizationFieldCacheReadTokens, wire.CachedContentTokenCount)
	if err != nil {
		return 0, 0, err
	}
	input, err := inference.SubtractTokenCounts(inference.UsageNormalizationFieldInputTokens, prompt, cacheRead, 0)
	if err != nil {
		return 0, 0, err
	}
	return input, cacheRead, nil
}

func normalizeOutputUsage(wire usageMetadata) (content.TokenCount, content.TokenCount, error) {
	candidates, err := inference.NormalizeTokenCount(inference.UsageNormalizationFieldOutputTokens, wire.CandidatesTokenCount)
	if err != nil {
		return 0, 0, err
	}
	reasoning, err := inference.NormalizeTokenCount(inference.UsageNormalizationFieldReasoningTokens, wire.ThoughtsTokenCount)
	if err != nil {
		return 0, 0, err
	}
	output, err := inference.AddTokenCounts(inference.UsageNormalizationFieldOutputTokens, candidates, reasoning)
	return output, reasoning, err
}

// buildBlocks maps candidate parts to content blocks, preserving Gemini's part
// order (which interleaves text, thoughts, and tool calls). A functionCall part
// becomes a ToolUseBlock; a thought-tagged text part becomes a ThinkingBlock; any
// other non-empty text part becomes a TextBlock. Empty text and unknown parts are
// skipped.
func buildBlocks(parts []geminiPart) []content.Block {
	var blocks []content.Block
	for _, p := range parts {
		switch {
		case p.FunctionCall != nil:
			blocks = append(blocks, &content.ToolUseBlock{
				ID:    p.FunctionCall.ID,
				Name:  p.FunctionCall.Name,
				Input: argsJSON(p.FunctionCall.Args),
			})
		case p.Thought && p.Text != "":
			blocks = append(blocks, &content.ThinkingBlock{Thinking: p.Text})
		case p.Text != "":
			blocks = append(blocks, &content.TextBlock{Text: p.Text})
		}
	}
	return blocks
}
