package anthropicapi

import (
	"encoding/json"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	failure "github.com/looprig/inference/failure"
	"github.com/looprig/inference/internal/usagenorm"
	usage "github.com/looprig/inference/usage"
)

// DecodeResponse parses a non-streaming Anthropic Messages API response body into
// a provider-neutral *inference.Response. An `error`-type envelope (a 200 body carrying
// {"type":"error",...}) is surfaced as a *failure.APIError. An empty content array
// is a valid response (e.g. a refusal or a pure stop), not an error.
func DecodeResponse(body []byte) (*inference.Response, error) {
	var wire messageResponse
	if err := json.Unmarshal(body, &wire); err != nil {
		return nil, err
	}

	if wire.Type == responseTypeError {
		return nil, failure.NewAPIError(0, "", "", 0)
	}

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
				Blocks: decodeBlocks(wire.Content),
			},
			Usage: messageUsage,
		},
		Model:        wire.Model,
		Usage:        usage,
		FinishReason: mapFinishReason(wire.StopReason),
	}, nil
}

func normalizeUsage(wire *messageUsage) (*usage.Usage, error) {
	if wire == nil {
		return nil, nil
	}
	input, err := wire.InputTokens.TokenCount(usagenorm.FieldInputTokens)
	if err != nil {
		return nil, err
	}
	output, err := wire.OutputTokens.TokenCount(usagenorm.FieldOutputTokens)
	if err != nil {
		return nil, err
	}
	// Anthropic declares both cache counts as anyOf[{integer, minimum 0}, {null}]
	// with "default": null, so a conforming response carries them as null on
	// every turn that read from or wrote to no cache. OptionalTokenCount maps
	// null to zero while keeping the strict numeric validation for present
	// values; TokenCount would reject the documented default and discard an
	// otherwise complete response over an accounting field.
	cacheRead, err := wire.CacheReadTokens.OptionalTokenCount(usagenorm.FieldCacheReadTokens)
	if err != nil {
		return nil, err
	}
	cacheCreation, err := wire.CacheCreationTokens.OptionalTokenCount(usagenorm.FieldCacheCreationTokens)
	if err != nil {
		return nil, err
	}
	var reasoning content.TokenCount
	if wire.OutputTokensDetails != nil {
		reasoning, err = wire.OutputTokensDetails.ThinkingTokens.TokenCount(usagenorm.FieldReasoningTokens)
		if err != nil {
			return nil, err
		}
	}
	usage := usage.Usage{
		InputTokens:         input,
		OutputTokens:        output,
		CacheReadTokens:     cacheRead,
		CacheCreationTokens: cacheCreation,
		ReasoningTokens:     reasoning,
	}
	return &usage, nil
}

// decodeBlocks maps Anthropic response content blocks to provider-neutral blocks,
// preserving order. Unknown block types (redacted_thinking, server-tool blocks,
// etc.) are skipped tolerantly rather than erroring.
func decodeBlocks(blocks []anthropicBlock) []content.Block {
	var out []content.Block
	for _, b := range blocks {
		switch b.Type {
		case blockTypeText:
			// Anthropic permits empty text in responses but requires request text
			// to contain at least one character. Empty text has no semantic state
			// to preserve, so retaining it would make an otherwise legal assistant
			// turn impossible to replay on the next request.
			if b.Text == "" {
				continue
			}
			out = append(out, &content.TextBlock{Text: b.Text})
		case blockTypeThinking:
			// The signature is stamped with THIS dialect as it comes off the
			// wire. Anthropic verifies it cryptographically, and Bedrock
			// Converse serves the same models with a structurally identical
			// block, so provenance is the only thing that keeps the two apart
			// once the block is in the neutral transcript.
			out = append(out, content.NewSignedThinkingBlock(
				b.Thinking, b.Signature, signatureFormatFor(b.Signature), nil, ""))
		case blockTypeRedactedThinking:
			out = append(out, content.NewThinkingBlock("", "", opaqueRedactedState(b.Data), providerStateFormatAnthropicRedacted))
		case blockTypeToolUse:
			out = append(out, &content.ToolUseBlock{ID: b.ID, Name: b.Name, Input: b.Input})
		default:
			// Skip block types the neutral vocabulary does not model.
		}
	}
	return out
}
