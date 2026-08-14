package openaiapi

import (
	"encoding/json"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	failure "github.com/looprig/inference/failure"
	"github.com/looprig/inference/internal/usagenorm"
	usage "github.com/looprig/inference/usage"
)

// DecodeResponse parses an OpenAI chat completions JSON response body into
// a provider-neutral *inference.Response.
func DecodeResponse(body []byte) (*inference.Response, error) {
	var wire chatResponse
	if err := json.Unmarshal(body, &wire); err != nil {
		return nil, err
	}

	// A top-level `error` member is the spec's ErrorResponse envelope. It can
	// arrive on an HTTP 200 body (OpenAI-compatible gateways document doing
	// exactly this), where transport's status-driven APIErrorFromResponse
	// never fires — so without this the caller gets a successful, empty
	// assistant turn instead of the upstream failure. Checked before the
	// choices guard so the error's own diagnostics survive rather than being
	// flattened into a bare "no choices" APIError.
	if wire.Error != nil {
		// APIErrorFromResponse re-parses the body so the code passes through
		// failure's allowlist, exactly as the non-2xx transport path does; the
		// status comes from a numeric `code` when the gateway smuggled the real
		// HTTP status there, so the retry layer can still classify it.
		return nil, failure.APIErrorFromResponse(wire.Error.httpStatus(), body, nil, 0)
	}

	if len(wire.Choices) == 0 {
		return nil, failure.NewAPIError(0, "", "", 0)
	}

	msg := wire.Choices[0].Message
	blocks, err := buildBlocks(msg)
	if err != nil {
		return nil, err
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
				Blocks: blocks,
			},
			Usage: messageUsage,
		},
		Model:        wire.Model,
		Usage:        usage,
		FinishReason: mapFinishReason(wire.Choices[0].FinishReason),
	}, nil
}

func normalizeUsage(wire *chatUsage) (*usage.Usage, error) {
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
	// Each count is validated individually above. The relationship BETWEEN
	// completion_tokens and its reasoning_tokens breakdown is not re-checked
	// here: OpenAI documents reasoning as a subset, but this format is spoken by
	// roughly fifty servers and one of them (OpenRouter, on
	// nvidia/nemotron-3-ultra-550b-a55b:free) returned completion_tokens=216
	// with reasoning_tokens=226 on a complete HTTP 200. Rejecting that pair
	// discarded the answer over an accounting field. Both numbers are reported
	// as received; content.Usage.ReasoningWithinOutput exposes the divergence.
	usage := usage.Usage{InputTokens: input, OutputTokens: output, CacheReadTokens: cacheRead, CacheCreationTokens: cacheCreation, ReasoningTokens: reasoning}
	return &usage, nil
}

func normalizePromptUsage(wire chatUsage) (content.TokenCount, content.TokenCount, content.TokenCount, error) {
	prompt, err := wire.PromptTokens.TokenCount(usagenorm.FieldInputTokens)
	if err != nil {
		return 0, 0, 0, err
	}
	cacheRead, err := wire.PromptTokensDetails.CachedTokens.OptionalTokenCount(usagenorm.FieldCacheReadTokens)
	if err != nil {
		return 0, 0, 0, err
	}
	cacheCreation, err := wire.PromptTokensDetails.CacheWriteTokens.OptionalTokenCount(usagenorm.FieldCacheCreationTokens)
	if err != nil {
		return 0, 0, 0, err
	}
	input, err := usagenorm.SubtractTokenCounts(usagenorm.FieldInputTokens, prompt, cacheRead, cacheCreation)
	if err != nil {
		return 0, 0, 0, err
	}
	return input, cacheRead, cacheCreation, nil
}

func normalizeCompletionUsage(wire chatUsage) (content.TokenCount, content.TokenCount, error) {
	output, err := wire.CompletionTokens.TokenCount(usagenorm.FieldOutputTokens)
	if err != nil {
		return 0, 0, err
	}
	reasoning, err := wire.CompletionTokensDetails.ReasoningTokens.OptionalTokenCount(usagenorm.FieldReasoningTokens)
	return output, reasoning, err
}

// buildBlocks constructs an ordered slice of content blocks from a decoded
// chatMessage. Reasoning comes first, then text, then any refusal, then tool
// calls.
func buildBlocks(msg chatMessage) ([]content.Block, error) {
	var blocks []content.Block

	if msg.ReasoningContent != "" {
		blocks = append(blocks, &content.ThinkingBlock{Thinking: msg.ReasoningContent})
	}

	if s, ok := msg.Content.(string); ok && s != "" {
		blocks = append(blocks, &content.TextBlock{Text: s})
	}

	blocks = append(blocks, refusalBlocks(msg.Refusal)...)

	for _, tc := range msg.ToolCalls {
		// `arguments` is spec-typed STRING — a JSON document carried inside a
		// JSON string — while ToolUseBlock.Input is the arguments OBJECT, the
		// same value openairesponses and anthropicapi decode and the same value
		// encodeAIMessage re-quotes on replay. Assigning the raw wire member
		// here left the quoted literal in Input, so the next request carried
		// "\"{\\\"city\\\":\\\"Paris\\\"}\"" and strict gateways answered 400
		// "Assistant tool call function.arguments must be a JSON object". The
		// streaming path never had the bug (its Arguments field is a Go string,
		// already unescaped), which is the divergence CLAUDE.md forbids; this
		// call is what makes the two paths agree. Same unwrapper as the server
		// request path, in the tolerant response direction — see
		// unwrapToolCallArguments (server_decode.go).
		input, err := unwrapToolCallArguments(tc.Function.Arguments, preserveInvalidToolArguments)
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, &content.ToolUseBlock{
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: input,
		})
	}

	return blocks, nil
}

// refusalBlocks maps a `refusal` member onto the neutral vocabulary's own
// refusal variant.
//
// PRESENCE, not content, is the signal: an absent (null) refusal is the normal
// case on every successful response and yields no block, while a present member
// yields a *content.RefusalBlock even when its text is empty — a provider may
// decline with no explanation, and dropping that would restore the exact bug
// the block type exists to fix, a refusal decoding as a successful empty reply.
// That is why the parameter is a *string; see chatMessage.Refusal (types.go).
//
// `refusal` and `content` are independently nullable in the schema and real
// responses populate only one. Both are emitted, in wire order — content first
// — for the pathological case where a provider sends both, rather than
// inventing an ordering or an exclusivity the wire does not define.
//
// The turn-level finish reason is deliberately NOT overridden for a refusal.
// OpenAI reports one with finish_reason "stop", the block now carries the
// per-block "declined" signal, and manufacturing content_filter would both
// report a filter intervention that did not occur and make a genuine
// content_filter finish indistinguishable from a model refusal.
func refusalBlocks(refusal *string) []content.Block {
	if refusal == nil {
		return nil
	}
	return []content.Block{&content.RefusalBlock{Text: *refusal}}
}
