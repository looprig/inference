package geminiapi

import (
	"encoding/json"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	failure "github.com/looprig/inference/failure"
	"github.com/looprig/inference/internal/usagenorm"
	"github.com/looprig/inference/stream"
	usage "github.com/looprig/inference/usage"
)

// DecodeResponse parses a Gemini generateContent JSON response body into a
// provider-neutral *inference.Response. It reads candidates[0]; a body with no
// candidates is a *failure.APIError (matching the sibling OpenAI codec) — or a
// *PromptBlockedError, which unwraps to one, when promptFeedback says why — and
// malformed JSON is a *DecodeError.
func DecodeResponse(body []byte) (*inference.Response, error) {
	var wire GenerateContentResponse
	if err := json.Unmarshal(body, &wire); err != nil {
		return nil, &DecodeError{Reason: "unmarshal response body", Err: err}
	}

	if len(wire.Candidates) == 0 {
		return nil, candidateLessError(wire)
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
		Model:        wire.ModelVersion,
		Usage:        usage,
		FinishReason: responseFinishReason(wire.Candidates[0]),
	}, nil
}

// candidateLessError explains a response that carried no candidates. Per the
// discovery document that happens "only if there was something wrong with the
// prompt (check prompt_feedback)", and the overwhelmingly common cause is a
// content-filter refusal — which arrives with a successful HTTP status, an empty
// candidates array and no error envelope. Reading promptFeedback is what
// separates that from an unknown failure; without it, every such body collapsed
// into one statusless, codeless *failure.APIError.
//
// The response's usage rides along on the block, because a refused prompt is
// still a charged prompt. Usage that fails normalization is dropped rather than
// returned in place of the block: an accounting field must never displace the
// one diagnostic the response carried.
func candidateLessError(wire GenerateContentResponse) error {
	if blocked := promptBlockedError(wire); blocked != nil {
		return blocked
	}
	return failure.NewAPIError(0, "", "", 0)
}

// promptBlockedError builds the block report a response's promptFeedback
// justifies, or nil when it justifies none. blockReason is "Optional. If set,
// the prompt was blocked", so feedback carrying only safetyRatings is
// classification detail about a prompt that was NOT refused and reports nothing.
// Shared with the streaming decoder (decodeEvent, codec.go), where the same
// frame otherwise skipped silently and the stream failed later as truncated.
func promptBlockedError(wire GenerateContentResponse) *PromptBlockedError {
	if wire.PromptFeedback == nil || wire.PromptFeedback.BlockReason == "" {
		return nil
	}
	blocked := &PromptBlockedError{
		BlockReason:   knownBlockReason(wire.PromptFeedback.BlockReason),
		SafetyRatings: safetyRatings(wire.PromptFeedback.SafetyRatings),
	}
	if usage, err := normalizeUsage(wire.UsageMetadata); err == nil {
		blocked.Usage = usage
	}
	return blocked
}

// blockReasons is PromptFeedback.blockReason's published enum. It is an
// allowlist, not a denylist: a member Google adds later is withheld rather than
// copied into an error, keeping this codec's failures free of unbounded provider
// strings for the same reason failure.APIError admits only allowlisted codes.
// The block itself is still reported — the field's presence, not its value, is
// what says the prompt was refused.
var blockReasons = map[string]struct{}{
	"BLOCK_REASON_UNSPECIFIED": {},
	"SAFETY":                   {},
	"OTHER":                    {},
	"BLOCKLIST":                {},
	"PROHIBITED_CONTENT":       {},
	"IMAGE_SAFETY":             {},
}

func knownBlockReason(reason string) string {
	if _, ok := blockReasons[reason]; !ok {
		return ""
	}
	return reason
}

// harmCategories and harmProbabilities are SafetyRating's published enums, held
// to the same allowlist discipline as blockReason. A rating whose category is
// unrecognized is dropped entirely rather than surfaced half-known, since the
// category is the only thing that makes a rating meaningful.
var harmCategories = map[string]struct{}{
	"HARM_CATEGORY_UNSPECIFIED": {}, "HARM_CATEGORY_DEROGATORY": {},
	"HARM_CATEGORY_TOXICITY": {}, "HARM_CATEGORY_VIOLENCE": {},
	"HARM_CATEGORY_SEXUAL": {}, "HARM_CATEGORY_MEDICAL": {},
	"HARM_CATEGORY_DANGEROUS": {}, "HARM_CATEGORY_HARASSMENT": {},
	"HARM_CATEGORY_HATE_SPEECH": {}, "HARM_CATEGORY_SEXUALLY_EXPLICIT": {},
	"HARM_CATEGORY_DANGEROUS_CONTENT": {}, "HARM_CATEGORY_CIVIC_INTEGRITY": {},
	"HARM_CATEGORY_JAILBREAK": {},
}

var harmProbabilities = map[string]struct{}{
	"HARM_PROBABILITY_UNSPECIFIED": {}, "NEGLIGIBLE": {},
	"LOW": {}, "MEDIUM": {}, "HIGH": {},
}

func safetyRatings(wire []wireSafetyRating) []SafetyRating {
	var out []SafetyRating
	for _, rating := range wire {
		if _, ok := harmCategories[rating.Category]; !ok {
			continue
		}
		probability := rating.Probability
		if _, ok := harmProbabilities[probability]; !ok {
			probability = ""
		}
		out = append(out, SafetyRating{Category: rating.Category, Probability: probability, Blocked: rating.Blocked})
	}
	return out
}

func responseFinishReason(candidate candidate) stream.FinishReason {
	if hasFunctionCall(candidate.Content.Parts) && (candidate.FinishReason == "" || candidate.FinishReason == "STOP") {
		return stream.FinishReasonToolUse
	}
	return mapFinishReason(candidate.FinishReason)
}

func normalizeUsage(wire *usageMetadata) (*usage.Usage, error) {
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
	if err := validateReportedTotal(*wire); err != nil {
		return nil, err
	}
	// totalTokenCount is deliberately NOT cross-checked against the components
	// this codec models. The neutral Usage carries no total, so the reported one
	// feeds nothing; a strict equality could therefore only ever destroy a
	// completed generation over an accounting field, which this module forbids.
	// It was not theoretical: toolUsePromptTokenCount is a published member this
	// codec did not model, and the discovery document's own prose for
	// totalTokenCount ("prompt + thoughts + response candidates") does not say
	// whether it participates — observed responses go both ways. Either the
	// reported total counted it and the equality failed, or it did not and the
	// next member Google adds will fail it instead. So every grounded and
	// code-execution turn was at risk — and because Gemini repeats
	// usageMetadata on every SSE frame, the
	// streaming path failed on frame 1, before a single character of the answer
	// was emitted, and reported it as a stream error rather than as the answer it
	// was. Modelling that field (below) fixes today's mismatch; dropping the
	// gate is what stops the next member Google adds to UsageMetadata from doing
	// the same thing again. The counts we do model keep every strict per-field
	// rule, and cachedContentTokenCount not exceeding promptTokenCount is still
	// enforced, by the subtraction in normalizeInputUsage.
	//
	// The neutral ReasoningTokens-within-OutputTokens convention needs no check
	// here because normalizeOutputUsage establishes it by construction: Gemini
	// is the one format that reports thoughts OUTSIDE the candidate count
	// ("prompt + thoughts + response candidates", per the discovery document's
	// totalTokenCount), so output is candidates PLUS thoughts and reasoning can
	// never exceed it.
	usage := usage.Usage{InputTokens: input, OutputTokens: output, CacheReadTokens: cacheRead, ReasoningTokens: reasoning}
	return &usage, nil
}

func normalizeInputUsage(wire usageMetadata) (content.TokenCount, content.TokenCount, error) {
	prompt, err := wire.PromptTokenCount.TokenCount(usagenorm.FieldInputTokens)
	if err != nil {
		return 0, 0, err
	}
	cacheRead, err := wire.CachedContentTokenCount.TokenCount(usagenorm.FieldCacheReadTokens)
	if err != nil {
		return 0, 0, err
	}
	// promptTokenCount is documented as "the total effective prompt size ...
	// includes the number of tokens in the cached content", so the uncached input
	// is the difference. toolUsePromptTokenCount is reported SEPARATELY, not
	// inside promptTokenCount: a full usageMetadata carries a promptTokensDetails
	// breakdown that sums to promptTokenCount exactly, with
	// toolUsePromptTokensDetails listed apart from it. So it is added rather than
	// subtracted — those are prompt tokens the caller was charged for, and the
	// neutral Usage has no distinct bucket for them. Dropping them would silently
	// under-report input on every grounded turn.
	input, err := usagenorm.SubtractTokenCounts(usagenorm.FieldInputTokens, prompt, cacheRead, 0)
	if err != nil {
		return 0, 0, err
	}
	toolUsePrompt, err := wire.ToolUsePromptTokenCount.TokenCount(usagenorm.FieldInputTokens)
	if err != nil {
		return 0, 0, err
	}
	input, err = usagenorm.AddTokenCounts(usagenorm.FieldInputTokens, input, toolUsePrompt)
	if err != nil {
		return 0, 0, err
	}
	return input, cacheRead, nil
}

func normalizeOutputUsage(wire usageMetadata) (content.TokenCount, content.TokenCount, error) {
	candidates, err := wire.CandidatesTokenCount.TokenCount(usagenorm.FieldOutputTokens)
	if err != nil {
		return 0, 0, err
	}
	reasoning, err := wire.ThoughtsTokenCount.TokenCount(usagenorm.FieldReasoningTokens)
	if err != nil {
		return 0, 0, err
	}
	output, err := usagenorm.AddTokenCounts(usagenorm.FieldOutputTokens, candidates, reasoning)
	return output, reasoning, err
}

// validateReportedTotal keeps totalTokenCount's own per-field validation without
// reconciling it against the components. A present total that is negative, null,
// fractional or out of int64 range is a malformed field and stays an error; a
// total that merely disagrees with the sum of the members this codec models is
// an accounting difference and must not cost the caller the generation. See
// normalizeUsage for why the reconciliation was removed.
func validateReportedTotal(wire usageMetadata) error {
	if !wire.TotalTokenCount.Present() {
		return nil
	}
	_, err := wire.TotalTokenCount.TokenCount(usagenorm.FieldTotalTokens)
	return err
}

// buildBlocks maps candidate parts to content blocks, preserving Gemini's part
// order (which interleaves text, thoughts, and tool calls). A functionCall part
// becomes a ToolUseBlock; a thought-tagged part becomes a ThinkingBlock (built via
// content.NewThinkingBlock so its thoughtSignature, if present, is defensively
// copied into ProviderState for a same-dialect replay — see
// providerStateFromThoughtSignature); any other non-empty text part becomes a
// TextBlock. Empty text and unknown parts are skipped, EXCEPT a thought part
// carrying only a signature and no text: that still yields a (Thinking: "")
// ThinkingBlock rather than being dropped, since dropping it would silently lose
// the opaque continuation state a same-dialect follow-up request needs to replay.
// A functionCall part with no wire `id` (the Developer API's normal shape for
// parallel calls) gets a synthetic per-turn ordinal via toolCallID so its result
// stays addressable; see toolcallid.go.
func buildBlocks(parts []geminiPart) []content.Block {
	var blocks []content.Block
	callOrdinal := 0
	for _, p := range parts {
		switch {
		case p.FunctionCall != nil:
			blocks = append(blocks, content.NewToolUseBlock(toolCallID(p.FunctionCall.ID, callOrdinal), p.FunctionCall.Name,
				argsJSON(p.FunctionCall.Args), providerStateFromThoughtSignature(p.ThoughtSignature), providerStateFormatFor(p.ThoughtSignature)))
			callOrdinal++
		case p.Thought && (p.Text != "" || p.ThoughtSignature != ""):
			blocks = append(blocks, content.NewThinkingBlock(p.Text, "", providerStateFromThoughtSignature(p.ThoughtSignature), providerStateFormatFor(p.ThoughtSignature)))
		case p.Text != "":
			blocks = append(blocks, &content.TextBlock{Text: p.Text})
		}
	}
	return blocks
}

// providerStateFormatGemini tags a ThinkingBlock.ProviderState as having been
// produced by this codec (i.e. containing a Gemini thoughtSignature). Per the
// invariant documented on content.ThinkingBlock, every site in this package
// that forwards ProviderState onto the wire as thoughtSignature must first
// check ProviderStateFormat == providerStateFormatGemini; a ProviderState
// tagged with any other format (or untagged) originated from a different
// dialect and must be treated as absent, never replayed here.
const providerStateFormatGemini = "gemini"

// providerStateFromThoughtSignature marshals the wire `thoughtSignature`
// string into the json.RawMessage form ThinkingBlock.ProviderState carries,
// or returns nil for an absent signature. Pairing with
// providerStateToThoughtSignature (encode.go) makes ProviderState always
// "the JSON-encoded form of the provider-opaque wire value" for this
// dialect, so it round-trips arbitrary bytes/characters through ordinary
// JSON string escaping — the same convention codec/openairesponses uses for
// its own encrypted_content field.
func providerStateFromThoughtSignature(sig string) json.RawMessage {
	if sig == "" {
		return nil
	}
	// json.Marshal of a string cannot fail.
	encoded, _ := json.Marshal(sig)
	return encoded
}

// providerStateFormatFor returns providerStateFormatGemini when sig is
// present, or "" otherwise — mirroring providerStateFromThoughtSignature's
// nil-for-empty behavior so a signature-less ThinkingBlock never carries a
// stray format tag with no ProviderState to justify it.
func providerStateFormatFor(sig string) string {
	if sig == "" {
		return ""
	}
	return providerStateFormatGemini
}
