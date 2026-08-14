package openairesponses

import (
	"encoding/base64"
	"encoding/json"
	"strings"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	failure "github.com/looprig/inference/failure"
	"github.com/looprig/inference/internal/usagenorm"
	stream "github.com/looprig/inference/stream"
	usage "github.com/looprig/inference/usage"
)

// DecodeResponse parses a non-streaming OpenAI Responses API response body
// into a provider-neutral *inference.Response. A status:"failed" response is
// surfaced as a *failure.APIError, mirroring how anthropicapi.DecodeResponse
// handles its type:"error" envelope. An empty output array is a valid
// response, not an error.
func DecodeResponse(body []byte) (*inference.Response, error) {
	var wire wireResponse
	if err := json.Unmarshal(body, &wire); err != nil {
		return nil, err
	}

	if wire.Status == statusFailed {
		return nil, failure.NewAPIError(0, "", "", 0)
	}

	blocks, err := decodeOutputBlocks(wire.Output)
	if err != nil {
		return nil, err
	}

	u, err := normalizeUsage(wire.Usage)
	if err != nil {
		return nil, err
	}
	var messageUsage *content.Usage
	if u != nil {
		cloned := *u
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
		Usage:        u,
		FinishReason: deriveFinishReason(wire),
	}, nil
}

// deriveFinishReason derives the neutral stream.FinishReason from Responses'
// status/incomplete_details/output shape: any function_call in output means the
// model wants to call a tool regardless of status; otherwise
// status:"incomplete" with incomplete_details.reason:"max_output_tokens" means
// length; any other incomplete reason (or a missing one) is unknown;
// status:"completed" is a clean stop. status:"failed" is handled earlier
// (DecodeResponse) as an error, never reaching here as a finish reason.
//
// A refusal deliberately does NOT influence this. It arrives on a
// status:"completed" response and now travels as its own
// *content.RefusalBlock, which is a per-block signal every exhaustive consumer
// must handle; overriding the turn-level reason on top of that would report a
// content-filter intervention the provider did not report, and would make a
// refusal indistinguishable from a genuine filtered response. It would also
// suppress a tool call the model emitted alongside the refusal, which the
// caller still has to act on. See refusalBlocks.
func deriveFinishReason(wire wireResponse) stream.FinishReason {
	for _, item := range wire.Output {
		if item.Type == itemTypeFunctionCall {
			return stream.FinishReasonToolUse
		}
	}
	switch wire.Status {
	case statusCompleted:
		return stream.FinishReasonStop
	case statusIncomplete:
		if wire.IncompleteDetails != nil && wire.IncompleteDetails.Reason == incompleteReasonMaxOutputTokens {
			return stream.FinishReasonLength
		}
		return stream.FinishReasonUnknown
	default:
		return stream.FinishReasonUnknown
	}
}

func normalizeUsage(wire *wireUsage) (*usage.Usage, error) {
	if wire == nil {
		return nil, nil
	}
	promptTotal, err := wire.InputTokens.TokenCount(usagenorm.FieldInputTokens)
	if err != nil {
		return nil, err
	}
	cacheRead, err := wire.InputTokensDetails.CachedTokens.TokenCount(usagenorm.FieldCacheReadTokens)
	if err != nil {
		return nil, err
	}
	cacheWrite, err := wire.InputTokensDetails.CacheWriteTokens.TokenCount(usagenorm.FieldCacheCreationTokens)
	if err != nil {
		return nil, err
	}
	// Responses reports input_tokens as the gross prompt total including
	// cached tokens (like Chat Completions' prompt_tokens), with cached_tokens
	// as a breakdown subset — not a separate additive count. Subtract it to
	// the neutral input/cache-read split, mirroring openaiapi's
	// normalizePromptUsage. Some compatible endpoints also report cache writes
	// in input_tokens_details, so subtract that breakdown as well.
	input, err := usagenorm.SubtractTokenCounts(usagenorm.FieldInputTokens, promptTotal, cacheRead, cacheWrite)
	if err != nil {
		return nil, err
	}
	output, err := wire.OutputTokens.TokenCount(usagenorm.FieldOutputTokens)
	if err != nil {
		return nil, err
	}
	reasoning, err := wire.OutputTokensDetails.ReasoningTokens.TokenCount(usagenorm.FieldReasoningTokens)
	if err != nil {
		return nil, err
	}
	// output_tokens_details.reasoning_tokens is documented as a subset of
	// output_tokens, so no addition is performed. The subset relationship is not
	// re-asserted as a gate either: it is an accounting fact about what the
	// server reported, and a violation must not cost the caller the response.
	// See codec/openaiapi.normalizeUsage for the report that proved it.
	u := usage.Usage{InputTokens: input, OutputTokens: output, CacheReadTokens: cacheRead, CacheCreationTokens: cacheWrite, ReasoningTokens: reasoning}
	return &u, nil
}

// decodeOutputBlocks maps a response's `output` items to neutral content
// blocks, in wire order: a message item's output_text parts become
// TextBlocks, a function_call item becomes a ToolUseBlock, and a reasoning
// item becomes a ThinkingBlock (its encrypted_content, if present, round-trips
// through ThinkingBlock.ProviderState for a same-dialect follow-up request —
// see opaqueStateFromWire). Unknown item types are skipped tolerantly, like
// anthropicapi.decodeBlocks skips unknown block types on response decode.
func decodeOutputBlocks(items []wireItem) ([]content.Block, error) {
	var out []content.Block
	for _, item := range items {
		switch item.Type {
		case itemTypeMessage:
			for _, part := range item.Content.parts(contentTypeOutputText) {
				switch part.Type {
				case contentTypeOutputText:
					out = append(out, &content.TextBlock{Text: part.Text})
				case contentTypeRefusal:
					out = append(out, refusalBlocks(part.Refusal)...)
				}
			}
		case itemTypeFunctionCall:
			args, err := decodeFunctionCallArguments(item.Arguments)
			if err != nil {
				return nil, err
			}
			id := item.CallID
			if id == "" {
				id = item.ID
			}
			out = append(out, &content.ToolUseBlock{ID: id, Name: item.Name, Input: args})
		case itemTypeReasoning:
			out = append(out, decodeReasoningItem(item.ID, item.Summary, item.EncryptedContent))
		default:
			// Skip item types the neutral vocabulary does not model
			// (tolerant response decode, matching anthropicapi.decodeBlocks).
		}
	}
	return out, nil
}

// refusalBlocks maps a `refusal` content part onto the neutral vocabulary's own
// refusal variant.
//
// The PART's presence is the signal, not its text. RefusalContent.required is
// ["type","refusal"], so a `type:"refusal"` part is a refusal even when the
// string is empty — a model may decline with no explanation, and dropping that
// would restore the exact bug the block type exists to fix: a refused turn
// decoding as a completed response with zero blocks. Callers therefore invoke
// this only from a confirmed refusal part.
//
// The turn-level finish reason is deliberately left alone; see
// deriveFinishReason.
func refusalBlocks(refusal string) []content.Block {
	return []content.Block{&content.RefusalBlock{Text: refusal}}
}

// decodeFunctionCallArguments converts the wire `arguments` JSON-string into
// the json.RawMessage a neutral ToolUseBlock.Input carries, defaulting an
// absent value to "{}".
func decodeFunctionCallArguments(raw string) (json.RawMessage, error) {
	if raw == "" {
		return json.RawMessage(emptyObject), nil
	}
	return json.RawMessage(raw), nil
}

// providerStateFormatOpenAIResponses tags a ThinkingBlock.ProviderState as
// having been produced by this codec (i.e. containing an OpenAI Responses
// encrypted_content value). Per the invariant documented on
// content.ThinkingBlock, every site in this package that forwards
// ProviderState onto the wire as encrypted_content must first check
// ProviderStateFormat == providerStateFormatOpenAIResponses; a ProviderState
// tagged with any other format (or untagged) originated from a different
// dialect and must be treated as absent, never replayed here.
const providerStateFormatOpenAIResponses = "openai-responses"

// decodeReasoningItem builds a ThinkingBlock from a reasoning item's id,
// summary parts (concatenated) and its opaque encrypted_content, via
// content.NewThinkingBlock so ProviderState is defensively copied.
func decodeReasoningItem(id string, summary []wireSummaryPart, encryptedContent string) *content.ThinkingBlock {
	var sb strings.Builder
	for i, part := range summary {
		if i > 0 {
			sb.WriteString("\n\n")
		}
		sb.WriteString(part.Text)
	}

	var providerState json.RawMessage
	var providerStateFormat string
	if id != "" || encryptedContent != "" {
		providerState = opaqueStateFromWire(id, encryptedContent)
		providerStateFormat = providerStateFormatOpenAIResponses
	}
	return content.NewThinkingBlock(sb.String(), "", providerState, providerStateFormat)
}

// reasoningState is the JSON shape ThinkingBlock.ProviderState carries for
// this dialect: everything a later request needs to replay the reasoning item
// verbatim. The id is not a diagnostic — ReasoningItem.required lists it, so
// without it the item cannot be replayed at all, and no neutral block field
// can carry it. Both members are provider-opaque; the codec never interprets
// them.
type reasoningState struct {
	ID               string `json:"id,omitempty"`
	EncryptedContent string `json:"encrypted_content,omitempty"`
}

// opaqueStateFromWire marshals a reasoning item's replayable members into the
// json.RawMessage form ThinkingBlock.ProviderState carries. Pairing with
// opaqueStateToWire (encode.go) makes ProviderState always "the JSON-encoded
// form of the provider-opaque wire item" for this dialect, so it round-trips
// arbitrary bytes/characters through ordinary JSON string escaping.
func opaqueStateFromWire(id, encryptedContent string) json.RawMessage {
	// json.Marshal of a struct of strings cannot fail.
	encoded, _ := json.Marshal(reasoningState{ID: id, EncryptedContent: encryptedContent})
	return encoded
}

// decodeImageURL parses an input_image `image_url`: a data: URI decodes to
// inline bytes, anything else is treated as a remote URL reference.
func decodeImageURL(raw string) (*content.ImageBlock, error) {
	if raw == "" {
		return nil, &ServerDecodeError{Reason: "missing_image_url"}
	}
	if strings.HasPrefix(raw, dataURIPrefix) {
		mediaType, data, err := parseDataURI(raw)
		if err != nil {
			return nil, err
		}
		return &content.ImageBlock{MediaType: content.MediaType(mediaType), Source: content.ImageSource{Data: data}}, nil
	}
	return &content.ImageBlock{Source: content.ImageSource{URL: raw}}, nil
}

// decodeInputFile parses an input_file content part into a DocumentBlock,
// inverting documentInputFilePart (encode.go). Both file_data spellings are
// accepted — a data: URI yields the document's media type alongside its bytes,
// a bare base64 payload yields the bytes alone — so either legal wire form
// round-trips. `file_id` and `file_url` are refused rather than dropped: they
// reference a file the neutral transcript does not contain, and accepting the
// part would hand the model an attachment nothing in the session holds.
func decodeInputFile(part wireContentPart) (*content.DocumentBlock, error) {
	if part.FileID != "" {
		return nil, &ServerDecodeError{Reason: "unsupported_file_reference", Detail: "file_id"}
	}
	if part.FileData == "" {
		return nil, &ServerDecodeError{Reason: "missing_file_data"}
	}
	if strings.HasPrefix(part.FileData, dataURIPrefix) {
		mediaType, data, err := parseDataURI(part.FileData)
		if err != nil {
			return nil, err
		}
		return &content.DocumentBlock{MediaType: content.MediaType(mediaType), Name: part.Filename, Data: data}, nil
	}
	data, err := base64.StdEncoding.DecodeString(part.FileData)
	if err != nil {
		return nil, &ServerDecodeError{Reason: "invalid_file_data", Detail: err.Error()}
	}
	return &content.DocumentBlock{Name: part.Filename, Data: data}, nil
}

func parseDataURI(raw string) (string, []byte, error) {
	rest := strings.TrimPrefix(raw, dataURIPrefix)
	comma := strings.IndexByte(rest, ',')
	if comma < 0 {
		return "", nil, &ServerDecodeError{Reason: "invalid_data_uri"}
	}
	meta := rest[:comma]
	payload := rest[comma+1:]
	if !strings.HasSuffix(meta, ";base64") {
		return "", nil, &ServerDecodeError{Reason: "unsupported_data_uri_encoding"}
	}
	mediaType := strings.TrimSuffix(meta, ";base64")
	data, err := base64.StdEncoding.DecodeString(payload)
	if err != nil {
		return "", nil, &ServerDecodeError{Reason: "invalid_data_uri_payload", Detail: err.Error()}
	}
	return mediaType, data, nil
}
