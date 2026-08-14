package anthropicapi

import (
	"fmt"
	"strconv"
)

// UnsupportedBlockError is returned by the encoder when a content block has a
// concrete type the Anthropic Messages API dialect does not model (e.g. audio or
// document blocks). Block holds the Go type name for diagnosis. Callers may
// errors.As to detect an unencodable block rather than string-matching.
type UnsupportedBlockError struct {
	Block string
}

func (e *UnsupportedBlockError) Error() string {
	return "anthropicapi: unsupported content block type " + e.Block
}

// EmptyTextBlockError is returned by the encoder when a text content block
// carries no text. Anthropic's RequestTextBlock declares `text` required with
// minLength 1, so an empty text block is not a representable wire shape: it
// would encode to the invalid `{"type":"text"}` and draw an HTTP 400. The codec
// refuses it here instead, fail-secure like UnsupportedBlockError, so the defect
// surfaces at the call site rather than as an opaque provider rejection.
type EmptyTextBlockError struct{}

func (e *EmptyTextBlockError) Error() string {
	return "anthropicapi: empty text content block (Anthropic requires text to be non-empty)"
}

// UnsupportedImageMediaTypeError is returned by the encoder when an inline
// image block carries a media type outside Anthropic's Base64ImageSource enum
// (image/jpeg, image/png, image/gif, image/webp). The shared content.MediaType
// vocabulary is wider — content.MediaTypeImageSVG has no Anthropic equivalent —
// so this is a representable Looprig block with no representable wire form,
// refused here for the same reason as EmptyTextBlockError.
type UnsupportedImageMediaTypeError struct {
	MediaType string
}

func (e *UnsupportedImageMediaTypeError) Error() string {
	return "anthropicapi: unsupported image media type " + e.MediaType +
		" (Anthropic accepts image/jpeg, image/png, image/gif, image/webp)"
}

// UnsupportedDocumentError is returned by the encoder when a document block has
// no legal RequestDocumentBlock form. Reason names the violated constraint.
//
// The neutral content.DocumentBlock is wider than Anthropic's source union in
// both directions that matter. Base64PDFSource.media_type is const
// "application/pdf" and PlainTextSource.media_type is const "text/plain", so a
// DOCX payload or a markdown body has no representable source at all — and
// re-labelling either one to satisfy the const would forward a document the
// caller never described. RequestDocumentBlock.title is additionally capped at
// 500 characters, which a file name can exceed.
type UnsupportedDocumentError struct {
	Reason string
}

func (e *UnsupportedDocumentError) Error() string {
	return "anthropicapi: unsupported document content block: " + e.Reason
}

// UnsupportedAudioError is returned by the encoder for an audio content block.
//
// This is a hard limitation of the format, not a gap in this codec: the
// Anthropic Messages API document declares no audio content block in any
// request or response shape, and the substring "audio" does not occur anywhere
// in it. An audio block therefore has no wire form to encode toward, and there
// is nothing to route it to.
//
// It gets a typed error of its own rather than the generic
// UnsupportedBlockError because this path is reachable in ordinary operation:
// an MCP tool returning an audio result produces a content.AudioBlock that the
// harness clones and persists, so the failure surfaces on every subsequent turn
// of that session and the message has to say why.
type UnsupportedAudioError struct {
	MediaType string
}

func (e *UnsupportedAudioError) Error() string {
	message := "anthropicapi: the Anthropic Messages API has no audio content block"
	if e.MediaType != "" {
		message += " (block media type " + e.MediaType + ")"
	}
	return message
}

// UnsupportedRefusalError is returned by the encoder for a
// content.RefusalBlock.
//
// This is a hard limitation of the format, not a gap in this codec. Anthropic
// models a refusal as RESPONSE metadata — stop_reason "refusal" alongside a
// RefusalStopDetails object carrying a category and an explanation — and the
// Messages API request document declares no refusal content block in any
// position. A refusal therefore has no wire form to encode toward, and the
// alternatives are all worse than failing: sending it as `text` shows the model
// its own decline quoted back as something it said (the exact defect
// content.RefusalBlock exists to remove), and dropping it silently loses the
// fact that the turn was declined.
//
// Like UnsupportedAudioError it gets a typed error of its own rather than the
// generic UnsupportedBlockError, and for the same reason: the path is reachable
// in ordinary operation. A refused turn is stored in session history, so the
// failure surfaces on every subsequent turn of that session and the message has
// to say why.
type UnsupportedRefusalError struct{}

func (e *UnsupportedRefusalError) Error() string {
	return "anthropicapi: the Anthropic Messages API has no refusal content block; a refusal is response-only metadata (stop_reason \"refusal\" with stop_details), so it cannot be replayed in a request"
}

// InvalidToolUseIDError is returned by the encoder when a tool_use id or a
// tool_result tool_use_id cannot satisfy Anthropic's ^[a-zA-Z0-9_-]+$.
// Identifiers are minted by whichever provider issued the call, so a
// conversation replayed from a dialect with a wider class (Bedrock Converse
// permits "." and ":") can carry one Anthropic rejects. An empty id is the
// worse case: it is omitempty on the wire, so it does not travel as "" — the
// required property simply disappears.
type InvalidToolUseIDError struct {
	ID     string
	Reason string
}

func (e *InvalidToolUseIDError) Error() string {
	return "anthropicapi: invalid tool-use id " + quoteForDiagnostic(e.ID) + ": " + e.Reason
}

// InvalidToolNameError is returned by the encoder when a tool name cannot
// satisfy Anthropic's ^[a-zA-Z0-9_-]{1,128}$. Tool servers, MCP ones in
// particular, routinely publish names containing "." or "/".
type InvalidToolNameError struct {
	Name   string
	Reason string
}

func (e *InvalidToolNameError) Error() string {
	return "anthropicapi: invalid tool name " + quoteForDiagnostic(e.Name) + ": " + e.Reason
}

// InvalidToolSchemaError is returned by the encoder when a tool's input schema
// is not a JSON object, which Anthropic's InputSchema requires.
type InvalidToolSchemaError struct {
	Name   string
	Reason string
}

func (e *InvalidToolSchemaError) Error() string {
	return "anthropicapi: invalid input_schema for tool " + quoteForDiagnostic(e.Name) + ": " + e.Reason
}

// SamplingRangeError is returned by the encoder when temperature or top_p falls
// outside the [0, 1] interval Anthropic declares. The shared model.Sampling
// vocabulary is wider (an OpenAI-shaped temperature runs to 2), so switching a
// session onto an Anthropic model can carry a value the API refuses.
type SamplingRangeError struct {
	Field string
	Value float64
}

func (e *SamplingRangeError) Error() string {
	return fmt.Sprintf("anthropicapi: %s must be between 0 and 1, got %v", e.Field, e.Value)
}

// UndeclaredThinkingDialectError is returned by the encoder when a request asks
// for reasoning from a model advertised as thinking-capable whose
// Caps.ThinkingDialect the catalogue never declared.
//
// Anthropic serves two on-modes concurrently and rejects the wrong one with an
// HTTP 400 rather than degrading: measured on 2026-08-13, claude-haiku-4-5
// answers `{"type":"adaptive"}` with "adaptive thinking is not supported on
// this model", and claude-sonnet-5 answers
// `{"type":"enabled","budget_tokens":N}` with "\"thinking.type.enabled\" is not
// supported for this model. Use \"thinking.type.adaptive\" and
// \"output_config.effort\"". Nothing in the request document distinguishes
// them, so with no declared dialect the encoder has a coin flip between two
// bodies, one of which is provably rejected.
//
// It therefore fails closed and names the model, because that is the piece of
// information the fix needs: a provider 400 says the request was wrong, this
// says which catalogue row is incomplete. The provider's WithThinking escape
// hatch remains available for a caller who knows better than the catalogue.
type UndeclaredThinkingDialectError struct {
	Model string
}

func (e *UndeclaredThinkingDialectError) Error() string {
	return "anthropicapi: model " + quoteForDiagnostic(e.Model) +
		" is marked thinking-capable but declares no Caps.ThinkingDialect;" +
		" Anthropic rejects the wrong thinking spelling with an HTTP 400, so the codec will not guess" +
		" between \"adaptive\" and \"budget\""
}

// ThinkingBudgetError is returned by the encoder when the budget dialect has no
// legal budget_tokens available for the request's max_tokens.
//
// Two constraints bound the field and only one of them is in the schema.
// ThinkingConfigEnabled declares budget_tokens with minimum 1024; Anthropic
// documents separately that it must be less than max_tokens, which is a
// cross-field rule no JSON Schema keyword in the derived document expresses —
// the gate was measured accepting budget_tokens 99999 against max_tokens 1024.
// Both together make max_tokens 1025 the smallest cap that admits any legal
// budget, so a smaller cap is refused here, where the diagnostic can name the
// field, rather than at the provider.
type ThinkingBudgetError struct {
	Model     string
	MaxTokens int
}

func (e *ThinkingBudgetError) Error() string {
	return fmt.Sprintf("anthropicapi: model %s uses the budget thinking dialect, whose budget_tokens must be at least %d"+
		" and strictly below max_tokens, so max_tokens %d admits no legal budget",
		quoteForDiagnostic(e.Model), minThinkingBudgetTokens, e.MaxTokens)
}

// quoteForDiagnostic renders an identifier for an error message. Identifiers
// are caller-supplied rather than provider-supplied, so quoting them is safe;
// the quoting exists to make an empty or whitespace value visible.
func quoteForDiagnostic(value string) string { return strconv.Quote(value) }

// UnsupportedConversationError is returned by the encoder when a conversation
// turn has a concrete type outside the closed content.Conversation union the
// dialect maps (user / assistant / tool-result / system). Conversation holds the
// Go type name for diagnosis.
type UnsupportedConversationError struct {
	Conversation string
}

func (e *UnsupportedConversationError) Error() string {
	return "anthropicapi: unsupported conversation type " + e.Conversation
}

// ConversationCollisionError reports adjacent neutral turns that cannot be
// combined without violating Anthropic's tool-result ordering rules.
type ConversationCollisionError struct {
	Reason string
}

func (e *ConversationCollisionError) Error() string {
	return "anthropicapi: conversation collision: " + e.Reason
}

// StreamEventDecodeError reports malformed JSON inside an otherwise
// successfully framed Anthropic Messages stream. A truncated or corrupt frame is
// indistinguishable from a dropped one, so it is a terminal decode failure
// rather than a skip: swallowing it would let a stream that lost content still
// report an authoritative clean success. Unknown-but-VALID event types remain
// tolerant skips — only unparseable bytes reach here.
type StreamEventDecodeError struct{ Err error }

func (e *StreamEventDecodeError) Error() string {
	return "anthropicapi: malformed stream event: " + e.Err.Error()
}

func (e *StreamEventDecodeError) Unwrap() error { return e.Err }

// StreamAPIError reports an Anthropic error event received after a streaming
// request crossed the successful HTTP-status boundary. It retains only the
// provider's structured error type and message, never the raw response frame.
type StreamAPIError struct {
	Type    string
	Message string
}

func (e *StreamAPIError) Error() string {
	message := "anthropicapi: stream error"
	if e.Type != "" {
		message += " (" + e.Type + ")"
	}
	if e.Message != "" {
		message += ": " + e.Message
	}
	return message
}

// ForeignThinkingSignatureError reports a reasoning signature this dialect
// cannot prove it minted, either because another dialect's label is attached
// (Format names it) or because no label is attached at all (Format is empty).
//
// It is a hard error rather than a degrade, which is the one design decision in
// this type worth stating. Probed against api.anthropic.com with
// claude-haiku-4-5 on 2026-08-13: a verbatim signature is accepted; a signature
// with eight characters changed draws
// `messages.N.content.0: Invalid "signature" in "thinking" block` (HTTP 400);
// and an EMPTY signature draws that same 400. So both available degrades lose.
// Forwarding a foreign signature sends a request that is certain to be
// rejected, and stripping it sends an unsigned thinking block that is equally
// certain to be rejected — while destroying the only copy of the continuation
// state on the way. Failing here costs the same turn and names the cause.
//
// The realistic source is not an attacker. Bedrock Converse and the Messages
// API are two endpoints for the same Claude models; their thinking blocks are
// structurally identical and their signatures are not interchangeable, so a
// session moved between them carries a block nothing else can tell apart.
type ForeignThinkingSignatureError struct {
	// Format is the label carried by the signature, or "" when it carries none.
	Format string
}

func (e *ForeignThinkingSignatureError) Error() string {
	origin := "no dialect label"
	if e.Format != "" {
		origin = "dialect " + quoteForDiagnostic(e.Format)
	}
	return "anthropicapi: refusing to replay a thinking signature minted by " + origin +
		"; this dialect replays only signatures labelled " + quoteForDiagnostic(signatureFormatAnthropic) +
		", and an unsigned thinking block is rejected too, so the signature can be neither forwarded nor dropped"
}
