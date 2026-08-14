package anthropicapi

import (
	"encoding/json"
	"fmt"

	"github.com/looprig/inference/internal/usagenorm"
)

// Wire-value constants for the Anthropic Messages API. Centralized so the encode,
// decode, and stream-event paths cannot drift on a string literal.
const (
	roleUser      = "user"
	roleAssistant = "assistant"

	blockTypeText             = "text"
	blockTypeImage            = "image"
	blockTypeDocument         = "document"
	blockTypeThinking         = "thinking"
	blockTypeRedactedThinking = "redacted_thinking"
	blockTypeToolUse          = "tool_use"
	blockTypeToolResult       = "tool_result"

	// `source.type` discriminators. base64 and url are shared by the image and
	// document blocks (Base64ImageSource/Base64PDFSource and
	// URLImageSource/URLPDFSource have identical shapes); text and content are
	// document-only (PlainTextSource, ContentBlockSource).
	sourceTypeBase64  = "base64"
	sourceTypeURL     = "url"
	sourceTypeText    = "text"
	sourceTypeContent = "content"

	// Base64PDFSource.media_type and PlainTextSource.media_type are both const
	// in the request document, so these are the only two media types a document
	// source may declare, whatever the caller's block carries.
	documentMediaTypePDF  = "application/pdf"
	documentMediaTypeText = "text/plain"

	// maxDocumentTitleLength is RequestDocumentBlock.title's maxLength.
	maxDocumentTitleLength = 500

	// The two ON-modes of ThinkingConfigParam, selected per model by
	// Caps.ThinkingDialect. The union's third arm, "disabled", has no constant
	// because the codec never emits it: omitting `thinking` entirely is the
	// same request, and naming a value the encoder cannot produce would invite
	// a caller to reach for it.
	thinkingTypeAdaptive = "adaptive"
	thinkingTypeEnabled  = "enabled"

	// minThinkingBudgetTokens is ThinkingConfigEnabled.budget_tokens' declared
	// minimum. Transcribed from the request document rather than assumed: the
	// gate rejects 1023 and accepts 1024.
	minThinkingBudgetTokens = 1024

	outputFormatJSONSchema = "json_schema"
	toolChoiceAny          = "any"
	toolChoiceTool         = "tool"

	cacheControlEphemeral = "ephemeral"

	// callerTypeDirect is the `caller.type` of a tool_use block the client is
	// expected to execute. The full union is DirectCaller | ServerToolCaller |
	// ServerToolCaller_20260120; the two server-tool members describe calls
	// Anthropic's own hosted tools issued, which content.ToolUseBlock cannot
	// represent, so direct is both the only value this codec emits and — as an
	// allowlist, so a member added later fails closed — the only one it accepts
	// (see checkCaller, server_decode.go).
	callerTypeDirect = "direct"

	responseTypeError = "error"

	// SSE event `type` values.
	eventContentBlockStart = "content_block_start"
	eventContentBlockDelta = "content_block_delta"
	eventContentBlockStop  = "content_block_stop"
	eventMessageStart      = "message_start"
	eventMessageDelta      = "message_delta"
	eventMessageStop       = "message_stop"

	// content_block_delta `delta.type` values.
	deltaText      = "text_delta"
	deltaThinking  = "thinking_delta"
	deltaSignature = "signature_delta"
	deltaInputJSON = "input_json_delta"

	// emptyObject is the fallback for a tool_use `input`: Anthropic requires
	// input to be a JSON object, so an empty ToolUseBlock.Input becomes "{}".
	emptyObject = "{}"

	// defaultSchema is the fallback for a tool with no schema; Anthropic requires
	// input_schema to be a JSON object.
	defaultSchema = `{"type":"object"}`
)

// defaultMaxTokens is the max_tokens value sent when the effective Sampling
// leaves MaxTokens unset. Anthropic REQUIRES max_tokens on every request, so a
// codec-level default is mandatory. 4096 is the conservative Anthropic-example
// default: large enough for typical replies, small enough to avoid SDK/HTTP
// timeouts on non-streaming calls. Callers wanting long outputs set
// Sampling.MaxTokens explicitly (and should stream — see the transport).
const defaultMaxTokens = 4096

// messagesRequest is the Anthropic `POST /v1/messages` request body. Field order
// is irrelevant on the wire (JSON is an unordered object); it is laid out to
// read model → conversation → tools → sampling → thinking → stream.
type messagesRequest struct {
	Model         string             `json:"model"`
	System        *systemPrompt      `json:"system,omitempty"`
	Messages      []anthropicMessage `json:"messages"`
	Tools         []anthropicTool    `json:"tools,omitempty"`
	ToolChoice    *toolChoice        `json:"tool_choice,omitempty"`
	MaxTokens     int                `json:"max_tokens"`
	Temperature   *float64           `json:"temperature,omitempty"`
	TopP          *float64           `json:"top_p,omitempty"`
	StopSequences []string           `json:"stop_sequences,omitempty"`
	Thinking      *thinkingConfig    `json:"thinking,omitempty"`
	OutputConfig  *outputConfig      `json:"output_config,omitempty"`
	Stream        bool               `json:"stream,omitempty"`
}

// cacheControl is the `cache_control` breakpoint marker on a content block.
// Only the ephemeral type exists on the wire today (default 5-minute TTL).
type cacheControl struct {
	Type string `json:"type"`
}

// systemPrompt marshals the top-level `system` field. Without a breakpoint it
// emits the plain-string form — byte-compatible with the pre-caching wire
// shape. With Cache set it emits the array-of-blocks form, because a bare
// string cannot carry a cache_control marker.
type systemPrompt struct {
	Text  string
	Cache bool
}

func (s systemPrompt) MarshalJSON() ([]byte, error) {
	if !s.Cache {
		return json.Marshal(s.Text)
	}
	return json.Marshal([]anthropicBlock{{
		Type:         blockTypeText,
		Text:         s.Text,
		CacheControl: &cacheControl{Type: cacheControlEphemeral},
	}})
}

// thinkingConfig is the `thinking` request field. ThinkingConfigParam is a
// three-arm tagged union whose arms have DIFFERENT required sets —
// ThinkingConfigEnabled requires [budget_tokens, type], ThinkingConfigAdaptive
// and ThinkingConfigDisabled require [type] alone — and all three close
// additionalProperties. A shared struct with an optional `budget_tokens` tag is
// therefore wrong in both directions, and the gate was measured rejecting both
// mistakes: `{"type":"enabled"}` fails "missing property 'budget_tokens'"
// (omitempty on a zero budget erases the required key), and
// `{"type":"adaptive","budget_tokens":2048}` fails "additional properties
// 'budget_tokens' not allowed". This is the same defect class that produced the
// illegal `thinking` CONTENT block (see thinkingWireBlock), so it gets the same
// remedy: a marshaller per variant.
//
// Which variant is emitted is the model's ThinkingDialect, never a default —
// see UndeclaredThinkingDialectError.
type thinkingConfig struct {
	Type string
	// BudgetTokens is meaningful only for thinkingTypeEnabled.
	BudgetTokens int
}

// thinkingEnabledWire is ThinkingConfigEnabled: budget_tokens is required, so
// it carries no omitempty.
type thinkingEnabledWire struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens"`
}

// thinkingAdaptiveWire is ThinkingConfigAdaptive, which closes
// additionalProperties and so must not carry budget_tokens at all.
type thinkingAdaptiveWire struct {
	Type string `json:"type"`
}

// MarshalJSON routes to the variant the Type names. Unknown types are an error
// rather than a best-effort object: an allowlist means a member added to
// ThinkingConfigParam later fails closed here instead of leaking a shape this
// codec never validated.
//
// The enabled arm converts rather than listing fields, and that is deliberate:
// a conversion stops compiling the moment thinkingConfig gains a field, forcing
// whoever adds it to decide what the wire should carry. A field-by-field
// literal would keep compiling and silently drop the new field — the "never
// silently drop caller intent" failure this codec has already been bitten by.
// The adaptive arm cannot convert, because ThinkingConfigAdaptive closes
// additionalProperties and so must not carry budget_tokens at all.
func (c thinkingConfig) MarshalJSON() ([]byte, error) {
	switch c.Type {
	case thinkingTypeAdaptive:
		return json.Marshal(thinkingAdaptiveWire{Type: c.Type})
	case thinkingTypeEnabled:
		return json.Marshal(thinkingEnabledWire(c))
	default:
		return nil, fmt.Errorf("anthropicapi: unknown thinking variant %q", c.Type)
	}
}

// outputConfig carries independent effort and structured-output controls.
type outputConfig struct {
	Effort string        `json:"effort,omitempty"`
	Format *outputFormat `json:"format,omitempty"`
}

type outputFormat struct {
	Type   string          `json:"type"`
	Schema json.RawMessage `json:"schema"`
}

// toolChoice is the `tool_choice` request field, a tagged union whose members
// carry different required sets: ToolChoiceAny and ToolChoiceAuto require only
// `type` and close the object, while ToolChoiceTool requires `name` as well.
// Name therefore keeps omitempty — an "any" object carrying it would be
// rejected by additionalProperties:false — and the non-empty-name guarantee
// for the "tool" variant is carried upstream by
// inference.ValidateRequestFeatures, which refuses a named choice whose name is
// not a declared tool before any encoding happens.
type toolChoice struct {
	Type string `json:"type"`
	Name string `json:"name,omitempty"`
}

// anthropicTool is one entry of the `tools` array.
type anthropicTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
}

// anthropicMessage is one entry of the `messages` array: a role plus an ordered
// array of content blocks.
type anthropicMessage struct {
	Role    string           `json:"role"`
	Content []anthropicBlock `json:"content"`
}

// anthropicBlock is the wire form of one Anthropic content block. Anthropic
// content blocks are a tagged union discriminated by `type`; this single struct
// (with omitempty on every optional field) is the serialization DTO for that
// union — a common Go pattern for tagged-union wire formats that keeps the codec
// strictly typed (no interface{}). Only the fields relevant to a given `type`
// are populated by the encoders; the rest stay zero and drop out via omitempty.
// It is reused for DECODE of response content blocks (text / thinking / tool_use),
// where the extra fields simply stay zero.
type anthropicBlock struct {
	Type string `json:"type"`

	// text
	Text string `json:"text,omitempty"`

	// image / document. Both variants name their payload `source`, and their
	// source unions overlap (base64 and url are shape-identical between
	// Base64ImageSource/Base64PDFSource and URLImageSource/URLPDFSource), so one
	// DTO carries both; blockSource.MarshalJSON emits only the members the
	// selected variant declares.
	Source *blockSource `json:"source,omitempty"`

	// document. RequestDocumentBlock.title is optional with minLength 1 and
	// maxLength 500, so it is omitempty here and length-checked at encode.
	Title string `json:"title,omitempty"`

	// thinking / redacted_thinking. Thinking and Signature are read on decode
	// and written by thinkingWireBlock, never through these tags — the
	// MarshalJSON below routes a `thinking` block to its own required-field
	// shape. Data still marshals from here for redacted_thinking.
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`
	Data      string `json:"data,omitempty"`

	// text (RESPONSE direction, and the replay of one). Citations is required
	// on ResponseTextBlock and optional-but-declared on RequestTextBlock, so a
	// client echoing a real Anthropic assistant turn back to this gateway
	// always carries it. Before this field existed the strict server decode
	// (DisallowUnknownFields) answered such a replay with
	// `malformed_body: json: unknown field "citations"` — an HTTP 400 on a body
	// Anthropic's OWN request schema declares legal. It is decode-direction
	// only here: omitempty keeps it off every outbound request block, where
	// only RequestTextBlock declares it and every other variant closes
	// additionalProperties. The response direction emits it unconditionally
	// through responseTextWireBlock (server_encode.go), which is the only shape
	// allowed to.
	Citations json.RawMessage `json:"citations,omitempty"`

	// tool_use
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	// tool_use `caller`, the ResponseToolUseBlock counterpart of Citations:
	// required on the response side, declared-optional on the request side, and
	// likewise rejected as an unknown field before this. Same omitempty
	// rationale; responseToolUseWireBlock owns the response-direction emit.
	Caller json.RawMessage `json:"caller,omitempty"`

	// tool_result
	ToolUseID string           `json:"tool_use_id,omitempty"`
	Content   []anthropicBlock `json:"content,omitempty"`
	IsError   bool             `json:"is_error,omitempty"`

	// cache breakpoint marker (encode only; never populated on decode)
	CacheControl *cacheControl `json:"cache_control,omitempty"`
}

// thinkingWireBlock is the dedicated marshal shape of a `thinking` content
// block. The shared anthropicBlock DTO puts omitempty on every optional field —
// load-bearing for the other variants — but Anthropic's RequestThinkingBlock
// declares required = [signature, thinking, type] with additionalProperties =
// false. Current-generation models default to display:"omitted" and return
// {"type":"thinking","thinking":"","signature":"..."}; marshaling that through
// the shared DTO drops the empty `thinking` key and the replayed block is
// rejected with an HTTP 400 on the next turn. The thinking variant therefore
// gets its own struct with no omitempty and no borrowed fields, so both required
// strings are always on the wire and nothing outside the schema ever is.
type thinkingWireBlock struct {
	Type      string `json:"type"`
	Thinking  string `json:"thinking"`
	Signature string `json:"signature"`
}

// redactedThinkingWireBlock is the dedicated marshal shape of a
// `redacted_thinking` content block, for the same reason as thinkingWireBlock:
// RequestRedactedThinkingBlock declares required = [data, type] with
// additionalProperties = false, and the shared DTO's omitempty drops `data`
// when the opaque payload decoded to an empty string.
type redactedThinkingWireBlock struct {
	Type string `json:"type"`
	Data string `json:"data"`
}

// MarshalJSON dispatches the tagged union to the variant's own wire shape. The
// two reasoning variants each declare additionalProperties = false with all
// their fields required, so they cannot borrow the shared DTO's omitempty
// behavior; every other type marshals through it via a defined type that drops
// this method and so cannot recurse.
func (b anthropicBlock) MarshalJSON() ([]byte, error) {
	switch b.Type {
	case blockTypeThinking:
		return json.Marshal(thinkingWireBlock{Type: b.Type, Thinking: b.Thinking, Signature: b.Signature})
	case blockTypeRedactedThinking:
		return json.Marshal(redactedThinkingWireBlock{Type: b.Type, Data: b.Data})
	}
	type sharedWireBlock anthropicBlock
	return json.Marshal(sharedWireBlock(b))
}

// blockSource is the `source` object of an image or document block. Anthropic
// models it as a tagged union discriminated by `type`, with four members
// between the two block kinds:
//
//	base64  Base64ImageSource / Base64PDFSource  required [data, media_type, type]
//	url     URLImageSource / URLPDFSource        required [type, url]
//	text    PlainTextSource                      required [data, media_type, type]
//	content ContentBlockSource                   required [content, type]
//
// Content is decode-direction only: the neutral vocabulary has no nested
// document-content representation, so the encoders never populate it and the
// server decoder refuses a source that carries it. It exists on the DTO so that
// form is REJECTED by name rather than by DisallowUnknownFields reporting an
// unrecognized field on a source the document plainly declares.
type blockSource struct {
	Type      string          `json:"type"`
	MediaType string          `json:"media_type,omitempty"`
	Data      string          `json:"data,omitempty"`
	URL       string          `json:"url,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
}

// payloadSourceWire is the marshal shape of the two payload-carrying source
// members, base64 and text. They differ only in their `type` and `media_type`
// constants, and both declare required = [data, media_type, type], so neither
// key may carry omitempty: an empty document body or an empty inline image
// would otherwise drop a required property, the exact defect the reasoning
// blocks were repaired for.
type payloadSourceWire struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

// urlSourceWire is the marshal shape of the url member, whose required set is
// [type, url] and which declares no media_type at all.
type urlSourceWire struct {
	Type string `json:"type"`
	URL  string `json:"url"`
}

// contentSourceWire is the marshal shape of ContentBlockSource, required =
// [content, type]. No encoder emits it today (see blockSource.Content); the
// shape exists so the union is complete rather than partially modelled.
type contentSourceWire struct {
	Type    string          `json:"type"`
	Content json.RawMessage `json:"content"`
}

// MarshalJSON dispatches the source union to the member's own wire shape, for
// the same reason anthropicBlock.MarshalJSON does: the members declare
// different required sets under additionalProperties = false, so no single
// omitempty-tagged struct can serialize all of them legally.
func (s blockSource) MarshalJSON() ([]byte, error) {
	switch s.Type {
	case sourceTypeBase64, sourceTypeText:
		return json.Marshal(payloadSourceWire{Type: s.Type, MediaType: s.MediaType, Data: s.Data})
	case sourceTypeURL:
		return json.Marshal(urlSourceWire{Type: s.Type, URL: s.URL})
	case sourceTypeContent:
		return json.Marshal(contentSourceWire{Type: s.Type, Content: s.Content})
	}
	type sharedWireSource blockSource
	return json.Marshal(sharedWireSource(s))
}

// messageResponse is the non-streaming `POST /v1/messages` response body and the
// shape carried inside a `message_start` stream event. Usage is a pointer so its
// absence is distinguishable from a zeroed count. Error is populated only when
// Type == "error".
type messageResponse struct {
	ID         string           `json:"id"`
	Type       string           `json:"type"`
	Role       string           `json:"role"`
	Model      string           `json:"model"`
	Content    []anthropicBlock `json:"content"`
	StopReason string           `json:"stop_reason"`
	Usage      *messageUsage    `json:"usage"`
	Error      *anthropicError  `json:"error"`
}

// messageUsage is the `usage` object of a message response.
type messageUsage struct {
	InputTokens         usagenorm.Count             `json:"input_tokens"`
	OutputTokens        usagenorm.Count             `json:"output_tokens"`
	CacheReadTokens     usagenorm.Count             `json:"cache_read_input_tokens"`
	CacheCreationTokens usagenorm.Count             `json:"cache_creation_input_tokens"`
	OutputTokensDetails *messageOutputTokensDetails `json:"output_tokens_details"`
}

type messageOutputTokensDetails struct {
	ThinkingTokens usagenorm.Count `json:"thinking_tokens"`
}

// anthropicError is the `error` object of an error-type response.
type anthropicError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// streamEvent is the union view of one de-framed SSE event the codec cares about.
// Content fields feed DecodeEvent; message, usage, and error fields feed the
// stream result collector without entering the content chunk vocabulary.
type streamEvent struct {
	Type         string           `json:"type"`
	Index        int              `json:"index"`
	ContentBlock *streamBlock     `json:"content_block"`
	Delta        *streamDelta     `json:"delta"`
	Message      *messageResponse `json:"message"`
	Usage        *messageUsage    `json:"usage"`
	Error        *anthropicError  `json:"error"`
}

// streamBlock is the `content_block` object on a content_block_start event. The
// codec reads Type (to detect tool_use) plus the tool_use ID and Name.
type streamBlock struct {
	Type string `json:"type"`
	ID   string `json:"id"`
	Name string `json:"name"`
	Data string `json:"data"`
}

// streamDelta is the `delta` object on content_block_delta and message_delta
// events. Content events populate one content field; message_delta can populate
// StopReason for terminal metadata.
type streamDelta struct {
	Type        string `json:"type"`
	Text        string `json:"text"`
	PartialJSON string `json:"partial_json"`
	Thinking    string `json:"thinking"`
	Signature   string `json:"signature"`
	StopReason  string `json:"stop_reason"`
}
