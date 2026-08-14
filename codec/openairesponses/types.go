// Package openairesponses is the OpenAI Responses API wire dialect
// (POST /v1/responses): a genuinely different, items-based shape from OpenAI
// Chat Completions (codec/openaiapi) — not a flat messages array. It is both
// a client-side codec.StreamingCodec (neutral -> wire, for calling a
// Responses-speaking target) and a server-side codec.ServerCodec (wire ->
// neutral, for serving a Responses-speaking harness such as Codex).
package openairesponses

import (
	"bytes"
	"encoding/json"

	"github.com/looprig/inference/internal/usagenorm"
)

// Wire-value constants. Centralized so encode, decode, and stream-event paths
// cannot drift on a string literal.
const (
	roleUser      = "user"
	roleAssistant = "assistant"
	roleSystem    = "system"
	// roleDeveloper is Responses' alternative to roleSystem for a system-role
	// input item; this codec treats it identically to roleSystem on decode.
	roleDeveloper = "developer"

	itemTypeMessage            = "message"
	itemTypeFunctionCall       = "function_call"
	itemTypeFunctionCallOutput = "function_call_output"
	itemTypeReasoning          = "reasoning"

	contentTypeInputText  = "input_text"
	contentTypeInputImage = "input_image"
	contentTypeInputFile  = "input_file"
	contentTypeOutputText = "output_text"
	// contentTypeRefusal is OutputContent's refusal member, carrying a
	// *content.RefusalBlock. It is emitted only where an OutputMessage is
	// buildable — the response direction, which synthesizes the item id the
	// schema requires — because OutputContent is reachable from no id-free
	// input item (see blocksToItems, encode.go).
	contentTypeRefusal = "refusal"

	toolTypeFunction = "function"

	toolChoiceAuto     = "auto"
	toolChoiceRequired = "required"
	toolChoiceNone     = "none"

	// toolChoiceRequiredJSON is the encoded "required" mode. tool_choice is
	// raw JSON on the wire struct, so the mode member has to arrive already
	// quoted.
	toolChoiceRequiredJSON = `"` + toolChoiceRequired + `"`

	textFormatPlainText  = "text"
	textFormatJSONSchema = "json_schema"

	summaryTypeText = "summary_text"

	// includeEncryptedReasoningContent is the `include` value that requests
	// encrypted reasoning content back from a Responses target, so a
	// same-dialect follow-up request can replay it via a `reasoning` input
	// item's encrypted_content.
	includeEncryptedReasoningContent = "reasoning.encrypted_content"

	statusInProgress = "in_progress"
	statusCompleted  = "completed"
	statusIncomplete = "incomplete"
	statusFailed     = "failed"

	incompleteReasonMaxOutputTokens = "max_output_tokens"

	imageDetailAuto = "auto"

	dataURIPrefix = "data:"

	// emptyObject is the fallback for a function_call's `arguments` (and a
	// tool's `parameters` schema): Responses requires these to be JSON
	// objects, so an empty neutral value becomes "{}".
	emptyObject = "{}"

	// defaultSchema is the fallback for a tool with no schema.
	defaultSchema = `{"type":"object"}`
)

// SSE event `type` values this codec handles, per the design doc's native
// event order plus the tool/reasoning/failure events required for full
// support.
const (
	eventResponseCreated = "response.created"
	// eventResponseInProgress is ResponseInProgressEvent. The union permits its
	// absence, so nothing in the schema or the frame gate notices a stream that
	// never sends one, but every real Responses provider emits it immediately
	// after response.created — and a client that treats "created" as "accepted,
	// not yet generating" and waits for it stalls against a server that does not.
	eventResponseInProgress = "response.in_progress"
	eventOutputItemAdded    = "response.output_item.added"
	eventContentPartAdded   = "response.content_part.added"
	eventOutputTextDelta    = "response.output_text.delta"
	// eventOutputTextDone is ResponseTextDoneEvent, the terminal for the
	// output_text channel. It is not optional in practice: a client that
	// reconstructs assistant text from output_text.* alone — which is the
	// documented way and what a client ignoring content_part.* does — has no
	// other signal that the text finished. The refusal channel already had its
	// symmetric refusal.done; this one was simply missing.
	eventOutputTextDone        = "response.output_text.done"
	eventContentPartDone       = "response.content_part.done"
	eventOutputItemDone        = "response.output_item.done"
	eventResponseCompleted     = "response.completed"
	eventResponseIncomplete    = "response.incomplete"
	eventFunctionCallArgsDelta = "response.function_call_arguments.delta"
	eventFunctionCallArgsDone  = "response.function_call_arguments.done"
	eventReasoningSummaryDelta = "response.reasoning_summary_text.delta"
	eventReasoningSummaryDone  = "response.reasoning_summary_text.done"
	// eventRefusalDelta / eventRefusalDone are the refusal channel's streaming
	// events. Only the delta yields a chunk; the done event repeats the whole
	// refusal, exactly as response.output_text.done repeats the whole text.
	eventRefusalDelta   = "response.refusal.delta"
	eventRefusalDone    = "response.refusal.done"
	eventResponseFailed = "response.failed"
	// eventError is the spec's ResponseErrorEvent: a top-level `type:"error"`
	// event carrying code/message/param directly, with no enclosing `response`
	// object. It is a distinct failure channel from response.failed and must
	// terminate the stream just as hard.
	eventError = "error"
)

// --- request-direction wire shape (shared by encode.go and server_decode.go) ---

// wireItem is the wire form of one `input` (or `output`) array entry: a
// tagged union discriminated by Type. The struct tags below govern DECODE
// only — every variant of the union is decoded into this one flat shape, the
// same way anthropicapi folds its content blocks into a single struct.
//
// ENCODE is variant-aware and owned by MarshalJSON, because "one struct,
// omitempty per field" cannot express this union's rules: each variant has
// required members that must be emitted even when empty
// (function_call_output.output, reasoning.summary, function_call.call_id)
// while those same members must never appear on the other variants. An empty
// Type marks the EasyInputMessage form (role + content, no discriminator).
type wireItem struct {
	Type string `json:"type"`

	// message
	Role    string           `json:"role,omitempty"`
	Content *wireItemContent `json:"content,omitempty"`

	// function_call / function_call_output
	ID        string `json:"id,omitempty"`
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"` // JSON-encoded STRING on the wire
	Output    string `json:"output,omitempty"`

	// reasoning
	Summary          []wireSummaryPart `json:"summary,omitempty"`
	EncryptedContent string            `json:"encrypted_content,omitempty"`

	// Status is every item variant's "in_progress" | "completed" |
	// "incomplete", which the API populates on items it returns. It is
	// required on an OutputMessage, so the response direction emits it;
	// decoding it also keeps a client replaying items verbatim from being
	// rejected by the strict server-side decode.
	Status string `json:"status,omitempty"`
}

// MarshalJSON emits precisely the members the item's variant owns, per the
// union documented on wireItem. Required-but-empty members are emitted
// (an empty tool result still carries "output":"", a summary-less reasoning
// item still carries "summary":[]); members belonging to other variants never
// leak in.
func (i wireItem) MarshalJSON() ([]byte, error) {
	switch i.Type {
	case itemTypeMessage:
		// OutputMessage / InputMessage: id and status are required on the
		// former, so the response direction populates them; the request
		// direction has no id to give and uses the EasyInputMessage form
		// below instead.
		return json.Marshal(struct {
			Type    string           `json:"type"`
			ID      string           `json:"id,omitempty"`
			Role    string           `json:"role"`
			Status  string           `json:"status,omitempty"`
			Content *wireItemContent `json:"content"`
		}{i.Type, i.ID, i.Role, i.Status, i.Content})

	case itemTypeFunctionCall:
		// FunctionToolCall.required: type, call_id, name, arguments.
		return json.Marshal(struct {
			Type      string `json:"type"`
			ID        string `json:"id,omitempty"`
			CallID    string `json:"call_id"`
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
			Status    string `json:"status,omitempty"`
		}{i.Type, i.ID, i.CallID, i.Name, i.Arguments, i.Status})

	case itemTypeFunctionCallOutput:
		// FunctionCallOutputItemParam.required: type, output. call_id is what
		// pairs the result with its call, so it is emitted unconditionally
		// too rather than vanishing on an empty value.
		return json.Marshal(struct {
			Type   string `json:"type"`
			ID     string `json:"id,omitempty"`
			CallID string `json:"call_id"`
			Output string `json:"output"`
			Status string `json:"status,omitempty"`
		}{i.Type, i.ID, i.CallID, i.Output, i.Status})

	case itemTypeReasoning:
		// ReasoningItem.required: id, summary, type.
		summary := i.Summary
		if summary == nil {
			summary = []wireSummaryPart{}
		}
		return json.Marshal(struct {
			Type             string            `json:"type"`
			ID               string            `json:"id"`
			Summary          []wireSummaryPart `json:"summary"`
			EncryptedContent string            `json:"encrypted_content,omitempty"`
			Status           string            `json:"status,omitempty"`
		}{i.Type, i.ID, summary, i.EncryptedContent, i.Status})

	default:
		// EasyInputMessage.required: role, content. `type` is optional there
		// and deliberately omitted: with it absent, EasyInputMessage is the
		// only InputItem variant that can match a role-bearing item with no
		// id, which is exactly what replayed assistant history is.
		return json.Marshal(struct {
			Role    string           `json:"role"`
			Content *wireItemContent `json:"content"`
		}{i.Role, i.Content})
	}
}

// wireItemContent is an item's `content`: either the array of typed parts
// (input_text / input_image / output_text) every Item-flavored message uses,
// or the bare string an EasyInputMessage may carry. Both are valid wire
// forms, so the codec must decode either and choose which to emit — the same
// string-or-array problem openaiapi solves with wireChatContent.
//
// STRICTNESS IS DIRECTIONAL HERE, and the split is deliberate. This one
// UnmarshalJSON serves four paths:
//
//	server request decode  server_decode.go   wireDecodeRequest.Input
//	client response decode decode.go          wireResponse.Output
//	client stream decode   stream.go          the embedded Response's output
//	(encode is MarshalJSON's problem, not this one)
//
// The array branch below decodes LENIENTLY, and stays that way, for the sake
// of the last three: a provider adding a member to a content part must not
// break live inference. That is not hypothetical — OutputTextContent gained
// logprobs, which this codec had to grow a field for, and a strict decode
// would have turned that addition into a hard failure on responses that were
// perfectly usable. Content already generated is never discarded over a member
// we do not model.
//
// The first path needs the opposite answer, because its failure mode is the
// opposite: an unrecognised member in a REQUEST is caller intent, and dropping
// it silently produced an empty prompt served as a 200. Ingress therefore gets
// its strictness from rejectUnknownContentPartMembers (server_decode.go),
// which runs over the raw body and so can be strict in one direction only.
// Tightening the code below instead would take the response paths with it.
//
// This is the same asymmetry wireContentPart's Annotations/Logprobs comment
// records from the other side, and it is why neither half may be "unified"
// with the other.
type wireItemContent struct {
	// Parts is the array form; Text/IsText is the bare-string form. IsText
	// distinguishes an empty string from an empty array.
	Parts  []wireContentPart
	Text   string
	IsText bool
}

// textContent builds the bare-string EasyInputMessage content form.
func textContent(text string) *wireItemContent {
	return &wireItemContent{Text: text, IsText: true}
}

// partsContent builds the array content form, normalizing nil to an empty
// (but present) array since `content` is required wherever it appears.
func partsContent(parts []wireContentPart) *wireItemContent {
	if parts == nil {
		parts = []wireContentPart{}
	}
	return &wireItemContent{Parts: parts}
}

func (c wireItemContent) MarshalJSON() ([]byte, error) {
	if c.IsText {
		return json.Marshal(c.Text)
	}
	parts := c.Parts
	if parts == nil {
		parts = []wireContentPart{}
	}
	return json.Marshal(parts)
}

func (c *wireItemContent) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		return nil
	}
	switch trimmed[0] {
	case '"':
		if err := json.Unmarshal(trimmed, &c.Text); err != nil {
			return &ServerDecodeError{Reason: "invalid_item_content", Detail: err.Error()}
		}
		c.IsText = true
		return nil
	case '[':
		if err := json.Unmarshal(trimmed, &c.Parts); err != nil {
			return &ServerDecodeError{Reason: "invalid_item_content", Detail: err.Error()}
		}
		return nil
	default:
		return &ServerDecodeError{Reason: "invalid_item_content"}
	}
}

// parts returns the item's content as the array form, flattening the
// bare-string EasyInputMessage form into a single part of partType so every
// reader sees one shape. A nil receiver (absent content) yields nil.
func (c *wireItemContent) parts(partType string) []wireContentPart {
	switch {
	case c == nil:
		return nil
	case c.IsText:
		return []wireContentPart{{Type: partType, Text: c.Text}}
	default:
		return c.Parts
	}
}

// wireContentPart is one entry of an item's `content` array: input_text,
// input_image, input_file (request direction) or output_text (response
// direction, reused by wireOutputItem). Unlike Chat Completions' nested
// objects, every Responses part keeps its payload flat on the part itself, so
// one struct with omitempty covers the whole union for DECODE: each variant's
// members are disjoint and `type` alone discriminates. Annotations and Logprobs
// are carried so a client replaying a real OutputTextContent part is not
// rejected by the strict server decode; this codec never interprets either
// (citations and logprobs are out of scope), and preserves whatever it was
// given.
type wireContentPart struct {
	Type        string            `json:"type"`
	Text        string            `json:"text,omitempty"`
	ImageURL    string            `json:"image_url,omitempty"`
	Detail      string            `json:"detail,omitempty"`
	Filename    string            `json:"filename,omitempty"`
	FileData    string            `json:"file_data,omitempty"`
	FileID      string            `json:"file_id,omitempty"`
	Refusal     string            `json:"refusal,omitempty"`
	Annotations []json.RawMessage `json:"annotations,omitempty"`
	Logprobs    []json.RawMessage `json:"logprobs,omitempty"`
}

// outputTextWire is the marshal shape of OutputTextContent, required =
// ["type","text","annotations","logprobs"] — in BOTH directions; the request
// and response documents declare it identically. All four keys therefore drop
// omitempty.
//
// annotations and logprobs are arrays with no null member, so "" and "unknown"
// are not expressible: the only legal way to say the gateway produced no
// citations and was asked for no logprobs is the empty array, and that is
// exactly what an empty array means here. Anything a client sent is passed
// through unchanged rather than reset.
type outputTextWire struct {
	Type        string            `json:"type"`
	Text        string            `json:"text"`
	Annotations []json.RawMessage `json:"annotations"`
	Logprobs    []json.RawMessage `json:"logprobs"`
}

// MarshalJSON keeps the one-struct union honest for the two variants whose
// required members can legitimately be empty, which the shared `omitempty` tags
// would erase:
//
//   - RefusalContent.required is ["type","refusal"], and a model may decline
//     with no explanation, so an explanation-free refusal would become an
//     illegal part.
//   - OutputTextContent.required is ["type","text","annotations","logprobs"],
//     and the ordinary case has an empty annotations and logprobs — plus an
//     empty text on the content_part.added frame that opens a text part.
//
// Every other variant's payload is either non-empty by construction or
// genuinely optional, so they keep the tagged shape.
func (p wireContentPart) MarshalJSON() ([]byte, error) {
	switch p.Type {
	case contentTypeRefusal:
		return json.Marshal(struct {
			Type    string `json:"type"`
			Refusal string `json:"refusal"`
		}{p.Type, p.Refusal})
	case contentTypeOutputText:
		return json.Marshal(outputTextWire{
			Type:        p.Type,
			Text:        p.Text,
			Annotations: nonNilRaw(p.Annotations),
			Logprobs:    nonNilRaw(p.Logprobs),
		})
	}
	type plain wireContentPart
	return json.Marshal(plain(p))
}

// nonNilRaw returns raw, or an empty (but present) slice when it is nil, so a
// required array-typed member never marshals as null.
func nonNilRaw(raw []json.RawMessage) []json.RawMessage {
	if raw == nil {
		return []json.RawMessage{}
	}
	return raw
}

type wireSummaryPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// wireTool is a Responses FunctionTool. Strict has no omitempty: the spec's
// FunctionTool.required is ["type","name","strict","parameters"], so a tool
// object that leaves `strict` out is not a legal request body. It is always
// encoded false, because `strict` true additionally requires the tool schema
// to be a strict subset (every property required, additionalProperties:false)
// and inference.Tool carries arbitrary caller-supplied schemas this codec
// cannot certify.
type wireTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
	Strict      bool            `json:"strict"`
}

// namedToolChoice is ToolChoiceFunction, the forced-tool member of
// tool_choice: `{"type":"function","name":...}`. Both members are required, so
// neither carries omitempty. The name sits beside `type` here; the Chat
// Completions dialect nests the same value under a `function` object.
type namedToolChoice struct {
	Type string `json:"type"`
	Name string `json:"name"`
}

type wireText struct {
	Format *wireTextFormat `json:"format,omitempty"`
}

type wireTextFormat struct {
	Type   string          `json:"type"`
	Name   string          `json:"name,omitempty"`
	Schema json.RawMessage `json:"schema,omitempty"`
	Strict bool            `json:"strict,omitempty"`
}

type wireReasoningConfig struct {
	Effort  string `json:"effort,omitempty"`
	Summary string `json:"summary,omitempty"`
}

// wireRequest is the encode-direction (client, neutral -> wire) POST
// /v1/responses request body. ToolChoice is ToolChoiceParam, an anyOf over a
// mode enum and eight tool-selection objects, so it is carried as raw JSON:
// the codec emits the "required" mode or a ToolChoiceFunction object, and
// nothing else. The decode direction uses a json.RawMessage sibling
// (wireDecodeRequest, server_decode.go) so it can classify — and reject — the
// remaining forms real clients may send.
// Store is intentionally a non-omitempty bool: this project excludes
// server-stored conversations, so every request explicitly sends
// "store":false rather than relying on an unverified provider default.
type wireRequest struct {
	Model           string               `json:"model"`
	Instructions    string               `json:"instructions,omitempty"`
	Input           []wireItem           `json:"input"`
	Tools           []wireTool           `json:"tools,omitempty"`
	ToolChoice      json.RawMessage      `json:"tool_choice,omitempty"`
	MaxOutputTokens *int                 `json:"max_output_tokens,omitempty"`
	Temperature     *float64             `json:"temperature,omitempty"`
	TopP            *float64             `json:"top_p,omitempty"`
	Text            *wireText            `json:"text,omitempty"`
	Reasoning       *wireReasoningConfig `json:"reasoning,omitempty"`
	Include         []string             `json:"include,omitempty"`
	Store           bool                 `json:"store"`
	Stream          bool                 `json:"stream,omitempty"`
}

// --- response-direction wire shape ---------------------------------------
//
// A response's `output` array reuses wireItem (above) rather than a separate
// type: it is marshal-safe (no usagenorm.Count fields) and structurally
// identical to an input item for every item type this codec models (message,
// function_call, reasoning), so wireItem is shared verbatim by the
// client-decode direction (decode.go, stream.go) and the server-encode
// direction (server_encode.go, server_stream.go) — mirroring how
// anthropicapi's anthropicBlock is reused both ways for content blocks.

type wireIncompleteDetails struct {
	Reason string `json:"reason"`
}

type wireResponseError struct {
	Code    string `json:"code,omitempty"`
	Message string `json:"message"`
}

// wireResponse is the DECODE-direction (client, wire -> neutral) response
// envelope: Usage uses usagenorm.Count so it can decode a real provider
// value. It backs both the non-streaming response body (decode.go) and the
// `response` object embedded in response.created/completed/failed stream
// events (stream.go).
type wireResponse struct {
	ID                string                 `json:"id"`
	Status            string                 `json:"status"`
	Model             string                 `json:"model"`
	Output            []wireItem             `json:"output"`
	Usage             *wireUsage             `json:"usage"`
	IncompleteDetails *wireIncompleteDetails `json:"incomplete_details"`
	Error             *wireResponseError     `json:"error"`
}

type wireUsage struct {
	InputTokens         usagenorm.Count         `json:"input_tokens"`
	OutputTokens        usagenorm.Count         `json:"output_tokens"`
	InputTokensDetails  wireInputTokensDetails  `json:"input_tokens_details"`
	OutputTokensDetails wireOutputTokensDetails `json:"output_tokens_details"`
}

type wireInputTokensDetails struct {
	CachedTokens     usagenorm.Count `json:"cached_tokens"`
	CacheWriteTokens usagenorm.Count `json:"cache_write_tokens"`
}

type wireOutputTokensDetails struct {
	ReasoningTokens usagenorm.Count `json:"reasoning_tokens"`
}
