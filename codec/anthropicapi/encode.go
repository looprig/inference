package anthropicapi

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	model "github.com/looprig/inference/model"
)

// EncodeRequest converts a provider-neutral inference.Request into an Anthropic
// `POST /v1/messages` JSON body. stream=true adds "stream":true to the body.
// Request.System becomes the top-level `system` field; any SystemMessage in the
// thread is folded into it (Anthropic has no in-thread system role).
func EncodeRequest(req inference.Request, stream bool) ([]byte, error) {
	if err := inference.ValidateRequestFeatures(req); err != nil {
		return nil, err
	}
	r, err := buildMessagesRequest(req, stream)
	if err != nil {
		return nil, err
	}
	return json.Marshal(r)
}

type projectedMessageKind uint8

const (
	projectedOrdinary projectedMessageKind = iota
	projectedToolResults
	projectedToolResultsThenOrdinary
)

// buildMessagesRequest assembles the typed request struct. Split from marshaling
// so the mapping is unit-testable without a JSON round-trip.
func buildMessagesRequest(req inference.Request, stream bool) (messagesRequest, error) {
	// Effective sampling: a non-nil per-call Override wins over Model.Sampling.
	sampling := req.Model.Sampling
	if req.Override != nil {
		sampling = *req.Override
	}

	system := req.System
	var messages []anthropicMessage
	var messageKinds []projectedMessageKind
	var committedBlocks []int
	committedSourceMessages := len(req.Messages) - req.TransientMessages
	transientSystemMessageNonEmpty := false
	appendMessage := func(message anthropicMessage, kind projectedMessageKind, volatile bool) error {
		if len(messages) == 0 || messages[len(messages)-1].Role != message.Role {
			messages = append(messages, message)
			messageKinds = append(messageKinds, kind)
			if volatile {
				committedBlocks = append(committedBlocks, 0)
			} else {
				committedBlocks = append(committedBlocks, len(message.Content))
			}
			return nil
		}

		last := len(messages) - 1
		switch {
		case kind == projectedToolResults && messageKinds[last] != projectedToolResults:
			return &ConversationCollisionError{Reason: "tool_result blocks would follow ordinary content in one user turn"}
		case messageKinds[last] == projectedToolResults && kind == projectedOrdinary:
			messageKinds[last] = projectedToolResultsThenOrdinary
		}
		messages[last].Content = append(messages[last].Content, message.Content...)
		if !volatile {
			committedBlocks[last] += len(message.Content)
		}
		return nil
	}
	for index, conv := range req.Messages {
		volatile := index >= committedSourceMessages
		switch m := conv.(type) {
		case *content.SystemMessage:
			// Anthropic has no in-thread system role: fold system text into the
			// top-level `system` field, preserving order after Request.System.
			systemText := textOf(m.Blocks)
			if index >= committedSourceMessages && systemText != "" {
				transientSystemMessageNonEmpty = true
			}
			system = appendSystem(system, systemText)
		case *content.UserMessage:
			blocks, err := encodeBlocks(m.Blocks)
			if err != nil {
				return messagesRequest{}, err
			}
			if err := appendMessage(anthropicMessage{Role: roleUser, Content: blocks}, projectedOrdinary, volatile); err != nil {
				return messagesRequest{}, err
			}
		case *content.AIMessage:
			blocks, err := encodeBlocks(m.Blocks)
			if err != nil {
				return messagesRequest{}, err
			}
			if err := appendMessage(anthropicMessage{Role: roleAssistant, Content: blocks}, projectedOrdinary, volatile); err != nil {
				return messagesRequest{}, err
			}
		case *content.ToolResultMessage:
			block, err := encodeToolResult(m)
			if err != nil {
				return messagesRequest{}, err
			}
			// Anthropic delivers tool results as a user-role message whose sole
			// block is a tool_result. IsError is a first-class field here (unlike
			// the OpenAI dialect), so ToolResultMessage.IsError passes through.
			if err := appendMessage(anthropicMessage{Role: roleUser, Content: []anthropicBlock{block}}, projectedToolResults, volatile); err != nil {
				return messagesRequest{}, err
			}
		default:
			return messagesRequest{}, &UnsupportedConversationError{Conversation: fmt.Sprintf("%T", conv)}
		}
	}
	var sys *systemPrompt
	if system != "" {
		sys = &systemPrompt{Text: system}
	}

	r := messagesRequest{
		Model:         req.Model.Name,
		System:        sys,
		Messages:      messages,
		MaxTokens:     effectiveMaxTokens(sampling.MaxTokens),
		StopSequences: sampling.Stop,
		Stream:        stream,
	}

	for _, t := range req.Tools {
		if reason := toolNameReason(t.Name); reason != "" {
			return messagesRequest{}, &InvalidToolNameError{Name: t.Name, Reason: reason}
		}
		schema, err := schemaOrDefault(t.Schema)
		if err != nil {
			return messagesRequest{}, &InvalidToolSchemaError{Name: t.Name, Reason: err.Error()}
		}
		r.Tools = append(r.Tools, anthropicTool{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: schema,
		})
	}
	switch req.ToolChoice.Mode() {
	case inference.ToolChoiceModeRequired:
		r.ToolChoice = &toolChoice{Type: toolChoiceAny}
	case inference.ToolChoiceModeNamed:
		name, _ := req.ToolChoice.Named()
		r.ToolChoice = &toolChoice{Type: toolChoiceTool, Name: name}
	}

	// effort → thinking, gated by the model's advertised Thinking capability
	// and shaped by its declared ThinkingDialect. EffortNone (or a model that
	// can't think) emits neither member.
	thinkingEnabled := false
	if req.Model.Caps.Thinking && sampling.Effort != model.EffortNone {
		if !sampling.Effort.Valid() || (req.Model.Caps.ThinkingDialect == model.ThinkingDialectAdaptive && sampling.Effort == model.EffortMinimal) {
			return messagesRequest{}, &UnsupportedEffortError{Effort: string(sampling.Effort)}
		}
		thinking, effort, err := thinkingFor(req.Model, sampling.Effort, r.MaxTokens)
		if err != nil {
			return messagesRequest{}, err
		}
		r.Thinking = thinking
		if effort != "" {
			r.OutputConfig = &outputConfig{Effort: effort}
		}
		thinkingEnabled = true
	}
	if req.Output != nil {
		if r.OutputConfig == nil {
			r.OutputConfig = &outputConfig{}
		}
		r.OutputConfig.Format = &outputFormat{
			Type:   outputFormatJSONSchema,
			Schema: req.Output.Schema,
		}
	}

	// temperature/top_p reconciliation: Anthropic rejects temperature or top_p
	// sent alongside thinking with an HTTP 400, under BOTH dialects — the
	// adaptive-thinking models removed the sampling parameters outright, and
	// the budget form documents that thinking is incompatible with modifying
	// them. So when thinking is enabled for this request the codec OMITS both,
	// whichever variant it emitted. Otherwise they pass through only when set
	// (omitempty on the wire struct). This is the codec's job per the sampling
	// design — the dialect-validity rule lives here, not on Sampling.
	if !thinkingEnabled {
		if err := checkUnitInterval("temperature", sampling.Temperature); err != nil {
			return messagesRequest{}, err
		}
		if err := checkUnitInterval("top_p", sampling.TopP); err != nil {
			return messagesRequest{}, err
		}
		r.Temperature = sampling.Temperature
		r.TopP = sampling.TopP
	}

	// A non-empty transient SystemMessage folds into the top-level system
	// prefix, so caching either breakpoint would include transient context.
	if req.Model.Caps.PromptCaching && !transientSystemMessageNonEmpty {
		applyCacheBreakpoints(&r, committedBlocks)
	}

	return r, nil
}

// applyCacheBreakpoints marks the codec's two ephemeral cache_control
// breakpoints: one on the system prompt (which caches tools + system, since
// tools render before system in Anthropic's prefix order) and one on the last
// cacheable block of the committed message history, so multi-turn requests
// accrue incremental cache hits without marking transient runtime messages.
// A thinking block cannot carry cache_control, so the message breakpoint walks
// back to the nearest non-thinking block. At most two breakpoints are emitted,
// well under the wire limit of four.
func applyCacheBreakpoints(r *messagesRequest, committedBlocks []int) {
	if r.System != nil {
		r.System.Cache = true
	}
	for i := len(r.Messages) - 1; i >= 0; i-- {
		blocks := r.Messages[i].Content
		for j := committedBlocks[i] - 1; j >= 0; j-- {
			if blocks[j].Type == blockTypeThinking || blocks[j].Type == blockTypeRedactedThinking {
				continue
			}
			blocks[j].CacheControl = &cacheControl{Type: cacheControlEphemeral}
			return
		}
	}
}

// encodeBlocks maps a slice of content blocks to their Anthropic wire form.
func encodeBlocks(blocks []content.Block) ([]anthropicBlock, error) {
	out := make([]anthropicBlock, 0, len(blocks))
	for _, b := range blocks {
		eb, err := encodeBlock(b)
		if err != nil {
			return nil, err
		}
		out = append(out, eb)
	}
	return out, nil
}

// encodeBlock maps one content.Block to its Anthropic wire block. Text, image,
// document, thinking, and tool_use are supported; audio and refusals fail
// closed with the typed UnsupportedAudioError / UnsupportedRefusalError naming
// the format's own limitation, and any other concrete type yields a typed
// UnsupportedBlockError — fail-secure, not silent.
// An empty text block is likewise refused: Anthropic's RequestTextBlock requires
// non-empty text, so it has no valid wire form and would draw an HTTP 400.
func encodeBlock(b content.Block) (anthropicBlock, error) {
	switch b := b.(type) {
	case *content.TextBlock:
		if b.Text == "" {
			return anthropicBlock{}, &EmptyTextBlockError{}
		}
		return anthropicBlock{Type: blockTypeText, Text: b.Text}, nil
	case *content.ImageBlock:
		source, err := imageSourceOf(b)
		if err != nil {
			return anthropicBlock{}, err
		}
		return anthropicBlock{Type: blockTypeImage, Source: source}, nil
	case *content.DocumentBlock:
		return encodeDocumentBlock(b)
	case *content.AudioBlock:
		return anthropicBlock{}, &UnsupportedAudioError{MediaType: string(b.MediaType)}
	case *content.RefusalBlock:
		return anthropicBlock{}, &UnsupportedRefusalError{}
	case *content.ThinkingBlock:
		if b.ReplayableAs(providerStateFormatAnthropicRedacted) {
			data, err := opaqueRedactedToWire(b.ProviderState)
			if err != nil {
				return anthropicBlock{}, err
			}
			return anthropicBlock{Type: blockTypeRedactedThinking, Data: data}, nil
		}
		signature, err := replayableSignature(b)
		if err != nil {
			return anthropicBlock{}, err
		}
		return anthropicBlock{Type: blockTypeThinking, Thinking: b.Thinking, Signature: signature}, nil
	case *content.ToolUseBlock:
		if reason := toolUseIDReason(b.ID); reason != "" {
			return anthropicBlock{}, &InvalidToolUseIDError{ID: b.ID, Reason: reason}
		}
		if reason := toolNameReason(b.Name); reason != "" {
			return anthropicBlock{}, &InvalidToolNameError{Name: b.Name, Reason: reason}
		}
		return anthropicBlock{Type: blockTypeToolUse, ID: b.ID, Name: b.Name, Input: inputOrEmpty(b.Input)}, nil
	default:
		return anthropicBlock{}, &UnsupportedBlockError{Block: fmt.Sprintf("%T", b)}
	}
}

// providerStateFormatAnthropicRedacted is the ProviderStateFormat tag carried by
// a redacted-thinking block's opaque payload, scoping it to this dialect.
const providerStateFormatAnthropicRedacted = "anthropic-redacted-thinking" // #nosec G101 -- a dialect tag, not a credential

// signatureFormatAnthropic labels a reasoning signature as minted by the
// Anthropic Messages API. Every site in this package that reads a signature off
// the wire stamps it, and every site that writes one to the wire demands it;
// see ForeignThinkingSignatureError for why a mismatch is fatal rather than a
// dropped field.
//
// It is deliberately NOT the redacted-thinking label above. The two are
// different channels of provider-private state on the same block — a visible
// signed block and an opaque redacted one — and sharing one label would let
// either authorize the other's replay.
const signatureFormatAnthropic = "anthropic"

// signatureFormatFor returns this dialect's signature label for a signature
// just read off the wire, and "" for an absent one, so a label never ends up
// attached to nothing. It is the decode-side mirror of the encode-side check.
func signatureFormatFor(signature string) string {
	if signature == "" {
		return ""
	}
	return signatureFormatAnthropic
}

// replayableSignature returns the signature to place on the wire for b, or an
// error when b carries one this dialect cannot claim. An absent signature is
// not an error: a reasoning block with nothing to replay is a legitimate shape
// (a partial block, or one from a dialect that seals nothing).
//
// This is the single home for the check so the three egress paths — outbound
// request encode, gateway response encode, and gateway stream encode — cannot
// drift apart. They did on redacted thinking once already, and the symptom was
// that the same turn served differently depending on whether the client asked
// to stream.
func replayableSignature(b *content.ThinkingBlock) (string, error) {
	if b.Signature == "" {
		return "", nil
	}
	signature, ok := b.SignatureReplayableAs(signatureFormatAnthropic)
	if !ok {
		return "", &ForeignThinkingSignatureError{Format: b.SignatureFormat}
	}
	return signature, nil
}

func opaqueRedactedState(data string) json.RawMessage {
	encoded, _ := json.Marshal(data)
	return encoded
}

func opaqueRedactedToWire(raw json.RawMessage) (string, error) {
	var data string
	if err := json.Unmarshal(raw, &data); err != nil {
		return "", fmt.Errorf("anthropicapi: invalid redacted thinking ProviderState: %w", err)
	}
	return data, nil
}

// encodeToolResult builds the tool_result block from a ToolResultMessage. The
// result content (typically text) is encoded through the same block encoder.
func encodeToolResult(m *content.ToolResultMessage) (anthropicBlock, error) {
	if reason := toolUseIDReason(m.ToolUseID); reason != "" {
		return anthropicBlock{}, &InvalidToolUseIDError{ID: m.ToolUseID, Reason: reason}
	}
	inner, err := encodeBlocks(m.Blocks)
	if err != nil {
		return anthropicBlock{}, err
	}
	return anthropicBlock{
		Type:      blockTypeToolResult,
		ToolUseID: m.ToolUseID,
		Content:   inner,
		IsError:   m.IsError,
	}, nil
}

// encodeDocumentBlock builds a `document` block from a neutral DocumentBlock.
//
// Only two of RequestDocumentBlock's four source members are reachable from the
// neutral vocabulary. Base64PDFSource carries binary Data, PlainTextSource
// carries extracted Text, and both declare `media_type` as a const — so the
// selection is driven by which payload the caller populated, and the media type
// is then held to the const that member declares rather than forwarded. The
// remaining two members are unreachable by construction: URLPDFSource needs a
// URL and ContentBlockSource needs a nested block array, and content.
// DocumentBlock has a field for neither.
//
// Data wins over Text when both are set: binary is the higher-fidelity source,
// and the extracted text is derived from it.
//
// Name becomes `title` rather than being dropped. It is the only place the wire
// shape can hold a document's name at all, and dropping it would strip the one
// piece of provenance the model has for a document it is asked to cite.
func encodeDocumentBlock(doc *content.DocumentBlock) (anthropicBlock, error) {
	source, err := documentSourceOf(doc)
	if err != nil {
		return anthropicBlock{}, err
	}
	if count := len([]rune(doc.Name)); count > maxDocumentTitleLength {
		return anthropicBlock{}, &UnsupportedDocumentError{
			Reason: fmt.Sprintf("document name is %d characters, which exceeds the %d-character maximum of RequestDocumentBlock.title", count, maxDocumentTitleLength),
		}
	}
	return anthropicBlock{Type: blockTypeDocument, Source: source, Title: doc.Name}, nil
}

// documentSourceOf selects the document's source union member. It never
// approximates: a media type outside the two consts the request document
// declares is refused, because the alternative is telling Anthropic the payload
// is a PDF or plain text when the caller said it was something else.
func documentSourceOf(doc *content.DocumentBlock) (*blockSource, error) {
	switch {
	case len(doc.Data) > 0:
		if doc.MediaType != content.MediaTypeDocumentPDF {
			return nil, &UnsupportedDocumentError{
				Reason: "binary document media type " + string(doc.MediaType) + " has no source member; Base64PDFSource.media_type is const application/pdf",
			}
		}
		return &blockSource{
			Type:      sourceTypeBase64,
			MediaType: documentMediaTypePDF,
			Data:      base64.StdEncoding.EncodeToString(doc.Data),
		}, nil
	case doc.Text != "":
		if doc.MediaType != content.MediaTypeDocumentText {
			return nil, &UnsupportedDocumentError{
				Reason: "text document media type " + string(doc.MediaType) + " has no source member; PlainTextSource.media_type is const text/plain",
			}
		}
		return &blockSource{
			Type:      sourceTypeText,
			MediaType: documentMediaTypeText,
			Data:      doc.Text,
		}, nil
	default:
		return nil, &UnsupportedDocumentError{Reason: "document carries neither data nor text, and RequestDocumentBlock declares source required"}
	}
}

// imageSourceOf builds the `source` for an image block. A URL takes precedence
// over inline Data; Data is base64-encoded with its media type.
//
// An inline source's media type is held to Anthropic's Base64ImageSource enum
// first. The shared content.MediaType vocabulary is wider than that enum —
// content.MediaTypeImageSVG is a first-class value with no Anthropic
// equivalent — so forwarding the field verbatim turns a representable Looprig
// block into an HTTP 400. A URL source carries no media type at all, which is
// why the check applies only to the inline branch.
func imageSourceOf(img *content.ImageBlock) (*blockSource, error) {
	if img.Source.URL != "" {
		return &blockSource{Type: sourceTypeURL, URL: img.Source.URL}, nil
	}
	if !supportedImageMediaTypes[img.MediaType] {
		return nil, &UnsupportedImageMediaTypeError{MediaType: string(img.MediaType)}
	}
	return &blockSource{
		Type:      sourceTypeBase64,
		MediaType: string(img.MediaType),
		Data:      base64.StdEncoding.EncodeToString(img.Source.Data),
	}, nil
}

// supportedImageMediaTypes is Anthropic's Base64ImageSource.media_type enum.
var supportedImageMediaTypes = map[content.MediaType]bool{
	content.MediaTypeImageJPEG: true,
	content.MediaTypeImagePNG:  true,
	content.MediaTypeImageGIF:  true,
	content.MediaTypeImageWebP: true,
}

// Anthropic's identifier constraints, transcribed from the request document.
// Both patterns are ANCHORED, so an illegal character anywhere in the value
// rejects it rather than a legal substring rescuing it. The tool-use id has no
// length bound; the tool name is capped at 128.
var (
	toolUseIDPattern = regexp.MustCompile(`^[a-zA-Z0-9_-]+$`)
	toolNamePattern  = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,128}$`)
)

// toolUseIDReason reports why a tool-call identifier is not a legal Anthropic
// tool_use id, or "" when it is one.
//
// The empty case is the sharper half. `id` is omitempty on the wire struct, so
// an empty identifier does not travel as "" — the property vanishes, and
// RequestToolUseBlock declares id required with additionalProperties:false.
// That silent drop is the same failure the thinking blocks were repaired for.
//
// The pattern half is a cross-dialect concern: Converse mints ids that may
// contain "." and ":", both of which Anthropic's class excludes, so a
// conversation replayed from Bedrock can carry an id Anthropic will not accept.
func toolUseIDReason(id string) string {
	switch {
	case id == "":
		return "tool-use id must not be empty"
	case !toolUseIDPattern.MatchString(id):
		return "tool-use id must match " + toolUseIDPattern.String()
	}
	return ""
}

// toolNameReason reports why a tool name is not a legal Anthropic tool name, or
// "" when it is one. Anthropic's class excludes "." and "/", which tool servers
// — MCP ones especially — hand out freely.
func toolNameReason(name string) string {
	switch {
	case name == "":
		return "tool name must not be empty"
	case !toolNamePattern.MatchString(name):
		return "tool name must match " + toolNamePattern.String()
	}
	return ""
}

// checkUnitInterval enforces Anthropic's [0, 1] bound on a sampling knob.
// The shared vocabulary is wider — an OpenAI-shaped temperature runs to 2 —
// so cross-provider model switching routes an ordinary value into a 400 unless
// it is caught here.
func checkUnitInterval(field string, value *float64) error {
	if value == nil || (*value >= 0 && *value <= 1) {
		return nil
	}
	return &SamplingRangeError{Field: field, Value: *value}
}

// inputOrEmpty guarantees a tool_use `input` is a JSON object: an empty raw
// value becomes "{}", which Anthropic requires.
func inputOrEmpty(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return json.RawMessage(emptyObject)
	}
	return raw
}

// schemaOrDefault guarantees a tool `input_schema` is a JSON object: an empty
// schema becomes {"type":"object"}, which Anthropic requires.
//
// A non-empty schema that is not an object is refused rather than forwarded.
// InputSchema is typed `object` in Anthropic's document, so an array or scalar
// there is an HTTP 400 that names neither the tool nor the field; the sibling
// bedrockconverse encoder has always made this check, and its absence here was
// the asymmetry.
func schemaOrDefault(schema json.RawMessage) (json.RawMessage, error) {
	trimmed := bytes.TrimSpace(schema)
	if len(trimmed) == 0 {
		return json.RawMessage(defaultSchema), nil
	}
	if !json.Valid(trimmed) || trimmed[0] != '{' {
		return nil, fmt.Errorf("input_schema must be a JSON object")
	}
	return schema, nil
}

// effectiveMaxTokens returns the caller's MaxTokens when set to a positive value,
// else the mandatory codec default.
func effectiveMaxTokens(p *int) int {
	if p != nil && *p > 0 {
		return *p
	}
	return defaultMaxTokens
}

// effortValue maps the dialect-neutral model.Effort to Anthropic's
// output_config.effort wire value. EffortNone (and any unknown value, fail-safe)
// yields "", which suppresses both the thinking and output_config fields.
func effortValue(e model.Effort) string {
	switch e {
	case model.EffortLow:
		return "low"
	case model.EffortMedium:
		return "medium"
	case model.EffortHigh:
		return "high"
	case model.EffortXHigh:
		return "xhigh"
	case model.EffortMax:
		return "max"
	default: // EffortNone or unknown → omit
		return ""
	}
}

// thinkingFor selects the thinking request shape for a model that has been
// asked to reason, returning the `thinking` member and the
// `output_config.effort` value ("" when the dialect must not carry one).
//
// The choice is the model's declared ThinkingDialect and nothing else. It is
// NOT derived from the model name: both spellings are served by the same
// endpoint for different generations, so a name match in the codec would be a
// catalogue baked into a wire encoder, wrong the day a model ships. It is not
// defaulted either — see UndeclaredThinkingDialectError.
//
// output_config.effort is withheld from the budget dialect deliberately. The
// two are not merely redundant there: the models that take budget_tokens reject
// `effort`, and the schema cannot see it — the gate was measured ACCEPTING
// `{"type":"enabled","budget_tokens":2048}` alongside
// `output_config.effort:"medium"`. This function is where that per-model
// constraint is carried, since the gate is blind to it.
func thinkingFor(m model.Model, effort model.Effort, maxTokens int) (*thinkingConfig, string, error) {
	switch m.Caps.ThinkingDialect {
	case model.ThinkingDialectAdaptive:
		return &thinkingConfig{Type: thinkingTypeAdaptive}, effortValue(effort), nil
	case model.ThinkingDialectBudget:
		budget, ok := thinkingBudgetTokens(effort, maxTokens)
		if !ok {
			return nil, "", &ThinkingBudgetError{Model: m.Name, MaxTokens: maxTokens}
		}
		return &thinkingConfig{Type: thinkingTypeEnabled, BudgetTokens: budget}, "", nil
	default:
		return nil, "", &UndeclaredThinkingDialectError{Model: m.Name}
	}
}

// thinkingBudgetTokens maps the dialect-neutral Effort to a budget_tokens value
// for the enabled variant, reporting ok=false when no legal value exists.
//
// The budget is a FRACTION of the request's own max_tokens rather than a fixed
// per-level constant, and that is the one design choice here worth stating. The
// sibling Gemini encoder uses fixed constants precisely because no Google
// document relates thinkingBudget to maxOutputTokens, so clamping there would
// invent a bound. Anthropic publishes the relationship — budget_tokens must be
// less than max_tokens — so the reverse holds: a fixed constant would be the
// invention, and would emit a 32k budget under a 4k cap. Scaling keeps every
// level legal for every cap above the floor, and keeps the levels ordered.
//
// The floor is the schema's own minimum, not a preference. Below it there is no
// legal value at all, which is what ok=false reports.
func thinkingBudgetTokens(e model.Effort, maxTokens int) (int, bool) {
	// Fractions of max_tokens per level. The top level stops short of the cap
	// because budget_tokens must be STRICTLY below it, and because a request
	// that spends its entire allowance on reasoning returns no answer.
	var numerator int
	switch e {
	case model.EffortMinimal:
		numerator = 10
	case model.EffortLow:
		numerator = 25
	case model.EffortMedium:
		numerator = 50
	case model.EffortHigh:
		numerator = 75
	case model.EffortXHigh:
		numerator = 85
	case model.EffortMax:
		numerator = 90
	default: // EffortNone or unknown; callers gate on effortValue first
		return 0, false
	}
	budget := maxTokens * numerator / 100
	if budget < minThinkingBudgetTokens {
		budget = minThinkingBudgetTokens
	}
	if budget >= maxTokens {
		// Only reachable for maxTokens <= minThinkingBudgetTokens, where the
		// declared minimum and the strictly-below-max_tokens rule cannot both
		// hold.
		return 0, false
	}
	return budget, true
}

// appendSystem joins two system-prompt fragments, inserting a blank-line
// separator only when both sides are non-empty.
func appendSystem(base, add string) string {
	switch {
	case add == "":
		return base
	case base == "":
		return add
	default:
		return base + "\n\n" + add
	}
}

// textOf concatenates the text of all TextBlocks in a slice, ignoring others.
func textOf(blocks []content.Block) string {
	var sb strings.Builder
	for _, b := range blocks {
		if t, ok := b.(*content.TextBlock); ok {
			sb.WriteString(t.Text)
		}
	}
	return sb.String()
}
