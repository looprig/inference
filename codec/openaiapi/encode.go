package openaiapi

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"regexp"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	model "github.com/looprig/inference/model"
)

// toolNamePattern is FunctionObject.name's published class, transcribed from
// the request document: "Must be a-z, A-Z, 0-9, or contain underscores and
// dashes, with a maximum length of 64." It is ANCHORED and carries the length
// cap itself, so an illegal character anywhere rejects the name rather than a
// legal substring rescuing it.
//
// The class is narrower than the names tool servers publish — MCP names
// routinely carry "." or "/" — and narrower than Anthropic's, which allows 128
// characters. A tool set that works against one provider is therefore not
// automatically encodable for this one.
var toolNamePattern = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

// maxToolNameLength is the same cap spelled out, so the diagnostic can report
// the overshoot in characters instead of only quoting a regular expression.
const maxToolNameLength = 64

// toolNameReason reports why a name is not a legal OpenAI function name, or ""
// when it is one. It returns a bare reason so each call site can attach the
// error its surrounding context uses, matching the sibling anthropicapi and
// bedrockconverse codecs.
func toolNameReason(name string) string {
	switch {
	case name == "":
		return "tool name must not be empty"
	case len(name) > maxToolNameLength:
		return fmt.Sprintf("tool name is %d characters, which exceeds OpenAI's %d-character limit", len(name), maxToolNameLength)
	case !toolNamePattern.MatchString(name):
		return "tool name must match " + toolNamePattern.String()
	}
	return ""
}

// The sampling intervals CreateChatCompletionRequest declares, reached through
// CreateModelResponseProperties -> ModelResponseProperties. Unlike the tool-name
// class these ARE in the derived schema (minimum/maximum), so the conformance
// gate rejects a violation too; checking here means the failure names the field
// at the moment the request is built rather than after a body has been marshalled.
const (
	minTemperature = 0.0
	maxTemperature = 2.0
	minTopP        = 0.0
	maxTopP        = 1.0
)

// checkSamplingRange holds one sampling knob to its declared interval. An unset
// value is legal — the field is omitted from the body entirely.
func checkSamplingRange(field string, value *float64, min, max float64) error {
	if value == nil || (*value >= min && *value <= max) {
		return nil
	}
	return &SamplingRangeError{Field: field, Value: *value, Min: min, Max: max}
}

// BuildChatRequest converts a provider-neutral inference.Request into a ChatRequest
// struct. Exported so provider packages can embed or extend the result before
// marshaling (e.g. a provider extension adds an encrypted-response public-key field).
func BuildChatRequest(req inference.Request, stream bool) (ChatRequest, error) {
	if err := inference.ValidateRequestFeatures(req); err != nil {
		return ChatRequest{}, err
	}

	// Effective sampling: a non-nil per-call Override wins over Model.Sampling.
	sampling := req.Model.Sampling
	if req.Override != nil {
		sampling = *req.Override
	}
	if err := checkSamplingRange("temperature", sampling.Temperature, minTemperature, maxTemperature); err != nil {
		return ChatRequest{}, err
	}
	if err := checkSamplingRange("top_p", sampling.TopP, minTopP, maxTopP); err != nil {
		return ChatRequest{}, err
	}

	cr := ChatRequest{
		Model: req.Model.Name,
		// Non-nil from the start: `messages` is spec-typed as an array with no
		// null alternative, so an empty conversation must still marshal as [].
		Messages:        []chatMessage{},
		Temperature:     sampling.Temperature,
		TopP:            sampling.TopP,
		Stop:            sampling.Stop,
		Stream:          stream,
		ReasoningEffort: reasoningEffort(sampling.Effort),
	}
	// Token limit: max_tokens is deprecated in OpenAI's schema and rejected
	// outright by reasoning models, which require max_completion_tokens. The
	// choice is gated on the model's advertised Thinking capability rather
	// than switched wholesale: this dialect is also spoken by many
	// OpenAI-compatible servers (older llama.cpp / Ollama builds among them)
	// that understand only max_tokens, so a non-reasoning model keeps the
	// legacy spelling. Exactly one field is ever populated.
	if req.Model.Caps.Thinking {
		cr.MaxCompletionTokens = sampling.MaxTokens
	} else {
		cr.MaxTokens = sampling.MaxTokens
	}
	if stream {
		cr.StreamOptions = &chatStreamOptions{IncludeUsage: true}
	}

	if req.System != "" {
		cr.Messages = append(cr.Messages, chatMessage{
			Role:    "system",
			Content: req.System,
		})
	}

	for _, conv := range req.Messages {
		msgs, err := encodeConversation(conv)
		if err != nil {
			return ChatRequest{}, err
		}
		cr.Messages = append(cr.Messages, msgs...)
	}

	for _, t := range req.Tools {
		if reason := toolNameReason(t.Name); reason != "" {
			return ChatRequest{}, &InvalidToolNameError{Name: t.Name, Reason: reason}
		}
		cr.Tools = append(cr.Tools, chatTool{
			Type: "function",
			Function: chatFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  schemaOrDefault(t.Schema),
			},
		})
	}
	if req.Output != nil {
		cr.ResponseFormat = &responseFormat{
			Type: responseFormatJSONSchema,
			JSONSchema: &jsonSchema{
				Name:   req.Output.Name,
				Strict: req.Output.Strict,
				Schema: req.Output.Schema,
			},
		}
	}
	switch req.ToolChoice.Mode() {
	case inference.ToolChoiceModeRequired:
		cr.ToolChoice = json.RawMessage(`"` + toolChoiceRequired + `"`)
	case inference.ToolChoiceModeNamed:
		// ChatCompletionNamedToolChoice.function.name is not re-checked here:
		// ValidateRequestFeatures (above) already refuses a forced choice that
		// names an undeclared tool, and every declared tool has just been held
		// to toolNameReason, so an illegal forced name is unreachable. The
		// pairing is pinned by TestEncodeRequestNamedToolChoiceInheritsTheRule.
		name, _ := req.ToolChoice.Named()
		choice, err := json.Marshal(namedToolChoice{
			Type:     toolChoiceTypeFunction,
			Function: namedToolChoiceFunc{Name: name},
		})
		if err != nil {
			return ChatRequest{}, err
		}
		cr.ToolChoice = choice
	}

	return cr, nil
}

// EncodeRequest converts a provider-neutral inference.Request to an OpenAI chat
// completions JSON body. stream=true adds "stream":true to the body.
// Request.System is prepended as a system message if non-empty.
func EncodeRequest(req inference.Request, stream bool) ([]byte, error) {
	cr, err := BuildChatRequest(req, stream)
	if err != nil {
		return nil, err
	}
	return json.Marshal(cr)
}

// reasoningEffort maps the dialect-neutral model.Effort to the OpenAI Chat
// Completions reasoning_effort wire value. The current request schema admits
// the complete neutral ladder, so every known non-none value is preserved
// exactly. EffortNone (and any unknown value, fail-safe) yields "", which the
// omitempty tag drops from the body.
// schemaOrDefault substitutes an empty object schema for a tool that declares
// no parameters, because FunctionObject.parameters is spec-typed `object`:
// emitting the nil json.RawMessage verbatim would put a bare `null` on the
// wire. Mirrors openairesponses' function of the same name.
func schemaOrDefault(schema json.RawMessage) json.RawMessage {
	if len(schema) == 0 {
		return json.RawMessage(defaultSchema)
	}
	return schema
}

func reasoningEffort(e model.Effort) string {
	switch e {
	case model.EffortMinimal:
		return "minimal"
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

// encodeConversation dispatches a content.Conversation to the appropriate
// chatMessage encoder.
func encodeConversation(conv content.Conversation) ([]chatMessage, error) {
	switch m := conv.(type) {
	case *content.SystemMessage:
		return []chatMessage{{
			Role:    "system",
			Content: textContent(m.Blocks),
		}}, nil

	case *content.UserMessage:
		parts, err := encodeContentParts(m.Blocks)
		if err != nil {
			return nil, err
		}
		return []chatMessage{{
			Role:    "user",
			Content: parts,
		}}, nil

	case *content.AIMessage:
		msg, err := encodeAIMessage(m)
		if err != nil {
			return nil, err
		}
		return []chatMessage{msg}, nil

	case *content.ToolResultMessage:
		// IsError reconciliation: the OpenAI Chat Completions tool message has no
		// structured error flag (unlike Anthropic's tool_result block), so
		// ToolResultMessage.IsError is intentionally NOT placed on the request —
		// emitting a non-standard is_error here would be a schema violation. The model
		// learns a tool errored via the result's text content, which for engine-level
		// failures (Go error, panic, empty result, pre-exec failure) the loop
		// error-prefixes; a tool's self-reported ToolResultBlock error passes through
		// verbatim, so there the message-level IsError is the only structured signal.
		// IsError exists for the internal wire form and the display layer, not for
		// this provider's request.
		text, err := toolResultText(m.Blocks)
		if err != nil {
			return nil, err
		}
		return []chatMessage{{
			Role:       "tool",
			Content:    text,
			ToolCallID: m.ToolUseID,
		}}, nil

	default:
		return nil, fmt.Errorf("openaiapi: unknown conversation type %T", conv)
	}
}

// textContent concatenates all text blocks into a single string.
func textContent(blocks []content.Block) string {
	var out string
	for _, b := range blocks {
		if t, ok := b.(*content.TextBlock); ok {
			out += t.Text
		}
	}
	return out
}

// toolResultText flattens a tool result's blocks to the plain string the
// text-only OpenAI tool message carries. Any non-text block yields a typed
// *UnsupportedBlockError — fail-secure, never a silent drop.
func toolResultText(blocks []content.Block) (string, error) {
	var out string
	for _, b := range blocks {
		t, ok := b.(*content.TextBlock)
		if !ok {
			return "", &UnsupportedBlockError{Block: fmt.Sprintf("%T", b)}
		}
		out += t.Text
	}
	return out, nil
}

// encodeContentParts returns a plain string when all blocks are text, or a
// []chatContentPart slice when any richer block is present. All four members
// of ChatCompletionRequestUserMessageContentPart are covered: text, image_url,
// file (DocumentBlock) and input_audio (AudioBlock). A block type the dialect
// does not model in a user turn, or one whose value the part's schema cannot
// carry, yields a typed *UnsupportedBlockError — fail-secure, never a silent
// drop.
func encodeContentParts(blocks []content.Block) (interface{}, error) {
	allText := true
	for _, b := range blocks {
		if _, ok := b.(*content.TextBlock); !ok {
			allText = false
			break
		}
	}
	if allText {
		return textContent(blocks), nil
	}

	parts := make([]chatContentPart, 0, len(blocks))
	for _, b := range blocks {
		switch b := b.(type) {
		case *content.TextBlock:
			parts = append(parts, chatContentPart{Type: "text", Text: b.Text})
		case *content.ImageBlock:
			parts = append(parts, chatContentPart{Type: "image_url", ImageURL: &imageURLPart{URL: imageURL(b)}})
		case *content.DocumentBlock:
			file, err := documentFilePart(b)
			if err != nil {
				return nil, err
			}
			parts = append(parts, chatContentPart{Type: contentPartTypeFile, File: file})
		case *content.AudioBlock:
			audio, err := audioInputPart(b)
			if err != nil {
				return nil, err
			}
			parts = append(parts, chatContentPart{Type: contentPartTypeInputAudio, InputAudio: audio})
		default:
			return nil, &UnsupportedBlockError{Block: fmt.Sprintf("%T", b)}
		}
	}
	return parts, nil
}

// imageURL builds the URL string for an ImageBlock. URL takes precedence over
// Data. If Data is set, a data URI is returned.
func imageURL(img *content.ImageBlock) string {
	if img.Source.URL != "" {
		return img.Source.URL
	}
	encoded := base64.StdEncoding.EncodeToString(img.Source.Data)
	return "data:" + string(img.MediaType) + ";base64," + encoded
}

// documentFilePart builds a file content part's `file` object from a
// DocumentBlock.
//
// The neutral block can only carry a document inline, so only the
// filename+file_data form is reachable; `file_id` names a file already
// uploaded to OpenAI's Files API, and no neutral field holds such a handle.
//
// file_data is spec-typed as a bare string ("The base64 encoded file data"),
// which loses DocumentBlock.MediaType. Emitting a data: URI instead keeps that
// caller intent on the wire, is the form OpenAI's own file-input guide uses,
// and matches how this same codec already encodes inline image bytes
// (imageURL). A block with no media type falls back to bare base64 rather than
// emitting an empty `data:;base64,` scheme — which is also what decodeFilePart
// produces for a bare-base64 wire value, so the two forms round-trip.
//
// The filename is required rather than optional: OpenAI's file input guide
// documents it as mandatory alongside file_data (the schema leaves it optional
// only because the file_id form does not need it), so an unnamed document is
// refused here where the error can name the field, not at the provider.
func documentFilePart(doc *content.DocumentBlock) (*filePart, error) {
	payload := documentPayload(doc)
	if len(payload) == 0 {
		return nil, &UnsupportedBlockError{Block: "*content.DocumentBlock", Reason: "document carries neither data nor text"}
	}
	if doc.Name == "" {
		return nil, &UnsupportedBlockError{Block: "*content.DocumentBlock", Reason: "document name is empty and file_data requires a filename"}
	}
	encoded := base64.StdEncoding.EncodeToString(payload)
	if doc.MediaType != "" {
		encoded = "data:" + string(doc.MediaType) + ";base64," + encoded
	}
	return &filePart{Filename: doc.Name, FileData: encoded}, nil
}

// documentPayload returns the bytes a DocumentBlock puts on the wire. Data and
// Text are alternative representations of the same document (an MCP embedded
// resource can populate both); binary data wins, because it is the original
// and the extracted text is derived from it.
func documentPayload(doc *content.DocumentBlock) []byte {
	if len(doc.Data) > 0 {
		return doc.Data
	}
	return []byte(doc.Text)
}

// audioInputPart builds an audio content part's `input_audio` object from an
// AudioBlock.
func audioInputPart(audio *content.AudioBlock) (*inputAudioPart, error) {
	format, ok := audioFormat(audio.MediaType)
	if !ok {
		return nil, &UnsupportedBlockError{
			Block:  "*content.AudioBlock",
			Reason: "input_audio.format accepts only wav and mp3, not " + string(audio.MediaType),
		}
	}
	if len(audio.Data) == 0 {
		return nil, &UnsupportedBlockError{Block: "*content.AudioBlock", Reason: "audio carries no data"}
	}
	return &inputAudioPart{Data: base64.StdEncoding.EncodeToString(audio.Data), Format: format}, nil
}

// audioFormat maps a neutral audio media type to `input_audio.format`. The
// spec's enum is exactly ["wav","mp3"], transcribed here as an allowlist so a
// format OpenAI adds later fails closed instead of leaking an unaccepted value
// onto the wire — and so the audio types content declares constants for but
// OpenAI does not accept (ogg, flac, aac, mp4, webm) are refused locally.
//
// The non-IANA spellings are listed alongside the registered ones because an
// AudioBlock's media type is frequently whatever an MCP server declared
// (mcp/pkg/harness passes it through verbatim), and those spellings name the
// same two containers.
func audioFormat(mediaType content.MediaType) (string, bool) {
	switch mediaType {
	case content.MediaTypeAudioWAV, "audio/x-wav", "audio/wave", "audio/vnd.wave":
		return audioFormatWAV, true
	case content.MediaTypeAudioMPEG, "audio/mp3", "audio/x-mp3":
		return audioFormatMP3, true
	default:
		return "", false
	}
}

// encodeAIMessage builds a chatMessage from an AIMessage, handling text,
// refusals and tool calls, and ignoring ThinkingBlock.
//
// A RefusalBlock replays into the message-level `refusal` member
// ChatCompletionRequestAssistantMessage declares — never into `content`. The
// distinction is the whole reason the block type exists: text is what the model
// produced, a refusal is the model saying it would not produce it, and quoting
// a refusal back as assistant prose shows the model its own decline as
// something it said. Several refusal blocks in one turn concatenate, matching
// how several text blocks do, because the member is a single string.
func encodeAIMessage(m *content.AIMessage) (chatMessage, error) {
	var text string
	var refusal *string
	var calls []toolCall

	for _, b := range m.Blocks {
		switch b := b.(type) {
		case *content.TextBlock:
			text += b.Text
		case *content.RefusalBlock:
			joined := b.Text
			if refusal != nil {
				joined = *refusal + b.Text
			}
			refusal = &joined
		case *content.ToolUseBlock:
			// A replayed call names the function it invoked, so it is bound by
			// the same class as the declaration — and it is the likelier
			// carrier of a violation, because the id and name were minted by
			// whichever dialect issued the call. Anthropic permits 128
			// characters and Gemini permits "." and ":", so a thread replayed
			// into this dialect can hold a name it will not accept.
			if reason := toolNameReason(b.Name); reason != "" {
				return chatMessage{}, &InvalidToolNameError{Name: b.Name, Reason: reason}
			}
			// OpenAI wire format: function.arguments MUST be a JSON-encoded
			// STRING (e.g. "{\"p\":1}"), never a raw object. b.Input holds the
			// raw JSON object, so quote it; empty input becomes "{}". Emitting a
			// bare object here makes strict OpenAI-compatible servers reject the
			// follow-up request with a 400.
			raw := string(b.Input)
			if raw == "" {
				raw = "{}"
			}
			quoted, err := json.Marshal(raw)
			if err != nil {
				return chatMessage{}, fmt.Errorf("openaiapi: encode tool arguments for %q: %w", b.Name, err)
			}
			calls = append(calls, toolCall{
				ID:       b.ID,
				Type:     "function",
				Function: toolCallFunction{Name: b.Name, Arguments: json.RawMessage(quoted)},
			})
		case *content.ThinkingBlock:
			// Deliberately ignored: thinking is not part of the OpenAI wire format.
		default:
			return chatMessage{}, &UnsupportedBlockError{Block: fmt.Sprintf("%T", b)}
		}
	}

	return chatMessage{
		Role:      "assistant",
		Content:   text,
		Refusal:   refusal,
		ToolCalls: calls,
	}, nil
}
