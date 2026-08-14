package bedrockconverse

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	model "github.com/looprig/inference/model"
)

func buildRequest(req inference.Request) (converseRequest, error) {
	if err := inference.ValidateRequestFeatures(req); err != nil {
		return converseRequest{}, err
	}

	if err := validateTools(req.Tools); err != nil {
		return converseRequest{}, err
	}

	r := converseRequest{}
	if req.System != "" {
		r.System = append(r.System, systemContentBlock{Text: req.System})
	}

	messages, err := encodeMessages(req.Messages, req.TransientMessages)
	if err != nil {
		return converseRequest{}, err
	}
	r.Messages = messages

	for _, conversation := range req.Messages {
		switch message := conversation.(type) {
		case *content.SystemMessage:
			blocks, err := encodeSystemBlocks(message.Blocks)
			if err != nil {
				return converseRequest{}, err
			}
			r.System = append(r.System, blocks...)
		case *content.UserMessage, *content.AIMessage, *content.ToolResultMessage:
			// Conversation turns were projected above; this pass only gathers
			// system messages, which Converse carries out of band.
		default:
			return converseRequest{}, &UnsupportedConversationError{Conversation: fmt.Sprintf("%T", conversation)}
		}
	}

	sampling := req.Model.Sampling
	if req.Override != nil {
		sampling = *req.Override
	}
	config, samplingErr := samplingConfig(sampling)
	if samplingErr != nil {
		return converseRequest{}, samplingErr
	}
	if config != nil {
		r.InferenceConfig = config
	}

	if len(req.Tools) > 0 {
		tools := make([]toolDefinition, 0, len(req.Tools))
		for _, tool := range req.Tools {
			tools = append(tools, toolDefinition{ToolSpec: toolSpec{
				Name:        tool.Name,
				Description: tool.Description,
				InputSchema: toolInputSchema{JSON: append(json.RawMessage(nil), tool.Schema...)},
			}})
		}
		r.ToolConfig = &toolConfig{Tools: tools}
		switch req.ToolChoice.Mode() {
		case inference.ToolChoiceModeRequired:
			r.ToolConfig.ToolChoice = &toolChoice{Any: &struct{}{}}
		case inference.ToolChoiceModeNamed:
			name, _ := req.ToolChoice.Named()
			r.ToolConfig.ToolChoice = &toolChoice{Tool: &specificToolChoice{Name: name}}
		}
	}

	if req.Output != nil {
		r.OutputConfig = &outputConfig{TextFormat: &textFormat{
			Type: "json_schema",
			Structure: &textStructure{JSONSchema: jsonSchema{
				Schema:      string(req.Output.Schema),
				Name:        req.Output.Name,
				Description: req.Output.Description,
			}},
		}}
	}

	return r, nil
}

type projectedMessageKind uint8

const (
	projectedOrdinary projectedMessageKind = iota
	projectedToolResults
)

func encodeMessages(messages content.AgenticMessages, transient int) ([]converseMessage, error) {
	committed := len(messages) - transient
	encoded := make([]converseMessage, 0, len(messages))
	kinds := make([]projectedMessageKind, 0, len(messages))

	appendMessage := func(message converseMessage, kind projectedMessageKind) error {
		if len(encoded) == 0 || encoded[len(encoded)-1].Role != message.Role {
			encoded = append(encoded, message)
			kinds = append(kinds, kind)
			return nil
		}
		last := len(encoded) - 1
		if kinds[last] != kind {
			return &ConversationCollisionError{Reason: "adjacent ordinary user and tool-result turns"}
		}
		encoded[last].Content = append(encoded[last].Content, message.Content...)
		return nil
	}

	for i, conversation := range messages {
		switch message := conversation.(type) {
		case *content.SystemMessage:
			continue
		case *content.UserMessage:
			blocks, err := encodeContentBlocks(message.Blocks, roleUser)
			if err != nil {
				return nil, err
			}
			if len(encoded) > 0 && encoded[len(encoded)-1].Role == roleUser && kinds[len(kinds)-1] == projectedToolResults {
				if i < committed {
					// A committed user turn after tool results is not a malformed
					// transcript: it is what Looprig's input fold produces when the
					// user types while a tool call is in flight. Separate the two
					// turns instead of refusing them — see insertedSeparatorText.
					separator := insertedSeparatorText
					encoded = append(encoded, converseMessage{Role: roleAssistant, Content: []converseContentBlock{{Text: &separator}}})
					kinds = append(kinds, projectedOrdinary)
				} else {
					if err := appendTransientTextToFinalToolResult(encoded[len(encoded)-1].Content, blocks); err != nil {
						return nil, err
					}
					continue
				}
			}
			if err := appendMessage(converseMessage{Role: roleUser, Content: blocks}, projectedOrdinary); err != nil {
				return nil, err
			}
		case *content.AIMessage:
			blocks, err := encodeContentBlocks(message.Blocks, roleAssistant)
			if err != nil {
				return nil, err
			}
			if err := appendMessage(converseMessage{Role: roleAssistant, Content: blocks}, projectedOrdinary); err != nil {
				return nil, err
			}
		case *content.ToolResultMessage:
			block, err := encodeToolResultMessage(message)
			if err != nil {
				return nil, err
			}
			if err := appendMessage(converseMessage{Role: roleUser, Content: []converseContentBlock{{ToolResult: block}}}, projectedToolResults); err != nil {
				return nil, err
			}
		default:
			return nil, &UnsupportedConversationError{Conversation: fmt.Sprintf("%T", conversation)}
		}
	}
	return encoded, nil
}

// insertedSeparatorText is the whole content of the assistant turn the projector
// inserts between a tool-result turn and a user turn that Looprig committed behind
// it. It is deliberately self-identifying: this string is the only thing in the
// request body that tells a reader — a human debugging a transcript, or the model
// itself — that the turn is Looprig's and not the model's.
//
// Why an inserted turn at all. Looprig folds queued user input at a tool-continuation
// boundary, so the committed transcript legitimately contains a user turn directly
// after tool results. Bedrock Converse cannot carry that: a Converse message is one
// role, ordinary conversation blocks and toolResult blocks do not mix in a single
// message, and consecutive same-role turns are rejected. Note that neither rule is in
// the Smithy model — ConverseRequest declares no alternation constraint at all — so
// the encoder, not the schema, is what holds the shape. Refusing to encode is the one
// option that is not available: the message is ALREADY in stored history, the refusal
// fires while measuring the candidate request before any HTTP call, and every later
// turn re-projects the same history, so the session wedges permanently and no retry
// can clear it.
//
// Why not fold the user's words into the final tool result, the way the transient
// runtime suffix is folded (appendTransientTextToFinalToolResult). That fold is
// justified for the suffix because the suffix is provider-generated supplemental
// content that never enters stored history; neither is true of a committed user
// message. Folding it would (1) reattribute the user's own words to a tool, which is
// the single worst misrepresentation available here — the model would answer them as
// data returned by a tool rather than as an instruction from the person it is working
// for, and prompt-injection reasoning about tool output would apply to the user's own
// request; (2) mutate bytes in the middle of the committed prefix on every subsequent
// request, invalidating a cache breakpoint that an appended turn leaves intact; and
// (3) still fail for a committed message carrying an image or a document, because the
// fold is text-only — so it would not even close the defect.
//
// Why not role "system". ConversationRole does declare a third member, SYSTEM, so a
// system-role separator is schema-legal. It is rejected on two counts. Semantically it
// would elevate whatever follows into instruction position — user text arriving as
// system authority is a privilege escalation, and this codec's neighbours route system
// content out of band through the top-level system field precisely to keep that
// boundary. Practically, Converse's own documentation drives system content through
// that field, and nothing in the model says an in-conversation system turn is accepted
// by any given model family; trading a local, nameable encode error for an unverifiable
// remote rejection is the opposite of the fail-closed rule this package follows.
//
// The cost that remains is real and is the reason this text exists: the transcript the
// model sees contains an assistant turn the model never produced. Saying so in the turn
// is what keeps the insertion honest rather than silent.
const insertedSeparatorText = "[looprig] Separator inserted by the Looprig runtime — not model output, and not part of the conversation. The user sent the message that follows while the tool call above was still running."

func appendTransientTextToFinalToolResult(target, supplemental []converseContentBlock) error {
	if len(target) == 0 || target[len(target)-1].ToolResult == nil {
		return &ConversationCollisionError{Reason: "tool-result projection has no final toolResult"}
	}
	result := target[len(target)-1].ToolResult
	for _, block := range supplemental {
		if block.Text == nil {
			return &ConversationCollisionError{Reason: "transient user content after tool results must be text only"}
		}
		text := *block.Text
		result.Content = append(result.Content, toolResultBlock{Text: &text})
	}
	return nil
}

func marshalRequest(req converseRequest) ([]byte, error) {
	encoded, err := json.Marshal(req)
	if err != nil {
		return nil, &EncodeError{Reason: "marshal request", Err: err}
	}
	return encoded, nil
}

func encodeSystemBlocks(blocks []content.Block) ([]systemContentBlock, error) {
	encoded := make([]systemContentBlock, 0, len(blocks))
	for _, block := range blocks {
		text, ok := block.(*content.TextBlock)
		if !ok || text == nil {
			return nil, unsupportedBlock(block, "system content supports text only")
		}
		// SystemContentBlock.text is Converse's NonEmptyString, and the field is
		// omitempty on the wire shape, so an empty system text does not encode
		// to {"text":""} — it encodes to {}, a SystemContentBlock that matches
		// no member of the union at all. Refusing it here is the only way the
		// caller learns which block was empty.
		if text.Text == "" {
			return nil, unsupportedBlock(block, "system text must not be empty")
		}
		encoded = append(encoded, systemContentBlock{Text: text.Text})
	}
	return encoded, nil
}

func encodeContentBlocks(blocks []content.Block, role string) ([]converseContentBlock, error) {
	encoded := make([]converseContentBlock, 0, len(blocks))
	hasDocument := false
	hasText := false
	for _, block := range blocks {
		if err := validateContentBlockRole(block, role); err != nil {
			return nil, err
		}
		wireBlock, err := encodeContentBlock(block)
		if err != nil {
			return nil, err
		}
		if wireBlock.Text != nil {
			hasText = true
		}
		if wireBlock.Document != nil {
			hasDocument = true
		}
		encoded = append(encoded, wireBlock)
	}
	if hasDocument && !hasText {
		return nil, &UnsupportedBlockError{Block: "*content.DocumentBlock", Reason: "a document requires a text block in the same message"}
	}
	return encoded, nil
}

func validateContentBlockRole(block content.Block, role string) error {
	switch block.(type) {
	case *content.ImageBlock, *content.DocumentBlock:
		if role != roleUser {
			return unsupportedBlock(block, "image and document blocks are only valid in user messages")
		}
	case *content.ThinkingBlock, *content.ToolUseBlock:
		if role != roleAssistant {
			return unsupportedBlock(block, "reasoning and tool-use blocks are only valid in assistant messages")
		}
	case *content.ToolResultBlock:
		if role != roleUser {
			return unsupportedBlock(block, "tool-result blocks are only valid in user messages")
		}
	}
	return nil
}

func encodeContentBlock(block content.Block) (converseContentBlock, error) {
	switch block := block.(type) {
	case *content.TextBlock:
		if block == nil {
			return converseContentBlock{}, unsupportedBlock(block, "nil block")
		}
		text := block.Text
		return converseContentBlock{Text: &text}, nil
	case *content.ImageBlock:
		image, err := encodeImage(block)
		if err != nil {
			return converseContentBlock{}, err
		}
		return converseContentBlock{Image: image}, nil
	case *content.DocumentBlock:
		document, err := encodeDocument(block)
		if err != nil {
			return converseContentBlock{}, err
		}
		return converseContentBlock{Document: document}, nil
	case *content.AudioBlock:
		audio, err := encodeAudio(block)
		if err != nil {
			return converseContentBlock{}, err
		}
		return converseContentBlock{Audio: audio}, nil
	case *content.ThinkingBlock:
		if block == nil {
			return converseContentBlock{}, unsupportedBlock(block, "nil block")
		}
		if block.ReplayableAs(providerStateFormatBedrockRedacted) {
			redacted, err := redactedStateToBytes(block.ProviderState)
			if err != nil {
				return converseContentBlock{}, err
			}
			return converseContentBlock{ReasoningContent: &reasoningContent{RedactedContent: redacted}}, nil
		}
		signature, err := replayableSignature(block)
		if err != nil {
			return converseContentBlock{}, err
		}
		text := block.Thinking
		return converseContentBlock{ReasoningContent: &reasoningContent{ReasoningText: &reasoningText{
			Text:      &text,
			Signature: signature,
		}}}, nil
	case *content.ToolUseBlock:
		toolUse, err := encodeToolUse(block)
		if err != nil {
			return converseContentBlock{}, err
		}
		return converseContentBlock{ToolUse: toolUse}, nil
	case *content.ToolResultBlock:
		result, err := encodeToolResultBlock(block)
		if err != nil {
			return converseContentBlock{}, err
		}
		return converseContentBlock{ToolResult: result}, nil
	case *content.RefusalBlock:
		return converseContentBlock{}, unsupportedRefusal(block)
	default:
		return converseContentBlock{}, unsupportedBlock(block, "no Converse content-block representation")
	}
}

// unsupportedRefusal is the single home for this dialect's refusal refusal, so
// every position names the same limitation.
//
// Converse reports a decline through the response's stopReason
// ("guardrail_intervened", "content_filtered"). Its ContentBlock union is
// text|image|document|video|audio|toolUse|toolResult|guardContent|
// reasoningContent|cachePoint, with no refusal member, and no request field
// carries one either — so there is nothing to route a replayed refusal to.
// Sending it as `text` would show the model its own decline quoted back as
// something it said, which is the exact defect content.RefusalBlock exists to
// remove; dropping it loses the fact that the turn was declined.
func unsupportedRefusal(block *content.RefusalBlock) error {
	return unsupportedBlock(block, "Converse has no refusal content block; a refusal is response-only metadata (stopReason), so it has no request wire form")
}

const providerStateFormatBedrockRedacted = "bedrock-converse-redacted-thinking"

// signatureFormatBedrockConverse labels a reasoning signature as minted by
// Bedrock Converse. Every site in this package that reads a signature off the
// wire stamps it, and every site that writes one demands it.
//
// Converse fronts the same Claude models as the Anthropic Messages API, with a
// structurally identical reasoning block, and the two endpoints' signatures are
// not interchangeable. This label is the only thing that distinguishes them,
// which is why it is separate from the redacted-reasoning label above: those
// are two different channels of provider-private state on one block, and
// sharing a label would let either authorize the other's replay.
const signatureFormatBedrockConverse = "bedrock-converse"

// signatureFormatFor returns this dialect's signature label for a signature
// just read off the wire, and "" for an absent one, so a label never ends up
// attached to nothing.
func signatureFormatFor(signature string) string {
	if signature == "" {
		return ""
	}
	return signatureFormatBedrockConverse
}

// replayableSignature returns the signature to place on the wire for block, or
// an error when block carries one this dialect cannot claim. An absent
// signature is not an error; a foreign or unlabelled one is. See
// ForeignReasoningSignatureError for why neither forwarding nor dropping is a
// usable degrade.
func replayableSignature(block *content.ThinkingBlock) (string, error) {
	if block.Signature == "" {
		return "", nil
	}
	signature, ok := block.SignatureReplayableAs(signatureFormatBedrockConverse)
	if !ok {
		return "", &ForeignReasoningSignatureError{Format: block.SignatureFormat}
	}
	return signature, nil
}

func redactedStateToBytes(raw json.RawMessage) ([]byte, error) {
	var encoded string
	if err := json.Unmarshal(raw, &encoded); err != nil {
		return nil, &EncodeError{Reason: "invalid redacted reasoning ProviderState", Err: err}
	}
	decoded, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, &EncodeError{Reason: "invalid redacted reasoning ProviderState", Err: err}
	}
	return decoded, nil
}

func encodeImage(image *content.ImageBlock) (*imageContent, error) {
	if image == nil {
		return nil, unsupportedBlock(image, "nil block")
	}
	if image.Source.URL != "" {
		return nil, unsupportedBlock(image, "Bedrock Converse accepts inline image bytes, not URLs")
	}
	format := imageFormat(image.MediaType)
	if format == "" {
		return nil, unsupportedBlock(image, "unsupported image format")
	}
	if len(image.Source.Data) == 0 {
		return nil, unsupportedBlock(image, "image source is empty")
	}
	return &imageContent{Format: format, Source: imageSource{Bytes: append([]byte(nil), image.Source.Data...)}}, nil
}

func encodeDocument(document *content.DocumentBlock) (*documentContent, error) {
	if document == nil {
		return nil, unsupportedBlock(document, "nil block")
	}
	format := documentFormat(document.MediaType, document.Name)
	if format == "" {
		return nil, unsupportedBlock(document, "unsupported document format")
	}
	name := document.Name
	if name == "" {
		name = "document"
	}
	if err := validateDocumentName(name); err != nil {
		return nil, err
	}
	result := &documentContent{Format: format, Name: name}
	switch {
	case len(document.Data) > 0:
		result.Source.Bytes = append([]byte(nil), document.Data...)
	case document.Text != "":
		text := document.Text
		result.Source.Text = &text
	default:
		return nil, unsupportedBlock(document, "document source is empty")
	}
	return result, nil
}

// encodeAudio builds Converse's AudioBlock from a neutral audio block.
//
// The neutral vocabulary can only ever select the `bytes` member of
// AudioSource: content.AudioBlock carries a media type and a payload, and has
// no field that could name an S3 object. The empty-payload check is the Smithy
// model's own @length min 1 on that member — a zero-length blob encodes to
// {"bytes":""}, which fails AWS's length constraint after the request has
// already left the process.
func encodeAudio(audio *content.AudioBlock) (*audioContent, error) {
	if audio == nil {
		return nil, unsupportedBlock(audio, "nil block")
	}
	format := audioFormat(audio.MediaType)
	if format == "" {
		return nil, unsupportedBlock(audio, "media type "+string(audio.MediaType)+" has no AudioFormat enum member, so the required format field has no legal value")
	}
	if len(audio.Data) == 0 {
		return nil, unsupportedBlock(audio, "audio source is empty and AudioSource.bytes declares a minimum length of 1")
	}
	return &audioContent{Format: format, Source: audioSource{Bytes: append([]byte(nil), audio.Data...)}}, nil
}

func encodeToolUse(toolUse *content.ToolUseBlock) (*toolUseContent, error) {
	if toolUse == nil {
		return nil, unsupportedBlock(toolUse, "nil block")
	}
	if toolUse.ID == "" || toolUse.Name == "" {
		return nil, &ToolInputError{Tool: toolUse.Name, Reason: "tool-use id and name must not be empty"}
	}
	if reason := toolUseIDReason(toolUse.ID); reason != "" {
		return nil, &ToolInputError{Tool: toolUse.Name, Reason: reason}
	}
	if reason := toolNameReason(toolUse.Name); reason != "" {
		return nil, &ToolSchemaError{Tool: toolUse.Name, Reason: reason}
	}
	input, err := normalizedObject(toolUse.Input)
	if err != nil {
		return nil, &ToolInputError{Tool: toolUse.Name, Reason: err.Error()}
	}
	return &toolUseContent{ToolUseID: toolUse.ID, Name: toolUse.Name, Input: input}, nil
}

func encodeToolResultMessage(message *content.ToolResultMessage) (*toolResultContent, error) {
	if message == nil {
		return nil, &EncodeError{Reason: "nil tool result message"}
	}
	if message.ToolUseID == "" {
		return nil, &EncodeError{Reason: "tool result is missing tool-use id"}
	}
	if reason := toolUseIDReason(message.ToolUseID); reason != "" {
		return nil, &EncodeError{Reason: reason}
	}
	blocks, err := encodeToolResultBlocks(message.Blocks)
	if err != nil {
		return nil, err
	}
	result := &toolResultContent{ToolUseID: message.ToolUseID, Content: blocks}
	if message.IsError {
		result.Status = toolResultStatusError
	}
	return result, nil
}

func encodeToolResultBlock(block *content.ToolResultBlock) (*toolResultContent, error) {
	if block == nil {
		return nil, unsupportedBlock(block, "nil block")
	}
	if block.ToolUseID == "" {
		return nil, &EncodeError{Reason: "tool result is missing tool-use id"}
	}
	if reason := toolUseIDReason(block.ToolUseID); reason != "" {
		return nil, &EncodeError{Reason: reason}
	}
	blocks, err := encodeToolResultBlocks(block.Content)
	if err != nil {
		return nil, err
	}
	result := &toolResultContent{ToolUseID: block.ToolUseID, Content: blocks}
	if block.IsError {
		result.Status = toolResultStatusError
	}
	return result, nil
}

func encodeToolResultBlocks(blocks []content.Block) ([]toolResultBlock, error) {
	if len(blocks) == 0 {
		return nil, &EncodeError{Reason: "tool result content must not be empty"}
	}
	encoded := make([]toolResultBlock, 0, len(blocks))
	for _, block := range blocks {
		switch block := block.(type) {
		case *content.TextBlock:
			if block == nil {
				return nil, unsupportedBlock(block, "nil block")
			}
			text := block.Text
			encoded = append(encoded, toolResultBlock{Text: &text})
		case *content.ImageBlock:
			image, err := encodeImage(block)
			if err != nil {
				return nil, err
			}
			encoded = append(encoded, toolResultBlock{Image: image})
		case *content.DocumentBlock:
			document, err := encodeDocument(block)
			if err != nil {
				return nil, err
			}
			encoded = append(encoded, toolResultBlock{Document: document})
		case *content.AudioBlock:
			// ToolResultContentBlock is a different union from ContentBlock and
			// has no audio member: json, text, image, document, video and
			// searchResult only. An audio block IS encodable in an ordinary
			// message, so the generic "supports text, image, and document"
			// reason would misdescribe the limitation — and this is the shape an
			// MCP tool returning audio actually produces, so it is the one that
			// most needs to say where the wall is.
			return nil, unsupportedBlock(block, "Converse's toolResult content union has no audio member, so an audio tool result has no wire form")
		case *content.RefusalBlock:
			return nil, unsupportedRefusal(block)
		default:
			return nil, unsupportedBlock(block, "tool result content supports text, image, and document blocks")
		}
	}
	return encoded, nil
}

func validateTools(tools []inference.Tool) error {
	seen := make(map[string]struct{}, len(tools))
	for _, tool := range tools {
		if tool.Name == "" {
			return &ToolSchemaError{Reason: "tool name is empty"}
		}
		if reason := toolNameReason(tool.Name); reason != "" {
			return &ToolSchemaError{Tool: tool.Name, Reason: reason}
		}
		if _, exists := seen[tool.Name]; exists {
			return &ToolSchemaError{Tool: tool.Name, Reason: "duplicate tool name"}
		}
		seen[tool.Name] = struct{}{}
		trimmed := bytes.TrimSpace(tool.Schema)
		if len(trimmed) == 0 {
			return &ToolSchemaError{Tool: tool.Name, Reason: "schema is empty"}
		}
		if !json.Valid(trimmed) || trimmed[0] != '{' {
			return &ToolSchemaError{Tool: tool.Name, Reason: "schema must be a JSON object"}
		}
	}
	return nil
}

// samplingConfig builds InferenceConfiguration, holding every value to the
// range the Smithy model declares for it.
//
// The ranges are narrower than the shared model.Sampling vocabulary allows:
// Converse constrains temperature and topP to [0, 1] and maxTokens to >= 1,
// whereas an OpenAI-shaped temperature runs to 2. Cross-provider model
// switching therefore routes perfectly ordinary sampling values into a body
// Bedrock rejects with an HTTP 400. Forwarding them unchecked turns a local,
// nameable configuration error into an opaque provider rejection, so the codec
// refuses here instead — the same fail-secure rule the block encoders follow.
func samplingConfig(sampling model.Sampling) (*inferenceConfig, error) {
	if err := checkUnitInterval("temperature", sampling.Temperature); err != nil {
		return nil, err
	}
	if err := checkUnitInterval("topP", sampling.TopP); err != nil {
		return nil, err
	}
	if sampling.MaxTokens != nil && *sampling.MaxTokens < 1 {
		return nil, &EncodeError{Reason: fmt.Sprintf("maxTokens must be at least 1, got %d", *sampling.MaxTokens)}
	}
	config := &inferenceConfig{
		MaxTokens:     sampling.MaxTokens,
		Temperature:   sampling.Temperature,
		TopP:          sampling.TopP,
		StopSequences: sampling.Stop,
	}
	if config.MaxTokens == nil && config.Temperature == nil && config.TopP == nil && len(config.StopSequences) == 0 {
		return nil, nil
	}
	return config, nil
}

// checkUnitInterval enforces Converse's [0, 1] bound on a sampling knob.
func checkUnitInterval(field string, value *float64) error {
	if value == nil || (*value >= 0 && *value <= 1) {
		return nil
	}
	return &EncodeError{Reason: fmt.Sprintf("%s must be between 0 and 1, got %v", field, *value)}
}

// Converse's ToolUseId and ToolName constraints, transcribed from the Smithy
// model. Both patterns are ANCHORED: an unanchored form would accept "bad
// id/here" on the strength of a legal substring. ToolUseId additionally allows
// "." and ":", which ToolName does not.
var (
	toolUseIDPattern = regexp.MustCompile(`^[a-zA-Z0-9_.:-]+$`)
	toolNamePattern  = regexp.MustCompile(`^[a-zA-Z0-9_-]+$`)
)

const (
	maxToolUseIDLength = 64
	maxToolNameLength  = 64
)

// toolUseIDReason reports why a replayed tool-call identifier is not a legal
// Converse ToolUseId, or "" when it is one. It returns a bare reason rather
// than an error so each call site can attach the typed error its surrounding
// context already uses.
//
// Identifiers are minted by whichever provider issued the tool call, and the
// dialects disagree: Anthropic's ids match ^[a-zA-Z0-9_-]+$ with no length cap,
// Gemini's are synthesised locally, and Converse's own are capped at 64
// characters. Replaying a conversation into Bedrock can therefore carry an id
// Bedrock will not accept. Rejecting it locally names the violated constraint;
// forwarding it produces a ValidationException with no such clue.
func toolUseIDReason(id string) string {
	switch {
	case id == "":
		return "tool-use id must not be empty"
	case len(id) > maxToolUseIDLength:
		return fmt.Sprintf("tool-use id is %d characters, which exceeds Converse's %d-character limit", len(id), maxToolUseIDLength)
	case !toolUseIDPattern.MatchString(id):
		return "tool-use id must match " + toolUseIDPattern.String()
	}
	return ""
}

// toolNameReason reports why a tool name is not a legal Converse ToolName, or
// "" when it is one. Converse's ToolName is stricter than the names tool
// servers actually hand out — MCP names routinely carry "." or "/" — so an
// otherwise working tool set can be unrepresentable here.
func toolNameReason(name string) string {
	switch {
	case name == "":
		return "tool name is empty"
	case len(name) > maxToolNameLength:
		return fmt.Sprintf("tool name is %d characters, which exceeds Converse's %d-character limit", len(name), maxToolNameLength)
	case !toolNamePattern.MatchString(name):
		return "tool name must match " + toolNamePattern.String()
	}
	return ""
}

func normalizedObject(raw []byte) (json.RawMessage, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return json.RawMessage(`{}`), nil
	}
	if !json.Valid(trimmed) || trimmed[0] != '{' {
		return nil, fmt.Errorf("arguments must be a JSON object")
	}
	return append(json.RawMessage(nil), trimmed...), nil
}

func imageFormat(mediaType content.MediaType) string {
	switch strings.ToLower(string(mediaType)) {
	case string(content.MediaTypeImageJPEG):
		return imageFormatJPEG
	case string(content.MediaTypeImagePNG):
		return imageFormatPNG
	case string(content.MediaTypeImageGIF):
		return imageFormatGIF
	case string(content.MediaTypeImageWebP):
		return imageFormatWebP
	default:
		return ""
	}
}

// audioFormat maps a shared media type onto an AudioFormat enum member, or ""
// when none names it.
//
// This is deliberately an allowlist over the media types Looprig defines rather
// than a translation of the enum: AudioFormat carries fifteen members and the
// two vocabularies are not in bijection. audio/mpeg selects "mp3" (both "mp3"
// and "mpeg" are legal members for that media type; "mp3" is the one that
// matches the extension the shared constant documents), and audio/mp4 selects
// "mp4" over the equivalent "m4a" for the same reason. A media type with no
// member — audio/opus, audio/amr, anything AWS adds later — returns "" and the
// caller fails closed.
func audioFormat(mediaType content.MediaType) string {
	switch content.MediaType(strings.ToLower(string(mediaType))) {
	case content.MediaTypeAudioMPEG:
		return audioFormatMP3
	case content.MediaTypeAudioWAV:
		return audioFormatWAV
	case content.MediaTypeAudioOGG:
		return audioFormatOGG
	case content.MediaTypeAudioFLAC:
		return audioFormatFLAC
	case content.MediaTypeAudioAAC:
		return audioFormatAAC
	case content.MediaTypeAudioMP4:
		return audioFormatMP4
	case content.MediaTypeAudioWebM:
		return audioFormatWebM
	default:
		return ""
	}
}

func documentFormat(mediaType content.MediaType, name string) string {
	switch strings.ToLower(string(mediaType)) {
	case string(content.MediaTypeDocumentPDF):
		return "pdf"
	case string(content.MediaTypeDocumentText):
		return "txt"
	case string(content.MediaTypeDocumentHTML):
		return "html"
	case string(content.MediaTypeDocumentCSV):
		return "csv"
	case string(content.MediaTypeDocumentMarkdown):
		return "md"
	case string(content.MediaTypeDocumentDOCX):
		return "docx"
	case string(content.MediaTypeDocumentXLSX):
		return "xlsx"
	case "application/msword":
		return "doc"
	case "application/vnd.ms-excel":
		return "xls"
	}
	ext := strings.TrimPrefix(strings.ToLower(filepath.Ext(name)), ".")
	switch ext {
	case "pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md":
		return ext
	default:
		return ""
	}
}

func validateDocumentName(name string) error {
	if name == "" {
		return &UnsupportedBlockError{Block: "*content.DocumentBlock", Reason: "document name is empty"}
	}
	runes := []rune(name)
	if len(runes) > 200 {
		return &UnsupportedBlockError{Block: "*content.DocumentBlock", Reason: "document name exceeds 200 characters"}
	}
	previousWhitespace := false
	for _, r := range runes {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			previousWhitespace = false
		case r == '-', r == '(', r == ')', r == '[', r == ']':
			previousWhitespace = false
		case isDocumentWhitespace(r):
			if previousWhitespace {
				return &UnsupportedBlockError{Block: "*content.DocumentBlock", Reason: "document name contains consecutive whitespace"}
			}
			previousWhitespace = true
		default:
			return &UnsupportedBlockError{Block: "*content.DocumentBlock", Reason: "document name contains unsupported characters"}
		}
	}
	return nil
}

func isDocumentWhitespace(r rune) bool {
	return r == ' ' || r == '\t' || r == '\n' || r == '\r'
}

func unsupportedBlock(block content.Block, reason string) error {
	return &UnsupportedBlockError{Block: fmt.Sprintf("%T", block), Reason: reason}
}
