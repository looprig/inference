package openairesponses

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	model "github.com/looprig/inference/model"
)

// EncodeRequest converts a provider-neutral inference.Request into an OpenAI
// Responses `POST /v1/responses` JSON body. stream=true adds "stream":true to
// the body. Request.System (plus any in-thread SystemMessage) becomes the
// top-level `instructions` field: Responses has no in-thread system role, so
// a SystemMessage folds into instructions exactly as anthropicapi folds it
// into the top-level `system` field.
func EncodeRequest(req inference.Request, stream bool) ([]byte, error) {
	if err := inference.ValidateRequestFeatures(req); err != nil {
		return nil, err
	}
	r, err := buildResponsesRequest(req, stream)
	if err != nil {
		return nil, err
	}
	return json.Marshal(r)
}

// The sampling intervals CreateResponse declares, reached through
// CreateModelResponseProperties -> ModelResponseProperties — the same schema
// Chat Completions inherits, hence the same numbers as openaiapi's. They ARE in
// the derived request document (minimum/maximum), so the conformance gate
// rejects a violation too; checking here means the failure names the field at
// the moment the request is built rather than after a body has been marshalled.
//
// Only sampling is checked. The sibling openaiapi encoder also enforces
// FunctionObject.name's published class (64 characters, [a-zA-Z0-9_-]) because
// Chat Completions states it in prose that the derived schema does not carry.
// That rule DOES NOT transfer: this dialect's FunctionTool types `name` as a
// bare string and states no constraint on it anywhere, so transplanting Chat's
// class would invent a rule Responses never published and would reject the
// dotted/slashed names MCP servers routinely publish. See
// TestTheResponsesRequestGateHoldsSamplingButNotToolNames, which measures both
// halves rather than asserting them.
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

// buildResponsesRequest assembles the typed request struct. Split from
// marshaling so the mapping is unit-testable without a JSON round-trip.
func buildResponsesRequest(req inference.Request, stream bool) (wireRequest, error) {
	// Effective sampling: a non-nil per-call Override REPLACES Model.Sampling
	// wholesale, so the check has to run on the effective value — the override
	// is the field cross-provider switching actually populates.
	sampling := req.Model.Sampling
	if req.Override != nil {
		sampling = *req.Override
	}
	if err := checkSamplingRange("temperature", sampling.Temperature, minTemperature, maxTemperature); err != nil {
		return wireRequest{}, err
	}
	if err := checkSamplingRange("top_p", sampling.TopP, minTopP, maxTopP); err != nil {
		return wireRequest{}, err
	}

	instructions := req.System
	// Non-nil from the start: `input` is spec-typed string-or-array with no
	// null alternative, so an empty conversation must still marshal as [].
	items := []wireItem{}
	for _, conv := range req.Messages {
		switch m := conv.(type) {
		case *content.SystemMessage:
			instructions = appendSystem(instructions, textOf(m.Blocks))
		case *content.UserMessage:
			parts, err := encodeUserContentParts(m.Blocks)
			if err != nil {
				return wireRequest{}, err
			}
			items = append(items, wireItem{Type: itemTypeMessage, Role: roleUser, Content: partsContent(parts)})
		case *content.AIMessage:
			encoded, err := blocksToItems(m.Blocks, nil)
			if err != nil {
				return wireRequest{}, err
			}
			items = append(items, encoded...)
		case *content.ToolResultMessage:
			item, err := encodeToolResultItem(m)
			if err != nil {
				return wireRequest{}, err
			}
			items = append(items, item)
		default:
			return wireRequest{}, &UnsupportedConversationError{Conversation: fmt.Sprintf("%T", conv)}
		}
	}

	r := wireRequest{
		Model:           req.Model.Name,
		Instructions:    instructions,
		Input:           items,
		MaxOutputTokens: sampling.MaxTokens,
		Temperature:     sampling.Temperature,
		TopP:            sampling.TopP,
		// Store is always explicit false: this project excludes server-stored
		// Responses conversations/previous_response_id, and the provider's
		// unstated default for an omitted `store` cannot be relied upon.
		//
		// The OpenAI Conversations API (POST /v1/conversations, and
		// CreateResponse's `conversation` member, which threads a request onto
		// a server-side conversation object) is excluded for the same reason
		// and two more. Server-side conversation storage contradicts Carbon's
		// RetentionNone declaration — the transcript would live on OpenAI's
		// servers past the request. It is OpenAI-only, with no equivalent on
		// Anthropic, Gemini or Bedrock, so adopting it would fragment the
		// neutral transcript: a session that switched models would have half
		// its history on a provider and half in the local store, and nothing
		// could reconcile the two. And stateless replay — every request
		// carrying its whole input array — is the deliberate architecture, the
		// same choice that makes cross-provider model switching and local
		// compaction possible at all.
		Store:  false,
		Stream: stream,
	}

	// Sampling.Stop has no Responses wire representation: the real API
	// rejects an unrecognized "stop" field outright ("Unknown parameter:
	// 'stop'"), unlike Chat Completions. Silently omitting it (rather than
	// failing the request) matches this codec's other dialect-specific
	// omissions, e.g. anthropicapi dropping temperature/top_p under thinking.

	for _, t := range req.Tools {
		r.Tools = append(r.Tools, wireTool{
			Type:        toolTypeFunction,
			Name:        t.Name,
			Description: t.Description,
			Parameters:  schemaOrDefault(t.Schema),
		})
	}
	switch req.ToolChoice.Mode() {
	case inference.ToolChoiceModeRequired:
		r.ToolChoice = json.RawMessage(toolChoiceRequiredJSON)
	case inference.ToolChoiceModeNamed:
		name, _ := req.ToolChoice.Named()
		choice, err := json.Marshal(namedToolChoice{Type: toolTypeFunction, Name: name})
		if err != nil {
			return wireRequest{}, err
		}
		r.ToolChoice = choice
	}

	if req.Output != nil {
		r.Text = &wireText{Format: &wireTextFormat{
			Type:   textFormatJSONSchema,
			Name:   req.Output.Name,
			Schema: req.Output.Schema,
			Strict: req.Output.Strict,
		}}
	}

	// effort -> reasoning, gated by the model's advertised Thinking
	// capability, mirroring exactly how anthropicapi gates its own thinking
	// config on req.Model.Caps.Thinking. When reasoning is enabled, `include`
	// requests encrypted reasoning content back so a same-dialect follow-up
	// request can replay it (ThinkingBlock.ProviderState).
	if req.Model.Caps.Thinking {
		if ev := effortValue(sampling.Effort); ev != "" {
			r.Reasoning = &wireReasoningConfig{Effort: ev, Summary: "auto"}
			r.Include = append(r.Include, includeEncryptedReasoningContent)
		}
	}

	return r, nil
}

// blocksToItems maps a slice of neutral content blocks (an AIMessage's
// Blocks, whether from an outbound-request replay or a server-encoded
// response) to Responses items, preserving block order and grouping
// consecutive text blocks into one message item — the inverse of the
// items-based wire shape's grouping on decode.
//
// ids selects the direction, because the two positions have genuinely
// different schemas. Non-nil (the server response direction) means "this
// process is the authority producing the response": items are OutputMessage /
// ReasoningItem / FunctionToolCall values, so ids synthesizes the item ids
// those require. Nil (the client request-replay direction) means every id
// must already be real, having come from a prior actual response; an item
// whose required id is unavailable takes the id-free EasyInputMessage form
// where one exists, and is dropped where none does.
func blocksToItems(blocks []content.Block, ids func() string) ([]wireItem, error) {
	response := ids != nil

	var items []wireItem
	var pendingParts []wireContentPart

	flush := func() {
		if len(pendingParts) == 0 {
			return
		}
		if response {
			// OutputMessage.required: id, type, role, content, status.
			items = append(items, wireItem{
				Type:    itemTypeMessage,
				ID:      ids(),
				Role:    roleAssistant,
				Status:  statusCompleted,
				Content: partsContent(pendingParts),
			})
		} else {
			// Replayed assistant history has no message id — no neutral block
			// carries one — so an OutputMessage is unbuildable. EasyInputMessage
			// requires only role and content, and its content may be the bare
			// string these parts concatenate to. Only TEXT parts ever reach
			// this branch: a refusal is refused below, precisely because it
			// cannot survive this degrade.
			items = append(items, wireItem{Role: roleAssistant, Content: textContent(joinParts(pendingParts))})
		}
		pendingParts = nil
	}

	for _, b := range blocks {
		switch b := b.(type) {
		case *content.TextBlock:
			// OutputTextContent.required is ["type","text","annotations",
			// "logprobs"]; both arrays are empty because a gateway-served text
			// block carries no citations and no logprobs, and neither member
			// admits null. outputTextWire (types.go) is what puts them on the
			// wire — they are stated here as well so the construction site
			// matches the streaming one, which does the same.
			pendingParts = append(pendingParts, wireContentPart{
				Type:        contentTypeOutputText,
				Text:        b.Text,
				Annotations: []json.RawMessage{},
				Logprobs:    []json.RawMessage{},
			})
		case *content.RefusalBlock:
			// OutputContent is output_text | refusal, so a refusal is another
			// part of the same message item — in the RESPONSE direction, where
			// this process synthesizes the OutputMessage id the schema requires.
			//
			// The request-replay direction has no legal wire form for it, and
			// that is measured, not assumed: a refusal part is only admissible
			// inside an OutputMessage (required: id, type, role, content,
			// status), replayed history carries no message id, and
			// EasyInputMessage — the one id-free assistant form — takes
			// InputContent (input_text|input_image|input_file), which has no
			// refusal member. Emitting an OutputMessage without an id produces a
			// body the provider's own request schema rejects; degrading to
			// output_text or a bare string re-sends the model its own refusal as
			// something it said, which is the defect content.RefusalBlock exists
			// to remove; dropping it loses the fact that the model declined. So
			// it fails closed, naming the limitation.
			if !response {
				return nil, &UnsupportedBlockError{
					Block:  fmt.Sprintf("%T", b),
					Reason: "a Responses request can carry a refusal only inside an OutputMessage, whose required id no replayed assistant block holds",
				}
			}
			pendingParts = append(pendingParts, wireContentPart{Type: contentTypeRefusal, Refusal: b.Text})
		case *content.ToolUseBlock:
			flush()
			id := b.ID
			if id == "" && ids != nil {
				id = ids()
			}
			items = append(items, wireItem{
				Type:      itemTypeFunctionCall,
				CallID:    id,
				Name:      b.Name,
				Arguments: argumentsOrEmpty(b.Input),
			})
		case *content.ThinkingBlock:
			flush()
			item := wireItem{Type: itemTypeReasoning}
			if b.Thinking != "" {
				item.Summary = []wireSummaryPart{{Type: summaryTypeText, Text: b.Thinking}}
			}
			if b.ReplayableAs(providerStateFormatOpenAIResponses) {
				state, err := opaqueStateToWire(b.ProviderState)
				if err != nil {
					return nil, err
				}
				item.ID = state.ID
				item.EncryptedContent = state.EncryptedContent
			}
			if item.ID == "" {
				if !response {
					// ReasoningItem.required includes id, and there is none to
					// replay: state predating id preservation, or thinking that
					// never came from a Responses target. Emitting the item
					// anyway is a guaranteed "Missing required parameter:
					// 'input[N].id'", and fabricating an id is worse still — the
					// provider would reject an id it never issued. Dropping it
					// only costs a summary the model regenerates server-side.
					continue
				}
				item.ID = ids()
			}
			items = append(items, item)
		default:
			return nil, &UnsupportedBlockError{Block: fmt.Sprintf("%T", b)}
		}
	}
	flush()
	return items, nil
}

// joinParts concatenates content parts' text, for the bare-string
// EasyInputMessage content form.
func joinParts(parts []wireContentPart) string {
	var sb strings.Builder
	for _, p := range parts {
		sb.WriteString(p.Text)
	}
	return sb.String()
}

// encodeToolResultItem builds the function_call_output item from a
// ToolResultMessage. Output is a plain string on the wire (unlike
// Anthropic's structured tool_result content), so non-text blocks fail
// closed rather than being silently dropped — matching openaiapi's
// toolResultText. IsError has no Responses wire representation (no
// function_call_output field models it); this is a documented, intentional
// loss for this ingress-independent direction, mirroring how openaiapi never
// emits ToolResultMessage.IsError either.
func encodeToolResultItem(m *content.ToolResultMessage) (wireItem, error) {
	text, err := toolResultText(m.Blocks)
	if err != nil {
		return wireItem{}, err
	}
	return wireItem{Type: itemTypeFunctionCallOutput, CallID: m.ToolUseID, Output: text}, nil
}

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

// encodeUserContentParts maps a user message's blocks to the three members of
// the Responses input content union: input_text, input_image and input_file.
// A block type this dialect does not model in a user turn yields a typed
// *UnsupportedBlockError naming the limitation.
func encodeUserContentParts(blocks []content.Block) ([]wireContentPart, error) {
	parts := make([]wireContentPart, 0, len(blocks))
	for _, b := range blocks {
		switch b := b.(type) {
		case *content.TextBlock:
			parts = append(parts, wireContentPart{Type: contentTypeInputText, Text: b.Text})
		case *content.ImageBlock:
			parts = append(parts, wireContentPart{
				Type:     contentTypeInputImage,
				ImageURL: imageURLOf(b),
				Detail:   imageDetailAuto,
			})
		case *content.DocumentBlock:
			part, err := documentInputFilePart(b)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part)
		case *content.AudioBlock:
			// Audio has no Responses input representation. The spec does
			// declare an InputAudio object, but nothing in a Responses request
			// references it: InputContent — the item type of
			// InputMessageContentList — is exactly
			// input_text|input_image|input_file, and InputAudio is reachable
			// only from EvalItemContentItem, the Evals API's content union.
			// Chat Completions is the OpenAI dialect that takes audio input
			// (codec/openaiapi encodes it as an input_audio part), so failing
			// here names a real routing decision rather than a missing feature.
			return nil, &UnsupportedBlockError{
				Block:  fmt.Sprintf("%T", b),
				Reason: "the Responses input content union has no audio member (input_text|input_image|input_file)",
			}
		default:
			return nil, &UnsupportedBlockError{Block: fmt.Sprintf("%T", b)}
		}
	}
	return parts, nil
}

// documentInputFilePart builds an input_file content part from a
// DocumentBlock, mirroring openaiapi's documentFilePart: only the inline
// filename+file_data form is reachable, because `file_id` and `file_url` name
// resources outside the neutral transcript. The media type rides in a data:
// URI so it is not lost to file_data's plain-string typing, exactly as this
// codec already does for inline image bytes (imageURLOf), and the optional
// `detail` is left unset so the provider's own default rendering applies.
func documentInputFilePart(doc *content.DocumentBlock) (wireContentPart, error) {
	payload := documentPayload(doc)
	if len(payload) == 0 {
		return wireContentPart{}, &UnsupportedBlockError{Block: "*content.DocumentBlock", Reason: "document carries neither data nor text"}
	}
	if doc.Name == "" {
		return wireContentPart{}, &UnsupportedBlockError{Block: "*content.DocumentBlock", Reason: "document name is empty and file_data requires a filename"}
	}
	encoded := base64.StdEncoding.EncodeToString(payload)
	if doc.MediaType != "" {
		encoded = dataURIPrefix + string(doc.MediaType) + ";base64," + encoded
	}
	return wireContentPart{Type: contentTypeInputFile, Filename: doc.Name, FileData: encoded}, nil
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

// imageURLOf builds the `image_url` string for an ImageBlock. A URL takes
// precedence over inline Data; Data becomes a data: URI.
func imageURLOf(img *content.ImageBlock) string {
	if img.Source.URL != "" {
		return img.Source.URL
	}
	encoded := base64.StdEncoding.EncodeToString(img.Source.Data)
	return dataURIPrefix + string(img.MediaType) + ";base64," + encoded
}

// argumentsOrEmpty returns raw's text as the wire `arguments` string,
// defaulting to "{}" for an empty/absent input (Responses requires function
// call arguments to be a JSON object).
func argumentsOrEmpty(raw json.RawMessage) string {
	if len(raw) == 0 {
		return emptyObject
	}
	return string(raw)
}

// schemaOrDefault guarantees a tool's `parameters` is a JSON object.
func schemaOrDefault(schema json.RawMessage) json.RawMessage {
	if len(schema) == 0 {
		return json.RawMessage(defaultSchema)
	}
	return schema
}

// effortValue maps the dialect-neutral model.Effort to Responses'
// reasoning.effort wire value. The current request schema admits the complete
// neutral ladder, so every known non-none value is preserved exactly.
// EffortNone (and any unknown value, fail-safe) yields "", which suppresses
// the whole `reasoning` field.
func effortValue(e model.Effort) string {
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
	default: // EffortNone or unknown -> omit
		return ""
	}
}

// appendSystem joins two instruction fragments, inserting a blank-line
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

// opaqueStateToWire unmarshals a ThinkingBlock.ProviderState (which this
// codec stores as a reasoningState object — see opaqueStateFromWire,
// decode.go) back into the wire members a replayed reasoning item needs.
//
// A bare JSON string is the legacy encoding, from before the reasoning item's
// required id was preserved; it is accepted as encrypted content with no id,
// so state persisted by an older build still parses rather than failing the
// whole request. Such an item is not replayable (see blocksToItems).
func opaqueStateToWire(raw json.RawMessage) (reasoningState, error) {
	if legacy := bytes.TrimSpace(raw); len(legacy) > 0 && legacy[0] == '"' {
		var s string
		if err := json.Unmarshal(legacy, &s); err != nil {
			return reasoningState{}, fmt.Errorf("openairesponses: invalid ThinkingBlock.ProviderState: %w", err)
		}
		return reasoningState{EncryptedContent: s}, nil
	}
	var state reasoningState
	if err := json.Unmarshal(raw, &state); err != nil {
		return reasoningState{}, fmt.Errorf("openairesponses: invalid ThinkingBlock.ProviderState: %w", err)
	}
	return state, nil
}
