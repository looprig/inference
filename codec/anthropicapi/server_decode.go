package anthropicapi

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"net/http"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	codec "github.com/looprig/inference/codec"
	"github.com/looprig/inference/internal/jsonstrict"
	model "github.com/looprig/inference/model"
	"github.com/looprig/inference/wire/jsonbody"
)

// pathMessages and pathCountTokens are the two native Anthropic Messages API
// routes this codec recognizes: the ServerCodec-owned `/v1/messages` and the
// narrower, separately-invoked `/v1/messages/count_tokens` auxiliary endpoint.
const (
	pathMessages     = "/v1/messages"
	pathCountTokens  = "/v1/messages/count_tokens" // #nosec G101 -- a URL path, not a credential
	toolChoiceAuto   = "auto"
	toolChoiceNone   = "none"
	thinkingBudgeted = "enabled" // real Anthropic manual-budget wire value; unsupported here
)

// matchMessagesRequest reports whether req is a POST /v1/messages request. It
// does not inspect the body or Content-Type; DecodeRequest owns that rejection.
func matchMessagesRequest(req *http.Request) bool {
	return req.Method == http.MethodPost && req.URL.Path == pathMessages
}

// decodeMessagesRequest decodes a matched POST /v1/messages request into a
// codec.DecodedRequest. Request.Model is left at its zero value: the harness
// alias travels only in RequestedModel, and resolving it to a real Target is the
// gateway's job (this codec has no routing table).
func decodeMessagesRequest(req *http.Request) (codec.DecodedRequest, error) {
	body, err := readJSONBody(req)
	if err != nil {
		return codec.DecodedRequest{}, err
	}
	r, requestedModel, streaming, err := decodeMessagesBody(body)
	if err != nil {
		return codec.DecodedRequest{}, err
	}
	return codec.DecodedRequest{Request: r, RequestedModel: requestedModel, Streaming: streaming}, nil
}

// MatchCountTokensRequest reports whether req is a POST
// /v1/messages/count_tokens request. It is not part of codec.ServerCodec: the
// count_tokens endpoint is a separate, narrower auxiliary the gateway wires up on
// its own route, composing this decode helper with its own target resolution and
// a contextcount.ContextCounter.
func MatchCountTokensRequest(req *http.Request) bool {
	return req.Method == http.MethodPost && req.URL.Path == pathCountTokens
}

// DecodeCountTokensRequest decodes a POST /v1/messages/count_tokens request
// using the same semantic message/tool/system/image/thinking decoding as
// DecodeRequest. The count_tokens wire body has the same shape as a Messages
// request minus max_tokens/stream (both are optional on decode already), so it
// shares decodeMessagesBody. Streaming is always false: count_tokens has no
// streaming mode. As with decodeMessagesRequest, Request.Model stays unresolved.
func DecodeCountTokensRequest(req *http.Request) (codec.DecodedRequest, error) {
	body, err := readJSONBody(req)
	if err != nil {
		return codec.DecodedRequest{}, err
	}
	r, requestedModel, _, err := decodeMessagesBody(body)
	if err != nil {
		return codec.DecodedRequest{}, err
	}
	return codec.DecodedRequest{Request: r, RequestedModel: requestedModel, Streaming: false}, nil
}

// readJSONBody enforces the application/json Content-Type and reads the full
// request body. It never blocks past the request's own body availability, so a
// caller with an already-canceled context still returns promptly for an
// in-memory or already-buffered body.
func readJSONBody(req *http.Request) ([]byte, error) {
	if err := checkJSONContentType(req); err != nil {
		return nil, err
	}
	body, err := io.ReadAll(req.Body)
	if err != nil {
		return nil, &ServerDecodeError{Reason: "read_body", Detail: err.Error()}
	}
	return body, nil
}

func checkJSONContentType(req *http.Request) error {
	ct := req.Header.Get("Content-Type")
	mediaType, _, err := mime.ParseMediaType(ct)
	if err != nil || mediaType != jsonbody.ContentType {
		return &ServerDecodeError{Reason: "unsupported_content_type", Detail: ct}
	}
	return nil
}

// wireDecodeRequest is the server-decode-direction wire form of the Messages API
// request body. It reuses the existing anthropicMessage/anthropicBlock/
// anthropicTool/toolChoice/thinkingConfig/outputConfig types (encode.go/types.go)
// wherever they losslessly model the same shape; System uses the decode-only
// wireSystem to accept both wire forms of the `system` field. Metadata is
// decoded and then intentionally dropped: it is a documented benign field.
//
// MaxTokens is a pointer (not required) because this same struct backs both
// POST /v1/messages (where a real client always sends it) and POST
// /v1/messages/count_tokens (which never carries it).
type wireDecodeRequest struct {
	Model         string             `json:"model"`
	System        wireSystem         `json:"system"`
	Messages      []anthropicMessage `json:"messages"`
	Tools         []anthropicTool    `json:"tools"`
	ToolChoice    *toolChoice        `json:"tool_choice"`
	MaxTokens     *int               `json:"max_tokens"`
	Temperature   *float64           `json:"temperature"`
	TopP          *float64           `json:"top_p"`
	StopSequences []string           `json:"stop_sequences"`
	Thinking      *thinkingConfig    `json:"thinking"`
	OutputConfig  *outputConfig      `json:"output_config"`
	Stream        bool               `json:"stream"`
	Metadata      json.RawMessage    `json:"metadata"`
}

// wireSystem decodes the top-level `system` field, which Anthropic allows as
// either a bare string or an array of text blocks (the array form exists solely
// to carry a cache_control breakpoint on the system prompt). Both forms
// normalize to plain Text; any cache_control markers on the array form are
// accepted and dropped, matching the block-level handling.
type wireSystem struct {
	Text string
}

func (s *wireSystem) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		return nil
	}
	switch trimmed[0] {
	case '"':
		return json.Unmarshal(data, &s.Text)
	case '[':
		dec := json.NewDecoder(bytes.NewReader(data))
		dec.DisallowUnknownFields()
		var blocks []anthropicBlock
		if err := dec.Decode(&blocks); err != nil {
			return &ServerDecodeError{Reason: "invalid_system", Detail: err.Error()}
		}
		var text bytes.Buffer
		for _, b := range blocks {
			if b.Type != blockTypeText {
				return &ServerDecodeError{Reason: "unsupported_system_block", Detail: b.Type}
			}
			text.WriteString(b.Text)
		}
		s.Text = text.String()
		return nil
	default:
		return &ServerDecodeError{Reason: "invalid_system", Detail: "must be a string or an array of text blocks"}
	}
}

// decodeMessagesBody is the shared semantic decode core behind both
// decodeMessagesRequest and DecodeCountTokensRequest. It enforces unique object
// keys, strict field recognition (DisallowUnknownFields, so any unsupported wire
// field fails closed rather than being silently dropped — except the two
// documented benign fields: cache_control anywhere and top-level metadata,
// which decode successfully but are never mapped into the neutral Request), and
// maps the wire shape into a provider-neutral inference.Request. It never
// resolves Model — only the string alias is returned.
func decodeMessagesBody(raw []byte) (req inference.Request, requestedModel string, streaming bool, err error) {
	if dupErr := rejectDuplicateObjectKeys(raw); dupErr != nil {
		return inference.Request{}, "", false, dupErr
	}

	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	var wire wireDecodeRequest
	if err := dec.Decode(&wire); err != nil {
		return inference.Request{}, "", false, &ServerDecodeError{Reason: "malformed_body", Detail: err.Error()}
	}
	var trailing json.RawMessage
	if err := dec.Decode(&trailing); !errors.Is(err, io.EOF) {
		return inference.Request{}, "", false, &ServerDecodeError{Reason: "trailing_data"}
	}

	if wire.Model == "" {
		return inference.Request{}, "", false, &ServerDecodeError{Reason: "missing_model"}
	}

	var messages content.AgenticMessages
	for _, m := range wire.Messages {
		decoded, err := decodeWireMessage(m)
		if err != nil {
			return inference.Request{}, "", false, err
		}
		messages = append(messages, decoded...)
	}

	var tools []inference.Tool
	for _, t := range wire.Tools {
		tools = append(tools, inference.Tool{Name: t.Name, Description: t.Description, Schema: t.InputSchema})
	}

	toolChoice, err := decodeToolChoice(wire.ToolChoice)
	if err != nil {
		return inference.Request{}, "", false, err
	}

	sampling := model.Sampling{}
	if wire.MaxTokens != nil {
		sampling.MaxTokens = wire.MaxTokens
	}
	if wire.Temperature != nil {
		sampling.Temperature = wire.Temperature
	}
	if wire.TopP != nil {
		sampling.TopP = wire.TopP
	}
	if len(wire.StopSequences) > 0 {
		sampling.Stop = wire.StopSequences
	}

	if wire.Thinking != nil {
		if wire.Thinking.Type != thinkingTypeAdaptive {
			return inference.Request{}, "", false, &ServerDecodeError{Reason: "unsupported_thinking_mode", Detail: wire.Thinking.Type}
		}
		if wire.OutputConfig == nil || wire.OutputConfig.Effort == "" {
			return inference.Request{}, "", false, &ServerDecodeError{Reason: "missing_effort_for_thinking"}
		}
		effort, err := parseEffort(wire.OutputConfig.Effort)
		if err != nil {
			return inference.Request{}, "", false, err
		}
		sampling.Effort = effort
	}

	var output *inference.OutputSchema
	if wire.OutputConfig != nil && wire.OutputConfig.Format != nil {
		output = &inference.OutputSchema{Schema: wire.OutputConfig.Format.Schema}
	}

	req = inference.Request{
		System:     wire.System.Text,
		Messages:   messages,
		Tools:      tools,
		Output:     output,
		ToolChoice: toolChoice,
		Override:   &sampling,
	}
	return req, wire.Model, wire.Stream, nil
}

// decodeToolChoice maps the wire tool_choice to the neutral
// inference.ToolChoice. A "none" (disable tools) choice is real Anthropic
// behavior the neutral vocabulary cannot represent, so it fails closed rather
// than silently degrading to auto/required; so does a "tool" choice with no
// name, which a named neutral choice cannot spell either.
func decodeToolChoice(tc *toolChoice) (inference.ToolChoice, error) {
	if tc == nil {
		return inference.ToolAuto(), nil
	}
	switch tc.Type {
	case toolChoiceAuto:
		return inference.ToolAuto(), nil
	case toolChoiceAny:
		return inference.ToolRequired(), nil
	case toolChoiceTool:
		if tc.Name == "" {
			return inference.ToolAuto(), &ServerDecodeError{Reason: "unsupported_tool_choice", Detail: tc.Type}
		}
		return inference.ToolNamed(tc.Name), nil
	default:
		return inference.ToolAuto(), &ServerDecodeError{Reason: "unsupported_tool_choice", Detail: tc.Type}
	}
}

// parseEffort maps the wire output_config.effort value to the neutral
// model.Effort, inverting effortValue (encode.go).
func parseEffort(wire string) (model.Effort, error) {
	switch wire {
	case "low":
		return model.EffortLow, nil
	case "medium":
		return model.EffortMedium, nil
	case "high":
		return model.EffortHigh, nil
	case "max":
		return model.EffortMax, nil
	default:
		return model.EffortNone, &ServerDecodeError{Reason: "unsupported_effort", Detail: wire}
	}
}

// decodeWireMessage maps one wire `messages` entry to one or more neutral
// Conversation turns. An assistant message maps 1:1 to a single AIMessage
// (parallel tool_use blocks stay together as multiple blocks on that one
// message, matching content.AIMessage's own multi-block shape). A user message
// may expand to several turns: Anthropic represents each tool result as a
// tool_result content block inside a user-role message (including several at
// once for parallel tool calls), but the neutral vocabulary models each tool
// result as its own ToolResultMessage turn.
func decodeWireMessage(m anthropicMessage) ([]content.Conversation, error) {
	switch m.Role {
	case roleAssistant:
		blocks, err := decodeRequestBlocks(m.Content, false)
		if err != nil {
			return nil, err
		}
		return []content.Conversation{&content.AIMessage{
			Message: content.Message{Role: content.RoleAssistant, Blocks: blocks},
		}}, nil
	case roleUser:
		return decodeUserMessage(m.Content)
	default:
		return nil, &ServerDecodeError{Reason: "unsupported_role", Detail: m.Role}
	}
}

// decodeUserMessage splits a wire user message's content into the neutral turns
// it represents: consecutive non-tool_result blocks are grouped into one
// UserMessage (preserving their order), and each tool_result block becomes its
// own ToolResultMessage, in the order both kinds appear on the wire.
func decodeUserMessage(blocks []anthropicBlock) ([]content.Conversation, error) {
	var out []content.Conversation
	var pending []content.Block

	flush := func() {
		if len(pending) > 0 {
			out = append(out, &content.UserMessage{
				Message: content.Message{Role: content.RoleUser, Blocks: pending},
			})
			pending = nil
		}
	}

	for _, b := range blocks {
		if b.Type == blockTypeToolResult {
			flush()
			trMsg, err := decodeToolResultBlock(b)
			if err != nil {
				return nil, err
			}
			out = append(out, trMsg)
			continue
		}
		db, err := decodeRequestBlock(b, true)
		if err != nil {
			return nil, err
		}
		pending = append(pending, db)
	}
	flush()
	return out, nil
}

// decodeToolResultBlock maps one wire tool_result block to a ToolResultMessage.
// Only the array-of-blocks form of `content` is supported (not Anthropic's
// alternative bare-string shorthand); a real client sending the shorthand form
// gets a typed decode error rather than silently losing content.
func decodeToolResultBlock(b anthropicBlock) (*content.ToolResultMessage, error) {
	inner, err := decodeRequestBlocks(b.Content, true)
	if err != nil {
		return nil, err
	}
	return &content.ToolResultMessage{
		Message:   content.Message{Role: content.RoleUser, Blocks: inner},
		ToolUseID: b.ToolUseID,
		IsError:   b.IsError,
	}, nil
}

// decodeRequestBlocks maps a slice of wire content blocks to neutral blocks.
func decodeRequestBlocks(blocks []anthropicBlock, allowMedia bool) ([]content.Block, error) {
	if len(blocks) == 0 {
		return nil, nil
	}
	out := make([]content.Block, 0, len(blocks))
	for _, b := range blocks {
		db, err := decodeRequestBlock(b, allowMedia)
		if err != nil {
			return nil, err
		}
		out = append(out, db)
	}
	return out, nil
}

// decodeRequestBlock maps one wire content block to a neutral content.Block.
// text/image/document/thinking/redacted_thinking/tool_use are supported;
// anything else (tool_result in the wrong position, or an out-of-scope
// provider-hosted block such as citations) fails closed with a typed error
// rather than being silently dropped, since this is untrusted client input,
// not a tolerant response decode.
//
// allowMedia is false for an assistant message, where an image or document is
// not a shape a client can legitimately replay: both are inputs to the model.
//
// Signature is copied byte-for-byte from the wire onto the decoded
// ThinkingBlock: a same-dialect replay of a prior assistant thinking-plus-
// tool-use turn must preserve Anthropic's required signature exactly. It is
// stamped with this dialect's label as it arrives — a client replaying to this
// gateway is replaying an Anthropic-minted signature — so the outbound
// Anthropic encoder will accept it and no other dialect's encoder will.
func decodeRequestBlock(b anthropicBlock, allowMedia bool) (content.Block, error) {
	if err := checkResponseOnlyMembers(b); err != nil {
		return nil, err
	}
	switch b.Type {
	case blockTypeText:
		if err := checkCitations(b.Citations); err != nil {
			return nil, err
		}
		return &content.TextBlock{Text: b.Text}, nil
	case blockTypeImage:
		if !allowMedia {
			return nil, &ServerDecodeError{Reason: "unsupported_block_placement", Detail: "image block not allowed in an assistant message"}
		}
		return decodeImageSource(b.Source)
	case blockTypeDocument:
		if !allowMedia {
			return nil, &ServerDecodeError{Reason: "unsupported_block_placement", Detail: "document block not allowed in an assistant message"}
		}
		return decodeDocumentSource(b.Source, b.Title)
	case blockTypeThinking:
		return content.NewSignedThinkingBlock(
			b.Thinking, b.Signature, signatureFormatFor(b.Signature), nil, ""), nil
	case blockTypeRedactedThinking:
		// The gateway itself serves this block (encodeResponseBlock and the
		// streaming writeRedactedThinking both emit it), so refusing it here
		// would reject a client replaying our own output. `data` is
		// provider-opaque and travels verbatim in ProviderState, tagged with
		// this dialect's format so it can never be replayed to another one.
		return content.NewThinkingBlock("", "", opaqueRedactedState(b.Data), providerStateFormatAnthropicRedacted), nil
	case blockTypeToolUse:
		if err := checkCaller(b.Caller); err != nil {
			return nil, err
		}
		return &content.ToolUseBlock{ID: b.ID, Name: b.Name, Input: b.Input}, nil
	default:
		return nil, &ServerDecodeError{Reason: "unsupported_block_type", Detail: b.Type}
	}
}

// checkResponseOnlyMembers refuses `citations` on anything but a text block and
// `caller` on anything but a tool_use block. anthropicBlock is one shared DTO
// for the whole tagged union, so accepting the two members at all necessarily
// accepts them everywhere; the request schema does not, declaring citations
// only on RequestTextBlock and caller only on RequestToolUseBlock, with
// additionalProperties = false on every variant. Without this check the
// widening would have quietly traded a strict decode for a lax one.
func checkResponseOnlyMembers(b anthropicBlock) error {
	if len(b.Citations) > 0 && b.Type != blockTypeText {
		return &ServerDecodeError{Reason: "unsupported_citations_placement", Detail: b.Type}
	}
	if len(b.Caller) > 0 && b.Type != blockTypeToolUse {
		return &ServerDecodeError{Reason: "unsupported_caller_placement", Detail: b.Type}
	}
	return nil
}

// checkCitations accepts the two forms a text block's `citations` can take that
// content.TextBlock loses nothing by discarding: absent/null ("no citations")
// and an empty array. A POPULATED citations array is refused rather than
// dropped — the neutral vocabulary has no citation representation, so decoding
// it would report success while discarding the document spans the assistant
// attributed its text to, exactly the silent-degradation failure this codec
// exists to prevent.
func checkCitations(raw json.RawMessage) error {
	switch string(bytes.TrimSpace(raw)) {
	case "", "null", "[]":
		return nil
	default:
		return &ServerDecodeError{Reason: "unsupported_citations", Detail: string(raw)}
	}
}

// checkCaller accepts an absent, null or `direct` caller and refuses the two
// server-tool members. A ServerToolCaller marks a tool_use that Anthropic's own
// hosted code-execution tool issued; content.ToolUseBlock records no issuer, so
// accepting one would turn a server-side call into an indistinguishable
// client-side call — and the gateway would then hand it to the client to
// execute. It fails closed by name instead.
func checkCaller(raw json.RawMessage) error {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		return nil
	}
	var caller struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(trimmed, &caller); err != nil {
		return &ServerDecodeError{Reason: "invalid_tool_caller", Detail: err.Error()}
	}
	if caller.Type != callerTypeDirect {
		return &ServerDecodeError{Reason: "unsupported_tool_caller", Detail: caller.Type}
	}
	return nil
}

func decodeImageSource(src *blockSource) (*content.ImageBlock, error) {
	if src == nil {
		return nil, &ServerDecodeError{Reason: "missing_image_source"}
	}
	switch src.Type {
	case sourceTypeURL:
		return &content.ImageBlock{
			MediaType: content.MediaType(src.MediaType),
			Source:    content.ImageSource{URL: src.URL},
		}, nil
	case sourceTypeBase64:
		data, err := base64.StdEncoding.DecodeString(src.Data)
		if err != nil {
			return nil, &ServerDecodeError{Reason: "invalid_image_data", Detail: err.Error()}
		}
		return &content.ImageBlock{
			MediaType: content.MediaType(src.MediaType),
			Source:    content.ImageSource{Data: data},
		}, nil
	default:
		return nil, &ServerDecodeError{Reason: "unsupported_image_source_type", Detail: src.Type}
	}
}

// decodeDocumentSource maps a wire document `source` to a neutral
// DocumentBlock, inverting encodeDocumentBlock. title carries the document's
// name back, which is the same field the encoder writes it to.
//
// Only the two members the neutral vocabulary can hold are accepted.
// URLPDFSource and ContentBlockSource are perfectly legal Anthropic sources —
// they are simply unrepresentable here, because content.DocumentBlock has
// neither a URL field nor a nested block list — so they are refused BY NAME.
// Decoding either one into an empty DocumentBlock would report success while
// discarding the entire document, which is the failure mode this codec exists
// to prevent.
func decodeDocumentSource(src *blockSource, title string) (*content.DocumentBlock, error) {
	if src == nil {
		return nil, &ServerDecodeError{Reason: "missing_document_source"}
	}
	switch src.Type {
	case sourceTypeBase64:
		if src.MediaType != documentMediaTypePDF {
			return nil, &ServerDecodeError{Reason: "unsupported_document_media_type", Detail: src.MediaType}
		}
		data, err := base64.StdEncoding.DecodeString(src.Data)
		if err != nil {
			return nil, &ServerDecodeError{Reason: "invalid_document_data", Detail: err.Error()}
		}
		return &content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Name: title, Data: data}, nil
	case sourceTypeText:
		if src.MediaType != documentMediaTypeText {
			return nil, &ServerDecodeError{Reason: "unsupported_document_media_type", Detail: src.MediaType}
		}
		return &content.DocumentBlock{MediaType: content.MediaTypeDocumentText, Name: title, Text: src.Data}, nil
	default:
		return nil, &ServerDecodeError{Reason: "unsupported_document_source_type", Detail: src.Type}
	}
}

// --- duplicate JSON object key detection -----------------------------------
//
// The actual scan lives in internal/jsonstrict, shared by every codec/*api
// dialect's server-decode path (extracted once a fourth identical copy of
// this logic appeared — see that package's doc comment). This wrapper only
// translates jsonstrict's dialect-neutral error types to this package's own
// ServerDecodeError/DuplicateKeyError, so callers and existing tests see no
// change in behavior.

// rejectDuplicateObjectKeys reports the first duplicate object member name found
// anywhere in raw (at any nesting depth), or nil if raw has none. A JSON syntax
// error is also propagated as an error: it is not this function's job to
// validate JSON, but it must never silently accept a body it cannot fully walk.
func rejectDuplicateObjectKeys(raw []byte) error {
	switch err := jsonstrict.RejectDuplicateKeys(raw).(type) {
	case nil:
		return nil
	case *jsonstrict.DuplicateKeyError:
		return &DuplicateKeyError{Key: err.Key}
	case *jsonstrict.MalformedError:
		return &ServerDecodeError{Reason: "malformed_body", Detail: err.Detail}
	default:
		return err
	}
}
