package openaiapi

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"net/http"
	"strings"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	codec "github.com/looprig/inference/codec"
	"github.com/looprig/inference/internal/jsonstrict"
	model "github.com/looprig/inference/model"
	"github.com/looprig/inference/wire/jsonbody"
)

// pathChatCompletions is the native Chat Completions route this codec
// recognizes. Streaming and non-streaming requests share this single route
// (distinguished by the body's "stream" flag), unlike geminiapi's two
// distinct routes.
const pathChatCompletions = "/v1/chat/completions"

// wire-value constants used only by the server-decode direction.
const (
	wireRoleSystem    = "system"
	wireRoleUser      = "user"
	wireRoleAssistant = "assistant"
	wireRoleTool      = "tool"

	contentTypeText     = "text"
	contentTypeImageURL = "image_url"

	// toolChoiceAutoValue is the wire "auto" value; "none" has no explicit
	// constant since it is only ever matched by decodeToolChoice's default
	// (rejecting) case.
	toolChoiceAutoValue = "auto"

	dataURIPrefix = "data:"

	// emptyObject is the fallback for a tool call's `arguments` (and, on
	// encode, an absent tool-use input): Chat Completions requires this to
	// be a JSON object, so an empty/absent neutral value becomes "{}".
	emptyObject = "{}"

	// nullLiteral is JSON's null, matched textually when a raw member has to be
	// told apart from a value.
	nullLiteral = "null"
)

// matchChatCompletionsRequest reports whether req is a POST
// /v1/chat/completions request. It does not inspect the body or
// Content-Type; DecodeRequest owns that rejection.
func matchChatCompletionsRequest(req *http.Request) bool {
	return req.Method == http.MethodPost && req.URL.Path == pathChatCompletions
}

// decodeChatCompletionsRequest decodes a matched POST /v1/chat/completions
// request into a codec.DecodedRequest. Request.Model is left at its zero
// value: the harness alias travels only in RequestedModel, and resolving it
// to a real Target is the gateway's job (this codec has no routing table).
func decodeChatCompletionsRequest(req *http.Request) (codec.DecodedRequest, error) {
	body, err := readJSONBody(req)
	if err != nil {
		return codec.DecodedRequest{}, err
	}
	r, requestedModel, streaming, err := decodeChatCompletionsBody(body)
	if err != nil {
		return codec.DecodedRequest{}, err
	}
	return codec.DecodedRequest{Request: r, RequestedModel: requestedModel, Streaming: streaming}, nil
}

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

// wireDecodeRequest is the server-decode-direction wire form of the Chat
// Completions request body. It intentionally does not reuse ChatRequest: the
// decode direction needs a few fields ChatRequest never sends (N, ToolChoice
// as a RawMessage so the object/"none" forms can be classified and rejected)
// and a stricter message/content shape (wireChatMessage) that can tell a
// bare string content apart from an absent one and from an array of parts.
// ParallelToolCalls, User, and Seed are decoded and then intentionally
// dropped: documented benign fields, matching the design's cache_control/
// metadata precedent for the Anthropic and Responses codecs.
type wireDecodeRequest struct {
	Model          string            `json:"model"`
	Messages       []wireChatMessage `json:"messages"`
	Tools          []chatTool        `json:"tools"`
	ResponseFormat *responseFormat   `json:"response_format"`
	ToolChoice     json.RawMessage   `json:"tool_choice"`
	Temperature    *float64          `json:"temperature"`
	TopP           *float64          `json:"top_p"`
	// MaxTokens and MaxCompletionTokens are the deprecated and current
	// spellings of the same limit. Both are recognized (the strict decode
	// would otherwise reject the modern name outright), but a body carrying
	// both is refused rather than silently resolved.
	MaxTokens           *int               `json:"max_tokens"`
	MaxCompletionTokens *int               `json:"max_completion_tokens"`
	Stop                []string           `json:"stop"`
	Stream              bool               `json:"stream"`
	StreamOptions       *chatStreamOptions `json:"stream_options"`
	ReasoningEffort     string             `json:"reasoning_effort"`
	N                   *int               `json:"n"`

	// Documented benign fields: decoded so a real client sending them is not
	// rejected outright by DisallowUnknownFields, then never mapped into the
	// neutral Request.
	ParallelToolCalls *bool  `json:"parallel_tool_calls"`
	User              string `json:"user"`
	Seed              *int   `json:"seed"`
}

// wireChatMessage is the server-decode-direction wire form of one
// `messages` array entry. Content uses wireChatContent so a bare string, an
// array of parts, and an absent/null value are all distinguishable — the
// interface{}-typed chatMessage.Content (types.go, encode direction only)
// cannot make that distinction on decode. Name is a documented-benign,
// decoded-and-ignored field some real clients still attach.
// Refusal is the message-level refusal ChatCompletionRequestAssistantMessage
// declares: a harness replaying a turn OpenAI refused sends it back, and with
// no field for it the strict decode would reject the entire request over one
// member. It is a *string so an absent/null member is distinguishable from an
// explanation-free refusal, exactly as on the client-decode side — see
// chatMessage.Refusal (types.go).
type wireChatMessage struct {
	Role             string          `json:"role"`
	Content          wireChatContent `json:"content"`
	Name             string          `json:"name"`
	ReasoningContent string          `json:"reasoning_content"`
	Refusal          *string         `json:"refusal"`
	ToolCalls        []toolCall      `json:"tool_calls"`
	ToolCallID       string          `json:"tool_call_id"`
}

// contentKind discriminates which wire shape a wireChatContent captured.
type contentKind uint8

const (
	contentKindAbsent contentKind = iota
	contentKindText
	contentKindParts
)

// wireChatContent decodes the Chat Completions `content` field, which is
// either a bare string, an array of content parts, or absent/null (valid for
// an assistant message carrying only tool_calls).
type wireChatContent struct {
	Kind  contentKind
	Text  string
	Parts []chatContentPart
}

func (c *wireChatContent) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		c.Kind = contentKindAbsent
		return nil
	}
	switch trimmed[0] {
	case '"':
		if err := json.Unmarshal(trimmed, &c.Text); err != nil {
			return &ServerDecodeError{Reason: "invalid_message_content", Detail: err.Error()}
		}
		c.Kind = contentKindText
		return nil
	case '[':
		dec := json.NewDecoder(bytes.NewReader(trimmed))
		dec.DisallowUnknownFields()
		var parts []chatContentPart
		if err := dec.Decode(&parts); err != nil {
			return &ServerDecodeError{Reason: "invalid_message_content", Detail: err.Error()}
		}
		c.Parts = parts
		c.Kind = contentKindParts
		return nil
	default:
		return &ServerDecodeError{Reason: "invalid_message_content"}
	}
}

// decodeChatCompletionsBody is the shared semantic decode core behind
// decodeChatCompletionsRequest. It enforces unique object keys and strict
// field recognition (DisallowUnknownFields, so any unsupported wire field
// fails closed rather than being silently dropped — except the documented
// benign fields above), then maps the wire shape into a provider-neutral
// inference.Request. It never resolves Model — only the string alias is
// returned.
//
// Every "system"-role message folds directly into the returned System
// string (joined with a blank line between entries): unlike Anthropic and
// Responses, Chat Completions has no separate top-level system field, so a
// harness-authored system message and this codec's own req.System are the
// same wire position and cannot be told apart positionally. No in-thread
// content.SystemMessage is ever produced by this decoder.
func decodeChatCompletionsBody(raw []byte) (req inference.Request, requestedModel string, streaming bool, err error) {
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
	if wire.N != nil && *wire.N > 1 {
		return inference.Request{}, "", false, &UnsupportedChoiceCountError{N: *wire.N}
	}

	var systemParts []string
	var messages content.AgenticMessages
	for _, m := range wire.Messages {
		switch m.Role {
		case wireRoleSystem:
			text, err := decodeTextOnlyContent(m.Content)
			if err != nil {
				return inference.Request{}, "", false, err
			}
			if text != "" {
				systemParts = append(systemParts, text)
			}
		case wireRoleUser:
			blocks, err := decodeUserContentBlocks(m.Content)
			if err != nil {
				return inference.Request{}, "", false, err
			}
			messages = append(messages, &content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: blocks}})
		case wireRoleAssistant:
			ai, err := decodeAssistantMessage(m)
			if err != nil {
				return inference.Request{}, "", false, err
			}
			messages = append(messages, ai)
		case wireRoleTool:
			if m.ToolCallID == "" {
				return inference.Request{}, "", false, &ServerDecodeError{Reason: "missing_tool_call_id"}
			}
			text, err := decodeTextOnlyContent(m.Content)
			if err != nil {
				return inference.Request{}, "", false, err
			}
			messages = append(messages, &content.ToolResultMessage{
				Message:   content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: text}}},
				ToolUseID: m.ToolCallID,
			})
		default:
			return inference.Request{}, "", false, &ServerDecodeError{Reason: "unsupported_role", Detail: m.Role}
		}
	}

	var tools []inference.Tool
	for _, t := range wire.Tools {
		if t.Type != "function" {
			return inference.Request{}, "", false, &ServerDecodeError{Reason: "unsupported_tool_type", Detail: t.Type}
		}
		tools = append(tools, inference.Tool{Name: t.Function.Name, Description: t.Function.Description, Schema: t.Function.Parameters})
	}

	toolChoice, err := decodeToolChoice(wire.ToolChoice)
	if err != nil {
		return inference.Request{}, "", false, err
	}

	sampling := model.Sampling{}
	switch {
	case wire.MaxTokens != nil && wire.MaxCompletionTokens != nil:
		return inference.Request{}, "", false, &ServerDecodeError{Reason: "conflicting_token_limit", Detail: "max_tokens and max_completion_tokens are mutually exclusive"}
	case wire.MaxCompletionTokens != nil:
		sampling.MaxTokens = wire.MaxCompletionTokens
	case wire.MaxTokens != nil:
		sampling.MaxTokens = wire.MaxTokens
	}
	if wire.Temperature != nil {
		sampling.Temperature = wire.Temperature
	}
	if wire.TopP != nil {
		sampling.TopP = wire.TopP
	}
	if len(wire.Stop) > 0 {
		sampling.Stop = wire.Stop
	}
	if wire.ReasoningEffort != "" {
		effort, err := parseEffort(wire.ReasoningEffort)
		if err != nil {
			return inference.Request{}, "", false, err
		}
		sampling.Effort = effort
	}

	output, err := decodeResponseFormat(wire.ResponseFormat)
	if err != nil {
		return inference.Request{}, "", false, err
	}

	req = inference.Request{
		System:     strings.Join(systemParts, "\n\n"),
		Messages:   messages,
		Tools:      tools,
		Output:     output,
		ToolChoice: toolChoice,
		Override:   &sampling,
	}
	return req, wire.Model, wire.Stream, nil
}

// decodeToolChoice maps the wire tool_choice to the neutral
// inference.ToolChoice. Chat Completions' "none" choice, and the
// allowed-tools and custom-tool members of the object form, are real behavior
// the neutral vocabulary cannot represent, so all of them fail closed rather
// than silently degrading to auto/required.
//
// The object form is matched by an allowlist on `type`: a member OpenAI adds
// later fails closed instead of being mistaken for a function choice.
func decodeToolChoice(raw json.RawMessage) (inference.ToolChoice, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		return inference.ToolAuto(), nil
	}
	var s string
	if err := json.Unmarshal(trimmed, &s); err != nil {
		return decodeToolChoiceObject(trimmed)
	}
	switch s {
	case toolChoiceAutoValue:
		return inference.ToolAuto(), nil
	case toolChoiceRequired:
		return inference.ToolRequired(), nil
	default:
		return inference.ToolAuto(), &ServerDecodeError{Reason: "unsupported_tool_choice", Detail: s}
	}
}

// decodeToolChoiceObject handles the non-string members of
// ChatCompletionToolChoiceOption. Only ChatCompletionNamedToolChoice maps:
// its `function.name` becomes the neutral inference.ToolNamed choice.
func decodeToolChoiceObject(raw json.RawMessage) (inference.ToolChoice, error) {
	var wire namedToolChoice
	if err := json.Unmarshal(raw, &wire); err != nil {
		return inference.ToolAuto(), &ServerDecodeError{Reason: "unsupported_tool_choice", Detail: "object form"}
	}
	if wire.Type != toolChoiceTypeFunction || wire.Function.Name == "" {
		return inference.ToolAuto(), &ServerDecodeError{Reason: "unsupported_tool_choice", Detail: "object form"}
	}
	return inference.ToolNamed(wire.Function.Name), nil
}

// parseEffort maps the wire reasoning_effort value to the neutral
// model.Effort, inverting reasoningEffort (encode.go). Chat Completions has
// no wire value corresponding to model.EffortMax (reasoningEffort clamps it
// down to "high" on encode), so an unrecognized value fails closed.
func parseEffort(wire string) (model.Effort, error) {
	switch wire {
	case "low":
		return model.EffortLow, nil
	case "medium":
		return model.EffortMedium, nil
	case "high":
		return model.EffortHigh, nil
	default:
		return model.EffortNone, &ServerDecodeError{Reason: "unsupported_effort", Detail: wire}
	}
}

// decodeResponseFormat maps the wire `response_format` to a neutral
// OutputSchema. An absent format, or type "text", means plain unstructured
// output.
func decodeResponseFormat(rf *responseFormat) (*inference.OutputSchema, error) {
	if rf == nil {
		return nil, nil
	}
	switch rf.Type {
	case "", contentTypeText:
		return nil, nil
	case responseFormatJSONSchema:
		if rf.JSONSchema == nil {
			return nil, &ServerDecodeError{Reason: "missing_json_schema"}
		}
		return &inference.OutputSchema{Name: rf.JSONSchema.Name, Schema: rf.JSONSchema.Schema, Strict: rf.JSONSchema.Strict}, nil
	default:
		return nil, &ServerDecodeError{Reason: "unsupported_response_format", Detail: rf.Type}
	}
}

// decodeTextOnlyContent extracts the plain text of a message whose content
// must not carry images: a system message (folds into req.System) and a
// tool-result message (Chat Completions' tool content is text-only on the
// wire). An array-of-parts form is accepted, but any non-text part is
// rejected rather than silently dropped, since this is untrusted client
// input, not a tolerant response decode.
func decodeTextOnlyContent(c wireChatContent) (string, error) {
	switch c.Kind {
	case contentKindAbsent:
		return "", nil
	case contentKindText:
		return c.Text, nil
	case contentKindParts:
		var sb strings.Builder
		for _, p := range c.Parts {
			if p.Type != contentTypeText {
				return "", &ServerDecodeError{Reason: "unsupported_content_part_type", Detail: p.Type}
			}
			sb.WriteString(p.Text)
		}
		return sb.String(), nil
	default:
		return "", nil
	}
}

// decodeUserContentBlocks maps a user message's content to neutral blocks:
// text and image_url parts (both remote URL and inline data: URI forms). A
// bare string content becomes a single TextBlock, even if empty.
func decodeUserContentBlocks(c wireChatContent) ([]content.Block, error) {
	switch c.Kind {
	case contentKindAbsent:
		return nil, nil
	case contentKindText:
		return []content.Block{&content.TextBlock{Text: c.Text}}, nil
	case contentKindParts:
		out := make([]content.Block, 0, len(c.Parts))
		for _, p := range c.Parts {
			switch p.Type {
			case contentTypeText:
				out = append(out, &content.TextBlock{Text: p.Text})
			case contentTypeImageURL:
				img, err := decodeImageURLPart(p.ImageURL)
				if err != nil {
					return nil, err
				}
				out = append(out, img)
			case contentPartTypeFile:
				doc, err := decodeFilePart(p.File)
				if err != nil {
					return nil, err
				}
				out = append(out, doc)
			case contentPartTypeInputAudio:
				audio, err := decodeInputAudioPart(p.InputAudio)
				if err != nil {
					return nil, err
				}
				out = append(out, audio)
			default:
				return nil, &ServerDecodeError{Reason: "unsupported_content_part_type", Detail: p.Type}
			}
		}
		return out, nil
	default:
		return nil, nil
	}
}

// decodeAssistantMessage maps a wire assistant message to a neutral
// AIMessage, mirroring buildBlocks' (decode.go) order: reasoning, then text,
// then tool calls — so a same-dialect encode-then-decode round trip
// reconstructs blocks in the same order the response decoder would have
// produced them.
func decodeAssistantMessage(m wireChatMessage) (*content.AIMessage, error) {
	var blocks []content.Block

	if m.ReasoningContent != "" {
		blocks = append(blocks, &content.ThinkingBlock{Thinking: m.ReasoningContent})
	}

	text, err := decodeTextOnlyContent(m.Content)
	if err != nil {
		return nil, err
	}
	if text != "" {
		blocks = append(blocks, &content.TextBlock{Text: text})
	}

	blocks = append(blocks, refusalBlocks(m.Refusal)...)

	for _, tc := range m.ToolCalls {
		if tc.ID == "" {
			return nil, &ServerDecodeError{Reason: "missing_tool_call_id"}
		}
		if tc.Type != "" && tc.Type != "function" {
			return nil, &ServerDecodeError{Reason: "unsupported_tool_call_type", Detail: tc.Type}
		}
		args, err := decodeToolCallArguments(tc.Function.Arguments)
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, &content.ToolUseBlock{ID: tc.ID, Name: tc.Function.Name, Input: args})
	}

	return &content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: blocks}}, nil
}

// toolArgumentsPolicy selects what unwrapToolCallArguments does with an
// `arguments` value it cannot read as an arguments object. The unwrapping
// itself is direction-independent and lives in ONE function: the request
// decoder here and the response decoder in decode.go invert the same encoder,
// and a second copy is how the two drifted apart in the first place (the
// response path used to hand the quoted literal straight to ToolUseBlock.Input,
// which encodeAIMessage then quoted a second time).
type toolArgumentsPolicy int

const (
	// rejectInvalidToolArguments is the REQUEST direction: this codec acting as
	// a server, reading a client's body. That is an untrusted API boundary
	// where a malformed value is a caller defect, so it fails closed with a
	// typed error naming the field rather than propagating garbage inward.
	rejectInvalidToolArguments toolArgumentsPolicy = iota
	// preserveInvalidToolArguments is the RESPONSE direction: a provider's
	// completion. OpenAI's schema says of this very member, "the model does not
	// always generate valid JSON ... Validate the arguments in your code before
	// calling your function" — invalid JSON here is documented MODEL output,
	// not a protocol violation, and the validation site is the tool dispatcher
	// (harness loopruntime checks json.Valid and fails the single call, letting
	// the model correct itself). Rejecting the response would discard a
	// completed generation — its text, its usage and its other, valid tool
	// calls — over one hallucinated argument string, and repairing it to "{}"
	// would be silent repair, showing the model a call it did not make.
	// Preserving the bytes keeps replay byte-exact and leaves the decision
	// where the spec puts it.
	preserveInvalidToolArguments
)

// decodeToolCallArguments is the untrusted-input inverse of
// encodeAIMessage's argument-quoting (encode.go): per toolCallFunction's own
// doc comment, `arguments` is always a JSON-encoded STRING on the wire. A
// bare-object form is tolerated (some non-strict clients send it), matching
// DecodeResponse's tolerance for a provider response. An empty or absent
// value defaults to "{}"; a malformed value fails closed.
func decodeToolCallArguments(raw json.RawMessage) (json.RawMessage, error) {
	return unwrapToolCallArguments(raw, rejectInvalidToolArguments)
}

// unwrapToolCallArguments turns a wire `arguments` member into the arguments
// OBJECT a neutral ToolUseBlock.Input carries. Absent, empty and the empty
// string all become "{}", because Input is an object and "" is not one.
func unwrapToolCallArguments(raw json.RawMessage, policy toolArgumentsPolicy) (json.RawMessage, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return json.RawMessage(emptyObject), nil
	}
	switch trimmed[0] {
	case '"':
		var s string
		if err := json.Unmarshal(trimmed, &s); err != nil {
			return nil, &ServerDecodeError{Reason: "invalid_tool_call_arguments", Detail: err.Error()}
		}
		if s == "" {
			return json.RawMessage(emptyObject), nil
		}
		if !json.Valid([]byte(s)) {
			if policy == preserveInvalidToolArguments {
				return json.RawMessage(s), nil
			}
			return nil, &ServerDecodeError{Reason: "invalid_tool_call_arguments"}
		}
		return json.RawMessage(s), nil
	case '{':
		return trimmed, nil
	default:
		// A JSON null is how some gateways spell "no arguments" on a response;
		// treat it as absent rather than letting the literal reach Input, where
		// it is valid JSON but not an object.
		if policy == preserveInvalidToolArguments && string(trimmed) == nullLiteral {
			return json.RawMessage(emptyObject), nil
		}
		if policy == preserveInvalidToolArguments {
			return trimmed, nil
		}
		return nil, &ServerDecodeError{Reason: "invalid_tool_call_arguments"}
	}
}

// decodeImageURLPart parses an image_url part's `url`: a data: URI decodes
// to inline bytes, anything else is treated as a remote URL reference. This
// is Chat Completions' one content-part shape for both cases — unlike
// Anthropic/Gemini's separate base64 field, the url string itself is
// overloaded to carry either form.
func decodeImageURLPart(u *imageURLPart) (*content.ImageBlock, error) {
	if u == nil || u.URL == "" {
		return nil, &ServerDecodeError{Reason: "missing_image_url"}
	}
	if strings.HasPrefix(u.URL, dataURIPrefix) {
		mediaType, data, err := parseDataURI(u.URL)
		if err != nil {
			return nil, err
		}
		return &content.ImageBlock{MediaType: content.MediaType(mediaType), Source: content.ImageSource{Data: data}}, nil
	}
	return &content.ImageBlock{Source: content.ImageSource{URL: u.URL}}, nil
}

// decodeFilePart parses a `file` content part into a DocumentBlock, inverting
// documentFilePart (encode.go). file_data is spec-typed as a plain string, so
// both wire forms are accepted: a data: URI yields the document's media type
// alongside its bytes, a bare base64 payload yields the bytes with no media
// type. A `file_id` is refused rather than dropped — it names a file in
// OpenAI's Files API and no neutral block can hold such a handle, so accepting
// the part would hand the model an attachment the neutral transcript does not
// contain.
func decodeFilePart(f *filePart) (*content.DocumentBlock, error) {
	if f == nil {
		return nil, &ServerDecodeError{Reason: "missing_file"}
	}
	if f.FileID != "" {
		return nil, &ServerDecodeError{Reason: "unsupported_file_reference", Detail: "file_id"}
	}
	if f.FileData == "" {
		return nil, &ServerDecodeError{Reason: "missing_file_data"}
	}
	if strings.HasPrefix(f.FileData, dataURIPrefix) {
		mediaType, data, err := parseDataURI(f.FileData)
		if err != nil {
			return nil, err
		}
		return &content.DocumentBlock{MediaType: content.MediaType(mediaType), Name: f.Filename, Data: data}, nil
	}
	data, err := base64.StdEncoding.DecodeString(f.FileData)
	if err != nil {
		return nil, &ServerDecodeError{Reason: "invalid_file_data", Detail: err.Error()}
	}
	return &content.DocumentBlock{Name: f.Filename, Data: data}, nil
}

// decodeInputAudioPart parses an `input_audio` content part into an
// AudioBlock, inverting audioInputPart (encode.go). The format is matched
// against the spec's ["wav","mp3"] enum as an allowlist, so an unlisted value
// fails closed rather than producing a block whose media type is a guess.
func decodeInputAudioPart(a *inputAudioPart) (*content.AudioBlock, error) {
	if a == nil {
		return nil, &ServerDecodeError{Reason: "missing_input_audio"}
	}
	var mediaType content.MediaType
	switch a.Format {
	case audioFormatWAV:
		mediaType = content.MediaTypeAudioWAV
	case audioFormatMP3:
		mediaType = content.MediaTypeAudioMPEG
	default:
		return nil, &ServerDecodeError{Reason: "unsupported_audio_format", Detail: a.Format}
	}
	data, err := base64.StdEncoding.DecodeString(a.Data)
	if err != nil {
		return nil, &ServerDecodeError{Reason: "invalid_audio_data", Detail: err.Error()}
	}
	return &content.AudioBlock{MediaType: mediaType, Data: data}, nil
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

// --- duplicate JSON object key detection -----------------------------------
//
// The actual scan lives in internal/jsonstrict, shared by every codec/*api
// dialect's server-decode path (extracted once a fourth identical copy of
// this logic appeared — see that package's doc comment). This wrapper only
// translates jsonstrict's dialect-neutral error types to this package's own
// ServerDecodeError/DuplicateKeyError, so callers and existing tests see no
// change in behavior.

// rejectDuplicateObjectKeys reports the first duplicate object member name
// found anywhere in raw (at any nesting depth), or nil if raw has none. A
// JSON syntax error is also propagated as an error: it is not this
// function's job to validate JSON, but it must never silently accept a body
// it cannot fully walk.
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
