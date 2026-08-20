package geminiapi

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"net/http"
	"strconv"
	"strings"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	codec "github.com/looprig/inference/codec"
	"github.com/looprig/inference/internal/jsonstrict"
	model "github.com/looprig/inference/model"
	"github.com/looprig/inference/wire/jsonbody"
)

// Route constants. Unlike every other dialect in this module, Gemini's model
// name travels in the URL PATH, not the JSON body, and this one codec owns
// TWO distinct routes (non-streaming and streaming) rather than
// distinguishing streaming via a body flag on one shared route.
const (
	modelPathPrefix         = "/v1beta/models/"
	generateSuffix          = ":generateContent"
	streamGenerateSuffix    = ":streamGenerateContent"
	functionCallingModeAuto = "AUTO"

	// emptyObject is the fallback for a tool call's `args` (and, on encode,
	// an absent tool-use input): Gemini requires this to be a JSON object,
	// so an empty/absent neutral value becomes "{}".
	emptyObject = "{}"
)

// matchGenerateContentRequest reports whether req is a POST to either
// /v1beta/models/{model}:generateContent or
// /v1beta/models/{model}:streamGenerateContent. It does not inspect the
// body, Content-Type, or the `alt=sse` query parameter — see parseModelRoute
// for why the path suffix alone is this codec's routing signal.
func matchGenerateContentRequest(req *http.Request) bool {
	if req.Method != http.MethodPost {
		return false
	}
	_, _, ok := parseModelRoute(req.URL.Path)
	return ok
}

// parseModelRoute extracts {model} and the streaming flag from one of this
// codec's two routes. The streaming route's `?alt=sse` query parameter is a
// Google convention, not this codec's routing signal: the path suffix
// (:generateContent vs :streamGenerateContent) already unambiguously
// determines streaming, and Google's own docs describe alt=sse as
// conventional rather than load-bearing, so requiring its presence would
// only add a way to reject an otherwise well-formed streaming request that a
// real client sent with a different or absent alt value. req.URL.Path is
// already percent-decoded by net/http.
func parseModelRoute(path string) (modelName string, streaming bool, ok bool) {
	if !strings.HasPrefix(path, modelPathPrefix) {
		return "", false, false
	}
	rest := strings.TrimPrefix(path, modelPathPrefix)
	switch {
	case strings.HasSuffix(rest, streamGenerateSuffix):
		name := strings.TrimSuffix(rest, streamGenerateSuffix)
		if name == "" {
			return "", false, false
		}
		return name, true, true
	case strings.HasSuffix(rest, generateSuffix):
		name := strings.TrimSuffix(rest, generateSuffix)
		if name == "" {
			return "", false, false
		}
		return name, false, true
	default:
		return "", false, false
	}
}

// decodeGenerateContentRequest decodes a matched request into a
// codec.DecodedRequest. Request.Model is left at its zero value: the
// harness alias travels only in RequestedModel (extracted from the URL
// path here, not the body), and resolving it to a real Target is the
// gateway's job.
func decodeGenerateContentRequest(req *http.Request) (codec.DecodedRequest, error) {
	modelName, streaming, ok := parseModelRoute(req.URL.Path)
	if !ok {
		return codec.DecodedRequest{}, &ServerDecodeError{Reason: "unrecognized_route", Detail: req.URL.Path}
	}
	body, err := readJSONBody(req)
	if err != nil {
		return codec.DecodedRequest{}, err
	}
	r, err := decodeGenerateContentBody(body)
	if err != nil {
		return codec.DecodedRequest{}, err
	}
	return codec.DecodedRequest{Request: r, RequestedModel: modelName, Streaming: streaming}, nil
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

// decodeGenerateContentBody is the shared semantic decode core behind
// decodeGenerateContentRequest. It enforces unique object keys and strict
// field recognition (DisallowUnknownFields — decoding directly into the
// existing GenerateContentRequest wire type, since unlike the other three
// dialects Gemini's request shape is already homogeneous enough that no
// separate server-decode-only wire struct is needed). A multi-candidate
// request (the wire `candidateCount` field) is thereby rejected for free:
// GenerateContentRequest's generationConfig has no CandidateCount field, so
// DisallowUnknownFields fails it closed exactly like any other unmodeled
// field, with no dedicated "n>1"-style check required.
func decodeGenerateContentBody(raw []byte) (inference.Request, error) {
	if dupErr := rejectDuplicateObjectKeys(raw); dupErr != nil {
		return inference.Request{}, dupErr
	}

	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	var wire GenerateContentRequest
	if err := dec.Decode(&wire); err != nil {
		return inference.Request{}, &ServerDecodeError{Reason: "malformed_body", Detail: err.Error()}
	}
	var trailing json.RawMessage
	if err := dec.Decode(&trailing); !errors.Is(err, io.EOF) {
		return inference.Request{}, &ServerDecodeError{Reason: "trailing_data"}
	}

	systemText, err := decodeSystemInstruction(wire.SystemInstruction)
	if err != nil {
		return inference.Request{}, err
	}

	messages, err := decodeContents(wire.Contents)
	if err != nil {
		return inference.Request{}, err
	}

	var tools []inference.Tool
	for _, t := range wire.Tools {
		for _, fd := range t.FunctionDeclarations {
			schema, err := decodeToolSchema(fd)
			if err != nil {
				return inference.Request{}, err
			}
			tools = append(tools, inference.Tool{Name: fd.Name, Description: fd.Description, Schema: schema})
		}
	}

	toolChoice, err := decodeFunctionCallingMode(wire.ToolConfig)
	if err != nil {
		return inference.Request{}, err
	}

	sampling := model.Sampling{}
	var output *inference.OutputSchema
	if wire.GenerationConfig != nil {
		gc := wire.GenerationConfig
		sampling.Temperature = gc.Temperature
		sampling.TopP = gc.TopP
		sampling.MaxTokens = gc.MaxOutputTokens
		if len(gc.StopSequences) > 0 {
			sampling.Stop = gc.StopSequences
		}
		if gc.ThinkingConfig != nil {
			effort, err := effortFromThinkingBudget(gc.ThinkingConfig.ThinkingBudget)
			if err != nil {
				return inference.Request{}, err
			}
			sampling.Effort = effort
		}
		schema, err := decodeResponseSchema(gc.ResponseMIMEType, gc.ResponseJSONSchema)
		if err != nil {
			return inference.Request{}, err
		}
		output = schema
	}

	return inference.Request{
		System:     systemText,
		Messages:   messages,
		Tools:      tools,
		Output:     output,
		ToolChoice: toolChoice,
		Override:   &sampling,
	}, nil
}

// decodeFunctionCallingMode maps the wire toolConfig.functionCallingConfig to
// the neutral inference.ToolChoice, inverting the encoder. ANY with a
// one-element allowedFunctionNames list is the neutral named choice; ANY with
// no list is ToolRequired.
//
// Two restrictions fail closed rather than degrading. An allowlist with more
// than one member restricts the model to a SET, which the single-name neutral
// vocabulary cannot express and must not be widened back to an unrestricted
// ToolRequired. An allowlist on any other mode is likewise unrepresentable
// — and Google documents it as only meaningful for ANY and VALIDATED. Any other
// real Gemini mode (NONE, VALIDATED) has no neutral spelling either.
func decodeFunctionCallingMode(tc *toolConfig) (inference.ToolChoice, error) {
	if tc == nil || tc.FunctionCallingConfig == nil {
		return inference.ToolAuto(), nil
	}
	config := tc.FunctionCallingConfig
	allowed := config.AllowedFunctionNames
	switch config.Mode {
	case "", functionCallingModeAuto:
		if len(allowed) > 0 {
			return inference.ToolAuto(), &ServerDecodeError{Reason: "unsupported_allowed_function_names", Detail: config.Mode}
		}
		return inference.ToolAuto(), nil
	case functionCallingModeAny:
		switch len(allowed) {
		case 0:
			return inference.ToolRequired(), nil
		case 1:
			return inference.ToolNamed(allowed[0]), nil
		default:
			return inference.ToolAuto(), &ServerDecodeError{Reason: "unsupported_allowed_function_names", Detail: config.Mode}
		}
	default:
		return inference.ToolAuto(), &ServerDecodeError{Reason: "unsupported_function_calling_mode", Detail: config.Mode}
	}
}

// effortFromThinkingBudget maps the wire thinkingConfig.thinkingBudget back
// to the neutral model.Effort, inverting thinkingBudget (encode.go)'s four
// canonical values. A nil budget (thinkingConfig present but budget
// omitted) means "no explicit effort was requested" (EffortNone); any other
// value this codec's own encoder would never have produced is rejected
// rather than silently coerced to the nearest tier — a real Gemini client is
// free to request an arbitrary token budget the neutral Effort vocabulary
// cannot represent losslessly, and this is a documented, known limitation of
// the cross-dialect effort mapping, not something this task resolves.
func effortFromThinkingBudget(budget *int) (model.Effort, error) {
	if budget == nil {
		return model.EffortNone, nil
	}
	switch *budget {
	case 1024:
		return model.EffortLow, nil
	case 8192:
		return model.EffortMedium, nil
	case 24576:
		return model.EffortHigh, nil
	default:
		return model.EffortNone, &ServerDecodeError{Reason: "unsupported_thinking_budget", Detail: strconv.Itoa(*budget)}
	}
}

// decodeToolSchema reads a declaration's argument schema from whichever of
// FunctionDeclaration's two mutually exclusive parameter fields carries it.
//
// parametersJsonSchema is already standard JSON Schema and becomes the neutral
// Tool.Schema as-is. `parameters` is Gemini's own dialect, so its uppercase
// Type enum is mapped back to JSON Schema's lowercase names — the exact inverse
// of the projection buildTools performs, which keeps a same-dialect round trip
// faithful and stops a Gemini-only spelling leaking into a neutral schema bound
// for another provider. A declaration setting BOTH fields is rejected: Google
// documents them as mutually exclusive, and guessing which one the client meant
// is exactly the kind of silent choice this codec refuses to make.
func decodeToolSchema(fd functionDeclaration) (json.RawMessage, error) {
	if len(fd.Parameters) > 0 && len(fd.ParametersJSONSchema) > 0 {
		return nil, &ServerDecodeError{Reason: "conflicting_function_parameters", Detail: fd.Name}
	}
	if len(fd.ParametersJSONSchema) > 0 {
		return fd.ParametersJSONSchema, nil
	}
	return jsonSchemaTypesFor(fd.Parameters), nil
}

// jsonSchemaTypesFor rewrites Gemini's uppercase Schema type names to the JSON
// Schema ones, recursing through the keywords that hold subschemas. It is
// deliberately tolerant: anything it does not recognize — including a schema
// that is not an object at all — is returned untouched, because a client's
// schema must reach the neutral model even when this codec cannot improve it.
func jsonSchemaTypesFor(schema json.RawMessage) json.RawMessage {
	normalized, changed := normalizeSchemaTypes(schema, 1)
	if !changed {
		return schema
	}
	return normalized
}

func normalizeSchemaTypes(schema json.RawMessage, depth int) (json.RawMessage, bool) {
	if depth > maxGeminiSchemaDepth {
		return schema, false
	}
	var node map[string]json.RawMessage
	if err := json.Unmarshal(schema, &node); err != nil || node == nil {
		return schema, false
	}
	changed := false
	for keyword, value := range node {
		switch keyword {
		case "type":
			if name, ok := jsonSchemaTypeName(value); ok {
				node[keyword] = name
				changed = true
			}
		case "items":
			if child, childChanged := normalizeSchemaTypes(value, depth+1); childChanged {
				node[keyword] = child
				changed = true
			}
		case "properties":
			if child, childChanged := normalizeSchemaTypeMap(value, depth); childChanged {
				node[keyword] = child
				changed = true
			}
		case "anyOf":
			if child, childChanged := normalizeSchemaTypeList(value, depth); childChanged {
				node[keyword] = child
				changed = true
			}
		}
	}
	if !changed {
		return schema, false
	}
	encoded, err := json.Marshal(node)
	if err != nil {
		return schema, false
	}
	return encoded, true
}

func normalizeSchemaTypeMap(raw json.RawMessage, depth int) (json.RawMessage, bool) {
	var members map[string]json.RawMessage
	if err := json.Unmarshal(raw, &members); err != nil || members == nil {
		return raw, false
	}
	changed := false
	for name, member := range members {
		if child, childChanged := normalizeSchemaTypes(member, depth+1); childChanged {
			members[name] = child
			changed = true
		}
	}
	if !changed {
		return raw, false
	}
	encoded, err := json.Marshal(members)
	if err != nil {
		return raw, false
	}
	return encoded, true
}

func normalizeSchemaTypeList(raw json.RawMessage, depth int) (json.RawMessage, bool) {
	var members []json.RawMessage
	if err := json.Unmarshal(raw, &members); err != nil || members == nil {
		return raw, false
	}
	changed := false
	for i, member := range members {
		if child, childChanged := normalizeSchemaTypes(member, depth+1); childChanged {
			members[i] = child
			changed = true
		}
	}
	if !changed {
		return raw, false
	}
	encoded, err := json.Marshal(members)
	if err != nil {
		return raw, false
	}
	return encoded, true
}

// jsonSchemaTypeNames inverts geminiSchemaTypes (encode.go). The two maps are
// held to being exact inverses by TestGeminiSchemaTypeMapsAreInverses.
var jsonSchemaTypeNames = map[string]json.RawMessage{
	"STRING":  json.RawMessage(`"string"`),
	"NUMBER":  json.RawMessage(`"number"`),
	"INTEGER": json.RawMessage(`"integer"`),
	"BOOLEAN": json.RawMessage(`"boolean"`),
	"ARRAY":   json.RawMessage(`"array"`),
	"OBJECT":  json.RawMessage(`"object"`),
	"NULL":    json.RawMessage(`"null"`),
}

// jsonSchemaTypeName maps "STRING" to "string". A value that is not one of
// Gemini's enum members — a lowercase name a tolerant client sent, say — is
// left alone.
func jsonSchemaTypeName(raw json.RawMessage) (json.RawMessage, bool) {
	var name string
	if err := json.Unmarshal(raw, &name); err != nil {
		return nil, false
	}
	jsonName, ok := jsonSchemaTypeNames[name]
	return jsonName, ok
}

// decodeResponseSchema maps the wire responseMimeType/responseJsonSchema
// pair to a neutral OutputSchema. An absent/empty mime type means plain
// unstructured output.
func decodeResponseSchema(mimeType string, schema json.RawMessage) (*inference.OutputSchema, error) {
	switch mimeType {
	case "":
		return nil, nil
	case responseMIMETypeJSON:
		if len(schema) == 0 {
			return nil, &ServerDecodeError{Reason: "missing_response_json_schema"}
		}
		return &inference.OutputSchema{Schema: schema}, nil
	default:
		return nil, &ServerDecodeError{Reason: "unsupported_response_mime_type", Detail: mimeType}
	}
}

// decodeSystemInstruction extracts the plain text of the top-level
// systemInstruction content. Gemini's systemInstruction is text-only in
// practice; a non-text part (image, function call, …) fails closed rather
// than being silently dropped, since this is untrusted client input.
func decodeSystemInstruction(sys *geminiContent) (string, error) {
	if sys == nil {
		return "", nil
	}
	var sb strings.Builder
	for _, p := range sys.Parts {
		if p.Text == "" || p.Thought || p.InlineData != nil || p.FileData != nil || p.FunctionCall != nil || p.FunctionResponse != nil {
			return "", &ServerDecodeError{Reason: "unsupported_system_instruction_part"}
		}
		sb.WriteString(p.Text)
	}
	return sb.String(), nil
}

// decodeContents maps the wire `contents` array to neutral Conversation
// turns: a "user" entry may expand into several turns (a functionResponse
// part becomes its own ToolResultMessage, matching the sibling dialects'
// identical splitting of a multi-purpose wire turn), and a "model" entry
// becomes one AIMessage.
func decodeContents(contents []geminiContent) ([]content.Conversation, error) {
	var out []content.Conversation
	calls := &serverToolCallLedger{}
	for _, c := range contents {
		switch c.Role {
		case roleUser:
			msgs, err := decodeUserParts(c.Parts, calls)
			if err != nil {
				return nil, err
			}
			out = append(out, msgs...)
		case roleModel:
			ai, err := decodeModelParts(c.Parts, calls)
			if err != nil {
				return nil, err
			}
			out = append(out, ai)
		default:
			return nil, &ServerDecodeError{Reason: "unsupported_role", Detail: c.Role}
		}
	}
	return out, nil
}

// serverToolCallLedger carries a decoded functionCall's identity forward to the
// functionResponse that answers it. Gemini pairs the two by NAME — Required on
// both FunctionCall and FunctionResponse in the v1beta discovery document,
// while `id` is Optional on both — so a native request may legally answer a
// call using only its name. The neutral vocabulary addresses a result solely by
// ToolUseID, so dropping the name (as this decoder previously did, keeping only
// the possibly-absent id) severed the pairing for exactly the id-less parallel
// case. The name is therefore the join key: a name-only response takes the id,
// real or synthetic, of the earliest still-unanswered call of that name.
type serverToolCallLedger struct {
	byName map[string][]string
}

// record queues a decoded call's identity under its function name.
func (l *serverToolCallLedger) record(name, id string) {
	if name == "" || id == "" {
		return
	}
	if l.byName == nil {
		l.byName = make(map[string][]string)
	}
	l.byName[name] = append(l.byName[name], id)
}

// resolve is the ToolUseID for a functionResponse: the wire id when the client
// supplied one, otherwise the queued identity of the oldest unanswered call of
// that name. Either way one queue entry is consumed, so a later response for
// the same name resolves to the next call rather than re-answering the first.
// A response naming no call this thread made yields "" — the same
// "not in this thread" outcome the encoder's toolCallIndex reports.
func (l *serverToolCallLedger) resolve(name, wireID string) string {
	queue := l.byName[name]
	if len(queue) == 0 {
		return wireID
	}
	l.byName[name] = queue[1:]
	if wireID != "" {
		return wireID
	}
	return queue[0]
}

// decodeUserParts splits a "user"-role content's parts into the neutral
// turns they represent: consecutive text/image parts are grouped into one
// UserMessage (preserving order), and each functionResponse part becomes its
// own ToolResultMessage, in the order both kinds appear on the wire —
// mirroring anthropicapi's decodeUserMessage splitting of tool_result blocks
// out of an otherwise-plain user turn. calls carries the preceding model turns'
// functionCall identities forward so a functionResponse that names its call but
// omits `id` (both `id` fields are Optional on the wire, while both `name`
// fields are Required) still produces an addressable ToolUseID.
func decodeUserParts(parts []geminiPart, calls *serverToolCallLedger) ([]content.Conversation, error) {
	var out []content.Conversation
	var pending []content.Block

	flush := func() {
		if len(pending) > 0 {
			out = append(out, &content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: pending}})
			pending = nil
		}
	}

	for _, p := range parts {
		switch {
		case p.FunctionResponse != nil:
			flush()
			text, err := decodeFunctionResponseText(p.FunctionResponse.Response)
			if err != nil {
				return nil, err
			}
			out = append(out, &content.ToolResultMessage{
				Message:   content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: text}}},
				ToolUseID: calls.resolve(p.FunctionResponse.Name, p.FunctionResponse.ID),
			})
		case p.InlineData != nil:
			block, err := decodeInlineData(p.InlineData)
			if err != nil {
				return nil, err
			}
			pending = append(pending, block)
		case p.FileData != nil:
			block, err := decodeFileData(p.FileData)
			if err != nil {
				return nil, err
			}
			pending = append(pending, block)
		case p.Text != "" && !p.Thought:
			pending = append(pending, &content.TextBlock{Text: p.Text})
		default:
			return nil, &ServerDecodeError{Reason: "unsupported_or_empty_part"}
		}
	}
	flush()
	return out, nil
}

// decodeInlineData maps an inlineData part to the neutral block its mimeType
// names. One Blob carries images, audio and documents alike, so the media type
// is the only discriminator there is: decoding every Blob as an ImageBlock (as
// this decoder previously did) silently rewrote an audio or PDF part into an
// image whose media type contradicted its own block type, and lost the round
// trip with the encoder, which puts documents and audio in this very member.
//
// The image default is deliberate and is the decoder's tolerance, not an
// allowlist gap: this is untrusted ingress, and refusing a media type Google
// adds to Blob later would reject a legal client request. Only the two types
// with a dedicated neutral block are routed away from it.
func decodeInlineData(blob *inlineData) (content.Block, error) {
	data, err := base64.StdEncoding.DecodeString(blob.Data)
	if err != nil {
		return nil, &ServerDecodeError{Reason: "invalid_inline_data", Detail: err.Error()}
	}
	mediaType := content.MediaType(blob.MimeType)
	switch {
	case isBlobAudioMIME(mediaType):
		return &content.AudioBlock{MediaType: mediaType, Data: data}, nil
	case isBlobDocumentMIME(mediaType):
		return &content.DocumentBlock{MediaType: mediaType, Data: data}, nil
	default:
		return &content.ImageBlock{MediaType: mediaType, Source: content.ImageSource{Data: data}}, nil
	}
}

// decodeFileData maps a fileData part to an ImageBlock carrying the URI.
//
// A harness-supplied fileUri is accepted as an opaque URL string, same as an
// ImageBlock.Source.URL from any other dialect — validating it is a "real"
// Gemini File API / gs:// / YouTube URI is the outbound Gemini encoder's job
// when THIS decoded request later gets re-encoded to a Gemini target, not this
// decoder's.
//
// ImageBlock is the only neutral media block with a URL source: AudioBlock and
// DocumentBlock hold bytes and nothing else. A fileUri whose mime type says
// audio or document therefore has no faithful destination, and fails closed
// rather than becoming an ImageBlock labelled audio/mpeg. An absent mimeType —
// Optional on FileData — keeps the image reading, which is what every fileData
// this codec itself emits carries.
func decodeFileData(file *fileData) (content.Block, error) {
	mediaType := content.MediaType(file.MimeType)
	if isBlobAudioMIME(mediaType) || isBlobDocumentMIME(mediaType) {
		return nil, &ServerDecodeError{Reason: "unsupported_file_data_mime_type", Detail: file.MimeType}
	}
	return &content.ImageBlock{MediaType: mediaType, Source: content.ImageSource{URL: file.FileURI}}, nil
}

// decodeFunctionResponseText extracts the plain text of a functionResponse's
// `response` object. This codec's own outbound encoder always wraps tool
// result text as {"result": "<text>"} (functionResponsePayload, encode.go);
// the decode direction requires that same shape rather than accepting
// arbitrary JSON, matching the sibling dialects' text-only tool-result
// convention (openaiapi.toolResultText, anthropicapi's tool_result content).
func decodeFunctionResponseText(raw json.RawMessage) (string, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	var payload functionResponsePayload
	if err := dec.Decode(&payload); err != nil {
		return "", &ServerDecodeError{Reason: "unsupported_function_response_shape", Detail: err.Error()}
	}
	return payload.Result, nil
}

// decodeModelParts maps a "model"-role content's parts to a single
// AIMessage's blocks, in wire order: a functionCall part becomes a
// ToolUseBlock; a thought part (text and/or thoughtSignature) becomes a
// ThinkingBlock via content.NewThinkingBlock, so ProviderState is populated
// exactly like the client-decode direction's buildBlocks (decode.go); any
// other non-empty text part becomes a TextBlock. A functionCall part's own
// thoughtSignature (Gemini 2.5+ may attach one directly to the call rather
// than a separate thought part) is preserved positionally in the resulting
// ToolUseBlock's Gemini-tagged ProviderState for exact same-dialect replay. An
// id-less call gets the same synthetic per-turn ordinal the client-decode
// direction assigns (toolCallID), recorded in calls under its name so the
// functionResponse that answers it can be given the same identity.
func decodeModelParts(parts []geminiPart, calls *serverToolCallLedger) (*content.AIMessage, error) {
	var blocks []content.Block
	callOrdinal := 0
	for _, p := range parts {
		switch {
		case p.FunctionCall != nil:
			id := toolCallID(p.FunctionCall.ID, callOrdinal)
			callOrdinal++
			calls.record(p.FunctionCall.Name, id)
			blocks = append(blocks, content.NewToolUseBlock(id, p.FunctionCall.Name,
				argsJSON(p.FunctionCall.Args), providerStateFromThoughtSignature(p.ThoughtSignature), providerStateFormatFor(p.ThoughtSignature)))
		case p.Thought && (p.Text != "" || p.ThoughtSignature != ""):
			blocks = append(blocks, content.NewThinkingBlock(p.Text, "", providerStateFromThoughtSignature(p.ThoughtSignature), providerStateFormatFor(p.ThoughtSignature)))
		case p.Text != "":
			blocks = append(blocks, &content.TextBlock{Text: p.Text})
		default:
			return nil, &ServerDecodeError{Reason: "unsupported_or_empty_part"}
		}
	}
	return &content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: blocks}}, nil
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
