package geminiapi

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"regexp"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	model "github.com/looprig/inference/model"
)

const terminalOutputValidationName = "terminal_output"

// toolNamePattern is FunctionDeclaration.name's published class, transcribed
// from the v1beta discovery document: "Must be a-z, A-Z, 0-9, or contain
// underscores, colons, dots, and dashes, with a maximum length of 128."
// ANCHORED, and carrying the cap itself, so a legal substring cannot rescue an
// illegal name.
//
// This is the WIDEST of the four dialects: Anthropic's class excludes "." and
// ":", and Converse's excludes both and caps at 64. A tool set encodable here is
// therefore not necessarily encodable elsewhere, which is exactly why each codec
// owns its own transcription rather than sharing one.
//
// Google's own document narrows the class for FunctionCall.name and
// FunctionResponse.name ("underscores and dashes", no colons or dots), while
// FunctionCallingConfig.allowedFunctionNames says names "should match
// [FunctionDeclaration.name]". The declaration's class is used at all three
// sites deliberately: enforcing the narrower prose would make a legally DECLARED
// tool — every namespaced MCP tool, in practice — impossible to call or to
// answer, which is a constraint Gemini does not actually impose. The cap is the
// same 128 at all three.
var toolNamePattern = regexp.MustCompile(`^[a-zA-Z0-9_.:-]{1,128}$`)

// maxToolNameLength is the same cap spelled out, so the diagnostic can report
// the overshoot in characters rather than only quoting a regular expression.
const maxToolNameLength = 128

// toolNameReason reports why a name is not a legal Gemini function name, or ""
// when it is one. It returns a bare reason so each call site can attach the
// error its surrounding context uses, matching the sibling anthropicapi and
// bedrockconverse codecs.
func toolNameReason(name string) string {
	switch {
	case name == "":
		return "function name must not be empty"
	case len(name) > maxToolNameLength:
		return fmt.Sprintf("function name is %d characters, which exceeds Gemini's %d-character limit", len(name), maxToolNameLength)
	case !toolNamePattern.MatchString(name):
		return "function name must match " + toolNamePattern.String()
	}
	return ""
}

// The sampling bounds this dialect declares. GenerationConfig.temperature is
// explicit — "Values can range from [0.0, 2.0]" — and topP is defined as "the
// maximum cumulative probability of tokens to consider when sampling", so its
// interval is [0, 1] by definition of a cumulative probability rather than by a
// declared numeric range. Both are documentation, not schema: the derived
// request document types each as a plain number, so the conformance gate accepts
// any value at all (measured — see TestTheGenerateContentGateHoldsNoneOfThis).
// The encoder is the only thing holding these.
//
// Note the Vertex flavour of this dialect states the temperature range as
// (0.0, 2.0] — zero EXCLUSIVE. The inclusive v1beta form is used because
// generativelanguage.googleapis.com is this format's canonical source per the
// module's source table, and because temperature 0 is what callers set for
// determinism: refusing it would reject input the canonical document permits.
const (
	minTemperature = 0.0
	maxTemperature = 2.0
	minTopP        = 0.0
	maxTopP        = 1.0
)

// checkSamplingRange holds one sampling knob to its interval. An unset value is
// legal — the field is omitted from generationConfig entirely.
func checkSamplingRange(field string, value *float64, min, max float64) error {
	if value == nil || (*value >= min && *value <= max) {
		return nil
	}
	return &SamplingRangeError{Field: field, Value: *value, Min: min, Max: max}
}

// BuildGenerateContentRequest converts a provider-neutral inference.Request into a
// GenerateContentRequest struct. Exported so provider packages can embed or
// extend the result before marshaling. The effective sampling is Request.Override
// when non-nil, else Model.Sampling — the same precedence every codec honors.
func BuildGenerateContentRequest(req inference.Request) (GenerateContentRequest, error) {
	if err := inference.ValidateRequestFeatures(req); err != nil {
		return GenerateContentRequest{}, err
	}

	sampling := req.Model.Sampling
	if req.Override != nil {
		sampling = *req.Override
	}
	if err := checkSamplingRange("temperature", sampling.Temperature, minTemperature, maxTemperature); err != nil {
		return GenerateContentRequest{}, err
	}
	if err := checkSamplingRange("topP", sampling.TopP, minTopP, maxTopP); err != nil {
		return GenerateContentRequest{}, err
	}

	contents, systemParts, err := buildContents(req.System, req.Messages)
	if err != nil {
		return GenerateContentRequest{}, err
	}
	tools, err := buildTools(req.Tools)
	if err != nil {
		return GenerateContentRequest{}, err
	}

	generatedConfig, err := buildGenerationConfig(sampling, req.Model.Caps)
	if err != nil {
		return GenerateContentRequest{}, err
	}
	out := GenerateContentRequest{
		Contents:         contents,
		Tools:            tools,
		GenerationConfig: generatedConfig,
	}
	switch req.ToolChoice.Mode() {
	case inference.ToolChoiceModeRequired:
		out.ToolConfig = &toolConfig{FunctionCallingConfig: &functionCallingConfig{Mode: functionCallingModeAny}}
	case inference.ToolChoiceModeNamed:
		// allowedFunctionNames is not re-checked either:
		// ValidateRequestFeatures (above) refuses a forced choice naming an
		// undeclared tool, and buildTools has held every declared name to the
		// class, so an illegal forced name is unreachable. Pinned by
		// TestEncodeRequestNamedToolChoiceInheritsTheRule.
		name, _ := req.ToolChoice.Named()
		out.ToolConfig = &toolConfig{FunctionCallingConfig: &functionCallingConfig{
			Mode:                 functionCallingModeAny,
			AllowedFunctionNames: []string{name},
		}}
	}
	if req.Output != nil {
		if out.GenerationConfig == nil {
			out.GenerationConfig = &generationConfig{}
		}
		out.GenerationConfig.ResponseMIMEType = responseMIMETypeJSON
		// generationConfig has both a Schema-typed responseSchema and an
		// untyped responseJsonSchema, and the discovery document marks
		// responseSchema DEPRECATED — so unlike functionDeclarations (below)
		// there is no compatibility argument for the lossy dialect here. The
		// caller's schema goes out exactly as written, including the
		// additionalProperties:false that ValidateOutputSchema requires and
		// that Gemini's Schema dialect cannot spell.
		out.GenerationConfig.ResponseJSONSchema = cloneRawJSON(req.Output.Schema)
	}
	if len(systemParts) > 0 {
		out.SystemInstruction = &geminiContent{Parts: systemParts}
	}
	return out, nil
}

// EncodeRequest converts a provider-neutral inference.Request to a Gemini
// generateContent JSON body. Note there is no stream parameter: Gemini's
// generateContent and streamGenerateContent bodies are byte-for-byte identical —
// the transport selects the endpoint and adds `?alt=sse`, so Codec.EncodeRequest
// ignores its RequestMode.
func EncodeRequest(req inference.Request) ([]byte, error) {
	gr, err := BuildGenerateContentRequest(req)
	if err != nil {
		return nil, err
	}
	return json.Marshal(gr)
}

// buildContents splits the request thread into Gemini's two homes for message
// text: the top-level systemInstruction (Request.System plus any in-thread
// SystemMessage) and the `contents` array (user/model/tool turns). It threads a
// toolCallIndex forward so a ToolResultMessage — which carries only an id, and
// possibly not even that — can name its functionResponse, which Gemini matches
// on name. The thread is ordered, so a tool call is always recorded before its
// result is encoded, and an id-less call's result is the next id-less result to
// arrive.
func buildContents(system string, msgs content.AgenticMessages) ([]geminiContent, []geminiPart, error) {
	// Non-nil so an empty thread marshals as `"contents": []` rather than
	// `"contents": null`. GenerateContentRequest.contents is declared an array;
	// proto3 JSON happens to tolerate null for a repeated field, but a null is
	// not a legal payload under Google's own request schema.
	contents := []geminiContent{}
	var systemParts []geminiPart
	if system != "" {
		systemParts = append(systemParts, geminiPart{Text: system})
	}

	calls := &toolCallIndex{}
	for _, conv := range msgs {
		switch m := conv.(type) {
		case *content.SystemMessage:
			systemParts = append(systemParts, textParts(m.Blocks)...)
		case *content.UserMessage:
			parts, err := encodeUserParts(m.Blocks)
			if err != nil {
				return nil, nil, err
			}
			contents = append(contents, geminiContent{Role: roleUser, Parts: parts})
		case *content.AIMessage:
			parts, err := encodeAIParts(m, calls)
			if err != nil {
				return nil, nil, err
			}
			contents = append(contents, geminiContent{Role: roleModel, Parts: parts})
		case *content.ToolResultMessage:
			parts, err := encodeToolResult(m, calls)
			if err != nil {
				return nil, nil, err
			}
			contents = append(contents, geminiContent{Role: roleUser, Parts: parts})
		default:
			return nil, nil, &EncodeError{Reason: fmt.Sprintf("unknown conversation type %T", conv)}
		}
	}
	return contents, systemParts, nil
}

// textParts extracts the text blocks of a message as Gemini text parts,
// discarding non-text blocks. Used for SystemMessage, which folds into
// systemInstruction where only text is meaningful.
func textParts(blocks []content.Block) []geminiPart {
	var parts []geminiPart
	for _, b := range blocks {
		if t, ok := b.(*content.TextBlock); ok {
			parts = append(parts, geminiPart{Text: t.Text})
		}
	}
	return parts
}

// encodeUserParts maps a user turn's blocks to Gemini parts: text -> text,
// image/audio/document bytes -> inlineData, an image URL -> fileData. Block
// order is preserved. A block type the dialect does not model on a user turn
// (thinking, a nested tool result, …) yields an *UnsupportedBlockError —
// fail-secure, never a silent drop.
func encodeUserParts(blocks []content.Block) ([]geminiPart, error) {
	parts := make([]geminiPart, 0, len(blocks))
	for _, b := range blocks {
		switch b := b.(type) {
		case *content.TextBlock:
			parts = append(parts, geminiPart{Text: b.Text})
		default:
			media, err := mediaParts(b)
			if err != nil {
				return nil, err
			}
			parts = append(parts, media...)
		}
	}
	return parts, nil
}

// mediaParts maps one media block onto the Part members that can carry it. It is
// the single home for that routing, shared by the user turn (encodeUserParts)
// and the media a tool result attaches to its functionResponse
// (encodeToolResult), so the two can never disagree about which member a block
// type belongs in. A block type with no media representation at all yields an
// *UnsupportedBlockError.
//
// A DocumentBlock is the only block that can produce two parts, because it is
// the only one whose neutral form holds two independent sources (bytes and
// extracted text) and Part.data admits exactly one member per part.
func mediaParts(block content.Block) ([]geminiPart, error) {
	switch b := block.(type) {
	case *content.ImageBlock:
		part, err := imagePart(b)
		if err != nil {
			return nil, err
		}
		return []geminiPart{part}, nil
	case *content.AudioBlock:
		part, err := audioPart(b)
		if err != nil {
			return nil, err
		}
		return []geminiPart{part}, nil
	case *content.DocumentBlock:
		return documentParts(b)
	case *content.RefusalBlock:
		return nil, unsupportedRefusal(b)
	default:
		return nil, &UnsupportedBlockError{Block: fmt.Sprintf("%T", block)}
	}
}

// unsupportedRefusal is the single home for this dialect's refusal refusal, so
// every position (user turn, model turn, tool result) names the same
// limitation.
//
// Gemini expresses a decline through the candidate's finishReason (SAFETY,
// PROHIBITED_CONTENT, …) and safetyRatings — response-only metadata. The
// discovery document's Part union is
// text|inlineData|fileData|functionCall|functionResponse|executableCode|
// codeExecutionResult, with no refusal member, so there is nothing to route a
// replayed refusal to. Encoding it as `text` would show the model its own
// decline quoted back as something it said, which is the exact defect
// content.RefusalBlock exists to remove; dropping it loses the fact that the
// turn was declined. Fail closed instead.
func unsupportedRefusal(b *content.RefusalBlock) *UnsupportedBlockError {
	return &UnsupportedBlockError{
		Block:  fmt.Sprintf("%T", b),
		Reason: "Gemini has no refusal Part; a refusal is response-only metadata (candidate finishReason/safetyRatings), so it has no request wire form",
	}
}

// imagePart maps an ImageBlock to a Gemini part. Inline bytes are preferred
// (inlineData is Gemini's robust multimodal path); a URL-only image goes to
// fileData.
//
// Both halves are held to what the destination member accepts, which is what
// audio and documents already were and images were not:
//
//   - inline bytes must carry a media type from Blob's published Images list
//     (isBlobImageMIME), so an SVG or a TIFF fails closed here instead of being
//     refused by the provider with no field named;
//   - a URI must be one fileUri actually accepts (fileURIReason). This used to
//     forward any string at all, so an arbitrary https:// image URL was written
//     into fileUri, which Gemini does not fetch — the request succeeded, the
//     picture never reached the model, and nobody learned why. That is the
//     silent drop of caller intent the module rule forbids, and it is now an
//     error naming the limitation.
func imagePart(img *content.ImageBlock) (geminiPart, error) {
	if len(img.Source.Data) > 0 {
		if !isBlobImageMIME(img.MediaType) {
			return geminiPart{}, &UnsupportedBlockError{
				Block:  fmt.Sprintf("%T", img),
				Reason: "media type " + string(img.MediaType) + " is not an image type Blob accepts",
			}
		}
		return inlineDataPart(img.MediaType, img.Source.Data), nil
	}
	if reason := fileURIReason(img.Source.URL); reason != "" {
		return geminiPart{}, &UnsupportedBlockError{Block: fmt.Sprintf("%T", img), Reason: reason}
	}
	return fileDataPart(img.MediaType, img.Source.URL), nil
}

// audioPart maps an AudioBlock to an inlineData part. AudioBlock has no URL
// source, so fileData — the Files API form — is unreachable from the neutral
// vocabulary for audio; the bytes are the only representation there is, and a
// block without them cannot be sent.
func audioPart(audio *content.AudioBlock) (geminiPart, error) {
	if len(audio.Data) == 0 {
		return geminiPart{}, &UnsupportedBlockError{Block: fmt.Sprintf("%T", audio), Reason: "audio block carries no data"}
	}
	if !isBlobAudioMIME(audio.MediaType) {
		return geminiPart{}, &UnsupportedBlockError{
			Block:  fmt.Sprintf("%T", audio),
			Reason: "media type " + string(audio.MediaType) + " is not an audio type Blob accepts",
		}
	}
	return inlineDataPart(audio.MediaType, audio.Data), nil
}

// documentParts maps a DocumentBlock to the parts that carry it: bytes travel
// as an inlineData Blob, and extracted text as Part.text — the destination
// Blob's own documentation names ("Text should not be sent as raw bytes, use
// the 'text' field"). A document holding both yields both, in that order, since
// dropping either would silently deliver less than the caller sent.
//
// DocumentBlock.Name has no home. Blob carries only mimeType and data, and the
// one member that could hold a source name — Part.partMetadata — is a v1beta
// addition this codec also encodes for Vertex, whose Part is a different
// message; sending it there risks a hard rejection of every named document for
// the sake of metadata. The name is therefore dropped by documented decision,
// not by oversight.
func documentParts(doc *content.DocumentBlock) ([]geminiPart, error) {
	var parts []geminiPart
	if len(doc.Data) > 0 {
		if !isBlobDocumentMIME(doc.MediaType) {
			return nil, &UnsupportedBlockError{
				Block:  fmt.Sprintf("%T", doc),
				Reason: "media type " + string(doc.MediaType) + " is not a document type Blob accepts",
			}
		}
		parts = append(parts, inlineDataPart(doc.MediaType, doc.Data))
	}
	if doc.Text != "" {
		parts = append(parts, geminiPart{Text: doc.Text})
	}
	if len(parts) == 0 {
		return nil, &UnsupportedBlockError{Block: fmt.Sprintf("%T", doc), Reason: "document block carries neither data nor text"}
	}
	return parts, nil
}

// inlineDataPart and fileDataPart are the only constructors of a media part.
// Each sets exactly one member of the Part.data union, which is what the
// discovery document requires ("A `Part` can only contain one of the accepted
// types in `Part.data`") and what the request schema — an ordinary object with
// every member optional — cannot enforce. Routing every media block through
// them makes a two-member part unrepresentable by construction rather than
// merely untested.
func inlineDataPart(mediaType content.MediaType, data []byte) geminiPart {
	return geminiPart{InlineData: &inlineData{
		MimeType: string(mediaType),
		Data:     base64.StdEncoding.EncodeToString(data),
	}}
}

func fileDataPart(mediaType content.MediaType, uri string) geminiPart {
	return geminiPart{FileData: &fileData{MimeType: string(mediaType), FileURI: uri}}
}

// encodeAIParts maps a model turn's blocks to Gemini parts: text -> text,
// tool_use -> functionCall, thinking -> a thought part IF it carries a
// thoughtSignature in ProviderState (added in an earlier task, so the
// domain model now DOES carry Gemini's opaque continuation token — this is
// no longer the silent, undocumented drop it once was). A ThinkingBlock with
// an empty/nil ProviderState (e.g. cross-dialect-sourced, or a same-dialect
// block that predates thinking) is still dropped: real Gemini 2.5+ models
// require a replayed thought part to carry the exact signature the model
// itself issued, and a synthesized or absent signature is not a faithful
// echo — sending a signature-less thought part risks the target rejecting
// the turn outright, so omitting it (as before) remains the safer, more
// defensible choice than fabricating or guessing a value. Each tool call is
// recorded in calls under the identity its functionResponse will use, and the
// id written to the wire is filtered through wireToolCallID so a synthetic
// in-process identity never masquerades as a model-issued one. Any other
// unmodeled block type yields an *UnsupportedBlockError — fail-secure, never
// a silent drop.
func encodeAIParts(m *content.AIMessage, calls *toolCallIndex) ([]geminiPart, error) {
	parts := make([]geminiPart, 0, len(m.Blocks))
	for _, b := range m.Blocks {
		switch b := b.(type) {
		case *content.TextBlock:
			parts = append(parts, geminiPart{Text: b.Text})
		case *content.ToolUseBlock:
			// A replayed functionCall names the function it invoked, so it is
			// bound by the declaration's class — and it is the likelier carrier
			// of a violation, since the name was minted by whichever dialect
			// issued the call.
			if reason := toolNameReason(b.Name); reason != "" {
				return nil, &InvalidToolNameError{Name: b.Name, Reason: reason}
			}
			calls.record(b.ID, b.Name)
			var sig string
			if b.ReplayableAs(providerStateFormatGemini) {
				var err error
				sig, err = providerStateToThoughtSignature(b.ProviderState)
				if err != nil {
					return nil, err
				}
			}
			parts = append(parts, geminiPart{ThoughtSignature: sig, FunctionCall: &functionCall{
				ID:   wireToolCallID(b.ID),
				Name: b.Name,
				Args: argsJSON(b.Input),
			}})
		case *content.ThinkingBlock:
			if !b.ReplayableAs(providerStateFormatGemini) {
				continue
			}
			sig, err := providerStateToThoughtSignature(b.ProviderState)
			if err != nil {
				return nil, err
			}
			parts = append(parts, geminiPart{Thought: true, Text: b.Thinking, ThoughtSignature: sig})
		case *content.RefusalBlock:
			return nil, unsupportedRefusal(b)
		default:
			return nil, &UnsupportedBlockError{Block: fmt.Sprintf("%T", b)}
		}
	}
	return parts, nil
}

// providerStateToThoughtSignature unmarshals a ThinkingBlock.ProviderState
// (which this codec always stores as the JSON-marshaled form of the wire
// thoughtSignature string — see providerStateFromThoughtSignature, decode.go)
// back into the plain string the wire `thoughtSignature` field carries.
func providerStateToThoughtSignature(raw json.RawMessage) (string, error) {
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return "", &EncodeError{Reason: "invalid ThinkingBlock.ProviderState", Err: err}
	}
	return s, nil
}

// functionResponsePayload is the JSON object wrapper for a tool result. Gemini's
// functionResponse.response is a Struct (object), but our tool output is text, so
// it is wrapped under "result" — the key the official Google GenAI SDK uses.
type functionResponsePayload struct {
	Result string `json:"result"`
}

// encodeToolResult maps a ToolResultMessage to the parts of one user turn: the
// functionResponse, followed by any media the result carried. The function name
// comes from the toolCallIndex (empty if the paired call was not in this
// thread), which resolves an id-less result positionally rather than letting
// every id-less call collide on the "" key.
//
// functionResponse.name is NOT re-checked against toolNameReason: every name
// the index can return was recorded from a ToolUseBlock that encodeAIParts
// already held to the class, so the only other value it produces is the
// documented "" of an unpaired result — a pre-existing pairing case, not a
// name-class one. Mirroring the OpenAI codec, IsError
// is NOT emitted — the classic Gemini functionResponse has no structured error
// flag, so the model learns of a failure through the (loop-prefixed) result
// text.
func encodeToolResult(m *content.ToolResultMessage, calls *toolCallIndex) ([]geminiPart, error) {
	text, media, err := toolResultContent(m.Blocks)
	if err != nil {
		return nil, err
	}
	payload, err := json.Marshal(functionResponsePayload{Result: text})
	if err != nil {
		return nil, &EncodeError{Reason: "marshal tool result", Err: err}
	}
	parts := make([]geminiPart, 0, 1+len(media))
	parts = append(parts, geminiPart{FunctionResponse: &functionResponse{
		ID:       wireToolCallID(m.ToolUseID),
		Name:     calls.resolve(m.ToolUseID),
		Response: payload,
	}})
	return append(parts, media...), nil
}

// toolResultContent splits a tool result's blocks into the plain string the
// text-only functionResponse carries and the media parts that must ride
// alongside it.
//
// Gemini's classic functionResponse.response is a JSON object with no media
// member, and the multimodal form the discovery document describes —
// FunctionResponse.parts referenced from the response by a `$ref` to an
// `inline_data.display_name` — cannot be built from that same document, which
// declares no display_name on FunctionResponseBlob. So media travels as
// ordinary inlineData parts of the same user turn, immediately after the
// functionResponse: the model still receives the bytes in the right place in the
// thread, and nothing the caller sent is dropped. This is the path an MCP tool
// returning audio, an image, or an embedded resource takes
// (mcp/pkg/harness/tools.go), and the harness persists those blocks, so a
// refusal here would break every subsequent turn of the session rather than one
// request.
//
// A document's extracted text is tool OUTPUT, not a separate user utterance, so
// it folds into the result string; only bytes become parts. Any block type with
// no representation at all still yields a typed *UnsupportedBlockError.
func toolResultContent(blocks []content.Block) (string, []geminiPart, error) {
	var text string
	var media []geminiPart
	for _, b := range blocks {
		if t, ok := b.(*content.TextBlock); ok {
			text += t.Text
			continue
		}
		parts, err := mediaParts(b)
		if err != nil {
			return "", nil, err
		}
		for _, part := range parts {
			if part.Text != "" {
				text += part.Text
				continue
			}
			media = append(media, part)
		}
	}
	return text, media, nil
}

// buildTools maps the exposed tools into Gemini's single tool entry holding all
// functionDeclarations. Returns nil when there are no tools (so the key is
// omitted). The reserved terminal-output tool is additionally held to the
// portable output subset before it is encoded like any other declaration.
func buildTools(tools []inference.Tool) ([]geminiTool, error) {
	if len(tools) == 0 {
		return nil, nil
	}
	decls := make([]functionDeclaration, 0, len(tools))
	for _, t := range tools {
		if reason := toolNameReason(t.Name); reason != "" {
			return nil, &InvalidToolNameError{Name: t.Name, Reason: reason}
		}
		if t.Name == inference.StructuredOutputToolName {
			terminalOutput := inference.OutputSchema{Name: terminalOutputValidationName, Schema: t.Schema, Strict: true}
			if err := inference.ValidateOutputSchema(terminalOutput); err != nil {
				return nil, err
			}
		}
		decls = append(decls, declareFunction(t))
	}
	return []geminiTool{{FunctionDeclarations: decls}}, nil
}

// declareFunction picks the parameter field a tool's schema belongs in.
//
// FunctionDeclaration has two mutually exclusive homes for it. `parameters` is
// typed as Gemini's own Schema — an OpenAPI 3.0 subset whose `type` is the
// uppercase enum STRING/OBJECT/… and which has no member for
// additionalProperties, $ref, $defs, oneOf, const or prefixItems.
// `parametersJsonSchema` is untyped and takes a standard JSON Schema as-is.
//
// The long-standing `parameters` is preferred when the caller's schema fits
// Gemini's dialect exactly, because it is the field every model and API version
// supports and the only one a Gemini request schema can type-check. Anything
// the dialect cannot express moves the WHOLE schema to parametersJsonSchema
// rather than being dropped: silently losing a tool's required/enum is worse
// than a 400, and a half-projected schema is worse than both.
func declareFunction(t inference.Tool) functionDeclaration {
	decl := functionDeclaration{Name: t.Name, Description: t.Description}
	if len(t.Schema) == 0 {
		return decl
	}
	if projected, ok := projectGeminiSchema(t.Schema); ok {
		decl.Parameters = projected
		return decl
	}
	decl.ParametersJSONSchema = cloneRawJSON(t.Schema)
	return decl
}

// maxGeminiSchemaDepth bounds the projector's recursion over an unvalidated
// caller schema. A schema nested deeper than this is reported as not
// projectable — it still reaches the model, through parametersJsonSchema.
const maxGeminiSchemaDepth = 64

// projectGeminiSchema rewrites a standard JSON Schema into Gemini's Schema
// dialect, returning ok=false when the schema uses anything that dialect cannot
// carry. It is deliberately all-or-nothing: a keyword is either translated
// faithfully or the whole projection is abandoned, so this function can never
// weaken a constraint the caller wrote.
func projectGeminiSchema(schema json.RawMessage) (json.RawMessage, bool) {
	node, ok := projectGeminiNode(schema, 1)
	if !ok {
		return nil, false
	}
	encoded, err := json.Marshal(node)
	if err != nil {
		return nil, false
	}
	return encoded, true
}

// projectGeminiNode projects one schema node. Every keyword Gemini's Schema
// models is listed explicitly; an unknown or unrepresentable one fails the
// projection. Keywords Gemini spells differently from JSON Schema (its
// int64-valued minItems/maxLength/… are strings on the wire) are deliberately
// NOT translated — the JSON-Schema field carries them exactly instead.
func projectGeminiNode(raw json.RawMessage, depth int) (map[string]json.RawMessage, bool) {
	if depth > maxGeminiSchemaDepth {
		return nil, false
	}
	var node map[string]json.RawMessage
	if err := json.Unmarshal(raw, &node); err != nil || node == nil {
		return nil, false
	}
	out := make(map[string]json.RawMessage, len(node))
	for keyword, value := range node {
		switch keyword {
		case "type":
			geminiType, ok := geminiSchemaType(value)
			if !ok {
				return nil, false
			}
			out[keyword] = geminiType
		case "description", "title", "format", "pattern":
			if !jsonIsString(value) {
				return nil, false
			}
			out[keyword] = value
		case "nullable":
			if !jsonIsBool(value) {
				return nil, false
			}
			out[keyword] = value
		case "minimum", "maximum":
			if !jsonIsNumber(value) {
				return nil, false
			}
			out[keyword] = value
		case "enum", "required", "propertyOrdering":
			// Schema.enum is an array of strings, so an integer or boolean
			// enum — which JSON Schema allows — is not representable.
			if !jsonIsStringArray(value) {
				return nil, false
			}
			out[keyword] = value
		case "properties":
			projected, ok := projectGeminiProperties(value, depth)
			if !ok {
				return nil, false
			}
			out[keyword] = projected
		case "items":
			projected, ok := projectGeminiChild(value, depth)
			if !ok {
				return nil, false
			}
			out[keyword] = projected
		case "anyOf":
			projected, ok := projectGeminiMembers(value, depth)
			if !ok {
				return nil, false
			}
			out[keyword] = projected
		default:
			return nil, false
		}
	}
	return out, true
}

func projectGeminiProperties(raw json.RawMessage, depth int) (json.RawMessage, bool) {
	var properties map[string]json.RawMessage
	if err := json.Unmarshal(raw, &properties); err != nil || properties == nil {
		return nil, false
	}
	projected := make(map[string]json.RawMessage, len(properties))
	for name, property := range properties {
		child, ok := projectGeminiChild(property, depth)
		if !ok {
			return nil, false
		}
		projected[name] = child
	}
	encoded, err := json.Marshal(projected)
	if err != nil {
		return nil, false
	}
	return encoded, true
}

func projectGeminiMembers(raw json.RawMessage, depth int) (json.RawMessage, bool) {
	var members []json.RawMessage
	if err := json.Unmarshal(raw, &members); err != nil || members == nil {
		return nil, false
	}
	projected := make([]json.RawMessage, 0, len(members))
	for _, member := range members {
		child, ok := projectGeminiChild(member, depth)
		if !ok {
			return nil, false
		}
		projected = append(projected, child)
	}
	encoded, err := json.Marshal(projected)
	if err != nil {
		return nil, false
	}
	return encoded, true
}

func projectGeminiChild(raw json.RawMessage, depth int) (json.RawMessage, bool) {
	node, ok := projectGeminiNode(raw, depth+1)
	if !ok {
		return nil, false
	}
	encoded, err := json.Marshal(node)
	if err != nil {
		return nil, false
	}
	return encoded, true
}

// geminiSchemaTypes maps the JSON Schema type names to the uppercase members of
// Gemini's Type enum. A union type (`"type": ["string","null"]`) has no member
// and is therefore not projectable.
var geminiSchemaTypes = map[string]json.RawMessage{
	"string":  json.RawMessage(`"STRING"`),
	"number":  json.RawMessage(`"NUMBER"`),
	"integer": json.RawMessage(`"INTEGER"`),
	"boolean": json.RawMessage(`"BOOLEAN"`),
	"array":   json.RawMessage(`"ARRAY"`),
	"object":  json.RawMessage(`"OBJECT"`),
	"null":    json.RawMessage(`"NULL"`),
}

func geminiSchemaType(raw json.RawMessage) (json.RawMessage, bool) {
	var name string
	if err := json.Unmarshal(raw, &name); err != nil {
		return nil, false
	}
	geminiType, ok := geminiSchemaTypes[name]
	return geminiType, ok
}

// The jsonIs* guards below all reject JSON null explicitly: json.Unmarshal
// treats null as a no-op for every Go type, so a `"description": null` would
// otherwise be copied through as a null the Schema dialect does not admit.
func jsonIsString(raw json.RawMessage) bool {
	var value string
	return firstJSONByte(raw) == '"' && json.Unmarshal(raw, &value) == nil
}

func jsonIsBool(raw json.RawMessage) bool {
	var value bool
	first := firstJSONByte(raw)
	return (first == 't' || first == 'f') && json.Unmarshal(raw, &value) == nil
}

func jsonIsNumber(raw json.RawMessage) bool {
	var value json.Number
	return json.Unmarshal(raw, &value) == nil && value != ""
}

func jsonIsStringArray(raw json.RawMessage) bool {
	var values []string
	return json.Unmarshal(raw, &values) == nil && values != nil
}

func firstJSONByte(raw json.RawMessage) byte {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return 0
	}
	return trimmed[0]
}

// cloneRawJSON copies a caller-owned raw schema so the returned request never
// aliases (and can never mutate) the caller's memory.
func cloneRawJSON(raw json.RawMessage) json.RawMessage {
	if raw == nil {
		return nil
	}
	return append(json.RawMessage(nil), raw...)
}

// buildGenerationConfig maps effective sampling to Gemini's generationConfig,
// returning nil when nothing is set so the whole key is omitted. Sampling
// pointers/slices are referenced (read-only) directly, not cloned.
func buildGenerationConfig(s model.Sampling, caps model.Capabilities) (*generationConfig, error) {
	thinking, err := thinkingFor(s.Effort, caps)
	if err != nil {
		return nil, err
	}
	gc := &generationConfig{
		Temperature:     s.Temperature,
		TopP:            s.TopP,
		MaxOutputTokens: s.MaxTokens,
		StopSequences:   s.Stop,
		ThinkingConfig:  thinking,
	}
	if gc.Temperature == nil && gc.TopP == nil && gc.MaxOutputTokens == nil &&
		len(gc.StopSequences) == 0 && gc.ThinkingConfig == nil {
		return nil, nil
	}
	return gc, nil
}

// thinkingFor maps dialect-neutral Effort to a Gemini thinkingConfig. It is
// fail-safe gated on Caps.Thinking: a thinkingConfig sent to a non-thinking model
// is a 400, so a model that does not advertise thinking never receives one.
// EffortNone yields nil — thinking untouched. Unsupported or unknown values
// fail closed rather than silently changing caller intent.
//
// HAZARD, deliberately not fixed here: the budget is derived from Effort alone
// and is independent of Sampling.MaxTokens, so EffortHigh with MaxTokens 1024
// emits thinkingBudget 24576 against a 1024-token output cap. Gemini bills
// thinking tokens as output tokens and draws them from the same allowance, so
// such a request can come back empty or truncated with finishReason
// MAX_TOKENS — the whole allowance spent on reasoning.
//
// It is neither clamped nor rejected, because no Google document declares any
// relationship between thinkingBudget and maxOutputTokens: the discovery
// document types thinkingBudget as a bare integer with no bounds and no
// cross-field rule, and the thinking guides publish only PER-MODEL budget
// ranges. Clamping would be worse than the hazard — those per-model ranges have
// non-zero FLOORS (128 on 2.5 Pro, 512 on 2.5 Flash-Lite), so clamping a budget
// down to a small cap would turn an accepted request into a rejected one, and
// would silently rewrite the caller's declared effort. Failing closed would
// invent a bound the provider never published and refuse requests that succeed
// today. The behaviour is pinned, with the full argument, by
// TestEncodeRequestKeepsTheEffortBudgetUnderASmallOutputCap.
func thinkingFor(e model.Effort, caps model.Capabilities) (*thinkingConfig, error) {
	if !caps.Thinking {
		return nil, nil
	}
	if e == model.EffortNone {
		return nil, nil
	}
	budget, ok := thinkingBudget(e)
	if !ok {
		return nil, &UnsupportedEffortError{Effort: string(e)}
	}
	return &thinkingConfig{ThinkingBudget: &budget, IncludeThoughts: true}, nil
}

// thinkingBudget maps the supported neutral efforts to conservative fixed token
// budgets. ok=false means the effort has no provider-safe mapping; the caller
// turns that into UnsupportedEffortError. Minimal and low intentionally share
// Gemini's smallest portable budget.
func thinkingBudget(e model.Effort) (int, bool) {
	switch e {
	case model.EffortMinimal:
		return 1024, true
	case model.EffortLow:
		return 1024, true
	case model.EffortMedium:
		return 8192, true
	case model.EffortHigh:
		return 24576, true
	default: // EffortNone or unknown → omit
		return 0, false
	}
}

// argsJSON normalizes a tool-call argument payload for the wire: an empty payload
// becomes an empty JSON object so functionCall.args is always a valid Struct.
func argsJSON(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return json.RawMessage("{}")
	}
	return raw
}
