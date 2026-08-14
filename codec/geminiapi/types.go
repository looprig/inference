package geminiapi

import (
	"encoding/json"

	"github.com/looprig/inference/internal/usagenorm"
)

// Wire roles for a Gemini `contents` entry. Gemini names the assistant turn
// "model" (not "assistant"), has no "system" role in `contents` (the system
// prompt goes to the top-level systemInstruction), and carries tool results as a
// "user" turn holding a functionResponse part.
const (
	roleUser  = "user"
	roleModel = "model"

	responseMIMETypeJSON   = "application/json"
	functionCallingModeAny = "ANY"
)

// GenerateContentRequest is the Gemini generateContent / streamGenerateContent
// wire request body. The two endpoints share an identical body — streaming is a
// URL + `?alt=sse` concern owned by the transport, not a body field — so there is
// no "stream" flag here (unlike the OpenAI dialect). Exported so provider
// packages can embed it in a typed extension struct without round-tripping
// through map[string]json.RawMessage.
type GenerateContentRequest struct {
	Contents          []geminiContent   `json:"contents"`
	SystemInstruction *geminiContent    `json:"systemInstruction,omitempty"`
	Tools             []geminiTool      `json:"tools,omitempty"`
	ToolConfig        *toolConfig       `json:"toolConfig,omitempty"`
	GenerationConfig  *generationConfig `json:"generationConfig,omitempty"`
}

// geminiContent is one turn in `contents` (or the systemInstruction). Role is
// "user" or "model" for `contents` entries and omitted for systemInstruction.
// The same type serves the response `candidate.content`, so it is shared by both
// encode and decode.
type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

// geminiPart is one part of a turn. Exactly one payload field is set per part on
// encode; on decode the unset fields stay at their zero value. The type is shared
// across request and response: Text/FunctionCall/Thought appear in responses;
// InlineData/FileData/FunctionResponse appear only in requests.
type geminiPart struct {
	Text             string            `json:"text,omitempty"`
	Thought          bool              `json:"thought,omitempty"`
	InlineData       *inlineData       `json:"inlineData,omitempty"`
	FileData         *fileData         `json:"fileData,omitempty"`
	FunctionCall     *functionCall     `json:"functionCall,omitempty"`
	FunctionResponse *functionResponse `json:"functionResponse,omitempty"`

	// ThoughtSignature is Gemini 2.5+'s opaque, provider-private continuation
	// token for a thought (or the function-call part it accompanies). It is a
	// plain base64-ish string on the wire, not structured JSON, so it is
	// carried as a Go string here — the domain-model mapping into
	// content.ThinkingBlock.ProviderState (a json.RawMessage) stores this
	// string's JSON-marshaled form, the same "wrap the opaque wire string as
	// a JSON string value" convention codec/openairesponses uses for its own
	// analogous encrypted_content field (see opaqueStateFromWire/
	// opaqueStateToWire, decode.go/encode.go). A same-dialect round trip must
	// echo it back byte-for-byte on a replayed thought part; this codec never
	// interprets its contents.
	ThoughtSignature string `json:"thoughtSignature,omitempty"`
}

// inlineData carries raw image (or other media) bytes inline as base64. This is
// Gemini's primary multimodal input mechanism.
type inlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"` // base64-encoded bytes
}

// fileData references media by URI. Gemini's fileUri accepts a File API URI, a
// gs:// object, or a YouTube URL — NOT an arbitrary web URL. It is used here only
// as the closest structural mapping for a URL-sourced image (see imagePart).
type fileData struct {
	MimeType string `json:"mimeType,omitempty"`
	FileURI  string `json:"fileUri"`
}

// functionCall is a model-issued tool call. Args is the raw JSON object of the
// arguments (a Struct on the wire). ID is present on models that support parallel
// calls and is used to match the paired functionResponse. Shared by encode (the
// model's prior call, echoed back) and decode (a fresh call from the model).
type functionCall struct {
	ID   string          `json:"id,omitempty"`
	Name string          `json:"name"`
	Args json.RawMessage `json:"args,omitempty"`
}

// functionResponse carries a tool result back to the model. Name MUST match the
// paired functionCall's name (Gemini matches on name, not id); ID echoes the call
// id for parallel-call disambiguation. Response is a JSON object (Struct).
type functionResponse struct {
	ID       string          `json:"id,omitempty"`
	Name     string          `json:"name"`
	Response json.RawMessage `json:"response"`
}

// geminiTool wraps a set of function declarations. Gemini groups declarations
// under a single tool entry (unlike OpenAI's one-tool-per-function shape).
type geminiTool struct {
	FunctionDeclarations []functionDeclaration `json:"functionDeclarations,omitempty"`
}

// functionDeclaration is a callable function exposed to the model. The argument
// schema lives in exactly one of two MUTUALLY EXCLUSIVE fields: Parameters, in
// Gemini's own Schema dialect (an OpenAPI 3.0 subset with an uppercase type
// enum), or ParametersJSONSchema, which takes a standard JSON Schema verbatim.
// declareFunction (encode.go) chooses between them; setting both is a 400.
type functionDeclaration struct {
	Name                 string          `json:"name"`
	Description          string          `json:"description,omitempty"`
	Parameters           json.RawMessage `json:"parameters,omitempty"`
	ParametersJSONSchema json.RawMessage `json:"parametersJsonSchema,omitempty"`
}

// generationConfig maps dialect-neutral Sampling to Gemini's sampling knobs.
// Field names are camelCase per the Gemini wire (topP, maxOutputTokens,
// stopSequences), unlike the OpenAI snake_case dialect.
type generationConfig struct {
	Temperature        *float64        `json:"temperature,omitempty"`
	TopP               *float64        `json:"topP,omitempty"`
	MaxOutputTokens    *int            `json:"maxOutputTokens,omitempty"`
	StopSequences      []string        `json:"stopSequences,omitempty"`
	ThinkingConfig     *thinkingConfig `json:"thinkingConfig,omitempty"`
	ResponseMIMEType   string          `json:"responseMimeType,omitempty"`
	ResponseJSONSchema json.RawMessage `json:"responseJsonSchema,omitempty"`
}

type toolConfig struct {
	FunctionCallingConfig *functionCallingConfig `json:"functionCallingConfig,omitempty"`
}

// functionCallingConfig is `toolConfig.functionCallingConfig`. Gemini has no
// mode meaning "call this one tool": the discovery document defines ANY as
// "constrained to always predicting a function call", limited to
// allowedFunctionNames when that list is set. A forced single tool is
// therefore ANY plus a one-element list, and an unrestricted required choice
// is ANY with the list omitted.
//
// AllowedFunctionNames is a list on the wire even though the neutral
// vocabulary carries one name, so it stays a slice rather than collapsing to
// a string: the extra shape is Gemini's, not ours to erase.
type functionCallingConfig struct {
	Mode                 string   `json:"mode"`
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

// thinkingConfig controls Gemini 2.5+ extended thinking. ThinkingBudget is a
// token budget: -1 requests dynamic (model-decided) thinking; a positive value
// caps it. IncludeThoughts asks the model to return thought summaries as parts
// tagged `"thought": true` (decoded into ThinkingBlock / ThinkingChunk).
type thinkingConfig struct {
	ThinkingBudget  *int `json:"thinkingBudget,omitempty"`
	IncludeThoughts bool `json:"includeThoughts,omitempty"`
}

// GenerateContentResponse is the Gemini generateContent response body and the
// per-chunk streamGenerateContent event (identical shape; a streamed chunk is a
// partial GenerateContentResponse).
type GenerateContentResponse struct {
	Candidates    []candidate    `json:"candidates"`
	UsageMetadata *usageMetadata `json:"usageMetadata"`
	ModelVersion  string         `json:"modelVersion"`

	// PromptFeedback explains a response that carries no candidates. The
	// discovery document states the API "returns no candidates at all only if
	// there was something wrong with the prompt (check prompt_feedback)", so
	// this is the only place such a response says WHY — there is no error
	// envelope and the HTTP status is a success. Decoded so that case becomes
	// a *PromptBlockedError instead of an anonymous failure (decode.go).
	PromptFeedback *promptFeedback `json:"promptFeedback"`

	// Error carries the `{"error":{...}}` envelope (a google.rpc.Status:
	// code/message/status) Google can emit as a stream frame AFTER the request
	// already returned a successful HTTP status. It is modeled here rather than
	// left unknown because such a frame is otherwise a perfectly valid object
	// with no candidates — indistinguishable, to a tolerant decoder, from an
	// uninteresting chunk — so ignoring it let a failed generation finish as a
	// clean, truncated success. It reuses the same geminiErrorBody this codec's
	// server direction writes (server_encode.go).
	Error *geminiErrorBody `json:"error"`
}

// candidate is one generated alternative. The codec reads candidates[0] only.
// A non-empty FinishReason is this dialect's ONLY end-of-generation signal —
// there is no [DONE]-style sentinel — and the v1beta discovery document is
// explicit that the field is "Optional. Output only. The reason why the model
// stopped generating tokens. If empty, the model has not stopped generating
// tokens."
type candidate struct {
	Content      geminiContent `json:"content"`
	FinishReason string        `json:"finishReason"`
	Index        int           `json:"index"`
}

// promptFeedback is the content-filter verdict on the REQUEST's prompt, as
// opposed to a candidate's own finishReason/safetyRatings. BlockReason is set
// only when the prompt was refused; SafetyRatings holds at most one rating per
// harm category, with the deciding one flagged.
type promptFeedback struct {
	BlockReason   string             `json:"blockReason"`
	SafetyRatings []wireSafetyRating `json:"safetyRatings"`
}

// wireSafetyRating is one harm-category rating. It is named apart from the
// exported SafetyRating (errors.go) because the wire values pass through an
// allowlist before they are surfaced.
type wireSafetyRating struct {
	Category    string `json:"category"`
	Probability string `json:"probability"`
	Blocked     bool   `json:"blocked"`
}

// usageMetadata reports token consumption.
//
// ToolUsePromptTokenCount is the discovery document's toolUsePromptTokenCount,
// "Output only. Number of tokens present in tool-use prompt(s)". It is a
// published member of UsageMetadata that appears on grounded and
// code-execution turns, it is billable input, and it is reported apart from
// promptTokenCount rather than inside it — a full response carries a
// promptTokensDetails breakdown that sums to promptTokenCount exactly, with
// toolUsePromptTokensDetails listed separately. Leaving it unmodelled dropped
// those tokens from the caller's accounting entirely; see normalizeInputUsage
// in decode.go.
//
// The remaining members of the published shape (promptTokensDetails,
// candidatesTokensDetails, toolUsePromptTokensDetails, cacheTokensDetails,
// serviceTier) are per-modality breakdowns and labels, not additional token
// buckets, so they are deliberately not modelled.
type usageMetadata struct {
	PromptTokenCount        usagenorm.Count `json:"promptTokenCount"`
	ToolUsePromptTokenCount usagenorm.Count `json:"toolUsePromptTokenCount"`
	CandidatesTokenCount    usagenorm.Count `json:"candidatesTokenCount"`
	CachedContentTokenCount usagenorm.Count `json:"cachedContentTokenCount"`
	ThoughtsTokenCount      usagenorm.Count `json:"thoughtsTokenCount"`
	TotalTokenCount         usagenorm.Count `json:"totalTokenCount"`
}
