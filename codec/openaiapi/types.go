package openaiapi

import (
	"encoding/json"

	"github.com/looprig/inference/internal/usagenorm"
)

const (
	responseFormatJSONSchema = "json_schema"
	toolChoiceRequired       = "required"
	toolChoiceTypeFunction   = "function"
	// defaultSchema is the fallback for a tool with no schema.
	// FunctionObject.parameters is spec-typed `object` with no null
	// alternative, so a parameterless tool must still carry an empty object
	// schema. Matches openairesponses' constant of the same name.
	defaultSchema = `{"type":"object"}`

	// contentPartTypeFile and contentPartTypeInputAudio are the `type`
	// discriminators of the two members of
	// ChatCompletionRequestUserMessageContentPart this codec added for
	// content.DocumentBlock and content.AudioBlock. They are used by both the
	// encode and the server-decode direction, unlike the decode-only role and
	// part tags in server_decode.go.
	contentPartTypeFile       = "file"
	contentPartTypeInputAudio = "input_audio"

	// audioFormatWAV and audioFormatMP3 are the complete membership of
	// ChatCompletionRequestMessageContentPartAudio's `input_audio.format`
	// enum. Anything else is not a wire value this dialect has.
	audioFormatWAV = "wav"
	audioFormatMP3 = "mp3"
)

// ChatRequest is the OpenAI chat completions wire request. Exported so
// provider packages can embed it in a typed extension struct (e.g. adding an
// encrypted-response public key) without round-tripping through map[string]json.RawMessage.
type ChatRequest struct {
	Model          string          `json:"model"`
	Messages       []chatMessage   `json:"messages"`
	Tools          []chatTool      `json:"tools,omitempty"`
	ResponseFormat *responseFormat `json:"response_format,omitempty"`
	// ToolChoice is ChatCompletionToolChoiceOption, a oneOf over a mode
	// string and three named-tool objects, so it is carried as raw JSON
	// rather than a string: BuildChatRequest emits either the "required"
	// mode or a ChatCompletionNamedToolChoice object.
	ToolChoice  json.RawMessage `json:"tool_choice,omitempty"`
	Temperature *float64        `json:"temperature,omitempty"`
	TopP        *float64        `json:"top_p,omitempty"`
	// MaxTokens and MaxCompletionTokens are the two mutually exclusive
	// token-limit spellings. OpenAI's spec marks max_tokens deprecated and
	// "not compatible with o-series models"; max_completion_tokens replaces
	// it and is the only form gpt-5 / o-series accept. BuildChatRequest
	// populates exactly one — see its capability gate — because plenty of
	// OpenAI-compatible servers speaking this dialect still know only
	// max_tokens.
	MaxTokens           *int               `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int               `json:"max_completion_tokens,omitempty"`
	Stop                []string           `json:"stop,omitempty"`
	Stream              bool               `json:"stream,omitempty"`
	StreamOptions       *chatStreamOptions `json:"stream_options,omitempty"`

	// o-series reasoning
	ReasoningEffort string `json:"reasoning_effort,omitempty"`
}

type responseFormat struct {
	Type       string      `json:"type"`
	JSONSchema *jsonSchema `json:"json_schema"`
}

type jsonSchema struct {
	Name   string          `json:"name"`
	Strict bool            `json:"strict"`
	Schema json.RawMessage `json:"schema"`
}

type chatStreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// chatMessage is one `messages` entry on encode and one choice's `message` on
// decode.
//
// Refusal is a *string, not a string, because both directions need to tell
// "absent" from "present but empty". ChatCompletionResponseMessage requires the
// member on every assistant message the provider returns, so it is present and
// null on every ordinary reply; a *content.RefusalBlock, by contrast, is
// meaningful whatever its text — a provider may decline with no explanation,
// and the block's presence is the signal (see content.RefusalBlock). Collapsing
// null and "" into one Go value would either invent a block for every ordinary
// reply or discard an explanation-free refusal. omitempty then does the right
// thing on the ENCODE side too: a replayed assistant turn carrying no refusal
// omits the member, while one carrying a RefusalBlock emits it — including the
// empty-string form — because ChatCompletionRequestAssistantMessage declares
// `refusal` on the request side as well.
type chatMessage struct {
	Role             string      `json:"role"`
	Content          interface{} `json:"content"`                     // string or []chatContentPart; interface{} required at JSON serialization boundary
	ReasoningContent string      `json:"reasoning_content,omitempty"` // DeepSeek / o-series
	Refusal          *string     `json:"refusal,omitempty"`
	ToolCalls        []toolCall  `json:"tool_calls,omitempty"`
	ToolCallID       string      `json:"tool_call_id,omitempty"`
}

// chatContentPart is one entry of a user message's `content` array. It folds
// all four members of ChatCompletionRequestUserMessageContentPart into a
// single struct: the members are mutually exclusive and each variant's own
// required member is a pointer or a non-empty string, so omitempty is enough
// to keep one variant's payload from leaking into another's object.
type chatContentPart struct {
	Type       string          `json:"type"`
	Text       string          `json:"text,omitempty"`
	ImageURL   *imageURLPart   `json:"image_url,omitempty"`
	File       *filePart       `json:"file,omitempty"`
	InputAudio *inputAudioPart `json:"input_audio,omitempty"`
}

type imageURLPart struct {
	URL string `json:"url"`
}

// filePart is ChatCompletionRequestMessageContentPartFile's `file` object.
// The schema declares no required member inside it — the file_id form sends
// only file_id, the inline form sends filename plus file_data — so every
// member carries omitempty and exactly one form is ever populated.
type filePart struct {
	Filename string `json:"filename,omitempty"`
	FileData string `json:"file_data,omitempty"`
	FileID   string `json:"file_id,omitempty"`
}

// inputAudioPart is ChatCompletionRequestMessageContentPartAudio's
// `input_audio` object. Its required set is ["data","format"], so neither
// member may carry omitempty: a zero value there would erase a key the
// provider demands.
type inputAudioPart struct {
	Data   string `json:"data"`
	Format string `json:"format"`
}

type toolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"` // always "function"
	Function toolCallFunction `json:"function"`
}

type toolCallFunction struct {
	Name string `json:"name"`
	// Arguments is json.RawMessage to tolerate both wire shapes on DECODE
	// (some servers send a JSON string, others a bare object). On ENCODE it
	// MUST be a JSON-encoded string — see encodeAIMessage, which quotes the
	// raw object before assigning it here. Do not assign a bare object.
	Arguments json.RawMessage `json:"arguments"`
}

type chatTool struct {
	Type     string       `json:"type"` // always "function"
	Function chatFunction `json:"function"`
}

type chatFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

// namedToolChoice is ChatCompletionNamedToolChoice, the forced-tool member of
// the tool_choice union: `{"type":"function","function":{"name":...}}`. The
// spec requires type, function and function.name, so no field carries
// omitempty — a name erased by Go's zero value would leave an object OpenAI
// rejects. Note the extra nesting: the Responses dialect spells the same
// intent with `name` beside `type`.
type namedToolChoice struct {
	Type     string              `json:"type"`
	Function namedToolChoiceFunc `json:"function"`
}

type namedToolChoiceFunc struct {
	Name string `json:"name"`
}

// chatResponse is the OpenAI chat completions wire response. Error models the
// spec's ErrorResponse envelope, which OpenAI-compatible gateways may deliver
// inside an HTTP 200 body rather than on a failing status.
type chatResponse struct {
	ID      string       `json:"id"`
	Model   string       `json:"model"`
	Choices []chatChoice `json:"choices"`
	Usage   *chatUsage   `json:"usage"`
	Error   *chatError   `json:"error"`
}

// chatError is the spec's Error object. Code is a json.RawMessage because it
// is a string in OpenAI's own schema but a numeric HTTP status in several
// compatible gateways (OpenRouter); a typed string field would turn the whole
// event into a spurious decode error. Type is the coarser classification the
// spec also requires; both feed the diagnostic code.
type chatError struct {
	Code    json.RawMessage `json:"code"`
	Type    string          `json:"type"`
	Message string          `json:"message"`
}

// codeString renders the error's `code` as a diagnostic string, preferring the
// string form and falling back to the number form's decimal digits. An absent
// or null code yields "".
func (e *chatError) codeString() string {
	if e == nil || len(e.Code) == 0 || string(e.Code) == "null" {
		return ""
	}
	var s string
	if json.Unmarshal(e.Code, &s) == nil {
		return s
	}
	var n json.Number
	if json.Unmarshal(e.Code, &n) == nil {
		return n.String()
	}
	return ""
}

// httpStatus reports the error's `code` when it is a numeric HTTP status. An
// OpenAI-compatible gateway that reports upstream failures over HTTP 200 puts
// the real status there, so recovering it lets the neutral failure.APIError
// carry a status the retry layer can act on. Anything outside the HTTP status
// range (or a non-numeric code) yields 0.
func (e *chatError) httpStatus() int {
	if e == nil || len(e.Code) == 0 {
		return 0
	}
	var n int
	if json.Unmarshal(e.Code, &n) != nil || n < 100 || n > 599 {
		return 0
	}
	return n
}

type chatChoice struct {
	Message      chatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type chatUsage struct {
	PromptTokens            usagenorm.Count             `json:"prompt_tokens"`
	CompletionTokens        usagenorm.Count             `json:"completion_tokens"`
	PromptTokensDetails     chatPromptTokensDetails     `json:"prompt_tokens_details"`
	CompletionTokensDetails chatCompletionTokensDetails `json:"completion_tokens_details"`
}

type chatPromptTokensDetails struct {
	CachedTokens     usagenorm.Count `json:"cached_tokens"`
	CacheWriteTokens usagenorm.Count `json:"cache_write_tokens"`
}

type chatCompletionTokensDetails struct {
	ReasoningTokens usagenorm.Count `json:"reasoning_tokens"`
}

// sseChunk is one streaming delta, terminal-usage event, or — for
// OpenAI-compatible gateways that report upstream failures inside an HTTP 200
// stream — an error envelope.
type sseChunk struct {
	Model   string      `json:"model"`
	Choices []sseChoice `json:"choices"`
	Usage   *chatUsage  `json:"usage"`
	Error   *chatError  `json:"error"`
}

type sseChoice struct {
	Delta        sseMessageDelta `json:"delta"`
	FinishReason string          `json:"finish_reason"`
}

type sseMessageDelta struct {
	Role             string             `json:"role"`
	Content          string             `json:"content"`
	ReasoningContent string             `json:"reasoning_content"` // DeepSeek / o-series
	Refusal          string             `json:"refusal"`
	ToolCalls        []sseToolCallDelta `json:"tool_calls"`
}

// sseToolCallDelta is one streaming tool-call delta entry. Unlike the
// non-streaming toolCall, it carries a per-call Index and delivers
// Function.Arguments as string fragments that the runner concatenates by Index.
type sseToolCallDelta struct {
	Index    int    `json:"index"`
	ID       string `json:"id"` // first delta only
	Function struct {
		Name      string `json:"name"`      // first delta only
		Arguments string `json:"arguments"` // FRAGMENT — concatenate across deltas
	} `json:"function"`
}
