package bedrockconverse

import (
	"encoding/json"

	"github.com/looprig/inference/internal/usagenorm"
)

const (
	roleUser      = "user"
	roleAssistant = "assistant"

	contentTypeText             = "text"
	contentTypeImage            = "image"
	contentTypeDocument         = "document"
	contentTypeReasoningContent = "reasoningContent"
	contentTypeToolUse          = "toolUse"
	contentTypeToolResult       = "toolResult"

	toolResultStatusSuccess = "success"
	toolResultStatusError   = "error"

	imageFormatJPEG = "jpeg"
	imageFormatPNG  = "png"
	imageFormatGIF  = "gif"
	imageFormatWebP = "webp"
)

type converseRequest struct {
	InferenceConfig                   *inferenceConfig     `json:"inferenceConfig,omitempty"`
	Messages                          []converseMessage    `json:"messages"`
	OutputConfig                      *outputConfig        `json:"outputConfig,omitempty"`
	System                            []systemContentBlock `json:"system,omitempty"`
	ToolConfig                        *toolConfig          `json:"toolConfig,omitempty"`
	AdditionalModelRequestFields      json.RawMessage      `json:"additionalModelRequestFields,omitempty"`
	AdditionalModelResponseFieldPaths []string             `json:"additionalModelResponseFieldPaths,omitempty"`
}

type converseCountTokensRequest struct {
	Messages   []converseMessage    `json:"messages"`
	System     []systemContentBlock `json:"system,omitempty"`
	ToolConfig *toolConfig          `json:"toolConfig,omitempty"`
}

type inferenceConfig struct {
	MaxTokens     *int     `json:"maxTokens,omitempty"`
	Temperature   *float64 `json:"temperature,omitempty"`
	TopP          *float64 `json:"topP,omitempty"`
	StopSequences []string `json:"stopSequences,omitempty"`
}

type converseMessage struct {
	Role    string                 `json:"role"`
	Content []converseContentBlock `json:"content"`
}

type systemContentBlock struct {
	Text string `json:"text,omitempty"`
}

// converseContentBlock is a tagged union. Exactly one pointer/string field is
// populated by the encoder for each block.
type converseContentBlock struct {
	Text             string             `json:"text,omitempty"`
	Image            *imageContent      `json:"image,omitempty"`
	Document         *documentContent   `json:"document,omitempty"`
	ReasoningContent *reasoningContent  `json:"reasoningContent,omitempty"`
	ToolUse          *toolUseContent    `json:"toolUse,omitempty"`
	ToolResult       *toolResultContent `json:"toolResult,omitempty"`
}

type imageContent struct {
	Format string      `json:"format"`
	Source imageSource `json:"source"`
}

type imageSource struct {
	Bytes []byte `json:"bytes,omitempty"`
}

type documentContent struct {
	Format string         `json:"format"`
	Name   string         `json:"name"`
	Source documentSource `json:"source"`
}

type documentSource struct {
	Bytes []byte `json:"bytes,omitempty"`
	Text  string `json:"text,omitempty"`
}

type reasoningContent struct {
	ReasoningText *reasoningText `json:"reasoningText,omitempty"`
}

type reasoningText struct {
	Text      string `json:"text"`
	Signature string `json:"signature,omitempty"`
}

type toolUseContent struct {
	ToolUseID string          `json:"toolUseId"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
}

type toolResultContent struct {
	ToolUseID string            `json:"toolUseId"`
	Content   []toolResultBlock `json:"content,omitempty"`
	Status    string            `json:"status"`
}

type toolResultBlock struct {
	Text     string           `json:"text,omitempty"`
	Image    *imageContent    `json:"image,omitempty"`
	Document *documentContent `json:"document,omitempty"`
}

type toolConfig struct {
	Tools      []toolDefinition `json:"tools,omitempty"`
	ToolChoice *toolChoice      `json:"toolChoice,omitempty"`
}

type toolDefinition struct {
	ToolSpec toolSpec `json:"toolSpec"`
}

type toolSpec struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"inputSchema"`
}

type toolChoice struct {
	Any *struct{} `json:"any,omitempty"`
}

type outputConfig struct {
	TextFormat *textFormat `json:"textFormat,omitempty"`
}

type textFormat struct {
	Type      string         `json:"type"`
	Structure *textStructure `json:"structure,omitempty"`
}

type textStructure struct {
	JSONSchema jsonSchema `json:"jsonSchema"`
}

type jsonSchema struct {
	Schema      string `json:"schema"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
}

// converseResponse is the non-streaming Converse response. The stream decoder
// uses the same content and usage DTOs for its terminal events.
type converseResponse struct {
	Output                        *responseOutput `json:"output"`
	StopReason                    string          `json:"stopReason"`
	Usage                         *responseUsage  `json:"usage"`
	Metrics                       json.RawMessage `json:"metrics,omitempty"`
	AdditionalModelResponseFields json.RawMessage `json:"additionalModelResponseFields,omitempty"`
}

type responseOutput struct {
	Message *converseMessage `json:"message"`
}

type responseUsage struct {
	InputTokens           usagenorm.Count `json:"inputTokens"`
	OutputTokens          usagenorm.Count `json:"outputTokens"`
	CacheReadInputTokens  usagenorm.Count `json:"cacheReadInputTokens"`
	CacheWriteInputTokens usagenorm.Count `json:"cacheWriteInputTokens"`
}
