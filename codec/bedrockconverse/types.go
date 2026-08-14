package bedrockconverse

import (
	"encoding/json"

	"github.com/looprig/inference/internal/usagenorm"
)

const (
	roleUser      = "user"
	roleAssistant = "assistant"

	toolResultStatusSuccess = "success"
	toolResultStatusError   = "error"

	imageFormatJPEG = "jpeg"
	imageFormatPNG  = "png"
	imageFormatGIF  = "gif"
	imageFormatWebP = "webp"

	// AudioFormat enum members this codec can name from the shared
	// content.MediaType vocabulary. The enum has fifteen members in all; the
	// remainder (opus, pcm, mka, mkv, mpga, m4a, x-aac) have no MediaType
	// constant to select them, and inventing one would let a caller's media
	// type be silently rewritten into a container it never named.
	audioFormatMP3  = "mp3"
	audioFormatWAV  = "wav"
	audioFormatOGG  = "ogg"
	audioFormatFLAC = "flac"
	audioFormatAAC  = "aac"
	audioFormatMP4  = "mp4"
	audioFormatWebM = "webm"
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
	Messages                     []converseMessage    `json:"messages"`
	System                       []systemContentBlock `json:"system,omitempty"`
	ToolConfig                   *toolConfig          `json:"toolConfig,omitempty"`
	AdditionalModelRequestFields json.RawMessage      `json:"additionalModelRequestFields,omitempty"`
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

// converseContentBlock is a tagged union. Exactly one field must be present.
type converseContentBlock struct {
	Text             *string            `json:"text,omitempty"`
	Image            *imageContent      `json:"image,omitempty"`
	Document         *documentContent   `json:"document,omitempty"`
	Audio            *audioContent      `json:"audio,omitempty"`
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

// audioContent is Converse's AudioBlock: format and source are both @required,
// so neither may be omitempty. The `error` member is response-only diagnostic
// metadata AWS attaches when it cannot process the block; the codec neither
// emits nor reads it.
type audioContent struct {
	Format string      `json:"format"`
	Source audioSource `json:"source"`
}

// audioSource is the AudioSource union: exactly one of bytes (@length min 1)
// or s3Location. S3Location is decode-direction only — content.AudioBlock has
// no URI field, so the encoder can never produce it — and exists here so a
// response carrying one is refused by name rather than by an arity check that
// cannot say which member it saw.
type audioSource struct {
	Bytes      []byte      `json:"bytes,omitempty"`
	S3Location *s3Location `json:"s3Location,omitempty"`
}

type documentContent struct {
	Format string         `json:"format"`
	Name   string         `json:"name"`
	Source documentSource `json:"source"`
}

// documentSource is the DocumentSource union: exactly one of bytes (@length
// min 1), s3Location, text or content. Only bytes and text are reachable from
// content.DocumentBlock; S3Location and Content are modelled for the same
// reason audioSource.S3Location is, so the decoder names the member it cannot
// represent instead of reporting a source with no recognized variant.
type documentSource struct {
	Bytes      []byte                 `json:"bytes,omitempty"`
	Text       *string                `json:"text,omitempty"`
	S3Location *s3Location            `json:"s3Location,omitempty"`
	Content    []documentContentBlock `json:"content,omitempty"`
}

// documentContentBlock is DocumentContentBlock, a union whose only member today
// is text.
type documentContentBlock struct {
	Text *string `json:"text,omitempty"`
}

// s3Location is a storage reference AWS accepts in place of inline bytes. uri
// is @required; bucketOwner is optional and only needed cross-account.
type s3Location struct {
	URI         string `json:"uri"`
	BucketOwner string `json:"bucketOwner,omitempty"`
}

type reasoningContent struct {
	ReasoningText   *reasoningText `json:"reasoningText,omitempty"`
	RedactedContent []byte         `json:"redactedContent,omitempty"`
}

type reasoningText struct {
	Text      *string `json:"text"`
	Signature string  `json:"signature,omitempty"`
}

type toolUseContent struct {
	ToolUseID string          `json:"toolUseId"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
}

type toolResultContent struct {
	ToolUseID string            `json:"toolUseId"`
	Content   []toolResultBlock `json:"content,omitempty"`
	Status    string            `json:"status,omitempty"`
}

type toolResultBlock struct {
	Text     *string          `json:"text,omitempty"`
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
	InputSchema toolInputSchema `json:"inputSchema"`
}

type toolInputSchema struct {
	JSON json.RawMessage `json:"json"`
}

// toolChoice is the Converse ToolChoice union. A Smithy union carries exactly
// one member, and the member key is the discriminator, so every field is a
// pointer with omitempty and exactly one is ever populated.
type toolChoice struct {
	Any  *struct{}           `json:"any,omitempty"`
	Tool *specificToolChoice `json:"tool,omitempty"`
}

// specificToolChoice is SpecificToolChoice, whose `name` is @required and
// targets ToolName (^[a-zA-Z0-9_-]+$, 1..64). Name therefore carries no
// omitempty. The pattern and length are not re-checked here: the neutral
// validator requires the forced name to match a declared tool, and
// validateTools already holds every declared name to toolNameReason.
type specificToolChoice struct {
	Name string `json:"name"`
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
