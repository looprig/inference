package openaiapi

import (
	"fmt"
	"strconv"
)

// StreamEventDecodeError reports malformed JSON inside an otherwise
// successfully framed Chat Completions stream.
type StreamEventDecodeError struct{ Err error }

func (e *StreamEventDecodeError) Error() string {
	return "openaiapi: malformed stream event: " + e.Err.Error()
}

func (e *StreamEventDecodeError) Unwrap() error { return e.Err }

// StreamAPIError reports a well-formed provider error object carried inside an
// otherwise successful Chat Completions stream — the spec's ErrorResponse
// envelope ({"error": {...}}) delivered over HTTP 200, as OpenAI-compatible
// gateways such as OpenRouter document. It is the streaming twin of the
// non-streaming path's failure.APIError, and is deliberately distinct from
// StreamEventDecodeError: the frame parsed fine, the provider is reporting a
// failure. Only the structured code and message are retained, never the raw
// frame. Code prefers the object's `code`, falling back to its `type`.
type StreamAPIError struct {
	Code    string
	Message string
}

func (e *StreamAPIError) Error() string {
	message := "openaiapi: stream error"
	if e.Code != "" {
		message += " (" + e.Code + ")"
	}
	if e.Message != "" {
		message += ": " + e.Message
	}
	return message
}

// UnsupportedBlockError is returned by the encoder when a content block cannot
// be placed on the wire: a concrete type the OpenAI chat completions dialect
// does not model in that position (any non-text block in a text-only tool
// message), or a block whose value falls outside what the position's schema
// accepts (an audio media type outside `input_audio.format`'s two-member enum,
// a file part with no filename to go with its inline data). Block holds the Go
// type name for diagnosis; Reason, when set, names the specific limitation —
// mirroring the sibling bedrockconverse codec's error of the same name.
// Fail-secure per CLAUDE.md and consistent with the sibling anthropicapi and
// geminiapi codecs: an unencodable block is refused, never silently dropped, so
// the model never receives less than the caller sent. Callers may errors.As to
// detect it.
type UnsupportedBlockError struct {
	Block  string
	Reason string
}

func (e *UnsupportedBlockError) Error() string {
	if e.Reason == "" {
		return "openai: unsupported content block type " + e.Block
	}
	return "openai: unsupported content block type " + e.Block + ": " + e.Reason
}

// InvalidToolNameError is returned by the encoder when a tool name cannot
// satisfy the class FunctionObject.name publishes — "Must be a-z, A-Z, 0-9, or
// contain underscores and dashes, with a maximum length of 64".
//
// The constraint is carried in the specification's prose, not in its JSON
// Schema: `name` is typed as a bare string with no `pattern` and no
// `maxLength`, so the conformance gate accepts a name OpenAI's server will
// refuse (measured — see TestTheChatRequestGateHoldsSamplingButNotToolNames).
// This error is the only thing holding the line, which is why it exists at all.
//
// Tool servers hand out names the class excludes: MCP names routinely carry "."
// or "/", and a namespaced name easily runs past 64 characters. Rejecting here
// names the tool and the violated rule; the provider's 400 names neither.
// Shaped after the sibling anthropicapi error of the same name.
type InvalidToolNameError struct {
	Name   string
	Reason string
}

func (e *InvalidToolNameError) Error() string {
	return "openaiapi: invalid tool name " + strconv.Quote(e.Name) + ": " + e.Reason
}

// SamplingRangeError is returned by the encoder when a sampling knob falls
// outside the interval CreateChatCompletionRequest declares for it —
// temperature [0, 2], top_p [0, 1].
//
// Min and Max are carried on the error rather than baked into the message
// because the two fields have DIFFERENT bounds here, and because the bound that
// matters is the destination's: Anthropic and Bedrock cap temperature at 1 and
// OpenAI at 2, so a session moved between providers carries a value that was
// legal at its source into a request where it is not. The shared
// model.Sampling vocabulary is wide enough to hold every dialect's range, so
// the narrowing has to happen in the codec that owns the destination contract.
type SamplingRangeError struct {
	Field string
	Value float64
	Min   float64
	Max   float64
}

func (e *SamplingRangeError) Error() string {
	return fmt.Sprintf("openaiapi: %s must be between %v and %v, got %v", e.Field, e.Min, e.Max, e.Value)
}

// ServerDecodeError reports a native Chat Completions request body this codec
// cannot decode into the provider-neutral vocabulary: malformed shape, a
// missing required field, or a recognized-but-unsupported feature. Reason is
// a short machine-checkable diagnostic code; Detail elaborates for
// logs/messages.
type ServerDecodeError struct {
	Reason string
	Detail string
}

func (e *ServerDecodeError) Error() string {
	msg := "openaiapi: invalid request: " + e.Reason
	if e.Detail != "" {
		msg += ": " + e.Detail
	}
	return msg
}

// DuplicateKeyError reports a request body with a duplicate JSON object
// member name. encoding/json silently takes the last occurrence; this codec
// rejects the request instead so a client cannot smuggle a semantically
// different value past a naive review of the first occurrence.
type DuplicateKeyError struct {
	Key string
}

func (e *DuplicateKeyError) Error() string {
	return "openaiapi: duplicate JSON object key " + e.Key
}

// UnsupportedChoiceCountError reports a request that asked for more than one
// completion choice ("n" > 1). The neutral one-response contract has no
// concept of multiple parallel choices, so this fails closed rather than
// silently returning only the first choice a harness may have expected N of.
type UnsupportedChoiceCountError struct {
	N int
}

func (e *UnsupportedChoiceCountError) Error() string {
	return "openaiapi: unsupported choice count: n=" + strconv.Itoa(e.N) + " (only n=1 is supported)"
}

// StreamTerminatedError is returned by StreamEncoder.WriteChunk, Finish, or
// Fail once the stream has already been terminated by a prior Finish or Fail
// call, per the single-termination-ownership rule in codec.StreamEncoder.
type StreamTerminatedError struct{}

func (e *StreamTerminatedError) Error() string {
	return "openaiapi: stream already terminated"
}

// UnsupportedChunkError is returned when a content.Chunk has a concrete type
// this dialect's stream encoder does not model. content.Chunk is a sealed
// interface, so this only guards against future variants added to the
// vocabulary.
type UnsupportedChunkError struct {
	Chunk string
}

func (e *UnsupportedChunkError) Error() string {
	return "openaiapi: unsupported stream chunk type " + e.Chunk
}
