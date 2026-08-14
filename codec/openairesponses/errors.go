package openairesponses

import "fmt"

// SamplingRangeError is returned by the encoder when a sampling knob falls
// outside the interval CreateResponse declares for it — temperature [0, 2],
// top_p [0, 1], reached through CreateModelResponseProperties ->
// ModelResponseProperties, the same schema Chat Completions inherits.
//
// Min and Max are carried on the error rather than baked into the message
// because the two fields have DIFFERENT bounds here, and because the bound that
// matters is the destination's: Anthropic and Bedrock cap temperature at 1 and
// OpenAI at 2, so a session moved between providers carries a value that was
// legal at its source into a request where it is not. The shared
// model.Sampling vocabulary is wide enough to hold every dialect's range, so
// the narrowing has to happen in the codec that owns the destination contract.
// This mirrors the sibling openaiapi error of the same name.
type SamplingRangeError struct {
	Field string
	Value float64
	Min   float64
	Max   float64
}

func (e *SamplingRangeError) Error() string {
	return fmt.Sprintf("openairesponses: %s must be between %v and %v, got %v", e.Field, e.Min, e.Max, e.Value)
}

// UnsupportedBlockError is returned by an encoder when a content block cannot
// be placed on the wire: a concrete type this dialect does not model in that
// position (audio anywhere, any non-text block in a text-only tool result), or
// a block whose value the position's schema cannot carry (a document with no
// name to attach its inline data to). Block holds the Go type name for
// diagnosis; Reason, when set, names the specific limitation — mirroring the
// sibling bedrockconverse and openaiapi codecs' errors of the same name.
type UnsupportedBlockError struct {
	Block  string
	Reason string
}

func (e *UnsupportedBlockError) Error() string {
	if e.Reason == "" {
		return "openairesponses: unsupported content block type " + e.Block
	}
	return "openairesponses: unsupported content block type " + e.Block + ": " + e.Reason
}

// UnsupportedConversationError is returned by the encoder when a
// conversation turn has a concrete type outside the closed
// content.Conversation union the dialect maps (user / assistant /
// tool-result / system). Conversation holds the Go type name for diagnosis.
type UnsupportedConversationError struct {
	Conversation string
}

func (e *UnsupportedConversationError) Error() string {
	return "openairesponses: unsupported conversation type " + e.Conversation
}

// ServerDecodeError reports a native Responses request body this codec
// cannot decode into the provider-neutral vocabulary: malformed shape, a
// missing required field, or a recognized-but-unsupported feature. Reason is
// a short machine-checkable diagnostic code; Detail elaborates for
// logs/messages.
type ServerDecodeError struct {
	Reason string
	Detail string
}

func (e *ServerDecodeError) Error() string {
	msg := "openairesponses: invalid request: " + e.Reason
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
	return "openairesponses: duplicate JSON object key " + e.Key
}

// StreamTerminatedError is returned by StreamEncoder.WriteChunk, Finish, or
// Fail once the stream has already been terminated by a prior Finish or Fail
// call, per the single-termination-ownership rule in codec.StreamEncoder.
type StreamTerminatedError struct{}

func (e *StreamTerminatedError) Error() string {
	return "openairesponses: stream already terminated"
}

// UnsupportedChunkError is returned when a content.Chunk has a concrete type
// this dialect's stream encoder does not model. content.Chunk is a sealed
// interface, so this only guards against future variants added to the
// vocabulary.
type UnsupportedChunkError struct {
	Chunk string
}

func (e *UnsupportedChunkError) Error() string {
	return "openairesponses: unsupported stream chunk type " + e.Chunk
}

// StreamAPIError reports a native `response.failed` event received after a
// streaming request crossed the successful HTTP-status boundary. It retains
// only the provider's structured error code and message, never the raw
// response frame.
type StreamAPIError struct {
	Code    string
	Message string
}

// StreamEventDecodeError reports malformed JSON inside an otherwise
// successfully framed Responses stream. Unknown well-formed events remain
// forward-compatible skips.
type StreamEventDecodeError struct{ Err error }

func (e *StreamEventDecodeError) Error() string {
	return "openairesponses: malformed stream event: " + e.Err.Error()
}

func (e *StreamEventDecodeError) Unwrap() error { return e.Err }

func (e *StreamAPIError) Error() string {
	message := "openairesponses: stream error"
	if e.Code != "" {
		message += " (" + e.Code + ")"
	}
	if e.Message != "" {
		message += ": " + e.Message
	}
	return message
}
