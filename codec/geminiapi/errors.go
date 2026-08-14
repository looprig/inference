package geminiapi

import (
	"fmt"
	"strconv"

	failure "github.com/looprig/inference/failure"
	usage "github.com/looprig/inference/usage"
)

// EncodeError is a failure while translating an inference.Request into the Gemini wire
// body — an unknown conversation type or a JSON marshal failure. Typed per
// CLAUDE.md so callers can errors.As it to distinguish an encode fault from a
// transport or API error.
type EncodeError struct {
	Reason string
	Err    error
}

func (e *EncodeError) Error() string {
	if e.Err != nil {
		return "gemini: encode: " + e.Reason + ": " + e.Err.Error()
	}
	return "gemini: encode: " + e.Reason
}

func (e *EncodeError) Unwrap() error { return e.Err }

// UnsupportedBlockError is returned by the encoder when a content block cannot
// be put on the wire: its concrete type has no home in the Gemini
// generateContent dialect (e.g. a media block on a model turn), or the block is
// modeled but carries something the dialect refuses — a media type absent from
// Blob's documented list, or no source at all. Block holds the Go type name and
// Reason the specific defect, both for diagnosis. Fail-secure per CLAUDE.md and
// consistent with the sibling bedrockconverse codec's identically shaped error:
// such a block is refused, never silently dropped, so the model never receives
// less than the caller sent. Callers may errors.As to detect it.
type UnsupportedBlockError struct {
	Block  string
	Reason string
}

func (e *UnsupportedBlockError) Error() string {
	message := "gemini: unsupported content block type " + e.Block
	if e.Reason != "" {
		message += ": " + e.Reason
	}
	return message
}

// InvalidToolNameError is returned by the encoder when a function name cannot
// satisfy the class FunctionDeclaration.name publishes — "Must be a-z, A-Z,
// 0-9, or contain underscores, colons, dots, and dashes, with a maximum length
// of 128".
//
// The constraint lives in the discovery document's prose, not in its schema:
// `name` is typed as a bare string, so the conformance gate accepts a name
// Gemini's server will refuse (measured — see
// TestTheGenerateContentGateHoldsNoneOfThis). This error is the only thing
// holding the line. Shaped after the sibling anthropicapi error of the same
// name, whose class is narrower — no "." and no ":" — which is why a tool set
// that encodes here may not encode there.
type InvalidToolNameError struct {
	Name   string
	Reason string
}

func (e *InvalidToolNameError) Error() string {
	return "gemini: invalid function name " + strconv.Quote(e.Name) + ": " + e.Reason
}

// SamplingRangeError is returned by the encoder when a sampling knob falls
// outside the interval this dialect documents — temperature [0, 2], topP
// [0, 1].
//
// Min and Max are carried on the error because the bounds differ per field and,
// more importantly, per provider: Anthropic and Bedrock cap temperature at 1
// where Gemini and OpenAI reach 2. The shared model.Sampling vocabulary spans
// every dialect, so a session moved onto a Gemini model can carry a value that
// was legal at its source, and only the destination codec knows it is not legal
// here.
type SamplingRangeError struct {
	Field string
	Value float64
	Min   float64
	Max   float64
}

func (e *SamplingRangeError) Error() string {
	return fmt.Sprintf("gemini: %s must be between %v and %v, got %v", e.Field, e.Min, e.Max, e.Value)
}

// SafetyRating is one category of Gemini's content classification of a prompt,
// carried on PromptBlockedError. Category and Probability are members of the
// closed enums the discovery document publishes for SafetyRating; a value
// outside them is withheld rather than copied through, so no unbounded provider
// string reaches an error. Blocked reports whether this particular category is
// the one that refused the prompt.
type SafetyRating struct {
	Category    string
	Probability string
	Blocked     bool
}

// PromptBlockedError reports a generateContent response that returned no
// candidates because the PROMPT was refused by Gemini's content filters, with
// the reason the response carried in promptFeedback.
//
// It exists because that response is otherwise indistinguishable from an
// unknown failure: it arrives with a successful HTTP status, no candidates and
// no error envelope, and this codec used to report it as a statusless, codeless
// *failure.APIError — a caller could not tell a policy block from a broken
// response. The discovery document is explicit that the API "returns no
// candidates at all only if there was something wrong with the prompt (check
// prompt_feedback)".
//
// Usage carries the token accounting the response reported. A blocked prompt is
// still a billed prompt, so those counts are retained on the failure rather than
// discarded with it; it is nil when the response carried no usage, or usage this
// codec could not normalize.
type PromptBlockedError struct {
	// BlockReason is a member of PromptFeedback's published blockReason enum,
	// or "" when the provider sent a value this codec does not recognize.
	BlockReason   string
	SafetyRatings []SafetyRating
	Usage         *usage.Usage
}

func (e *PromptBlockedError) Error() string {
	message := "gemini: prompt blocked by the provider's content filter"
	if e.BlockReason != "" {
		message += ": " + e.BlockReason
	}
	return message
}

// Unwrap keeps every existing caller that classifies on *failure.APIError
// working, and upgrades what it sees: the neutral error this codec used to
// return for a candidate-less body carried no code at all, where a blocked
// prompt is precisely a content-policy refusal. promptFeedback is defined as
// "the prompt's feedback related to the content filters", so every blockReason
// it can hold is one — the code does not vary by reason.
//
// The APIError is built on demand so a PromptBlockedError composed by hand
// (in a test, or by a future caller) unwraps the same way one built here does.
func (e *PromptBlockedError) Unwrap() error {
	return failure.NewAPIError(0, contentPolicyViolationCode, "", 0)
}

// contentPolicyViolationCode is the member of failure's closed provider-code
// allowlist that names a content-filter refusal.
const contentPolicyViolationCode = "content_policy_violation"

// DecodeError is a failure while parsing a Gemini response body into a
// provider-neutral Response (a JSON unmarshal failure). The distinct
// "no candidates, no explanation" case returns *failure.APIError instead,
// matching the sibling OpenAI codec so the transport and callers treat every
// dialect uniformly; a candidate-less body that DOES explain itself returns
// *PromptBlockedError, which unwraps to that same APIError.
type DecodeError struct {
	Reason string
	Err    error
}

func (e *DecodeError) Error() string {
	if e.Err != nil {
		return "gemini: decode: " + e.Reason + ": " + e.Err.Error()
	}
	return "gemini: decode: " + e.Reason
}

func (e *DecodeError) Unwrap() error { return e.Err }

// StreamEventDecodeError reports malformed JSON inside an otherwise
// successfully framed streamGenerateContent stream. Mirrors the sibling
// openaiapi/openairesponses codecs: invalid or truncated streaming JSON is an
// error, never a successful response with silently missing content. A
// well-formed chunk this dialect simply has no mapping for stays a tolerant
// skip, so forward compatibility is unaffected.
type StreamEventDecodeError struct{ Err error }

func (e *StreamEventDecodeError) Error() string {
	return "gemini: malformed stream event: " + e.Err.Error()
}

func (e *StreamEventDecodeError) Unwrap() error { return e.Err }

// StreamDecodeError reports a streamGenerateContent response that is framed
// and parseable but structurally wrong — currently only a stream that ends
// without any candidate carrying a finishReason, which per the v1beta
// discovery document ("If empty, the model has not stopped generating tokens")
// means the answer was truncated in flight. It never includes the raw provider
// payload in its diagnostic. Named and shaped after the equivalent type in
// codec/bedrockconverse, the module's strictest streaming dialect.
type StreamDecodeError struct {
	Reason string
	Err    error
}

func (e *StreamDecodeError) Error() string {
	if e.Err != nil {
		return "gemini: stream " + e.Reason + ": " + e.Err.Error()
	}
	return "gemini: stream " + e.Reason
}

func (e *StreamDecodeError) Unwrap() error { return e.Err }

// StreamAPIError reports a Gemini `{"error":{...}}` envelope (a google.rpc.Status
// carrying code/message/status) received after a streaming request already
// crossed the successful HTTP-status boundary. Only the structured fields are
// retained, never the raw frame.
type StreamAPIError struct {
	Code    int
	Status  string
	Message string
}

func (e *StreamAPIError) Error() string {
	message := "gemini: stream error"
	if e.Status != "" {
		message += " (" + e.Status + ")"
	}
	if e.Message != "" {
		message += ": " + e.Message
	}
	return message
}

// ServerDecodeError reports a native generateContent/streamGenerateContent
// request this codec cannot decode into the provider-neutral vocabulary:
// malformed shape, an unrecognized route, or a recognized-but-unsupported
// feature. Reason is a short machine-checkable diagnostic code; Detail
// elaborates for logs/messages.
type ServerDecodeError struct {
	Reason string
	Detail string
}

func (e *ServerDecodeError) Error() string {
	msg := "gemini: invalid request: " + e.Reason
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
	return "gemini: duplicate JSON object key " + e.Key
}

// StreamTerminatedError is returned by StreamEncoder.WriteChunk, Finish, or
// Fail once the stream has already been terminated by a prior Finish or Fail
// call, per the single-termination-ownership rule in codec.StreamEncoder.
type StreamTerminatedError struct{}

func (e *StreamTerminatedError) Error() string {
	return "gemini: stream already terminated"
}

// UnsupportedChunkError is returned when a content.Chunk has a concrete type
// this dialect's stream encoder does not model. content.Chunk is a sealed
// interface, so this only guards against future variants added to the
// vocabulary.
type UnsupportedChunkError struct {
	Chunk string
}

func (e *UnsupportedChunkError) Error() string {
	return "gemini: unsupported stream chunk type " + e.Chunk
}
