package bedrockconverse

import (
	"fmt"
	"strconv"
)

// UnsupportedEffortError reports a neutral reasoning effort that the generic
// Converse request has no model-independent field to represent.
type UnsupportedEffortError struct {
	Effort string
}

func (e *UnsupportedEffortError) Error() string {
	return "bedrockconverse: unsupported reasoning effort " + strconv.Quote(e.Effort)
}

// UnsupportedBlockError reports a content block that Bedrock Converse cannot
// represent in the shared request vocabulary.
type UnsupportedBlockError struct {
	Block  string
	Reason string
}

func (e *UnsupportedBlockError) Error() string {
	if e.Reason == "" {
		return "bedrockconverse: unsupported content block " + e.Block
	}
	return "bedrockconverse: unsupported content block " + e.Block + ": " + e.Reason
}

// UnsupportedConversationError reports an unexpected conversation variant.
type UnsupportedConversationError struct {
	Conversation string
}

// ConversationCollisionError reports adjacent neutral turns that Bedrock
// cannot represent without mixing ordinary user content and toolResult blocks
// in one Converse message.
type ConversationCollisionError struct {
	Reason string
}

func (e *ConversationCollisionError) Error() string {
	return "bedrockconverse: conversation collision: " + e.Reason
}

func (e *UnsupportedConversationError) Error() string {
	return "bedrockconverse: unsupported conversation " + e.Conversation
}

// ToolSchemaError reports a missing, malformed, or otherwise invalid tool
// input schema before it reaches the Bedrock endpoint.
type ToolSchemaError struct {
	Tool   string
	Reason string
}

func (e *ToolSchemaError) Error() string {
	if e.Tool == "" {
		return "bedrockconverse: invalid tool schema: " + e.Reason
	}
	return fmt.Sprintf("bedrockconverse: invalid tool schema for %q: %s", e.Tool, e.Reason)
}

// ToolInputError reports malformed arguments on a replayed tool-use block.
type ToolInputError struct {
	Tool   string
	Reason string
}

func (e *ToolInputError) Error() string {
	if e.Tool == "" {
		return "bedrockconverse: invalid tool input: " + e.Reason
	}
	return fmt.Sprintf("bedrockconverse: invalid tool input for %q: %s", e.Tool, e.Reason)
}

// EncodeError wraps a request-construction failure that is not a feature or
// content-type validation error. It intentionally carries bounded diagnostics
// rather than raw provider payloads.
type EncodeError struct {
	Reason string
	Err    error
}

func (e *EncodeError) Error() string {
	if e.Err != nil {
		return "bedrockconverse: " + e.Reason + ": " + e.Err.Error()
	}
	return "bedrockconverse: " + e.Reason
}

func (e *EncodeError) Unwrap() error { return e.Err }

// DecodeError is used by the response decoder for malformed successful
// Bedrock payloads. It is defined with the request codec so callers can use one
// stable typed error family across the codec's directions.
type DecodeError struct {
	Reason string
	Err    error
}

func (e *DecodeError) Error() string {
	if e.Err != nil {
		return "bedrockconverse: " + e.Reason + ": " + e.Err.Error()
	}
	return "bedrockconverse: " + e.Reason
}

func (e *DecodeError) Unwrap() error { return e.Err }

// StreamDecodeError reports malformed or out-of-order ConverseStream events.
// It never includes the raw provider event body in its diagnostic.
type StreamDecodeError struct {
	Reason string
	Err    error
}

func (e *StreamDecodeError) Error() string {
	if e.Err != nil {
		return "bedrockconverse: stream " + e.Reason + ": " + e.Err.Error()
	}
	return "bedrockconverse: stream " + e.Reason
}

func (e *StreamDecodeError) Unwrap() error { return e.Err }

// StreamAPIError reports an AWS event-stream exception after the HTTP success
// boundary. Only its typed exception name and bounded message are retained.
type StreamAPIError struct {
	Type    string
	Message string
}

func (e *StreamAPIError) Error() string {
	message := "bedrockconverse: stream error"
	if e.Type != "" {
		message += " (" + e.Type + ")"
	}
	if e.Message != "" {
		message += ": " + e.Message
	}
	return message
}

// ForeignReasoningSignatureError reports a reasoning signature this dialect
// cannot prove it minted, either because another dialect's label is attached
// (Format names it) or because no label is attached at all (Format is empty).
//
// It is a hard error, not a degrade. A reasoning signature is verified by its
// issuer: measured against api.anthropic.com on 2026-08-13, a verbatim
// signature is accepted, a tampered one draws HTTP 400, and an EMPTY one draws
// the same 400. Both degrades therefore lose — forwarding sends a request
// certain to be rejected, stripping sends an unsigned reasoning block equally
// certain to be rejected while destroying the only copy of the continuation
// state. Failing here costs the same turn and names the cause.
//
// Converse fronts the same Claude models as the Anthropic Messages API, so the
// foreign case is an ordinary session moved between endpoints, not an attack.
type ForeignReasoningSignatureError struct {
	// Format is the label carried by the signature, or "" when it carries none.
	Format string
}

func (e *ForeignReasoningSignatureError) Error() string {
	origin := "no dialect label"
	if e.Format != "" {
		origin = "dialect " + strconv.Quote(e.Format)
	}
	return "bedrockconverse: refusing to replay a reasoning signature minted by " + origin +
		"; this dialect replays only signatures labelled " + strconv.Quote(signatureFormatBedrockConverse) +
		", and an unsigned reasoning block is rejected too, so the signature can be neither forwarded nor dropped"
}
