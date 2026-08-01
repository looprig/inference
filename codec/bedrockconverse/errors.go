package bedrockconverse

import "fmt"

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
