package inference

import (
	"context"
	"encoding/json"
	"unicode/utf8"

	"github.com/looprig/core/content"
	"github.com/looprig/inference/model"

	"github.com/looprig/inference/stream"
)

// Client is the provider-neutral inference interface.
type Client interface {
	Invoke(ctx context.Context, req Request) (*Response, error)
	Stream(ctx context.Context, req Request) (*stream.StreamReader[content.Chunk], error)
}

// Request is the provider-neutral inference request. It carries a secret-free
// Model descriptor for this turn, the per-agent System prompt, the message
// thread, the count of trailing transient messages, the exposed tools, an
// optional structured Output contract, the ToolChoice (zero value: auto), and
// an optional per-call sampling Override (nil means use Model.Sampling).
type Request struct {
	Model             model.Model
	System            string
	Messages          content.AgenticMessages
	TransientMessages int
	Tools             []Tool
	Output            *OutputSchema
	ToolChoice        ToolChoice
	Override          *model.Sampling
}

// InvalidTransientMessagesError reports a transient-message count that falls
// outside the request's message slice.
type InvalidTransientMessagesError struct {
	Transient int
	Messages  int
}

func (e *InvalidTransientMessagesError) Error() string {
	return "inference: transient message count is outside request messages"
}

// ValidateRequestFeatures validates provider-neutral request feature
// combinations before a codec attempts to encode them.
func ValidateRequestFeatures(req Request) error {
	if req.TransientMessages < 0 || req.TransientMessages > len(req.Messages) {
		return &InvalidTransientMessagesError{
			Transient: req.TransientMessages,
			Messages:  len(req.Messages),
		}
	}

	// The ToolChoice type makes the name inseparable from the named variant,
	// so the only tool-choice invariants left are the cross-field ones no
	// type can encode: both are about the request's Tools, which the choice
	// cannot see.
	switch req.ToolChoice.mode {
	case ToolChoiceModeAuto:
	case ToolChoiceModeRequired:
		if len(req.Tools) == 0 {
			return &StructuredOutputConflictError{Feature: "tool_choice_required_without_tools"}
		}
	case ToolChoiceModeNamed:
		// A forced name must name a tool this request actually declares:
		// every dialect resolves the choice against its own tools array, so
		// an undeclared name is a guaranteed provider 400 — and no provider
		// request schema catches it. Checking it here also subsumes the
		// empty-tools and empty-name cases.
		if !declaresTool(req.Tools, req.ToolChoice.name) {
			return &StructuredOutputConflictError{Feature: "tool_choice_tool_undeclared_name"}
		}
	default:
		// Unreachable from outside this package: the discriminant is
		// unexported. Fails closed so a variant added here later cannot be
		// silently encoded as auto by codecs that do not recognize it.
		return &StructuredOutputConflictError{Feature: "tool_choice"}
	}

	if !req.Model.Caps.AcceptsImages && messagesCarryImages(req.Messages) {
		return &ImageInputUnsupportedError{Model: boundedStructuredDiagnostic(req.Model.Name)}
	}

	if req.Output == nil {
		return nil
	}
	if err := ValidateOutputSchema(*req.Output); err != nil {
		return err
	}

	seenTools := make(map[string]struct{}, len(req.Tools))
	for _, tool := range req.Tools {
		if tool.Name == StructuredOutputToolName {
			return &StructuredOutputConflictError{Feature: "reserved_structured_output_tool"}
		}
		if _, ok := seenTools[tool.Name]; ok {
			return &StructuredOutputConflictError{Feature: "duplicate_tool_name"}
		}
		seenTools[tool.Name] = struct{}{}
	}

	if !req.Model.Caps.StructuredOutput {
		return &StructuredOutputUnsupportedError{Model: boundedStructuredDiagnostic(req.Model.Name)}
	}
	if len(req.Tools) > 0 && !req.Model.Caps.StructuredOutputWithTools {
		return &StructuredOutputWithToolsUnsupportedError{Model: boundedStructuredDiagnostic(req.Model.Name)}
	}
	return nil
}

// declaresTool reports whether name matches a tool the request exposes.
func declaresTool(tools []Tool, name string) bool {
	for _, tool := range tools {
		if tool.Name == name {
			return true
		}
	}
	return false
}

// messagesCarryImages reports whether any message in the thread carries an
// ImageBlock, including images nested inside ToolResultBlock content. The
// Conversation interface is sealed, so the four concrete message types are
// enumerated; an unknown type contributes no blocks.
func messagesCarryImages(msgs content.AgenticMessages) bool {
	for _, conv := range msgs {
		var blocks []content.Block
		switch m := conv.(type) {
		case *content.SystemMessage:
			blocks = m.Blocks
		case *content.UserMessage:
			blocks = m.Blocks
		case *content.AIMessage:
			blocks = m.Blocks
		case *content.ToolResultMessage:
			blocks = m.Blocks
		}
		if blocksCarryImages(blocks) {
			return true
		}
	}
	return false
}

func blocksCarryImages(blocks []content.Block) bool {
	for _, b := range blocks {
		switch b := b.(type) {
		case *content.ImageBlock:
			return true
		case *content.ToolResultBlock:
			if blocksCarryImages(b.Content) {
				return true
			}
		}
	}
	return false
}

func boundedStructuredDiagnostic(value string) string {
	if !utf8.ValidString(value) {
		return "invalid-utf8"
	}
	if len(value) <= MaxStructuredOutputDiagnosticBytes {
		return value
	}
	end := MaxStructuredOutputDiagnosticBytes
	for end > 0 && !utf8.RuneStart(value[end]) {
		end--
	}
	return value[:end]
}

// Response is the complete provider-neutral response.
type Response struct {
	Message      *content.AIMessage
	Usage        *content.Usage
	Model        string
	FinishReason stream.FinishReason
	// Attempts is how many attempts produced this response when served
	// through a retrying decorator; 0 means the serving client does not
	// count attempts, 1 means first-try success.
	Attempts int
}

// Tool is a callable function definition exposed to the model.
type Tool struct {
	Name        string
	Description string
	Schema      json.RawMessage
}
