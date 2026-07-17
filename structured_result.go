package inference

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"strings"
	"unicode/utf8"

	"github.com/looprig/core/content"
	"github.com/looprig/inference/stream"
)

// StructuredResult extracts one structured JSON object from a complete
// response and verifies that the finish reason agrees with its representation.
func StructuredResult(resp *Response) (json.RawMessage, error) {
	if resp == nil {
		return nil, malformedError(MalformedReasonNilResponse, nil)
	}

	switch resp.FinishReason {
	case stream.FinishReasonLength, stream.FinishReasonContentFilter:
		return nil, &StructuredOutputFinishError{Reason: resp.FinishReason}
	case stream.FinishReasonStop:
		if containsToolCall(resp.Message) {
			return nil, &StructuredOutputFinishError{Reason: resp.FinishReason}
		}
	case stream.FinishReasonToolUse:
		if !isTerminalToolRepresentation(resp.Message) {
			return nil, &StructuredOutputFinishError{Reason: resp.FinishReason}
		}
	case stream.FinishReasonUnknown:
	default:
		return nil, &StructuredOutputFinishError{Reason: StructuredOutputFinishReasonOther}
	}

	return StructuredMessageResult(resp.Message)
}

// StructuredMessageResult extracts exactly one JSON-object representation
// from assistant text fragments or one reserved terminal-tool input. Thinking
// blocks are ignored. The returned bytes are compacted and independently owned.
func StructuredMessageResult(msg *content.AIMessage) (json.RawMessage, error) {
	if msg == nil {
		return nil, malformedError(MalformedReasonNilMessage, nil)
	}
	if msg.Role != content.RoleAssistant {
		return nil, malformedError(MalformedReasonWrongRole, nil)
	}

	var text strings.Builder
	var terminalInput json.RawMessage
	textSeen := false
	terminalCount := 0
	ordinaryCount := 0

	for _, block := range msg.Blocks {
		switch typed := block.(type) {
		case *content.TextBlock:
			if typed == nil {
				return nil, malformedError(MalformedReasonNilBlock, nil)
			}
			textSeen = true
			text.WriteString(typed.Text)
		case *content.ThinkingBlock:
			if typed == nil {
				return nil, malformedError(MalformedReasonNilBlock, nil)
			}
		case *content.ToolUseBlock:
			if typed == nil {
				return nil, malformedError(MalformedReasonNilBlock, nil)
			}
			if typed.Name == StructuredOutputToolName {
				terminalCount++
				terminalInput = typed.Input
			} else {
				ordinaryCount++
			}
		case *content.ImageBlock:
			if typed == nil {
				return nil, malformedError(MalformedReasonNilBlock, nil)
			}
			return nil, malformedError(MalformedReasonInvalidBlock, nil)
		case *content.AudioBlock:
			if typed == nil {
				return nil, malformedError(MalformedReasonNilBlock, nil)
			}
			return nil, malformedError(MalformedReasonInvalidBlock, nil)
		case *content.DocumentBlock:
			if typed == nil {
				return nil, malformedError(MalformedReasonNilBlock, nil)
			}
			return nil, malformedError(MalformedReasonInvalidBlock, nil)
		case *content.ToolResultBlock:
			if typed == nil {
				return nil, malformedError(MalformedReasonNilBlock, nil)
			}
			return nil, malformedError(MalformedReasonInvalidBlock, nil)
		default:
			return nil, malformedError(MalformedReasonNilBlock, nil)
		}
	}

	if textSeen && terminalCount+ordinaryCount > 0 {
		return nil, malformedError(MalformedReasonAmbiguous, nil)
	}
	if terminalCount > 1 || terminalCount == 1 && ordinaryCount > 0 {
		return nil, malformedError(MalformedReasonAmbiguous, nil)
	}
	if ordinaryCount > 0 {
		return nil, malformedError(MalformedReasonInvalidRepresentation, nil)
	}
	if textSeen {
		return parseStructuredObject([]byte(text.String()))
	}
	if terminalCount == 1 {
		return parseStructuredObject(terminalInput)
	}
	return nil, malformedError(MalformedReasonEmpty, nil)
}

// DecodeOutput extracts and strictly decodes a response into a non-nil concrete
// pointer. Required and other domain invariants remain the caller's concern.
func DecodeOutput(resp *Response, out any) error {
	raw, err := StructuredResult(resp)
	if err != nil {
		return err
	}
	return decodeStructuredOutput(raw, out)
}

// DecodeMessageOutput extracts and strictly decodes a message into a non-nil
// concrete pointer. Required and other domain invariants remain caller-owned.
func DecodeMessageOutput(msg *content.AIMessage, out any) error {
	raw, err := StructuredMessageResult(msg)
	if err != nil {
		return err
	}
	return decodeStructuredOutput(raw, out)
}

func parseStructuredObject(raw []byte) (json.RawMessage, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return nil, malformedError(MalformedReasonEmpty, raw)
	}
	if !utf8.Valid(trimmed) || !json.Valid(trimmed) {
		return nil, malformedError(MalformedReasonMalformedJSON, raw)
	}
	if trimmed[0] != '{' {
		return nil, malformedError(MalformedReasonRootNotObject, raw)
	}

	compact := make([]byte, 0, len(trimmed))
	compact = append(compact, trimmed...)
	buffer := bytes.NewBuffer(compact[:0])
	if err := json.Compact(buffer, trimmed); err != nil {
		return nil, malformedError(MalformedReasonMalformedJSON, raw)
	}
	result := make(json.RawMessage, buffer.Len())
	copy(result, buffer.Bytes())
	return result, nil
}

func malformedError(reason MalformedStructuredOutputReason, raw []byte) error {
	return &MalformedStructuredOutputError{
		ReasonCode: reason,
		Length:     len(raw),
		SHA256:     sha256.Sum256(raw),
	}
}

func containsToolCall(msg *content.AIMessage) bool {
	if msg == nil {
		return false
	}
	for _, block := range msg.Blocks {
		if tool, ok := block.(*content.ToolUseBlock); ok && tool != nil {
			return true
		}
	}
	return false
}

func isTerminalToolRepresentation(msg *content.AIMessage) bool {
	if msg == nil || msg.Role != content.RoleAssistant {
		return false
	}
	terminalCount := 0
	for _, block := range msg.Blocks {
		switch typed := block.(type) {
		case *content.ThinkingBlock:
			if typed == nil {
				return false
			}
		case *content.ToolUseBlock:
			if typed == nil || typed.Name != StructuredOutputToolName {
				return false
			}
			terminalCount++
		default:
			return false
		}
	}
	return terminalCount == 1
}

func decodeStructuredOutput(raw json.RawMessage, out any) error {
	if out == nil {
		return &SchemaValidationError{Field: SchemaFieldOutput, ReasonCode: SchemaReasonInvalidTarget}
	}
	target := reflect.ValueOf(out)
	if target.Kind() != reflect.Pointer || target.IsNil() || target.Elem().Kind() != reflect.Struct {
		return &SchemaValidationError{Field: SchemaFieldOutput, ReasonCode: SchemaReasonInvalidTarget}
	}

	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(out); err != nil {
		return &SchemaValidationError{Field: SchemaFieldOutput, ReasonCode: SchemaReasonDecodeFailed}
	}
	var trailing json.RawMessage
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return &SchemaValidationError{Field: SchemaFieldOutput, ReasonCode: SchemaReasonDecodeFailed}
	}
	return nil
}
