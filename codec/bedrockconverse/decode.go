package bedrockconverse

import (
	"bytes"
	"encoding/json"
	"strings"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/internal/usagenorm"
	stream "github.com/looprig/inference/stream"
	usage "github.com/looprig/inference/usage"
)

// DecodeResponse parses a native Bedrock Converse response into the shared
// inference response. Bedrock does not echo the model ID in this envelope, so
// Model is intentionally left empty for the provider client to fill from the
// bound request.
func DecodeResponse(body []byte) (*inference.Response, error) {
	var wire converseResponse
	if err := json.Unmarshal(body, &wire); err != nil {
		return nil, &DecodeError{Reason: "unmarshal response body", Err: err}
	}
	if wire.Output == nil || wire.Output.Message == nil {
		return nil, &DecodeError{Reason: "response is missing output.message"}
	}

	blocks, err := decodeContentBlocks(wire.Output.Message.Content)
	if err != nil {
		return nil, err
	}
	normalized, err := normalizeUsage(wire.Usage)
	if err != nil {
		return nil, err
	}
	var messageUsage *content.Usage
	if normalized != nil {
		copyUsage := *normalized
		messageUsage = &copyUsage
	}
	return &inference.Response{
		Message: &content.AIMessage{
			Message: content.Message{Role: content.RoleAssistant, Blocks: blocks},
			Usage:   messageUsage,
		},
		Usage:        normalized,
		FinishReason: mapFinishReason(wire.StopReason),
	}, nil
}

func normalizeUsage(wire *responseUsage) (*usage.Usage, error) {
	if wire == nil {
		return nil, nil
	}
	input, err := wire.InputTokens.TokenCount(usagenorm.FieldInputTokens)
	if err != nil {
		return nil, err
	}
	output, err := wire.OutputTokens.TokenCount(usagenorm.FieldOutputTokens)
	if err != nil {
		return nil, err
	}
	cacheRead, err := wire.CacheReadInputTokens.TokenCount(usagenorm.FieldCacheReadTokens)
	if err != nil {
		return nil, err
	}
	cacheWrite, err := wire.CacheWriteInputTokens.TokenCount(usagenorm.FieldCacheCreationTokens)
	if err != nil {
		return nil, err
	}
	normalized := usage.Usage{
		InputTokens:         input,
		OutputTokens:        output,
		CacheReadTokens:     cacheRead,
		CacheCreationTokens: cacheWrite,
	}
	if err := usagenorm.ValidateUsage(normalized); err != nil {
		return nil, err
	}
	return &normalized, nil
}

func decodeContentBlocks(blocks []converseContentBlock) ([]content.Block, error) {
	decoded := make([]content.Block, 0, len(blocks))
	for _, block := range blocks {
		switch {
		case block.Text != "":
			decoded = append(decoded, &content.TextBlock{Text: block.Text})
		case block.Image != nil:
			image, err := decodeImage(block.Image)
			if err != nil {
				return nil, err
			}
			decoded = append(decoded, image)
		case block.Document != nil:
			document, err := decodeDocument(block.Document)
			if err != nil {
				return nil, err
			}
			decoded = append(decoded, document)
		case block.ReasoningContent != nil:
			if block.ReasoningContent.ReasoningText == nil {
				continue
			}
			reasoning := block.ReasoningContent.ReasoningText
			decoded = append(decoded, &content.ThinkingBlock{Thinking: reasoning.Text, Signature: reasoning.Signature})
		case block.ToolUse != nil:
			input, err := decodeToolInput(block.ToolUse.Input)
			if err != nil {
				return nil, err
			}
			if block.ToolUse.ToolUseID == "" || block.ToolUse.Name == "" {
				return nil, &DecodeError{Reason: "toolUse is missing toolUseId or name"}
			}
			decoded = append(decoded, &content.ToolUseBlock{ID: block.ToolUse.ToolUseID, Name: block.ToolUse.Name, Input: input})
		case block.ToolResult != nil:
			result, err := decodeToolResult(block.ToolResult)
			if err != nil {
				return nil, err
			}
			decoded = append(decoded, result)
		default:
			// Unknown provider-only content blocks are intentionally skipped. The
			// shared vocabulary has no safe representation for them.
		}
	}
	return decoded, nil
}

func decodeToolInput(raw json.RawMessage) (json.RawMessage, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || !json.Valid(trimmed) || trimmed[0] != '{' {
		return nil, &DecodeError{Reason: "toolUse.input is not a JSON object"}
	}
	return append(json.RawMessage(nil), trimmed...), nil
}

func decodeImage(image *imageContent) (*content.ImageBlock, error) {
	if image == nil || image.Format == "" || len(image.Source.Bytes) == 0 {
		return nil, &DecodeError{Reason: "image content block is incomplete"}
	}
	return &content.ImageBlock{
		MediaType: content.MediaType("image/" + strings.ToLower(image.Format)),
		Source:    content.ImageSource{Data: append([]byte(nil), image.Source.Bytes...)},
	}, nil
}

func decodeDocument(document *documentContent) (*content.DocumentBlock, error) {
	if document == nil || document.Format == "" || document.Name == "" {
		return nil, &DecodeError{Reason: "document content block is incomplete"}
	}
	decoded := &content.DocumentBlock{
		MediaType: documentMediaType(document.Format),
		Name:      document.Name,
	}
	if len(document.Source.Bytes) > 0 {
		decoded.Data = append([]byte(nil), document.Source.Bytes...)
	} else if document.Source.Text != "" {
		decoded.Text = document.Source.Text
	} else {
		return nil, &DecodeError{Reason: "document content block has empty source"}
	}
	return decoded, nil
}

func decodeToolResult(result *toolResultContent) (*content.ToolResultBlock, error) {
	if result.ToolUseID == "" {
		return nil, &DecodeError{Reason: "toolResult is missing toolUseId"}
	}
	status := result.Status
	if status != "" && status != toolResultStatusSuccess && status != toolResultStatusError {
		return nil, &DecodeError{Reason: "toolResult has unknown status"}
	}
	blocks := make([]content.Block, 0, len(result.Content))
	for _, block := range result.Content {
		switch {
		case block.Text != "":
			blocks = append(blocks, &content.TextBlock{Text: block.Text})
		case block.Image != nil:
			image, err := decodeImage(block.Image)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, image)
		case block.Document != nil:
			document, err := decodeDocument(block.Document)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, document)
		default:
			// Unknown tool-result sub-blocks have no neutral representation.
		}
	}
	return &content.ToolResultBlock{ToolUseID: result.ToolUseID, Content: blocks, IsError: status == toolResultStatusError}, nil
}

func documentMediaType(format string) content.MediaType {
	switch strings.ToLower(format) {
	case "pdf":
		return content.MediaTypeDocumentPDF
	case "txt":
		return content.MediaTypeDocumentText
	case "html":
		return content.MediaTypeDocumentHTML
	case "csv":
		return content.MediaTypeDocumentCSV
	case "md":
		return content.MediaTypeDocumentMarkdown
	case "docx":
		return content.MediaTypeDocumentDOCX
	case "xlsx":
		return content.MediaTypeDocumentXLSX
	default:
		return content.MediaType("application/" + strings.ToLower(format))
	}
}

func mapFinishReason(reason string) stream.FinishReason {
	switch reason {
	case "end_turn", "stop_sequence":
		return stream.FinishReasonStop
	case "max_tokens", "model_context_window_exceeded":
		return stream.FinishReasonLength
	case "tool_use":
		return stream.FinishReasonToolUse
	case "content_filtered", "guardrail_intervened":
		return stream.FinishReasonContentFilter
	default:
		return stream.FinishReasonUnknown
	}
}
