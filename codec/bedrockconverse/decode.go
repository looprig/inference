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
		decodedBlock, err := decodeContentBlock(block)
		if err != nil {
			return nil, err
		}
		decoded = append(decoded, decodedBlock)
	}
	return decoded, nil
}

func decodeContentBlock(block converseContentBlock) (content.Block, error) {
	if variants := contentBlockVariantCount(block); variants != 1 {
		return nil, &DecodeError{Reason: "content block must contain exactly one recognized variant"}
	}
	switch {
	case block.Text != nil:
		return &content.TextBlock{Text: *block.Text}, nil
	case block.Image != nil:
		return decodeImage(block.Image)
	case block.Document != nil:
		return decodeDocument(block.Document)
	case block.ReasoningContent != nil:
		reasoning := block.ReasoningContent
		reasoningVariants := 0
		if reasoning.ReasoningText != nil {
			reasoningVariants++
		}
		if len(reasoning.RedactedContent) > 0 {
			reasoningVariants++
		}
		if reasoningVariants != 1 {
			return nil, &DecodeError{Reason: "reasoningContent must contain exactly one recognized variant"}
		}
		if len(reasoning.RedactedContent) > 0 {
			return nil, &DecodeError{Reason: "redacted reasoning content has no shared representation"}
		}
		if reasoning.ReasoningText.Text == nil {
			return nil, &DecodeError{Reason: "reasoningText is missing text"}
		}
		return &content.ThinkingBlock{Thinking: *reasoning.ReasoningText.Text, Signature: reasoning.ReasoningText.Signature}, nil
	case block.ToolUse != nil:
		input, err := decodeToolInput(block.ToolUse.Input)
		if err != nil {
			return nil, err
		}
		if block.ToolUse.ToolUseID == "" || block.ToolUse.Name == "" {
			return nil, &DecodeError{Reason: "toolUse is missing toolUseId or name"}
		}
		return &content.ToolUseBlock{ID: block.ToolUse.ToolUseID, Name: block.ToolUse.Name, Input: input}, nil
	case block.ToolResult != nil:
		return decodeToolResult(block.ToolResult)
	default:
		return nil, &DecodeError{Reason: "content block has no recognized variant"}
	}
}

func contentBlockVariantCount(block converseContentBlock) int {
	count := 0
	if block.Text != nil {
		count++
	}
	if block.Image != nil {
		count++
	}
	if block.Document != nil {
		count++
	}
	if block.ReasoningContent != nil {
		count++
	}
	if block.ToolUse != nil {
		count++
	}
	if block.ToolResult != nil {
		count++
	}
	return count
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
	if imageFormat(content.MediaType("image/"+strings.ToLower(image.Format))) == "" {
		return nil, &DecodeError{Reason: "image content block has unsupported format"}
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
	if !isDocumentFormat(document.Format) {
		return nil, &DecodeError{Reason: "document content block has unsupported format"}
	}
	if err := validateDocumentName(document.Name); err != nil {
		return nil, &DecodeError{Reason: "document content block has invalid name"}
	}
	decoded := &content.DocumentBlock{
		MediaType: documentMediaType(document.Format),
		Name:      document.Name,
	}
	hasBytes := len(document.Source.Bytes) > 0
	hasText := document.Source.Text != nil
	if hasBytes == hasText {
		return nil, &DecodeError{Reason: "document content block source must contain exactly one variant"}
	}
	if hasBytes {
		decoded.Data = append([]byte(nil), document.Source.Bytes...)
	} else if document.Source.Text != nil {
		decoded.Text = *document.Source.Text
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
		if variants := toolResultBlockVariantCount(block); variants != 1 {
			return nil, &DecodeError{Reason: "tool result content block must contain exactly one recognized variant"}
		}
		switch {
		case block.Text != nil:
			blocks = append(blocks, &content.TextBlock{Text: *block.Text})
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
		}
	}
	return &content.ToolResultBlock{ToolUseID: result.ToolUseID, Content: blocks, IsError: status == toolResultStatusError}, nil
}

func toolResultBlockVariantCount(block toolResultBlock) int {
	count := 0
	if block.Text != nil {
		count++
	}
	if block.Image != nil {
		count++
	}
	if block.Document != nil {
		count++
	}
	return count
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
	case "doc":
		return content.MediaType("application/msword")
	case "xls":
		return content.MediaType("application/vnd.ms-excel")
	default:
		return content.MediaType("application/" + strings.ToLower(format))
	}
}

func isDocumentFormat(format string) bool {
	switch strings.ToLower(format) {
	case "pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md":
		return true
	default:
		return false
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
