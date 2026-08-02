package bedrockconverse

import (
	"bytes"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	model "github.com/looprig/inference/model"
)

func buildRequest(req inference.Request) (converseRequest, error) {
	if err := inference.ValidateRequestFeatures(req); err != nil {
		return converseRequest{}, err
	}

	if err := validateTools(req.Tools); err != nil {
		return converseRequest{}, err
	}

	r := converseRequest{Messages: make([]converseMessage, 0, len(req.Messages))}
	if req.System != "" {
		r.System = append(r.System, systemContentBlock{Text: req.System})
	}

	for _, conversation := range req.Messages {
		switch message := conversation.(type) {
		case *content.SystemMessage:
			blocks, err := encodeSystemBlocks(message.Blocks)
			if err != nil {
				return converseRequest{}, err
			}
			r.System = append(r.System, blocks...)
		case *content.UserMessage:
			blocks, err := encodeContentBlocks(message.Blocks, roleUser)
			if err != nil {
				return converseRequest{}, err
			}
			r.Messages = append(r.Messages, converseMessage{Role: roleUser, Content: blocks})
		case *content.AIMessage:
			blocks, err := encodeContentBlocks(message.Blocks, roleAssistant)
			if err != nil {
				return converseRequest{}, err
			}
			r.Messages = append(r.Messages, converseMessage{Role: roleAssistant, Content: blocks})
		case *content.ToolResultMessage:
			block, err := encodeToolResultMessage(message)
			if err != nil {
				return converseRequest{}, err
			}
			r.Messages = append(r.Messages, converseMessage{Role: roleUser, Content: []converseContentBlock{{ToolResult: block}}})
		default:
			return converseRequest{}, &UnsupportedConversationError{Conversation: fmt.Sprintf("%T", conversation)}
		}
	}

	sampling := req.Model.Sampling
	if req.Override != nil {
		sampling = *req.Override
	}
	if config := samplingConfig(sampling); config != nil {
		r.InferenceConfig = config
	}

	if len(req.Tools) > 0 {
		tools := make([]toolDefinition, 0, len(req.Tools))
		for _, tool := range req.Tools {
			tools = append(tools, toolDefinition{ToolSpec: toolSpec{
				Name:        tool.Name,
				Description: tool.Description,
				InputSchema: toolInputSchema{JSON: append(json.RawMessage(nil), tool.Schema...)},
			}})
		}
		r.ToolConfig = &toolConfig{Tools: tools}
		if req.ToolChoice == inference.ToolChoiceRequired {
			r.ToolConfig.ToolChoice = &toolChoice{Any: &struct{}{}}
		}
	}

	if req.Output != nil {
		r.OutputConfig = &outputConfig{TextFormat: &textFormat{
			Type: "json_schema",
			Structure: &textStructure{JSONSchema: jsonSchema{
				Schema:      string(req.Output.Schema),
				Name:        req.Output.Name,
				Description: req.Output.Description,
			}},
		}}
	}

	return r, nil
}

func marshalRequest(req converseRequest) ([]byte, error) {
	encoded, err := json.Marshal(req)
	if err != nil {
		return nil, &EncodeError{Reason: "marshal request", Err: err}
	}
	return encoded, nil
}

func encodeSystemBlocks(blocks []content.Block) ([]systemContentBlock, error) {
	encoded := make([]systemContentBlock, 0, len(blocks))
	for _, block := range blocks {
		text, ok := block.(*content.TextBlock)
		if !ok || text == nil {
			return nil, unsupportedBlock(block, "system content supports text only")
		}
		encoded = append(encoded, systemContentBlock{Text: text.Text})
	}
	return encoded, nil
}

func encodeContentBlocks(blocks []content.Block, role string) ([]converseContentBlock, error) {
	encoded := make([]converseContentBlock, 0, len(blocks))
	hasDocument := false
	hasText := false
	for _, block := range blocks {
		if err := validateContentBlockRole(block, role); err != nil {
			return nil, err
		}
		wireBlock, err := encodeContentBlock(block)
		if err != nil {
			return nil, err
		}
		if wireBlock.Text != nil {
			hasText = true
		}
		if wireBlock.Document != nil {
			hasDocument = true
		}
		encoded = append(encoded, wireBlock)
	}
	if hasDocument && !hasText {
		return nil, &UnsupportedBlockError{Block: "*content.DocumentBlock", Reason: "a document requires a text block in the same message"}
	}
	return encoded, nil
}

func validateContentBlockRole(block content.Block, role string) error {
	switch block.(type) {
	case *content.ImageBlock, *content.DocumentBlock:
		if role != roleUser {
			return unsupportedBlock(block, "image and document blocks are only valid in user messages")
		}
	case *content.ThinkingBlock, *content.ToolUseBlock:
		if role != roleAssistant {
			return unsupportedBlock(block, "reasoning and tool-use blocks are only valid in assistant messages")
		}
	case *content.ToolResultBlock:
		if role != roleUser {
			return unsupportedBlock(block, "tool-result blocks are only valid in user messages")
		}
	}
	return nil
}

func encodeContentBlock(block content.Block) (converseContentBlock, error) {
	switch block := block.(type) {
	case *content.TextBlock:
		if block == nil {
			return converseContentBlock{}, unsupportedBlock(block, "nil block")
		}
		text := block.Text
		return converseContentBlock{Text: &text}, nil
	case *content.ImageBlock:
		image, err := encodeImage(block)
		if err != nil {
			return converseContentBlock{}, err
		}
		return converseContentBlock{Image: image}, nil
	case *content.DocumentBlock:
		document, err := encodeDocument(block)
		if err != nil {
			return converseContentBlock{}, err
		}
		return converseContentBlock{Document: document}, nil
	case *content.ThinkingBlock:
		if block == nil {
			return converseContentBlock{}, unsupportedBlock(block, "nil block")
		}
		text := block.Thinking
		return converseContentBlock{ReasoningContent: &reasoningContent{ReasoningText: &reasoningText{
			Text:      &text,
			Signature: block.Signature,
		}}}, nil
	case *content.ToolUseBlock:
		toolUse, err := encodeToolUse(block)
		if err != nil {
			return converseContentBlock{}, err
		}
		return converseContentBlock{ToolUse: toolUse}, nil
	case *content.ToolResultBlock:
		result, err := encodeToolResultBlock(block)
		if err != nil {
			return converseContentBlock{}, err
		}
		return converseContentBlock{ToolResult: result}, nil
	default:
		return converseContentBlock{}, unsupportedBlock(block, "no Converse content-block representation")
	}
}

func encodeImage(image *content.ImageBlock) (*imageContent, error) {
	if image == nil {
		return nil, unsupportedBlock(image, "nil block")
	}
	if image.Source.URL != "" {
		return nil, unsupportedBlock(image, "Bedrock Converse accepts inline image bytes, not URLs")
	}
	format := imageFormat(image.MediaType)
	if format == "" {
		return nil, unsupportedBlock(image, "unsupported image format")
	}
	if len(image.Source.Data) == 0 {
		return nil, unsupportedBlock(image, "image source is empty")
	}
	return &imageContent{Format: format, Source: imageSource{Bytes: append([]byte(nil), image.Source.Data...)}}, nil
}

func encodeDocument(document *content.DocumentBlock) (*documentContent, error) {
	if document == nil {
		return nil, unsupportedBlock(document, "nil block")
	}
	format := documentFormat(document.MediaType, document.Name)
	if format == "" {
		return nil, unsupportedBlock(document, "unsupported document format")
	}
	name := document.Name
	if name == "" {
		name = "document"
	}
	if err := validateDocumentName(name); err != nil {
		return nil, err
	}
	result := &documentContent{Format: format, Name: name}
	switch {
	case len(document.Data) > 0:
		result.Source.Bytes = append([]byte(nil), document.Data...)
	case document.Text != "":
		text := document.Text
		result.Source.Text = &text
	default:
		return nil, unsupportedBlock(document, "document source is empty")
	}
	return result, nil
}

func encodeToolUse(toolUse *content.ToolUseBlock) (*toolUseContent, error) {
	if toolUse == nil {
		return nil, unsupportedBlock(toolUse, "nil block")
	}
	if toolUse.ID == "" || toolUse.Name == "" {
		return nil, &ToolInputError{Tool: toolUse.Name, Reason: "tool-use id and name must not be empty"}
	}
	input, err := normalizedObject(toolUse.Input)
	if err != nil {
		return nil, &ToolInputError{Tool: toolUse.Name, Reason: err.Error()}
	}
	return &toolUseContent{ToolUseID: toolUse.ID, Name: toolUse.Name, Input: input}, nil
}

func encodeToolResultMessage(message *content.ToolResultMessage) (*toolResultContent, error) {
	if message == nil {
		return nil, &EncodeError{Reason: "nil tool result message"}
	}
	if message.ToolUseID == "" {
		return nil, &EncodeError{Reason: "tool result is missing tool-use id"}
	}
	blocks, err := encodeToolResultBlocks(message.Blocks)
	if err != nil {
		return nil, err
	}
	result := &toolResultContent{ToolUseID: message.ToolUseID, Content: blocks}
	if message.IsError {
		result.Status = toolResultStatusError
	}
	return result, nil
}

func encodeToolResultBlock(block *content.ToolResultBlock) (*toolResultContent, error) {
	if block == nil {
		return nil, unsupportedBlock(block, "nil block")
	}
	if block.ToolUseID == "" {
		return nil, &EncodeError{Reason: "tool result is missing tool-use id"}
	}
	blocks, err := encodeToolResultBlocks(block.Content)
	if err != nil {
		return nil, err
	}
	result := &toolResultContent{ToolUseID: block.ToolUseID, Content: blocks}
	if block.IsError {
		result.Status = toolResultStatusError
	}
	return result, nil
}

func encodeToolResultBlocks(blocks []content.Block) ([]toolResultBlock, error) {
	if len(blocks) == 0 {
		return nil, &EncodeError{Reason: "tool result content must not be empty"}
	}
	encoded := make([]toolResultBlock, 0, len(blocks))
	for _, block := range blocks {
		switch block := block.(type) {
		case *content.TextBlock:
			if block == nil {
				return nil, unsupportedBlock(block, "nil block")
			}
			text := block.Text
			encoded = append(encoded, toolResultBlock{Text: &text})
		case *content.ImageBlock:
			image, err := encodeImage(block)
			if err != nil {
				return nil, err
			}
			encoded = append(encoded, toolResultBlock{Image: image})
		case *content.DocumentBlock:
			document, err := encodeDocument(block)
			if err != nil {
				return nil, err
			}
			encoded = append(encoded, toolResultBlock{Document: document})
		default:
			return nil, unsupportedBlock(block, "tool result content supports text, image, and document blocks")
		}
	}
	return encoded, nil
}

func validateTools(tools []inference.Tool) error {
	seen := make(map[string]struct{}, len(tools))
	for _, tool := range tools {
		if tool.Name == "" {
			return &ToolSchemaError{Reason: "tool name is empty"}
		}
		if _, exists := seen[tool.Name]; exists {
			return &ToolSchemaError{Tool: tool.Name, Reason: "duplicate tool name"}
		}
		seen[tool.Name] = struct{}{}
		trimmed := bytes.TrimSpace(tool.Schema)
		if len(trimmed) == 0 {
			return &ToolSchemaError{Tool: tool.Name, Reason: "schema is empty"}
		}
		if !json.Valid(trimmed) || trimmed[0] != '{' {
			return &ToolSchemaError{Tool: tool.Name, Reason: "schema must be a JSON object"}
		}
	}
	return nil
}

func samplingConfig(sampling model.Sampling) *inferenceConfig {
	config := &inferenceConfig{
		MaxTokens:     sampling.MaxTokens,
		Temperature:   sampling.Temperature,
		TopP:          sampling.TopP,
		StopSequences: sampling.Stop,
	}
	if config.MaxTokens == nil && config.Temperature == nil && config.TopP == nil && len(config.StopSequences) == 0 {
		return nil
	}
	return config
}

func normalizedObject(raw []byte) (json.RawMessage, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return json.RawMessage(`{}`), nil
	}
	if !json.Valid(trimmed) || trimmed[0] != '{' {
		return nil, fmt.Errorf("arguments must be a JSON object")
	}
	return append(json.RawMessage(nil), trimmed...), nil
}

func imageFormat(mediaType content.MediaType) string {
	switch strings.ToLower(string(mediaType)) {
	case string(content.MediaTypeImageJPEG):
		return imageFormatJPEG
	case string(content.MediaTypeImagePNG):
		return imageFormatPNG
	case string(content.MediaTypeImageGIF):
		return imageFormatGIF
	case string(content.MediaTypeImageWebP):
		return imageFormatWebP
	default:
		return ""
	}
}

func documentFormat(mediaType content.MediaType, name string) string {
	switch strings.ToLower(string(mediaType)) {
	case string(content.MediaTypeDocumentPDF):
		return "pdf"
	case string(content.MediaTypeDocumentText):
		return "txt"
	case string(content.MediaTypeDocumentHTML):
		return "html"
	case string(content.MediaTypeDocumentCSV):
		return "csv"
	case string(content.MediaTypeDocumentMarkdown):
		return "md"
	case string(content.MediaTypeDocumentDOCX):
		return "docx"
	case string(content.MediaTypeDocumentXLSX):
		return "xlsx"
	case "application/msword":
		return "doc"
	case "application/vnd.ms-excel":
		return "xls"
	}
	ext := strings.TrimPrefix(strings.ToLower(filepath.Ext(name)), ".")
	switch ext {
	case "pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md":
		return ext
	default:
		return ""
	}
}

func validateDocumentName(name string) error {
	if name == "" {
		return &UnsupportedBlockError{Block: "*content.DocumentBlock", Reason: "document name is empty"}
	}
	runes := []rune(name)
	if len(runes) > 200 {
		return &UnsupportedBlockError{Block: "*content.DocumentBlock", Reason: "document name exceeds 200 characters"}
	}
	previousWhitespace := false
	for _, r := range runes {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			previousWhitespace = false
		case r == '-', r == '(', r == ')', r == '[', r == ']':
			previousWhitespace = false
		case isDocumentWhitespace(r):
			if previousWhitespace {
				return &UnsupportedBlockError{Block: "*content.DocumentBlock", Reason: "document name contains consecutive whitespace"}
			}
			previousWhitespace = true
		default:
			return &UnsupportedBlockError{Block: "*content.DocumentBlock", Reason: "document name contains unsupported characters"}
		}
	}
	return nil
}

func isDocumentWhitespace(r rune) bool {
	return r == ' ' || r == '\t' || r == '\n' || r == '\r'
}

func unsupportedBlock(block content.Block, reason string) error {
	return &UnsupportedBlockError{Block: fmt.Sprintf("%T", block), Reason: reason}
}
