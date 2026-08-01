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
			blocks, err := encodeContentBlocks(message.Blocks)
			if err != nil {
				return converseRequest{}, err
			}
			r.Messages = append(r.Messages, converseMessage{Role: roleUser, Content: blocks})
		case *content.AIMessage:
			blocks, err := encodeContentBlocks(message.Blocks)
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
				InputSchema: append(json.RawMessage(nil), tool.Schema...),
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

func encodeContentBlocks(blocks []content.Block) ([]converseContentBlock, error) {
	encoded := make([]converseContentBlock, 0, len(blocks))
	for _, block := range blocks {
		wireBlock, err := encodeContentBlock(block)
		if err != nil {
			return nil, err
		}
		encoded = append(encoded, wireBlock)
	}
	return encoded, nil
}

func encodeContentBlock(block content.Block) (converseContentBlock, error) {
	switch block := block.(type) {
	case *content.TextBlock:
		if block == nil {
			return converseContentBlock{}, unsupportedBlock(block, "nil block")
		}
		return converseContentBlock{Text: block.Text}, nil
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
		return converseContentBlock{ReasoningContent: &reasoningContent{ReasoningText: &reasoningText{
			Text:      block.Thinking,
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
	result := &documentContent{Format: format, Name: name}
	switch {
	case len(document.Data) > 0:
		result.Source.Bytes = append([]byte(nil), document.Data...)
	case document.Text != "":
		result.Source.Text = document.Text
	default:
		return nil, unsupportedBlock(document, "document source is empty")
	}
	return result, nil
}

func encodeToolUse(toolUse *content.ToolUseBlock) (*toolUseContent, error) {
	if toolUse == nil {
		return nil, unsupportedBlock(toolUse, "nil block")
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
	blocks, err := encodeToolResultBlocks(message.Blocks)
	if err != nil {
		return nil, err
	}
	status := toolResultStatusSuccess
	if message.IsError {
		status = toolResultStatusError
	}
	return &toolResultContent{ToolUseID: message.ToolUseID, Content: blocks, Status: status}, nil
}

func encodeToolResultBlock(block *content.ToolResultBlock) (*toolResultContent, error) {
	if block == nil {
		return nil, unsupportedBlock(block, "nil block")
	}
	blocks, err := encodeToolResultBlocks(block.Content)
	if err != nil {
		return nil, err
	}
	status := toolResultStatusSuccess
	if block.IsError {
		status = toolResultStatusError
	}
	return &toolResultContent{ToolUseID: block.ToolUseID, Content: blocks, Status: status}, nil
}

func encodeToolResultBlocks(blocks []content.Block) ([]toolResultBlock, error) {
	encoded := make([]toolResultBlock, 0, len(blocks))
	for _, block := range blocks {
		switch block := block.(type) {
		case *content.TextBlock:
			if block == nil {
				return nil, unsupportedBlock(block, "nil block")
			}
			encoded = append(encoded, toolResultBlock{Text: block.Text})
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
	}
	ext := strings.TrimPrefix(strings.ToLower(filepath.Ext(name)), ".")
	switch ext {
	case "pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md":
		return ext
	default:
		return ""
	}
}

func unsupportedBlock(block content.Block, reason string) error {
	return &UnsupportedBlockError{Block: fmt.Sprintf("%T", block), Reason: reason}
}
