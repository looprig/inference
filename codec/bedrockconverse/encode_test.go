package bedrockconverse_test

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	codec "github.com/looprig/inference/codec"
	"github.com/looprig/inference/codec/bedrockconverse"
	model "github.com/looprig/inference/model"
)

func decodeObject(t *testing.T, raw []byte) map[string]json.RawMessage {
	t.Helper()
	var object map[string]json.RawMessage
	if err := json.Unmarshal(raw, &object); err != nil {
		t.Fatalf("json.Unmarshal() error = %v; body = %s", err, raw)
	}
	return object
}

func decodeArray(t *testing.T, raw json.RawMessage) []map[string]json.RawMessage {
	t.Helper()
	var values []map[string]json.RawMessage
	if err := json.Unmarshal(raw, &values); err != nil {
		t.Fatalf("json.Unmarshal(array) error = %v; raw = %s", err, raw)
	}
	return values
}

func asString(t *testing.T, raw json.RawMessage) string {
	t.Helper()
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		t.Fatalf("json.Unmarshal(string) error = %v; raw = %s", err, raw)
	}
	return value
}

func baseModel() model.Model {
	return model.Model{
		Provider:  model.ProviderName("bedrock"),
		APIFormat: model.APIFormatBedrockConverse,
		BaseURL:   "https://bedrock-runtime.us-east-1.amazonaws.com",
		Name:      "anthropic.claude-sonnet-4-20250514-v1:0",
	}
}

func userMessage(blocks ...content.Block) *content.UserMessage {
	return &content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: blocks}}
}

func assistantMessage(blocks ...content.Block) *content.AIMessage {
	return &content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: blocks}}
}

func systemMessage(blocks ...content.Block) *content.SystemMessage {
	return &content.SystemMessage{Message: content.Message{Role: content.RoleSystem, Blocks: blocks}}
}

func toolResultMessage(id string, isError bool, blocks ...content.Block) *content.ToolResultMessage {
	return &content.ToolResultMessage{
		Message:   content.Message{Role: content.RoleTool, Blocks: blocks},
		ToolUseID: id,
		IsError:   isError,
	}
}

func TestEncodeRequest_SystemAndMessages(t *testing.T) {
	t.Parallel()

	req := inference.Request{
		Model:  baseModel(),
		System: "base instruction",
		Messages: content.AgenticMessages{
			systemMessage(&content.TextBlock{Text: "thread instruction"}),
			userMessage(&content.TextBlock{Text: "hello"}),
			assistantMessage(&content.TextBlock{Text: "hi"}),
		},
	}
	raw, err := bedrockconverse.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	body := decodeObject(t, raw)
	if _, ok := body["model"]; ok {
		t.Fatal("request body contains model; Bedrock takes model ID in the URL")
	}
	systems := decodeArray(t, body["system"])
	if len(systems) != 2 || asString(t, systems[0]["text"]) != "base instruction" || asString(t, systems[1]["text"]) != "thread instruction" {
		t.Fatalf("system = %#v, want two ordered text blocks", systems)
	}
	messages := decodeArray(t, body["messages"])
	if len(messages) != 2 || asString(t, messages[0]["role"]) != "user" || asString(t, messages[1]["role"]) != "assistant" {
		t.Fatalf("messages = %#v, want user then assistant", messages)
	}
	if got := asString(t, decodeArray(t, messages[0]["content"])[0]["text"]); got != "hello" {
		t.Errorf("user text = %q, want hello", got)
	}
}

func TestEncodeRequest_ImagesAndDocuments(t *testing.T) {
	t.Parallel()

	m := baseModel()
	m.Caps.AcceptsImages = true
	req := inference.Request{
		Model: m,
		Messages: content.AgenticMessages{userMessage(
			&content.ImageBlock{MediaType: content.MediaTypeImagePNG, Source: content.ImageSource{Data: []byte{1, 2, 3}}},
			&content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Name: "report-pdf", Data: []byte("pdf-bytes")},
			&content.DocumentBlock{MediaType: content.MediaTypeDocumentMarkdown, Name: "notes-md", Text: "# Notes"},
			&content.TextBlock{Text: "The attached documents contain the details."},
		)},
	}
	raw, err := bedrockconverse.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	messages := decodeArray(t, decodeObject(t, raw)["messages"])
	blocks := decodeArray(t, messages[0]["content"])
	var image map[string]json.RawMessage
	if err := json.Unmarshal(blocks[0]["image"], &image); err != nil {
		t.Fatalf("image block: %v", err)
	}
	if got := asString(t, image["format"]); got != "png" {
		t.Errorf("image.format = %q, want png", got)
	}
	var imageSource map[string]json.RawMessage
	if err := json.Unmarshal(image["source"], &imageSource); err != nil {
		t.Fatalf("image source: %v", err)
	}
	if got := asString(t, imageSource["bytes"]); got != base64.StdEncoding.EncodeToString([]byte{1, 2, 3}) {
		t.Errorf("image.source.bytes = %q, want base64 data", got)
	}

	var document map[string]json.RawMessage
	if err := json.Unmarshal(blocks[1]["document"], &document); err != nil {
		t.Fatalf("document block: %v", err)
	}
	if got := asString(t, document["format"]); got != "pdf" {
		t.Errorf("document.format = %q, want pdf", got)
	}
	if got := asString(t, document["name"]); got != "report-pdf" {
		t.Errorf("document.name = %q, want report-pdf", got)
	}
	var documentSource map[string]json.RawMessage
	if err := json.Unmarshal(document["source"], &documentSource); err != nil {
		t.Fatalf("document source: %v", err)
	}
	if got := asString(t, documentSource["bytes"]); got != base64.StdEncoding.EncodeToString([]byte("pdf-bytes")) {
		t.Errorf("document.source.bytes = %q, want base64 data", got)
	}

	var textDocument map[string]json.RawMessage
	if err := json.Unmarshal(blocks[2]["document"], &textDocument); err != nil {
		t.Fatalf("text document block: %v", err)
	}
	var textSource map[string]json.RawMessage
	if err := json.Unmarshal(textDocument["source"], &textSource); err != nil {
		t.Fatalf("text document source: %v", err)
	}
	if got := asString(t, textSource["text"]); got != "# Notes" {
		t.Errorf("document.source.text = %q, want # Notes", got)
	}
}

func TestEncodeRequest_ReasoningToolsAndToolResult(t *testing.T) {
	t.Parallel()

	m := baseModel()
	m.Caps.Thinking = true
	m.Caps.Tools = true
	req := inference.Request{
		Model: m,
		Messages: content.AgenticMessages{
			assistantMessage(
				content.NewThinkingBlock("consider", "sig", nil, ""),
				&content.ToolUseBlock{ID: "call-1", Name: "lookup", Input: json.RawMessage(`{"query":"go"}`)},
			),
			toolResultMessage("call-1", true, &content.TextBlock{Text: "not found"}),
		},
		Tools:      []inference.Tool{{Name: "lookup", Description: "Find a thing", Schema: json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}},"required":["query"],"additionalProperties":false}`)}},
		ToolChoice: inference.ToolChoiceRequired,
	}
	raw, err := bedrockconverse.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	body := decodeObject(t, raw)
	messages := decodeArray(t, body["messages"])
	assistantBlocks := decodeArray(t, messages[0]["content"])
	var reasoning map[string]json.RawMessage
	if err := json.Unmarshal(assistantBlocks[0]["reasoningContent"], &reasoning); err != nil {
		t.Fatalf("reasoningContent: %v", err)
	}
	var reasoningText map[string]json.RawMessage
	if err := json.Unmarshal(reasoning["reasoningText"], &reasoningText); err != nil {
		t.Fatalf("reasoningText: %v", err)
	}
	if got := asString(t, reasoningText["text"]); got != "consider" {
		t.Errorf("reasoning text = %q, want consider", got)
	}
	if got := asString(t, reasoningText["signature"]); got != "sig" {
		t.Errorf("reasoning signature = %q, want sig", got)
	}
	var toolUse map[string]json.RawMessage
	if err := json.Unmarshal(assistantBlocks[1]["toolUse"], &toolUse); err != nil {
		t.Fatalf("toolUse: %v", err)
	}
	if got := asString(t, toolUse["toolUseId"]); got != "call-1" {
		t.Errorf("toolUse.toolUseId = %q, want call-1", got)
	}

	resultBlocks := decodeArray(t, messages[1]["content"])
	var toolResult map[string]json.RawMessage
	if err := json.Unmarshal(resultBlocks[0]["toolResult"], &toolResult); err != nil {
		t.Fatalf("toolResult: %v", err)
	}
	if got := asString(t, toolResult["status"]); got != "error" {
		t.Errorf("toolResult.status = %q, want error", got)
	}

	var toolConfig map[string]json.RawMessage
	if err := json.Unmarshal(body["toolConfig"], &toolConfig); err != nil {
		t.Fatalf("toolConfig: %v", err)
	}
	choices := decodeArray(t, toolConfig["tools"])
	var spec map[string]json.RawMessage
	if err := json.Unmarshal(choices[0]["toolSpec"], &spec); err != nil {
		t.Fatalf("toolSpec: %v", err)
	}
	if got := asString(t, spec["name"]); got != "lookup" {
		t.Errorf("toolSpec.name = %q, want lookup", got)
	}
	var inputSchema map[string]json.RawMessage
	if err := json.Unmarshal(spec["inputSchema"], &inputSchema); err != nil {
		t.Fatalf("toolSpec.inputSchema: %v", err)
	}
	var schema map[string]json.RawMessage
	if err := json.Unmarshal(inputSchema["json"], &schema); err != nil {
		t.Fatalf("toolSpec.inputSchema.json: %v", err)
	}
	if got := asString(t, schema["type"]); got != "object" {
		t.Errorf("tool schema type = %q, want object", got)
	}
	var choice map[string]json.RawMessage
	if err := json.Unmarshal(toolConfig["toolChoice"], &choice); err != nil {
		t.Fatalf("toolChoice: %v", err)
	}
	if _, ok := choice["any"]; !ok {
		t.Errorf("toolChoice = %#v, want any", choice)
	}
}

func TestEncodeRequest_SamplingAndStructuredOutput(t *testing.T) {
	t.Parallel()

	m := baseModel()
	m.Caps.StructuredOutput = true
	sampling := model.Sampling{MaxTokens: intPtr(321), Temperature: floatPtr(0.2), TopP: floatPtr(0.8), Stop: []string{"END"}}
	req := inference.Request{
		Model:    m,
		Override: &sampling,
		Output: &inference.OutputSchema{
			Name:        "answer",
			Description: "The answer object",
			Schema:      json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}`),
		},
	}
	raw, err := bedrockconverse.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	body := decodeObject(t, raw)
	var inferenceConfig map[string]json.RawMessage
	if err := json.Unmarshal(body["inferenceConfig"], &inferenceConfig); err != nil {
		t.Fatalf("inferenceConfig: %v", err)
	}
	if got := asInt(t, inferenceConfig["maxTokens"]); got != 321 {
		t.Errorf("maxTokens = %d, want 321", got)
	}
	if got := asFloat(t, inferenceConfig["temperature"]); got != 0.2 {
		t.Errorf("temperature = %v, want 0.2", got)
	}
	if got := asFloat(t, inferenceConfig["topP"]); got != 0.8 {
		t.Errorf("topP = %v, want 0.8", got)
	}
	var stops []string
	if err := json.Unmarshal(inferenceConfig["stopSequences"], &stops); err != nil {
		t.Fatalf("stopSequences: %v", err)
	}
	if len(stops) != 1 || stops[0] != "END" {
		t.Fatalf("stopSequences = %#v, want [END]", stops)
	}

	var outputConfig map[string]json.RawMessage
	if err := json.Unmarshal(body["outputConfig"], &outputConfig); err != nil {
		t.Fatalf("outputConfig: %v", err)
	}
	var textFormat map[string]json.RawMessage
	if err := json.Unmarshal(outputConfig["textFormat"], &textFormat); err != nil {
		t.Fatalf("textFormat: %v", err)
	}
	if got := asString(t, textFormat["type"]); got != "json_schema" {
		t.Errorf("textFormat.type = %q, want json_schema", got)
	}
	var structure map[string]json.RawMessage
	if err := json.Unmarshal(textFormat["structure"], &structure); err != nil {
		t.Fatalf("structure: %v", err)
	}
	var schema map[string]json.RawMessage
	if err := json.Unmarshal(structure["jsonSchema"], &schema); err != nil {
		t.Fatalf("jsonSchema: %v", err)
	}
	if got := asString(t, schema["name"]); got != "answer" {
		t.Errorf("jsonSchema.name = %q, want answer", got)
	}
	if got := asString(t, schema["schema"]); got != string(req.Output.Schema) {
		t.Errorf("jsonSchema.schema = %q, want original schema string", got)
	}
}

func TestEncodeRequest_OmitsUnrequestedOptionalFields(t *testing.T) {
	t.Parallel()

	raw, err := bedrockconverse.EncodeRequest(inference.Request{Model: baseModel()})
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	body := decodeObject(t, raw)
	if len(body) != 1 {
		t.Fatalf("body keys = %#v, want only messages", body)
	}
	if _, ok := body["messages"]; !ok {
		t.Fatal("messages field omitted")
	}
}

func TestEncodeCountTokensInput_UsesOnlyConverseCountableFields(t *testing.T) {
	t.Parallel()

	m := baseModel()
	m.Caps.Tools = true
	m.Caps.StructuredOutput = true
	m.Caps.StructuredOutputWithTools = true
	maxTokens := 128
	req := inference.Request{
		Model:    m,
		System:   "count this system",
		Messages: content.AgenticMessages{userMessage(&content.TextBlock{Text: "hello"})},
		Tools:    []inference.Tool{{Name: "lookup", Schema: json.RawMessage(`{"type":"object","additionalProperties":false}`)}},
		Override: &model.Sampling{MaxTokens: &maxTokens, Temperature: floatPtr(0.1)},
		Output: &inference.OutputSchema{
			Name:   "answer",
			Schema: json.RawMessage(`{"type":"object","additionalProperties":false}`),
		},
	}
	raw, err := bedrockconverse.EncodeCountTokensInput(req)
	if err != nil {
		t.Fatalf("EncodeCountTokensInput() error = %v", err)
	}
	body := decodeObject(t, raw)
	for _, field := range []string{"messages", "system", "toolConfig"} {
		if _, ok := body[field]; !ok {
			t.Errorf("count input missing %q", field)
		}
	}
	for _, field := range []string{"inferenceConfig", "outputConfig", "model", "guardrailConfig", "serviceTier"} {
		if _, ok := body[field]; ok {
			t.Errorf("count input unexpectedly includes %q", field)
		}
	}
}

func TestCodecEncodeRequest_HeadersAndMode(t *testing.T) {
	t.Parallel()

	encoded, err := (bedrockconverse.Codec{}).EncodeRequest(inference.Request{Model: baseModel()}, codec.RequestModeStream)
	if err != nil {
		t.Fatalf("Codec.EncodeRequest() error = %v", err)
	}
	if got := encoded.Header.Get("Content-Type"); got != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", got)
	}
	body, err := io.ReadAll(encoded.Body)
	if err != nil {
		t.Fatalf("ReadAll() error = %v", err)
	}
	if strings.Contains(string(body), "stream") {
		t.Errorf("Converse request unexpectedly contains stream flag: %s", body)
	}
}

func TestEncodeRequest_RejectsUnsupportedAndMalformedTools(t *testing.T) {
	t.Parallel()

	m := baseModel()
	m.Caps.AcceptsImages = true
	cases := []struct {
		name string
		req  inference.Request
	}{
		{
			name: "audio",
			req:  inference.Request{Model: m, Messages: content.AgenticMessages{userMessage(&content.AudioBlock{MediaType: content.MediaTypeAudioMPEG, Data: []byte("audio")})}},
		},
		{
			name: "remote image",
			req:  inference.Request{Model: m, Messages: content.AgenticMessages{userMessage(&content.ImageBlock{MediaType: content.MediaTypeImagePNG, Source: content.ImageSource{URL: "https://example.test/image.png"}})}},
		},
		{
			name: "empty schema",
			req:  inference.Request{Model: m, Tools: []inference.Tool{{Name: "tool"}}},
		},
		{
			name: "malformed schema",
			req:  inference.Request{Model: m, Tools: []inference.Tool{{Name: "tool", Schema: json.RawMessage(`{"type":`)}}},
		},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := bedrockconverse.EncodeRequest(tc.req)
			if err == nil {
				t.Fatal("EncodeRequest() error = nil, want typed validation error")
			}
			var unsupported *bedrockconverse.UnsupportedBlockError
			var schemaErr *bedrockconverse.ToolSchemaError
			if tc.name == "audio" || tc.name == "remote image" {
				if !errors.As(err, &unsupported) {
					t.Fatalf("error = %T (%v), want UnsupportedBlockError", err, err)
				}
			} else if !errors.As(err, &schemaErr) {
				t.Fatalf("error = %T (%v), want ToolSchemaError", err, err)
			}
		})
	}
}

func TestEncodeRequest_EnforcesBedrockMessageRolesAndDocumentRules(t *testing.T) {
	t.Parallel()

	m := baseModel()
	m.Caps.AcceptsImages = true
	m.Caps.Thinking = true
	m.Caps.Tools = true
	cases := []struct {
		name string
		req  inference.Request
	}{
		{
			name: "assistant image",
			req:  inference.Request{Model: m, Messages: content.AgenticMessages{assistantMessage(&content.ImageBlock{MediaType: content.MediaTypeImagePNG, Source: content.ImageSource{Data: []byte{1}}})}},
		},
		{
			name: "user thinking",
			req:  inference.Request{Model: m, Messages: content.AgenticMessages{userMessage(&content.ThinkingBlock{Thinking: "think"})}},
		},
		{
			name: "assistant tool result",
			req:  inference.Request{Model: m, Messages: content.AgenticMessages{assistantMessage(&content.ToolResultBlock{ToolUseID: "call", Content: []content.Block{&content.TextBlock{Text: "result"}}})}},
		},
		{
			name: "document without text",
			req:  inference.Request{Model: m, Messages: content.AgenticMessages{userMessage(&content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Name: "report-pdf", Data: []byte("pdf")})}},
		},
		{
			name: "document name with period",
			req:  inference.Request{Model: m, Messages: content.AgenticMessages{userMessage(&content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Name: "report.pdf", Data: []byte("pdf")}, &content.TextBlock{Text: "attached"})}},
		},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := bedrockconverse.EncodeRequest(tc.req)
			var unsupported *bedrockconverse.UnsupportedBlockError
			if !errors.As(err, &unsupported) {
				t.Fatalf("EncodeRequest() error = %T (%v), want UnsupportedBlockError", err, err)
			}
		})
	}
}

func TestEncodeRequest_RejectsInvalidSharedFeatures(t *testing.T) {
	t.Parallel()

	_, err := bedrockconverse.EncodeRequest(inference.Request{
		Model: baseModel(),
		Output: &inference.OutputSchema{
			Name:   "answer",
			Schema: json.RawMessage(`{"type":"object","additionalProperties":false}`),
		},
	})
	var unsupported *inference.StructuredOutputUnsupportedError
	if !errors.As(err, &unsupported) {
		t.Fatalf("error = %T (%v), want StructuredOutputUnsupportedError", err, err)
	}
}

func intPtr(value int) *int           { return &value }
func floatPtr(value float64) *float64 { return &value }

func asInt(t *testing.T, raw json.RawMessage) int {
	t.Helper()
	var value int
	if err := json.Unmarshal(raw, &value); err != nil {
		t.Fatalf("json.Unmarshal(int) error = %v; raw = %s", err, raw)
	}
	return value
}

func asFloat(t *testing.T, raw json.RawMessage) float64 {
	t.Helper()
	var value float64
	if err := json.Unmarshal(raw, &value); err != nil {
		t.Fatalf("json.Unmarshal(float) error = %v; raw = %s", err, raw)
	}
	return value
}
