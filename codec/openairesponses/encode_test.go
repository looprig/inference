package openairesponses_test

import (
	"encoding/json"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/openairesponses"
	model "github.com/looprig/inference/model"
)

func decodeRequestBody(t *testing.T, body []byte) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("unmarshal encoded request: %v (body=%s)", err, body)
	}
	return m
}

func TestEncodeRequest_InstructionsFromSystem(t *testing.T) {
	t.Parallel()
	req := inference.Request{
		Model:  model.Model{Name: "gpt-test"},
		System: "be terse",
	}
	body, err := openairesponses.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	m := decodeRequestBody(t, body)
	if m["instructions"] != "be terse" {
		t.Errorf("instructions = %v, want %q", m["instructions"], "be terse")
	}
	if m["model"] != "gpt-test" {
		t.Errorf("model = %v", m["model"])
	}
	if m["store"] != false {
		t.Errorf("store = %v, want explicit false", m["store"])
	}
	if _, present := m["stream"]; present {
		t.Errorf("stream should be omitted when false, got %v", m["stream"])
	}
}

func TestEncodeRequest_InThreadSystemMessageFoldsIntoInstructions(t *testing.T) {
	t.Parallel()
	req := inference.Request{
		Model:  model.Model{Name: "gpt-test"},
		System: "base",
		Messages: content.AgenticMessages{
			&content.SystemMessage{Message: content.Message{
				Role:   content.RoleSystem,
				Blocks: []content.Block{&content.TextBlock{Text: "extra"}},
			}},
		},
	}
	body, err := openairesponses.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	m := decodeRequestBody(t, body)
	if m["instructions"] != "base\n\nextra" {
		t.Errorf("instructions = %v, want %q", m["instructions"], "base\n\nextra")
	}
	// The SystemMessage must not also appear as an input item.
	input, _ := m["input"].([]any)
	if len(input) != 0 {
		t.Errorf("input = %v, want empty (system folded into instructions)", input)
	}
}

func TestEncodeRequest_UserMessageText(t *testing.T) {
	t.Parallel()
	req := inference.Request{
		Model: model.Model{Name: "gpt-test"},
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{
				Role:   content.RoleUser,
				Blocks: []content.Block{&content.TextBlock{Text: "hello"}},
			}},
		},
	}
	body, err := openairesponses.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	m := decodeRequestBody(t, body)
	input := m["input"].([]any)
	if len(input) != 1 {
		t.Fatalf("input len = %d, want 1", len(input))
	}
	item := input[0].(map[string]any)
	if item["type"] != "message" || item["role"] != "user" {
		t.Errorf("item = %#v", item)
	}
	parts := item["content"].([]any)
	part := parts[0].(map[string]any)
	if part["type"] != "input_text" || part["text"] != "hello" {
		t.Errorf("part = %#v", part)
	}
}

func TestEncodeRequest_ImagesInlineAndURL(t *testing.T) {
	t.Parallel()
	req := inference.Request{
		Model: model.Model{Name: "gpt-test", Caps: model.Capabilities{AcceptsImages: true}},
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{
				Role: content.RoleUser,
				Blocks: []content.Block{
					&content.ImageBlock{MediaType: content.MediaTypeImagePNG, Source: content.ImageSource{Data: []byte{1, 2, 3}}},
					&content.ImageBlock{MediaType: content.MediaTypeImageJPEG, Source: content.ImageSource{URL: "https://example.com/x.jpg"}},
				},
			}},
		},
	}
	body, err := openairesponses.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	m := decodeRequestBody(t, body)
	input := m["input"].([]any)
	item := input[0].(map[string]any)
	parts := item["content"].([]any)
	if len(parts) != 2 {
		t.Fatalf("parts len = %d, want 2", len(parts))
	}
	inlinePart := parts[0].(map[string]any)
	if inlinePart["type"] != "input_image" {
		t.Errorf("inlinePart type = %v", inlinePart["type"])
	}
	inlineURL, _ := inlinePart["image_url"].(string)
	if inlineURL == "" || inlineURL[:5] != "data:" {
		t.Errorf("inline image_url = %q, want data: URI", inlineURL)
	}
	if inlinePart["detail"] != "auto" {
		t.Errorf("detail = %v, want auto", inlinePart["detail"])
	}
	urlPart := parts[1].(map[string]any)
	if urlPart["image_url"] != "https://example.com/x.jpg" {
		t.Errorf("urlPart image_url = %v", urlPart["image_url"])
	}
}

func TestEncodeRequest_ToolsCallsAndOutputs(t *testing.T) {
	t.Parallel()
	req := inference.Request{
		Model: model.Model{Name: "gpt-test", Caps: model.Capabilities{Tools: true}},
		Tools: []inference.Tool{{
			Name:        "get_weather",
			Description: "gets weather",
			Schema:      json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
		}},
		Messages: content.AgenticMessages{
			&content.AIMessage{Message: content.Message{
				Role: content.RoleAssistant,
				Blocks: []content.Block{
					&content.ToolUseBlock{ID: "call_1", Name: "get_weather", Input: json.RawMessage(`{"city":"nyc"}`)},
				},
			}},
			&content.ToolResultMessage{
				Message:   content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "sunny"}}},
				ToolUseID: "call_1",
			},
		},
	}
	body, err := openairesponses.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	m := decodeRequestBody(t, body)

	tools := m["tools"].([]any)
	tool := tools[0].(map[string]any)
	if tool["type"] != "function" || tool["name"] != "get_weather" {
		t.Errorf("tool = %#v", tool)
	}

	input := m["input"].([]any)
	if len(input) != 2 {
		t.Fatalf("input len = %d, want 2", len(input))
	}
	call := input[0].(map[string]any)
	if call["type"] != "function_call" || call["call_id"] != "call_1" || call["name"] != "get_weather" {
		t.Errorf("call = %#v", call)
	}
	// arguments MUST be a JSON-encoded string, not a bare object.
	argsStr, ok := call["arguments"].(string)
	if !ok {
		t.Fatalf("arguments type = %T, want string", call["arguments"])
	}
	var argsObj map[string]any
	if err := json.Unmarshal([]byte(argsStr), &argsObj); err != nil {
		t.Fatalf("arguments not valid JSON: %v", err)
	}
	if argsObj["city"] != "nyc" {
		t.Errorf("arguments city = %v", argsObj["city"])
	}

	result := input[1].(map[string]any)
	if result["type"] != "function_call_output" || result["call_id"] != "call_1" || result["output"] != "sunny" {
		t.Errorf("result = %#v", result)
	}
}

func TestEncodeRequest_ParallelToolCalls(t *testing.T) {
	t.Parallel()
	req := inference.Request{
		Model: model.Model{Name: "gpt-test", Caps: model.Capabilities{Tools: true}},
		Messages: content.AgenticMessages{
			&content.AIMessage{Message: content.Message{
				Role: content.RoleAssistant,
				Blocks: []content.Block{
					&content.ToolUseBlock{ID: "call_1", Name: "get_weather", Input: json.RawMessage(`{}`)},
					&content.ToolUseBlock{ID: "call_2", Name: "get_time", Input: json.RawMessage(`{}`)},
				},
			}},
		},
	}
	body, err := openairesponses.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	m := decodeRequestBody(t, body)
	input := m["input"].([]any)
	if len(input) != 2 {
		t.Fatalf("input len = %d, want 2", len(input))
	}
	first := input[0].(map[string]any)
	second := input[1].(map[string]any)
	if first["call_id"] != "call_1" || second["call_id"] != "call_2" {
		t.Errorf("call_ids = %v, %v", first["call_id"], second["call_id"])
	}
}

func TestEncodeRequest_ToolChoiceRequired(t *testing.T) {
	t.Parallel()
	req := inference.Request{
		Model:      model.Model{Name: "gpt-test", Caps: model.Capabilities{Tools: true}},
		Tools:      []inference.Tool{{Name: "f", Schema: json.RawMessage(`{"type":"object"}`)}},
		ToolChoice: inference.ToolChoiceRequired,
	}
	body, err := openairesponses.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	m := decodeRequestBody(t, body)
	if m["tool_choice"] != "required" {
		t.Errorf("tool_choice = %v, want required", m["tool_choice"])
	}
}

func TestEncodeRequest_ToolChoiceAutoOmitted(t *testing.T) {
	t.Parallel()
	req := inference.Request{Model: model.Model{Name: "gpt-test"}}
	body, err := openairesponses.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	m := decodeRequestBody(t, body)
	if _, present := m["tool_choice"]; present {
		t.Errorf("tool_choice should be omitted for auto, got %v", m["tool_choice"])
	}
}

func TestEncodeRequest_ReasoningSummaryAndInclude(t *testing.T) {
	t.Parallel()
	low := model.EffortLow
	req := inference.Request{
		Model:    model.Model{Name: "gpt-test", Caps: model.Capabilities{Thinking: true}},
		Override: &model.Sampling{Effort: low},
	}
	body, err := openairesponses.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	m := decodeRequestBody(t, body)
	reasoning, ok := m["reasoning"].(map[string]any)
	if !ok {
		t.Fatalf("reasoning missing or wrong type: %#v", m["reasoning"])
	}
	if reasoning["effort"] != "low" {
		t.Errorf("effort = %v", reasoning["effort"])
	}
	if reasoning["summary"] != "auto" {
		t.Errorf("summary = %v", reasoning["summary"])
	}
	include, ok := m["include"].([]any)
	if !ok || len(include) != 1 || include[0] != "reasoning.encrypted_content" {
		t.Errorf("include = %#v, want [reasoning.encrypted_content]", m["include"])
	}
}

func TestEncodeRequest_ReasoningOmittedWithoutThinkingCapability(t *testing.T) {
	t.Parallel()
	req := inference.Request{
		Model:    model.Model{Name: "gpt-test"}, // Caps.Thinking false
		Override: &model.Sampling{Effort: model.EffortLow},
	}
	body, err := openairesponses.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	m := decodeRequestBody(t, body)
	if _, present := m["reasoning"]; present {
		t.Errorf("reasoning should be omitted without Thinking capability, got %v", m["reasoning"])
	}
	if _, present := m["include"]; present {
		t.Errorf("include should be omitted without Thinking capability, got %v", m["include"])
	}
}

func TestEncodeRequest_EncryptedReasoningReplay(t *testing.T) {
	t.Parallel()
	providerState := json.RawMessage(`"opaque-blob-abc123"`)
	req := inference.Request{
		Model: model.Model{Name: "gpt-test", Caps: model.Capabilities{Thinking: true}},
		Messages: content.AgenticMessages{
			&content.AIMessage{Message: content.Message{
				Role: content.RoleAssistant,
				Blocks: []content.Block{
					content.NewThinkingBlock("step by step", "", providerState, "openai-responses"),
				},
			}},
		},
	}
	body, err := openairesponses.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	m := decodeRequestBody(t, body)
	input := m["input"].([]any)
	item := input[0].(map[string]any)
	if item["type"] != "reasoning" {
		t.Fatalf("item type = %v, want reasoning", item["type"])
	}
	if item["encrypted_content"] != "opaque-blob-abc123" {
		t.Errorf("encrypted_content = %v", item["encrypted_content"])
	}
	summary := item["summary"].([]any)
	part := summary[0].(map[string]any)
	if part["type"] != "summary_text" || part["text"] != "step by step" {
		t.Errorf("summary part = %#v", part)
	}
}

func TestEncodeRequest_OutputSchema(t *testing.T) {
	t.Parallel()
	req := inference.Request{
		Model: model.Model{Name: "gpt-test", Caps: model.Capabilities{StructuredOutput: true}},
		Output: &inference.OutputSchema{
			Name:   "answer",
			Strict: true,
			Schema: json.RawMessage(`{"type":"object","properties":{"x":{"type":"string"}},"required":["x"],"additionalProperties":false}`),
		},
	}
	body, err := openairesponses.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	m := decodeRequestBody(t, body)
	text := m["text"].(map[string]any)
	format := text["format"].(map[string]any)
	if format["type"] != "json_schema" || format["name"] != "answer" || format["strict"] != true {
		t.Errorf("format = %#v", format)
	}
}

func TestEncodeRequest_Streaming(t *testing.T) {
	t.Parallel()
	req := inference.Request{Model: model.Model{Name: "gpt-test"}}
	body, err := openairesponses.EncodeRequest(req, true)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	m := decodeRequestBody(t, body)
	if m["stream"] != true {
		t.Errorf("stream = %v, want true", m["stream"])
	}
}

func TestEncodeRequest_StopSequencesOmitted(t *testing.T) {
	t.Parallel()
	// Responses has no wire "stop" field (the real API rejects it); this
	// codec must never emit one even when the caller sets Sampling.Stop.
	req := inference.Request{
		Model:    model.Model{Name: "gpt-test"},
		Override: &model.Sampling{Stop: []string{"END"}},
	}
	body, err := openairesponses.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	m := decodeRequestBody(t, body)
	if _, present := m["stop"]; present {
		t.Errorf("stop should never be emitted, got %v", m["stop"])
	}
}

func TestEncodeRequest_AssistantTextReplay(t *testing.T) {
	t.Parallel()
	req := inference.Request{
		Model: model.Model{Name: "gpt-test"},
		Messages: content.AgenticMessages{
			&content.AIMessage{Message: content.Message{
				Role:   content.RoleAssistant,
				Blocks: []content.Block{&content.TextBlock{Text: "prior answer"}},
			}},
		},
	}
	body, err := openairesponses.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	m := decodeRequestBody(t, body)
	input := m["input"].([]any)
	item := input[0].(map[string]any)
	if item["type"] != "message" || item["role"] != "assistant" {
		t.Fatalf("item = %#v", item)
	}
	parts := item["content"].([]any)
	part := parts[0].(map[string]any)
	if part["type"] != "output_text" || part["text"] != "prior answer" {
		t.Errorf("part = %#v", part)
	}
}

func TestEncodeRequest_UnsupportedConversation(t *testing.T) {
	t.Parallel()
	req := inference.Request{
		Model:    model.Model{Name: "gpt-test"},
		Messages: content.AgenticMessages{nil},
	}
	if _, err := openairesponses.EncodeRequest(req, false); err == nil {
		t.Fatal("EncodeRequest() error = nil, want error for unsupported conversation")
	}
}
