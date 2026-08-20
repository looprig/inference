package anthropicapi_test

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/anthropicapi"
	model "github.com/looprig/inference/model"
)

// --- shared helpers -------------------------------------------------------

func f64ptr(v float64) *float64 { return &v }
func intptr(v int) *int         { return &v }

// decodeObj unmarshals raw JSON into a field-addressable map.
func decodeObj(t *testing.T, data []byte) map[string]json.RawMessage {
	t.Helper()
	var m map[string]json.RawMessage
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatalf("unmarshal object: %v (%s)", err, data)
	}
	return m
}

func asString(t *testing.T, raw json.RawMessage) string {
	t.Helper()
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatalf("unmarshal string: %v (%s)", err, raw)
	}
	return s
}

func asInt(t *testing.T, raw json.RawMessage) int {
	t.Helper()
	var n int
	if err := json.Unmarshal(raw, &n); err != nil {
		t.Fatalf("unmarshal int: %v (%s)", err, raw)
	}
	return n
}

func asObjs(t *testing.T, raw json.RawMessage) []map[string]json.RawMessage {
	t.Helper()
	var objs []map[string]json.RawMessage
	if err := json.Unmarshal(raw, &objs); err != nil {
		t.Fatalf("unmarshal array: %v (%s)", err, raw)
	}
	return objs
}

func messagesOf(t *testing.T, body map[string]json.RawMessage) []map[string]json.RawMessage {
	t.Helper()
	return asObjs(t, body["messages"])
}

func blocksOf(t *testing.T, msg map[string]json.RawMessage) []map[string]json.RawMessage {
	t.Helper()
	return asObjs(t, msg["content"])
}

func userMsg(blocks ...content.Block) *content.UserMessage {
	return &content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: blocks}}
}

func aiMsg(blocks ...content.Block) *content.AIMessage {
	return &content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: blocks}}
}

func sysMsg(blocks ...content.Block) *content.SystemMessage {
	return &content.SystemMessage{Message: content.Message{Role: content.RoleSystem, Blocks: blocks}}
}

func toolResultMsg(id string, isErr bool, blocks ...content.Block) *content.ToolResultMessage {
	return &content.ToolResultMessage{
		Message:   content.Message{Role: content.RoleTool, Blocks: blocks},
		ToolUseID: id,
		IsError:   isErr,
	}
}

func textBlock(s string) content.Block { return &content.TextBlock{Text: s} }

func structuredOutput() *inference.OutputSchema {
	return &inference.OutputSchema{
		Name:        "answer",
		Description: "One answer",
		Strict:      true,
		Schema:      json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}`),
	}
}

func structuredModel(withTools bool) model.Model {
	m := baseModel()
	m.Caps.StructuredOutput = true
	if withTools {
		m.Caps.Tools = true
		m.Caps.StructuredOutputWithTools = true
	}
	return m
}

func TestEncodeRequestIgnoresUsage(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		usage *content.Usage
	}{
		{name: "nil", usage: nil},
		{name: "present zero", usage: &content.Usage{}},
		{name: "populated", usage: &content.Usage{InputTokens: 1, OutputTokens: 2, CacheReadTokens: 3, CacheCreationTokens: 4, ReasoningTokens: 1}},
	}

	want, err := anthropicapi.EncodeRequest(inference.Request{Model: baseModel(), Messages: content.AgenticMessages{aiMsg(textBlock("answer"))}}, false)
	if err != nil {
		t.Fatalf("EncodeRequest() baseline error = %v", err)
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			message := aiMsg(textBlock("answer"))
			message.Usage = tt.usage
			got, err := anthropicapi.EncodeRequest(inference.Request{Model: baseModel(), Messages: content.AgenticMessages{message}}, false)
			if err != nil {
				t.Fatalf("EncodeRequest() error = %v", err)
			}
			if string(got) != string(want) {
				t.Errorf("EncodeRequest() = %s, want byte-identical %s", got, want)
			}
		})
	}
}

func imageURLBlock(url string) content.Block {
	return &content.ImageBlock{MediaType: content.MediaTypeImageJPEG, Source: content.ImageSource{URL: url}}
}

func imageDataBlock(mt content.MediaType, data []byte) content.Block {
	return &content.ImageBlock{MediaType: mt, Source: content.ImageSource{Data: data}}
}

// baseModel is a minimal valid Model for encode tests. Caps default off.
func baseModel() model.Model {
	return model.Model{
		Provider:  model.ProviderName("phala"),
		APIFormat: model.APIFormatAnthropic,
		BaseURL:   "https://example.test",
		Name:      "claude-opus-4-8",
	}
}

// --- TestEncodeRequest_SystemAndMessages ----------------------------------

func TestEncodeRequest_SystemAndMessages(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name          string
		system        string
		messages      content.AgenticMessages
		wantSystem    string // "" means the field must be absent
		wantSystemKey bool
		wantRoles     []string
	}{
		{
			name:          "system + user + assistant",
			system:        "You are helpful.",
			messages:      content.AgenticMessages{userMsg(textBlock("hi")), aiMsg(textBlock("hello"))},
			wantSystem:    "You are helpful.",
			wantSystemKey: true,
			wantRoles:     []string{"user", "assistant"},
		},
		{
			name:          "empty system omits the field",
			system:        "",
			messages:      content.AgenticMessages{userMsg(textBlock("hi"))},
			wantSystemKey: false,
			wantRoles:     []string{"user"},
		},
		{
			name:          "in-thread SystemMessage folds into top-level system",
			system:        "Base.",
			messages:      content.AgenticMessages{sysMsg(textBlock("More.")), userMsg(textBlock("hi"))},
			wantSystem:    "Base.\n\nMore.",
			wantSystemKey: true,
			wantRoles:     []string{"user"}, // system message does not become a wire message
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			data, err := anthropicapi.EncodeRequest(inference.Request{Model: baseModel(), System: tc.system, Messages: tc.messages}, false)
			if err != nil {
				t.Fatalf("EncodeRequest: %v", err)
			}
			body := decodeObj(t, data)

			if got := asString(t, body["model"]); got != "claude-opus-4-8" {
				t.Errorf("model = %q, want claude-opus-4-8", got)
			}
			_, hasSystem := body["system"]
			if hasSystem != tc.wantSystemKey {
				t.Errorf("system present = %v, want %v", hasSystem, tc.wantSystemKey)
			}
			if tc.wantSystemKey && asString(t, body["system"]) != tc.wantSystem {
				t.Errorf("system = %q, want %q", asString(t, body["system"]), tc.wantSystem)
			}

			msgs := messagesOf(t, body)
			if len(msgs) != len(tc.wantRoles) {
				t.Fatalf("message count = %d, want %d", len(msgs), len(tc.wantRoles))
			}
			for i, wantRole := range tc.wantRoles {
				if got := asString(t, msgs[i]["role"]); got != wantRole {
					t.Errorf("message[%d].role = %q, want %q", i, got, wantRole)
				}
			}
		})
	}
}

func TestEncodeRequest_ProjectsConversationTurns(t *testing.T) {
	t.Parallel()

	m := baseModel()
	m.Caps.Tools = true

	t.Run("adjacent ordinary user messages merge in order", func(t *testing.T) {
		t.Parallel()
		data, err := anthropicapi.EncodeRequest(inference.Request{
			Model: m,
			Messages: content.AgenticMessages{
				userMsg(textBlock("first")),
				userMsg(textBlock("second")),
			},
		}, false)
		if err != nil {
			t.Fatalf("EncodeRequest: %v", err)
		}
		messages := messagesOf(t, decodeObj(t, data))
		if len(messages) != 1 || asString(t, messages[0]["role"]) != "user" {
			t.Fatalf("messages = %v, want one user turn", messages)
		}
		blocks := blocksOf(t, messages[0])
		if len(blocks) != 2 || asString(t, blocks[0]["text"]) != "first" || asString(t, blocks[1]["text"]) != "second" {
			t.Fatalf("blocks = %v, want first then second", blocks)
		}
	})

	t.Run("parallel tool results share one user turn", func(t *testing.T) {
		t.Parallel()
		data, err := anthropicapi.EncodeRequest(inference.Request{
			Model: m,
			Messages: content.AgenticMessages{
				userMsg(textBlock("start")),
				aiMsg(
					&content.ToolUseBlock{ID: "toolu_1", Name: "lookup", Input: json.RawMessage(`{"q":"one"}`)},
					&content.ToolUseBlock{ID: "toolu_2", Name: "lookup", Input: json.RawMessage(`{"q":"two"}`)},
				),
				toolResultMsg("toolu_1", false, textBlock("one")),
				toolResultMsg("toolu_2", false, textBlock("two")),
			},
		}, false)
		if err != nil {
			t.Fatalf("EncodeRequest: %v", err)
		}
		messages := messagesOf(t, decodeObj(t, data))
		if len(messages) != 3 {
			t.Fatalf("len(messages) = %d, want 3", len(messages))
		}
		blocks := blocksOf(t, messages[2])
		if len(blocks) != 2 || asString(t, blocks[0]["tool_use_id"]) != "toolu_1" || asString(t, blocks[1]["tool_use_id"]) != "toolu_2" {
			t.Fatalf("tool-result blocks = %v, want both results in call order", blocks)
		}
	})

	t.Run("transient runtime text follows tool results without another user turn", func(t *testing.T) {
		t.Parallel()
		data, err := anthropicapi.EncodeRequest(inference.Request{
			Model: m,
			Messages: content.AgenticMessages{
				userMsg(textBlock("start")),
				aiMsg(&content.ToolUseBlock{ID: "toolu_1", Name: "lookup", Input: json.RawMessage(`{"q":"one"}`)}),
				toolResultMsg("toolu_1", false, textBlock("one")),
				userMsg(textBlock("runtime context")),
			},
			TransientMessages: 1,
		}, false)
		if err != nil {
			t.Fatalf("EncodeRequest: %v", err)
		}
		messages := messagesOf(t, decodeObj(t, data))
		if len(messages) != 3 {
			t.Fatalf("len(messages) = %d, want 3", len(messages))
		}
		blocks := blocksOf(t, messages[2])
		if len(blocks) != 2 || asString(t, blocks[0]["type"]) != "tool_result" || asString(t, blocks[1]["type"]) != "text" {
			t.Fatalf("final user blocks = %v, want tool_result then runtime text", blocks)
		}
		contentBlocks := asObjs(t, blocks[0]["content"])
		if len(contentBlocks) != 1 || asString(t, contentBlocks[0]["text"]) != "one" {
			t.Fatalf("tool_result content = %v, want only the tool's result", contentBlocks)
		}
		if got := asString(t, blocks[1]["text"]); got != "runtime context" {
			t.Fatalf("runtime text = %q, want runtime context", got)
		}
	})

	t.Run("ordinary user content before a tool result fails locally", func(t *testing.T) {
		t.Parallel()
		_, err := anthropicapi.EncodeRequest(inference.Request{
			Model: m,
			Messages: content.AgenticMessages{
				userMsg(textBlock("ordinary user content")),
				toolResultMsg("toolu_1", false, textBlock("result")),
			},
		}, false)
		if err == nil {
			t.Fatal("EncodeRequest accepted text before tool_result in one user turn")
		}
	})
}

// --- TestEncodeRequest_Tools ----------------------------------------------

func TestEncodeRequest_Tools(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name       string
		tools      []inference.Tool
		wantTools  bool
		wantName   string
		wantDesc   string
		wantSchema string
	}{
		{
			name: "tool with schema maps name/description/input_schema",
			tools: []inference.Tool{{
				Name:        "get_weather",
				Description: "Look up weather",
				Schema:      json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
			}},
			wantTools:  true,
			wantName:   "get_weather",
			wantDesc:   "Look up weather",
			wantSchema: `{"type":"object","properties":{"city":{"type":"string"}}}`,
		},
		{
			name:       "tool with empty schema defaults to object schema",
			tools:      []inference.Tool{{Name: "noargs"}},
			wantTools:  true,
			wantName:   "noargs",
			wantSchema: `{"type":"object"}`,
		},
		{
			name:      "no tools omits the tools field",
			tools:     nil,
			wantTools: false,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			req := inference.Request{Model: baseModel(), Messages: content.AgenticMessages{userMsg(textBlock("hi"))}, Tools: tc.tools}
			data, err := anthropicapi.EncodeRequest(req, false)
			if err != nil {
				t.Fatalf("EncodeRequest: %v", err)
			}
			body := decodeObj(t, data)
			_, hasTools := body["tools"]
			if hasTools != tc.wantTools {
				t.Fatalf("tools present = %v, want %v", hasTools, tc.wantTools)
			}
			if !tc.wantTools {
				return
			}
			tools := asObjs(t, body["tools"])
			if len(tools) != 1 {
				t.Fatalf("tool count = %d, want 1", len(tools))
			}
			if got := asString(t, tools[0]["name"]); got != tc.wantName {
				t.Errorf("tool name = %q, want %q", got, tc.wantName)
			}
			if tc.wantDesc != "" && asString(t, tools[0]["description"]) != tc.wantDesc {
				t.Errorf("tool description = %q, want %q", asString(t, tools[0]["description"]), tc.wantDesc)
			}
			if got := string(tools[0]["input_schema"]); got != tc.wantSchema {
				t.Errorf("tool input_schema = %s, want %s", got, tc.wantSchema)
			}
		})
	}
}

func TestEncodeRequest_StructuredOutput(t *testing.T) {
	t.Parallel()

	m := structuredModel(true)
	m.Caps.Thinking = true
	m.Caps.ThinkingDialect = model.ThinkingDialectAdaptive
	m.Sampling.Effort = model.EffortHigh
	req := inference.Request{
		Model:      m,
		Messages:   content.AgenticMessages{userMsg(textBlock("hi"))},
		Output:     structuredOutput(),
		ToolChoice: inference.ToolRequired(),
		Tools: []inference.Tool{{
			Name:        "lookup",
			Description: "Look something up",
			Schema:      json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`),
		}},
	}

	data, err := anthropicapi.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	body := decodeObj(t, data)
	outputConfig := decodeObj(t, body["output_config"])
	if got := asString(t, outputConfig["effort"]); got != "high" {
		t.Errorf("output_config.effort = %q, want high", got)
	}
	format := decodeObj(t, outputConfig["format"])
	if len(format) != 2 {
		t.Errorf("output_config.format fields = %v, want only type and schema", format)
	}
	if got := asString(t, format["type"]); got != "json_schema" {
		t.Errorf("output_config.format.type = %q, want json_schema", got)
	}
	if got, want := string(format["schema"]), string(req.Output.Schema); got != want {
		t.Errorf("output_config.format.schema = %s, want %s", got, want)
	}

	tools := asObjs(t, body["tools"])
	if len(tools) != 1 || asString(t, tools[0]["name"]) != "lookup" {
		t.Fatalf("tools = %s, want the one ordinary lookup tool", body["tools"])
	}
	toolChoice := decodeObj(t, body["tool_choice"])
	if got := asString(t, toolChoice["type"]); got != "any" {
		t.Errorf("tool_choice.type = %q, want any", got)
	}
}

func TestEncodeRequest_StructuredOutputWithoutEffort(t *testing.T) {
	t.Parallel()

	data, err := anthropicapi.EncodeRequest(inference.Request{
		Model:  structuredModel(false),
		Output: structuredOutput(),
	}, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	outputConfig := decodeObj(t, decodeObj(t, data)["output_config"])
	if _, ok := outputConfig["format"]; !ok {
		t.Fatal("output_config.format is absent without effort")
	}
	if _, ok := outputConfig["effort"]; ok {
		t.Error("output_config.effort is present when effort is unset")
	}
}

func TestEncodeRequest_StructuredOutputFeatureValidation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		req     inference.Request
		wantErr func(error) bool
	}{
		{
			name: "structured output capability is required",
			req:  inference.Request{Model: baseModel(), Output: structuredOutput()},
			wantErr: func(err error) bool {
				var target *inference.StructuredOutputUnsupportedError
				return errors.As(err, &target)
			},
		},
		{
			name: "combined capability is required with tools",
			req: inference.Request{
				Model:  structuredModel(false),
				Output: structuredOutput(),
				Tools:  []inference.Tool{{Name: "lookup"}},
			},
			wantErr: func(err error) bool {
				var target *inference.StructuredOutputWithToolsUnsupportedError
				return errors.As(err, &target)
			},
		},
		{
			name: "invalid schema returns schema validation error before marshal",
			req: inference.Request{
				Model: structuredModel(false),
				Output: &inference.OutputSchema{
					Name:   "answer",
					Schema: json.RawMessage(`{"type":"object"`),
				},
			},
			wantErr: func(err error) bool {
				var target *inference.SchemaValidationError
				return errors.As(err, &target)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			_, err := anthropicapi.EncodeRequest(tt.req, false)
			if !tt.wantErr(err) {
				t.Fatalf("EncodeRequest() error = %T %v, want typed feature-validation error", err, err)
			}
		})
	}
}

func TestEncodeRequest_NilStructuredOutputPreservesWireShape(t *testing.T) {
	t.Parallel()

	got, err := anthropicapi.EncodeRequest(inference.Request{Model: baseModel()}, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	const want = `{"model":"claude-opus-4-8","messages":null,"max_tokens":4096}`
	if string(got) != want {
		t.Errorf("EncodeRequest() = %s, want byte-identical prior shape %s", got, want)
	}
}

func TestEncodeRequest_ToolChoiceAutoIsOmitted(t *testing.T) {
	t.Parallel()

	got, err := anthropicapi.EncodeRequest(inference.Request{
		Model:  structuredModel(false),
		Output: structuredOutput(),
	}, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	if _, ok := decodeObj(t, got)["tool_choice"]; ok {
		t.Error("tool_choice is present for ToolChoiceAuto")
	}
}

// TestEncodeRequest_ToolChoiceNamedTool pins the named tool-choice variant:
// Anthropic's ToolChoiceTool is `{"type":"tool","name":...}` with both
// properties required and additionalProperties:false, so the object carries
// exactly those two keys.
func TestEncodeRequest_ToolChoiceNamedTool(t *testing.T) {
	t.Parallel()

	m := structuredModel(false)
	m.Caps.Tools = true
	got, err := anthropicapi.EncodeRequest(inference.Request{
		Model:    m,
		Messages: content.AgenticMessages{userMsg(textBlock("hi"))},
		Tools: []inference.Tool{
			{Name: "lookup", Schema: json.RawMessage(`{"type":"object"}`)},
			{Name: "search", Schema: json.RawMessage(`{"type":"object"}`)},
		},
		ToolChoice: inference.ToolNamed("search"),
	}, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	toolChoice := decodeObj(t, decodeObj(t, got)["tool_choice"])
	if len(toolChoice) != 2 {
		t.Errorf("tool_choice fields = %v, want only type and name", toolChoice)
	}
	if choiceType := asString(t, toolChoice["type"]); choiceType != "tool" {
		t.Errorf("tool_choice.type = %q, want tool", choiceType)
	}
	if name := asString(t, toolChoice["name"]); name != "search" {
		t.Errorf("tool_choice.name = %q, want search", name)
	}
}

// --- TestEncodeRequest_Images ---------------------------------------------

func TestEncodeRequest_Images(t *testing.T) {
	t.Parallel()

	raw := []byte{0x1, 0x2, 0x3, 0x4}
	cases := []struct {
		name          string
		block         content.Block
		wantType      string
		wantMediaType string
		wantURL       string
		wantData      string
	}{
		{
			name:     "url image maps to a url source",
			block:    imageURLBlock("https://img.test/a.jpg"),
			wantType: "url",
			wantURL:  "https://img.test/a.jpg",
		},
		{
			name:          "inline data image maps to a base64 source",
			block:         imageDataBlock(content.MediaTypeImagePNG, raw),
			wantType:      "base64",
			wantMediaType: "image/png",
			wantData:      base64.StdEncoding.EncodeToString(raw),
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			m := baseModel()
			m.Caps.AcceptsImages = true
			req := inference.Request{Model: m, Messages: content.AgenticMessages{userMsg(tc.block)}}
			data, err := anthropicapi.EncodeRequest(req, false)
			if err != nil {
				t.Fatalf("EncodeRequest: %v", err)
			}
			body := decodeObj(t, data)
			blocks := blocksOf(t, messagesOf(t, body)[0])
			if len(blocks) != 1 {
				t.Fatalf("block count = %d, want 1", len(blocks))
			}
			if got := asString(t, blocks[0]["type"]); got != "image" {
				t.Fatalf("block type = %q, want image", got)
			}
			source := decodeObj(t, blocks[0]["source"])
			if got := asString(t, source["type"]); got != tc.wantType {
				t.Errorf("source.type = %q, want %q", got, tc.wantType)
			}
			if tc.wantURL != "" && asString(t, source["url"]) != tc.wantURL {
				t.Errorf("source.url = %q, want %q", asString(t, source["url"]), tc.wantURL)
			}
			if tc.wantMediaType != "" && asString(t, source["media_type"]) != tc.wantMediaType {
				t.Errorf("source.media_type = %q, want %q", asString(t, source["media_type"]), tc.wantMediaType)
			}
			if tc.wantData != "" && asString(t, source["data"]) != tc.wantData {
				t.Errorf("source.data = %q, want %q", asString(t, source["data"]), tc.wantData)
			}
		})
	}
}

// --- TestEncodeRequest_EffortThinking -------------------------------------

func TestEncodeRequest_EffortThinking(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name         string
		thinkingCap  bool
		effort       model.Effort
		wantThinking bool
		wantEffort   string // "" when output_config must be absent
	}{
		{name: "thinking-capable + high effort emits adaptive + effort", thinkingCap: true, effort: model.EffortHigh, wantThinking: true, wantEffort: "high"},
		{name: "thinking-capable + xhigh effort maps to xhigh", thinkingCap: true, effort: model.EffortXHigh, wantThinking: true, wantEffort: "xhigh"},
		{name: "thinking-capable + max effort maps to max", thinkingCap: true, effort: model.EffortMax, wantThinking: true, wantEffort: "max"},
		{name: "thinking-capable + low effort maps to low", thinkingCap: true, effort: model.EffortLow, wantThinking: true, wantEffort: "low"},
		{name: "thinking-capable + no effort emits neither", thinkingCap: true, effort: model.EffortNone, wantThinking: false},
		{name: "not thinking-capable ignores effort", thinkingCap: false, effort: model.EffortHigh, wantThinking: false},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			m := baseModel()
			m.Caps.Thinking = tc.thinkingCap
			// This table measures the effort→wire mapping of ONE dialect; which
			// dialect a model gets, and what an undeclared one does, is
			// encode_thinking_dialect_test.go's subject.
			m.Caps.ThinkingDialect = model.ThinkingDialectAdaptive
			m.Sampling = model.Sampling{Effort: tc.effort}
			req := inference.Request{Model: m, Messages: content.AgenticMessages{userMsg(textBlock("hi"))}}
			data, err := anthropicapi.EncodeRequest(req, false)
			if err != nil {
				t.Fatalf("EncodeRequest: %v", err)
			}
			body := decodeObj(t, data)

			_, hasThinking := body["thinking"]
			if hasThinking != tc.wantThinking {
				t.Errorf("thinking present = %v, want %v", hasThinking, tc.wantThinking)
			}
			if tc.wantThinking {
				th := decodeObj(t, body["thinking"])
				if got := asString(t, th["type"]); got != "adaptive" {
					t.Errorf("thinking.type = %q, want adaptive", got)
				}
			}

			_, hasOC := body["output_config"]
			wantOC := tc.wantEffort != ""
			if hasOC != wantOC {
				t.Errorf("output_config present = %v, want %v", hasOC, wantOC)
			}
			if wantOC {
				oc := decodeObj(t, body["output_config"])
				if got := asString(t, oc["effort"]); got != tc.wantEffort {
					t.Errorf("output_config.effort = %q, want %q", got, tc.wantEffort)
				}
			}
		})
	}
}

func TestEncodeRequest_RejectsMinimalEffort(t *testing.T) {
	t.Parallel()
	m := baseModel()
	m.Caps.Thinking = true
	m.Caps.ThinkingDialect = model.ThinkingDialectAdaptive
	m.Sampling = model.Sampling{Effort: model.EffortMinimal}
	_, err := anthropicapi.EncodeRequest(inference.Request{
		Model: m, Messages: content.AgenticMessages{userMsg(textBlock("hi"))},
	}, false)
	if err == nil {
		t.Fatal("EncodeRequest() error = nil, want rejection of unsupported minimal effort")
	}
}

// --- TestEncodeRequest_ThinkingOmitsSampling ------------------------------

// Current adaptive-thinking Anthropic models reject temperature/top_p sent
// alongside thinking (HTTP 400). The codec reconciles this: when thinking is
// enabled for the request (Caps.Thinking AND a real Effort), temperature and
// top_p are omitted from the body even if the caller set them; otherwise they
// pass through unchanged.
func TestEncodeRequest_ThinkingOmitsSampling(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name         string
		thinkingCap  bool
		effort       model.Effort
		wantThinking bool // thinking + output_config present
		wantSampling bool // temperature + top_p present
	}{
		{
			name:         "thinking enabled omits temperature and top_p",
			thinkingCap:  true,
			effort:       model.EffortHigh,
			wantThinking: true,
			wantSampling: false,
		},
		{
			name:         "thinking-capable but effort none keeps temperature and top_p",
			thinkingCap:  true,
			effort:       model.EffortNone,
			wantThinking: false,
			wantSampling: true,
		},
		{
			name:         "not thinking-capable keeps temperature and top_p even with effort",
			thinkingCap:  false,
			effort:       model.EffortHigh,
			wantThinking: false,
			wantSampling: true,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			m := baseModel()
			m.Caps.Thinking = tc.thinkingCap
			m.Caps.ThinkingDialect = model.ThinkingDialectAdaptive
			// Both temperature and top_p are set on every case so absence is the
			// codec's reconciliation, not a missing input.
			m.Sampling = model.Sampling{Temperature: f64ptr(0.7), TopP: f64ptr(0.9), Effort: tc.effort}
			req := inference.Request{Model: m, Messages: content.AgenticMessages{userMsg(textBlock("hi"))}}
			data, err := anthropicapi.EncodeRequest(req, false)
			if err != nil {
				t.Fatalf("EncodeRequest: %v", err)
			}
			body := decodeObj(t, data)

			if _, ok := body["thinking"]; ok != tc.wantThinking {
				t.Errorf("thinking present = %v, want %v", ok, tc.wantThinking)
			}
			if _, ok := body["output_config"]; ok != tc.wantThinking {
				t.Errorf("output_config present = %v, want %v", ok, tc.wantThinking)
			}
			if _, ok := body["temperature"]; ok != tc.wantSampling {
				t.Errorf("temperature present = %v, want %v", ok, tc.wantSampling)
			}
			if _, ok := body["top_p"]; ok != tc.wantSampling {
				t.Errorf("top_p present = %v, want %v", ok, tc.wantSampling)
			}
		})
	}
}

// --- TestEncodeRequest_MaxTokensAndSampling -------------------------------

func TestEncodeRequest_MaxTokensAndSampling(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name          string
		sampling      model.Sampling
		override      *model.Sampling
		wantMaxTokens int
		wantTempKey   bool
		wantTemp      float64
		wantTopPKey   bool
		wantStopKey   bool
		wantStop      []string
	}{
		{
			name:          "unset max_tokens uses the codec default",
			sampling:      model.Sampling{},
			wantMaxTokens: 4096,
		},
		{
			name:          "explicit max_tokens is honored",
			sampling:      model.Sampling{MaxTokens: intptr(2000)},
			wantMaxTokens: 2000,
		},
		{
			name:          "temperature/top_p/stop_sequences map through",
			sampling:      model.Sampling{Temperature: f64ptr(0.7), TopP: f64ptr(0.9), Stop: []string{"STOP"}},
			wantMaxTokens: 4096,
			wantTempKey:   true,
			wantTemp:      0.7,
			wantTopPKey:   true,
			wantStopKey:   true,
			wantStop:      []string{"STOP"},
		},
		{
			name:          "override wins over model sampling",
			sampling:      model.Sampling{MaxTokens: intptr(10)},
			override:      &model.Sampling{MaxTokens: intptr(999)},
			wantMaxTokens: 999,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			m := baseModel()
			m.Sampling = tc.sampling
			req := inference.Request{Model: m, Messages: content.AgenticMessages{userMsg(textBlock("hi"))}, Override: tc.override}
			data, err := anthropicapi.EncodeRequest(req, false)
			if err != nil {
				t.Fatalf("EncodeRequest: %v", err)
			}
			body := decodeObj(t, data)

			if got := asInt(t, body["max_tokens"]); got != tc.wantMaxTokens {
				t.Errorf("max_tokens = %d, want %d", got, tc.wantMaxTokens)
			}
			if _, ok := body["temperature"]; ok != tc.wantTempKey {
				t.Errorf("temperature present = %v, want %v", ok, tc.wantTempKey)
			}
			if tc.wantTempKey {
				var got float64
				if err := json.Unmarshal(body["temperature"], &got); err != nil || got != tc.wantTemp {
					t.Errorf("temperature = %v (err %v), want %v", got, err, tc.wantTemp)
				}
			}
			if _, ok := body["top_p"]; ok != tc.wantTopPKey {
				t.Errorf("top_p present = %v, want %v", ok, tc.wantTopPKey)
			}
			if _, ok := body["stop_sequences"]; ok != tc.wantStopKey {
				t.Errorf("stop_sequences present = %v, want %v", ok, tc.wantStopKey)
			}
			if tc.wantStopKey {
				var got []string
				if err := json.Unmarshal(body["stop_sequences"], &got); err != nil {
					t.Fatalf("unmarshal stop_sequences: %v", err)
				}
				if len(got) != len(tc.wantStop) || got[0] != tc.wantStop[0] {
					t.Errorf("stop_sequences = %v, want %v", got, tc.wantStop)
				}
			}
		})
	}
}

// --- TestEncodeRequest_Stream ---------------------------------------------

func TestEncodeRequest_Stream(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name       string
		stream     bool
		wantHasKey bool
		wantStream bool
	}{
		{name: "invoke omits stream", stream: false, wantHasKey: false},
		{name: "stream sets stream true", stream: true, wantHasKey: true, wantStream: true},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			req := inference.Request{Model: baseModel(), Messages: content.AgenticMessages{userMsg(textBlock("hi"))}}
			data, err := anthropicapi.EncodeRequest(req, tc.stream)
			if err != nil {
				t.Fatalf("EncodeRequest: %v", err)
			}
			body := decodeObj(t, data)
			raw, ok := body["stream"]
			if ok != tc.wantHasKey {
				t.Fatalf("stream present = %v, want %v", ok, tc.wantHasKey)
			}
			if !tc.wantHasKey {
				return
			}
			var got bool
			if err := json.Unmarshal(raw, &got); err != nil || got != tc.wantStream {
				t.Errorf("stream = %v (err %v), want %v", got, err, tc.wantStream)
			}
		})
	}
}

// --- TestEncodeRequest_Blocks ---------------------------------------------

// Exercises assistant blocks (text/thinking/tool_use), tool_use default input,
// and tool_result mapping (tool_use_id + is_error).
func TestEncodeRequest_Blocks(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name     string
		conv     content.Conversation
		wantRole string
		assert   func(t *testing.T, blocks []map[string]json.RawMessage)
	}{
		{
			name:     "assistant text + thinking + tool_use",
			conv:     aiMsg(&content.TextBlock{Text: "sure"}, content.NewSignedThinkingBlock("reason", "sig", signatureFormatAnthropic, nil, ""), &content.ToolUseBlock{ID: "toolu_1", Name: "run", Input: json.RawMessage(`{"x":1}`)}),
			wantRole: "assistant",
			assert: func(t *testing.T, blocks []map[string]json.RawMessage) {
				if len(blocks) != 3 {
					t.Fatalf("block count = %d, want 3", len(blocks))
				}
				if got := asString(t, blocks[0]["type"]); got != "text" {
					t.Errorf("blocks[0].type = %q, want text", got)
				}
				if got := asString(t, blocks[1]["type"]); got != "thinking" {
					t.Errorf("blocks[1].type = %q, want thinking", got)
				}
				if got := asString(t, blocks[1]["signature"]); got != "sig" {
					t.Errorf("blocks[1].signature = %q, want sig", got)
				}
				if got := asString(t, blocks[2]["type"]); got != "tool_use" {
					t.Errorf("blocks[2].type = %q, want tool_use", got)
				}
				if got := asString(t, blocks[2]["id"]); got != "toolu_1" {
					t.Errorf("blocks[2].id = %q, want toolu_1", got)
				}
				if got := string(blocks[2]["input"]); got != `{"x":1}` {
					t.Errorf("blocks[2].input = %s, want {\"x\":1}", got)
				}
			},
		},
		{
			name:     "tool_use with empty input defaults to empty object",
			conv:     aiMsg(&content.ToolUseBlock{ID: "toolu_2", Name: "noargs"}),
			wantRole: "assistant",
			assert: func(t *testing.T, blocks []map[string]json.RawMessage) {
				if got := string(blocks[0]["input"]); got != `{}` {
					t.Errorf("input = %s, want {}", got)
				}
			},
		},
		{
			name:     "tool_result maps to a user tool_result block with is_error",
			conv:     toolResultMsg("toolu_1", true, textBlock("boom")),
			wantRole: "user",
			assert: func(t *testing.T, blocks []map[string]json.RawMessage) {
				if len(blocks) != 1 {
					t.Fatalf("block count = %d, want 1", len(blocks))
				}
				if got := asString(t, blocks[0]["type"]); got != "tool_result" {
					t.Errorf("type = %q, want tool_result", got)
				}
				if got := asString(t, blocks[0]["tool_use_id"]); got != "toolu_1" {
					t.Errorf("tool_use_id = %q, want toolu_1", got)
				}
				var isErr bool
				if err := json.Unmarshal(blocks[0]["is_error"], &isErr); err != nil || !isErr {
					t.Errorf("is_error = %v (err %v), want true", isErr, err)
				}
				inner := asObjs(t, blocks[0]["content"])
				if len(inner) != 1 || asString(t, inner[0]["type"]) != "text" {
					t.Errorf("tool_result content = %v, want one text block", inner)
				}
			},
		},
		{
			name:     "tool_result without error omits is_error",
			conv:     toolResultMsg("toolu_9", false, textBlock("ok")),
			wantRole: "user",
			assert: func(t *testing.T, blocks []map[string]json.RawMessage) {
				if _, ok := blocks[0]["is_error"]; ok {
					t.Errorf("is_error present, want absent for a non-error result")
				}
			},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			req := inference.Request{Model: baseModel(), Messages: content.AgenticMessages{tc.conv}}
			data, err := anthropicapi.EncodeRequest(req, false)
			if err != nil {
				t.Fatalf("EncodeRequest: %v", err)
			}
			body := decodeObj(t, data)
			msg := messagesOf(t, body)[0]
			if got := asString(t, msg["role"]); got != tc.wantRole {
				t.Errorf("role = %q, want %q", got, tc.wantRole)
			}
			tc.assert(t, blocksOf(t, msg))
		})
	}
}

// --- TestEncodeRequest_Errors ---------------------------------------------

// TestEncodeRequest_Errors covers the generic unsupported-block fallback. Audio
// and document blocks no longer arrive here: document is a real
// RequestDocumentBlock and audio has a typed error naming the format's own
// limitation, both exercised in encode_document_test.go. A bare
// ToolResultBlock nested inside a message keeps the fallback itself covered —
// the neutral vocabulary allows it, and the dialect carries tool results as
// their own turn rather than as a nested block.
func TestEncodeRequest_Errors(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		messages  content.AgenticMessages
		wantBlock bool
	}{
		{
			name: "nested tool-result block is unsupported",
			messages: content.AgenticMessages{userMsg(&content.ToolResultBlock{
				ToolUseID: "toolu_nested",
				Content:   []content.Block{&content.TextBlock{Text: "result"}},
			})},
			wantBlock: true,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := anthropicapi.EncodeRequest(inference.Request{Model: baseModel(), Messages: tc.messages}, false)
			if err == nil {
				t.Fatalf("expected error, got nil")
			}
			var ube *anthropicapi.UnsupportedBlockError
			if tc.wantBlock && !errors.As(err, &ube) {
				t.Errorf("error = %v, want *UnsupportedBlockError", err)
			}
		})
	}
}

// TestEncodeRequest_RefusalBlockFailsClosed pins the fail-closed encoding of a
// content.RefusalBlock. Anthropic reports a refusal as RESPONSE metadata —
// stop_reason "refusal" with a RefusalStopDetails object — and its request
// document declares no refusal content block of any kind. There is therefore
// nothing to route a replayed refusal to, and inventing one (text, or a
// system note) would show the model its own decline as something it said.
func TestEncodeRequest_RefusalBlockFailsClosed(t *testing.T) {
	t.Parallel()

	_, err := anthropicapi.EncodeRequest(inference.Request{
		Model:    baseModel(),
		Messages: content.AgenticMessages{aiMsg(&content.RefusalBlock{Text: "I cannot help with that."})},
	}, false)
	if err == nil {
		t.Fatal("expected an error, got nil")
	}
	var refusal *anthropicapi.UnsupportedRefusalError
	if !errors.As(err, &refusal) {
		t.Fatalf("error = %v (%T), want *UnsupportedRefusalError", err, err)
	}
	if !strings.Contains(err.Error(), "refusal") {
		t.Errorf("error = %q, want it to name the refusal limitation", err)
	}
}
