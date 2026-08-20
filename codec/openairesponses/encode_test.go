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

// TestEncodeRequest_EmptyInputIsAnEmptyArrayNotNull pins that `input` is
// always an array on the wire. The spec types it as string-or-array with no
// null alternative, so a Go nil slice marshalling as `null` is a type error in
// the request body, not merely an empty one.
func TestEncodeRequest_EmptyInputIsAnEmptyArrayNotNull(t *testing.T) {
	t.Parallel()
	body, err := openairesponses.EncodeRequest(inference.Request{
		Model: model.Model{Name: "gpt-test"},
	}, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	input, ok := decodeRequestBody(t, body)["input"]
	if !ok {
		t.Fatalf("body = %s, want an input member", body)
	}
	if _, ok := input.([]any); !ok {
		t.Errorf("input = %#v (%T), want an array", input, input)
	}
}

// TestEncodeRequest_FunctionToolAlwaysCarriesStrict pins a REQUIRED member of
// the Responses FunctionTool shape. OpenAI's own spec lists
// ["type","name","strict","parameters"] as required, so a tool object without
// `strict` is not a legal request body, however permissively any one server
// happens to treat it. The value is always false: `strict` true additionally
// demands the tool schema be a strict subset (every property required,
// additionalProperties:false), and inference.Tool carries arbitrary caller
// schemas that this codec cannot certify.
func TestEncodeRequest_FunctionToolAlwaysCarriesStrict(t *testing.T) {
	t.Parallel()
	body, err := openairesponses.EncodeRequest(inference.Request{
		Model: model.Model{Name: "gpt-test", Caps: model.Capabilities{Tools: true}},
		Tools: []inference.Tool{{
			Name:   "get_weather",
			Schema: json.RawMessage(`{"type":"object"}`),
		}},
	}, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	tool := decodeRequestBody(t, body)["tools"].([]any)[0].(map[string]any)
	strict, ok := tool["strict"]
	if !ok {
		t.Fatalf("tool = %#v, want a `strict` member (FunctionTool.required)", tool)
	}
	if strict != false {
		t.Errorf("strict = %#v, want false", strict)
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
		ToolChoice: inference.ToolRequired(),
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

// TestEncodeRequest_ToolChoiceNamedTool pins the named tool-choice variant.
// Responses' ToolChoiceFunction is flat — `{"type":"function","name":...}` —
// unlike Chat Completions, which nests the name under `function`.
func TestEncodeRequest_ToolChoiceNamedTool(t *testing.T) {
	t.Parallel()
	req := inference.Request{
		Model: model.Model{Name: "gpt-test", Caps: model.Capabilities{Tools: true}},
		Tools: []inference.Tool{
			{Name: "f", Schema: json.RawMessage(`{"type":"object"}`)},
			{Name: "g", Schema: json.RawMessage(`{"type":"object"}`)},
		},
		ToolChoice: inference.ToolNamed("g"),
	}
	body, err := openairesponses.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	var wire struct {
		ToolChoice json.RawMessage `json:"tool_choice"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	const want = `{"type":"function","name":"g"}`
	if string(wire.ToolChoice) != want {
		t.Errorf("tool_choice = %s, want %s", wire.ToolChoice, want)
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
	for _, effort := range []model.Effort{
		model.EffortMinimal, model.EffortLow, model.EffortMedium, model.EffortHigh, model.EffortXHigh, model.EffortMax,
	} {
		effort := effort
		t.Run(string(effort), func(t *testing.T) {
			t.Parallel()
			req := inference.Request{
				Model:    model.Model{Name: "gpt-test", Caps: model.Capabilities{Thinking: true}},
				Override: &model.Sampling{Effort: effort},
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
			if reasoning["effort"] != string(effort) {
				t.Errorf("effort = %v, want %q", reasoning["effort"], effort)
			}
			if reasoning["summary"] != "auto" {
				t.Errorf("summary = %v", reasoning["summary"])
			}
			include, ok := m["include"].([]any)
			if !ok || len(include) != 1 || include[0] != "reasoning.encrypted_content" {
				t.Errorf("include = %#v, want [reasoning.encrypted_content]", m["include"])
			}
		})
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

// TestEncodeRequest_EncryptedReasoningReplay covers the encrypted-content
// round trip through ThinkingBlock.ProviderState. It previously asserted that
// the replayed item carried NO id — a rule that read as "never invent an id"
// but in practice emitted a reasoning item missing a required member, which
// the provider rejects outright. The rule is unchanged in spirit (no id is
// ever fabricated); the item now replays the id the response itself carried,
// which ProviderState transports alongside the encrypted content.
func TestEncodeRequest_EncryptedReasoningReplay(t *testing.T) {
	t.Parallel()
	providerState := json.RawMessage(`{"id":"rs_abc123","encrypted_content":"opaque-blob-abc123"}`)
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
	if item["id"] != "rs_abc123" {
		t.Errorf("id = %#v, want the response's own rs_abc123", item["id"])
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

// TestEncodeRequest_AssistantTextReplay locks the input shape of replayed
// assistant history. It previously asserted the OutputMessage form
// ({"type":"message","role":"assistant","content":[{"type":"output_text",...}]})
// with no id — but OutputMessage.required is ["id","type","role","content",
// "status"], so that item is invalid as input and no id is available to
// supply, since a message item's id is not carried by any neutral block.
// The valid alternative is EasyInputMessage, whose required members are only
// ["role","content"] and whose content may be a bare string. The `type`
// discriminator is deliberately omitted: with role "assistant" present,
// EasyInputMessage is then the only variant of the InputItem union that can
// match (InputMessage's role enum excludes assistant, OutputMessage demands
// an id and status).
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
	if item["role"] != "assistant" {
		t.Fatalf("item = %#v", item)
	}
	if item["content"] != "prior answer" {
		t.Errorf("content = %#v, want the plain string %q", item["content"], "prior answer")
	}
	for _, forbidden := range []string{"type", "id", "status"} {
		if _, present := item[forbidden]; present {
			t.Errorf("assistant history item carries %q, want the bare EasyInputMessage form: %#v", forbidden, item)
		}
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

// TestEncodeRequest_ToolResultEmptyOutput locks the required `output` member.
// FunctionCallOutputItemParam.required is ["type","output"], so an empty tool
// result must still carry "output":"" — an omitempty drop makes the provider
// reject the whole request with a missing-required-parameter 400, which is the
// single most common way an empty tool result poisons a turn.
func TestEncodeRequest_ToolResultEmptyOutput(t *testing.T) {
	t.Parallel()
	req := inference.Request{
		Model: model.Model{Name: "gpt-test"},
		Messages: content.AgenticMessages{
			&content.ToolResultMessage{
				Message:   content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: ""}}},
				ToolUseID: "call_1",
			},
		},
	}
	body, err := openairesponses.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	m := decodeRequestBody(t, body)
	item := m["input"].([]any)[0].(map[string]any)
	if item["type"] != "function_call_output" {
		t.Fatalf("item = %#v", item)
	}
	out, present := item["output"]
	if !present {
		t.Fatalf("output absent; it is required even when empty (item=%#v)", item)
	}
	if out != "" {
		t.Errorf("output = %#v, want empty string", out)
	}
	if item["call_id"] != "call_1" {
		t.Errorf("call_id = %#v, want call_1", item["call_id"])
	}
}

// TestEncodeRequest_ItemsCarryOnlyTheirOwnMembers guards the tagged union:
// making `output`/`call_id` unconditional must not leak them onto message,
// reasoning, or function_call items, which have no such properties.
func TestEncodeRequest_ItemsCarryOnlyTheirOwnMembers(t *testing.T) {
	t.Parallel()
	req := inference.Request{
		Model: model.Model{Name: "gpt-test", Caps: model.Capabilities{Thinking: true}},
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{
				Role:   content.RoleUser,
				Blocks: []content.Block{&content.TextBlock{Text: "hi"}},
			}},
			&content.AIMessage{Message: content.Message{
				Role: content.RoleAssistant,
				Blocks: []content.Block{
					content.NewThinkingBlock("think", "", json.RawMessage(`{"id":"rs_1","encrypted_content":"blob"}`), "openai-responses"),
					&content.ToolUseBlock{ID: "call_1", Name: "calc", Input: json.RawMessage(`{"x":1}`)},
				},
			}},
		},
	}
	body, err := openairesponses.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	m := decodeRequestBody(t, body)
	for i, raw := range m["input"].([]any) {
		item := raw.(map[string]any)
		if item["type"] == "function_call_output" {
			continue
		}
		if _, present := item["output"]; present {
			t.Errorf("input[%d] (%v) carries an `output` member: %#v", i, item["type"], item)
		}
		if item["type"] != "function_call" {
			if _, present := item["call_id"]; present {
				t.Errorf("input[%d] (%v) carries a `call_id` member: %#v", i, item["type"], item)
			}
		}
	}
}

// TestEncodeRequest_ReasoningReplayPreservesItemID locks the fix for the
// reasoning-replay 400. ReasoningItem.required is ["id","summary","type"], so
// replaying a reasoning item without its id is rejected with "Missing required
// parameter: 'input[N].id'". The id is not invented — it is the one the
// provider's own response carried, round-tripped through
// ThinkingBlock.ProviderState alongside the encrypted content.
func TestEncodeRequest_ReasoningReplayPreservesItemID(t *testing.T) {
	t.Parallel()
	providerState := json.RawMessage(`{"id":"rs_abc123","encrypted_content":"opaque-blob-abc123"}`)
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
	item := m["input"].([]any)[0].(map[string]any)
	if item["type"] != "reasoning" {
		t.Fatalf("item type = %v, want reasoning", item["type"])
	}
	if item["id"] != "rs_abc123" {
		t.Errorf("id = %#v, want the response's own rs_abc123", item["id"])
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

// TestEncodeRequest_ReasoningSummaryAlwaysPresent covers a reasoning item whose
// summary is empty: `summary` is required, so it must marshal as [] rather
// than being dropped.
func TestEncodeRequest_ReasoningSummaryAlwaysPresent(t *testing.T) {
	t.Parallel()
	providerState := json.RawMessage(`{"id":"rs_abc123","encrypted_content":"blob"}`)
	req := inference.Request{
		Model: model.Model{Name: "gpt-test", Caps: model.Capabilities{Thinking: true}},
		Messages: content.AgenticMessages{
			&content.AIMessage{Message: content.Message{
				Role:   content.RoleAssistant,
				Blocks: []content.Block{content.NewThinkingBlock("", "", providerState, "openai-responses")},
			}},
		},
	}
	body, err := openairesponses.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	m := decodeRequestBody(t, body)
	item := m["input"].([]any)[0].(map[string]any)
	summary, present := item["summary"]
	if !present {
		t.Fatalf("summary absent; it is required even when empty (item=%#v)", item)
	}
	if got := summary.([]any); len(got) != 0 {
		t.Errorf("summary = %#v, want []", got)
	}
}

// TestEncodeRequest_ReasoningWithoutIDIsDropped documents the degrade for a
// ThinkingBlock whose provider state predates id preservation (the legacy bare
// -string encoding, which carried encrypted content and nothing else). There
// is no valid reasoning item to build without an id, and inventing one would
// be worse than omitting the item: OpenAI regenerates reasoning server-side,
// whereas a fabricated id is a guaranteed 400.
func TestEncodeRequest_ReasoningWithoutIDIsDropped(t *testing.T) {
	t.Parallel()
	req := inference.Request{
		Model: model.Model{Name: "gpt-test", Caps: model.Capabilities{Thinking: true}},
		Messages: content.AgenticMessages{
			&content.AIMessage{Message: content.Message{
				Role: content.RoleAssistant,
				Blocks: []content.Block{
					content.NewThinkingBlock("legacy", "", json.RawMessage(`"legacy-blob"`), "openai-responses"),
					&content.TextBlock{Text: "answer"},
				},
			}},
		},
	}
	body, err := openairesponses.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	m := decodeRequestBody(t, body)
	for i, raw := range m["input"].([]any) {
		if item := raw.(map[string]any); item["type"] == "reasoning" {
			t.Errorf("input[%d] is an id-less reasoning item, want it dropped: %#v", i, item)
		}
	}
}
