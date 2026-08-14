package openaiapi_test

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/openaiapi"
	model "github.com/looprig/inference/model"
)

// mustDecode unmarshals raw JSON into a map for field inspection.
func mustDecode(t *testing.T, data []byte) map[string]json.RawMessage {
	t.Helper()
	var m map[string]json.RawMessage
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return m
}

func messagesFromRaw(t *testing.T, raw map[string]json.RawMessage) []map[string]json.RawMessage {
	t.Helper()
	var msgs []map[string]json.RawMessage
	if err := json.Unmarshal(raw["messages"], &msgs); err != nil {
		t.Fatalf("unmarshal messages: %v", err)
	}
	return msgs
}

func roleOf(t *testing.T, msg map[string]json.RawMessage) string {
	t.Helper()
	var r string
	if err := json.Unmarshal(msg["role"], &r); err != nil {
		t.Fatalf("unmarshal role: %v", err)
	}
	return r
}

func contentStr(t *testing.T, msg map[string]json.RawMessage) string {
	t.Helper()
	var s string
	if err := json.Unmarshal(msg["content"], &s); err != nil {
		t.Fatalf("unmarshal content as string: %v", err)
	}
	return s
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

func toolMsg(id string, blocks ...content.Block) *content.ToolResultMessage {
	return &content.ToolResultMessage{Message: content.Message{Role: content.RoleTool, Blocks: blocks}, ToolUseID: id}
}

func textBlock(s string) content.Block {
	return &content.TextBlock{Text: s}
}

func imageURLBlock(url string) content.Block {
	return &content.ImageBlock{
		MediaType: content.MediaTypeImageJPEG,
		Source:    content.ImageSource{URL: url},
	}
}

func imageDataBlock(mediaType content.MediaType, data []byte) content.Block {
	return &content.ImageBlock{
		MediaType: mediaType,
		Source:    content.ImageSource{Data: data},
	}
}

func thinkingBlock(text string) content.Block {
	return &content.ThinkingBlock{Thinking: text}
}

func toolUseBlock(id, name string, input json.RawMessage) content.Block {
	return &content.ToolUseBlock{ID: id, Name: name, Input: input}
}

func structuredOutput() *inference.OutputSchema {
	return &inference.OutputSchema{
		Name:        "answer",
		Description: "One answer",
		Strict:      true,
		Schema:      json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}`),
	}
}

func structuredModel(withTools bool) model.Model {
	m := model.Model{Name: "gpt-4o"}
	m.Caps.StructuredOutput = true
	if withTools {
		m.Caps.Tools = true
		m.Caps.StructuredOutputWithTools = true
	}
	return m
}

func TestEncodeRequestStructuredOutput(t *testing.T) {
	t.Parallel()

	req := inference.Request{Model: structuredModel(false), Output: structuredOutput()}
	body, err := openaiapi.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}

	var wire struct {
		ResponseFormat *struct {
			Type       string `json:"type"`
			JSONSchema *struct {
				Name        string          `json:"name"`
				Description json.RawMessage `json:"description"`
				Strict      bool            `json:"strict"`
				Schema      json.RawMessage `json:"schema"`
			} `json:"json_schema"`
		} `json:"response_format"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if wire.ResponseFormat == nil || wire.ResponseFormat.JSONSchema == nil {
		t.Fatalf("response_format.json_schema missing: %s", body)
	}
	got := wire.ResponseFormat.JSONSchema
	if wire.ResponseFormat.Type != "json_schema" || got.Name != "answer" || !got.Strict {
		t.Errorf("response format = type %q name %q strict %v", wire.ResponseFormat.Type, got.Name, got.Strict)
	}
	if got.Description != nil {
		t.Errorf("response_format.json_schema.description unexpectedly present: %s", got.Description)
	}
	if string(got.Schema) != string(req.Output.Schema) {
		t.Errorf("response_format.json_schema.schema = %s, want %s", got.Schema, req.Output.Schema)
	}
	raw := mustDecode(t, body)
	want := `{"type":"json_schema","json_schema":{"name":"answer","strict":true,"schema":{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}}}`
	if string(raw["response_format"]) != want {
		t.Errorf("response_format = %s, want exact %s", raw["response_format"], want)
	}
}

func TestEncodeRequestStructuredOutputWithTools(t *testing.T) {
	t.Parallel()

	req := inference.Request{
		Model:      structuredModel(true),
		Output:     structuredOutput(),
		ToolChoice: inference.ToolRequired(),
		Tools: []inference.Tool{{
			Name:        "lookup",
			Description: "Look up a value",
			Schema:      json.RawMessage(`{"type":"object","properties":{"key":{"type":"string"}},"required":["key"],"additionalProperties":false}`),
		}},
	}
	body, err := openaiapi.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	var wire struct {
		ResponseFormat json.RawMessage `json:"response_format"`
		ToolChoice     string          `json:"tool_choice"`
		Tools          []struct {
			Type     string `json:"type"`
			Function struct {
				Name        string          `json:"name"`
				Description string          `json:"description"`
				Parameters  json.RawMessage `json:"parameters"`
			} `json:"function"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if wire.ResponseFormat == nil {
		t.Fatalf("response_format missing: %s", body)
	}
	if wire.ToolChoice != "required" {
		t.Errorf("tool_choice = %q, want required", wire.ToolChoice)
	}
	if len(wire.Tools) != 1 {
		t.Fatalf("len(tools) = %d, want 1", len(wire.Tools))
	}
	tool := wire.Tools[0]
	if tool.Type != "function" || tool.Function.Name != req.Tools[0].Name || tool.Function.Description != req.Tools[0].Description || string(tool.Function.Parameters) != string(req.Tools[0].Schema) {
		t.Errorf("tool changed on combined request: %+v", tool)
	}
}

// TestEncodeRequestNamedToolChoice pins the named tool-choice variant.
// ChatCompletionNamedToolChoice nests the name one level deeper than the
// Responses dialect does: {"type":"function","function":{"name":...}}, with
// type, function and function.name all required.
func TestEncodeRequestNamedToolChoice(t *testing.T) {
	t.Parallel()

	body, err := openaiapi.EncodeRequest(inference.Request{
		Model: structuredModel(false),
		Tools: []inference.Tool{
			{Name: "lookup", Schema: json.RawMessage(`{"type":"object"}`)},
			{Name: "search", Schema: json.RawMessage(`{"type":"object"}`)},
		},
		ToolChoice: inference.ToolNamed("search"),
	}, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	var wire struct {
		ToolChoice json.RawMessage `json:"tool_choice"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	const want = `{"type":"function","function":{"name":"search"}}`
	if string(wire.ToolChoice) != want {
		t.Errorf("tool_choice = %s, want %s", wire.ToolChoice, want)
	}
}

func TestBuildChatRequestStructuredOutputValidationPrecedesEncoding(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		req          inference.Request
		wantBase     bool
		wantCombined bool
		wantSchema   bool
	}{
		{
			name:     "base capability error",
			req:      inference.Request{Model: model.Model{Name: "unsupported"}, Messages: content.AgenticMessages{nil}, Output: structuredOutput()},
			wantBase: true,
		},
		{
			name: "combined capability error",
			req: inference.Request{
				Model:    model.Model{Name: "unsupported-combined", Caps: model.Capabilities{StructuredOutput: true, Tools: true}},
				Messages: content.AgenticMessages{nil},
				Output:   structuredOutput(),
				Tools:    []inference.Tool{{Name: "lookup"}},
			},
			wantCombined: true,
		},
		{
			name: "schema error",
			req: func() inference.Request {
				output := structuredOutput()
				output.Schema = json.RawMessage(`{"type":"array"}`)
				return inference.Request{Model: structuredModel(false), Messages: content.AgenticMessages{nil}, Output: output}
			}(),
			wantSchema: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			_, err := openaiapi.BuildChatRequest(tt.req, false)
			if tt.wantBase {
				var target *inference.StructuredOutputUnsupportedError
				if !errors.As(err, &target) {
					t.Fatalf("BuildChatRequest() error = %T %v, want *StructuredOutputUnsupportedError", err, err)
				}
			}
			if tt.wantCombined {
				var target *inference.StructuredOutputWithToolsUnsupportedError
				if !errors.As(err, &target) {
					t.Fatalf("BuildChatRequest() error = %T %v, want *StructuredOutputWithToolsUnsupportedError", err, err)
				}
			}
			if tt.wantSchema {
				var target *inference.SchemaValidationError
				if !errors.As(err, &target) {
					t.Fatalf("BuildChatRequest() error = %T %v, want *SchemaValidationError", err, err)
				}
			}
		})
	}
}

func TestEncodeRequestStructuredOutputStreamParity(t *testing.T) {
	t.Parallel()

	req := inference.Request{Model: structuredModel(false), Output: structuredOutput()}
	oneshoot, err := openaiapi.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest(non-stream) error = %v", err)
	}
	streamed, err := openaiapi.EncodeRequest(req, true)
	if err != nil {
		t.Fatalf("EncodeRequest(stream) error = %v", err)
	}
	oneshootRaw := mustDecode(t, oneshoot)
	streamedRaw := mustDecode(t, streamed)
	if string(oneshootRaw["response_format"]) != string(streamedRaw["response_format"]) {
		t.Errorf("response_format differs: non-stream %s, stream %s", oneshootRaw["response_format"], streamedRaw["response_format"])
	}
	if string(streamedRaw["stream"]) != "true" {
		t.Errorf("stream = %s, want true", streamedRaw["stream"])
	}
}

// TestEncodeRequestNilOutputByteShape pins the byte shape of the emptiest
// legal-to-attempt request. `messages` is spec-typed as an array with no null
// alternative, so a Go nil slice must still marshal as [] — this test
// previously asserted `"messages":null`, which is a type error on the wire
// however few servers bother to reject it. (The array being EMPTY is a
// separate matter: CreateChatCompletionRequest.messages carries minItems 1, so
// a caller passing no messages at all is still sending something OpenAI will
// refuse. That is the caller's error to make, not a reason to emit null.)
func TestEncodeRequestNilOutputByteShape(t *testing.T) {
	t.Parallel()

	body, err := openaiapi.EncodeRequest(inference.Request{Model: model.Model{Name: "m"}}, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	const want = `{"model":"m","messages":[]}`
	if string(body) != want {
		t.Errorf("EncodeRequest() = %s, want byte-identical %s", body, want)
	}
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

	want, err := openaiapi.EncodeRequest(inference.Request{Model: model.Model{Name: "m"}, Messages: content.AgenticMessages{aiMsg(textBlock("answer"))}}, false)
	if err != nil {
		t.Fatalf("EncodeRequest() baseline error = %v", err)
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			message := aiMsg(textBlock("answer"))
			message.Usage = tt.usage
			got, err := openaiapi.EncodeRequest(inference.Request{Model: model.Model{Name: "m"}, Messages: content.AgenticMessages{message}}, false)
			if err != nil {
				t.Fatalf("EncodeRequest() error = %v", err)
			}
			if string(got) != string(want) {
				t.Errorf("EncodeRequest() = %s, want byte-identical %s", got, want)
			}
		})
	}
}

// --- TestEncodeRequest_System ---

func TestEncodeRequest_System(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name        string
		systemSpec  string
		messages    content.AgenticMessages
		wantFirst   string
		wantMsgRole string
	}{
		{
			name:       "system from Request prepended",
			systemSpec: "You are helpful.",
			messages:   content.AgenticMessages{userMsg(textBlock("hi"))},
			wantFirst:  "system",
		},
		{
			name:       "no system: first message is user",
			systemSpec: "",
			messages:   content.AgenticMessages{userMsg(textBlock("hi"))},
			wantFirst:  "user",
		},
		{
			name:       "empty system string treated as absent",
			systemSpec: "",
			messages:   content.AgenticMessages{userMsg(textBlock("hello"))},
			wantFirst:  "user",
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			req := inference.Request{
				Model:    model.Model{Name: "test-model"},
				System:   tc.systemSpec,
				Messages: tc.messages,
			}
			got, err := openaiapi.EncodeRequest(req, false)
			if err != nil {
				t.Fatalf("EncodeRequest error: %v", err)
			}

			raw := mustDecode(t, got)
			msgs := messagesFromRaw(t, raw)
			if len(msgs) == 0 {
				t.Fatal("expected at least one message")
			}
			if got := roleOf(t, msgs[0]); got != tc.wantFirst {
				t.Errorf("first message role = %q, want %q", got, tc.wantFirst)
			}
		})
	}
}

// --- TestEncodeRequest_StreamFlag ---

func TestEncodeRequest_StreamFlag(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name       string
		stream     bool
		wantStream bool
	}{
		{name: "stream true", stream: true, wantStream: true},
		{name: "stream false", stream: false, wantStream: false},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			req := inference.Request{
				Model:    model.Model{Name: "m"},
				Messages: content.AgenticMessages{userMsg(textBlock("x"))},
			}
			got, err := openaiapi.EncodeRequest(req, tc.stream)
			if err != nil {
				t.Fatalf("EncodeRequest error: %v", err)
			}

			raw := mustDecode(t, got)
			streamRaw, exists := raw["stream"]
			if tc.wantStream {
				if !exists {
					t.Fatal("expected stream key in JSON")
				}
				var v bool
				if err := json.Unmarshal(streamRaw, &v); err != nil {
					t.Fatalf("unmarshal stream: %v", err)
				}
				if !v {
					t.Error("expected stream=true")
				}
			} else {
				if exists {
					var v bool
					if err := json.Unmarshal(streamRaw, &v); err == nil && v {
						t.Error("expected stream to be absent or false")
					}
				}
			}
		})
	}
}

func TestEncodeRequestIncludeUsage(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name              string
		stream            bool
		wantStreamOptions bool
	}{
		{name: "stream requests usage trailer", stream: true, wantStreamOptions: true},
		{name: "oneshot omits stream options", stream: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			body, err := openaiapi.EncodeRequest(inference.Request{Model: model.Model{Name: "m"}}, tt.stream)
			if err != nil {
				t.Fatalf("EncodeRequest() error = %v", err)
			}
			var got struct {
				StreamOptions *struct {
					IncludeUsage bool `json:"include_usage"`
				} `json:"stream_options"`
			}
			if err := json.Unmarshal(body, &got); err != nil {
				t.Fatalf("json.Unmarshal() error = %v", err)
			}
			if (got.StreamOptions != nil) != tt.wantStreamOptions {
				t.Fatalf("stream_options present = %v, want %v; body = %s", got.StreamOptions != nil, tt.wantStreamOptions, body)
			}
			if got.StreamOptions != nil && !got.StreamOptions.IncludeUsage {
				t.Errorf("stream_options.include_usage = false, want true")
			}
		})
	}
}

// --- TestEncodeRequest_Messages ---

func TestEncodeRequest_Messages(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		msgs    content.AgenticMessages
		checkFn func(t *testing.T, msgs []map[string]json.RawMessage)
	}{
		{
			name: "text-only user message: content is string",
			msgs: content.AgenticMessages{userMsg(textBlock("hello world"))},
			checkFn: func(t *testing.T, msgs []map[string]json.RawMessage) {
				t.Helper()
				if len(msgs) < 1 {
					t.Fatal("expected at least 1 message")
				}
				s := contentStr(t, msgs[0])
				if s != "hello world" {
					t.Errorf("content = %q, want %q", s, "hello world")
				}
			},
		},
		{
			name: "mixed user message (text + image URL): content is []chatContentPart",
			msgs: content.AgenticMessages{userMsg(textBlock("look"), imageURLBlock("https://example.com/img.jpg"))},
			checkFn: func(t *testing.T, msgs []map[string]json.RawMessage) {
				t.Helper()
				if len(msgs) < 1 {
					t.Fatal("expected at least 1 message")
				}
				var parts []map[string]json.RawMessage
				if err := json.Unmarshal(msgs[0]["content"], &parts); err != nil {
					t.Fatalf("expected content to be array: %v", err)
				}
				if len(parts) != 2 {
					t.Fatalf("expected 2 parts, got %d", len(parts))
				}
				var typ0 string
				if err := json.Unmarshal(parts[0]["type"], &typ0); err != nil {
					t.Fatalf("failed to unmarshal parts[0].type: %v", err)
				}
				if typ0 != "text" {
					t.Errorf("parts[0].type = %q, want \"text\"", typ0)
				}
				var typ1 string
				if err := json.Unmarshal(parts[1]["type"], &typ1); err != nil {
					t.Fatalf("failed to unmarshal parts[1].type: %v", err)
				}
				if typ1 != "image_url" {
					t.Errorf("parts[1].type = %q, want \"image_url\"", typ1)
				}
			},
		},
		{
			name: "AI message with text: content is string",
			msgs: content.AgenticMessages{aiMsg(textBlock("I am the AI"))},
			checkFn: func(t *testing.T, msgs []map[string]json.RawMessage) {
				t.Helper()
				if len(msgs) < 1 {
					t.Fatal("expected at least 1 message")
				}
				s := contentStr(t, msgs[0])
				if s != "I am the AI" {
					t.Errorf("content = %q, want %q", s, "I am the AI")
				}
			},
		},
		{
			name: "AI message with tool call: has tool_calls, content is empty string",
			msgs: content.AgenticMessages{aiMsg(toolUseBlock("call-1", "my_tool", json.RawMessage(`{"key":"val"}`)))},
			checkFn: func(t *testing.T, msgs []map[string]json.RawMessage) {
				t.Helper()
				if len(msgs) < 1 {
					t.Fatal("expected at least 1 message")
				}
				tcRaw, ok := msgs[0]["tool_calls"]
				if !ok {
					t.Fatal("expected tool_calls key")
				}
				var tc []map[string]json.RawMessage
				if err := json.Unmarshal(tcRaw, &tc); err != nil {
					t.Fatalf("unmarshal tool_calls: %v", err)
				}
				if len(tc) != 1 {
					t.Fatalf("expected 1 tool call, got %d", len(tc))
				}
				s := contentStr(t, msgs[0])
				if s != "" {
					t.Errorf("content = %q, want empty string", s)
				}
			},
		},
		{
			name: "tool message: role=tool, has tool_call_id",
			msgs: content.AgenticMessages{toolMsg("call-99", textBlock("result"))},
			checkFn: func(t *testing.T, msgs []map[string]json.RawMessage) {
				t.Helper()
				if len(msgs) < 1 {
					t.Fatal("expected at least 1 message")
				}
				if r := roleOf(t, msgs[0]); r != "tool" {
					t.Errorf("role = %q, want \"tool\"", r)
				}
				var id string
				if err := json.Unmarshal(msgs[0]["tool_call_id"], &id); err != nil {
					t.Fatalf("unmarshal tool_call_id: %v", err)
				}
				if id != "call-99" {
					t.Errorf("tool_call_id = %q, want \"call-99\"", id)
				}
			},
		},
		{
			name: "system message in conversation: role=system",
			msgs: content.AgenticMessages{sysMsg(textBlock("Be concise."))},
			checkFn: func(t *testing.T, msgs []map[string]json.RawMessage) {
				t.Helper()
				if len(msgs) < 1 {
					t.Fatal("expected at least 1 message")
				}
				if r := roleOf(t, msgs[0]); r != "system" {
					t.Errorf("role = %q, want \"system\"", r)
				}
			},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			req := inference.Request{
				Model:    model.Model{Name: "m", Caps: model.Capabilities{AcceptsImages: true}},
				Messages: tc.msgs,
			}
			got, err := openaiapi.EncodeRequest(req, false)
			if err != nil {
				t.Fatalf("EncodeRequest error: %v", err)
			}

			raw := mustDecode(t, got)
			msgs := messagesFromRaw(t, raw)
			tc.checkFn(t, msgs)
		})
	}
}

// --- TestEncodeRequest_ToolResultErrorReachesModel ---

// TestEncodeRequest_ToolResultErrorReachesModel locks the IsError reconciliation:
// the OpenAI Chat Completions tool message has NO structured error flag, so
// ToolResultMessage.IsError is intentionally NOT emitted on the request. The model
// learns a tool errored only through the result's text content (the loop
// error-prefixes it). The encoded tool message must carry role=tool, the
// tool_call_id, and the error text as content — and must NOT carry an is_error field.
func TestEncodeRequest_ToolResultErrorReachesModel(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name        string
		msg         *content.ToolResultMessage
		wantContent string
	}{
		{
			name: "error result: IsError true, error text reaches model",
			msg: func() *content.ToolResultMessage {
				m := toolMsg("call-err", textBlock("tool error: boom"))
				m.IsError = true
				return m
			}(),
			wantContent: "tool error: boom",
		},
		{
			name:        "success result: IsError false, result text reaches model",
			msg:         toolMsg("call-ok", textBlock("ok")),
			wantContent: "ok",
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			req := inference.Request{
				Model:    model.Model{Name: "m"},
				Messages: content.AgenticMessages{tc.msg},
			}
			got, err := openaiapi.EncodeRequest(req, false)
			if err != nil {
				t.Fatalf("EncodeRequest error: %v", err)
			}

			raw := mustDecode(t, got)
			msgs := messagesFromRaw(t, raw)
			if len(msgs) != 1 {
				t.Fatalf("expected exactly 1 message, got %d", len(msgs))
			}
			if r := roleOf(t, msgs[0]); r != "tool" {
				t.Errorf("role = %q, want \"tool\"", r)
			}
			var id string
			if err := json.Unmarshal(msgs[0]["tool_call_id"], &id); err != nil {
				t.Fatalf("unmarshal tool_call_id: %v", err)
			}
			if id != tc.msg.ToolUseID {
				t.Errorf("tool_call_id = %q, want %q", id, tc.msg.ToolUseID)
			}
			if s := contentStr(t, msgs[0]); s != tc.wantContent {
				t.Errorf("content = %q, want %q (error text must reach the model)", s, tc.wantContent)
			}
			// The OpenAI schema has no is_error field on a tool message: it must NOT
			// be emitted, regardless of ToolResultMessage.IsError.
			if _, ok := msgs[0]["is_error"]; ok {
				t.Error("tool message carries a non-standard is_error field; OpenAI Chat Completions has no such field")
			}
		})
	}
}

// --- TestEncodeRequest_Tools ---

// TestEncodeRequest_ToolWithoutSchemaSendsAnEmptyObjectSchema pins the
// parameterless-tool case. FunctionObject.parameters is spec-typed `object`
// with no null alternative, so passing an inference.Tool with no Schema
// through verbatim produced `"parameters":null` — a body OpenAI's own schema
// rejects. openairesponses already substituted {"type":"object"} here
// (schemaOrDefault); this dialect did not, and the two must agree.
func TestEncodeRequest_ToolWithoutSchemaSendsAnEmptyObjectSchema(t *testing.T) {
	t.Parallel()

	body, err := openaiapi.EncodeRequest(inference.Request{
		Model:    model.Model{Name: "m"},
		Messages: content.AgenticMessages{userMsg(textBlock("hello"))},
		Tools:    []inference.Tool{{Name: "ping", Description: "no arguments"}},
	}, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}

	var decoded struct {
		Tools []struct {
			Function struct {
				Parameters json.RawMessage `json:"parameters"`
			} `json:"function"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	if len(decoded.Tools) != 1 {
		t.Fatalf("tools = %d, want 1\n%s", len(decoded.Tools), body)
	}
	if got := string(decoded.Tools[0].Function.Parameters); got != `{"type":"object"}` {
		t.Errorf("parameters = %s, want {\"type\":\"object\"}", got)
	}
}

func TestEncodeRequest_Tools(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		tools     []inference.Tool
		wantKey   bool
		wantCount int
	}{
		{
			name:    "no tools: wire has no tools key",
			tools:   nil,
			wantKey: false,
		},
		{
			name: "one tool: wire has tools array",
			tools: []inference.Tool{
				{Name: "search", Description: "search the web", Schema: json.RawMessage(`{"type":"object","properties":{}}`)},
			},
			wantKey:   true,
			wantCount: 1,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			req := inference.Request{
				Model:    model.Model{Name: "m"},
				Messages: content.AgenticMessages{userMsg(textBlock("hello"))},
				Tools:    tc.tools,
			}
			got, err := openaiapi.EncodeRequest(req, false)
			if err != nil {
				t.Fatalf("EncodeRequest error: %v", err)
			}

			raw := mustDecode(t, got)
			toolsRaw, exists := raw["tools"]
			if tc.wantKey && !exists {
				t.Fatal("expected tools key in JSON")
			}
			if !tc.wantKey && exists {
				t.Fatal("expected no tools key in JSON")
			}
			if tc.wantKey {
				var tools []map[string]json.RawMessage
				if err := json.Unmarshal(toolsRaw, &tools); err != nil {
					t.Fatalf("unmarshal tools: %v", err)
				}
				if len(tools) != tc.wantCount {
					t.Errorf("tools count = %d, want %d", len(tools), tc.wantCount)
				}
				// Verify shape: type and function fields exist
				var typ string
				if err := json.Unmarshal(tools[0]["type"], &typ); err != nil {
					t.Fatalf("failed to unmarshal tool type: %v", err)
				}
				if typ != "function" {
					t.Errorf("tool type = %q, want \"function\"", typ)
				}
				if _, ok := tools[0]["function"]; !ok {
					t.Error("expected function key in tool")
				}
			}
		})
	}
}

// --- TestEncodeRequest_ThinkingIgnored ---

func TestEncodeRequest_ThinkingIgnored(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		msgs content.AgenticMessages
	}{
		{
			name: "thinking block alone: content empty string, no tool_calls",
			msgs: content.AgenticMessages{aiMsg(thinkingBlock("secret thoughts"))},
		},
		{
			name: "thinking block mixed with text: only text survives",
			msgs: content.AgenticMessages{aiMsg(thinkingBlock("hidden"), textBlock("visible"))},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			req := inference.Request{
				Model:    model.Model{Name: "m"},
				Messages: tc.msgs,
			}
			got, err := openaiapi.EncodeRequest(req, false)
			if err != nil {
				t.Fatalf("EncodeRequest error: %v", err)
			}

			raw := mustDecode(t, got)
			msgs := messagesFromRaw(t, raw)
			if len(msgs) < 1 {
				t.Fatal("expected at least 1 message")
			}

			// Content must not contain thinking text
			contentBytes := msgs[0]["content"]
			var contentStr string
			if err := json.Unmarshal(contentBytes, &contentStr); err == nil {
				if contentStr == "secret thoughts" || contentStr == "hidden" {
					t.Errorf("thinking text leaked into content: %q", contentStr)
				}
			}

			// No tool_calls
			if _, ok := msgs[0]["tool_calls"]; ok {
				t.Error("unexpected tool_calls key for thinking-only message")
			}
		})
	}
}

// --- TestEncodeRequest_ImageBlock_DataURL ---

func TestEncodeRequest_ImageBlock_DataURL(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		mediaType content.MediaType
		data      []byte
	}{
		{
			name:      "PNG data becomes data URI",
			mediaType: content.MediaTypeImagePNG,
			data:      []byte{0x89, 0x50, 0x4E, 0x47},
		},
		{
			name:      "JPEG data becomes data URI",
			mediaType: content.MediaTypeImageJPEG,
			data:      []byte{0xFF, 0xD8, 0xFF},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			req := inference.Request{
				Model:    model.Model{Name: "m", Caps: model.Capabilities{AcceptsImages: true}},
				Messages: content.AgenticMessages{userMsg(imageDataBlock(tc.mediaType, tc.data))},
			}
			got, err := openaiapi.EncodeRequest(req, false)
			if err != nil {
				t.Fatalf("EncodeRequest error: %v", err)
			}

			raw := mustDecode(t, got)
			msgs := messagesFromRaw(t, raw)
			if len(msgs) < 1 {
				t.Fatal("expected at least 1 message")
			}

			var parts []map[string]json.RawMessage
			if err := json.Unmarshal(msgs[0]["content"], &parts); err != nil {
				t.Fatalf("expected content to be array: %v", err)
			}

			found := false
			for _, p := range parts {
				var typ string
				if err := json.Unmarshal(p["type"], &typ); err != nil {
					t.Fatalf("failed to unmarshal part type: %v", err)
				}
				if typ != "image_url" {
					continue
				}
				found = true
				var imgURL map[string]json.RawMessage
				if err := json.Unmarshal(p["image_url"], &imgURL); err != nil {
					t.Fatalf("failed to unmarshal image_url object: %v", err)
				}
				var urlStr string
				if err := json.Unmarshal(imgURL["url"], &urlStr); err != nil {
					t.Fatalf("failed to unmarshal url string: %v", err)
				}

				expectedPrefix := "data:" + string(tc.mediaType) + ";base64,"
				if len(urlStr) < len(expectedPrefix) || urlStr[:len(expectedPrefix)] != expectedPrefix {
					snippet := urlStr
					if len(snippet) > len(expectedPrefix)+10 {
						snippet = snippet[:len(expectedPrefix)+10]
					}
					t.Errorf("URL prefix = %q, want prefix %q", snippet, expectedPrefix)
				}
				expectedB64 := base64.StdEncoding.EncodeToString(tc.data)
				if urlStr != expectedPrefix+expectedB64 {
					t.Errorf("data URI = %q, want %q", urlStr, expectedPrefix+expectedB64)
				}
			}
			if !found {
				t.Error("no image_url part found in content")
			}
		})
	}
}

// --- TestEncodeRequest_ValidJSON ---

func TestEncodeRequest_ValidJSON(t *testing.T) {
	t.Parallel()

	temp := 0.7
	maxTok := 100
	cases := []struct {
		name string
		req  inference.Request
	}{
		{
			name: "minimal request",
			req: inference.Request{
				Model:    model.Model{Name: "m"},
				Messages: content.AgenticMessages{userMsg(textBlock("hi"))},
			},
		},
		{
			name: "full request with tools and system",
			req: inference.Request{
				Model: model.Model{
					Name: "gpt-4o",
					Caps: model.Capabilities{AcceptsImages: true},
					Sampling: model.Sampling{
						Temperature: &temp,
						MaxTokens:   &maxTok,
						Stop:        []string{"STOP"},
					},
				},
				System: "Be helpful.",
				Messages: content.AgenticMessages{
					userMsg(textBlock("hello"), imageURLBlock("https://x.com/img.jpg")),
					aiMsg(textBlock("hi there")),
					toolMsg("id1", textBlock("result")),
				},
				Tools: []inference.Tool{
					{Name: "calc", Description: "math", Schema: json.RawMessage(`{"type":"object"}`)},
				},
			},
		},
		{
			name: "stream=true",
			req: inference.Request{
				Model:    model.Model{Name: "m"},
				Messages: content.AgenticMessages{userMsg(textBlock("stream me"))},
			},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			stream := tc.name == "stream=true"
			got, err := openaiapi.EncodeRequest(tc.req, stream)
			if err != nil {
				t.Fatalf("EncodeRequest error: %v", err)
			}
			if !json.Valid(got) {
				t.Errorf("output is not valid JSON: %s", got)
			}
		})
	}
}

// --- TestEncodeRequest_Sampling ---

// TestEncodeRequest_Sampling locks the reshaped-domain sampling migration:
//   - effective sampling is Request.Override when non-nil, else Model.Sampling;
//   - Temperature/TopP/MaxTokens/Stop from the effective Sampling reach the wire;
//   - model.Effort maps to the reasoning_effort wire value, with EffortNone (and
//     any unknown value, fail-safe) omitting the field and EffortMax clamping to
//     "high" (OpenAI's o-series accepts only low|medium|high — there is no "max").
func TestEncodeRequest_Sampling(t *testing.T) {
	t.Parallel()

	temp := 0.3
	topP := 0.9
	maxTok := 256
	stopVals := []string{"STOP", "END"}
	overrideTemp := 0.99

	cases := []struct {
		name          string
		model         model.Model
		override      *model.Sampling
		wantTemp      *float64 // nil = temperature key must be absent
		wantTopP      *float64 // nil = top_p key must be absent
		wantMaxTokens *int     // nil = max_tokens key must be absent
		wantStop      []string // nil = stop key must be absent
		wantEffort    string   // "" = reasoning_effort key must be absent
	}{
		{
			name:          "model sampling: temperature/top_p/max_tokens/stop on wire",
			model:         model.Model{Name: "m", Sampling: model.Sampling{Temperature: &temp, TopP: &topP, MaxTokens: &maxTok, Stop: stopVals}},
			wantTemp:      &temp,
			wantTopP:      &topP,
			wantMaxTokens: &maxTok,
			wantStop:      stopVals,
		},
		{
			name:       "effort none omits reasoning_effort",
			model:      model.Model{Name: "m", Sampling: model.Sampling{Effort: model.EffortNone}},
			wantEffort: "",
		},
		{
			name:       "effort low maps to low",
			model:      model.Model{Name: "m", Sampling: model.Sampling{Effort: model.EffortLow}},
			wantEffort: "low",
		},
		{
			name:       "effort medium maps to medium",
			model:      model.Model{Name: "m", Sampling: model.Sampling{Effort: model.EffortMedium}},
			wantEffort: "medium",
		},
		{
			name:       "effort high maps to high",
			model:      model.Model{Name: "m", Sampling: model.Sampling{Effort: model.EffortHigh}},
			wantEffort: "high",
		},
		{
			name:       "effort max clamps to high",
			model:      model.Model{Name: "m", Sampling: model.Sampling{Effort: model.EffortMax}},
			wantEffort: "high",
		},
		{
			// Unknown enum value hits the fail-safe default branch: omit rather
			// than forward a value the server would reject.
			name:       "unknown effort value omits reasoning_effort",
			model:      model.Model{Name: "m", Sampling: model.Sampling{Effort: model.Effort("garbage")}},
			wantEffort: "",
		},
		{
			name:       "override wins over model sampling",
			model:      model.Model{Name: "m", Sampling: model.Sampling{Temperature: &temp, Effort: model.EffortLow}},
			override:   &model.Sampling{Temperature: &overrideTemp, Effort: model.EffortMax},
			wantTemp:   &overrideTemp,
			wantEffort: "high",
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			req := inference.Request{
				Model:    tc.model,
				Override: tc.override,
				Messages: content.AgenticMessages{userMsg(textBlock("hi"))},
			}
			got, err := openaiapi.EncodeRequest(req, false)
			if err != nil {
				t.Fatalf("EncodeRequest error: %v", err)
			}
			raw := mustDecode(t, got)

			// reasoning_effort: mapped from Effort; absent for EffortNone.
			effRaw, effOK := raw["reasoning_effort"]
			if tc.wantEffort == "" {
				if effOK {
					t.Errorf("reasoning_effort present (%s), want absent", effRaw)
				}
			} else {
				if !effOK {
					t.Fatal("reasoning_effort absent, want present")
				}
				var eff string
				if err := json.Unmarshal(effRaw, &eff); err != nil {
					t.Fatalf("unmarshal reasoning_effort: %v", err)
				}
				if eff != tc.wantEffort {
					t.Errorf("reasoning_effort = %q, want %q", eff, tc.wantEffort)
				}
			}

			// temperature: from the effective sampling (override or model).
			if tc.wantTemp != nil {
				tRaw, ok := raw["temperature"]
				if !ok {
					t.Fatal("temperature absent, want present")
				}
				var tv float64
				if err := json.Unmarshal(tRaw, &tv); err != nil {
					t.Fatalf("unmarshal temperature: %v", err)
				}
				if tv != *tc.wantTemp {
					t.Errorf("temperature = %v, want %v", tv, *tc.wantTemp)
				}
			}

			// top_p: from the effective sampling.
			if tc.wantTopP != nil {
				pRaw, ok := raw["top_p"]
				if !ok {
					t.Fatal("top_p absent, want present")
				}
				var pv float64
				if err := json.Unmarshal(pRaw, &pv); err != nil {
					t.Fatalf("unmarshal top_p: %v", err)
				}
				if pv != *tc.wantTopP {
					t.Errorf("top_p = %v, want %v", pv, *tc.wantTopP)
				}
			}

			// max_tokens: from the effective sampling.
			if tc.wantMaxTokens != nil {
				mRaw, ok := raw["max_tokens"]
				if !ok {
					t.Fatal("max_tokens absent, want present")
				}
				var mv int
				if err := json.Unmarshal(mRaw, &mv); err != nil {
					t.Fatalf("unmarshal max_tokens: %v", err)
				}
				if mv != *tc.wantMaxTokens {
					t.Errorf("max_tokens = %d, want %d", mv, *tc.wantMaxTokens)
				}
			}

			// stop: from the effective sampling (array must match element-wise).
			if tc.wantStop != nil {
				sRaw, ok := raw["stop"]
				if !ok {
					t.Fatal("stop absent, want present")
				}
				var sv []string
				if err := json.Unmarshal(sRaw, &sv); err != nil {
					t.Fatalf("unmarshal stop: %v", err)
				}
				if len(sv) != len(tc.wantStop) {
					t.Fatalf("stop len = %d, want %d (%v)", len(sv), len(tc.wantStop), sv)
				}
				for i, want := range tc.wantStop {
					if sv[i] != want {
						t.Errorf("stop[%d] = %q, want %q", i, sv[i], want)
					}
				}
			}
		})
	}
}

// --- TestEncodeRequest_MaxCompletionTokens ---

// TestEncodeRequest_MaxCompletionTokens locks the token-limit field selection.
// OpenAI's own spec marks CreateChatCompletionRequest.max_tokens deprecated and
// "not compatible with o-series models", replacing it with
// max_completion_tokens; gpt-5 / o-series reject a request carrying max_tokens.
// The two fields are therefore selected on the model's advertised Thinking
// capability rather than migrated unconditionally: openaiapi is the shared Chat
// dialect for the OpenAI-compatible ecosystem, and older llama.cpp / Ollama
// builds still accept only max_tokens. Exactly one of the two is ever emitted.
func TestEncodeRequest_MaxCompletionTokens(t *testing.T) {
	t.Parallel()

	maxTok := 256

	cases := []struct {
		name      string
		caps      model.Capabilities
		wantKey   string
		absentKey string
	}{
		{
			name:      "thinking model uses max_completion_tokens",
			caps:      model.Capabilities{Thinking: true},
			wantKey:   "max_completion_tokens",
			absentKey: "max_tokens",
		},
		{
			name:      "non-thinking model keeps deprecated max_tokens",
			caps:      model.Capabilities{},
			wantKey:   "max_tokens",
			absentKey: "max_completion_tokens",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			req := inference.Request{
				Model: model.Model{Name: "m", Caps: tc.caps, Sampling: model.Sampling{MaxTokens: &maxTok}},
			}
			body, err := openaiapi.EncodeRequest(req, false)
			if err != nil {
				t.Fatalf("EncodeRequest() error = %v", err)
			}
			raw := mustDecode(t, body)
			got, ok := raw[tc.wantKey]
			if !ok {
				t.Fatalf("%s absent, want present (body=%s)", tc.wantKey, body)
			}
			var v int
			if err := json.Unmarshal(got, &v); err != nil {
				t.Fatalf("unmarshal %s: %v", tc.wantKey, err)
			}
			if v != maxTok {
				t.Errorf("%s = %d, want %d", tc.wantKey, v, maxTok)
			}
			if _, present := raw[tc.absentKey]; present {
				t.Errorf("%s must not be emitted alongside %s (body=%s)", tc.absentKey, tc.wantKey, body)
			}
		})
	}
}

// TestEncodeRequest_MaxCompletionTokensOmittedWhenUnset confirms neither
// token-limit field appears when the effective sampling carries no limit.
func TestEncodeRequest_MaxCompletionTokensOmittedWhenUnset(t *testing.T) {
	t.Parallel()

	req := inference.Request{Model: model.Model{Name: "m", Caps: model.Capabilities{Thinking: true}}}
	body, err := openaiapi.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	raw := mustDecode(t, body)
	for _, key := range []string{"max_tokens", "max_completion_tokens"} {
		if _, present := raw[key]; present {
			t.Errorf("%s present with no MaxTokens set (body=%s)", key, body)
		}
	}
}
