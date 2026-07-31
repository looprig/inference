package geminiapi_test

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/geminiapi"
	model "github.com/looprig/inference/model"
)

// --- shared helpers (used across the package's test files) ---

// mustDecode unmarshals raw JSON into a map for field inspection.
func mustDecode(t *testing.T, data []byte) map[string]json.RawMessage {
	t.Helper()
	var m map[string]json.RawMessage
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return m
}

// contentsFromRaw pulls the `contents` array out as inspectable maps.
func contentsFromRaw(t *testing.T, raw map[string]json.RawMessage) []map[string]json.RawMessage {
	t.Helper()
	var cs []map[string]json.RawMessage
	if err := json.Unmarshal(raw["contents"], &cs); err != nil {
		t.Fatalf("unmarshal contents: %v", err)
	}
	return cs
}

// partsOf pulls the `parts` array out of a content entry.
func partsOf(t *testing.T, entry map[string]json.RawMessage) []map[string]json.RawMessage {
	t.Helper()
	var ps []map[string]json.RawMessage
	if err := json.Unmarshal(entry["parts"], &ps); err != nil {
		t.Fatalf("unmarshal parts: %v", err)
	}
	return ps
}

func roleOf(t *testing.T, entry map[string]json.RawMessage) string {
	t.Helper()
	var r string
	if err := json.Unmarshal(entry["role"], &r); err != nil {
		t.Fatalf("unmarshal role: %v", err)
	}
	return r
}

func strField(t *testing.T, m map[string]json.RawMessage, key string) string {
	t.Helper()
	var s string
	if err := json.Unmarshal(m[key], &s); err != nil {
		t.Fatalf("unmarshal %q as string: %v", key, err)
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

func textBlock(s string) content.Block { return &content.TextBlock{Text: s} }

func imageURLBlock(url string) content.Block {
	return &content.ImageBlock{MediaType: content.MediaTypeImageJPEG, Source: content.ImageSource{URL: url}}
}

func imageDataBlock(mt content.MediaType, data []byte) content.Block {
	return &content.ImageBlock{MediaType: mt, Source: content.ImageSource{Data: data}}
}

func thinkingBlock(s string) content.Block { return &content.ThinkingBlock{Thinking: s} }

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

	want, err := geminiapi.EncodeRequest(inference.Request{Model: model.Model{Name: "m"}, Messages: content.AgenticMessages{aiMsg(textBlock("answer"))}})
	if err != nil {
		t.Fatalf("EncodeRequest() baseline error = %v", err)
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			message := aiMsg(textBlock("answer"))
			message.Usage = tt.usage
			got, err := geminiapi.EncodeRequest(inference.Request{Model: model.Model{Name: "m"}, Messages: content.AgenticMessages{message}})
			if err != nil {
				t.Fatalf("EncodeRequest() error = %v", err)
			}
			if string(got) != string(want) {
				t.Errorf("EncodeRequest() = %s, want byte-identical %s", got, want)
			}
		})
	}
}

func toolUseBlock(id, name string, input json.RawMessage) content.Block {
	return &content.ToolUseBlock{ID: id, Name: name, Input: input}
}

// --- TestEncodeRequest_SystemInstruction ---

func TestEncodeRequest_SystemInstruction(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name       string
		system     string
		messages   content.AgenticMessages
		wantSysKey bool
		wantSysTxt string
	}{
		{
			name:       "system from Request becomes systemInstruction",
			system:     "You are helpful.",
			messages:   content.AgenticMessages{userMsg(textBlock("hi"))},
			wantSysKey: true,
			wantSysTxt: "You are helpful.",
		},
		{
			name:       "no system: systemInstruction omitted",
			system:     "",
			messages:   content.AgenticMessages{userMsg(textBlock("hi"))},
			wantSysKey: false,
		},
		{
			name:       "in-thread SystemMessage folds into systemInstruction",
			system:     "",
			messages:   content.AgenticMessages{sysMsg(textBlock("Be concise.")), userMsg(textBlock("hi"))},
			wantSysKey: true,
			wantSysTxt: "Be concise.",
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			req := inference.Request{Model: model.Model{Name: "m"}, System: tc.system, Messages: tc.messages}
			got, err := geminiapi.EncodeRequest(req)
			if err != nil {
				t.Fatalf("EncodeRequest error: %v", err)
			}
			raw := mustDecode(t, got)

			sysRaw, ok := raw["systemInstruction"]
			if ok != tc.wantSysKey {
				t.Fatalf("systemInstruction present=%v, want %v (%s)", ok, tc.wantSysKey, sysRaw)
			}
			if !tc.wantSysKey {
				return
			}
			var sys map[string]json.RawMessage
			if err := json.Unmarshal(sysRaw, &sys); err != nil {
				t.Fatalf("unmarshal systemInstruction: %v", err)
			}
			// systemInstruction must have no role and its first part must be the text.
			if _, hasRole := sys["role"]; hasRole {
				t.Error("systemInstruction should not carry a role")
			}
			parts := partsOf(t, sys)
			if len(parts) == 0 {
				t.Fatal("expected at least one systemInstruction part")
			}
			if txt := strField(t, parts[0], "text"); txt != tc.wantSysTxt {
				t.Errorf("systemInstruction text = %q, want %q", txt, tc.wantSysTxt)
			}
		})
	}
}

// --- TestEncodeRequest_Roles ---

// TestEncodeRequest_Roles locks the Gemini role mapping: user -> "user",
// assistant -> "model", and a tool result -> a "user" turn holding a
// functionResponse (Gemini has no "assistant"/"tool"/"system" roles in contents).
func TestEncodeRequest_Roles(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name     string
		msgs     content.AgenticMessages
		wantRole string
	}{
		{name: "user message -> user", msgs: content.AgenticMessages{userMsg(textBlock("hi"))}, wantRole: "user"},
		{name: "assistant message -> model", msgs: content.AgenticMessages{aiMsg(textBlock("yo"))}, wantRole: "model"},
		{name: "tool result -> user", msgs: content.AgenticMessages{toolMsg("id1", textBlock("ok"))}, wantRole: "user"},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			req := inference.Request{Model: model.Model{Name: "m"}, Messages: tc.msgs}
			got, err := geminiapi.EncodeRequest(req)
			if err != nil {
				t.Fatalf("EncodeRequest error: %v", err)
			}
			contents := contentsFromRaw(t, mustDecode(t, got))
			if len(contents) != 1 {
				t.Fatalf("expected 1 content entry, got %d", len(contents))
			}
			if r := roleOf(t, contents[0]); r != tc.wantRole {
				t.Errorf("role = %q, want %q", r, tc.wantRole)
			}
		})
	}
}

// --- TestEncodeRequest_FunctionCall ---

func TestEncodeRequest_FunctionCall(t *testing.T) {
	t.Parallel()

	req := inference.Request{
		Model: model.Model{Name: "m"},
		Messages: content.AgenticMessages{
			aiMsg(toolUseBlock("call-1", "get_weather", json.RawMessage(`{"location":"Boston"}`))),
		},
	}
	got, err := geminiapi.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest error: %v", err)
	}
	contents := contentsFromRaw(t, mustDecode(t, got))
	if len(contents) != 1 {
		t.Fatalf("expected 1 content entry, got %d", len(contents))
	}
	if r := roleOf(t, contents[0]); r != "model" {
		t.Errorf("role = %q, want model", r)
	}
	parts := partsOf(t, contents[0])
	if len(parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(parts))
	}
	fcRaw, ok := parts[0]["functionCall"]
	if !ok {
		t.Fatal("expected functionCall part")
	}
	var fc map[string]json.RawMessage
	if err := json.Unmarshal(fcRaw, &fc); err != nil {
		t.Fatalf("unmarshal functionCall: %v", err)
	}
	if name := strField(t, fc, "name"); name != "get_weather" {
		t.Errorf("functionCall.name = %q, want get_weather", name)
	}
	if id := strField(t, fc, "id"); id != "call-1" {
		t.Errorf("functionCall.id = %q, want call-1", id)
	}
	var args map[string]string
	if err := json.Unmarshal(fc["args"], &args); err != nil {
		t.Fatalf("unmarshal functionCall.args: %v", err)
	}
	if args["location"] != "Boston" {
		t.Errorf("functionCall.args.location = %q, want Boston", args["location"])
	}
}

// --- TestEncodeRequest_FunctionResponse ---

// TestEncodeRequest_FunctionResponse locks the tool-result mapping: a
// ToolResultMessage becomes a functionResponse whose `name` is looked up from the
// preceding tool call (Gemini matches on name), whose `id` echoes the tool-use id,
// and whose `response` is a JSON object wrapping the result text. IsError must NOT
// appear (the classic functionResponse has no error flag).
func TestEncodeRequest_FunctionResponse(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name       string
		msgs       content.AgenticMessages
		wantName   string
		wantID     string
		wantResult string
	}{
		{
			name: "name resolved from preceding tool call",
			msgs: content.AgenticMessages{
				aiMsg(toolUseBlock("call-9", "search", json.RawMessage(`{}`))),
				toolMsg("call-9", textBlock("42 results")),
			},
			wantName:   "search",
			wantID:     "call-9",
			wantResult: "42 results",
		},
		{
			name: "error result: text reaches model, no error flag",
			msgs: content.AgenticMessages{
				aiMsg(toolUseBlock("call-e", "run", json.RawMessage(`{}`))),
				func() *content.ToolResultMessage {
					m := toolMsg("call-e", textBlock("tool error: boom"))
					m.IsError = true
					return m
				}(),
			},
			wantName:   "run",
			wantID:     "call-e",
			wantResult: "tool error: boom",
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			req := inference.Request{Model: model.Model{Name: "m"}, Messages: tc.msgs}
			got, err := geminiapi.EncodeRequest(req)
			if err != nil {
				t.Fatalf("EncodeRequest error: %v", err)
			}
			contents := contentsFromRaw(t, mustDecode(t, got))
			// last content entry is the tool result turn
			last := contents[len(contents)-1]
			if r := roleOf(t, last); r != "user" {
				t.Errorf("functionResponse turn role = %q, want user", r)
			}
			parts := partsOf(t, last)
			frRaw, ok := parts[0]["functionResponse"]
			if !ok {
				t.Fatal("expected functionResponse part")
			}
			var fr map[string]json.RawMessage
			if err := json.Unmarshal(frRaw, &fr); err != nil {
				t.Fatalf("unmarshal functionResponse: %v", err)
			}
			if name := strField(t, fr, "name"); name != tc.wantName {
				t.Errorf("functionResponse.name = %q, want %q", name, tc.wantName)
			}
			if id := strField(t, fr, "id"); id != tc.wantID {
				t.Errorf("functionResponse.id = %q, want %q", id, tc.wantID)
			}
			var resp map[string]string
			if err := json.Unmarshal(fr["response"], &resp); err != nil {
				t.Fatalf("unmarshal functionResponse.response: %v", err)
			}
			if resp["result"] != tc.wantResult {
				t.Errorf("functionResponse.response.result = %q, want %q", resp["result"], tc.wantResult)
			}
			if _, ok := fr["is_error"]; ok {
				t.Error("functionResponse carries a non-standard is_error field")
			}
		})
	}
}

// --- TestEncodeRequest_ImageInlineData ---

func TestEncodeRequest_ImageInlineData(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		mediaType content.MediaType
		data      []byte
	}{
		{name: "PNG bytes -> inlineData", mediaType: content.MediaTypeImagePNG, data: []byte{0x89, 0x50, 0x4E, 0x47}},
		{name: "JPEG bytes -> inlineData", mediaType: content.MediaTypeImageJPEG, data: []byte{0xFF, 0xD8, 0xFF}},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			req := inference.Request{
				Model:    model.Model{Name: "m", Caps: model.Capabilities{AcceptsImages: true}},
				Messages: content.AgenticMessages{userMsg(textBlock("look"), imageDataBlock(tc.mediaType, tc.data))},
			}
			got, err := geminiapi.EncodeRequest(req)
			if err != nil {
				t.Fatalf("EncodeRequest error: %v", err)
			}
			contents := contentsFromRaw(t, mustDecode(t, got))
			parts := partsOf(t, contents[0])
			if len(parts) != 2 {
				t.Fatalf("expected 2 parts (text + image), got %d", len(parts))
			}
			// parts[0] is the text; parts[1] is the inlineData image, order preserved.
			if txt := strField(t, parts[0], "text"); txt != "look" {
				t.Errorf("parts[0].text = %q, want look", txt)
			}
			idRaw, ok := parts[1]["inlineData"]
			if !ok {
				t.Fatalf("expected inlineData part, got %v", parts[1])
			}
			var id map[string]json.RawMessage
			if err := json.Unmarshal(idRaw, &id); err != nil {
				t.Fatalf("unmarshal inlineData: %v", err)
			}
			if mt := strField(t, id, "mimeType"); mt != string(tc.mediaType) {
				t.Errorf("inlineData.mimeType = %q, want %q", mt, tc.mediaType)
			}
			if data := strField(t, id, "data"); data != base64.StdEncoding.EncodeToString(tc.data) {
				t.Errorf("inlineData.data = %q, want base64 of input", data)
			}
		})
	}
}

// --- TestEncodeRequest_ImageURLFileData ---

// A URL-sourced image (no inline bytes) degrades to a fileData part.
func TestEncodeRequest_ImageURLFileData(t *testing.T) {
	t.Parallel()

	req := inference.Request{
		Model:    model.Model{Name: "m", Caps: model.Capabilities{AcceptsImages: true}},
		Messages: content.AgenticMessages{userMsg(imageURLBlock("https://example.com/x.jpg"))},
	}
	got, err := geminiapi.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest error: %v", err)
	}
	contents := contentsFromRaw(t, mustDecode(t, got))
	parts := partsOf(t, contents[0])
	fdRaw, ok := parts[0]["fileData"]
	if !ok {
		t.Fatalf("expected fileData part, got %v", parts[0])
	}
	var fd map[string]json.RawMessage
	if err := json.Unmarshal(fdRaw, &fd); err != nil {
		t.Fatalf("unmarshal fileData: %v", err)
	}
	if uri := strField(t, fd, "fileUri"); uri != "https://example.com/x.jpg" {
		t.Errorf("fileData.fileUri = %q, want the image URL", uri)
	}
}

// --- TestEncodeRequest_Tools ---

func TestEncodeRequest_Tools(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		tools     []inference.Tool
		wantKey   bool
		wantDecls int
	}{
		{name: "no tools: tools key absent", tools: nil, wantKey: false},
		{
			name: "two tools grouped under one functionDeclarations",
			tools: []inference.Tool{
				{Name: "search", Description: "search web", Schema: json.RawMessage(`{"type":"object"}`)},
				{Name: "calc", Description: "math", Schema: json.RawMessage(`{"type":"object"}`)},
			},
			wantKey:   true,
			wantDecls: 2,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			req := inference.Request{
				Model:    model.Model{Name: "m"},
				Messages: content.AgenticMessages{userMsg(textBlock("hi"))},
				Tools:    tc.tools,
			}
			got, err := geminiapi.EncodeRequest(req)
			if err != nil {
				t.Fatalf("EncodeRequest error: %v", err)
			}
			raw := mustDecode(t, got)
			toolsRaw, ok := raw["tools"]
			if ok != tc.wantKey {
				t.Fatalf("tools present=%v, want %v", ok, tc.wantKey)
			}
			if !tc.wantKey {
				return
			}
			var tools []map[string]json.RawMessage
			if err := json.Unmarshal(toolsRaw, &tools); err != nil {
				t.Fatalf("unmarshal tools: %v", err)
			}
			if len(tools) != 1 {
				t.Fatalf("expected exactly 1 tool group, got %d", len(tools))
			}
			var decls []map[string]json.RawMessage
			if err := json.Unmarshal(tools[0]["functionDeclarations"], &decls); err != nil {
				t.Fatalf("unmarshal functionDeclarations: %v", err)
			}
			if len(decls) != tc.wantDecls {
				t.Fatalf("functionDeclarations count = %d, want %d", len(decls), tc.wantDecls)
			}
			if name := strField(t, decls[0], "name"); name != "search" {
				t.Errorf("decls[0].name = %q, want search", name)
			}
			if _, ok := decls[0]["parameters"]; !ok {
				t.Error("expected parameters on declaration")
			}
		})
	}
}

// --- TestEncodeRequest_GenerationConfig ---

func TestEncodeRequest_GenerationConfig(t *testing.T) {
	t.Parallel()

	temp := 0.3
	topP := 0.9
	maxTok := 256
	overrideTemp := 0.99

	cases := []struct {
		name          string
		model         model.Model
		override      *model.Sampling
		wantCfgKey    bool
		wantTemp      *float64
		wantTopP      *float64
		wantMaxTokens *int
		wantStop      []string
	}{
		{
			name:       "empty sampling: generationConfig omitted",
			model:      model.Model{Name: "m"},
			wantCfgKey: false,
		},
		{
			name:          "temperature/topP/maxOutputTokens/stopSequences on wire",
			model:         model.Model{Name: "m", Sampling: model.Sampling{Temperature: &temp, TopP: &topP, MaxTokens: &maxTok, Stop: []string{"STOP", "END"}}},
			wantCfgKey:    true,
			wantTemp:      &temp,
			wantTopP:      &topP,
			wantMaxTokens: &maxTok,
			wantStop:      []string{"STOP", "END"},
		},
		{
			name:       "override wins over model sampling",
			model:      model.Model{Name: "m", Sampling: model.Sampling{Temperature: &temp}},
			override:   &model.Sampling{Temperature: &overrideTemp},
			wantCfgKey: true,
			wantTemp:   &overrideTemp,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			req := inference.Request{Model: tc.model, Override: tc.override, Messages: content.AgenticMessages{userMsg(textBlock("hi"))}}
			got, err := geminiapi.EncodeRequest(req)
			if err != nil {
				t.Fatalf("EncodeRequest error: %v", err)
			}
			raw := mustDecode(t, got)
			cfgRaw, ok := raw["generationConfig"]
			if ok != tc.wantCfgKey {
				t.Fatalf("generationConfig present=%v, want %v", ok, tc.wantCfgKey)
			}
			if !tc.wantCfgKey {
				return
			}
			var cfg map[string]json.RawMessage
			if err := json.Unmarshal(cfgRaw, &cfg); err != nil {
				t.Fatalf("unmarshal generationConfig: %v", err)
			}
			if tc.wantTemp != nil {
				var v float64
				if err := json.Unmarshal(cfg["temperature"], &v); err != nil {
					t.Fatalf("unmarshal temperature: %v", err)
				}
				if v != *tc.wantTemp {
					t.Errorf("temperature = %v, want %v", v, *tc.wantTemp)
				}
			}
			if tc.wantTopP != nil {
				var v float64
				if err := json.Unmarshal(cfg["topP"], &v); err != nil {
					t.Fatalf("unmarshal topP: %v", err)
				}
				if v != *tc.wantTopP {
					t.Errorf("topP = %v, want %v", v, *tc.wantTopP)
				}
			}
			if tc.wantMaxTokens != nil {
				var v int
				if err := json.Unmarshal(cfg["maxOutputTokens"], &v); err != nil {
					t.Fatalf("unmarshal maxOutputTokens: %v", err)
				}
				if v != *tc.wantMaxTokens {
					t.Errorf("maxOutputTokens = %d, want %d", v, *tc.wantMaxTokens)
				}
			}
			if tc.wantStop != nil {
				var v []string
				if err := json.Unmarshal(cfg["stopSequences"], &v); err != nil {
					t.Fatalf("unmarshal stopSequences: %v", err)
				}
				if len(v) != len(tc.wantStop) {
					t.Fatalf("stopSequences len = %d, want %d", len(v), len(tc.wantStop))
				}
				for i := range tc.wantStop {
					if v[i] != tc.wantStop[i] {
						t.Errorf("stopSequences[%d] = %q, want %q", i, v[i], tc.wantStop[i])
					}
				}
			}
		})
	}
}

// --- TestEncodeRequest_ThinkingConfig ---

// TestEncodeRequest_ThinkingConfig locks the effort/Caps gating: thinkingConfig
// appears only when the model advertises Caps.Thinking AND Effort is a real level;
// EffortMax maps to thinkingBudget -1 (dynamic).
func TestEncodeRequest_ThinkingConfig(t *testing.T) {
	t.Parallel()

	thinkingCaps := model.Capabilities{Thinking: true}

	cases := []struct {
		name       string
		caps       model.Capabilities
		effort     model.Effort
		wantKey    bool
		wantBudget int
	}{
		{name: "no thinking capability: config omitted even with effort", caps: model.Capabilities{}, effort: model.EffortHigh, wantKey: false},
		{name: "thinking-capable, effort none: config omitted", caps: thinkingCaps, effort: model.EffortNone, wantKey: false},
		{name: "thinking-capable, effort low", caps: thinkingCaps, effort: model.EffortLow, wantKey: true, wantBudget: 4096},
		{name: "thinking-capable, effort medium", caps: thinkingCaps, effort: model.EffortMedium, wantKey: true, wantBudget: 8192},
		{name: "thinking-capable, effort high", caps: thinkingCaps, effort: model.EffortHigh, wantKey: true, wantBudget: 16384},
		{name: "thinking-capable, effort max -> dynamic -1", caps: thinkingCaps, effort: model.EffortMax, wantKey: true, wantBudget: -1},
		{name: "thinking-capable, unknown effort: config omitted", caps: thinkingCaps, effort: model.Effort("garbage"), wantKey: false},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			req := inference.Request{
				Model:    model.Model{Name: "m", Caps: tc.caps, Sampling: model.Sampling{Effort: tc.effort}},
				Messages: content.AgenticMessages{userMsg(textBlock("hi"))},
			}
			got, err := geminiapi.EncodeRequest(req)
			if err != nil {
				t.Fatalf("EncodeRequest error: %v", err)
			}
			raw := mustDecode(t, got)

			cfgRaw, hasCfg := raw["generationConfig"]
			if !tc.wantKey {
				if hasCfg {
					var cfg map[string]json.RawMessage
					if err := json.Unmarshal(cfgRaw, &cfg); err != nil {
						t.Fatalf("unmarshal generationConfig: %v", err)
					}
					if _, ok := cfg["thinkingConfig"]; ok {
						t.Error("thinkingConfig present, want absent")
					}
				}
				return
			}
			if !hasCfg {
				t.Fatal("generationConfig absent, want present")
			}
			var cfg map[string]json.RawMessage
			if err := json.Unmarshal(cfgRaw, &cfg); err != nil {
				t.Fatalf("unmarshal generationConfig: %v", err)
			}
			var tcfg map[string]json.RawMessage
			if err := json.Unmarshal(cfg["thinkingConfig"], &tcfg); err != nil {
				t.Fatalf("unmarshal thinkingConfig: %v", err)
			}
			var budget int
			if err := json.Unmarshal(tcfg["thinkingBudget"], &budget); err != nil {
				t.Fatalf("unmarshal thinkingBudget: %v", err)
			}
			if budget != tc.wantBudget {
				t.Errorf("thinkingBudget = %d, want %d", budget, tc.wantBudget)
			}
		})
	}
}

// --- TestEncodeRequest_UnsupportedBlock ---

// A user or model block the Gemini dialect does not model (audio, document) must
// fail secure with a *geminiapi.UnsupportedBlockError rather than being silently
// dropped — the model must never receive less than the caller sent. This mirrors
// the sibling anthropicapi codec. A ThinkingBlock on an assistant turn remains an
// intentional (documented) skip, not an error, UNLESS it carries a thoughtSignature
// (ProviderState) — covered by TestEncodeRequest_ThinkingWithoutSignatureIsDropped
// and TestEncodeRequest_ThinkingWithSignatureRoundTrips.
func TestEncodeRequest_UnsupportedBlock(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		msgs content.AgenticMessages
	}{
		{
			name: "audio block in a user turn is unsupported",
			msgs: content.AgenticMessages{userMsg(&content.AudioBlock{MediaType: content.MediaTypeAudioMPEG, Data: []byte{1}})},
		},
		{
			name: "document block in a user turn is unsupported",
			msgs: content.AgenticMessages{userMsg(&content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Data: []byte{1}})},
		},
		{
			name: "audio block in an assistant turn is unsupported",
			msgs: content.AgenticMessages{aiMsg(&content.AudioBlock{MediaType: content.MediaTypeAudioMPEG, Data: []byte{1}})},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := geminiapi.EncodeRequest(inference.Request{Model: model.Model{Name: "m"}, Messages: tc.msgs})
			if err == nil {
				t.Fatalf("expected error, got nil")
			}
			var ube *geminiapi.UnsupportedBlockError
			if !errors.As(err, &ube) {
				t.Errorf("error = %v (%T), want *geminiapi.UnsupportedBlockError", err, err)
			}
		})
	}
}

// --- TestEncodeRequest_ThinkingWithoutSignatureIsDropped ---

// A ThinkingBlock with no ProviderState (no thoughtSignature to replay) is
// still dropped: real Gemini 2.5+ models require a replayed thought part to
// carry the model's own exact signature, and there is nothing faithful to
// send without one. See TestEncodeRequest_ThinkingWithSignatureRoundTrips for
// the case that DOES now survive encoding.
func TestEncodeRequest_ThinkingWithoutSignatureIsDropped(t *testing.T) {
	t.Parallel()

	req := inference.Request{
		Model:    model.Model{Name: "m"},
		Messages: content.AgenticMessages{aiMsg(thinkingBlock("secret"), textBlock("visible"))},
	}
	got, err := geminiapi.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest error: %v", err)
	}
	contents := contentsFromRaw(t, mustDecode(t, got))
	parts := partsOf(t, contents[0])
	if len(parts) != 1 {
		t.Fatalf("expected only the text part to survive, got %d parts", len(parts))
	}
	if txt := strField(t, parts[0], "text"); txt != "visible" {
		t.Errorf("surviving text = %q, want visible", txt)
	}
}

// --- TestEncodeRequest_ThinkingWithSignatureRoundTrips ---

// A ThinkingBlock built via content.NewThinkingBlock with a non-nil
// ProviderState (the JSON-marshaled form of a Gemini thoughtSignature, per
// this codec's convention) is now replayed as a thought part carrying that
// exact signature — the encode.go fix this task makes, replacing the prior
// silent, undocumented drop.
func TestEncodeRequest_ThinkingWithSignatureRoundTrips(t *testing.T) {
	t.Parallel()

	providerState := json.RawMessage(`"opaque-thought-sig"`)
	req := inference.Request{
		Model: model.Model{Name: "m"},
		Messages: content.AgenticMessages{
			&content.AIMessage{Message: content.Message{
				Role: content.RoleAssistant,
				Blocks: []content.Block{
					content.NewThinkingBlock("secret", "", providerState, "gemini"),
					textBlock("visible"),
				},
			}},
		},
	}
	got, err := geminiapi.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest error: %v", err)
	}
	contents := contentsFromRaw(t, mustDecode(t, got))
	parts := partsOf(t, contents[0])
	if len(parts) != 2 {
		t.Fatalf("expected thought and text parts to survive, got %d parts", len(parts))
	}
	if txt := strField(t, parts[0], "text"); txt != "secret" {
		t.Errorf("thought text = %q, want secret", txt)
	}
	thoughtVal, ok := parts[0]["thought"]
	if !ok {
		t.Fatalf("parts[0] missing thought field: %#v", parts[0])
	}
	var thought bool
	if err := json.Unmarshal(thoughtVal, &thought); err != nil || !thought {
		t.Errorf("thought = %v, want true", thoughtVal)
	}
	if sig := strField(t, parts[0], "thoughtSignature"); sig != "opaque-thought-sig" {
		t.Errorf("thoughtSignature = %q, want opaque-thought-sig", sig)
	}
	if txt := strField(t, parts[1], "text"); txt != "visible" {
		t.Errorf("surviving text = %q, want visible", txt)
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
			name: "minimal",
			req:  inference.Request{Model: model.Model{Name: "m"}, Messages: content.AgenticMessages{userMsg(textBlock("hi"))}},
		},
		{
			name: "full with tools, system, thinking",
			req: inference.Request{
				Model: model.Model{
					Name:     "gemini-2.5-flash",
					Caps:     model.Capabilities{Thinking: true, AcceptsImages: true},
					Sampling: model.Sampling{Temperature: &temp, MaxTokens: &maxTok, Stop: []string{"STOP"}, Effort: model.EffortHigh},
				},
				System: "Be helpful.",
				Messages: content.AgenticMessages{
					userMsg(textBlock("hello"), imageDataBlock(content.MediaTypeImagePNG, []byte{1, 2, 3})),
					aiMsg(toolUseBlock("id1", "t", json.RawMessage(`{"a":1}`))),
					toolMsg("id1", textBlock("done")),
				},
				Tools: []inference.Tool{{Name: "t", Description: "d", Schema: json.RawMessage(`{"type":"object"}`)}},
			},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := geminiapi.EncodeRequest(tc.req)
			if err != nil {
				t.Fatalf("EncodeRequest error: %v", err)
			}
			if !json.Valid(got) {
				t.Errorf("output is not valid JSON: %s", got)
			}
		})
	}
}

func TestEncodeRequest_StructuredOutput(t *testing.T) {
	t.Parallel()

	schema := json.RawMessage(`{"type":"object","description":"root","properties":{"answer":{"type":"string","enum":["yes","no"]},"nested":{"type":"object","properties":{"values":{"type":"array","items":{"type":"integer"}}},"required":["values"],"additionalProperties":false}},"required":["answer","nested"],"additionalProperties":false}`)
	wantOriginal := append(json.RawMessage(nil), schema...)
	output := &inference.OutputSchema{Name: "answer", Schema: schema, Strict: true}
	req := inference.Request{
		Model:    model.Model{Name: "gemini-structured", Caps: model.Capabilities{StructuredOutput: true}},
		Messages: content.AgenticMessages{userMsg(textBlock("answer"))},
		Output:   output,
	}

	wire, err := geminiapi.BuildGenerateContentRequest(req)
	if err != nil {
		t.Fatalf("BuildGenerateContentRequest() error = %v", err)
	}
	if wire.GenerationConfig == nil {
		t.Fatal("GenerationConfig = nil, want structured output fields")
	}
	if got := wire.GenerationConfig.ResponseMIMEType; got != "application/json" {
		t.Errorf("ResponseMIMEType = %q, want application/json", got)
	}
	wantProjected := json.RawMessage(`{"type":"object","description":"root","properties":{"answer":{"type":"string","enum":["yes","no"]},"nested":{"type":"object","properties":{"values":{"type":"array","items":{"type":"integer"}}},"required":["values"]}},"required":["answer","nested"]}`)
	if string(wire.GenerationConfig.ResponseJSONSchema) != string(wantProjected) {
		t.Errorf("ResponseJSONSchema = %s, want exact projection %s", wire.GenerationConfig.ResponseJSONSchema, wantProjected)
	}
	if string(output.Schema) != string(wantOriginal) {
		t.Fatalf("BuildGenerateContentRequest() mutated caller schema to %s", output.Schema)
	}
	wire.GenerationConfig.ResponseJSONSchema[0] = '['
	if string(output.Schema) != string(wantOriginal) {
		t.Error("projected schema aliases caller schema")
	}

	encoded, err := geminiapi.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	raw := mustDecode(t, encoded)
	var config map[string]json.RawMessage
	if err := json.Unmarshal(raw["generationConfig"], &config); err != nil {
		t.Fatalf("unmarshal generationConfig: %v", err)
	}
	if got := strField(t, config, "responseMimeType"); got != "application/json" {
		t.Errorf("responseMimeType = %q, want application/json", got)
	}
	if string(config["responseJsonSchema"]) != string(wantProjected) {
		t.Errorf("responseJsonSchema = %s, want %s", config["responseJsonSchema"], wantProjected)
	}
}

func TestEncodeRequest_StructuredOutputWithToolsPreservesOrdinarySchema(t *testing.T) {
	t.Parallel()

	outputSchema := json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}`)
	ordinarySchema := json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}},"additionalProperties":false}`)
	req := inference.Request{
		Model: model.Model{Name: "gemini-combined", Caps: model.Capabilities{
			Tools:                     true,
			StructuredOutput:          true,
			StructuredOutputWithTools: true,
		}},
		Messages: content.AgenticMessages{userMsg(textBlock("answer"))},
		Tools:    []inference.Tool{{Name: "search", Schema: ordinarySchema}},
		Output:   &inference.OutputSchema{Name: "answer", Schema: outputSchema, Strict: true},
	}

	encoded, err := geminiapi.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	raw := mustDecode(t, encoded)
	var tools []struct {
		FunctionDeclarations []struct {
			Name       string          `json:"name"`
			Parameters json.RawMessage `json:"parameters"`
		} `json:"functionDeclarations"`
	}
	if err := json.Unmarshal(raw["tools"], &tools); err != nil {
		t.Fatalf("unmarshal tools: %v", err)
	}
	if len(tools) != 1 || len(tools[0].FunctionDeclarations) != 1 {
		t.Fatalf("tools = %+v, want one declaration", tools)
	}
	declaration := tools[0].FunctionDeclarations[0]
	if declaration.Name != "search" || string(declaration.Parameters) != string(ordinarySchema) {
		t.Errorf("ordinary tool = %q %s, want unchanged %q %s", declaration.Name, declaration.Parameters, "search", ordinarySchema)
	}
}

func TestEncodeRequest_TerminalToolProjectionAndRequiredChoice(t *testing.T) {
	t.Parallel()

	terminalSchema := json.RawMessage(`{"type":"object","properties":{"result":{"type":"object","properties":{"items":{"type":"array","items":{"type":"string"}}},"required":["items"],"additionalProperties":false}},"required":["result"],"additionalProperties":false}`)
	ordinarySchema := json.RawMessage(`{"type":"object","additionalProperties":false,"properties":{"q":{"type":"string"}}}`)
	req := inference.Request{
		Model: model.Model{Name: "gemini-tools", Caps: model.Capabilities{Tools: true}},
		Tools: []inference.Tool{
			{Name: "search", Schema: ordinarySchema},
			{Name: inference.StructuredOutputToolName, Description: "final", Schema: terminalSchema},
		},
		ToolChoice: inference.ToolChoiceRequired,
	}

	encoded, err := geminiapi.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	raw := mustDecode(t, encoded)
	var tools []struct {
		FunctionDeclarations []struct {
			Name       string          `json:"name"`
			Parameters json.RawMessage `json:"parameters"`
		} `json:"functionDeclarations"`
	}
	if err := json.Unmarshal(raw["tools"], &tools); err != nil {
		t.Fatalf("unmarshal tools: %v", err)
	}
	if len(tools) != 1 || len(tools[0].FunctionDeclarations) != 2 {
		t.Fatalf("tools = %+v, want two declarations", tools)
	}
	if got := tools[0].FunctionDeclarations[0].Parameters; string(got) != string(ordinarySchema) {
		t.Errorf("ordinary schema = %s, want unchanged %s", got, ordinarySchema)
	}
	wantTerminal := `{"type":"object","properties":{"result":{"type":"object","properties":{"items":{"type":"array","items":{"type":"string"}}},"required":["items"]}},"required":["result"]}`
	if got := tools[0].FunctionDeclarations[1].Parameters; string(got) != wantTerminal {
		t.Errorf("terminal schema = %s, want %s", got, wantTerminal)
	}
	var toolConfig struct {
		FunctionCallingConfig struct {
			Mode string `json:"mode"`
		} `json:"functionCallingConfig"`
	}
	if err := json.Unmarshal(raw["toolConfig"], &toolConfig); err != nil {
		t.Fatalf("unmarshal toolConfig: %v", err)
	}
	if toolConfig.FunctionCallingConfig.Mode != "ANY" {
		t.Errorf("functionCallingConfig.mode = %q, want ANY", toolConfig.FunctionCallingConfig.Mode)
	}
	if string(req.Tools[1].Schema) != string(terminalSchema) {
		t.Error("terminal projection mutated caller schema")
	}
}

func TestBuildGenerateContentRequest_StructuredFeatureValidation(t *testing.T) {
	t.Parallel()

	type errorKind uint8
	const (
		errorKindUnsupported errorKind = iota + 1
		errorKindCombinedUnsupported
		errorKindSchema
	)

	validSchema := json.RawMessage(`{"type":"object","properties":{},"additionalProperties":false}`)
	tests := []struct {
		name    string
		req     inference.Request
		wantErr errorKind
	}{
		{
			name:    "native capability missing",
			req:     inference.Request{Model: model.Model{Name: "plain"}, Output: &inference.OutputSchema{Name: "out", Schema: validSchema}},
			wantErr: errorKindUnsupported,
		},
		{
			name: "combined capability missing",
			req: inference.Request{
				Model:  model.Model{Name: "output-only", Caps: model.Capabilities{StructuredOutput: true, Tools: true}},
				Output: &inference.OutputSchema{Name: "out", Schema: validSchema},
				Tools:  []inference.Tool{{Name: "search", Schema: json.RawMessage(`{"type":"object"}`)}},
			},
			wantErr: errorKindCombinedUnsupported,
		},
		{
			name: "malformed schema",
			req: inference.Request{
				Model:  model.Model{Name: "structured", Caps: model.Capabilities{StructuredOutput: true}},
				Output: &inference.OutputSchema{Name: "out", Schema: json.RawMessage(`{"type":"object"}`)},
			},
			wantErr: errorKindSchema,
		},
		{
			name: "invalid terminal schema",
			req: inference.Request{
				Model: model.Model{Name: "tools", Caps: model.Capabilities{Tools: true}},
				Tools: []inference.Tool{{Name: inference.StructuredOutputToolName, Schema: json.RawMessage(`{"type":"object"}`)}},
			},
			wantErr: errorKindSchema,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			_, err := geminiapi.BuildGenerateContentRequest(tt.req)
			if err == nil {
				t.Fatal("BuildGenerateContentRequest() error = nil")
			}
			switch tt.wantErr {
			case errorKindUnsupported:
				var target *inference.StructuredOutputUnsupportedError
				if !errors.As(err, &target) {
					t.Fatalf("error = %T, want StructuredOutputUnsupportedError", err)
				}
			case errorKindCombinedUnsupported:
				var target *inference.StructuredOutputWithToolsUnsupportedError
				if !errors.As(err, &target) {
					t.Fatalf("error = %T, want StructuredOutputWithToolsUnsupportedError", err)
				}
			case errorKindSchema:
				var target *inference.SchemaValidationError
				if !errors.As(err, &target) {
					t.Fatalf("error = %T, want SchemaValidationError", err)
				}
			}
		})
	}
}

func TestEncodeRequest_NilOutputPreservesWireShape(t *testing.T) {
	t.Parallel()

	encoded, err := geminiapi.EncodeRequest(inference.Request{Model: model.Model{Name: "m"}, Messages: content.AgenticMessages{userMsg(textBlock("hi"))}})
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	raw := mustDecode(t, encoded)
	if _, ok := raw["generationConfig"]; ok {
		t.Error("generationConfig present for nil output and zero sampling")
	}
	if _, ok := raw["toolConfig"]; ok {
		t.Error("toolConfig present for automatic choice")
	}
}
