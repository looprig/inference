package anthropicapi_test

import (
	"bytes"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/anthropicapi"
	model "github.com/looprig/inference/model"
)

// newDecodeRequest builds an httptest request for POST /v1/messages carrying body.
func newDecodeRequest(body string) *http.Request {
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")
	return req
}

func mustDecode(t *testing.T, body string) inference.Request {
	t.Helper()
	decoded, err := anthropicapi.Codec{}.DecodeRequest(newDecodeRequest(body))
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v (body=%s)", err, body)
	}
	return decoded.Request
}

func TestDecodeRequest_MatchRequest(t *testing.T) {
	t.Parallel()
	c := anthropicapi.Codec{}
	if !c.MatchRequest(newDecodeRequest(`{}`)) {
		t.Error("MatchRequest(POST /v1/messages) = false, want true")
	}
	if c.MatchRequest(httptest.NewRequest(http.MethodGet, "/v1/messages", nil)) {
		t.Error("MatchRequest(GET /v1/messages) = true, want false")
	}
	if c.MatchRequest(httptest.NewRequest(http.MethodPost, "/v1/other", nil)) {
		t.Error("MatchRequest(POST /v1/other) = true, want false")
	}
}

func TestDecodeRequest_SystemAndUserText(t *testing.T) {
	t.Parallel()
	body := `{
		"model": "claude-test",
		"system": "be terse",
		"max_tokens": 256,
		"messages": [
			{"role": "user", "content": [{"type": "text", "text": "hello"}]}
		]
	}`
	req := mustDecode(t, body)

	if req.System != "be terse" {
		t.Errorf("System = %q, want %q", req.System, "be terse")
	}
	if len(req.Messages) != 1 {
		t.Fatalf("Messages len = %d, want 1", len(req.Messages))
	}
	um, ok := req.Messages[0].(*content.UserMessage)
	if !ok {
		t.Fatalf("Messages[0] type = %T, want *content.UserMessage", req.Messages[0])
	}
	if len(um.Blocks) != 1 {
		t.Fatalf("UserMessage.Blocks len = %d, want 1", len(um.Blocks))
	}
	tb, ok := um.Blocks[0].(*content.TextBlock)
	if !ok || tb.Text != "hello" {
		t.Errorf("Blocks[0] = %#v, want TextBlock{hello}", um.Blocks[0])
	}
}

// TestDecodeRequest_ModelStaysUnresolved pins that Request.Model is left at its
// zero value: the harness model alias travels only in RequestedModel, and
// resolving it to a real Target is the gateway's job, not this codec's.
func TestDecodeRequest_ModelStaysUnresolved(t *testing.T) {
	t.Parallel()
	body := `{"model":"claude-alias","max_tokens":16,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`
	decoded, err := anthropicapi.Codec{}.DecodeRequest(newDecodeRequest(body))
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if decoded.RequestedModel != "claude-alias" {
		t.Errorf("RequestedModel = %q, want %q", decoded.RequestedModel, "claude-alias")
	}
	if !reflect.DeepEqual(decoded.Request.Model, model.Model{}) {
		t.Errorf("Request.Model = %+v, want zero value", decoded.Request.Model)
	}
}

func TestDecodeRequest_StreamFlag(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name string
		body string
		want bool
	}{
		{name: "absent", body: `{"model":"m","max_tokens":1,"messages":[]}`, want: false},
		{name: "false", body: `{"model":"m","max_tokens":1,"stream":false,"messages":[]}`, want: false},
		{name: "true", body: `{"model":"m","max_tokens":1,"stream":true,"messages":[]}`, want: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			decoded, err := anthropicapi.Codec{}.DecodeRequest(newDecodeRequest(tc.body))
			if err != nil {
				t.Fatalf("DecodeRequest() error = %v", err)
			}
			if decoded.Streaming != tc.want {
				t.Errorf("Streaming = %v, want %v", decoded.Streaming, tc.want)
			}
		})
	}
}

func TestDecodeRequest_AssistantParallelToolUse(t *testing.T) {
	t.Parallel()
	body := `{
		"model": "m", "max_tokens": 16,
		"messages": [
			{"role": "assistant", "content": [
				{"type": "text", "text": "calling tools"},
				{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city":"nyc"}},
				{"type": "tool_use", "id": "toolu_2", "name": "get_time", "input": {"tz":"utc"}}
			]}
		]
	}`
	req := mustDecode(t, body)
	if len(req.Messages) != 1 {
		t.Fatalf("Messages len = %d, want 1", len(req.Messages))
	}
	ai, ok := req.Messages[0].(*content.AIMessage)
	if !ok {
		t.Fatalf("Messages[0] type = %T, want *content.AIMessage", req.Messages[0])
	}
	if len(ai.Blocks) != 3 {
		t.Fatalf("AIMessage.Blocks len = %d, want 3", len(ai.Blocks))
	}
	tu1, ok := ai.Blocks[1].(*content.ToolUseBlock)
	if !ok || tu1.ID != "toolu_1" || tu1.Name != "get_weather" {
		t.Errorf("Blocks[1] = %#v, want ToolUseBlock{toolu_1,get_weather}", ai.Blocks[1])
	}
	tu2, ok := ai.Blocks[2].(*content.ToolUseBlock)
	if !ok || tu2.ID != "toolu_2" || tu2.Name != "get_time" {
		t.Errorf("Blocks[2] = %#v, want ToolUseBlock{toolu_2,get_time}", ai.Blocks[2])
	}
}

func TestDecodeRequest_ParallelToolResults(t *testing.T) {
	t.Parallel()
	body := `{
		"model": "m", "max_tokens": 16,
		"messages": [
			{"role": "user", "content": [
				{"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type":"text","text":"sunny"}]},
				{"type": "tool_result", "tool_use_id": "toolu_2", "content": [{"type":"text","text":"12:00"}], "is_error": true}
			]}
		]
	}`
	req := mustDecode(t, body)
	if len(req.Messages) != 2 {
		t.Fatalf("Messages len = %d, want 2 (one ToolResultMessage per result)", len(req.Messages))
	}
	tr1, ok := req.Messages[0].(*content.ToolResultMessage)
	if !ok || tr1.ToolUseID != "toolu_1" || tr1.IsError {
		t.Errorf("Messages[0] = %#v, want ToolResultMessage{toolu_1,false}", req.Messages[0])
	}
	tr2, ok := req.Messages[1].(*content.ToolResultMessage)
	if !ok || tr2.ToolUseID != "toolu_2" || !tr2.IsError {
		t.Errorf("Messages[1] = %#v, want ToolResultMessage{toolu_2,true}", req.Messages[1])
	}
}

func TestDecodeRequest_MixedUserTextAndToolResult(t *testing.T) {
	t.Parallel()
	body := `{
		"model": "m", "max_tokens": 16,
		"messages": [
			{"role": "user", "content": [
				{"type": "text", "text": "before"},
				{"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type":"text","text":"result"}]}
			]}
		]
	}`
	req := mustDecode(t, body)
	if len(req.Messages) != 2 {
		t.Fatalf("Messages len = %d, want 2", len(req.Messages))
	}
	if _, ok := req.Messages[0].(*content.UserMessage); !ok {
		t.Errorf("Messages[0] type = %T, want *content.UserMessage", req.Messages[0])
	}
	if _, ok := req.Messages[1].(*content.ToolResultMessage); !ok {
		t.Errorf("Messages[1] type = %T, want *content.ToolResultMessage", req.Messages[1])
	}
}

func TestDecodeRequest_Images(t *testing.T) {
	t.Parallel()

	t.Run("base64", func(t *testing.T) {
		t.Parallel()
		body := `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":[
			{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aGVsbG8="}}
		]}]}`
		req := mustDecode(t, body)
		um := req.Messages[0].(*content.UserMessage)
		img, ok := um.Blocks[0].(*content.ImageBlock)
		if !ok {
			t.Fatalf("Blocks[0] type = %T, want *content.ImageBlock", um.Blocks[0])
		}
		if string(img.MediaType) != "image/png" {
			t.Errorf("MediaType = %q, want image/png", img.MediaType)
		}
		if string(img.Source.Data) != "hello" {
			t.Errorf("Source.Data = %q, want %q", img.Source.Data, "hello")
		}
	})

	t.Run("url", func(t *testing.T) {
		t.Parallel()
		body := `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":[
			{"type":"image","source":{"type":"url","url":"https://example.test/x.png"}}
		]}]}`
		req := mustDecode(t, body)
		um := req.Messages[0].(*content.UserMessage)
		img := um.Blocks[0].(*content.ImageBlock)
		if img.Source.URL != "https://example.test/x.png" {
			t.Errorf("Source.URL = %q", img.Source.URL)
		}
	})

	t.Run("not allowed in assistant message", func(t *testing.T) {
		t.Parallel()
		body := `{"model":"m","max_tokens":8,"messages":[{"role":"assistant","content":[
			{"type":"image","source":{"type":"url","url":"https://example.test/x.png"}}
		]}]}`
		_, err := anthropicapi.Codec{}.DecodeRequest(newDecodeRequest(body))
		if err == nil {
			t.Error("DecodeRequest() error = nil, want error (image in assistant message)")
		}
	})
}

func TestDecodeRequest_Tools(t *testing.T) {
	t.Parallel()
	body := `{
		"model": "m", "max_tokens": 8,
		"tools": [{"name":"get_weather","description":"gets weather","input_schema":{"type":"object"}}],
		"messages": []
	}`
	req := mustDecode(t, body)
	if len(req.Tools) != 1 {
		t.Fatalf("Tools len = %d, want 1", len(req.Tools))
	}
	if req.Tools[0].Name != "get_weather" || req.Tools[0].Description != "gets weather" {
		t.Errorf("Tools[0] = %#v", req.Tools[0])
	}
	if string(req.Tools[0].Schema) != `{"type":"object"}` {
		t.Errorf("Tools[0].Schema = %s", req.Tools[0].Schema)
	}
}

func TestDecodeRequest_ToolChoice(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name      string
		toolChoiceJSON string
		want      inference.ToolChoice
		wantError bool
	}{
		{name: "absent", toolChoiceJSON: ``, want: inference.ToolChoiceAuto},
		{name: "auto", toolChoiceJSON: `,"tool_choice":{"type":"auto"}`, want: inference.ToolChoiceAuto},
		{name: "any", toolChoiceJSON: `,"tool_choice":{"type":"any"}`, want: inference.ToolChoiceRequired},
		{name: "tool (unsupported)", toolChoiceJSON: `,"tool_choice":{"type":"tool","name":"x"}`, wantError: true},
		{name: "none (unsupported)", toolChoiceJSON: `,"tool_choice":{"type":"none"}`, wantError: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			body := `{"model":"m","max_tokens":8,"messages":[]` + tc.toolChoiceJSON + `}`
			decoded, err := anthropicapi.Codec{}.DecodeRequest(newDecodeRequest(body))
			if tc.wantError {
				if err == nil {
					t.Fatal("DecodeRequest() error = nil, want error")
				}
				return
			}
			if err != nil {
				t.Fatalf("DecodeRequest() error = %v", err)
			}
			if decoded.Request.ToolChoice != tc.want {
				t.Errorf("ToolChoice = %v, want %v", decoded.Request.ToolChoice, tc.want)
			}
		})
	}
}

func TestDecodeRequest_Sampling(t *testing.T) {
	t.Parallel()
	body := `{
		"model": "m", "max_tokens": 512,
		"temperature": 0.5, "top_p": 0.9, "stop_sequences": ["STOP","END"],
		"messages": []
	}`
	req := mustDecode(t, body)
	if req.Override == nil {
		t.Fatal("Override = nil, want non-nil")
	}
	s := req.Override
	if s.MaxTokens == nil || *s.MaxTokens != 512 {
		t.Errorf("MaxTokens = %v, want 512", s.MaxTokens)
	}
	if s.Temperature == nil || *s.Temperature != 0.5 {
		t.Errorf("Temperature = %v, want 0.5", s.Temperature)
	}
	if s.TopP == nil || *s.TopP != 0.9 {
		t.Errorf("TopP = %v, want 0.9", s.TopP)
	}
	if len(s.Stop) != 2 || s.Stop[0] != "STOP" || s.Stop[1] != "END" {
		t.Errorf("Stop = %v, want [STOP END]", s.Stop)
	}
}

func TestDecodeRequest_VisibleThinking(t *testing.T) {
	t.Parallel()
	cases := []struct {
		wire string
		want model.Effort
	}{
		{wire: "low", want: model.EffortLow},
		{wire: "medium", want: model.EffortMedium},
		{wire: "high", want: model.EffortHigh},
		{wire: "max", want: model.EffortMax},
	}
	for _, tc := range cases {
		t.Run(tc.wire, func(t *testing.T) {
			t.Parallel()
			body := `{
				"model": "m", "max_tokens": 512,
				"thinking": {"type":"adaptive"},
				"output_config": {"effort":"` + tc.wire + `"},
				"messages": []
			}`
			req := mustDecode(t, body)
			if req.Override == nil || req.Override.Effort != tc.want {
				t.Errorf("Effort = %v, want %v", req.Override, tc.want)
			}
		})
	}

	t.Run("thinking without effort is a decode error", func(t *testing.T) {
		t.Parallel()
		body := `{"model":"m","max_tokens":8,"thinking":{"type":"adaptive"},"messages":[]}`
		_, err := anthropicapi.Codec{}.DecodeRequest(newDecodeRequest(body))
		if err == nil {
			t.Error("DecodeRequest() error = nil, want error")
		}
	})

	t.Run("manual budget thinking is unsupported", func(t *testing.T) {
		t.Parallel()
		body := `{"model":"m","max_tokens":8,"thinking":{"type":"enabled"},"output_config":{"effort":"low"},"messages":[]}`
		_, err := anthropicapi.Codec{}.DecodeRequest(newDecodeRequest(body))
		if err == nil {
			t.Error("DecodeRequest() error = nil, want error")
		}
	})
}

// TestDecodeRequest_SignedThinkingReplay pins that a replayed assistant
// thinking block's signature is preserved byte-for-byte: Anthropic requires the
// exact signature on a thinking-plus-tool-use replay turn.
func TestDecodeRequest_SignedThinkingReplay(t *testing.T) {
	t.Parallel()
	const signature = "sig-abc123-should-round-trip-exactly=="
	body := `{
		"model": "m", "max_tokens": 8,
		"messages": [
			{"role": "assistant", "content": [
				{"type": "thinking", "thinking": "let me think", "signature": "` + signature + `"},
				{"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}}
			]}
		]
	}`
	req := mustDecode(t, body)
	ai := req.Messages[0].(*content.AIMessage)
	tb, ok := ai.Blocks[0].(*content.ThinkingBlock)
	if !ok {
		t.Fatalf("Blocks[0] type = %T, want *content.ThinkingBlock", ai.Blocks[0])
	}
	if tb.Signature != signature {
		t.Errorf("Signature = %q, want %q", tb.Signature, signature)
	}
	if tb.Thinking != "let me think" {
		t.Errorf("Thinking = %q", tb.Thinking)
	}
}

func TestDecodeRequest_CacheControlIgnored(t *testing.T) {
	t.Parallel()

	t.Run("block level", func(t *testing.T) {
		t.Parallel()
		body := `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":[
			{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}
		]}]}`
		req := mustDecode(t, body)
		um := req.Messages[0].(*content.UserMessage)
		tb := um.Blocks[0].(*content.TextBlock)
		if tb.Text != "hi" {
			t.Errorf("Text = %q, want hi", tb.Text)
		}
	})

	t.Run("system array form", func(t *testing.T) {
		t.Parallel()
		body := `{"model":"m","max_tokens":8,
			"system":[{"type":"text","text":"sys prompt","cache_control":{"type":"ephemeral"}}],
			"messages":[]}`
		req := mustDecode(t, body)
		if req.System != "sys prompt" {
			t.Errorf("System = %q, want %q", req.System, "sys prompt")
		}
	})
}

func TestDecodeRequest_MetadataIgnored(t *testing.T) {
	t.Parallel()
	body := `{"model":"m","max_tokens":8,"metadata":{"user_id":"abc"},"messages":[]}`
	req := mustDecode(t, body)
	if req.System != "" {
		t.Errorf("unexpected System = %q", req.System)
	}
}

func TestDecodeRequest_UnknownMaterialFieldFails(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		body string
	}{
		{name: "top-level unknown field", body: `{"model":"m","max_tokens":8,"messages":[],"top_k":5}`},
		{name: "unknown block field (citations)", body: `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":[{"type":"text","text":"hi","citations":[]}]}]}`},
		{name: "unrecognized block type (redacted_thinking)", body: `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":[{"type":"redacted_thinking","data":"x"}]}]}`},
		{name: "unrecognized block type (document)", body: `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":[{"type":"document"}]}]}`},
		{name: "thinking config unknown field (budget_tokens)", body: `{"model":"m","max_tokens":8,"thinking":{"type":"adaptive","budget_tokens":1024},"output_config":{"effort":"low"},"messages":[]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := anthropicapi.Codec{}.DecodeRequest(newDecodeRequest(tc.body))
			if err == nil {
				t.Error("DecodeRequest() error = nil, want error")
			}
		})
	}
}

func TestDecodeRequest_DuplicateJSONKeyFails(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		body string
	}{
		{name: "top level", body: `{"model":"m","max_tokens":8,"max_tokens":16,"messages":[]}`},
		{name: "nested block", body: `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":[{"type":"text","text":"a","text":"b"}]}]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := anthropicapi.Codec{}.DecodeRequest(newDecodeRequest(tc.body))
			if err == nil {
				t.Fatal("DecodeRequest() error = nil, want error")
			}
			var dupErr *anthropicapi.DuplicateKeyError
			if !errors.As(err, &dupErr) {
				t.Errorf("error = %T (%v), want *anthropicapi.DuplicateKeyError", err, err)
			}
		})
	}
}

func TestDecodeRequest_MissingModelFails(t *testing.T) {
	t.Parallel()
	_, err := anthropicapi.Codec{}.DecodeRequest(newDecodeRequest(`{"max_tokens":8,"messages":[]}`))
	if err == nil {
		t.Error("DecodeRequest() error = nil, want error")
	}
}

func TestDecodeRequest_MalformedBodyNeverPanics(t *testing.T) {
	t.Parallel()
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("DecodeRequest panicked: %v", r)
		}
	}()
	_, err := anthropicapi.Codec{}.DecodeRequest(newDecodeRequest(`{"model":`))
	if err == nil {
		t.Error("DecodeRequest() error = nil, want error")
	}
}

func TestDecodeRequest_WrongContentTypeRejected(t *testing.T) {
	t.Parallel()
	req := newDecodeRequest(`{"model":"m","max_tokens":8,"messages":[]}`)
	req.Header.Set("Content-Type", "text/plain")
	_, err := anthropicapi.Codec{}.DecodeRequest(req)
	if err == nil {
		t.Error("DecodeRequest() error = nil, want error")
	}
}

// --- count_tokens -----------------------------------------------------------

func TestCountTokens_MatchAndDecode(t *testing.T) {
	t.Parallel()
	body := `{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")

	if !anthropicapi.MatchCountTokensRequest(req) {
		t.Fatal("MatchCountTokensRequest() = false, want true")
	}
	if anthropicapi.MatchCountTokensRequest(httptest.NewRequest(http.MethodPost, "/v1/messages", nil)) {
		t.Error("MatchCountTokensRequest(/v1/messages) = true, want false")
	}

	decoded, err := anthropicapi.DecodeCountTokensRequest(req)
	if err != nil {
		t.Fatalf("DecodeCountTokensRequest() error = %v", err)
	}
	if decoded.RequestedModel != "m" {
		t.Errorf("RequestedModel = %q, want m", decoded.RequestedModel)
	}
	if decoded.Streaming {
		t.Error("Streaming = true, want false")
	}
	if len(decoded.Request.Messages) != 1 {
		t.Fatalf("Messages len = %d, want 1", len(decoded.Request.Messages))
	}
}

// TestCountTokens_OmittedMaxTokensAndStreamAreFine pins that count_tokens bodies
// (which never carry max_tokens or stream) decode successfully through the
// shared decode core.
func TestCountTokens_OmittedMaxTokensAndStreamAreFine(t *testing.T) {
	t.Parallel()
	body := `{"model":"m","system":"be terse","messages":[]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")

	decoded, err := anthropicapi.DecodeCountTokensRequest(req)
	if err != nil {
		t.Fatalf("DecodeCountTokensRequest() error = %v", err)
	}
	if decoded.Request.System != "be terse" {
		t.Errorf("System = %q", decoded.Request.System)
	}
	if decoded.Request.Override == nil || decoded.Request.Override.MaxTokens != nil {
		t.Errorf("Override.MaxTokens = %v, want nil (not sent)", decoded.Request.Override)
	}
}
