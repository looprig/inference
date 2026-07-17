package inference_test

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/stream"
)

func assistantMessage(blocks ...content.Block) *content.AIMessage {
	return &content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: blocks}}
}

func TestStructuredMessageResultAcceptsExactlyOneRepresentation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		msg    *content.AIMessage
		want   string
		mutate func()
	}{
		{
			name: "ordered text fragments with thinking",
			msg: assistantMessage(
				&content.ThinkingBlock{Thinking: "private"},
				&content.TextBlock{Text: "  {\"message\":\"hé"},
				&content.TextBlock{Text: "llo 世界\", \"ok\": true}  "},
			),
			want: `{"message":"héllo 世界","ok":true}`,
		},
	}

	toolInput := json.RawMessage(" \n { \"ok\" : true } \t")
	tests = append(tests, struct {
		name   string
		msg    *content.AIMessage
		want   string
		mutate func()
	}{
		name: "one terminal tool with thinking",
		msg: assistantMessage(
			&content.ThinkingBlock{Thinking: "private"},
			&content.ToolUseBlock{ID: "terminal", Name: inference.StructuredOutputToolName, Input: toolInput},
		),
		want: `{"ok":true}`,
		mutate: func() {
			for i := range toolInput {
				toolInput[i] = 'x'
			}
		},
	})

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := inference.StructuredMessageResult(tt.msg)
			if err != nil {
				t.Fatalf("StructuredMessageResult() error = %v", err)
			}
			if string(got) != tt.want {
				t.Fatalf("StructuredMessageResult() = %q, want %q", got, tt.want)
			}
			if tt.mutate != nil {
				tt.mutate()
				if string(got) != tt.want {
					t.Fatalf("result aliases message storage: got %q, want %q", got, tt.want)
				}
			}
		})
	}
}

func TestStructuredMessageResultRejectsInvalidRepresentations(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		msg        *content.AIMessage
		wantReason inference.MalformedStructuredOutputReason
		payload    string
	}{
		{name: "nil message", wantReason: inference.MalformedReasonNilMessage},
		{name: "wrong role", msg: &content.AIMessage{Message: content.Message{Role: content.RoleUser}}, wantReason: inference.MalformedReasonWrongRole},
		{name: "no blocks", msg: assistantMessage(), wantReason: inference.MalformedReasonEmpty},
		{name: "empty text", msg: assistantMessage(&content.TextBlock{}), wantReason: inference.MalformedReasonEmpty},
		{name: "whitespace text", msg: assistantMessage(&content.TextBlock{Text: " \n\t"}), wantReason: inference.MalformedReasonEmpty, payload: " \n\t"},
		{name: "malformed text", msg: assistantMessage(&content.TextBlock{Text: `{"secret":"do-not-leak"`}), wantReason: inference.MalformedReasonMalformedJSON, payload: `{"secret":"do-not-leak"`},
		{name: "invalid UTF-8", msg: assistantMessage(&content.TextBlock{Text: "{\"value\":\"\xff\"}"}), wantReason: inference.MalformedReasonMalformedJSON, payload: "{\"value\":\"\xff\"}"},
		{name: "trailing json", msg: assistantMessage(&content.TextBlock{Text: `{} {}`}), wantReason: inference.MalformedReasonMalformedJSON, payload: `{} {}`},
		{name: "scalar", msg: assistantMessage(&content.TextBlock{Text: `"value"`}), wantReason: inference.MalformedReasonRootNotObject, payload: `"value"`},
		{name: "array", msg: assistantMessage(&content.TextBlock{Text: `[]`}), wantReason: inference.MalformedReasonRootNotObject, payload: `[]`},
		{name: "null", msg: assistantMessage(&content.TextBlock{Text: `null`}), wantReason: inference.MalformedReasonRootNotObject, payload: `null`},
		{name: "ordinary tool", msg: assistantMessage(&content.ToolUseBlock{Name: "read", Input: json.RawMessage(`{}`)}), wantReason: inference.MalformedReasonInvalidRepresentation},
		{name: "mixed text and terminal", msg: assistantMessage(&content.TextBlock{Text: `{}`}, &content.ToolUseBlock{Name: inference.StructuredOutputToolName, Input: json.RawMessage(`{}`)}), wantReason: inference.MalformedReasonAmbiguous},
		{name: "terminal and ordinary", msg: assistantMessage(&content.ToolUseBlock{Name: inference.StructuredOutputToolName, Input: json.RawMessage(`{}`)}, &content.ToolUseBlock{Name: "read", Input: json.RawMessage(`{}`)}), wantReason: inference.MalformedReasonAmbiguous},
		{name: "duplicate terminal", msg: assistantMessage(&content.ToolUseBlock{Name: inference.StructuredOutputToolName, Input: json.RawMessage(`{}`)}, &content.ToolUseBlock{Name: inference.StructuredOutputToolName, Input: json.RawMessage(`{}`)}), wantReason: inference.MalformedReasonAmbiguous},
		{name: "empty terminal input", msg: assistantMessage(&content.ToolUseBlock{Name: inference.StructuredOutputToolName}), wantReason: inference.MalformedReasonEmpty},
		{name: "terminal scalar", msg: assistantMessage(&content.ToolUseBlock{Name: inference.StructuredOutputToolName, Input: json.RawMessage(`7`)}), wantReason: inference.MalformedReasonRootNotObject, payload: `7`},
		{name: "image block", msg: assistantMessage(&content.ImageBlock{}), wantReason: inference.MalformedReasonInvalidBlock},
		{name: "audio block", msg: assistantMessage(&content.AudioBlock{}), wantReason: inference.MalformedReasonInvalidBlock},
		{name: "document block", msg: assistantMessage(&content.DocumentBlock{}), wantReason: inference.MalformedReasonInvalidBlock},
		{name: "tool result block", msg: assistantMessage(&content.ToolResultBlock{}), wantReason: inference.MalformedReasonInvalidBlock},
		{name: "typed nil text", msg: assistantMessage((*content.TextBlock)(nil)), wantReason: inference.MalformedReasonNilBlock},
		{name: "typed nil thinking", msg: assistantMessage((*content.ThinkingBlock)(nil)), wantReason: inference.MalformedReasonNilBlock},
		{name: "typed nil terminal", msg: assistantMessage((*content.ToolUseBlock)(nil)), wantReason: inference.MalformedReasonNilBlock},
		{name: "typed nil image", msg: assistantMessage((*content.ImageBlock)(nil)), wantReason: inference.MalformedReasonNilBlock},
		{name: "typed nil audio", msg: assistantMessage((*content.AudioBlock)(nil)), wantReason: inference.MalformedReasonNilBlock},
		{name: "typed nil document", msg: assistantMessage((*content.DocumentBlock)(nil)), wantReason: inference.MalformedReasonNilBlock},
		{name: "typed nil tool result", msg: assistantMessage((*content.ToolResultBlock)(nil)), wantReason: inference.MalformedReasonNilBlock},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := inference.StructuredMessageResult(tt.msg)
			if err == nil {
				t.Fatal("StructuredMessageResult() error = nil")
			}
			var malformed *inference.MalformedStructuredOutputError
			if !errors.As(err, &malformed) {
				t.Fatalf("error = %T %v, want *MalformedStructuredOutputError", err, err)
			}
			if malformed.ReasonCode != tt.wantReason {
				t.Errorf("ReasonCode = %q, want %q", malformed.ReasonCode, tt.wantReason)
			}
			if malformed.Length != len(tt.payload) {
				t.Errorf("Length = %d, want %d", malformed.Length, len(tt.payload))
			}
			if malformed.SHA256 != sha256.Sum256([]byte(tt.payload)) {
				t.Errorf("SHA256 = %x, want %x", malformed.SHA256, sha256.Sum256([]byte(tt.payload)))
			}
			if strings.Contains(err.Error(), "do-not-leak") {
				t.Errorf("error leaks raw output: %q", err)
			}
		})
	}
}

func TestStructuredResultEnforcesFinishReason(t *testing.T) {
	t.Parallel()

	text := assistantMessage(&content.TextBlock{Text: `{"ok":true}`})
	terminal := assistantMessage(&content.ToolUseBlock{Name: inference.StructuredOutputToolName, Input: json.RawMessage(`{"ok":true}`)})
	ordinary := assistantMessage(&content.ToolUseBlock{Name: "read", Input: json.RawMessage(`{}`)})

	tests := []struct {
		name       string
		resp       *inference.Response
		want       string
		wantFinish stream.FinishReason
		wantBad    inference.MalformedStructuredOutputReason
	}{
		{name: "unknown text", resp: &inference.Response{Message: text}, want: `{"ok":true}`},
		{name: "unknown terminal", resp: &inference.Response{Message: terminal}, want: `{"ok":true}`},
		{name: "stop text", resp: &inference.Response{Message: text, FinishReason: stream.FinishReasonStop}, want: `{"ok":true}`},
		{name: "tool use terminal", resp: &inference.Response{Message: terminal, FinishReason: stream.FinishReasonToolUse}, want: `{"ok":true}`},
		{name: "length valid partial object", resp: &inference.Response{Message: text, FinishReason: stream.FinishReasonLength}, wantFinish: stream.FinishReasonLength},
		{name: "filtered malformed", resp: &inference.Response{Message: assistantMessage(&content.TextBlock{Text: `{"secret"`}), FinishReason: stream.FinishReasonContentFilter}, wantFinish: stream.FinishReasonContentFilter},
		{name: "stop terminal contradiction", resp: &inference.Response{Message: terminal, FinishReason: stream.FinishReasonStop}, wantFinish: stream.FinishReasonStop},
		{name: "stop ordinary contradiction", resp: &inference.Response{Message: ordinary, FinishReason: stream.FinishReasonStop}, wantFinish: stream.FinishReasonStop},
		{name: "tool use text contradiction", resp: &inference.Response{Message: text, FinishReason: stream.FinishReasonToolUse}, wantFinish: stream.FinishReasonToolUse},
		{name: "tool use ordinary contradiction", resp: &inference.Response{Message: ordinary, FinishReason: stream.FinishReasonToolUse}, wantFinish: stream.FinishReasonToolUse},
		{name: "future reason fails closed", resp: &inference.Response{Message: text, FinishReason: stream.FinishReason("future")}, wantFinish: stream.FinishReason("future")},
		{name: "nil response", wantBad: inference.MalformedReasonNilResponse},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := inference.StructuredResult(tt.resp)
			switch {
			case tt.wantFinish != "":
				var finishErr *inference.StructuredOutputFinishError
				if !errors.As(err, &finishErr) {
					t.Fatalf("error = %T %v, want *StructuredOutputFinishError", err, err)
				}
				if finishErr.Reason != tt.wantFinish {
					t.Errorf("Reason = %q, want %q", finishErr.Reason, tt.wantFinish)
				}
			case tt.wantBad != "":
				var malformed *inference.MalformedStructuredOutputError
				if !errors.As(err, &malformed) || malformed.ReasonCode != tt.wantBad {
					t.Fatalf("error = %T %v, want malformed reason %q", err, err, tt.wantBad)
				}
			default:
				if err != nil {
					t.Fatalf("StructuredResult() error = %v", err)
				}
				if string(got) != tt.want {
					t.Errorf("StructuredResult() = %q, want %q", got, tt.want)
				}
			}
		})
	}
}

func TestDecodeOutputStrictlyDecodesConcretePointer(t *testing.T) {
	t.Parallel()

	type result struct {
		Name  string `json:"name"`
		Count int    `json:"count"`
	}
	response := func(raw string) *inference.Response {
		return &inference.Response{
			Message:      assistantMessage(&content.TextBlock{Text: raw}),
			FinishReason: stream.FinishReasonStop,
		}
	}

	var got result
	if err := inference.DecodeOutput(response(`{"name":"世界","count":3}`), &got); err != nil {
		t.Fatalf("DecodeOutput() error = %v", err)
	}
	if got != (result{Name: "世界", Count: 3}) {
		t.Errorf("DecodeOutput() result = %+v", got)
	}

	var missing result
	if err := inference.DecodeMessageOutput(assistantMessage(&content.TextBlock{Text: `{"name":"ok"}`}), &missing); err != nil {
		t.Fatalf("DecodeMessageOutput() missing field error = %v; required fields are domain validation", err)
	}

	var nilResult *result
	var interfaceTarget any
	tests := []struct {
		name string
		resp *inference.Response
		out  any
	}{
		{name: "nil output", resp: response(`{}`)},
		{name: "non pointer", resp: response(`{}`), out: result{}},
		{name: "nil pointer", resp: response(`{}`), out: nilResult},
		{name: "interface pointer", resp: response(`{}`), out: &interfaceTarget},
		{name: "unknown field", resp: response(`{"name":"ok","extra":"secret-value"}`), out: &result{}},
		{name: "type mismatch", resp: response(`{"name":7}`), out: &result{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := inference.DecodeOutput(tt.resp, tt.out)
			var schemaErr *inference.SchemaValidationError
			if !errors.As(err, &schemaErr) {
				t.Fatalf("error = %T %v, want *SchemaValidationError", err, err)
			}
			if strings.Contains(err.Error(), "secret-value") {
				t.Errorf("error leaks raw output: %q", err)
			}
		})
	}
}

func TestResponseFinishReasonZeroValue(t *testing.T) {
	t.Parallel()

	var response inference.Response
	if response.FinishReason != stream.FinishReasonUnknown {
		t.Errorf("FinishReason = %q, want unknown zero value", response.FinishReason)
	}
}
