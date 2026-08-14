package anthropicapi_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/anthropicapi"
	failure "github.com/looprig/inference/failure"
	model "github.com/looprig/inference/model"
	stream "github.com/looprig/inference/stream"
	usage "github.com/looprig/inference/usage"
)

// TestDecodeResponse_CompileTimeCheck pins the free function signature.
func TestDecodeResponse_CompileTimeCheck(t *testing.T) {
	t.Parallel()
	var _ func([]byte) (*inference.Response, error) = anthropicapi.DecodeResponse
}

func TestDecodeResponse_RedactedThinkingPreservesOpaqueState(t *testing.T) {
	t.Parallel()
	resp, err := anthropicapi.DecodeResponse([]byte(`{"type":"message","role":"assistant","model":"claude","content":[{"type":"redacted_thinking","data":"opaque+/="}],"stop_reason":"tool_use"}`))
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	thinking, ok := resp.Message.Blocks[0].(*content.ThinkingBlock)
	if !ok {
		t.Fatalf("block = %T, want ThinkingBlock", resp.Message.Blocks[0])
	}
	if got := string(thinking.ProviderState); got != `"opaque+/="` || thinking.ProviderStateFormat != "anthropic-redacted-thinking" {
		t.Fatalf("opaque state = %s/%q", got, thinking.ProviderStateFormat)
	}
	req := inference.Request{Model: model.Model{Name: "claude"}, Messages: content.AgenticMessages{aiMsg(thinking)}}
	raw, err := anthropicapi.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	if !strings.Contains(string(raw), `{"type":"redacted_thinking","data":"opaque+/="}`) {
		t.Fatalf("replay body = %s", raw)
	}
}

func TestDecodeResponseFinishReason(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		stopReason string
		want       stream.FinishReason
	}{
		{name: "stop", stopReason: "end_turn", want: stream.FinishReasonStop},
		{name: "tool use", stopReason: "tool_use", want: stream.FinishReasonToolUse},
		{name: "max tokens", stopReason: "max_tokens", want: stream.FinishReasonLength},
		{name: "refusal", stopReason: "refusal", want: stream.FinishReasonContentFilter},
		{name: "unknown", stopReason: "future_reason", want: stream.FinishReasonUnknown},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			body := []byte(`{"type":"message","model":"claude-test","content":[],"stop_reason":"` + tt.stopReason + `"}`)
			got, err := anthropicapi.DecodeResponse(body)
			if err != nil {
				t.Fatalf("DecodeResponse() error = %v", err)
			}
			if got.FinishReason != tt.want {
				t.Errorf("FinishReason = %q, want %q", got.FinishReason, tt.want)
			}
		})
	}
}

// blockType maps a decoded block to its wire tag for order assertions.
func blockType(b content.Block) content.BlockType {
	switch b.(type) {
	case *content.TextBlock:
		return content.TypeText
	case *content.ThinkingBlock:
		return content.TypeThinking
	case *content.ToolUseBlock:
		return content.TypeToolUse
	case *content.ImageBlock:
		return content.TypeImage
	default:
		return ""
	}
}

func TestDecodeResponseUsageNormalization(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		usageField string
		want       *content.Usage
		wantField  usage.UsageNormalizationField
		wantReason usage.UsageNormalizationReason
	}{
		{name: "absent usage is unknown", want: nil},
		{name: "present zero is known", usageField: `,"usage":{}`, want: &content.Usage{}},
		{name: "cache categories are disjoint", usageField: `,"usage":{"input_tokens":5,"output_tokens":6,"cache_read_input_tokens":3,"cache_creation_input_tokens":2}`, want: &content.Usage{InputTokens: 5, OutputTokens: 6, CacheReadTokens: 3, CacheCreationTokens: 2}},
		{name: "thinking tokens are an output subset", usageField: `,"usage":{"input_tokens":5,"output_tokens":6,"output_tokens_details":{"thinking_tokens":4}}`, want: &content.Usage{InputTokens: 5, OutputTokens: 6, ReasoningTokens: 4}},
		{name: "negative input", usageField: `,"usage":{"input_tokens":-1}`, wantField: usage.UsageNormalizationFieldInputTokens, wantReason: usage.UsageNormalizationReasonNegative},
		{name: "negative output", usageField: `,"usage":{"output_tokens":-1}`, wantField: usage.UsageNormalizationFieldOutputTokens, wantReason: usage.UsageNormalizationReasonNegative},
		{name: "negative cache read", usageField: `,"usage":{"cache_read_input_tokens":-1}`, wantField: usage.UsageNormalizationFieldCacheReadTokens, wantReason: usage.UsageNormalizationReasonNegative},
		{name: "negative cache creation", usageField: `,"usage":{"cache_creation_input_tokens":-1}`, wantField: usage.UsageNormalizationFieldCacheCreationTokens, wantReason: usage.UsageNormalizationReasonNegative},
		{name: "negative thinking", usageField: `,"usage":{"output_tokens_details":{"thinking_tokens":-1}}`, wantField: usage.UsageNormalizationFieldReasoningTokens, wantReason: usage.UsageNormalizationReasonNegative},
		// Anthropic declares both cache counts as anyOf[{integer, minimum 0},
		// {null}] with "default": null, so null is the documented value for a
		// turn that neither read nor wrote a cache. It normalizes to zero.
		{name: "null cache creation is zero", usageField: `,"usage":{"cache_creation_input_tokens":null}`, want: &content.Usage{}},
		{name: "null cache read is zero", usageField: `,"usage":{"cache_read_input_tokens":null}`, want: &content.Usage{}},
		// The relaxation is scoped to the two nullable members: input_tokens and
		// output_tokens are declared type: integer with no null branch, so an
		// explicit null there is still malformed.
		{name: "null input is still an error", usageField: `,"usage":{"input_tokens":null}`, wantField: usage.UsageNormalizationFieldInputTokens, wantReason: usage.UsageNormalizationReasonNull},
		{name: "null output is still an error", usageField: `,"usage":{"output_tokens":null}`, wantField: usage.UsageNormalizationFieldOutputTokens, wantReason: usage.UsageNormalizationReasonNull},
		{name: "fractional input", usageField: `,"usage":{"input_tokens":1.5}`, wantField: usage.UsageNormalizationFieldInputTokens, wantReason: usage.UsageNormalizationReasonFractional},
		{name: "out of range output", usageField: `,"usage":{"output_tokens":9223372036854775808}`, wantField: usage.UsageNormalizationFieldOutputTokens, wantReason: usage.UsageNormalizationReasonOutOfRange},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			body := []byte(`{"type":"message","model":"claude-test","content":[{"type":"text","text":"ok"}]` + tt.usageField + `}`)
			response, err := anthropicapi.DecodeResponse(body)
			if tt.wantReason != "" {
				var normalizationErr *usage.UsageNormalizationError
				if !errors.As(err, &normalizationErr) {
					t.Fatalf("DecodeResponse() error = %T %v, want *UsageNormalizationError", err, err)
				}
				if normalizationErr.Field != tt.wantField || normalizationErr.Reason != tt.wantReason {
					t.Errorf("normalization error = %+v, want field=%q reason=%q", normalizationErr, tt.wantField, tt.wantReason)
				}
				return
			}
			if err != nil {
				t.Fatalf("DecodeResponse() error = %v", err)
			}
			assertIndependentUsage(t, response, tt.want)
		})
	}
}

func assertIndependentUsage(t *testing.T, response *inference.Response, want *content.Usage) {
	t.Helper()
	if response.Usage == nil || response.Message.Usage == nil {
		if response.Usage != nil || response.Message.Usage != nil || want != nil {
			t.Fatalf("usage pointers = response:%+v message:%+v, want %+v", response.Usage, response.Message.Usage, want)
		}
		return
	}
	if want == nil || *response.Usage != *want || *response.Message.Usage != *want {
		t.Fatalf("usage = response:%+v message:%+v, want %+v", response.Usage, response.Message.Usage, want)
	}
	if response.Usage == response.Message.Usage {
		t.Fatal("Response.Usage and Message.Usage alias")
	}
	before := *response.Message.Usage
	response.Usage.InputTokens++
	if *response.Message.Usage != before {
		t.Errorf("mutating Response.Usage changed Message.Usage to %+v", response.Message.Usage)
	}
}

func TestDecodeResponse(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		body []byte

		wantErr    bool
		wantAPIErr bool

		wantModel        string
		wantTypes        []content.BlockType
		wantText         string
		wantThinking     string
		wantToolID       string
		wantToolName     string
		wantToolInput    string
		wantUsageNil     bool
		wantInputTokens  content.TokenCount
		wantOutputTokens content.TokenCount
	}{
		{
			name: "text response with usage",
			body: []byte(`{
				"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-4-8",
				"content":[{"type":"text","text":"Hello, world!"}],
				"stop_reason":"end_turn",
				"usage":{"input_tokens":10,"output_tokens":5}
			}`),
			wantModel:        "claude-opus-4-8",
			wantTypes:        []content.BlockType{content.TypeText},
			wantText:         "Hello, world!",
			wantInputTokens:  10,
			wantOutputTokens: 5,
		},
		{
			name: "tool_use response",
			body: []byte(`{
				"id":"msg_2","type":"message","role":"assistant","model":"claude-opus-4-8",
				"content":[{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Paris"}}],
				"stop_reason":"tool_use",
				"usage":{"input_tokens":20,"output_tokens":8}
			}`),
			wantModel:        "claude-opus-4-8",
			wantTypes:        []content.BlockType{content.TypeToolUse},
			wantToolID:       "toolu_1",
			wantToolName:     "get_weather",
			wantToolInput:    `{"city":"Paris"}`,
			wantInputTokens:  20,
			wantOutputTokens: 8,
		},
		{
			name: "text + tool_use in order",
			body: []byte(`{
				"id":"msg_3","type":"message","role":"assistant","model":"claude-opus-4-8",
				"content":[
					{"type":"text","text":"Let me check."},
					{"type":"tool_use","id":"toolu_2","name":"search","input":{"q":"x"}}
				],
				"usage":{"input_tokens":3,"output_tokens":4}
			}`),
			wantModel:        "claude-opus-4-8",
			wantTypes:        []content.BlockType{content.TypeText, content.TypeToolUse},
			wantText:         "Let me check.",
			wantToolID:       "toolu_2",
			wantToolName:     "search",
			wantToolInput:    `{"q":"x"}`,
			wantInputTokens:  3,
			wantOutputTokens: 4,
		},
		{
			name: "thinking block decodes with signature",
			body: []byte(`{
				"id":"msg_4","type":"message","role":"assistant","model":"claude-opus-4-8",
				"content":[{"type":"thinking","thinking":"pondering","signature":"sig123"},{"type":"text","text":"done"}],
				"usage":{"input_tokens":1,"output_tokens":2}
			}`),
			wantModel:        "claude-opus-4-8",
			wantTypes:        []content.BlockType{content.TypeThinking, content.TypeText},
			wantThinking:     "pondering",
			wantText:         "done",
			wantInputTokens:  1,
			wantOutputTokens: 2,
		},
		{
			name: "redacted thinking is preserved and unknown block types are skipped",
			body: []byte(`{
				"id":"msg_5","type":"message","role":"assistant","model":"claude-opus-4-8",
				"content":[{"type":"redacted_thinking","data":"..."},{"type":"future_block"},{"type":"text","text":"visible"}],
				"usage":{"input_tokens":1,"output_tokens":1}
			}`),
			wantModel:        "claude-opus-4-8",
			wantTypes:        []content.BlockType{content.TypeThinking, content.TypeText},
			wantText:         "visible",
			wantInputTokens:  1,
			wantOutputTokens: 1,
		},
		{
			name: "empty content is valid (no error)",
			body: []byte(`{
				"id":"msg_6","type":"message","role":"assistant","model":"claude-opus-4-8",
				"content":[],
				"stop_reason":"end_turn",
				"usage":{"input_tokens":7,"output_tokens":0}
			}`),
			wantModel:        "claude-opus-4-8",
			wantTypes:        nil,
			wantInputTokens:  7,
			wantOutputTokens: 0,
		},
		{
			name: "missing usage yields nil Usage",
			body: []byte(`{
				"id":"msg_7","type":"message","role":"assistant","model":"claude-opus-4-8",
				"content":[{"type":"text","text":"hi"}]
			}`),
			wantModel:    "claude-opus-4-8",
			wantTypes:    []content.BlockType{content.TypeText},
			wantText:     "hi",
			wantUsageNil: true,
		},
		{
			name:       "error envelope is an APIError",
			body:       []byte(`{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`),
			wantErr:    true,
			wantAPIErr: true,
		},
		{
			name:    "malformed json is an error",
			body:    []byte(`{not json`),
			wantErr: true,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			resp, err := anthropicapi.DecodeResponse(tc.body)
			if (err != nil) != tc.wantErr {
				t.Fatalf("DecodeResponse err = %v, wantErr %v", err, tc.wantErr)
			}
			if tc.wantErr {
				if tc.wantAPIErr {
					var apiErr *failure.APIError
					if !errors.As(err, &apiErr) {
						t.Errorf("error = %v, want *failure.APIError", err)
					}
				}
				return
			}

			if resp.Model != tc.wantModel {
				t.Errorf("Model = %q, want %q", resp.Model, tc.wantModel)
			}
			if resp.Message == nil || resp.Message.Role != content.RoleAssistant {
				t.Fatalf("Message = %+v, want assistant role", resp.Message)
			}

			blocks := resp.Message.Blocks
			if len(blocks) != len(tc.wantTypes) {
				t.Fatalf("block count = %d, want %d", len(blocks), len(tc.wantTypes))
			}
			for i, wt := range tc.wantTypes {
				if got := blockType(blocks[i]); got != wt {
					t.Errorf("block[%d] type = %q, want %q", i, got, wt)
				}
			}

			for _, b := range blocks {
				switch v := b.(type) {
				case *content.TextBlock:
					if tc.wantText != "" && v.Text != tc.wantText {
						t.Errorf("text = %q, want %q", v.Text, tc.wantText)
					}
				case *content.ThinkingBlock:
					if tc.wantThinking != "" && v.Thinking != tc.wantThinking {
						t.Errorf("thinking = %q, want %q", v.Thinking, tc.wantThinking)
					}
				case *content.ToolUseBlock:
					if v.ID != tc.wantToolID {
						t.Errorf("tool id = %q, want %q", v.ID, tc.wantToolID)
					}
					if v.Name != tc.wantToolName {
						t.Errorf("tool name = %q, want %q", v.Name, tc.wantToolName)
					}
					if string(v.Input) != tc.wantToolInput {
						t.Errorf("tool input = %s, want %s", v.Input, tc.wantToolInput)
					}
				}
			}

			if tc.wantUsageNil {
				if resp.Usage != nil {
					t.Errorf("Usage = %+v, want nil", resp.Usage)
				}
				return
			}
			if resp.Usage == nil {
				t.Fatalf("Usage = nil, want non-nil")
			}
			if resp.Usage.InputTokens != tc.wantInputTokens || resp.Usage.OutputTokens != tc.wantOutputTokens {
				t.Errorf("Usage = %+v, want {In:%d Out:%d}", resp.Usage, tc.wantInputTokens, tc.wantOutputTokens)
			}
		})
	}
}
