package openaiapi_test

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/openaiapi"
	failure "github.com/looprig/inference/failure"
	stream "github.com/looprig/inference/stream"
	usage "github.com/looprig/inference/usage"
)

// TestDecodeResponse_CompileTimeCheck asserts the exact signature of DecodeResponse.
func TestDecodeResponse_CompileTimeCheck(t *testing.T) {
	t.Parallel()
	var _ func([]byte) (*inference.Response, error) = openaiapi.DecodeResponse
}

func TestDecodeResponseFinishReason(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		wire string
		want stream.FinishReason
	}{
		{name: "stop", wire: "stop", want: stream.FinishReasonStop},
		{name: "tool calls", wire: "tool_calls", want: stream.FinishReasonToolUse},
		{name: "legacy function call", wire: "function_call", want: stream.FinishReasonToolUse},
		{name: "length", wire: "length", want: stream.FinishReasonLength},
		{name: "content filter", wire: "content_filter", want: stream.FinishReasonContentFilter},
		{name: "absent", wire: "", want: stream.FinishReasonUnknown},
		{name: "future value fails safe", wire: "future_reason", want: stream.FinishReasonUnknown},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			body := []byte(`{"model":"gpt-test","choices":[{"message":{"role":"assistant","content":"{}"},"finish_reason":"` + tt.wire + `"}]}`)
			response, err := openaiapi.DecodeResponse(body)
			if err != nil {
				t.Fatalf("DecodeResponse() error = %v", err)
			}
			if response.FinishReason != tt.want {
				t.Errorf("FinishReason = %q, want %q", response.FinishReason, tt.want)
			}
		})
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
		{
			name:       "cache reads writes and reasoning are disjoint",
			usageField: `,"usage":{"prompt_tokens":10,"completion_tokens":6,"prompt_tokens_details":{"cached_tokens":3,"cache_write_tokens":2},"completion_tokens_details":{"reasoning_tokens":4}}`,
			want:       &content.Usage{InputTokens: 5, OutputTokens: 6, CacheReadTokens: 3, CacheCreationTokens: 2, ReasoningTokens: 4},
		},
		{name: "negative prompt", usageField: `,"usage":{"prompt_tokens":-1}`, wantField: usage.UsageNormalizationFieldInputTokens, wantReason: usage.UsageNormalizationReasonNegative},
		{name: "negative completion", usageField: `,"usage":{"completion_tokens":-1}`, wantField: usage.UsageNormalizationFieldOutputTokens, wantReason: usage.UsageNormalizationReasonNegative},
		{name: "negative cache read", usageField: `,"usage":{"prompt_tokens_details":{"cached_tokens":-1}}`, wantField: usage.UsageNormalizationFieldCacheReadTokens, wantReason: usage.UsageNormalizationReasonNegative},
		{name: "negative cache creation", usageField: `,"usage":{"prompt_tokens_details":{"cache_write_tokens":-1}}`, wantField: usage.UsageNormalizationFieldCacheCreationTokens, wantReason: usage.UsageNormalizationReasonNegative},
		{name: "negative reasoning", usageField: `,"usage":{"completion_tokens_details":{"reasoning_tokens":-1}}`, wantField: usage.UsageNormalizationFieldReasoningTokens, wantReason: usage.UsageNormalizationReasonNegative},
		{name: "null prompt", usageField: `,"usage":{"prompt_tokens":null}`, wantField: usage.UsageNormalizationFieldInputTokens, wantReason: usage.UsageNormalizationReasonNull},
		{name: "fractional completion", usageField: `,"usage":{"completion_tokens":1.5}`, wantField: usage.UsageNormalizationFieldOutputTokens, wantReason: usage.UsageNormalizationReasonFractional},
		{name: "out of range cache read", usageField: `,"usage":{"prompt_tokens_details":{"cached_tokens":9223372036854775808}}`, wantField: usage.UsageNormalizationFieldCacheReadTokens, wantReason: usage.UsageNormalizationReasonOutOfRange},
		{name: "cache totals exceed prompt", usageField: `,"usage":{"prompt_tokens":4,"prompt_tokens_details":{"cached_tokens":3,"cache_write_tokens":2}}`, wantField: usage.UsageNormalizationFieldInputTokens, wantReason: usage.UsageNormalizationReasonComponentsExceedTotal},
		{name: "reasoning exceeds output", usageField: `,"usage":{"completion_tokens":2,"completion_tokens_details":{"reasoning_tokens":3}}`, wantField: usage.UsageNormalizationFieldReasoningTokens, wantReason: usage.UsageNormalizationReasonReasoningExceedsOutput},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			body := []byte(`{"model":"gpt-test","choices":[{"message":{"role":"assistant","content":"ok"}}]` + tt.usageField + `}`)
			response, err := openaiapi.DecodeResponse(body)
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

// blockType maps a concrete sealed block to its wire-tag BlockType, used to
// assert decoded block ordering without a Type field on the value.
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

func TestDecodeResponse(t *testing.T) {
	t.Parallel()

	rawArgs := json.RawMessage(`{"key":"value"}`)

	cases := []struct {
		name string
		body []byte

		// expected success fields
		wantModel        string
		wantBlockCount   int
		wantBlockTypes   []content.BlockType
		wantThinking     string
		wantText         string
		wantToolUseID    string
		wantToolUseName  string
		wantToolUseInput json.RawMessage
		wantUsageNil     bool
		wantInputTokens  content.TokenCount
		wantOutputTokens content.TokenCount

		// expected error fields
		wantErr       bool
		wantAPIErr    bool
		wantNoChoices bool
	}{
		{
			name: "text response",
			body: []byte(`{
				"id": "chatcmpl-1",
				"model": "gpt-4o",
				"choices": [{"message": {"role": "assistant", "content": "Hello, world!"}, "finish_reason": "stop"}],
				"usage": {"prompt_tokens": 10, "completion_tokens": 5}
			}`),
			wantModel:        "gpt-4o",
			wantBlockCount:   1,
			wantBlockTypes:   []content.BlockType{content.TypeText},
			wantText:         "Hello, world!",
			wantInputTokens:  10,
			wantOutputTokens: 5,
		},
		{
			name: "tool call response with null content",
			body: []byte(`{
				"id": "chatcmpl-2",
				"model": "gpt-4o",
				"choices": [{"message": {"role": "assistant", "content": null, "tool_calls": [{"id": "call_abc", "type": "function", "function": {"name": "get_weather", "arguments": {"key":"value"}}}]}, "finish_reason": "tool_calls"}],
				"usage": {"prompt_tokens": 20, "completion_tokens": 8}
			}`),
			wantModel:        "gpt-4o",
			wantBlockCount:   1,
			wantBlockTypes:   []content.BlockType{content.TypeToolUse},
			wantToolUseID:    "call_abc",
			wantToolUseName:  "get_weather",
			wantToolUseInput: rawArgs,
			wantInputTokens:  20,
			wantOutputTokens: 8,
		},
		{
			name: "reasoning content only",
			body: []byte(`{
				"id": "chatcmpl-3",
				"model": "o1-preview",
				"choices": [{"message": {"role": "assistant", "content": null, "reasoning_content": "I need to think about this..."}, "finish_reason": "stop"}],
				"usage": {"prompt_tokens": 30, "completion_tokens": 15}
			}`),
			wantModel:        "o1-preview",
			wantBlockCount:   1,
			wantBlockTypes:   []content.BlockType{content.TypeThinking},
			wantThinking:     "I need to think about this...",
			wantInputTokens:  30,
			wantOutputTokens: 15,
		},
		{
			name: "both reasoning and content",
			body: []byte(`{
				"id": "chatcmpl-4",
				"model": "o1-preview",
				"choices": [{"message": {"role": "assistant", "content": "My answer.", "reasoning_content": "First I thought..."}, "finish_reason": "stop"}],
				"usage": {"prompt_tokens": 50, "completion_tokens": 25}
			}`),
			wantModel:        "o1-preview",
			wantBlockCount:   2,
			wantBlockTypes:   []content.BlockType{content.TypeThinking, content.TypeText},
			wantThinking:     "First I thought...",
			wantText:         "My answer.",
			wantInputTokens:  50,
			wantOutputTokens: 25,
		},
		{
			name: "null usage",
			body: []byte(`{
				"id": "chatcmpl-5",
				"model": "gpt-4o",
				"choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
				"usage": null
			}`),
			wantModel:      "gpt-4o",
			wantBlockCount: 1,
			wantBlockTypes: []content.BlockType{content.TypeText},
			wantText:       "hello",
			wantUsageNil:   true,
		},
		{
			name: "usage present",
			body: []byte(`{
				"id": "chatcmpl-6",
				"model": "gpt-4o",
				"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
				"usage": {"prompt_tokens": 7, "completion_tokens": 3}
			}`),
			wantModel:        "gpt-4o",
			wantBlockCount:   1,
			wantText:         "hi",
			wantInputTokens:  7,
			wantOutputTokens: 3,
		},
		{
			name:          "empty choices",
			body:          []byte(`{"id":"x","model":"gpt-4o","choices":[],"usage":null}`),
			wantErr:       true,
			wantAPIErr:    true,
			wantNoChoices: true,
		},
		{
			name:    "invalid JSON",
			body:    []byte(`not json at all`),
			wantErr: true,
		},
		{
			name: "model field preserved",
			body: []byte(`{
				"id": "chatcmpl-7",
				"model": "deepseek-chat",
				"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
				"usage": null
			}`),
			wantModel:      "deepseek-chat",
			wantBlockCount: 1,
			wantText:       "ok",
			wantUsageNil:   true,
		},
		{
			name: "multiple tool calls",
			body: []byte(`{
				"id": "chatcmpl-8",
				"model": "gpt-4o",
				"choices": [{"message": {"role": "assistant", "content": null, "tool_calls": [
					{"id": "call_1", "type": "function", "function": {"name": "fn_a", "arguments": {"key":"value"}}},
					{"id": "call_2", "type": "function", "function": {"name": "fn_b", "arguments": {"key":"value"}}}
				]}, "finish_reason": "tool_calls"}],
				"usage": {"prompt_tokens": 40, "completion_tokens": 12}
			}`),
			wantModel:        "gpt-4o",
			wantBlockCount:   2,
			wantBlockTypes:   []content.BlockType{content.TypeToolUse, content.TypeToolUse},
			wantInputTokens:  40,
			wantOutputTokens: 12,
		},
		{
			name: "empty string content produces no text block",
			body: []byte(`{
				"id": "chatcmpl-9",
				"model": "gpt-4o",
				"choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
				"usage": null
			}`),
			wantModel:      "gpt-4o",
			wantBlockCount: 0,
			wantUsageNil:   true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			resp, err := openaiapi.DecodeResponse(tc.body)

			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got nil")
				}
				if tc.wantAPIErr {
					apiErr, ok := err.(*failure.APIError)
					if !ok {
						t.Fatalf("expected *failure.APIError, got %T: %v", err, err)
					}
					if tc.wantNoChoices && apiErr.Status != 0 {
						t.Errorf("no-choices APIError: want Status=0, got %d", apiErr.Status)
					}
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if resp == nil {
				t.Fatal("expected non-nil response")
			}

			if resp.Model != tc.wantModel {
				t.Errorf("Model: want %q, got %q", tc.wantModel, resp.Model)
			}

			if resp.Message == nil {
				t.Fatal("expected non-nil Message")
			}

			if got := len(resp.Message.Blocks); got != tc.wantBlockCount {
				t.Fatalf("block count: want %d, got %d", tc.wantBlockCount, got)
			}

			for i, wantType := range tc.wantBlockTypes {
				if i >= len(resp.Message.Blocks) {
					break
				}
				if got := blockType(resp.Message.Blocks[i]); got != wantType {
					t.Errorf("block[%d].Type: want %q, got %q", i, wantType, got)
				}
			}

			if tc.wantThinking != "" {
				var found bool
				for _, b := range resp.Message.Blocks {
					if tb, ok := b.(*content.ThinkingBlock); ok && tb.Thinking == tc.wantThinking {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("expected ThinkingBlock with %q", tc.wantThinking)
				}
			}

			if tc.wantText != "" {
				var found bool
				for _, b := range resp.Message.Blocks {
					if tb, ok := b.(*content.TextBlock); ok && tb.Text == tc.wantText {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("expected TextBlock with %q", tc.wantText)
				}
			}

			if tc.wantToolUseID != "" || tc.wantToolUseName != "" {
				var found bool
				for _, b := range resp.Message.Blocks {
					tu, ok := b.(*content.ToolUseBlock)
					if ok && tu.ID == tc.wantToolUseID && tu.Name == tc.wantToolUseName {
						found = true
						if string(tc.wantToolUseInput) != "" {
							if string(tu.Input) != string(tc.wantToolUseInput) {
								t.Errorf("ToolUseBlock.Input: want %s, got %s", tc.wantToolUseInput, tu.Input)
							}
						}
						break
					}
				}
				if !found {
					t.Errorf("expected ToolUseBlock id=%q name=%q", tc.wantToolUseID, tc.wantToolUseName)
				}
			}

			if tc.wantUsageNil {
				if resp.Usage != nil {
					t.Errorf("expected nil Usage, got %+v", resp.Usage)
				}
			} else {
				if resp.Usage == nil {
					t.Fatal("expected non-nil Usage")
				}
				if resp.Usage.InputTokens != tc.wantInputTokens {
					t.Errorf("InputTokens: want %d, got %d", tc.wantInputTokens, resp.Usage.InputTokens)
				}
				if resp.Usage.OutputTokens != tc.wantOutputTokens {
					t.Errorf("OutputTokens: want %d, got %d", tc.wantOutputTokens, resp.Usage.OutputTokens)
				}
			}
		})
	}
}
