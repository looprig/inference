package geminiapi_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/geminiapi"
)

// TestDecodeResponse_CompileTimeCheck asserts the exact signature of DecodeResponse.
func TestDecodeResponse_CompileTimeCheck(t *testing.T) {
	t.Parallel()
	var _ func([]byte) (*inference.Response, error) = geminiapi.DecodeResponse
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
	default:
		return ""
	}
}

func TestDecodeResponseUsageNormalization(t *testing.T) {
	t.Parallel()

	maxInt := int(^uint(0) >> 1)
	tests := []struct {
		name       string
		usageField string
		want       *content.Usage
		wantField  inference.UsageNormalizationField
		wantReason inference.UsageNormalizationReason
	}{
		{name: "absent usage is unknown", want: nil},
		{name: "present zero is known", usageField: `,"usageMetadata":{}`, want: &content.Usage{}},
		{name: "cache read and thoughts are disjoint", usageField: `,"usageMetadata":{"promptTokenCount":10,"cachedContentTokenCount":3,"candidatesTokenCount":5,"thoughtsTokenCount":2}`, want: &content.Usage{InputTokens: 7, OutputTokens: 7, CacheReadTokens: 3, ReasoningTokens: 2}},
		{name: "negative prompt", usageField: `,"usageMetadata":{"promptTokenCount":-1}`, wantField: inference.UsageNormalizationFieldInputTokens, wantReason: inference.UsageNormalizationReasonNegative},
		{name: "negative candidates", usageField: `,"usageMetadata":{"candidatesTokenCount":-1}`, wantField: inference.UsageNormalizationFieldOutputTokens, wantReason: inference.UsageNormalizationReasonNegative},
		{name: "negative cache read", usageField: `,"usageMetadata":{"cachedContentTokenCount":-1}`, wantField: inference.UsageNormalizationFieldCacheReadTokens, wantReason: inference.UsageNormalizationReasonNegative},
		{name: "negative thoughts", usageField: `,"usageMetadata":{"thoughtsTokenCount":-1}`, wantField: inference.UsageNormalizationFieldReasoningTokens, wantReason: inference.UsageNormalizationReasonNegative},
		{name: "cache exceeds prompt", usageField: `,"usageMetadata":{"promptTokenCount":2,"cachedContentTokenCount":3}`, wantField: inference.UsageNormalizationFieldInputTokens, wantReason: inference.UsageNormalizationReasonComponentsExceedTotal},
		{name: "max int sum is representable", usageField: fmt.Sprintf(`,"usageMetadata":{"candidatesTokenCount":%d,"thoughtsTokenCount":%d}`, maxInt, maxInt), want: &content.Usage{OutputTokens: content.TokenCount(maxInt) + content.TokenCount(maxInt), ReasoningTokens: content.TokenCount(maxInt)}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			body := []byte(`{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]` + tt.usageField + `}`)
			response, err := geminiapi.DecodeResponse(body)
			if tt.wantReason != "" {
				var normalizationErr *inference.UsageNormalizationError
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

		wantModel        string
		wantBlockTypes   []content.BlockType
		wantText         string
		wantThinking     string
		wantToolUseID    string
		wantToolUseName  string
		wantToolUseInput string
		wantUsageNil     bool
		wantInputTokens  content.TokenCount
		wantOutputTokens content.TokenCount

		wantErr       bool
		wantAPIErr    bool
		wantNoCandido bool
	}{
		{
			name: "text response with usage and modelVersion",
			body: []byte(`{
				"candidates": [{"content": {"parts": [{"text": "Hello, world!"}], "role": "model"}, "finishReason": "STOP", "index": 0}],
				"usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 12, "totalTokenCount": 16},
				"modelVersion": "gemini-2.5-flash"
			}`),
			wantModel:        "gemini-2.5-flash",
			wantBlockTypes:   []content.BlockType{content.TypeText},
			wantText:         "Hello, world!",
			wantInputTokens:  4,
			wantOutputTokens: 12,
		},
		{
			name: "functionCall response",
			body: []byte(`{
				"candidates": [{"content": {"parts": [{"functionCall": {"name": "get_weather", "args": {"location": "Boston, MA"}}}], "role": "model"}, "finishReason": "STOP"}],
				"usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 8},
				"modelVersion": "gemini-2.5-pro"
			}`),
			wantModel:        "gemini-2.5-pro",
			wantBlockTypes:   []content.BlockType{content.TypeToolUse},
			wantToolUseName:  "get_weather",
			wantToolUseInput: `{"location": "Boston, MA"}`,
			wantInputTokens:  20,
			wantOutputTokens: 8,
		},
		{
			name: "functionCall with id preserved",
			body: []byte(`{
				"candidates": [{"content": {"parts": [{"functionCall": {"id": "call_7", "name": "run", "args": {}}}], "role": "model"}}]
			}`),
			wantBlockTypes:  []content.BlockType{content.TypeToolUse},
			wantToolUseID:   "call_7",
			wantToolUseName: "run",
			wantUsageNil:    true,
		},
		{
			name: "thought part then text: two ordered blocks",
			body: []byte(`{
				"candidates": [{"content": {"parts": [{"text": "planning", "thought": true}, {"text": "the answer"}], "role": "model"}}],
				"usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 9}
			}`),
			wantBlockTypes:   []content.BlockType{content.TypeThinking, content.TypeText},
			wantThinking:     "planning",
			wantText:         "the answer",
			wantInputTokens:  5,
			wantOutputTokens: 9,
		},
		{
			name:           "empty text part yields no block",
			body:           []byte(`{"candidates": [{"content": {"parts": [{"text": ""}], "role": "model"}}]}`),
			wantBlockTypes: nil,
			wantUsageNil:   true,
		},
		{
			name:          "no candidates is an APIError",
			body:          []byte(`{"candidates": [], "usageMetadata": null}`),
			wantErr:       true,
			wantAPIErr:    true,
			wantNoCandido: true,
		},
		{
			name:    "invalid JSON is an error",
			body:    []byte(`not json`),
			wantErr: true,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			resp, err := geminiapi.DecodeResponse(tc.body)

			if tc.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if tc.wantAPIErr {
					apiErr, ok := err.(*inference.APIError)
					if !ok {
						t.Fatalf("expected *inference.APIError, got %T: %v", err, err)
					}
					if tc.wantNoCandido && apiErr.Status != 0 {
						t.Errorf("no-candidates APIError: want Status=0, got %d", apiErr.Status)
					}
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if resp == nil || resp.Message == nil {
				t.Fatal("expected non-nil response and message")
			}
			if resp.Model != tc.wantModel {
				t.Errorf("Model = %q, want %q", resp.Model, tc.wantModel)
			}
			if got := len(resp.Message.Blocks); got != len(tc.wantBlockTypes) {
				t.Fatalf("block count = %d, want %d", got, len(tc.wantBlockTypes))
			}
			for i, want := range tc.wantBlockTypes {
				if got := blockType(resp.Message.Blocks[i]); got != want {
					t.Errorf("block[%d] type = %q, want %q", i, got, want)
				}
			}
			if tc.wantText != "" {
				assertHasText(t, resp.Message.Blocks, tc.wantText)
			}
			if tc.wantThinking != "" {
				assertHasThinking(t, resp.Message.Blocks, tc.wantThinking)
			}
			if tc.wantToolUseName != "" {
				assertHasToolUse(t, resp.Message.Blocks, tc.wantToolUseID, tc.wantToolUseName, tc.wantToolUseInput)
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
					t.Errorf("InputTokens = %d, want %d", resp.Usage.InputTokens, tc.wantInputTokens)
				}
				if resp.Usage.OutputTokens != tc.wantOutputTokens {
					t.Errorf("OutputTokens = %d, want %d", resp.Usage.OutputTokens, tc.wantOutputTokens)
				}
			}
		})
	}
}

func assertHasText(t *testing.T, blocks []content.Block, want string) {
	t.Helper()
	for _, b := range blocks {
		if tb, ok := b.(*content.TextBlock); ok && tb.Text == want {
			return
		}
	}
	t.Errorf("expected TextBlock with %q", want)
}

func assertHasThinking(t *testing.T, blocks []content.Block, want string) {
	t.Helper()
	for _, b := range blocks {
		if tb, ok := b.(*content.ThinkingBlock); ok && tb.Thinking == want {
			return
		}
	}
	t.Errorf("expected ThinkingBlock with %q", want)
}

func assertHasToolUse(t *testing.T, blocks []content.Block, id, name, input string) {
	t.Helper()
	for _, b := range blocks {
		tu, ok := b.(*content.ToolUseBlock)
		if !ok || tu.Name != name || tu.ID != id {
			continue
		}
		if input != "" && string(tu.Input) != input {
			t.Errorf("ToolUseBlock.Input = %s, want %s", tu.Input, input)
		}
		return
	}
	t.Errorf("expected ToolUseBlock id=%q name=%q", id, name)
}

// TestDecodeResponse_EmptyArgsNormalized confirms a functionCall with no args
// decodes to an empty-object Input rather than an empty/invalid RawMessage.
func TestDecodeResponse_EmptyArgsNormalized(t *testing.T) {
	t.Parallel()

	body := []byte(`{"candidates": [{"content": {"parts": [{"functionCall": {"name": "noop"}}], "role": "model"}}]}`)
	resp, err := geminiapi.DecodeResponse(body)
	if err != nil {
		t.Fatalf("DecodeResponse error: %v", err)
	}
	if len(resp.Message.Blocks) != 1 {
		t.Fatalf("expected 1 block, got %d", len(resp.Message.Blocks))
	}
	tu, ok := resp.Message.Blocks[0].(*content.ToolUseBlock)
	if !ok {
		t.Fatalf("expected ToolUseBlock, got %T", resp.Message.Blocks[0])
	}
	if string(tu.Input) != "{}" {
		t.Errorf("Input = %s, want {}", tu.Input)
	}
	if !json.Valid(tu.Input) {
		t.Errorf("Input is not valid JSON: %s", tu.Input)
	}
}
