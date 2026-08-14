package geminiapi_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/geminiapi"
	failure "github.com/looprig/inference/failure"
	model "github.com/looprig/inference/model"
	stream "github.com/looprig/inference/stream"
	usage "github.com/looprig/inference/usage"
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
		wantField  usage.UsageNormalizationField
		wantReason usage.UsageNormalizationReason
	}{
		{name: "absent usage is unknown", want: nil},
		{name: "present zero is known", usageField: `,"usageMetadata":{}`, want: &content.Usage{}},
		{name: "cache read and thoughts are disjoint", usageField: `,"usageMetadata":{"promptTokenCount":10,"cachedContentTokenCount":3,"candidatesTokenCount":5,"thoughtsTokenCount":2}`, want: &content.Usage{InputTokens: 7, OutputTokens: 7, CacheReadTokens: 3, ReasoningTokens: 2}},
		{name: "negative prompt", usageField: `,"usageMetadata":{"promptTokenCount":-1}`, wantField: usage.UsageNormalizationFieldInputTokens, wantReason: usage.UsageNormalizationReasonNegative},
		{name: "negative candidates", usageField: `,"usageMetadata":{"candidatesTokenCount":-1}`, wantField: usage.UsageNormalizationFieldOutputTokens, wantReason: usage.UsageNormalizationReasonNegative},
		{name: "negative cache read", usageField: `,"usageMetadata":{"cachedContentTokenCount":-1}`, wantField: usage.UsageNormalizationFieldCacheReadTokens, wantReason: usage.UsageNormalizationReasonNegative},
		{name: "negative thoughts", usageField: `,"usageMetadata":{"thoughtsTokenCount":-1}`, wantField: usage.UsageNormalizationFieldReasoningTokens, wantReason: usage.UsageNormalizationReasonNegative},
		{name: "null prompt", usageField: `,"usageMetadata":{"promptTokenCount":null}`, wantField: usage.UsageNormalizationFieldInputTokens, wantReason: usage.UsageNormalizationReasonNull},
		{name: "fractional candidates", usageField: `,"usageMetadata":{"candidatesTokenCount":1.5}`, wantField: usage.UsageNormalizationFieldOutputTokens, wantReason: usage.UsageNormalizationReasonFractional},
		{name: "out of range thoughts", usageField: `,"usageMetadata":{"thoughtsTokenCount":9223372036854775808}`, wantField: usage.UsageNormalizationFieldReasoningTokens, wantReason: usage.UsageNormalizationReasonOutOfRange},
		{name: "cache exceeds prompt", usageField: `,"usageMetadata":{"promptTokenCount":2,"cachedContentTokenCount":3}`, wantField: usage.UsageNormalizationFieldInputTokens, wantReason: usage.UsageNormalizationReasonComponentsExceedTotal},
		{name: "max int sum is representable", usageField: fmt.Sprintf(`,"usageMetadata":{"candidatesTokenCount":%d,"thoughtsTokenCount":%d}`, maxInt, maxInt), want: &content.Usage{OutputTokens: content.TokenCount(maxInt) + content.TokenCount(maxInt), ReasoningTokens: content.TokenCount(maxInt)}},
		{name: "total absent", usageField: `,"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":3,"thoughtsTokenCount":4}`, want: &content.Usage{InputTokens: 2, OutputTokens: 7, ReasoningTokens: 4}},
		{name: "explicit zero total exact", usageField: `,"usageMetadata":{"totalTokenCount":0}`, want: &content.Usage{}},
		// A reported total that disagrees with the modelled components is an
		// accounting difference, not a corrupt response: totalTokenCount feeds no
		// field of the neutral Usage, so failing on it could only ever discard a
		// completed generation. Its own per-field validation below still applies.
		{name: "total below the modelled components is tolerated", usageField: `,"usageMetadata":{"promptTokenCount":1,"totalTokenCount":0}`, want: &content.Usage{InputTokens: 1}},
		{name: "total above the modelled components is tolerated", usageField: `,"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":999}`, want: &content.Usage{InputTokens: 1, OutputTokens: 1}},
		{name: "negative total", usageField: `,"usageMetadata":{"totalTokenCount":-1}`, wantField: usage.UsageNormalizationFieldTotalTokens, wantReason: usage.UsageNormalizationReasonNegative},
		{name: "null total", usageField: `,"usageMetadata":{"totalTokenCount":null}`, wantField: usage.UsageNormalizationFieldTotalTokens, wantReason: usage.UsageNormalizationReasonNull},
		{name: "fractional total", usageField: `,"usageMetadata":{"totalTokenCount":1.5}`, wantField: usage.UsageNormalizationFieldTotalTokens, wantReason: usage.UsageNormalizationReasonFractional},
		{name: "out of range total", usageField: `,"usageMetadata":{"totalTokenCount":9223372036854775808}`, wantField: usage.UsageNormalizationFieldTotalTokens, wantReason: usage.UsageNormalizationReasonOutOfRange},
		{name: "exact nonzero total", usageField: `,"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":3,"thoughtsTokenCount":4,"totalTokenCount":9}`, want: &content.Usage{InputTokens: 2, OutputTokens: 7, ReasoningTokens: 4}},
		{name: "components far beyond the reported total are tolerated", usageField: fmt.Sprintf(`,"usageMetadata":{"promptTokenCount":%d,"candidatesTokenCount":%d,"thoughtsTokenCount":2,"totalTokenCount":0}`, maxInt, maxInt), want: &content.Usage{InputTokens: content.TokenCount(maxInt), OutputTokens: content.TokenCount(maxInt) + 2, ReasoningTokens: 2}},
		{name: "maximum total boundary", usageField: `,"usageMetadata":{"promptTokenCount":9223372036854775807,"totalTokenCount":9223372036854775807}`, want: &content.Usage{InputTokens: 9223372036854775807}},
		// toolUsePromptTokenCount is billable input reported apart from
		// promptTokenCount, so it is added to InputTokens rather than dropped.
		{name: "tool-use prompt tokens join the input", usageField: `,"usageMetadata":{"promptTokenCount":50,"toolUsePromptTokenCount":30,"candidatesTokenCount":20}`, want: &content.Usage{InputTokens: 80, OutputTokens: 20}},
		{name: "tool-use prompt tokens net of cached content", usageField: `,"usageMetadata":{"promptTokenCount":50,"cachedContentTokenCount":10,"toolUsePromptTokenCount":30}`, want: &content.Usage{InputTokens: 70, CacheReadTokens: 10}},
		{name: "negative tool-use prompt tokens", usageField: `,"usageMetadata":{"toolUsePromptTokenCount":-1}`, wantField: usage.UsageNormalizationFieldInputTokens, wantReason: usage.UsageNormalizationReasonNegative},
		{name: "null tool-use prompt tokens", usageField: `,"usageMetadata":{"toolUsePromptTokenCount":null}`, wantField: usage.UsageNormalizationFieldInputTokens, wantReason: usage.UsageNormalizationReasonNull},
		{name: "tool-use prompt sum at the representable boundary", usageField: fmt.Sprintf(`,"usageMetadata":{"promptTokenCount":%d,"toolUsePromptTokenCount":%d}`, maxInt, maxInt), want: &content.Usage{InputTokens: content.TokenCount(maxInt) + content.TokenCount(maxInt)}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			body := []byte(`{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]` + tt.usageField + `}`)
			response, err := geminiapi.DecodeResponse(body)
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
			wantModel:      "gemini-2.5-pro",
			wantBlockTypes: []content.BlockType{content.TypeToolUse},
			// FunctionCall.id is Optional on the wire, so an absent id becomes
			// the codec's internal per-turn ordinal (never re-emitted to Gemini).
			wantToolUseID:    "gemini-positional-call-0",
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
					apiErr, ok := err.(*failure.APIError)
					if !ok {
						t.Fatalf("expected *failure.APIError, got %T: %v", err, err)
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

func TestDecodeResponse_FinishReason(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		body string
		want stream.FinishReason
	}{
		{name: "stop", body: `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`, want: stream.FinishReasonStop},
		{name: "function call stop", body: `{"candidates":[{"content":{"parts":[{"functionCall":{"name":"search","args":{}}}]},"finishReason":"STOP"}]}`, want: stream.FinishReasonToolUse},
		{name: "function call without reason", body: `{"candidates":[{"content":{"parts":[{"functionCall":{"name":"search","args":{}}}]}}]}`, want: stream.FinishReasonToolUse},
		{name: "length overrides function call", body: `{"candidates":[{"content":{"parts":[{"functionCall":{"name":"search","args":{}}}]},"finishReason":"MAX_TOKENS"}]}`, want: stream.FinishReasonLength},
		{name: "safety", body: `{"candidates":[{"content":{"parts":[]},"finishReason":"SAFETY"}]}`, want: stream.FinishReasonContentFilter},
		{name: "unknown fails closed to unknown", body: `{"candidates":[{"content":{"parts":[]},"finishReason":"FUTURE_REASON"}]}`, want: stream.FinishReasonUnknown},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			response, err := geminiapi.DecodeResponse([]byte(tt.body))
			if err != nil {
				t.Fatalf("DecodeResponse() error = %v", err)
			}
			if response.FinishReason != tt.want {
				t.Errorf("FinishReason = %q, want %q", response.FinishReason, tt.want)
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

// TestDecodeResponse_ThoughtSignaturePopulatesProviderState proves buildBlocks
// now populates ThinkingBlock.ProviderState from the wire thoughtSignature
// (via content.NewThinkingBlock) instead of the prior bare struct literal that
// dropped it — the decode.go fix this task makes, mirroring the sibling
// encode.go fix that stops silently dropping a ThinkingBlock's signature on
// the outbound leg.
func TestDecodeResponse_ThoughtSignaturePopulatesProviderState(t *testing.T) {
	t.Parallel()

	body := []byte(`{
		"candidates": [{"content": {"parts": [
			{"text": "planning", "thought": true, "thoughtSignature": "opaque-sig"}
		], "role": "model"}}]
	}`)
	resp, err := geminiapi.DecodeResponse(body)
	if err != nil {
		t.Fatalf("DecodeResponse error: %v", err)
	}
	if len(resp.Message.Blocks) != 1 {
		t.Fatalf("expected 1 block, got %d", len(resp.Message.Blocks))
	}
	tb, ok := resp.Message.Blocks[0].(*content.ThinkingBlock)
	if !ok {
		t.Fatalf("expected ThinkingBlock, got %T", resp.Message.Blocks[0])
	}
	if tb.Thinking != "planning" {
		t.Errorf("Thinking = %q, want planning", tb.Thinking)
	}
	var sig string
	if err := json.Unmarshal(tb.ProviderState, &sig); err != nil {
		t.Fatalf("ProviderState not a JSON string: %v (%s)", err, tb.ProviderState)
	}
	if sig != "opaque-sig" {
		t.Errorf("ProviderState = %q, want opaque-sig", sig)
	}
}

func TestDecodeResponse_FunctionCallThoughtSignatureRoundTripsPositionally(t *testing.T) {
	t.Parallel()
	body := []byte(`{"candidates":[{"content":{"role":"model","parts":[{"thoughtSignature":"call-sig","functionCall":{"id":"call_1","name":"tool","args":{}}}]},"finishReason":"STOP"}]}`)
	resp, err := geminiapi.DecodeResponse(body)
	if err != nil {
		t.Fatalf("DecodeResponse: %v", err)
	}
	tool, ok := resp.Message.Blocks[0].(*content.ToolUseBlock)
	if !ok || string(tool.ProviderState) != `"call-sig"` || tool.ProviderStateFormat != "gemini" {
		t.Fatalf("tool block = %#v", resp.Message.Blocks[0])
	}
	req := inference.Request{Model: model.Model{Name: "gemini-test"}, Messages: content.AgenticMessages{
		&content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{tool}}},
	}}
	raw, err := geminiapi.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}
	var request map[string]any
	if err := json.Unmarshal(raw, &request); err != nil {
		t.Fatalf("request json: %v", err)
	}
	part := request["contents"].([]any)[0].(map[string]any)["parts"].([]any)[0].(map[string]any)
	if part["thoughtSignature"] != "call-sig" || part["functionCall"] == nil {
		t.Fatalf("signature moved off functionCall part: %#v", part)
	}
}

// TestDecodeResponse_ThoughtWithoutSignatureLeavesProviderStateNil proves a
// thought part with no thoughtSignature still decodes to a ThinkingBlock, but
// with ProviderState left nil rather than an empty-but-non-nil value.
func TestDecodeResponse_ThoughtWithoutSignatureLeavesProviderStateNil(t *testing.T) {
	t.Parallel()

	body := []byte(`{"candidates": [{"content": {"parts": [{"text": "planning", "thought": true}], "role": "model"}}]}`)
	resp, err := geminiapi.DecodeResponse(body)
	if err != nil {
		t.Fatalf("DecodeResponse error: %v", err)
	}
	tb, ok := resp.Message.Blocks[0].(*content.ThinkingBlock)
	if !ok {
		t.Fatalf("expected ThinkingBlock, got %T", resp.Message.Blocks[0])
	}
	if tb.ProviderState != nil {
		t.Errorf("ProviderState = %q, want nil", tb.ProviderState)
	}
}

// TestDecodeResponse_PromptBlocked covers the shape a safety-blocked PROMPT
// takes: no candidates at all, the reason in promptFeedback.blockReason with its
// safetyRatings alongside, and prompt tokens in usageMetadata that were billed
// even though nothing was generated. The discovery document is explicit that a
// response returns no candidates "only if there was something wrong with the
// prompt (check prompt_feedback)", so reporting it as a bare, statusless
// APIError throws away the one diagnostic the response carried.
func TestDecodeResponse_PromptBlocked(t *testing.T) {
	t.Parallel()

	body := []byte(`{
		"promptFeedback": {
			"blockReason": "PROHIBITED_CONTENT",
			"safetyRatings": [
				{"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "probability": "HIGH", "blocked": true},
				{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "NEGLIGIBLE"}
			]
		},
		"usageMetadata": {"promptTokenCount": 17, "totalTokenCount": 17},
		"modelVersion": "gemini-2.5-flash"
	}`)

	_, err := geminiapi.DecodeResponse(body)
	var blocked *geminiapi.PromptBlockedError
	if !errors.As(err, &blocked) {
		t.Fatalf("DecodeResponse() error = %v (%T), want *geminiapi.PromptBlockedError", err, err)
	}
	if blocked.BlockReason != "PROHIBITED_CONTENT" {
		t.Errorf("BlockReason = %q, want PROHIBITED_CONTENT", blocked.BlockReason)
	}
	if len(blocked.SafetyRatings) != 2 {
		t.Fatalf("SafetyRatings = %d, want 2", len(blocked.SafetyRatings))
	}
	if got := blocked.SafetyRatings[0]; got.Category != "HARM_CATEGORY_SEXUALLY_EXPLICIT" || got.Probability != "HIGH" || !got.Blocked {
		t.Errorf("SafetyRatings[0] = %+v, want the blocking rating", got)
	}
	if blocked.SafetyRatings[1].Blocked {
		t.Errorf("SafetyRatings[1] = %+v, want blocked=false", blocked.SafetyRatings[1])
	}
	// The prompt was charged, so the usage must survive the failure.
	if blocked.Usage == nil {
		t.Fatal("Usage = nil, want the billed prompt tokens")
	}
	if got := blocked.Usage.InputTokens; got != 17 {
		t.Errorf("Usage.InputTokens = %d, want 17", got)
	}
	if !strings.Contains(blocked.Error(), "PROHIBITED_CONTENT") {
		t.Errorf("Error() = %q, want the block reason named", blocked.Error())
	}

	// Callers that already classify on *failure.APIError keep working, and now
	// see a policy code instead of an empty one.
	var apiErr *failure.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error does not unwrap to *failure.APIError: %v", err)
	}
	if apiErr.Code != "content_policy_violation" {
		t.Errorf("APIError.Code = %q, want content_policy_violation", apiErr.Code)
	}
}

// A blockReason outside the enum the discovery document publishes is not copied
// into the error: provider strings are not retained in this module's failures
// (see failure.APIError's closed code allowlist). The response is still reported
// as a block, since the field's presence is what says the prompt was refused.
func TestDecodeResponse_PromptBlockedUnknownReason(t *testing.T) {
	t.Parallel()

	body := []byte(`{"promptFeedback": {"blockReason": "SOMETHING_GOOGLE_ADDED_LATER"}}`)
	_, err := geminiapi.DecodeResponse(body)
	var blocked *geminiapi.PromptBlockedError
	if !errors.As(err, &blocked) {
		t.Fatalf("DecodeResponse() error = %v (%T), want *geminiapi.PromptBlockedError", err, err)
	}
	if blocked.BlockReason != "" {
		t.Errorf("BlockReason = %q, want it withheld as unrecognized", blocked.BlockReason)
	}
}

// A candidate-less response with nothing to explain it stays the generic,
// statusless APIError it has always been — there is no reason to report.
func TestDecodeResponse_NoCandidatesWithoutFeedback(t *testing.T) {
	t.Parallel()

	_, err := geminiapi.DecodeResponse([]byte(`{"candidates": []}`))
	var blocked *geminiapi.PromptBlockedError
	if errors.As(err, &blocked) {
		t.Fatalf("error = %v, want no block reason invented", err)
	}
	var apiErr *failure.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error = %v (%T), want *failure.APIError", err, err)
	}
}

// TestDecodeResponse_GroundedToolUsePromptTokensKeepTheGeneration is the
// grounded / code-execution regression. UsageMetadata in the Gemini discovery
// document declares toolUsePromptTokenCount ("Output only. Number of tokens
// present in tool-use prompt(s)") as a first-class member, and it is a component
// of totalTokenCount. A codec that does not model it computes a short total and
// then throws away a completed, fully-formed answer over the difference — an
// accounting field discarding a generation.
func TestDecodeResponse_GroundedToolUsePromptTokensKeepTheGeneration(t *testing.T) {
	t.Parallel()

	// 50 prompt + 30 tool-use prompt + 20 candidates + 5 thoughts = 105.
	body := []byte(`{"candidates":[{"content":{"parts":[{"text":"grounded answer"}]},"finishReason":"STOP"}],` +
		`"usageMetadata":{"promptTokenCount":50,"toolUsePromptTokenCount":30,"candidatesTokenCount":20,` +
		`"thoughtsTokenCount":5,"totalTokenCount":105}}`)

	response, err := geminiapi.DecodeResponse(body)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v, want the generation preserved", err)
	}
	if response.Message == nil || len(response.Message.Blocks) != 1 {
		t.Fatalf("Message = %#v, want the answer block", response.Message)
	}
	if got := response.Message.Blocks[0].(*content.TextBlock).Text; got != "grounded answer" {
		t.Errorf("text = %q, want %q", got, "grounded answer")
	}
	// Tool-use prompt tokens are billable input the caller paid for, and the
	// neutral Usage has no separate bucket for them, so they belong in
	// InputTokens rather than being silently dropped: 50 - 0 cached + 30.
	want := &content.Usage{InputTokens: 80, OutputTokens: 25, ReasoningTokens: 5}
	assertIndependentUsage(t, response, want)
}

// TestDecodeResponse_UnmodelledTotalComponentKeepsTheGeneration pins the
// strictness decision itself. Google has repeatedly added members to
// UsageMetadata (serviceTier, cacheTokensDetails, toolUsePromptTokensDetails);
// the next token bucket it adds must degrade the accounting, never the answer.
// A reported total larger than the components this codec models is therefore
// tolerated.
func TestDecodeResponse_UnmodelledTotalComponentKeepsTheGeneration(t *testing.T) {
	t.Parallel()

	body := []byte(`{"candidates":[{"content":{"parts":[{"text":"answer"}]},"finishReason":"STOP"}],` +
		`"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":999}}`)

	response, err := geminiapi.DecodeResponse(body)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v, want the generation preserved", err)
	}
	if response.Message == nil || len(response.Message.Blocks) != 1 {
		t.Fatalf("Message = %#v, want the answer block", response.Message)
	}
	assertIndependentUsage(t, response, &content.Usage{InputTokens: 10, OutputTokens: 5})
}

// TestDecodeStream_GroundedUsageDoesNotSwallowTheAnswer is the streaming half of
// the same defect, and the worse one. Gemini repeats usageMetadata on EVERY SSE
// frame, so a usage error raised while normalizing it aborts the stream from the
// first frame — before a single character of the answer reaches the caller, and
// reported as a stream failure rather than as the completed generation it was.
// The test lives beside the non-streaming case because the normalization it
// exercises is decode.go's, reached through the stream.
func TestDecodeStream_GroundedUsageDoesNotSwallowTheAnswer(t *testing.T) {
	t.Parallel()

	body := "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"grounded \"}]}}]," +
		"\"usageMetadata\":{\"promptTokenCount\":50,\"toolUsePromptTokenCount\":30,\"candidatesTokenCount\":10,\"totalTokenCount\":90}}\n\n" +
		"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"answer\"}]},\"finishReason\":\"STOP\"}]," +
		"\"usageMetadata\":{\"promptTokenCount\":50,\"toolUsePromptTokenCount\":30,\"candidatesTokenCount\":20,\"totalTokenCount\":100}}\n\n"

	reader, err := (geminiapi.Codec{}).DecodeStream(&http.Response{Body: io.NopCloser(strings.NewReader(body))})
	if err != nil {
		t.Fatalf("DecodeStream() error = %v", err)
	}
	defer func() { _ = reader.Close() }()

	var text strings.Builder
	for {
		chunk, err := reader.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Next() error = %v, want the streamed answer preserved", err)
		}
		if textChunk, ok := chunk.(*content.TextChunk); ok {
			text.WriteString(textChunk.Text)
		}
	}
	if text.String() != "grounded answer" {
		t.Fatalf("streamed text = %q, want %q", text.String(), "grounded answer")
	}
	result, ok := reader.Result()
	if !ok {
		t.Fatal("Result() unavailable after a clean stream")
	}
	if result.Usage == nil || result.Usage.InputTokens != 80 || result.Usage.OutputTokens != 20 {
		t.Fatalf("result usage = %+v, want input 80 / output 20", result.Usage)
	}
}
