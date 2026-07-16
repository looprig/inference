package contextcount

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"testing"
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	model "github.com/looprig/inference/model"
)

func TestEstimatorGoldenCounts(t *testing.T) {
	tests := []struct {
		name   string
		format model.APIFormat
		want   content.TokenCount
	}{
		{name: "OpenAI", format: model.APIFormatOpenAI, want: 199},
		{name: "Anthropic", format: model.APIFormatAnthropic, want: 231},
		{name: "Gemini", format: model.APIFormatGemini, want: 213},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NewEstimator().CountContext(context.Background(), completeRequest(tt.format, nil))
			if err != nil {
				t.Fatalf("CountContext() error = %v", err)
			}
			if got.InputTokens != tt.want {
				t.Errorf("InputTokens = %d, want %d", got.InputTokens, tt.want)
			}
			if got.Model != (model.ModelKey{Provider: "test-provider", Model: "test-model"}) {
				t.Errorf("Model = %#v, want test-provider/test-model", got.Model)
			}
			if got.Quality != CountQualityHeuristicEstimate {
				t.Errorf("Quality = %v, want heuristic estimate", got.Quality)
			}
		})
	}
}

func TestEstimatorRejectsUnavailableContext(t *testing.T) {
	canceled, cancelCanceled := context.WithCancel(context.Background())
	cancelCanceled()
	expired, cancelExpired := context.WithDeadline(context.Background(), time.Unix(1, 0))
	defer cancelExpired()

	tests := []struct {
		name      string
		ctx       context.Context
		wantState EstimatorStateReason
		wantCause error
	}{
		{name: "nil context", ctx: nil, wantState: EstimatorStateNilContext},
		{name: "already canceled", ctx: canceled, wantCause: context.Canceled},
		{name: "expired deadline", ctx: expired, wantCause: context.DeadlineExceeded},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := requestWithModel("provider", "model", model.APIFormat("unsupported-before-encode"))
			_, err := NewEstimator().CountContext(tt.ctx, req)
			if tt.wantState != "" {
				var stateErr *EstimatorStateError
				if !errors.As(err, &stateErr) || stateErr.Reason != tt.wantState {
					t.Fatalf("error = %#v, want EstimatorStateError reason %q", err, tt.wantState)
				}
				return
			}
			var countErr *ContextCountError
			if !errors.As(err, &countErr) || !errors.Is(err, tt.wantCause) {
				t.Fatalf("error = %#v, want ContextCountError wrapping %v", err, tt.wantCause)
			}
			if countErr.Model != req.Model.Key() || countErr.Quality != CountQualityHeuristicEstimate {
				t.Errorf("ContextCountError = %#v, want model %#v and heuristic quality", countErr, req.Model.Key())
			}
		})
	}
}

func TestEstimatorCountsCompleteRequest(t *testing.T) {
	tests := []struct {
		name   string
		format model.APIFormat
		mutate func(*inference.Request)
	}{
		{name: "OpenAI system", format: model.APIFormatOpenAI, mutate: addSystem},
		{name: "OpenAI model", format: model.APIFormatOpenAI, mutate: addModelName},
		{name: "OpenAI message", format: model.APIFormatOpenAI, mutate: addMessage},
		{name: "OpenAI tool and schema", format: model.APIFormatOpenAI, mutate: addTool},
		{name: "OpenAI image", format: model.APIFormatOpenAI, mutate: addImage},
		{name: "OpenAI sampling", format: model.APIFormatOpenAI, mutate: addSampling},
		{name: "Anthropic system", format: model.APIFormatAnthropic, mutate: addSystem},
		{name: "Anthropic model", format: model.APIFormatAnthropic, mutate: addModelName},
		{name: "Anthropic message", format: model.APIFormatAnthropic, mutate: addMessage},
		{name: "Anthropic tool and schema", format: model.APIFormatAnthropic, mutate: addTool},
		{name: "Anthropic image", format: model.APIFormatAnthropic, mutate: addImage},
		{name: "Anthropic sampling", format: model.APIFormatAnthropic, mutate: addSampling},
		{name: "Gemini system", format: model.APIFormatGemini, mutate: addSystem},
		{name: "Gemini message", format: model.APIFormatGemini, mutate: addMessage},
		{name: "Gemini tool and schema", format: model.APIFormatGemini, mutate: addTool},
		{name: "Gemini image", format: model.APIFormatGemini, mutate: addImage},
		{name: "Gemini sampling", format: model.APIFormatGemini, mutate: addSampling},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			base := minimalRequest(tt.format)
			before, err := NewEstimator().CountContext(context.Background(), base)
			if err != nil {
				t.Fatalf("base CountContext() error = %v", err)
			}
			tt.mutate(&base)
			after, err := NewEstimator().CountContext(context.Background(), base)
			if err != nil {
				t.Fatalf("mutated CountContext() error = %v", err)
			}
			if after.InputTokens == before.InputTokens {
				t.Errorf("InputTokens unchanged at %d", after.InputTokens)
			}
		})
	}
}

func TestEstimatorCountsToolResultErrorMetadata(t *testing.T) {
	tests := []struct {
		name        string
		format      model.APIFormat
		wantChanged bool
	}{
		{name: "OpenAI intentionally omits IsError", format: model.APIFormatOpenAI, wantChanged: false},
		{name: "Anthropic includes IsError", format: model.APIFormatAnthropic, wantChanged: true},
		{name: "Gemini intentionally omits IsError", format: model.APIFormatGemini, wantChanged: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			before := toolFlowRequest(tt.format, testToolID, testToolName, testToolInput, testToolResult, false)
			after := toolFlowRequest(tt.format, testToolID, testToolName, testToolInput, testToolResult, true)
			assertRequestChange(t, before, after, tt.wantChanged)
		})
	}
}

func TestEstimatorCountsHistoricalThinking(t *testing.T) {
	tests := []struct {
		name        string
		format      model.APIFormat
		wantChanged bool
	}{
		{name: "OpenAI intentionally omits thinking", format: model.APIFormatOpenAI, wantChanged: false},
		{name: "Anthropic includes thinking and signature", format: model.APIFormatAnthropic, wantChanged: true},
		{name: "Gemini intentionally omits thinking", format: model.APIFormatGemini, wantChanged: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assertRequestChange(t, requestWithThinking(tt.format, false), requestWithThinking(tt.format, true), tt.wantChanged)
		})
	}
}

func TestEstimatorCountsEffortByDialect(t *testing.T) {
	tests := []struct {
		name        string
		before      inference.Request
		after       inference.Request
		wantChanged bool
	}{
		{name: "OpenAI effort ignores capability gate", before: requestWithEffort(model.APIFormatOpenAI, model.EffortNone, false), after: requestWithEffort(model.APIFormatOpenAI, model.EffortLow, false), wantChanged: true},
		{name: "Anthropic effort with thinking", before: requestWithEffort(model.APIFormatAnthropic, model.EffortNone, true), after: requestWithEffort(model.APIFormatAnthropic, model.EffortLow, true), wantChanged: true},
		{name: "Anthropic gate off restores no-effort encoding", before: requestWithEffort(model.APIFormatAnthropic, model.EffortNone, false), after: requestWithEffort(model.APIFormatAnthropic, model.EffortLow, false), wantChanged: false},
		{name: "Gemini effort with thinking", before: requestWithEffort(model.APIFormatGemini, model.EffortNone, true), after: requestWithEffort(model.APIFormatGemini, model.EffortLow, true), wantChanged: true},
		{name: "Gemini gate off restores no-effort encoding", before: requestWithEffort(model.APIFormatGemini, model.EffortNone, false), after: requestWithEffort(model.APIFormatGemini, model.EffortLow, false), wantChanged: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assertRequestChange(t, tt.before, tt.after, tt.wantChanged)
		})
	}
}

const (
	testToolID     string = "call-rich-123"
	testToolName   string = "lookup_rich_value"
	testToolInput  string = `{"query":"rich tool input","limit":7}`
	testToolResult string = "rich tool result content"
)

func assertRequestChange(t *testing.T, before, after inference.Request, wantChanged bool) {
	t.Helper()
	if before.Model.APIFormat != after.Model.APIFormat {
		t.Fatalf("test request format mismatch: before=%q after=%q", before.Model.APIFormat, after.Model.APIFormat)
	}
	bodyChanged, countChanged := requestChanges(t, before, after)
	if bodyChanged != wantChanged {
		t.Errorf("encoded body changed=%v, want %v", bodyChanged, wantChanged)
	}
	if countChanged != wantChanged {
		t.Errorf("estimated count changed=%v, want %v", countChanged, wantChanged)
	}
}

func requestChanges(t *testing.T, before, after inference.Request) (bool, bool) {
	t.Helper()
	beforeBody, err := encodeRequest(before)
	if err != nil {
		t.Fatalf("encode before request: %v", err)
	}
	afterBody, err := encodeRequest(after)
	if err != nil {
		t.Fatalf("encode after request: %v", err)
	}

	beforeCount, err := NewEstimator().CountContext(context.Background(), before)
	if err != nil {
		t.Fatalf("before CountContext() error = %v", err)
	}
	afterCount, err := NewEstimator().CountContext(context.Background(), after)
	if err != nil {
		t.Fatalf("after CountContext() error = %v", err)
	}
	return !bytes.Equal(beforeBody, afterBody), beforeCount.InputTokens != afterCount.InputTokens
}

func TestEstimatorIgnoresHistoricalUsage(t *testing.T) {
	tests := []struct {
		name   string
		format model.APIFormat
	}{
		{name: "OpenAI", format: model.APIFormatOpenAI},
		{name: "Anthropic", format: model.APIFormatAnthropic},
		{name: "Gemini", format: model.APIFormatGemini},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			without := completeRequest(tt.format, nil)
			with := completeRequest(tt.format, &content.Usage{
				InputTokens:         10_000,
				OutputTokens:        20_000,
				CacheReadTokens:     30_000,
				CacheCreationTokens: 40_000,
				ReasoningTokens:     20_000,
			})

			withoutCount, err := NewEstimator().CountContext(context.Background(), without)
			if err != nil {
				t.Fatalf("without usage CountContext() error = %v", err)
			}
			withCount, err := NewEstimator().CountContext(context.Background(), with)
			if err != nil {
				t.Fatalf("with usage CountContext() error = %v", err)
			}
			if withCount != withoutCount {
				t.Errorf("count with usage = %#v, want %#v", withCount, withoutCount)
			}
		})
	}
}

func TestEstimatorTypedFailures(t *testing.T) {
	tests := []struct {
		name            string
		estimator       *Estimator
		req             inference.Request
		wantState       EstimatorStateReason
		wantModelField  model.ModelKeyField
		wantUnsupported model.APIFormat
		wantEncoding    model.APIFormat
	}{
		{
			name:      "nil receiver",
			estimator: nil,
			req:       minimalRequest(model.APIFormatOpenAI),
			wantState: EstimatorStateNilReceiver,
		},
		{
			name:           "missing provider identity",
			estimator:      NewEstimator(),
			req:            requestWithModel("", "model", model.APIFormatOpenAI),
			wantModelField: model.ModelKeyFieldProvider,
		},
		{
			name:           "missing model identity",
			estimator:      NewEstimator(),
			req:            requestWithModel("provider", "", model.APIFormatOpenAI),
			wantModelField: model.ModelKeyFieldModel,
		},
		{
			name:      "unsupported empty format",
			estimator: NewEstimator(),
			req:       requestWithModel("provider", "model", ""),
		},
		{
			name:            "unsupported custom format",
			estimator:       NewEstimator(),
			req:             requestWithModel("provider", "model", model.APIFormat("custom")),
			wantUnsupported: model.APIFormat("custom"),
		},
		{
			name:         "OpenAI encoding failure",
			estimator:    NewEstimator(),
			req:          requestWithInvalidToolSchema(model.APIFormatOpenAI),
			wantEncoding: model.APIFormatOpenAI,
		},
		{
			name:         "Anthropic unsupported document",
			estimator:    NewEstimator(),
			req:          requestWithDocument(model.APIFormatAnthropic),
			wantEncoding: model.APIFormatAnthropic,
		},
		{
			name:         "Gemini unsupported document",
			estimator:    NewEstimator(),
			req:          requestWithDocument(model.APIFormatGemini),
			wantEncoding: model.APIFormatGemini,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.estimator.CountContext(context.Background(), tt.req)
			if err == nil {
				t.Fatal("CountContext() error = nil, want typed error")
			}

			switch {
			case tt.wantState != "":
				var got *EstimatorStateError
				if !errors.As(err, &got) || got.Reason != tt.wantState {
					t.Fatalf("error = %#v, want EstimatorStateError reason %q", err, tt.wantState)
				}
			case tt.wantModelField != "":
				var got *ModelIdentityError
				if !errors.As(err, &got) {
					t.Fatalf("error = %T, want *ModelIdentityError", err)
				}
				var cause *model.ModelKeyValidationError
				if !errors.As(err, &cause) || cause.Field != tt.wantModelField {
					t.Errorf("cause = %#v, want ModelKeyValidationError field %q", cause, tt.wantModelField)
				}
				if got.Model != tt.req.Model.Key() {
					t.Errorf("error Model = %#v, want %#v", got.Model, tt.req.Model.Key())
				}
			case tt.wantUnsupported != "":
				var got *UnsupportedAPIFormatError
				if !errors.As(err, &got) || got.APIFormat != tt.wantUnsupported {
					t.Fatalf("error = %#v, want unsupported format %q", err, tt.wantUnsupported)
				}
			case tt.req.Model.APIFormat == "":
				var got *UnsupportedAPIFormatError
				if !errors.As(err, &got) || got.APIFormat != "" {
					t.Fatalf("error = %#v, want unsupported empty format", err)
				}
			case tt.wantEncoding != "":
				var got *RequestEncodingError
				if !errors.As(err, &got) || got.APIFormat != tt.wantEncoding || got.Err == nil {
					t.Fatalf("error = %#v, want encoding error for %q with cause", err, tt.wantEncoding)
				}
				if errors.Unwrap(got) != got.Err {
					t.Errorf("Unwrap() = %v, want Err %v", errors.Unwrap(got), got.Err)
				}
			}
		})
	}
}

func TestEstimatorCapability(t *testing.T) {
	want := CounterCapability{
		Transport:    CounterTransportLocal,
		Retention:    RetentionNone,
		TokenizerRev: TokenizerRevision("bundled-openai-anthropic-gemini-request-bytes-div4-v1"),
		Quality:      CountQualityHeuristicEstimate,
	}
	tests := []struct {
		name      string
		estimator *Estimator
		want      CounterCapability
		wantValid bool
	}{
		{name: "constructed", estimator: NewEstimator(), want: want, wantValid: true},
		{name: "zero value", estimator: &Estimator{}, want: want, wantValid: true},
		{name: "nil receiver is invalid", estimator: nil, want: CounterCapability{}, wantValid: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.estimator.CounterCapability()
			if got != tt.want {
				t.Errorf("CounterCapability() = %#v, want %#v", got, tt.want)
			}
			if (got.Validate() == nil) != tt.wantValid {
				t.Errorf("Validate() success = %v, want %v", got.Validate() == nil, tt.wantValid)
			}
		})
	}
}

func TestEstimatorDeterminismAndInputIntegrity(t *testing.T) {
	tests := []struct {
		name   string
		format model.APIFormat
	}{
		{name: "OpenAI", format: model.APIFormatOpenAI},
		{name: "Anthropic", format: model.APIFormatAnthropic},
		{name: "Gemini", format: model.APIFormatGemini},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := completeRequest(tt.format, nil)
			before := completeRequest(tt.format, nil)
			first, err := NewEstimator().CountContext(context.Background(), req)
			if err != nil {
				t.Fatalf("first CountContext() error = %v", err)
			}
			second, err := NewEstimator().CountContext(context.Background(), req)
			if err != nil {
				t.Fatalf("second CountContext() error = %v", err)
			}
			if first != second {
				t.Errorf("second count = %#v, want %#v", second, first)
			}
			if !reflect.DeepEqual(req, before) {
				t.Errorf("request mutated:\n got  %#v\n want %#v", req, before)
			}
		})
	}
}

func TestEstimatedTokensForBytes(t *testing.T) {
	tests := []struct {
		name string
		size uint64
		want content.TokenCount
	}{
		{name: "empty", size: 0, want: 0},
		{name: "one byte", size: 1, want: 1},
		{name: "exact word", size: 4, want: 1},
		{name: "remainder", size: 5, want: 2},
		{name: "maximum byte count", size: math.MaxUint64, want: content.TokenCount(math.MaxUint64/4 + 1)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := estimatedTokensForBytes(tt.size); got != tt.want {
				t.Errorf("estimatedTokensForBytes(%d) = %d, want %d", tt.size, got, tt.want)
			}
		})
	}
}

func minimalRequest(format model.APIFormat) inference.Request {
	return requestWithModel("test-provider", "test-model", format)
}

func requestWithModel(provider model.ProviderName, name string, format model.APIFormat) inference.Request {
	return inference.Request{Model: model.Model{Provider: provider, Name: name, APIFormat: format}}
}

func toolFlowRequest(format model.APIFormat, id, name, input, result string, isError bool) inference.Request {
	req := minimalRequest(format)
	req.Messages = content.AgenticMessages{
		&content.AIMessage{Message: content.Message{
			Role: content.RoleAssistant,
			Blocks: []content.Block{&content.ToolUseBlock{
				ID:    id,
				Name:  name,
				Input: json.RawMessage(input),
			}},
		}},
		&content.ToolResultMessage{
			Message: content.Message{
				Role:   content.RoleTool,
				Blocks: []content.Block{&content.TextBlock{Text: result}},
			},
			ToolUseID: id,
			IsError:   isError,
		},
	}
	return req
}

func requestWithThinking(format model.APIFormat, include bool) inference.Request {
	req := minimalRequest(format)
	blocks := []content.Block{&content.TextBlock{Text: "assistant answer"}}
	if include {
		blocks = append(blocks, &content.ThinkingBlock{
			Thinking:  "a deliberately long historical reasoning trace",
			Signature: "a-deliberately-long-provider-signature",
		})
	}
	req.Messages = content.AgenticMessages{&content.AIMessage{Message: content.Message{
		Role:   content.RoleAssistant,
		Blocks: blocks,
	}}}
	return req
}

func requestWithEffort(format model.APIFormat, effort model.Effort, thinkingCap bool) inference.Request {
	req := minimalRequest(format)
	req.Model.Caps.Thinking = thinkingCap
	req.Override = &model.Sampling{Effort: effort}
	return req
}

func completeRequest(format model.APIFormat, usage *content.Usage) inference.Request {
	temperature := 0.25
	topP := 0.9
	maxTokens := 64
	return inference.Request{
		Model: model.Model{
			Provider:  "test-provider",
			APIFormat: format,
			Name:      "test-model",
			Caps:      model.Capabilities{Tools: true, AcceptsImages: true, Thinking: true},
			Sampling: model.Sampling{
				Temperature: &temperature,
				TopP:        &topP,
				MaxTokens:   &maxTokens,
				Stop:        []string{"END"},
				Effort:      model.EffortLow,
			},
		},
		System: "system prompt",
		Messages: content.AgenticMessages{
			&content.SystemMessage{Message: content.Message{Role: content.RoleSystem, Blocks: []content.Block{
				&content.TextBlock{Text: "in-thread system instruction"},
			}}},
			&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{
				&content.TextBlock{Text: "hello"},
			}}},
			&content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{
				&content.TextBlock{Text: "world"},
				&content.ThinkingBlock{Thinking: "historical reasoning", Signature: "provider-signature"},
				&content.ToolUseBlock{ID: testToolID, Name: testToolName, Input: json.RawMessage(testToolInput)},
			}}, Usage: usage},
			&content.ToolResultMessage{
				Message:   content.Message{Role: content.RoleTool, Blocks: []content.Block{&content.TextBlock{Text: testToolResult}}},
				ToolUseID: testToolID,
				IsError:   true,
			},
			&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{
				&content.ImageBlock{MediaType: content.MediaTypeImagePNG, Source: content.ImageSource{Data: []byte{1, 2, 3}}},
			}}},
		},
		Tools: []inference.Tool{{
			Name:        "lookup",
			Description: "look up a value",
			Schema:      json.RawMessage(`{"type":"object","properties":{"x":{"type":"string"}}}`),
		}},
	}
}

func addSystem(req *inference.Request) { req.System = "a deliberately long system prompt" }

func addModelName(req *inference.Request) { req.Model.Name = "a-deliberately-long-model-name" }

func addMessage(req *inference.Request) {
	req.Messages = content.AgenticMessages{&content.UserMessage{Message: content.Message{
		Role:   content.RoleUser,
		Blocks: []content.Block{&content.TextBlock{Text: "a deliberately long user message"}},
	}}}
}

func addTool(req *inference.Request) {
	req.Tools = []inference.Tool{{
		Name:        "lookup",
		Description: "look up a deliberately long value",
		Schema:      json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}}}`),
	}}
}

func addImage(req *inference.Request) {
	req.Messages = content.AgenticMessages{&content.UserMessage{Message: content.Message{
		Role: content.RoleUser,
		Blocks: []content.Block{&content.ImageBlock{
			MediaType: content.MediaTypeImagePNG,
			Source:    content.ImageSource{Data: []byte("a deliberately nonempty image payload")},
		}},
	}}}
}

func addSampling(req *inference.Request) {
	temperature := 0.125
	topP := 0.875
	maxTokens := 321
	req.Override = &model.Sampling{
		Temperature: &temperature,
		TopP:        &topP,
		MaxTokens:   &maxTokens,
		Stop:        []string{"STOP", "HALT"},
	}
}

func requestWithInvalidToolSchema(format model.APIFormat) inference.Request {
	req := minimalRequest(format)
	req.Tools = []inference.Tool{{Name: "lookup", Schema: json.RawMessage(`{"broken":`)}}
	return req
}

func requestWithDocument(format model.APIFormat) inference.Request {
	req := minimalRequest(format)
	req.Messages = content.AgenticMessages{&content.UserMessage{Message: content.Message{
		Role: content.RoleUser,
		Blocks: []content.Block{&content.DocumentBlock{
			MediaType: content.MediaTypeDocumentText,
			Name:      "notes.txt",
			Text:      "document contents",
		}},
	}}}
	return req
}
