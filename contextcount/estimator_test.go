package contextcount

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
)

func TestEstimatorGoldenCounts(t *testing.T) {
	tests := []struct {
		name   string
		format inference.APIFormat
		want   content.TokenCount
	}{
		{name: "OpenAI", format: inference.APIFormatOpenAI, want: 124},
		{name: "Anthropic", format: inference.APIFormatAnthropic, want: 133},
		{name: "Gemini", format: inference.APIFormatGemini, want: 139},
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
			if got.Model != (inference.ModelKey{Provider: "test-provider", Model: "test-model"}) {
				t.Errorf("Model = %#v, want test-provider/test-model", got.Model)
			}
			if got.Quality != inference.CountQualityHeuristicEstimate {
				t.Errorf("Quality = %v, want heuristic estimate", got.Quality)
			}
		})
	}
}

func TestEstimatorCountsCompleteRequest(t *testing.T) {
	tests := []struct {
		name   string
		format inference.APIFormat
		mutate func(*inference.Request)
	}{
		{name: "OpenAI system", format: inference.APIFormatOpenAI, mutate: addSystem},
		{name: "OpenAI model", format: inference.APIFormatOpenAI, mutate: addModelName},
		{name: "OpenAI message", format: inference.APIFormatOpenAI, mutate: addMessage},
		{name: "OpenAI tool and schema", format: inference.APIFormatOpenAI, mutate: addTool},
		{name: "OpenAI image", format: inference.APIFormatOpenAI, mutate: addImage},
		{name: "OpenAI sampling", format: inference.APIFormatOpenAI, mutate: addSampling},
		{name: "Anthropic system", format: inference.APIFormatAnthropic, mutate: addSystem},
		{name: "Anthropic model", format: inference.APIFormatAnthropic, mutate: addModelName},
		{name: "Anthropic message", format: inference.APIFormatAnthropic, mutate: addMessage},
		{name: "Anthropic tool and schema", format: inference.APIFormatAnthropic, mutate: addTool},
		{name: "Anthropic image", format: inference.APIFormatAnthropic, mutate: addImage},
		{name: "Anthropic sampling", format: inference.APIFormatAnthropic, mutate: addSampling},
		{name: "Gemini system", format: inference.APIFormatGemini, mutate: addSystem},
		{name: "Gemini message", format: inference.APIFormatGemini, mutate: addMessage},
		{name: "Gemini tool and schema", format: inference.APIFormatGemini, mutate: addTool},
		{name: "Gemini image", format: inference.APIFormatGemini, mutate: addImage},
		{name: "Gemini sampling", format: inference.APIFormatGemini, mutate: addSampling},
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

func TestEstimatorIgnoresHistoricalUsage(t *testing.T) {
	tests := []struct {
		name   string
		format inference.APIFormat
	}{
		{name: "OpenAI", format: inference.APIFormatOpenAI},
		{name: "Anthropic", format: inference.APIFormatAnthropic},
		{name: "Gemini", format: inference.APIFormatGemini},
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
		wantModelField  inference.ModelKeyField
		wantUnsupported inference.APIFormat
		wantEncoding    inference.APIFormat
	}{
		{
			name:      "nil receiver",
			estimator: nil,
			req:       minimalRequest(inference.APIFormatOpenAI),
			wantState: EstimatorStateNilReceiver,
		},
		{
			name:           "missing provider identity",
			estimator:      NewEstimator(),
			req:            requestWithModel("", "model", inference.APIFormatOpenAI),
			wantModelField: inference.ModelKeyFieldProvider,
		},
		{
			name:           "missing model identity",
			estimator:      NewEstimator(),
			req:            requestWithModel("provider", "", inference.APIFormatOpenAI),
			wantModelField: inference.ModelKeyFieldModel,
		},
		{
			name:      "unsupported empty format",
			estimator: NewEstimator(),
			req:       requestWithModel("provider", "model", ""),
		},
		{
			name:            "unsupported custom format",
			estimator:       NewEstimator(),
			req:             requestWithModel("provider", "model", inference.APIFormat("custom")),
			wantUnsupported: inference.APIFormat("custom"),
		},
		{
			name:         "OpenAI encoding failure",
			estimator:    NewEstimator(),
			req:          requestWithInvalidToolSchema(inference.APIFormatOpenAI),
			wantEncoding: inference.APIFormatOpenAI,
		},
		{
			name:         "Anthropic unsupported document",
			estimator:    NewEstimator(),
			req:          requestWithDocument(inference.APIFormatAnthropic),
			wantEncoding: inference.APIFormatAnthropic,
		},
		{
			name:         "Gemini unsupported document",
			estimator:    NewEstimator(),
			req:          requestWithDocument(inference.APIFormatGemini),
			wantEncoding: inference.APIFormatGemini,
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
				var cause *inference.ModelKeyValidationError
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
	want := inference.CounterCapability{
		Transport:    inference.CounterTransportLocal,
		Retention:    inference.RetentionNone,
		TokenizerRev: EstimatorRevision,
		Quality:      inference.CountQualityHeuristicEstimate,
	}
	tests := []struct {
		name      string
		estimator *Estimator
		want      inference.CounterCapability
		wantValid bool
	}{
		{name: "constructed", estimator: NewEstimator(), want: want, wantValid: true},
		{name: "zero value", estimator: &Estimator{}, want: want, wantValid: true},
		{name: "nil receiver is invalid", estimator: nil, want: inference.CounterCapability{}, wantValid: false},
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
		format inference.APIFormat
	}{
		{name: "OpenAI", format: inference.APIFormatOpenAI},
		{name: "Anthropic", format: inference.APIFormatAnthropic},
		{name: "Gemini", format: inference.APIFormatGemini},
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

func minimalRequest(format inference.APIFormat) inference.Request {
	return requestWithModel("test-provider", "test-model", format)
}

func requestWithModel(provider inference.ProviderName, model string, format inference.APIFormat) inference.Request {
	return inference.Request{Model: inference.Model{Provider: provider, Name: model, APIFormat: format}}
}

func completeRequest(format inference.APIFormat, usage *content.Usage) inference.Request {
	temperature := 0.25
	topP := 0.9
	maxTokens := 64
	return inference.Request{
		Model: inference.Model{
			Provider:  "test-provider",
			APIFormat: format,
			Name:      "test-model",
			Caps:      inference.Capabilities{Tools: true, AcceptsImages: true, Thinking: true},
			Sampling: inference.Sampling{
				Temperature: &temperature,
				TopP:        &topP,
				MaxTokens:   &maxTokens,
				Stop:        []string{"END"},
				Effort:      inference.EffortLow,
			},
		},
		System: "system prompt",
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{
				&content.TextBlock{Text: "hello"},
			}}},
			&content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{
				&content.TextBlock{Text: "world"},
			}}, Usage: usage},
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
	req.Override = &inference.Sampling{
		Temperature: &temperature,
		TopP:        &topP,
		MaxTokens:   &maxTokens,
		Stop:        []string{"STOP", "HALT"},
	}
}

func requestWithInvalidToolSchema(format inference.APIFormat) inference.Request {
	req := minimalRequest(format)
	req.Tools = []inference.Tool{{Name: "lookup", Schema: json.RawMessage(`{"broken":`)}}
	return req
}

func requestWithDocument(format inference.APIFormat) inference.Request {
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
