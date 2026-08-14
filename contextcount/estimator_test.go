package contextcount

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/anthropicapi"
	"github.com/looprig/inference/codec/bedrockconverse"
	"github.com/looprig/inference/codec/geminiapi"
	"github.com/looprig/inference/codec/openaiapi"
	"github.com/looprig/inference/codec/openairesponses"
	model "github.com/looprig/inference/model"
)

func TestEstimatorGoldenCounts(t *testing.T) {
	tests := []struct {
		name   string
		format model.APIFormat
		want   content.TokenCount
	}{
		{name: "OpenAI", format: model.APIFormatOpenAI, want: 201},
		// 222, up from 219: the Responses encoder now emits FunctionTool's
		// required `strict` member, which it previously omitted. The golden
		// measures the real encoded body, so a correct encoder change moves it.
		{name: "OpenAIResponses", format: model.APIFormatOpenAIResponses, want: 222},
		// 224, down from 231: the Anthropic encoder projects adjacent neutral
		// user-role turns into the single user message its wire conversation
		// model requires. The tool-result turn and following image turn in this
		// fixture now share one message envelope.
		{name: "Anthropic", format: model.APIFormatAnthropic, want: 224},
		{name: "Gemini", format: model.APIFormatGemini, want: 213},
		{name: "BedrockConverse", format: model.APIFormatBedrockConverse, want: 213},
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

// TestEstimatorProjectsDialectRequestBody pins the invariant that counting
// measures exactly the body inference would send in the model's dialect: the
// bundled encoder's output, byte for byte.
func TestEstimatorProjectsDialectRequestBody(t *testing.T) {
	tests := []struct {
		name   string
		format model.APIFormat
		encode func(inference.Request) ([]byte, error)
	}{
		{name: "OpenAI", format: model.APIFormatOpenAI, encode: func(req inference.Request) ([]byte, error) {
			return openaiapi.EncodeRequest(req, false)
		}},
		{name: "OpenAIResponses", format: model.APIFormatOpenAIResponses, encode: func(req inference.Request) ([]byte, error) {
			return openairesponses.EncodeRequest(req, false)
		}},
		{name: "Anthropic", format: model.APIFormatAnthropic, encode: func(req inference.Request) ([]byte, error) {
			return anthropicapi.EncodeRequest(req, false)
		}},
		{name: "Gemini", format: model.APIFormatGemini, encode: geminiapi.EncodeRequest},
		{name: "BedrockConverse", format: model.APIFormatBedrockConverse, encode: bedrockconverse.EncodeRequest},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := completeRequest(tt.format, nil)
			want, err := tt.encode(req)
			if err != nil {
				t.Fatalf("dialect encode error = %v", err)
			}
			got, err := encodeRequest(req)
			if err != nil {
				t.Fatalf("encodeRequest() error = %v", err)
			}
			if !bytes.Equal(got, want) {
				t.Errorf("encoded body =\n%s\nwant\n%s", got, want)
			}
			count, err := NewEstimator().CountContext(context.Background(), req)
			if err != nil {
				t.Fatalf("CountContext() error = %v", err)
			}
			if wantTokens := estimatedTokensForBytes(uint64(len(want))); count.InputTokens != wantTokens {
				t.Errorf("InputTokens = %d, want %d", count.InputTokens, wantTokens)
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
		{name: "OpenAIResponses system", format: model.APIFormatOpenAIResponses, mutate: addSystem},
		{name: "OpenAIResponses model", format: model.APIFormatOpenAIResponses, mutate: addModelName},
		{name: "OpenAIResponses message", format: model.APIFormatOpenAIResponses, mutate: addMessage},
		{name: "OpenAIResponses tool and schema", format: model.APIFormatOpenAIResponses, mutate: addTool},
		{name: "OpenAIResponses image", format: model.APIFormatOpenAIResponses, mutate: addImage},
		{name: "OpenAIResponses sampling", format: model.APIFormatOpenAIResponses, mutate: addSampling},
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
		{name: "BedrockConverse system", format: model.APIFormatBedrockConverse, mutate: addSystem},
		{name: "BedrockConverse message", format: model.APIFormatBedrockConverse, mutate: addMessage},
		{name: "BedrockConverse tool and schema", format: model.APIFormatBedrockConverse, mutate: addTool},
		{name: "BedrockConverse image", format: model.APIFormatBedrockConverse, mutate: addImage},
		{name: "BedrockConverse sampling", format: model.APIFormatBedrockConverse, mutate: addSampling},
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
		{name: "OpenAIResponses intentionally omits IsError", format: model.APIFormatOpenAIResponses, wantChanged: false},
		{name: "Anthropic includes IsError", format: model.APIFormatAnthropic, wantChanged: true},
		{name: "Gemini intentionally omits IsError", format: model.APIFormatGemini, wantChanged: false},
		{name: "BedrockConverse includes IsError as a status", format: model.APIFormatBedrockConverse, wantChanged: true},
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
		{name: "OpenAIResponses includes thinking as a reasoning item", format: model.APIFormatOpenAIResponses, wantChanged: true},
		{name: "Anthropic includes thinking and signature", format: model.APIFormatAnthropic, wantChanged: true},
		{name: "Gemini intentionally omits thinking", format: model.APIFormatGemini, wantChanged: false},
		{name: "BedrockConverse includes thinking as reasoning content", format: model.APIFormatBedrockConverse, wantChanged: true},
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
		{name: "OpenAIResponses effort with thinking", before: requestWithEffort(model.APIFormatOpenAIResponses, model.EffortNone, true), after: requestWithEffort(model.APIFormatOpenAIResponses, model.EffortLow, true), wantChanged: true},
		{name: "OpenAIResponses gate off restores no-effort encoding", before: requestWithEffort(model.APIFormatOpenAIResponses, model.EffortNone, false), after: requestWithEffort(model.APIFormatOpenAIResponses, model.EffortLow, false), wantChanged: false},
		{name: "Anthropic effort with thinking", before: requestWithEffort(model.APIFormatAnthropic, model.EffortNone, true), after: requestWithEffort(model.APIFormatAnthropic, model.EffortLow, true), wantChanged: true},
		{name: "Anthropic gate off restores no-effort encoding", before: requestWithEffort(model.APIFormatAnthropic, model.EffortNone, false), after: requestWithEffort(model.APIFormatAnthropic, model.EffortLow, false), wantChanged: false},
		{name: "Gemini effort with thinking", before: requestWithEffort(model.APIFormatGemini, model.EffortNone, true), after: requestWithEffort(model.APIFormatGemini, model.EffortLow, true), wantChanged: true},
		{name: "Gemini gate off restores no-effort encoding", before: requestWithEffort(model.APIFormatGemini, model.EffortNone, false), after: requestWithEffort(model.APIFormatGemini, model.EffortLow, false), wantChanged: false},
		{name: "BedrockConverse intentionally omits effort", before: requestWithEffort(model.APIFormatBedrockConverse, model.EffortNone, true), after: requestWithEffort(model.APIFormatBedrockConverse, model.EffortLow, true), wantChanged: false},
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
		{name: "OpenAIResponses", format: model.APIFormatOpenAIResponses},
		{name: "Anthropic", format: model.APIFormatAnthropic},
		{name: "Gemini", format: model.APIFormatGemini},
		{name: "BedrockConverse", format: model.APIFormatBedrockConverse},
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
			name:         "OpenAIResponses encoding failure",
			estimator:    NewEstimator(),
			req:          requestWithInvalidToolSchema(model.APIFormatOpenAIResponses),
			wantEncoding: model.APIFormatOpenAIResponses,
		},
		{
			// Responses models documents (input_file) but not audio: its input
			// content union is input_text|input_image|input_file, and the
			// spec's InputAudio object is reachable only from the Evals API.
			// Chat Completions, which does model audio, is deliberately not
			// listed here.
			name:         "OpenAIResponses unsupported audio",
			estimator:    NewEstimator(),
			req:          requestWithAudio(model.APIFormatOpenAIResponses),
			wantEncoding: model.APIFormatOpenAIResponses,
		},
		{
			// Anthropic models RequestDocumentBlock, so a text/plain document
			// now encodes. What it has no source member for is a non-PDF
			// BINARY document: Base64PDFSource.media_type is const
			// "application/pdf".
			name:         "Anthropic unsupported document",
			estimator:    NewEstimator(),
			req:          requestWithBinaryDocument(model.APIFormatAnthropic),
			wantEncoding: model.APIFormatAnthropic,
		},
		{
			// Gemini models documents and audio through Part's inlineData
			// (Blob) and text members, so a PDF, an audio clip and extracted
			// document text all encode now. What remains unencodable is a
			// document whose BYTES carry a media type absent from Blob's
			// documented list — .docx and .xlsx among them.
			name:         "Gemini unsupported document",
			estimator:    NewEstimator(),
			req:          requestWithBinaryDocument(model.APIFormatGemini),
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
		TokenizerRev: TokenizerRevision("bundled-openai-responses-anthropic-gemini-bedrock-request-bytes-div4-v3"),
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
		{name: "OpenAIResponses", format: model.APIFormatOpenAIResponses},
		{name: "Anthropic", format: model.APIFormatAnthropic},
		{name: "Gemini", format: model.APIFormatGemini},
		{name: "BedrockConverse", format: model.APIFormatBedrockConverse},
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
		// Carry Responses-issued provider state so the reasoning item is
		// replayable. Responses drops a reasoning item that has no issued id,
		// because ReasoningItem.required includes "id" and inventing one is a
		// 400; without state this fixture would encode identically with and
		// without thinking and the test would assert nothing. Dialects that do
		// not own this format ignore the state via ReplayableAs and continue to
		// project Thinking/Signature as before.
		blocks = append(blocks, content.NewSignedThinkingBlock(
			"a deliberately long historical reasoning trace",
			"a-deliberately-long-provider-signature",
			signatureFormatOf(format),
			json.RawMessage(`{"id":"rs_historical","type":"reasoning","summary":[],"encrypted_content":"a-deliberately-long-encrypted-reasoning-blob"}`),
			"openai-responses",
		))
	}
	req.Messages = content.AgenticMessages{&content.AIMessage{Message: content.Message{
		Role:   content.RoleAssistant,
		Blocks: blocks,
	}}}
	return req
}

// signatureFormatOf labels a fixture's reasoning signature with the dialect
// under test. A signature is verified by the endpoint that minted it, so these
// dialect-parameterised fixtures cannot share one label: the Anthropic and
// Bedrock encoders refuse a signature that is not theirs, which is the point of
// the label. The dialects with no signature wire field at all get none, and
// their encoders never read the field.
func signatureFormatOf(format model.APIFormat) string {
	switch format {
	case model.APIFormatAnthropic:
		return "anthropic"
	case model.APIFormatBedrockConverse:
		return "bedrock-converse"
	default:
		return ""
	}
}

func requestWithEffort(format model.APIFormat, effort model.Effort, thinkingCap bool) inference.Request {
	req := minimalRequest(format)
	req.Model.Caps.Thinking = thinkingCap
	// A thinking-capable model must also say WHICH thinking request shape it
	// takes: the Anthropic encoder refuses to guess between the two spellings
	// its API serves concurrently (UndeclaredThinkingDialectError).
	req.Model.Caps.ThinkingDialect = model.ThinkingDialectAdaptive
	req.Override = &model.Sampling{Effort: effort}
	return req
}

func completeRequest(format model.APIFormat, usage *content.Usage) inference.Request {
	req := dialectNeutralCompleteRequest(format, usage)
	if format == model.APIFormatBedrockConverse {
		// Converse rejects a committed user turn after tool results, so the
		// image turn precedes the tool exchange. The content is otherwise
		// identical to the other dialects' complete request.
		m := req.Messages
		req.Messages = content.AgenticMessages{m[0], m[1], m[4], m[2], m[3]}
	}
	return req
}

func dialectNeutralCompleteRequest(format model.APIFormat, usage *content.Usage) inference.Request {
	temperature := 0.25
	topP := 0.9
	maxTokens := 64
	return inference.Request{
		Model: model.Model{
			Provider:  "test-provider",
			APIFormat: format,
			Name:      "test-model",
			Caps: model.Capabilities{
				Tools: true, AcceptsImages: true,
				Thinking: true, ThinkingDialect: model.ThinkingDialectAdaptive,
			},
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
				content.NewSignedThinkingBlock("historical reasoning", "provider-signature", signatureFormatOf(format), nil, ""),
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
	req.Model.Caps.AcceptsImages = true
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

// requestWithBinaryDocument carries a document whose payload is binary and
// whose media type is not application/pdf — the one document shape the
// Anthropic dialect cannot source, because both of its payload-carrying source
// members declare their media_type as a const.
func requestWithBinaryDocument(format model.APIFormat) inference.Request {
	req := minimalRequest(format)
	req.Messages = content.AgenticMessages{&content.UserMessage{Message: content.Message{
		Role: content.RoleUser,
		Blocks: []content.Block{&content.DocumentBlock{
			MediaType: content.MediaTypeDocumentDOCX,
			Name:      "spec.docx",
			Data:      []byte{0x50, 0x4b, 0x03, 0x04},
		}},
	}}}
	return req
}

func requestWithAudio(format model.APIFormat) inference.Request {
	req := minimalRequest(format)
	req.Messages = content.AgenticMessages{&content.UserMessage{Message: content.Message{
		Role: content.RoleUser,
		Blocks: []content.Block{&content.AudioBlock{
			MediaType: content.MediaTypeAudioWAV,
			Data:      []byte("RIFF"),
		}},
	}}}
	return req
}

// TestEstimatorCountsTextDocumentContents pins that a text document's contents
// reach the estimate. The sibling binary-document case is already covered by
// TestEstimatorRejects..., which asserts a refusal; this is the accepted half,
// and it was the one left unexercised — requestWithDocument sat unused until
// staticcheck reported it, which is exactly the shape of an untested modality.
//
// The assertion is that the count RISES, not that it merely changes: an
// estimator that dropped the document would leave a request whose only block is
// the document indistinguishable from one carrying an empty message.
func TestEstimatorCountsTextDocumentContents(t *testing.T) {
	formats := []model.APIFormat{
		model.APIFormatOpenAI,
		model.APIFormatOpenAIResponses,
		model.APIFormatAnthropic,
		model.APIFormatGemini,
		model.APIFormatBedrockConverse,
	}

	for _, format := range formats {
		t.Run(string(format), func(t *testing.T) {
			base, err := NewEstimator().CountContext(context.Background(), minimalRequest(format))
			if err != nil {
				t.Fatalf("base CountContext() error = %v", err)
			}
			// "notes", not "notes.txt": a period is legal in every other
			// dialect and illegal in Bedrock's, which is the subject of
			// TestEstimatorRejectsADottedDocumentNameOnBedrockOnly below.
			withDoc, err := NewEstimator().CountContext(context.Background(), requestWithDocument(format, "notes"))
			if err != nil {
				t.Fatalf("document CountContext() error = %v", err)
			}
			if withDoc.InputTokens <= base.InputTokens {
				t.Errorf("InputTokens = %d with a text document, want more than the %d without one; the document's contents are not reaching the encoded request",
					withDoc.InputTokens, base.InputTokens)
			}
		})
	}
}

// TestEstimatorRejectsADottedDocumentNameOnBedrockOnly pins a cross-provider
// divergence the conformance gate cannot see. AWS's Smithy model constrains
// DocumentBlock.name only by length (minLength 1, maxLength 200) and declares
// no pattern, so the schema accepts "notes.txt"; the Converse documentation
// separately restricts the name to alphanumerics, hyphens, parentheses, square
// brackets and non-consecutive whitespace. bedrockconverse therefore carries
// that rule as an encoder allowlist, which is where a gate-blind constraint
// belongs — and this test is what keeps it honest.
//
// The same name is legal for Anthropic, so this is a genuine dialect
// difference and not a shared restriction: a document that counts fine for one
// model must be renamed before it can be counted for the other.
func TestEstimatorRejectsADottedDocumentNameOnBedrockOnly(t *testing.T) {
	_, err := NewEstimator().CountContext(context.Background(), requestWithDocument(model.APIFormatBedrockConverse, "notes.txt"))
	if err == nil {
		t.Fatal("bedrock-converse accepted a document name containing a period; the Converse name allowlist is no longer enforced")
	}
	// Assert the REASON, not merely that something failed. Converse has more
	// than one document rule, and an earlier draft of this test passed against
	// the "requires a text block" error instead — proving nothing about names.
	var unsupported *bedrockconverse.UnsupportedBlockError
	if !errors.As(err, &unsupported) {
		t.Fatalf("bedrock-converse failed with %v, which is not the block rejection this test is about", err)
	}
	if !strings.Contains(unsupported.Reason, "name") {
		t.Fatalf("bedrock-converse rejected the document for %q, not for its name; this case no longer tests the name allowlist", unsupported.Reason)
	}

	if _, err := NewEstimator().CountContext(context.Background(), requestWithDocument(model.APIFormatAnthropic, "notes.txt")); err != nil {
		t.Fatalf("anthropic rejected a document name containing a period, so the restriction is not Bedrock-specific after all: %v", err)
	}
}

func requestWithDocument(format model.APIFormat, name string) inference.Request {
	req := minimalRequest(format)
	req.Messages = content.AgenticMessages{&content.UserMessage{Message: content.Message{
		Role: content.RoleUser,
		// The anchor text block is required, not decorative: Converse rejects a
		// message whose only content is a document ("a document requires a text
		// block in the same message"). Carrying it in every dialect keeps the
		// five formats comparable, and keeps the name test below failing for the
		// reason it claims to test.
		Blocks: []content.Block{
			&content.TextBlock{Text: "anchor"},
			&content.DocumentBlock{
				MediaType: content.MediaTypeDocumentText,
				Name:      name,
				Text:      "document contents",
			},
		},
	}}}
	return req
}
