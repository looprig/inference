package anthropicapi_test

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/anthropicapi"
	"github.com/looprig/inference/codec/conformance"
	model "github.com/looprig/inference/model"
)

// This file pins the per-model thinking dialect: WHICH of Anthropic's three
// `thinking` variants the encoder emits, and which it refuses to guess at.
//
// The defect it closes was measured against api.anthropic.com on 2026-08-13,
// from one run of the same encoder:
//
//	claude-haiku-4-5 : "adaptive thinking is not supported on this model"
//	claude-sonnet-5  : "\"thinking.type.enabled\" is not supported for this
//	                    model. Use \"thinking.type.adaptive\" and
//	                    \"output_config.effort\""
//
// Both are HTTP 400s. The encoder chose from a single boolean — Caps.Thinking
// set meant `{"type":"adaptive"}` plus `output_config.effort`, always — so a
// model whose only on-mode is the budget form could not be given reasoning at
// all through the shared codec.
//
// The schema cannot carry this rule and it was measured saying so: the gate
// ACCEPTS `{"type":"enabled","budget_tokens":2048}` and
// `{"type":"adaptive"}` for the same model id, and accepts `budget_tokens`
// alongside `output_config.effort`. It is a per-model server behaviour, so the
// constraint is carried by Caps.ThinkingDialect and by the assertions below,
// which name the exact variant each dialect must produce.

// dialectMessages is the one-user-turn conversation every case here encodes;
// the subject under test is the request's thinking members, not its content.
func dialectMessages() content.AgenticMessages {
	return content.AgenticMessages{userMsg(textBlock("hi"))}
}

// TestEncodeRequestThinkingDialectSelectsTheModelsWireShape is the RED test for
// the defect: a budget-dialect model must reach the wire as
// `{"type":"enabled","budget_tokens":N}` with NO output_config.effort, and an
// adaptive-dialect model as `{"type":"adaptive"}` WITH output_config.effort.
func TestEncodeRequestThinkingDialectSelectsTheModelsWireShape(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name       string
		modelName  string
		dialect    model.ThinkingDialect
		effort     model.Effort
		maxTokens  int
		wantType   string
		wantBudget int    // 0 when budget_tokens must be absent
		wantEffort string // "" when output_config must be absent
	}{
		{
			name: "adaptive dialect emits adaptive plus effort", modelName: "claude-sonnet-5",
			dialect: model.ThinkingDialectAdaptive, effort: model.EffortHigh, maxTokens: 8000,
			wantType: "adaptive", wantEffort: "high",
		},
		{
			name: "budget dialect emits enabled with budget_tokens and no effort", modelName: "claude-haiku-4-5",
			dialect: model.ThinkingDialectBudget, effort: model.EffortHigh, maxTokens: 8000,
			wantType: "enabled", wantBudget: 6000,
		},
		{
			name: "budget dialect maps minimal below low", modelName: "claude-haiku-4-5",
			dialect: model.ThinkingDialectBudget, effort: model.EffortMinimal, maxTokens: 20000,
			wantType: "enabled", wantBudget: 2000,
		},
		{
			name: "budget dialect scales the budget with effort", modelName: "claude-haiku-4-5",
			dialect: model.ThinkingDialectBudget, effort: model.EffortLow, maxTokens: 8000,
			wantType: "enabled", wantBudget: 2000,
		},
		{
			name: "budget dialect maps xhigh between high and max", modelName: "claude-haiku-4-5",
			dialect: model.ThinkingDialectBudget, effort: model.EffortXHigh, maxTokens: 8000,
			wantType: "enabled", wantBudget: 6800,
		},
		{
			name: "budget dialect floors the budget at the schema minimum", modelName: "claude-haiku-4-5",
			dialect: model.ThinkingDialectBudget, effort: model.EffortLow, maxTokens: 2048,
			wantType: "enabled", wantBudget: 1024,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			m := baseModel()
			m.Name = tc.modelName
			m.Caps.Thinking = true
			m.Caps.ThinkingDialect = tc.dialect
			maxTokens := tc.maxTokens
			m.Sampling = model.Sampling{Effort: tc.effort, MaxTokens: &maxTokens}

			data, err := anthropicapi.EncodeRequest(inference.Request{Model: m, Messages: dialectMessages()}, false)
			if err != nil {
				t.Fatalf("EncodeRequest() error = %v", err)
			}
			// The gate runs BEFORE the structural assertions: a body that is
			// not a legal CreateMessageParams proves nothing about its members.
			conformance.MustValidateRequest(t, "anthropic", kindCreateMessageRequest, data)

			body := decodeObj(t, data)
			th := decodeObj(t, body["thinking"])
			if got := asString(t, th["type"]); got != tc.wantType {
				t.Errorf("thinking.type = %q, want %q", got, tc.wantType)
			}
			if tc.wantBudget == 0 {
				if _, ok := th["budget_tokens"]; ok {
					t.Errorf("thinking.budget_tokens is present for the %q dialect", tc.dialect)
				}
			} else {
				var budget int
				if err := json.Unmarshal(th["budget_tokens"], &budget); err != nil {
					t.Fatalf("thinking.budget_tokens is not an integer: %v (raw %s)", err, th["budget_tokens"])
				}
				if budget != tc.wantBudget {
					t.Errorf("thinking.budget_tokens = %d, want %d", budget, tc.wantBudget)
				}
				if budget >= tc.maxTokens {
					t.Errorf("thinking.budget_tokens = %d, must be strictly below max_tokens %d", budget, tc.maxTokens)
				}
			}

			oc, hasOC := body["output_config"]
			if tc.wantEffort == "" {
				if hasOC {
					t.Errorf("output_config is present for the %q dialect: %s", tc.dialect, oc)
				}
				return
			}
			if !hasOC {
				t.Fatalf("output_config is absent for the %q dialect", tc.dialect)
			}
			if got := asString(t, decodeObj(t, oc)["effort"]); got != tc.wantEffort {
				t.Errorf("output_config.effort = %q, want %q", got, tc.wantEffort)
			}
		})
	}
}

// TestEncodeRequestRefusesAnUndeclaredThinkingDialect pins the fail-closed
// half. A model advertised as thinking-capable whose dialect the catalogue does
// not describe has no known-good spelling: one of the two is a 400 and nothing
// local says which. Guessing sends a request we cannot justify, so the encoder
// refuses with a diagnostic that names the model.
func TestEncodeRequestRefusesAnUndeclaredThinkingDialect(t *testing.T) {
	t.Parallel()

	m := baseModel()
	m.Name = "claude-unknown-9"
	m.Caps.Thinking = true // dialect deliberately left undeclared
	m.Sampling = model.Sampling{Effort: model.EffortHigh}

	_, err := anthropicapi.EncodeRequest(inference.Request{Model: m, Messages: dialectMessages()}, false)
	var typed *anthropicapi.UndeclaredThinkingDialectError
	if !errors.As(err, &typed) {
		t.Fatalf("EncodeRequest() error = %T %v, want *anthropicapi.UndeclaredThinkingDialectError", err, err)
	}
	if typed.Model != "claude-unknown-9" {
		t.Errorf("error Model = %q, want the model that could not be described", typed.Model)
	}
	if !strings.Contains(err.Error(), "claude-unknown-9") {
		t.Errorf("error message %q does not name the model", err.Error())
	}
}

// TestEncodeRequestRefusesABudgetThatCannotBeLegal transcribes the two
// constraints the budget form carries — ThinkingConfigEnabled.budget_tokens has
// minimum 1024, and Anthropic documents that it must be less than max_tokens —
// into encoder validation. Below 1025 max_tokens no value satisfies both, so
// the codec fails locally where the diagnostic can name the field instead of
// sending a request it can prove will 400.
func TestEncodeRequestRefusesABudgetThatCannotBeLegal(t *testing.T) {
	t.Parallel()

	for _, maxTokens := range []int{512, 1024} {
		m := baseModel()
		m.Name = "claude-haiku-4-5"
		m.Caps.Thinking = true
		m.Caps.ThinkingDialect = model.ThinkingDialectBudget
		cap := maxTokens
		m.Sampling = model.Sampling{Effort: model.EffortHigh, MaxTokens: &cap}

		_, err := anthropicapi.EncodeRequest(inference.Request{Model: m, Messages: dialectMessages()}, false)
		var typed *anthropicapi.ThinkingBudgetError
		if !errors.As(err, &typed) {
			t.Fatalf("max_tokens %d: EncodeRequest() error = %T %v, want *anthropicapi.ThinkingBudgetError", maxTokens, err, err)
		}
		if typed.MaxTokens != maxTokens || typed.Model != "claude-haiku-4-5" {
			t.Errorf("max_tokens %d: error = %+v, want it to name the model and the cap", maxTokens, typed)
		}
	}
}

// TestEncodeRequestIgnoresTheDialectWithoutEffort keeps the dialect from
// becoming a second switch that turns thinking on: EffortNone still emits no
// thinking member at all, for either dialect.
func TestEncodeRequestIgnoresTheDialectWithoutEffort(t *testing.T) {
	t.Parallel()

	for _, dialect := range []model.ThinkingDialect{model.ThinkingDialectAdaptive, model.ThinkingDialectBudget} {
		m := baseModel()
		m.Caps.Thinking = true
		m.Caps.ThinkingDialect = dialect
		m.Sampling = model.Sampling{Effort: model.EffortNone}

		data, err := anthropicapi.EncodeRequest(inference.Request{Model: m, Messages: dialectMessages()}, false)
		if err != nil {
			t.Fatalf("dialect %q: EncodeRequest() error = %v", dialect, err)
		}
		conformance.MustValidateRequest(t, "anthropic", kindCreateMessageRequest, data)
		if _, ok := decodeObj(t, data)["thinking"]; ok {
			t.Errorf("dialect %q: thinking is present with EffortNone", dialect)
		}
	}
}
