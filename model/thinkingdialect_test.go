package model_test

import (
	"errors"
	"testing"

	model "github.com/looprig/inference/model"
)

// TestThinkingDialectValid pins the allowlist. A dialect the vocabulary does
// not define must be invalid rather than pass through as an opaque string: a
// codec switching on it would otherwise fall into whatever its default arm is,
// which is the failure mode the type exists to remove.
func TestThinkingDialectValid(t *testing.T) {
	t.Parallel()

	valid := []model.ThinkingDialect{
		model.ThinkingDialectUnknown,
		model.ThinkingDialectAdaptive,
		model.ThinkingDialectBudget,
	}
	for _, d := range valid {
		if !d.Valid() {
			t.Errorf("ThinkingDialect(%q).Valid() = false, want true", d)
		}
	}
	for _, d := range []model.ThinkingDialect{"enabled", "Adaptive", "extended", "true"} {
		if d.Valid() {
			t.Errorf("ThinkingDialect(%q).Valid() = true, want false", d)
		}
	}
}

// TestWithThinkingDialectSetsItsPrerequisite mirrors
// WithStructuredOutputWithTools: an option that declares a refinement also sets
// the capability it refines, so a caller cannot produce a descriptor that says
// "budget dialect, but not thinking-capable".
func TestWithThinkingDialectSetsItsPrerequisite(t *testing.T) {
	t.Parallel()

	m := model.CustomModel("anthropic", model.APIFormatAnthropic, "https://api.anthropic.com/v1", "claude-haiku-4-5",
		model.WithThinkingDialect(model.ThinkingDialectBudget))
	if !m.Caps.Thinking {
		t.Error("Caps.Thinking = false, want true")
	}
	if m.Caps.ThinkingDialect != model.ThinkingDialectBudget {
		t.Errorf("Caps.ThinkingDialect = %q, want %q", m.Caps.ThinkingDialect, model.ThinkingDialectBudget)
	}

	// WithThinking alone leaves the dialect UNDECLARED. That is the honest
	// answer for a caller who knows the model reasons but not how to ask, and
	// it is what makes the codec's fail-closed path reachable.
	bare := model.CustomModel("anthropic", model.APIFormatAnthropic, "https://api.anthropic.com/v1", "claude-x",
		model.WithThinking())
	if bare.Caps.ThinkingDialect != model.ThinkingDialectUnknown {
		t.Errorf("WithThinking() set Caps.ThinkingDialect = %q, want it left undeclared", bare.Caps.ThinkingDialect)
	}
}

// TestValidateThinkingDialect pins the two structural rules, in the same shape
// as the existing StructuredOutputWithTools rule.
func TestValidateThinkingDialect(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		caps      model.Capabilities
		wantField string // "" means Validate must accept
	}{
		{name: "undeclared dialect on a thinking model", caps: model.Capabilities{Thinking: true}},
		{name: "undeclared dialect on a non-thinking model", caps: model.Capabilities{}},
		{name: "adaptive with thinking", caps: model.Capabilities{Thinking: true, ThinkingDialect: model.ThinkingDialectAdaptive}},
		{name: "budget with thinking", caps: model.Capabilities{Thinking: true, ThinkingDialect: model.ThinkingDialectBudget}},
		{
			name:      "dialect without thinking",
			caps:      model.Capabilities{ThinkingDialect: model.ThinkingDialectBudget},
			wantField: "Caps.ThinkingDialect",
		},
		{
			name:      "unknown dialect",
			caps:      model.Capabilities{Thinking: true, ThinkingDialect: "enabled"},
			wantField: "Caps.ThinkingDialect",
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			m := model.Model{
				Provider:  "anthropic",
				APIFormat: model.APIFormatAnthropic,
				Name:      "claude-test",
				Caps:      tc.caps,
			}
			err := m.Validate()
			if tc.wantField == "" {
				if err != nil {
					t.Fatalf("Validate() error = %v, want nil", err)
				}
				return
			}
			var verr *model.ValidationError
			if !errors.As(err, &verr) {
				t.Fatalf("Validate() error = %T %v, want *model.ValidationError", err, err)
			}
			if verr.Field != tc.wantField {
				t.Errorf("Validate() field = %q, want %q", verr.Field, tc.wantField)
			}
		})
	}
}
