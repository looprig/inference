package usagenorm_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference/internal/usagenorm"
	usage "github.com/looprig/inference/usage"
)

func TestAddTokenCounts(t *testing.T) {
	t.Parallel()

	max := content.TokenCount(^uint64(0))
	tests := []struct {
		name       string
		left       content.TokenCount
		right      content.TokenCount
		want       content.TokenCount
		wantReason usage.UsageNormalizationReason
	}{
		{name: "zero", want: 0},
		{name: "boundary", left: max - 1, right: 1, want: max},
		{name: "overflow", left: max, right: 1, wantReason: usage.UsageNormalizationReasonOverflow},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := usagenorm.AddTokenCounts(usagenorm.FieldOutputTokens, tt.left, tt.right)
			if tt.wantReason == "" {
				if err != nil {
					t.Fatalf("AddTokenCounts() error = %v", err)
				}
				if got != tt.want {
					t.Errorf("AddTokenCounts() = %d, want %d", got, tt.want)
				}
				return
			}
			assertNormalizationError(t, err, usage.UsageNormalizationFieldOutputTokens, tt.wantReason, 0, tt.left, tt.right, false)
			if !strings.Contains(err.Error(), "left=18446744073709551615") || !strings.Contains(err.Error(), "right=1") {
				t.Errorf("Error() = %q, want arithmetic operands", err)
			}
		})
	}
}

func TestSubtractTokenCounts(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		total      content.TokenCount
		first      content.TokenCount
		second     content.TokenCount
		want       content.TokenCount
		wantReason usage.UsageNormalizationReason
		wantRight  content.TokenCount
	}{
		{name: "zero", want: 0},
		{name: "components equal total", total: 5, first: 3, second: 2, want: 0},
		{name: "components exceed total", total: 4, first: 3, second: 2, wantReason: usage.UsageNormalizationReasonComponentsExceedTotal, wantRight: 5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := usagenorm.SubtractTokenCounts(usagenorm.FieldInputTokens, tt.total, tt.first, tt.second)
			if tt.wantReason == "" {
				if err != nil {
					t.Fatalf("SubtractTokenCounts() error = %v", err)
				}
				if got != tt.want {
					t.Errorf("SubtractTokenCounts() = %d, want %d", got, tt.want)
				}
				return
			}
			assertNormalizationError(t, err, usage.UsageNormalizationFieldInputTokens, tt.wantReason, 0, tt.total, tt.wantRight, false)
			if !strings.Contains(err.Error(), "left=4") || !strings.Contains(err.Error(), "right=5") {
				t.Errorf("Error() = %q, want arithmetic operands", err)
			}
		})
	}
}

func TestRequireEqual(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		reported   content.TokenCount
		calculated content.TokenCount
		wantErr    bool
	}{
		{name: "zero equal", reported: 0, calculated: 0},
		{name: "nonzero equal", reported: 7, calculated: 7},
		{name: "mismatch", reported: 3, calculated: 4, wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			err := usagenorm.RequireEqual(usagenorm.FieldTotalTokens, tt.reported, tt.calculated)
			if !tt.wantErr {
				if err != nil {
					t.Errorf("RequireEqual() error = %v", err)
				}
				return
			}
			assertNormalizationError(t, err, usage.UsageNormalizationFieldTotalTokens, usage.UsageNormalizationReasonTotalMismatch, 0, tt.reported, tt.calculated, false)
			if !strings.Contains(err.Error(), "left=3") || !strings.Contains(err.Error(), "right=4") {
				t.Errorf("Error() = %q, want mismatch operands", err)
			}
		})
	}
}

func TestNormalizeValidationError(t *testing.T) {
	t.Parallel()

	known := &content.UsageValidationError{Field: content.UsageFieldReasoningTokens, Reason: content.UsageValidationReasonReasoningExceedsOutput}
	unknown := &content.UsageValidationError{Field: content.UsageField("FutureTokens"), Reason: content.UsageValidationReason("future invariant")}
	unknownKnownField := &content.UsageValidationError{Field: content.UsageFieldReasoningTokens, Reason: content.UsageValidationReason("future reasoning invariant")}
	tests := []struct {
		name       string
		input      error
		wantField  usage.UsageNormalizationField
		wantReason usage.UsageNormalizationReason
	}{
		{name: "known reasoning invariant", input: known, wantField: usage.UsageNormalizationFieldReasoningTokens, wantReason: usage.UsageNormalizationReasonReasoningExceedsOutput},
		{name: "future invariant", input: unknown, wantField: usage.UsageNormalizationField("FutureTokens"), wantReason: usage.UsageNormalizationReasonDomainValidation},
		{name: "future invariant on known field", input: unknownKnownField, wantField: usage.UsageNormalizationFieldReasoningTokens, wantReason: usage.UsageNormalizationReasonDomainValidation},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			err := usagenorm.NormalizeValidationError(tt.input)
			assertNormalizationError(t, err, tt.wantField, tt.wantReason, 0, 0, 0, true)
			if !errors.Is(err, tt.input) {
				t.Errorf("NormalizeValidationError() does not preserve exact cause: %v", err)
			}
			if strings.Contains(err.Error(), "left=0") || strings.Contains(err.Error(), "right=0") {
				t.Errorf("NormalizeValidationError() invents unavailable operands: %q", err)
			}
		})
	}
}

func TestValidateUsage(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		usage      content.Usage
		wantReason usage.UsageNormalizationReason
		wantLeft   content.TokenCount
		wantRight  content.TokenCount
	}{
		{name: "zero", usage: content.Usage{}},
		{name: "reasoning subset", usage: content.Usage{OutputTokens: 3, ReasoningTokens: 3}},
		{name: "reasoning exceeds output", usage: content.Usage{OutputTokens: 2, ReasoningTokens: 3}, wantReason: usage.UsageNormalizationReasonReasoningExceedsOutput, wantLeft: 2, wantRight: 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			err := usagenorm.ValidateUsage(tt.usage)
			if tt.wantReason == "" {
				if err != nil {
					t.Errorf("ValidateUsage() error = %v", err)
				}
				return
			}
			assertNormalizationError(t, err, usage.UsageNormalizationFieldReasoningTokens, tt.wantReason, 0, tt.wantLeft, tt.wantRight, true)
			if !strings.Contains(err.Error(), "left=2") || !strings.Contains(err.Error(), "right=3") {
				t.Errorf("Error() = %q, want usage operands", err)
			}
		})
	}
}
