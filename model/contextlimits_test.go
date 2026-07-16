package model_test

import (
	"errors"
	"testing"

	"github.com/looprig/core/content"
	model "github.com/looprig/inference/model"
)

func TestContextLimits_Validate(t *testing.T) {
	t.Parallel()
	maximum := content.TokenCount(^uint64(0))
	tests := []struct {
		name       string
		limits     model.ContextLimits
		wantField  model.ContextLimitField
		wantReason model.ContextLimitValidationReason
		wantErr    bool
	}{
		{name: "all limits unknown"},
		{
			name:   "known shared window",
			limits: model.ContextLimits{WindowTokens: 200_000},
		},
		{
			name:   "independent input and output caps with unknown window",
			limits: model.ContextLimits{MaxInputTokens: 1_000_000, MaxOutputTokens: 64_000},
		},
		{
			name:   "caps equal shared window boundary",
			limits: model.ContextLimits{WindowTokens: 100, MaxInputTokens: 100, MaxOutputTokens: 100},
		},
		{
			name:   "independent caps need not sum below shared window",
			limits: model.ContextLimits{WindowTokens: 100, MaxInputTokens: 80, MaxOutputTokens: 30},
		},
		{
			name:   "maximum values valid when equal",
			limits: model.ContextLimits{WindowTokens: maximum, MaxInputTokens: maximum, MaxOutputTokens: maximum},
		},
		{
			name:       "input cap exceeds shared window",
			limits:     model.ContextLimits{WindowTokens: 100, MaxInputTokens: 101},
			wantField:  model.ContextLimitFieldMaxInputTokens,
			wantReason: model.ContextLimitValidationReasonExceedsWindow,
			wantErr:    true,
		},
		{
			name:       "output cap exceeds shared window",
			limits:     model.ContextLimits{WindowTokens: 100, MaxOutputTokens: 101},
			wantField:  model.ContextLimitFieldMaxOutputTokens,
			wantReason: model.ContextLimitValidationReasonExceedsWindow,
			wantErr:    true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			err := tt.limits.Validate()
			if (err != nil) != tt.wantErr {
				t.Fatalf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
			if !tt.wantErr {
				return
			}
			var validationErr *model.ContextLimitsValidationError
			if !errors.As(err, &validationErr) {
				t.Fatalf("Validate() error = %T, want *model.ContextLimitsValidationError", err)
			}
			if validationErr.Field != tt.wantField || validationErr.Reason != tt.wantReason {
				t.Errorf("Validate() error = %+v, want field %q reason %q", validationErr, tt.wantField, tt.wantReason)
			}
			if validationErr.Value != 101 || validationErr.WindowTokens != 100 {
				t.Errorf("Validate() values = value %d window %d, want 101 and 100", validationErr.Value, validationErr.WindowTokens)
			}
		})
	}
}
