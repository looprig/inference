package inference_test

import (
	"errors"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
)

func TestUsageAlias(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		usage inference.Usage
		want  content.Usage
	}{
		{name: "zero", usage: inference.Usage{}, want: content.Usage{}},
		{
			name: "all fields",
			usage: inference.Usage{
				InputTokens:         1,
				OutputTokens:        2,
				CacheReadTokens:     3,
				CacheCreationTokens: 4,
				ReasoningTokens:     1,
			},
			want: content.Usage{
				InputTokens:         1,
				OutputTokens:        2,
				CacheReadTokens:     3,
				CacheCreationTokens: 4,
				ReasoningTokens:     1,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			var got content.Usage = tt.usage
			if got != tt.want {
				t.Errorf("content.Usage = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestNormalizeTokenCount(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		field      inference.UsageNormalizationField
		value      int
		want       content.TokenCount
		wantReason inference.UsageNormalizationReason
	}{
		{name: "zero", field: inference.UsageNormalizationFieldInputTokens, value: 0, want: 0},
		{name: "positive", field: inference.UsageNormalizationFieldOutputTokens, value: 7, want: 7},
		{name: "negative", field: inference.UsageNormalizationFieldCacheReadTokens, value: -1, wantReason: inference.UsageNormalizationReasonNegative},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := inference.NormalizeTokenCount(tt.field, tt.value)
			if tt.wantReason == "" {
				if err != nil {
					t.Fatalf("NormalizeTokenCount() error = %v", err)
				}
				if got != tt.want {
					t.Errorf("NormalizeTokenCount() = %d, want %d", got, tt.want)
				}
				return
			}
			var normalizationErr *inference.UsageNormalizationError
			if !errors.As(err, &normalizationErr) {
				t.Fatalf("NormalizeTokenCount() error = %T, want *UsageNormalizationError", err)
			}
			if normalizationErr.Field != tt.field || normalizationErr.Reason != tt.wantReason || normalizationErr.Value != tt.value {
				t.Errorf("normalization error = %+v, want field=%q reason=%q value=%d", normalizationErr, tt.field, tt.wantReason, tt.value)
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
		wantReason inference.UsageNormalizationReason
	}{
		{name: "no components", total: 9, want: 9},
		{name: "two components", total: 10, first: 3, second: 2, want: 5},
		{name: "components equal total", total: 5, first: 3, second: 2, want: 0},
		{name: "components exceed total", total: 4, first: 3, second: 2, wantReason: inference.UsageNormalizationReasonComponentsExceedTotal},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := inference.SubtractTokenCounts(inference.UsageNormalizationFieldInputTokens, tt.total, tt.first, tt.second)
			if tt.wantReason == "" {
				if err != nil {
					t.Fatalf("SubtractTokenCounts() error = %v", err)
				}
				if got != tt.want {
					t.Errorf("SubtractTokenCounts() = %d, want %d", got, tt.want)
				}
				return
			}
			assertUsageNormalizationError(t, err, inference.UsageNormalizationFieldInputTokens, tt.wantReason)
		})
	}
}

func TestAddTokenCounts(t *testing.T) {
	t.Parallel()

	max := content.TokenCount(^uint64(0))
	tests := []struct {
		name       string
		left       content.TokenCount
		right      content.TokenCount
		want       content.TokenCount
		wantReason inference.UsageNormalizationReason
	}{
		{name: "zero", left: 0, right: 0, want: 0},
		{name: "boundary", left: max - 1, right: 1, want: max},
		{name: "overflow", left: max, right: 1, wantReason: inference.UsageNormalizationReasonOverflow},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := inference.AddTokenCounts(inference.UsageNormalizationFieldOutputTokens, tt.left, tt.right)
			if tt.wantReason == "" {
				if err != nil {
					t.Fatalf("AddTokenCounts() error = %v", err)
				}
				if got != tt.want {
					t.Errorf("AddTokenCounts() = %d, want %d", got, tt.want)
				}
				return
			}
			assertUsageNormalizationError(t, err, inference.UsageNormalizationFieldOutputTokens, tt.wantReason)
		})
	}
}

func TestValidateNormalizedUsage(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		usage      inference.Usage
		wantReason inference.UsageNormalizationReason
	}{
		{name: "zero", usage: inference.Usage{}},
		{name: "reasoning subset", usage: inference.Usage{OutputTokens: 3, ReasoningTokens: 3}},
		{name: "reasoning exceeds output", usage: inference.Usage{OutputTokens: 2, ReasoningTokens: 3}, wantReason: inference.UsageNormalizationReasonReasoningExceedsOutput},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			err := inference.ValidateNormalizedUsage(tt.usage)
			if tt.wantReason == "" {
				if err != nil {
					t.Errorf("ValidateNormalizedUsage() error = %v", err)
				}
				return
			}
			assertUsageNormalizationError(t, err, inference.UsageNormalizationFieldReasoningTokens, tt.wantReason)
			var contentErr *content.UsageValidationError
			if !errors.As(err, &contentErr) {
				t.Errorf("ValidateNormalizedUsage() error does not preserve *content.UsageValidationError: %v", err)
			}
		})
	}
}

func assertUsageNormalizationError(t *testing.T, err error, field inference.UsageNormalizationField, reason inference.UsageNormalizationReason) {
	t.Helper()
	var normalizationErr *inference.UsageNormalizationError
	if !errors.As(err, &normalizationErr) {
		t.Fatalf("error = %T, want *UsageNormalizationError", err)
	}
	if normalizationErr.Field != field || normalizationErr.Reason != reason {
		t.Errorf("normalization error = %+v, want field=%q reason=%q", normalizationErr, field, reason)
	}
}
