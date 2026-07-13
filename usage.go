package inference

import "github.com/looprig/core/content"

const maximumTokenCount content.TokenCount = ^content.TokenCount(0)

// Usage is the canonical normalized token-usage domain owned by core/content.
type Usage = content.Usage

// NormalizeTokenCount rejects negative provider integers before conversion.
func NormalizeTokenCount(field UsageNormalizationField, value int) (content.TokenCount, error) {
	if value < 0 {
		return 0, &UsageNormalizationError{
			Field:  field,
			Reason: UsageNormalizationReasonNegative,
			Value:  value,
		}
	}
	return content.TokenCount(value), nil
}

// AddTokenCounts adds two normalized counts without wrapping.
func AddTokenCounts(field UsageNormalizationField, left, right content.TokenCount) (content.TokenCount, error) {
	if right > maximumTokenCount-left {
		return 0, &UsageNormalizationError{
			Field:  field,
			Reason: UsageNormalizationReasonOverflow,
			Left:   left,
			Right:  right,
		}
	}
	return left + right, nil
}

// SubtractTokenCounts removes two disjoint components from a reported total.
func SubtractTokenCounts(field UsageNormalizationField, total, first, second content.TokenCount) (content.TokenCount, error) {
	components, err := AddTokenCounts(field, first, second)
	if err != nil {
		return 0, err
	}
	if components > total {
		return 0, &UsageNormalizationError{
			Field:  field,
			Reason: UsageNormalizationReasonComponentsExceedTotal,
			Left:   total,
			Right:  components,
		}
	}
	return total - components, nil
}

// ValidateNormalizedUsage verifies cross-field normalized usage invariants.
func ValidateNormalizedUsage(usage Usage) error {
	if err := usage.Validate(); err != nil {
		return &UsageNormalizationError{
			Field:  UsageNormalizationFieldReasoningTokens,
			Reason: UsageNormalizationReasonReasoningExceedsOutput,
			Left:   usage.OutputTokens,
			Right:  usage.ReasoningTokens,
			Cause:  err,
		}
	}
	return nil
}
