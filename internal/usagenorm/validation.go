package usagenorm

import (
	"errors"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
)

// ValidateUsage applies the canonical core usage invariants and preserves typed
// core validation failures through normalization.
func ValidateUsage(usage content.Usage) error {
	if err := usage.Validate(); err != nil {
		normalized := NormalizeValidationError(err)
		var normalizationErr *inference.UsageNormalizationError
		if errors.As(normalized, &normalizationErr) &&
			normalizationErr.Reason == inference.UsageNormalizationReasonReasoningExceedsOutput {
			normalizationErr.Left = usage.OutputTokens
			normalizationErr.Right = usage.ReasoningTokens
		}
		return normalized
	}
	return nil
}

// NormalizeValidationError maps known core invariants specifically and keeps
// future invariants truthful through a generic reason and exact cause.
func NormalizeValidationError(err error) error {
	var validationErr *content.UsageValidationError
	if !errors.As(err, &validationErr) {
		return &inference.UsageNormalizationError{
			Reason: inference.UsageNormalizationReasonDomainValidation,
			Cause:  err,
		}
	}
	field := coreValidationField(validationErr.Field)
	reason := inference.UsageNormalizationReasonDomainValidation
	if validationErr.Field == content.UsageFieldReasoningTokens &&
		validationErr.Reason == content.UsageValidationReasonReasoningExceedsOutput {
		reason = inference.UsageNormalizationReasonReasoningExceedsOutput
	}
	return &inference.UsageNormalizationError{Field: field, Reason: reason, Cause: validationErr}
}

func coreValidationField(field content.UsageField) inference.UsageNormalizationField {
	switch field {
	case content.UsageFieldInputTokens:
		return inference.UsageNormalizationFieldInputTokens
	case content.UsageFieldOutputTokens:
		return inference.UsageNormalizationFieldOutputTokens
	case content.UsageFieldCacheReadTokens:
		return inference.UsageNormalizationFieldCacheReadTokens
	case content.UsageFieldCacheCreationTokens:
		return inference.UsageNormalizationFieldCacheCreationTokens
	case content.UsageFieldReasoningTokens:
		return inference.UsageNormalizationFieldReasoningTokens
	case content.UsageFieldContextTokens:
		return inference.UsageNormalizationFieldContextTokens
	case content.UsageFieldTotalTokens:
		return inference.UsageNormalizationFieldTotalTokens
	default:
		return inference.UsageNormalizationField(field)
	}
}
