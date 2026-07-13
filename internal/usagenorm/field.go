// Package usagenorm validates and normalizes provider token-usage wire values.
package usagenorm

import "github.com/looprig/inference"

// Field is the closed set of usage fields normalization may report.
type Field uint8

const (
	FieldInputTokens Field = iota + 1
	FieldOutputTokens
	FieldCacheReadTokens
	FieldCacheCreationTokens
	FieldReasoningTokens
	FieldContextTokens
	FieldTotalTokens
)

func normalizationField(field Field) (inference.UsageNormalizationField, error) {
	switch field {
	case FieldInputTokens:
		return inference.UsageNormalizationFieldInputTokens, nil
	case FieldOutputTokens:
		return inference.UsageNormalizationFieldOutputTokens, nil
	case FieldCacheReadTokens:
		return inference.UsageNormalizationFieldCacheReadTokens, nil
	case FieldCacheCreationTokens:
		return inference.UsageNormalizationFieldCacheCreationTokens, nil
	case FieldReasoningTokens:
		return inference.UsageNormalizationFieldReasoningTokens, nil
	case FieldContextTokens:
		return inference.UsageNormalizationFieldContextTokens, nil
	case FieldTotalTokens:
		return inference.UsageNormalizationFieldTotalTokens, nil
	default:
		return "", &inference.UsageNormalizationError{Reason: inference.UsageNormalizationReasonInvalidField}
	}
}
