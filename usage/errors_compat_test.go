package usage_test

import (
	"errors"
	"testing"

	"github.com/looprig/inference/usage"
)

func TestUsageNormalizationErrorCauseCompatibility(t *testing.T) {
	cause := errors.New("legacy validation")
	err := &usage.UsageNormalizationError{
		Reason: usage.UsageNormalizationReasonDomainValidation,
		Cause:  cause,
	}
	if !errors.Is(err, cause) {
		t.Fatalf("errors.Is(%v, cause) = false", err)
	}
}
