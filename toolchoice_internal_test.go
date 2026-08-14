package inference

import (
	"errors"
	"testing"
)

// TestValidateRequestFeaturesRejectsUnknownToolChoiceMode pins the fail-closed
// arm of the tool-choice switch. The discriminant is unexported, so no caller
// outside this package can build a mode that is not one of the three
// constructors — this test has to be internal to construct one at all. The arm
// exists so that a variant added to this package later, but not taught to
// ValidateRequestFeatures, is refused rather than silently encoded as auto:
// every codec omits tool_choice for the modes it does not recognize, so a
// missing arm would drop caller intent on the wire.
func TestValidateRequestFeaturesRejectsUnknownToolChoiceMode(t *testing.T) {
	t.Parallel()

	err := ValidateRequestFeatures(Request{ToolChoice: ToolChoice{mode: ToolChoiceMode(99)}})
	var target *StructuredOutputConflictError
	if !errors.As(err, &target) {
		t.Fatalf("ValidateRequestFeatures() error = %T %v, want *StructuredOutputConflictError", err, err)
	}
	if target.Feature != "tool_choice" {
		t.Errorf("StructuredOutputConflictError.Feature = %q, want tool_choice", target.Feature)
	}
}
