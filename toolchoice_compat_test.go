package inference_test

import (
	"testing"

	"github.com/looprig/inference"
)

func TestLegacyToolChoiceValuesRemainSourceCompatible(t *testing.T) {
	auto := inference.Request{ToolChoice: inference.ToolChoiceAuto}
	required := inference.Request{ToolChoice: inference.ToolChoiceRequired}
	if auto.ToolChoice != inference.ToolAuto() {
		t.Errorf("ToolChoiceAuto = %v, want ToolAuto()", auto.ToolChoice)
	}
	if required.ToolChoice != inference.ToolRequired() {
		t.Errorf("ToolChoiceRequired = %v, want ToolRequired()", required.ToolChoice)
	}
}
