package gateway

import "testing"

func TestTargetAuthoritativeEffortDefaultsFalse(t *testing.T) {
	t.Parallel()
	var target Target
	if target.AuthoritativeEffort {
		t.Fatal("zero Target must not claim effort authority")
	}
}
