package model

import "testing"

func TestEffortValid(t *testing.T) {
	tests := []struct {
		name string
		e    Effort
		want bool
	}{
		{name: "none/empty", e: EffortNone, want: true},
		{name: "minimal", e: EffortMinimal, want: true},
		{name: "low", e: EffortLow, want: true},
		{name: "medium", e: EffortMedium, want: true},
		{name: "high", e: EffortHigh, want: true},
		{name: "xhigh", e: EffortXHigh, want: true},
		{name: "max", e: EffortMax, want: true},
		{name: "unknown", e: "ultra", want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := tt.e.Valid(); got != tt.want {
				t.Errorf("Effort(%q).Valid() = %v, want %v", tt.e, got, tt.want)
			}
		})
	}
}
