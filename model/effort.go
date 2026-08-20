package model

// Effort is dialect-neutral "how hard to think" intent. Each codec maps it to its wire
// mechanism (openaiapi → reasoning_effort; anthropicapi → adaptive thinking + effort). Zero
// value (EffortNone) means the model decides / thinking off.
type Effort string

const (
	EffortNone    Effort = ""
	EffortMinimal Effort = "minimal"
	EffortLow     Effort = "low"
	EffortMedium  Effort = "medium"
	EffortHigh    Effort = "high"
	EffortXHigh   Effort = "xhigh"
	EffortMax     Effort = "max"
)

// Valid reports whether e is a known effort level (the empty value is valid = unset).
func (e Effort) Valid() bool {
	switch e {
	case EffortNone, EffortMinimal, EffortLow, EffortMedium, EffortHigh, EffortXHigh, EffortMax:
		return true
	default:
		return false
	}
}
