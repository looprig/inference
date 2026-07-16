package usagenorm_test

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/looprig/inference/internal/usagenorm"
	usage "github.com/looprig/inference/usage"
)

func FuzzCountJSON(f *testing.F) {
	seeds := []string{
		`{}`, `{"count":0}`, `{"count":null}`, `{"count":-1}`,
		`{"count":1.5}`, `{"count":9223372036854775808}`, `{"count":"1"}`,
	}
	for _, seed := range seeds {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, input string) {
		var wire struct {
			Count usagenorm.Count `json:"count"`
		}
		if err := json.Unmarshal([]byte(input), &wire); err != nil {
			return
		}
		_, err := wire.Count.TokenCount(usagenorm.FieldInputTokens)
		if err == nil {
			return
		}
		var normalizationErr *usage.UsageNormalizationError
		if !errors.As(err, &normalizationErr) {
			t.Fatalf("TokenCount() error = %T %v, want *UsageNormalizationError", err, err)
		}
	})
}
