package inference_test

import (
	"encoding/json"
	"testing"

	"github.com/looprig/inference"
)

func FuzzValidateOutputSchema(f *testing.F) {
	seeds := []string{
		`{"type":"object","properties":{},"required":[],"additionalProperties":false}`,
		`{"type":"object","properties":{"x":{"type":"array","items":{"type":"integer"}}},"required":["x"],"additionalProperties":false}`,
		`{"type":"object","additionalProperties":true}`,
		`{"type":["object","null"]}`,
		`{`,
		`[]`,
		"\xff",
	}
	for _, seed := range seeds {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, schema string) {
		output := inference.OutputSchema{
			Name:        "fuzz_result",
			Description: "Fuzzed schema.",
			Schema:      json.RawMessage(schema),
			Strict:      true,
		}
		_ = inference.ValidateOutputSchema(output)
	})
}
