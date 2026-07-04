package inference_test

import (
	"testing"

	"github.com/looprig/inference"
)

// TestAPIFormat_BuiltinConstants confirms the built-in convenience labels keep
// their canonical string values.
func TestAPIFormat_BuiltinConstants(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		f    inference.APIFormat
		want string
	}{
		{name: "openai", f: inference.APIFormatOpenAI, want: "openai"},
		{name: "anthropic", f: inference.APIFormatAnthropic, want: "anthropic"},
		{name: "gemini", f: inference.APIFormatGemini, want: "gemini"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if string(tc.f) != tc.want {
				t.Errorf("APIFormat = %q, want %q", tc.f, tc.want)
			}
		})
	}
}

// TestAPIFormat_OpenLabel asserts APIFormat is an open string label with NO
// fail-closed validation: an unknown/custom value is accepted, and a Model that
// carries one still validates (structural validation never rejects on APIFormat).
func TestAPIFormat_OpenLabel(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		f    inference.APIFormat
	}{
		{name: "empty", f: inference.APIFormat("")},
		{name: "unknown dialect", f: inference.APIFormat("some-custom-wire")},
		{name: "provider-shaped but unbundled", f: inference.APIFormat("bedrock-converse")},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			// A Model carrying an unknown APIFormat passes structural validation:
			// nothing at the inference layer fails closed on an unknown dialect.
			m := inference.CustomModel(inference.ProviderName("custom"), tc.f, "https://api.example.test", "some-model")
			if err := m.Validate(); err != nil {
				t.Errorf("Model.Validate() with APIFormat %q = %v, want nil (open label, no fail-closed gate)", tc.f, err)
			}
		})
	}
}
