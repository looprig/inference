package inference

import (
	"reflect"
	"testing"
)

func TestDecodeStructuredOutputCommitsOnlyAfterCompleteDecode(t *testing.T) {
	t.Parallel()

	type result struct {
		Name   string   `json:"name"`
		Count  int      `json:"count"`
		Labels []string `json:"labels"`
	}
	original := result{Name: "preserve", Count: 42, Labels: []string{"a", "b"}}
	tests := []struct {
		name string
		raw  string
	}{
		{name: "unknown field", raw: `{"name":"changed","extra":true}`},
		{name: "type mismatch after valid fields", raw: `{"name":"changed","count":"wrong"}`},
		{name: "trailing JSON", raw: `{"name":"changed"}{}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := result{Name: original.Name, Count: original.Count, Labels: append([]string(nil), original.Labels...)}
			if err := decodeStructuredOutput([]byte(tt.raw), &got); err == nil {
				t.Fatal("decodeStructuredOutput() error = nil")
			}
			if !reflect.DeepEqual(got, original) {
				t.Fatalf("target changed on failure: got %+v, want %+v", got, original)
			}
		})
	}
}
