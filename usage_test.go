package inference_test

import (
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
)

func TestUsageAlias(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		usage inference.Usage
		want  content.Usage
	}{
		{name: "zero", usage: inference.Usage{}, want: content.Usage{}},
		{
			name: "all fields",
			usage: inference.Usage{
				InputTokens: 1, OutputTokens: 2, CacheReadTokens: 3,
				CacheCreationTokens: 4, ReasoningTokens: 1,
			},
			want: content.Usage{
				InputTokens: 1, OutputTokens: 2, CacheReadTokens: 3,
				CacheCreationTokens: 4, ReasoningTokens: 1,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			var got content.Usage = tt.usage
			if got != tt.want {
				t.Errorf("content.Usage = %+v, want %+v", got, tt.want)
			}
		})
	}
}
