package usage_test

import (
	"testing"

	"github.com/looprig/core/content"
	usage "github.com/looprig/inference/usage"
)

func TestUsageAlias(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		usage usage.Usage
		want  content.Usage
	}{
		{name: "zero", usage: usage.Usage{}, want: content.Usage{}},
		{
			name: "all fields",
			usage: usage.Usage{
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
