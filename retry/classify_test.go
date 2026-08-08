package retry

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/looprig/inference/failure"
)

func TestRetryable(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"network", &failure.NetworkError{Err: errors.New("dial")}, true},
		{"wrapped network", fmt.Errorf("x: %w", &failure.NetworkError{Err: errors.New("d")}), true},
		{"api 408", &failure.APIError{Status: 408}, true},
		{"api 429", &failure.APIError{Status: 429}, true},
		{"api 500", &failure.APIError{Status: 500}, true},
		{"api 599", &failure.APIError{Status: 599}, true},
		{"api 400", &failure.APIError{Status: 400}, false},
		{"api 401", &failure.APIError{Status: 401}, false},
		{"api 404", &failure.APIError{Status: 404}, false},
		{"ctx canceled", context.Canceled, false},
		{"ctx deadline", context.DeadlineExceeded, false},
		{"canceled wrapping network", fmt.Errorf("%w: %w", context.Canceled, &failure.NetworkError{Err: errors.New("d")}), false},
		{"exhausted wrapping api", &ExhaustedError{Attempts: 6, Cause: &failure.APIError{Status: 429}}, false},
		{"other", errors.New("boom"), false},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := Retryable(tc.err); got != tc.want {
				t.Fatalf("Retryable(%v) = %v, want %v", tc.err, got, tc.want)
			}
		})
	}
}
