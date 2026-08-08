package retry

import (
	"errors"
	"testing"
	"time"

	"github.com/looprig/inference/failure"
)

func TestBaseDelay(t *testing.T) {
	t.Parallel()
	p := Policy{StableRetries: 3, StableDelay: 2 * time.Second, MaxAttempts: 8, MaxDelay: 10 * time.Second}
	want := []time.Duration{
		2 * time.Second,
		2 * time.Second,
		2 * time.Second,
		4 * time.Second,
		8 * time.Second,
		10 * time.Second,
		10 * time.Second,
	}
	for i, w := range want {
		if got := baseDelay(p, i+1); got != w {
			t.Fatalf("baseDelay(attempt %d) = %v, want %v", i+1, got, w)
		}
	}
}

func TestJitteredDelay_Bounds(t *testing.T) {
	t.Parallel()
	d := 10 * time.Second
	if got, want := jittered(d, 0.0), 9*time.Second; got != want {
		t.Fatalf("jittered(%v, 0.0) = %v, want %v", d, got, want)
	}
	if got, want := jittered(d, 1.0), 11*time.Second; got != want {
		t.Fatalf("jittered(%v, 1.0) = %v, want %v", d, got, want)
	}
}

func TestNextDelay_RetryAfterOverride(t *testing.T) {
	t.Parallel()
	p := Policy{StableRetries: 3, StableDelay: 2 * time.Second, MaxAttempts: 8, MaxDelay: 10 * time.Second}
	if got, want := nextDelay(p, 1, &failure.APIError{RetryAfter: 60 * time.Second}, 0.5), 60*time.Second; got != want {
		t.Fatalf("large Retry-After delay = %v, want %v", got, want)
	}
	if got, want := nextDelay(p, 1, &failure.APIError{RetryAfter: time.Second}, 0.5), 2*time.Second; got != want {
		t.Fatalf("small Retry-After delay = %v, want %v", got, want)
	}
	if got, want := nextDelay(p, 1, &failure.APIError{RetryAfter: 10 * time.Minute}, 0.5), 5*time.Minute; got != want {
		t.Fatalf("capped Retry-After delay = %v, want %v", got, want)
	}
	if got, want := nextDelay(p, 1, &failure.NetworkError{Err: errors.New("dial")}, 0.5), 2*time.Second; got != want {
		t.Fatalf("network delay = %v, want %v", got, want)
	}
}
