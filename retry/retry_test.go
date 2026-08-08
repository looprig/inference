package retry

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/stream"
)

func validPolicy() Policy {
	return Policy{StableRetries: 3, StableDelay: 2 * time.Second, MaxAttempts: 6, MaxDelay: 30 * time.Second}
}

type stubClient struct{}

func (stubClient) Invoke(context.Context, inference.Request) (*inference.Response, error) {
	return nil, nil
}

func (stubClient) Stream(context.Context, inference.Request) (*stream.StreamReader[content.Chunk], error) {
	return nil, nil
}

func TestNew_Validation(t *testing.T) {
	t.Parallel()
	if _, err := New(nil, validPolicy()); err == nil {
		t.Fatal("nil client accepted")
	}
	bad := []Policy{
		{},
		{StableRetries: -1, StableDelay: time.Second, MaxAttempts: 2, MaxDelay: time.Second},
		{StableRetries: 0, StableDelay: 0, MaxAttempts: 2, MaxDelay: time.Second},
		{StableRetries: 1, StableDelay: time.Second, MaxAttempts: 0, MaxDelay: time.Second},
		{StableRetries: 5, StableDelay: time.Second, MaxAttempts: 3, MaxDelay: time.Second},
		{StableRetries: 1, StableDelay: time.Second, MaxAttempts: 2, MaxDelay: 0},
	}
	for i, p := range bad {
		if _, err := New(stubClient{}, p); err == nil {
			t.Fatalf("bad policy %d accepted: %+v", i, p)
		} else {
			var cfg *ConfigError
			if !errors.As(err, &cfg) {
				t.Fatalf("bad policy %d: error %T is not *ConfigError", i, err)
			}
		}
	}
	if _, err := New(stubClient{}, validPolicy()); err != nil {
		t.Fatalf("valid policy rejected: %v", err)
	}
}
