package retry

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/failure"
	"github.com/looprig/inference/stream"
)

type outcome struct {
	resp   *inference.Response
	reader *stream.StreamReader[content.Chunk]
	err    error
}

// scriptedClient returns queued outcomes in order and records call counts.
type scriptedClient struct {
	mu             sync.Mutex
	invokeOutcomes []outcome
	streamOutcomes []outcome
	calls          int
	streamCalls    int
	invokeStarted  chan struct{}
	streamStarted  chan struct{}
}

func (c *scriptedClient) Invoke(context.Context, inference.Request) (*inference.Response, error) {
	c.mu.Lock()
	c.calls++
	if c.calls == 1 && c.invokeStarted != nil {
		close(c.invokeStarted)
		c.invokeStarted = nil
	}
	index := c.calls - 1
	if len(c.invokeOutcomes) == 0 {
		c.mu.Unlock()
		return nil, errors.New("scripted client: no Invoke outcome")
	}
	if index >= len(c.invokeOutcomes) {
		index = len(c.invokeOutcomes) - 1
	}
	result := c.invokeOutcomes[index]
	c.mu.Unlock()
	return result.resp, result.err
}

func (c *scriptedClient) Stream(context.Context, inference.Request) (*stream.StreamReader[content.Chunk], error) {
	c.mu.Lock()
	c.streamCalls++
	if c.streamCalls == 1 && c.streamStarted != nil {
		close(c.streamStarted)
		c.streamStarted = nil
	}
	index := c.streamCalls - 1
	if len(c.streamOutcomes) == 0 {
		c.mu.Unlock()
		return nil, errors.New("scripted client: no Stream outcome")
	}
	if index >= len(c.streamOutcomes) {
		index = len(c.streamOutcomes) - 1
	}
	result := c.streamOutcomes[index]
	c.mu.Unlock()
	return result.reader, result.err
}

func (c *scriptedClient) callCounts() (int, int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.calls, c.streamCalls
}

func newTestClient(t *testing.T, inner *scriptedClient) (*Client, *[]time.Duration) {
	t.Helper()
	c, err := New(inner, validPolicy())
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	delays := make([]time.Duration, 0)
	c.after = func(delay time.Duration) <-chan time.Time {
		delays = append(delays, delay)
		done := make(chan time.Time)
		close(done)
		return done
	}
	c.randFloat = func() float64 { return 0.5 }
	return c, &delays
}

func TestInvoke_SuccessFirstTry(t *testing.T) {
	inner := &scriptedClient{invokeOutcomes: []outcome{{resp: &inference.Response{Model: "model"}}}}
	c, delays := newTestClient(t, inner)

	resp, err := c.Invoke(context.Background(), inference.Request{})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if resp.Attempts != 1 {
		t.Fatalf("Attempts = %d, want 1", resp.Attempts)
	}
	calls, _ := inner.callCounts()
	if calls != 1 {
		t.Fatalf("Invoke calls = %d, want 1", calls)
	}
	if len(*delays) != 0 {
		t.Fatalf("delays = %v, want none", *delays)
	}
}

func TestInvoke_RetriesThenSucceeds(t *testing.T) {
	inner := &scriptedClient{invokeOutcomes: []outcome{
		{err: &failure.APIError{Status: 429}},
		{err: &failure.APIError{Status: 429}},
		{resp: &inference.Response{Model: "model"}},
	}}
	c, delays := newTestClient(t, inner)

	resp, err := c.Invoke(context.Background(), inference.Request{})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if resp.Attempts != 3 {
		t.Fatalf("Attempts = %d, want 3", resp.Attempts)
	}
	calls, _ := inner.callCounts()
	if calls != 3 {
		t.Fatalf("Invoke calls = %d, want 3", calls)
	}
	if want := []time.Duration{2 * time.Second, 2 * time.Second}; !reflect.DeepEqual(*delays, want) {
		t.Fatalf("delays = %v, want %v", *delays, want)
	}
}

func TestInvoke_StableThenExponential(t *testing.T) {
	inner := &scriptedClient{invokeOutcomes: []outcome{
		{err: &failure.APIError{Status: 429}},
		{err: &failure.APIError{Status: 429}},
		{err: &failure.APIError{Status: 429}},
		{err: &failure.APIError{Status: 429}},
		{err: &failure.APIError{Status: 429}},
		{resp: &inference.Response{Model: "model"}},
	}}
	c, delays := newTestClient(t, inner)

	if _, err := c.Invoke(context.Background(), inference.Request{}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if want := []time.Duration{2 * time.Second, 2 * time.Second, 2 * time.Second, 4 * time.Second, 8 * time.Second}; !reflect.DeepEqual(*delays, want) {
		t.Fatalf("delays = %v, want %v", *delays, want)
	}
}

func TestInvoke_NonRetryableFailsFast(t *testing.T) {
	wantErr := &failure.APIError{Status: 400}
	inner := &scriptedClient{invokeOutcomes: []outcome{{err: wantErr}}}
	c, delays := newTestClient(t, inner)

	_, err := c.Invoke(context.Background(), inference.Request{})
	if err != wantErr {
		t.Fatalf("error = %v (%T), want original %v", err, err, wantErr)
	}
	var apiErr *failure.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error %T does not unwrap to *failure.APIError", err)
	}
	calls, _ := inner.callCounts()
	if calls != 1 {
		t.Fatalf("Invoke calls = %d, want 1", calls)
	}
	if len(*delays) != 0 {
		t.Fatalf("delays = %v, want none", *delays)
	}
}

func TestInvoke_Exhaustion(t *testing.T) {
	lastErr := &failure.APIError{Status: 429, Message: "last"}
	inner := &scriptedClient{invokeOutcomes: []outcome{
		{err: &failure.APIError{Status: 429}},
		{err: &failure.APIError{Status: 429}},
		{err: &failure.APIError{Status: 429}},
		{err: &failure.APIError{Status: 429}},
		{err: &failure.APIError{Status: 429}},
		{err: lastErr},
	}}
	c, _ := newTestClient(t, inner)

	_, err := c.Invoke(context.Background(), inference.Request{})
	var exhausted *ExhaustedError
	if !errors.As(err, &exhausted) {
		t.Fatalf("error = %T, want *ExhaustedError", err)
	}
	if exhausted.Attempts != 6 {
		t.Fatalf("Attempts = %d, want 6", exhausted.Attempts)
	}
	var apiErr *failure.APIError
	if !errors.As(err, &apiErr) || apiErr != lastErr {
		t.Fatalf("error chain does not expose final API error: %v", err)
	}
	calls, _ := inner.callCounts()
	if calls != 6 {
		t.Fatalf("Invoke calls = %d, want 6", calls)
	}
}

func TestInvoke_RetryAfterOverridesSchedule(t *testing.T) {
	inner := &scriptedClient{invokeOutcomes: []outcome{
		{err: &failure.APIError{Status: 429, RetryAfter: 60 * time.Second}},
		{resp: &inference.Response{Model: "model"}},
	}}
	c, delays := newTestClient(t, inner)

	if _, err := c.Invoke(context.Background(), inference.Request{}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if want := []time.Duration{60 * time.Second}; !reflect.DeepEqual(*delays, want) {
		t.Fatalf("delays = %v, want %v", *delays, want)
	}
}

func TestInvoke_ContextCancelDuringWait(t *testing.T) {
	started := make(chan struct{})
	inner := &scriptedClient{
		invokeOutcomes: []outcome{{err: &failure.APIError{Status: 429}}},
		invokeStarted:  started,
	}
	c, _ := newTestClient(t, inner)
	c.after = func(time.Duration) <-chan time.Time {
		return make(chan time.Time)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	result := make(chan error, 1)
	go func() {
		_, err := c.Invoke(ctx, inference.Request{})
		result <- err
	}()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("first Invoke attempt did not start")
	}
	cancel()
	select {
	case err := <-result:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("error = %v, want context.Canceled", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Invoke did not return after context cancellation")
	}
	calls, _ := inner.callCounts()
	if calls != 1 {
		t.Fatalf("Invoke calls = %d, want 1", calls)
	}
}

func TestInvoke_NilResponseFailsWithTypedError(t *testing.T) {
	inner := &scriptedClient{invokeOutcomes: []outcome{{resp: nil, err: nil}}}
	c, _ := newTestClient(t, inner)

	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("Invoke panicked on nil response: %v", recovered)
		}
	}()
	_, err := c.Invoke(context.Background(), inference.Request{})
	var invalid *InvalidResponseError
	if !errors.As(err, &invalid) {
		t.Fatalf("error = %T, want *InvalidResponseError", err)
	}
}
