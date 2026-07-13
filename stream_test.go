package inference_test

import (
	"errors"
	"io"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
)

func TestStreamReader_Next(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		items     []string
		wantItems []string
	}{
		{
			name:      "three item stream reads all items then EOF",
			items:     []string{"alpha", "beta", "gamma"},
			wantItems: []string{"alpha", "beta", "gamma"},
		},
		{
			name:      "single item stream",
			items:     []string{"only"},
			wantItems: []string{"only"},
		},
		{
			name:      "empty stream immediately returns EOF",
			items:     []string{},
			wantItems: []string{},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			idx := 0
			next := func() (string, error) {
				if idx >= len(tc.items) {
					return "", io.EOF
				}
				v := tc.items[idx]
				idx++
				return v, nil
			}

			r := inference.NewStreamReader(next, nil)

			var got []string
			for {
				v, err := r.Next()
				if errors.Is(err, io.EOF) {
					break
				}
				if err != nil {
					t.Fatalf("unexpected error before EOF: %v", err)
				}
				got = append(got, v)
			}

			if len(got) != len(tc.wantItems) {
				t.Fatalf("got %d items, want %d", len(got), len(tc.wantItems))
			}
			for i, want := range tc.wantItems {
				if got[i] != want {
					t.Errorf("item[%d]: got %q, want %q", i, got[i], want)
				}
			}
		})
	}
}

// TestStreamReader_CloseIdempotent asserts Close runs the wrapped closer at most
// once across repeated calls, and every call returns the first call's result.
func TestStreamReader_CloseIdempotent(t *testing.T) {
	t.Parallel()

	errClose := errors.New("close failed")

	cases := []struct {
		name     string
		closer   func(calls *int) func() error
		calls    int // number of Close() invocations
		wantRuns int // how many times the wrapped closer must run
		wantErr  error
	}{
		{
			name:     "double close on nil closer runs nothing, returns nil",
			closer:   nil,
			calls:    2,
			wantRuns: 0,
			wantErr:  nil,
		},
		{
			name: "double close runs closer once, returns nil twice",
			closer: func(calls *int) func() error {
				return func() error { *calls++; return nil }
			},
			calls:    2,
			wantRuns: 1,
			wantErr:  nil,
		},
		{
			name: "triple close runs error-closer once, returns same err each call",
			closer: func(calls *int) func() error {
				return func() error { *calls++; return errClose }
			},
			calls:    3,
			wantRuns: 1,
			wantErr:  errClose,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			runs := 0
			var closer func() error
			if tc.closer != nil {
				closer = tc.closer(&runs)
			}
			next := func() (string, error) { return "", io.EOF }
			r := inference.NewStreamReader(next, closer)

			for i := 0; i < tc.calls; i++ {
				if err := r.Close(); !errors.Is(err, tc.wantErr) {
					t.Errorf("Close() call %d error = %v, want %v", i, err, tc.wantErr)
				}
			}
			if runs != tc.wantRuns {
				t.Errorf("closer ran %d times, want %d", runs, tc.wantRuns)
			}
		})
	}
}

func TestStreamReader_Close(t *testing.T) {
	t.Parallel()

	errClose := errors.New("close failed")

	cases := []struct {
		name       string
		closer     func() error
		wantCalled bool
		wantErr    error
	}{
		{
			name:       "nil closer returns nil and is a no-op",
			closer:     nil,
			wantCalled: false,
			wantErr:    nil,
		},
		{
			name: "explicit closer sets flag and returns nil",
			closer: func() error {
				return nil
			},
			wantCalled: true,
			wantErr:    nil,
		},
		{
			name: "closer that returns error propagates it",
			closer: func() error {
				return errClose
			},
			wantCalled: true,
			wantErr:    errClose,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			called := false
			var closer func() error
			if tc.closer != nil {
				original := tc.closer
				closer = func() error {
					called = true
					return original()
				}
			}

			next := func() (string, error) { return "", io.EOF }
			r := inference.NewStreamReader(next, closer)

			err := r.Close()

			if tc.closer != nil && !called {
				t.Error("closer was not called")
			}
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("Close() error = %v, want %v", err, tc.wantErr)
			}
		})
	}
}

func TestStreamReader_ErrorPropagates(t *testing.T) {
	t.Parallel()

	sentinel := errors.New("stream error")

	cases := []struct {
		name      string
		items     []string
		errOnCall int
		wantItems []string
		wantErr   error
	}{
		{
			name:      "error on first call returns no items",
			items:     []string{"a", "b"},
			errOnCall: 0,
			wantItems: nil,
			wantErr:   sentinel,
		},
		{
			name:      "error after two items returns those items then error",
			items:     []string{"x", "y", "z"},
			errOnCall: 2,
			wantItems: []string{"x", "y"},
			wantErr:   sentinel,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			call := 0
			next := func() (string, error) {
				if call == tc.errOnCall {
					return "", sentinel
				}
				if call >= len(tc.items) {
					return "", io.EOF
				}
				v := tc.items[call]
				call++
				return v, nil
			}

			r := inference.NewStreamReader(next, nil)

			var got []string
			var finalErr error
			for {
				v, err := r.Next()
				if err != nil {
					finalErr = err
					break
				}
				got = append(got, v)
			}

			if !errors.Is(finalErr, tc.wantErr) {
				t.Errorf("final error = %v, want %v", finalErr, tc.wantErr)
			}
			if len(got) != len(tc.wantItems) {
				t.Fatalf("got %d items, want %d", len(got), len(tc.wantItems))
			}
			for i, want := range tc.wantItems {
				if got[i] != want {
					t.Errorf("item[%d]: got %q, want %q", i, got[i], want)
				}
			}
		})
	}
}

func TestStreamReader_Result(t *testing.T) {
	t.Parallel()

	errProducer := errors.New("producer failed")
	invalidUsage := &content.Usage{OutputTokens: 1, ReasoningTokens: 2}
	tests := []struct {
		name       string
		simple     bool
		nextErrors []error
		producer   inference.StreamResultProducer
		beforeOK   bool
		wantErr    error
		wantOK     bool
		wantResult inference.StreamResult
		wantCalls  int
	}{
		{
			name:       "simple reader has no result after clean EOF",
			simple:     true,
			nextErrors: []error{io.EOF},
		},
		{
			name:       "result aware reader with nil producer has no result",
			nextErrors: []error{io.EOF},
		},
		{
			name:       "terminal result becomes available after clean EOF",
			nextErrors: []error{io.EOF},
			producer: func() (inference.StreamResult, bool, error) {
				return inference.StreamResult{Model: "model-a", FinishReason: inference.FinishReasonStop}, true, nil
			},
			wantOK:     true,
			wantResult: inference.StreamResult{Model: "model-a", FinishReason: inference.FinishReasonStop},
			wantCalls:  1,
		},
		{
			name:       "absent producer result remains unavailable",
			nextErrors: []error{io.EOF},
			producer: func() (inference.StreamResult, bool, error) {
				return inference.StreamResult{}, false, nil
			},
			wantCalls: 1,
		},
		{
			name:       "non EOF error permanently prevents result",
			nextErrors: []error{errProducer, io.EOF},
			producer: func() (inference.StreamResult, bool, error) {
				return inference.StreamResult{Model: "must-not-appear"}, true, nil
			},
			wantErr:   errProducer,
			wantCalls: 0,
		},
		{
			name:       "producer error makes EOF non authoritative",
			nextErrors: []error{io.EOF},
			producer: func() (inference.StreamResult, bool, error) {
				return inference.StreamResult{}, false, errProducer
			},
			wantErr:   errProducer,
			wantCalls: 1,
		},
		{
			name:       "invalid producer usage makes EOF non authoritative",
			nextErrors: []error{io.EOF},
			producer: func() (inference.StreamResult, bool, error) {
				return inference.StreamResult{Usage: invalidUsage}, true, nil
			},
			wantErr:   &content.UsageValidationError{},
			wantCalls: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			var nextCalls atomic.Int32
			next := func() (string, error) {
				i := int(nextCalls.Add(1)) - 1
				if i >= len(tt.nextErrors) {
					return "", io.EOF
				}
				return "", tt.nextErrors[i]
			}
			producerCalls := 0
			producer := tt.producer
			if producer != nil {
				producer = func() (inference.StreamResult, bool, error) {
					producerCalls++
					return tt.producer()
				}
			}
			var reader *inference.StreamReader[string]
			if tt.simple {
				reader = inference.NewStreamReader(next, nil)
			} else {
				reader = inference.NewStreamReaderWithResult(next, nil, producer)
			}

			_, beforeOK := reader.Result()
			if beforeOK != tt.beforeOK {
				t.Fatalf("Result() before EOF ok = %v, want %v", beforeOK, tt.beforeOK)
			}
			_, err := reader.Next()
			if tt.wantErr == nil {
				if !errors.Is(err, io.EOF) {
					t.Fatalf("Next() error = %v, want io.EOF", err)
				}
			} else if !errors.Is(err, tt.wantErr) {
				var usageErr *content.UsageValidationError
				if _, ok := tt.wantErr.(*content.UsageValidationError); !ok || !errors.As(err, &usageErr) {
					t.Fatalf("Next() error = %v, want cause %T/%v", err, tt.wantErr, tt.wantErr)
				}
			}

			got, ok := reader.Result()
			if ok != tt.wantOK {
				t.Fatalf("Result() ok = %v, want %v (result %+v)", ok, tt.wantOK, got)
			}
			if ok && got != tt.wantResult {
				t.Errorf("Result() = %+v, want %+v", got, tt.wantResult)
			}
			_, secondErr := reader.Next()
			if tt.wantErr == nil {
				if !errors.Is(secondErr, io.EOF) {
					t.Errorf("repeated Next() error = %v, want stable io.EOF", secondErr)
				}
			} else if !errors.Is(secondErr, tt.wantErr) {
				var usageErr *content.UsageValidationError
				if _, expectedUsage := tt.wantErr.(*content.UsageValidationError); !expectedUsage || !errors.As(secondErr, &usageErr) {
					t.Errorf("repeated Next() error = %v, want stable cause %T/%v", secondErr, tt.wantErr, tt.wantErr)
				}
			}
			if producerCalls != tt.wantCalls {
				t.Errorf("producer calls = %d, want %d", producerCalls, tt.wantCalls)
			}
		})
	}
}

func TestStreamReader_ResultDefensiveCopy(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
	}{
		{name: "producer stays live until EOF then result snapshots every read"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			source := &content.Usage{InputTokens: 1, OutputTokens: 2}
			reader := inference.NewStreamReaderWithResult(
				func() (string, error) { return "", io.EOF },
				nil,
				func() (inference.StreamResult, bool, error) {
					return inference.StreamResult{
						Usage:        source,
						Model:        "provider-model",
						FinishReason: inference.FinishReasonLength,
					}, true, nil
				},
			)

			// Producer state is intentionally live while the stream is accumulating.
			source.InputTokens = 3
			if _, err := reader.Next(); !errors.Is(err, io.EOF) {
				t.Fatalf("Next() error = %v, want io.EOF", err)
			}
			source.InputTokens = 4
			first, ok := reader.Result()
			if !ok {
				t.Fatal("Result() unavailable after clean EOF")
			}
			if first.Usage == nil || first.Usage.InputTokens != 3 {
				t.Fatalf("first usage = %+v, want EOF snapshot input=3", first.Usage)
			}
			first.Usage.InputTokens = 5
			second, ok := reader.Result()
			if !ok || second.Usage == nil || second.Usage.InputTokens != 3 {
				t.Errorf("second Result() = %+v, %v; want independent input=3", second, ok)
			}
			if second.Model != "provider-model" || second.FinishReason != inference.FinishReasonLength {
				t.Errorf("terminal metadata = model %q reason %q", second.Model, second.FinishReason)
			}
		})
	}
}

func TestStreamReader_ResultCloseSemantics(t *testing.T) {
	t.Parallel()

	errClose := errors.New("close failed")
	tests := []struct {
		name          string
		closeBefore   bool
		closerError   error
		wantBeforeOK  bool
		wantAfterOK   bool
		wantCloseRuns int
	}{
		{name: "close before EOF does not authorize result", closeBefore: true, wantAfterOK: true, wantCloseRuns: 1},
		{name: "close error before EOF does not authorize or prevent later EOF result", closeBefore: true, closerError: errClose, wantAfterOK: true, wantCloseRuns: 1},
		{name: "close after EOF preserves result", wantAfterOK: true, wantCloseRuns: 1},
		{name: "close error after EOF preserves result", closerError: errClose, wantAfterOK: true, wantCloseRuns: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			closeRuns := 0
			reader := inference.NewStreamReaderWithResult(
				func() (string, error) { return "", io.EOF },
				func() error { closeRuns++; return tt.closerError },
				func() (inference.StreamResult, bool, error) {
					return inference.StreamResult{Model: "m"}, true, nil
				},
			)
			if tt.closeBefore {
				if err := reader.Close(); !errors.Is(err, tt.closerError) {
					t.Fatalf("Close() error = %v, want %v", err, tt.closerError)
				}
				_, ok := reader.Result()
				if ok != tt.wantBeforeOK {
					t.Fatalf("Result() after early Close ok = %v, want %v", ok, tt.wantBeforeOK)
				}
			}
			if _, err := reader.Next(); !errors.Is(err, io.EOF) {
				t.Fatalf("Next() error = %v, want io.EOF", err)
			}
			if err := reader.Close(); !errors.Is(err, tt.closerError) {
				t.Errorf("Close() error = %v, want %v", err, tt.closerError)
			}
			if err := reader.Close(); !errors.Is(err, tt.closerError) {
				t.Errorf("second Close() error = %v, want %v", err, tt.closerError)
			}
			_, ok := reader.Result()
			if ok != tt.wantAfterOK {
				t.Errorf("Result() after EOF and Close ok = %v, want %v", ok, tt.wantAfterOK)
			}
			if closeRuns != tt.wantCloseRuns {
				t.Errorf("closer runs = %d, want %d", closeRuns, tt.wantCloseRuns)
			}
		})
	}
}

func TestStreamReader_InvalidBoundary(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		reader *inference.StreamReader[string]
		next   bool
	}{
		{name: "nil receiver Next", reader: nil, next: true},
		{name: "nil receiver Close", reader: nil},
		{name: "missing next callback", reader: inference.NewStreamReader[string](nil, nil), next: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			var err error
			if tt.next {
				_, err = tt.reader.Next()
			} else {
				err = tt.reader.Close()
			}
			var streamErr *inference.StreamReaderError
			if !errors.As(err, &streamErr) {
				t.Fatalf("error = %T %v, want *inference.StreamReaderError", err, err)
			}
			if _, ok := tt.reader.Result(); ok {
				t.Error("invalid reader unexpectedly has a result")
			}
		})
	}
}

func TestStreamReader_ResultConcurrentWithClose(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
	}{
		{name: "Result and Close remain race free around EOF"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			reader := inference.NewStreamReaderWithResult(
				func() (string, error) { return "", io.EOF },
				func() error { return nil },
				func() (inference.StreamResult, bool, error) {
					return inference.StreamResult{Usage: &content.Usage{InputTokens: 1}}, true, nil
				},
			)
			var wg sync.WaitGroup
			wg.Add(3)
			go func() {
				defer wg.Done()
				_, _ = reader.Next()
			}()
			go func() {
				defer wg.Done()
				for i := 0; i < 100; i++ {
					_, _ = reader.Result()
				}
			}()
			go func() {
				defer wg.Done()
				for i := 0; i < 100; i++ {
					_ = reader.Close()
				}
			}()
			wg.Wait()
			result, ok := reader.Result()
			if !ok || result.Usage == nil || result.Usage.InputTokens != 1 {
				t.Errorf("Result() = %+v, %v; want authoritative usage", result, ok)
			}
		})
	}
}
