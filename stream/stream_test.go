package stream_test

import (
	"errors"
	"io"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/looprig/core/content"
	stream "github.com/looprig/inference/stream"
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

			r := stream.NewStreamReader(next, nil)

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
			r := stream.NewStreamReader(next, closer)

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
			r := stream.NewStreamReader(next, closer)

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

			r := stream.NewStreamReader(next, nil)

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
	tests := []struct {
		name            string
		simple          bool
		nextErrors      []error
		producer        stream.StreamResultProducer
		beforeOK        bool
		wantErr         error
		wantResultError bool
		wantOK          bool
		wantResult      stream.StreamResult
		wantCalls       int
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
			producer: func() (stream.StreamResult, bool, error) {
				return stream.StreamResult{Model: "model-a", FinishReason: stream.FinishReasonStop}, true, nil
			},
			wantOK:     true,
			wantResult: stream.StreamResult{Model: "model-a", FinishReason: stream.FinishReasonStop},
			wantCalls:  1,
		},
		{
			name:       "absent producer result remains unavailable",
			nextErrors: []error{io.EOF},
			producer: func() (stream.StreamResult, bool, error) {
				return stream.StreamResult{}, false, nil
			},
			wantCalls: 1,
		},
		{
			name:       "non EOF error permanently prevents result",
			nextErrors: []error{errProducer, io.EOF},
			producer: func() (stream.StreamResult, bool, error) {
				return stream.StreamResult{Model: "must-not-appear"}, true, nil
			},
			wantErr:   errProducer,
			wantCalls: 0,
		},
		{
			name:       "producer error makes EOF non authoritative",
			nextErrors: []error{io.EOF},
			producer: func() (stream.StreamResult, bool, error) {
				return stream.StreamResult{}, false, errProducer
			},
			wantErr:         errProducer,
			wantResultError: true,
			wantCalls:       1,
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
				producer = func() (stream.StreamResult, bool, error) {
					producerCalls++
					return tt.producer()
				}
			}
			var reader *stream.StreamReader[string]
			if tt.simple {
				reader = stream.NewStreamReader(next, nil)
			} else {
				reader = stream.NewStreamReaderWithResult(next, nil, producer)
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
			} else if !streamErrorHasCause(err, tt.wantErr, tt.wantResultError) {
				t.Fatalf("Next() error = %v, want cause %T/%v", err, tt.wantErr, tt.wantErr)
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
			} else if !streamErrorHasCause(secondErr, tt.wantErr, tt.wantResultError) {
				t.Errorf("repeated Next() error = %v, want stable cause %T/%v", secondErr, tt.wantErr, tt.wantErr)
			}
			if producerCalls != tt.wantCalls {
				t.Errorf("producer calls = %d, want %d", producerCalls, tt.wantCalls)
			}
		})
	}
}

// TestStreamReader_ResultKeepsUsageDivergingFromReasoningConvention pins the
// counts from a live OpenRouter HTTP 200 against nvidia/nemotron-3-ultra-550b-
// a55b:free: completion_tokens=216 with reasoning_tokens=226. Usage is metrics
// and the chunks are the product, so a count that disagrees with the documented
// reasoning-subset convention must reach the caller as reported instead of
// turning a cleanly finished stream into a failed one.
func TestStreamReader_ResultKeepsUsageDivergingFromReasoningConvention(t *testing.T) {
	t.Parallel()

	divergent := content.Usage{OutputTokens: 216, ReasoningTokens: 226}
	reader := stream.NewStreamReaderWithResult(
		func() (string, error) { return "", io.EOF },
		nil,
		func() (stream.StreamResult, bool, error) {
			usage := divergent
			return stream.StreamResult{Usage: &usage, Model: "model-a"}, true, nil
		},
	)

	if _, err := reader.Next(); !errors.Is(err, io.EOF) {
		t.Fatalf("Next() error = %v, want io.EOF", err)
	}
	got, ok := reader.Result()
	if !ok {
		t.Fatal("Result() ok = false; an accounting mismatch must not withdraw terminal metadata")
	}
	if got.Usage == nil || *got.Usage != divergent {
		t.Errorf("Result().Usage = %+v, want %+v", got.Usage, divergent)
	}
	if got.Usage.ReasoningWithinOutput() {
		t.Error("ReasoningWithinOutput() = true, want false for the reported counts")
	}
}

func streamErrorHasCause(got error, want error, resultError bool) bool {
	if !resultError {
		return errors.Is(got, want)
	}
	var streamResultErr *stream.StreamResultError
	if !errors.As(got, &streamResultErr) {
		return false
	}
	return streamResultErr.Cause == want
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
			reader := stream.NewStreamReaderWithResult(
				func() (string, error) { return "", io.EOF },
				nil,
				func() (stream.StreamResult, bool, error) {
					return stream.StreamResult{
						Usage:        source,
						Model:        "provider-model",
						FinishReason: stream.FinishReasonLength,
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
			if second.Model != "provider-model" || second.FinishReason != stream.FinishReasonLength {
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
			reader := stream.NewStreamReaderWithResult(
				func() (string, error) { return "", io.EOF },
				func() error { closeRuns++; return tt.closerError },
				func() (stream.StreamResult, bool, error) {
					return stream.StreamResult{Model: "m"}, true, nil
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
		name          string
		reader        *stream.StreamReader[string]
		next          bool
		wantOperation stream.StreamOperation
		wantFailure   stream.StreamReaderFailure
	}{
		{
			name:          "nil receiver Next",
			reader:        nil,
			next:          true,
			wantOperation: stream.StreamOperationNext,
			wantFailure:   stream.StreamReaderFailureNilReceiver,
		},
		{
			name:          "nil receiver Close",
			reader:        nil,
			wantOperation: stream.StreamOperationClose,
			wantFailure:   stream.StreamReaderFailureNilReceiver,
		},
		{
			name:          "missing next callback",
			reader:        stream.NewStreamReader[string](nil, nil),
			next:          true,
			wantOperation: stream.StreamOperationNext,
			wantFailure:   stream.StreamReaderFailureMissingNext,
		},
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
			var streamErr *stream.StreamReaderError
			if !errors.As(err, &streamErr) {
				t.Fatalf("error = %T %v, want *stream.StreamReaderError", err, err)
			}
			if streamErr.Operation != tt.wantOperation || streamErr.Failure != tt.wantFailure {
				t.Errorf("error = operation %q failure %q, want %q/%q", streamErr.Operation, streamErr.Failure, tt.wantOperation, tt.wantFailure)
			}
			if _, ok := tt.reader.Result(); ok {
				t.Error("invalid reader unexpectedly has a result")
			}
		})
	}
}

func TestStreamReader_ResultErrorIsNotEOF(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		cause       error
		wantIs      error
		wantUsageAs bool
	}{
		{name: "producer EOF is metadata failure not clean stream exhaustion", cause: io.EOF},
		{name: "producer error wrapping EOF is metadata failure not clean stream exhaustion", cause: wrappedCause{cause: io.EOF}},
		{name: "ordinary sentinel cause remains visible to errors Is", cause: errStreamMetadata, wantIs: errStreamMetadata},
		{
			name:        "typed non EOF cause remains visible to errors As",
			cause:       &content.UsageOverflowError{Field: content.UsageFieldReasoningTokens, Left: 1, Right: 2},
			wantUsageAs: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			reader := stream.NewStreamReaderWithResult(
				func() (string, error) { return "", io.EOF },
				nil,
				func() (stream.StreamResult, bool, error) {
					return stream.StreamResult{}, false, tt.cause
				},
			)
			for call := 0; call < 2; call++ {
				_, err := reader.Next()
				if errors.Is(err, io.EOF) {
					t.Fatalf("Next() call %d error = %v matches io.EOF; metadata failure must be non-EOF", call, err)
				}
				var resultErr *stream.StreamResultError
				if !errors.As(err, &resultErr) {
					t.Fatalf("Next() call %d error = %T %v, want *StreamResultError", call, err, err)
				}
				if resultErr.Cause != tt.cause {
					t.Errorf("Next() call %d cause = %v, want exact %v", call, resultErr.Cause, tt.cause)
				}
				if tt.wantIs != nil && !errors.Is(err, tt.wantIs) {
					t.Errorf("Next() call %d error = %v, want errors.Is cause %v", call, err, tt.wantIs)
				}
				var usageErr *content.UsageOverflowError
				if got := errors.As(err, &usageErr); got != tt.wantUsageAs {
					t.Errorf("Next() call %d errors.As UsageOverflowError = %v, want %v", call, got, tt.wantUsageAs)
				}
			}
			if _, ok := reader.Result(); ok {
				t.Error("metadata failure unexpectedly authorized a result")
			}
		})
	}
}

type wrappedCause struct {
	cause error
}

func (e wrappedCause) Error() string { return "wrapped: " + e.cause.Error() }
func (e wrappedCause) Unwrap() error { return e.cause }

var errStreamMetadata = errors.New("stream metadata failed")

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
			reader := stream.NewStreamReaderWithResult(
				func() (string, error) { return "", io.EOF },
				func() error { return nil },
				func() (stream.StreamResult, bool, error) {
					return stream.StreamResult{Usage: &content.Usage{InputTokens: 1}}, true, nil
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
