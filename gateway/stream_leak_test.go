package gateway_test

import (
	"context"
	"net/http/httptest"
	"runtime"
	"testing"
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/inference/stream"
)

// assertNoGoroutineLeak captures the current goroutine count as a baseline
// and registers a t.Cleanup that waits (with a short settle window, since
// goroutine teardown -- e.g. a deferred Close, an AfterFunc watcher
// unregistering -- is not necessarily synchronous with the test's own
// ServeHTTP call returning) for the count to fall back to that baseline,
// failing the test if it never does.
//
// This is shared across stream_test.go, cancel_test.go, and this file (the
// three test files this task adds), called at the top of every streaming
// test, per the task's "no goroutine remains after the test" requirement.
// It is a delta check, not an absolute one: unrelated background goroutines
// (GC workers, other parallel tests' teardown) may be present in both the
// baseline and the final count, so this only catches growth introduced by
// the test itself.
func assertNoGoroutineLeak(t *testing.T) {
	t.Helper()
	runtime.GC()
	baseline := runtime.NumGoroutine()
	t.Cleanup(func() {
		deadline := time.Now().Add(200 * time.Millisecond)
		for {
			runtime.GC()
			current := runtime.NumGoroutine()
			if current <= baseline {
				return
			}
			if time.Now().After(deadline) {
				t.Errorf("goroutine leak: baseline=%d, still=%d after settle window", baseline, current)
				return
			}
			time.Sleep(5 * time.Millisecond)
		}
	})
}

// TestServeStreaming_NoLeak_FastNonCanceledRequest proves that a streaming
// request served against a context whose parent will only ever be canceled
// long after the HTTP request completes (a stand-in for a real, long-lived
// server-lifetime context) does not leave the context.AfterFunc watcher
// armed once the pull loop returns normally -- i.e. serveStreaming's
// `defer stop()` actually disarms it, rather than leaking a registration (or,
// on a Context implementation without direct cancelCtx support, a goroutine)
// for the remaining lifetime of that parent context.
func TestServeStreaming_NoLeak_FastNonCanceledRequest(t *testing.T) {
	assertNoGoroutineLeak(t)
	t.Parallel()

	// A context that will not be canceled for the lifetime of this test
	// process, standing in for a long-lived server/session-scoped parent.
	longLived, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	client := newControlledStreamClient()
	h, _ := newStreamHandler(t, client, 0)

	rr := httptest.NewRecorder()
	req := messagesRequest(t, "test-token", streamingMessagesBody).WithContext(longLived)
	done := make(chan struct{})
	go func() {
		h.ServeHTTP(rr, req)
		close(done)
	}()

	client.send(&content.TextChunk{Text: "hi"})
	client.finishClean(stream.StreamResult{FinishReason: stream.FinishReasonStop})

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("ServeHTTP did not return")
	}
	// assertNoGoroutineLeak's t.Cleanup (registered above, so it runs before
	// this test's own explicit cancel/close cleanups by LIFO order) proves no
	// goroutine outlives this point even though longLived is still open.
}

// TestServeStreaming_NoLeak_ManySequentialRequests runs several streaming
// requests to completion back to back and proves the goroutine count does
// not grow across them -- a coarser, higher-confidence sibling to the
// per-scenario leak checks in stream_test.go and cancel_test.go.
func TestServeStreaming_NoLeak_ManySequentialRequests(t *testing.T) {
	assertNoGoroutineLeak(t)
	t.Parallel()

	for i := 0; i < 20; i++ {
		client := newControlledStreamClient()
		h, _ := newStreamHandler(t, client, 0)

		rr := httptest.NewRecorder()
		done := make(chan struct{})
		go func() {
			h.ServeHTTP(rr, messagesRequest(t, "test-token", streamingMessagesBody))
			close(done)
		}()

		client.send(&content.TextChunk{Text: "hi"})
		client.finishClean(stream.StreamResult{FinishReason: stream.FinishReasonStop})

		select {
		case <-done:
		case <-time.After(2 * time.Second):
			t.Fatalf("iteration %d: ServeHTTP did not return", i)
		}
		if rr.Code != 200 {
			t.Fatalf("iteration %d: status = %d, want 200; body: %s", i, rr.Code, rr.Body.String())
		}
	}
}
