package gateway_test

import (
	"context"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/inference/stream"
)

// TestServeStreaming_ContextCanceled_InterruptsBlockingNext proves the
// gateway's own cancellation mechanism (a context.AfterFunc watcher that
// closes the upstream StreamReader when the inbound request's context is
// canceled -- see stream.go) actually unblocks a Next() call the pull loop
// is currently blocked in: controlledStreamClient's next() only ever
// unblocks via either a value on steps or its closed channel (never
// ctx.Done() directly -- see its doc comment), so if this test's cancel
// call did NOT reach reader.Close() through the gateway's own watcher, the
// blocked Next() would hang until the 2s test timeout instead of returning
// promptly.
//
// Because we are already past OpenStream by the time Next() is blocked
// (headers are committed as soon as the pull loop starts), the resulting
// error is always a post-header failure: it must call the StreamEncoder's
// Fail, never fall back to sc.WriteError.
func TestServeStreaming_ContextCanceled_InterruptsBlockingNext(t *testing.T) {
	assertNoGoroutineLeak(t)
	client := newControlledStreamClient()
	h, spy := newStreamHandler(t, client, 0)

	ctx, cancel := context.WithCancel(context.Background())
	req := messagesRequest(t, "test-token", streamingMessagesBody).WithContext(ctx)
	rr := httptest.NewRecorder()

	done := make(chan struct{})
	go func() {
		h.ServeHTTP(rr, req)
		close(done)
	}()

	// Let the handler reach OpenStream and block inside Next() (no chunk or
	// finish has been sent) before canceling.
	deadline := time.After(2 * time.Second)
	for spy.openStreamCalls() == 0 {
		select {
		case <-deadline:
			t.Fatal("OpenStream was never reached before the cancel window elapsed")
		case <-time.After(time.Millisecond):
		}
	}

	cancel()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("ServeHTTP did not return after context cancellation -- blocking Next() was not interrupted")
	}

	if calls := client.closeCalls(); calls != 1 {
		t.Errorf("upstream StreamReader Close called %d times, want exactly 1", calls)
	}
	if fails := spy.failCalls(); len(fails) != 1 {
		t.Fatalf("Fail called %d times, want 1 (finishes: %v)", len(fails), spy.finishCalls())
	}
	if finishes := spy.finishCalls(); len(finishes) != 0 {
		t.Errorf("Finish called %d times on a canceled path, want 0", len(finishes))
	}
	// Headers were already committed by OpenStream before cancellation, so
	// the recorder still reports 200 -- a canceled in-stream failure is
	// never a fresh WriteError status.
	if rr.Code != 200 {
		t.Errorf("status = %d, want 200 (headers already committed by OpenStream)", rr.Code)
	}
}

// TestServeStreaming_ContextCanceledAfterCleanCompletion_NoDoubleClose
// proves that canceling the request's context AFTER the pull loop has
// already completed cleanly does not panic, double-close, or otherwise
// disturb the already-finished response -- because serveStreaming's
// `defer stop()` disarms the context.AfterFunc watcher as soon as the pull
// loop returns, a cancellation arriving afterward has nothing left to
// interrupt.
func TestServeStreaming_ContextCanceledAfterCleanCompletion_NoDoubleClose(t *testing.T) {
	assertNoGoroutineLeak(t)
	client := newControlledStreamClient()
	h, spy := newStreamHandler(t, client, 0)

	ctx, cancel := context.WithCancel(context.Background())
	req := messagesRequest(t, "test-token", streamingMessagesBody).WithContext(ctx)
	rr := httptest.NewRecorder()

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

	if rr.Code != 200 {
		t.Fatalf("status = %d, want 200; body: %s", rr.Code, rr.Body.String())
	}
	if finishes := spy.finishCalls(); len(finishes) != 1 {
		t.Fatalf("Finish called %d times, want 1", len(finishes))
	}
	if calls := client.closeCalls(); calls != 1 {
		t.Fatalf("upstream StreamReader Close called %d times before cancel, want 1", calls)
	}

	// Cancel now, well after completion. The disarmed watcher must not fire
	// a second Close, and nothing should panic.
	cancel()
	time.Sleep(20 * time.Millisecond)

	if calls := client.closeCalls(); calls != 1 {
		t.Errorf("upstream StreamReader Close called %d times after a late cancel, want still 1 (watcher not disarmed?)", calls)
	}
	if fails := spy.failCalls(); len(fails) != 0 {
		t.Errorf("Fail called %d times after a late cancel following clean completion, want 0", len(fails))
	}
}
