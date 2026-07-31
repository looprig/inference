package gateway_test

import (
	"context"
	"errors"
	"net/http"
	"sync"
	"testing"
	"time"

	"github.com/looprig/inference/gateway"
)

// --- Bounded forced shutdown --------------------------------------------------

// TestServer_Close_BoundedForcedShutdown_SlowInFlightRequest proves that a
// slow, still-in-flight request does not block Close past ServerConfig's
// ShutdownTimeout: Close force-closes the listener/connections and returns a
// *gateway.ShutdownTimeoutError within a bounded margin of that timeout,
// rather than hanging until the handler chooses to return.
func TestServer_Close_BoundedForcedShutdown_SlowInFlightRequest(t *testing.T) {
	assertNoGoroutineLeak(t)
	t.Parallel()

	const shutdownTimeout = 150 * time.Millisecond

	block := make(chan struct{})
	var closeBlockOnce sync.Once
	closeBlock := func() { closeBlockOnce.Do(func() { close(block) }) }
	t.Cleanup(closeBlock)

	started := make(chan struct{}, 1)
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case started <- struct{}{}:
		default:
		}
		<-block
		w.WriteHeader(http.StatusOK)
	})

	srv, err := gateway.NewServer(gateway.ServerConfig{Handler: handler, ShutdownTimeout: shutdownTimeout})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	if err := srv.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	baseURL, token, ready := srv.Binding()
	if !ready {
		t.Fatal("ready = false after Start")
	}

	reqDone := make(chan struct{})
	go func() {
		defer close(reqDone)
		req, err := http.NewRequest(http.MethodGet, baseURL+"/", nil)
		if err != nil {
			return
		}
		req.Header.Set("Authorization", "Bearer "+token)
		resp, err := http.DefaultClient.Do(req)
		if err == nil {
			resp.Body.Close()
		}
	}()

	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("handler never observed the slow in-flight request")
	}

	closeStart := time.Now()
	closeErr := srv.Close(context.Background())
	elapsed := time.Since(closeStart)

	const margin = 850 * time.Millisecond
	if elapsed > shutdownTimeout+margin {
		t.Fatalf("Close took %v, want within %v of ShutdownTimeout=%v", elapsed, margin, shutdownTimeout)
	}

	var shutdownErr *gateway.ShutdownTimeoutError
	if !errors.As(closeErr, &shutdownErr) {
		t.Fatalf("Close error = %v (%T), want *ShutdownTimeoutError", closeErr, closeErr)
	}

	closeBlock() // release the still-blocked handler goroutine
	<-reqDone
}

// --- Concurrent lifecycle races (run under -race) ----------------------------

func TestServer_ConcurrentStartCalls_OnlyOneSucceeds(t *testing.T) {
	assertNoGoroutineLeak(t)
	t.Parallel()

	srv := newTestServer(t, okHandler())

	const n = 8
	errs := make([]error, n)
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			errs[i] = srv.Start(context.Background())
		}(i)
	}
	wg.Wait()

	successes := 0
	for i, err := range errs {
		if err == nil {
			successes++
			continue
		}
		var stateErr *gateway.ServerStateError
		if !errors.As(err, &stateErr) {
			t.Fatalf("Start call %d: unexpected error %v (%T), want nil or *ServerStateError", i, err, err)
		}
	}
	if successes != 1 {
		t.Fatalf("successful concurrent Start calls = %d, want exactly 1", successes)
	}
	if _, _, ready := srv.Binding(); !ready {
		t.Fatal("server not ready after the concurrent Start race settled")
	}
}

func TestServer_ConcurrentClose_AllCallsReturnSameOutcome(t *testing.T) {
	assertNoGoroutineLeak(t)
	t.Parallel()

	srv := newTestServer(t, okHandler())
	mustStart(t, srv)

	const n = 8
	errs := make([]error, n)
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			errs[i] = srv.Close(context.Background())
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Fatalf("concurrent Close call %d = %v, want nil", i, err)
		}
	}
	if _, _, ready := srv.Binding(); ready {
		t.Fatal("ready = true after concurrent Close calls settled")
	}
}

// TestServer_ConcurrentBindingDuringStartAndClose_NoRace hammers Binding()
// from a background goroutine while Start and then Close run on the main
// goroutine. It makes no behavioral assertions beyond "no panic, no data
// race" -- its entire purpose is to be run under `go test -race`.
func TestServer_ConcurrentBindingDuringStartAndClose_NoRace(t *testing.T) {
	assertNoGoroutineLeak(t)
	t.Parallel()

	srv := newTestServer(t, okHandler())

	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
				srv.Binding()
			}
		}
	}()

	if err := srv.Start(context.Background()); err != nil {
		close(stop)
		wg.Wait()
		t.Fatalf("Start: %v", err)
	}
	if err := srv.Close(context.Background()); err != nil {
		close(stop)
		wg.Wait()
		t.Fatalf("Close: %v", err)
	}
	close(stop)
	wg.Wait()
}
