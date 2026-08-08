# Retry Client Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** A provider-neutral retry decorator (`inference/retry`) that wraps any `inference.Client` with exponential backoff (3 stable retries of 2s, then doubling), wired into coderig's model loader.

**Architecture:** Leaf subpackage `inference/retry` implementing `inference.Client` by delegation. Retries cover `Invoke` and `Stream` *establishment* only; a mid-stream `Next()` error stays terminal. Classification reuses the typed `inference/failure` errors. `Retry-After` is plumbed onto `failure.APIError` by the transport. Attempt counts surface via new `Attempts` fields on `inference.Response` and `stream.StreamResult`.

**Tech Stack:** Go (module `github.com/looprig/inference`, consumer `coderig`). Design doc: `inference/docs/plans/2026-08-08-retry-client-design.md` — read it first; its non-goals are binding.

**Repos touched:** `inference` (tasks 1–7), `coderig` (task 8). Each is its own git repo; commit in the repo you edited. Workspace `go.work` at looprig root already covers both — do NOT tag/release or touch go.mod versions. Commit style: no Co-Authored-By trailer.

**Test commands:** `cd /Users/ipotter/code/looprig/inference && go test ./...` and `cd /Users/ipotter/code/looprig/coderig && go test ./...`. Also `go vet ./...` before each commit.

---

### Task 1: `failure.APIError` carries Retry-After

**Files:**
- Modify: `inference/failure/errors.go`
- Modify: `inference/transport/client.go:248` and `:285` (the two `&failure.APIError{...}` sites)
- Test: `inference/transport/retry_after_test.go` (new)

**Step 1: Write the failing test**

Test `parseRetryAfter` directly (unexported, same package `transport`... note: existing transport tests may be external `transport_test`; if so make this an internal test file `package transport`):

```go
package transport

import (
	"net/http"
	"testing"
	"time"
)

func TestParseRetryAfter(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		val  string
		want time.Duration
	}{
		{"absent", "", 0},
		{"integer seconds", "30", 30 * time.Second},
		{"zero", "0", 0},
		{"negative rejected", "-5", 0},
		{"garbage rejected", "soon", 0},
		{"http-date rejected (unsupported by design)", "Fri, 08 Aug 2026 12:00:00 GMT", 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			h := http.Header{}
			if tc.val != "" {
				h.Set("Retry-After", tc.val)
			}
			if got := parseRetryAfter(h); got != tc.want {
				t.Fatalf("parseRetryAfter(%q) = %v, want %v", tc.val, got, tc.want)
			}
		})
	}
}
```

**Step 2: Run it, expect FAIL** — `go test ./transport/ -run TestParseRetryAfter -v` → undefined: parseRetryAfter.

**Step 3: Implement.** In `failure/errors.go`, add the field with a doc comment (import `time`):

```go
type APIError struct {
	Status  int
	Message string
	Body    []byte
	// RetryAfter is the server-advertised wait from a Retry-After header,
	// integer-seconds form only. Zero means absent or unparseable.
	RetryAfter time.Duration
}
```

In `transport/client.go`, add:

```go
// parseRetryAfter reads Retry-After's integer-seconds form. The HTTP-date
// form is deliberately unsupported (needs a clock; no provider we bind uses it).
func parseRetryAfter(h http.Header) time.Duration {
	secs, err := strconv.Atoi(strings.TrimSpace(h.Get("Retry-After")))
	if err != nil || secs <= 0 {
		return 0
	}
	return time.Duration(secs) * time.Second
}
```

and extend BOTH construction sites (Invoke ~:248, Stream ~:285):

```go
return nil, &failure.APIError{Status: httpResp.StatusCode, Message: string(body), Body: body, RetryAfter: parseRetryAfter(httpResp.Header)}
```

**Step 4: Run** `go test ./transport/ ./failure/ -v` → PASS. Then `go test ./...` (whole module) to catch any struct-literal breakage (there is none — all existing literals use field names).

**Step 5: Commit** (in `inference/`): `git add -A && git commit -m "feat(failure): carry Retry-After on APIError from transport"`

---

### Task 2: `retry` package — Policy and fail-closed construction

**Files:**
- Create: `inference/retry/retry.go`
- Test: `inference/retry/retry_test.go` (internal, `package retry` — tests need seam fields later)

**Step 1: Failing test**

```go
package retry

import (
	"errors"
	"testing"
	"time"
)

func validPolicy() Policy {
	return Policy{StableRetries: 3, StableDelay: 2 * time.Second, MaxAttempts: 6, MaxDelay: 30 * time.Second}
}

func TestNew_Validation(t *testing.T) {
	t.Parallel()
	if _, err := New(nil, validPolicy()); err == nil {
		t.Fatal("nil client accepted")
	}
	bad := []Policy{
		{},                                    // zero value invalid
		{StableRetries: -1, StableDelay: time.Second, MaxAttempts: 2, MaxDelay: time.Second},
		{StableRetries: 0, StableDelay: 0, MaxAttempts: 2, MaxDelay: time.Second}, // zero delay with... see rule below
		{StableRetries: 1, StableDelay: time.Second, MaxAttempts: 0, MaxDelay: time.Second}, // no attempts
		{StableRetries: 5, StableDelay: time.Second, MaxAttempts: 3, MaxDelay: time.Second}, // stable exceeds attempts
		{StableRetries: 1, StableDelay: time.Second, MaxAttempts: 2, MaxDelay: 0},           // no cap
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
```

`stubClient` is a test double implementing `inference.Client` (define it in this test file; later tasks extend it into a scripted fake — see Task 4).

**Step 2: Run, expect FAIL** (compile error). `go test ./retry/ -v`

**Step 3: Implement** `retry.go`:

```go
// Package retry decorates an inference.Client with bounded, classified
// retry and exponential backoff. It retries Invoke calls and Stream
// establishment only; once a StreamReader is handed out, a mid-stream
// failure is terminal for the wrapper exactly as for the inner client.
package retry

import (
	"context"
	"fmt"
	"math/rand"
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/stream"
)

// Policy is the immutable retry schedule: StableRetries retries at
// StableDelay, then exponential doubling (starting at 2*StableDelay)
// capped at MaxDelay, until MaxAttempts total attempts have been made.
type Policy struct {
	StableRetries int           // retries at fixed StableDelay (agreed: 3)
	StableDelay   time.Duration // agreed: 2s
	MaxAttempts   int           // total attempts including the first (agreed: 6)
	MaxDelay      time.Duration // cap on the exponential leg (agreed: 30s)
}

// Validate reports the first structural defect. The zero value is invalid:
// this package never invents a schedule the caller did not state.
func (p Policy) Validate() error {
	switch {
	case p.MaxAttempts < 1:
		return &ConfigError{Field: "MaxAttempts", Reason: "must be at least 1"}
	case p.StableRetries < 0:
		return &ConfigError{Field: "StableRetries", Reason: "must not be negative"}
	case p.StableRetries >= p.MaxAttempts:
		return &ConfigError{Field: "StableRetries", Reason: "must be less than MaxAttempts (attempt 1 is not a retry)"}
	case p.StableDelay <= 0:
		return &ConfigError{Field: "StableDelay", Reason: "must be positive"}
	case p.MaxDelay < p.StableDelay:
		return &ConfigError{Field: "MaxDelay", Reason: "must be at least StableDelay"}
	}
	return nil
}

// ConfigError reports an invalid retry configuration at construction.
type ConfigError struct {
	Field  string
	Reason string
}

func (e *ConfigError) Error() string {
	return fmt.Sprintf("retry: invalid config: %s %s", e.Field, e.Reason)
}

// Client decorates an inner inference.Client with the Policy's schedule.
type Client struct {
	inner  inference.Client
	policy Policy

	// Test seams; production values set by New.
	after     func(time.Duration) <-chan time.Time
	randFloat func() float64
}

var _ inference.Client = (*Client)(nil)

// New validates policy and returns the decorated client.
func New(inner inference.Client, policy Policy) (*Client, error) {
	if inner == nil {
		return nil, &ConfigError{Field: "inner", Reason: "client must not be nil"}
	}
	if err := policy.Validate(); err != nil {
		return nil, err
	}
	return &Client{inner: inner, policy: policy, after: time.After, randFloat: rand.Float64}, nil
}
```

Add temporary no-op `Invoke`/`Stream` delegating to `c.inner` so `var _ inference.Client` compiles (replaced in Tasks 4/6).

**Step 4: Run** `go test ./retry/ -v` → PASS. `go vet ./retry/`.

**Step 5: Commit**: `git commit -m "feat(retry): package skeleton with fail-closed Policy validation"`

---

### Task 3: classification and the delay schedule (pure functions)

**Files:**
- Create: `inference/retry/classify.go`, `inference/retry/delay.go`
- Test: `inference/retry/classify_test.go`, `inference/retry/delay_test.go`

**Step 1: Failing tests**

```go
// classify_test.go
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
		{"other", errors.New("boom"), false},
	}
	// assert Retryable(tc.err) == tc.want ...
}
```

```go
// delay_test.go — baseDelay is jitter-free; jitter tested separately.
func TestBaseDelay(t *testing.T) {
	t.Parallel()
	p := Policy{StableRetries: 3, StableDelay: 2 * time.Second, MaxAttempts: 8, MaxDelay: 10 * time.Second}
	// attempt = the attempt number that just failed (1-based).
	want := []time.Duration{
		2 * time.Second,  // after attempt 1
		2 * time.Second,  // after attempt 2
		2 * time.Second,  // after attempt 3 — stable leg done
		4 * time.Second,  // after attempt 4
		8 * time.Second,  // after attempt 5
		10 * time.Second, // after attempt 6 — capped
		10 * time.Second, // after attempt 7 — capped
	}
	for i, w := range want {
		if got := baseDelay(p, i+1); got != w {
			t.Fatalf("baseDelay(attempt %d) = %v, want %v", i+1, got, w)
		}
	}
}

func TestJitteredDelay_Bounds(t *testing.T) {
	t.Parallel()
	// randFloat pinned to 0 and 1 must give exactly ±10%.
	// jittered(d, 0.0) == 0.9*d ; jittered(d, 1.0) == 1.1*d
}

func TestNextDelay_RetryAfterOverride(t *testing.T) {
	t.Parallel()
	// APIError{RetryAfter: 60s} beats a 2s schedule slot: nextDelay returns 60s (jitter-free).
	// APIError{RetryAfter: 1s} loses to a 2s slot: schedule (jittered) wins.
	// RetryAfter above the 5-minute safety cap is clamped to 5 minutes.
	// Non-APIError (NetworkError): pure schedule.
}
```

**Step 2: Run, expect FAIL.**

**Step 3: Implement**

```go
// classify.go

// Retryable reports whether err is worth another attempt: a transient
// transport/network failure or a provider status that signals pressure
// (408, 429, 5xx). Context cancellation is never retryable, even when it
// wraps a retryable cause. Exported so other callers (hustleruntime holds a
// private copy of this predicate today) can converge on one definition.
func Retryable(err error) bool {
	if err == nil || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	var netErr *failure.NetworkError
	if errors.As(err, &netErr) {
		return true
	}
	var apiErr *failure.APIError
	if errors.As(err, &apiErr) {
		return apiErr.Status == 408 || apiErr.Status == 429 || apiErr.Status >= 500 && apiErr.Status <= 599
	}
	return false
}
```

```go
// delay.go

// retryAfterCap bounds a server-advertised Retry-After so a pathological
// header cannot park a turn for an hour; context cancellation remains the
// user's escape hatch below this bound.
const retryAfterCap = 5 * time.Minute

// baseDelay is the jitter-free schedule slot after the given 1-based failed
// attempt: StableDelay for the stable leg, then doubling from 2*StableDelay,
// capped at MaxDelay.
func baseDelay(p Policy, attempt int) time.Duration {
	if attempt <= p.StableRetries {
		return p.StableDelay
	}
	d := p.StableDelay
	for i := 0; i < attempt-p.StableRetries; i++ {
		d *= 2
		if d >= p.MaxDelay {
			return p.MaxDelay
		}
	}
	return d
}

// jittered spreads d by ±10% using r in [0,1).
func jittered(d time.Duration, r float64) time.Duration {
	return time.Duration(float64(d) * (0.9 + 0.2*r))
}

// nextDelay combines the jittered schedule slot with any server-advertised
// Retry-After on err: the larger wins (a server saying "60s" must not be
// hammered at 2s; a server saying "1s" doesn't shrink our own pacing).
func nextDelay(p Policy, attempt int, err error, r float64) time.Duration {
	d := jittered(baseDelay(p, attempt), r)
	var apiErr *failure.APIError
	if errors.As(err, &apiErr) && apiErr.RetryAfter > 0 {
		ra := min(apiErr.RetryAfter, retryAfterCap)
		if ra > d {
			return ra
		}
	}
	return d
}
```

**Step 4: Run** `go test ./retry/ -v` → PASS.

**Step 5: Commit**: `git commit -m "feat(retry): classification and delay schedule"`

---### Task 4: `Invoke` retry loop

**Files:**
- Modify: `inference/retry/retry.go`
- Create: `inference/retry/exhausted.go`
- Test: `inference/retry/invoke_test.go`

**Step 1: Failing tests.** Extend the scripted fake:

```go
// scriptedClient returns queued outcomes in order; records call count.
type scriptedClient struct {
	invokeOutcomes []outcome // {resp *inference.Response, err error}
	calls          int
}
```

Tests (all with `after` seam capturing requested delays into a slice and returning a closed channel; `randFloat` pinned to 0.5 so jitter is exact):

1. `TestInvoke_SuccessFirstTry` — one outcome, success: inner called once, no delays, `resp.Attempts == 1`.
2. `TestInvoke_RetriesThenSucceeds` — 2×`APIError{Status: 429}` then success: 3 calls, delays `[2s, 2s]` (jitter 0.5 ⇒ exact), `resp.Attempts == 3`.
3. `TestInvoke_StableThenExponential` — 5 failures then success with `MaxAttempts: 6`: delays `[2s, 2s, 2s, 4s, 8s]`.
4. `TestInvoke_NonRetryableFailsFast` — `APIError{Status: 400}` first: 1 call, error returned as-is (`errors.As` still finds `*failure.APIError`), no delay.
5. `TestInvoke_Exhaustion` — 6×429 with `MaxAttempts: 6`: 6 calls, returns `*ExhaustedError` with `Attempts == 6`, `errors.As` reaches the final `*failure.APIError`.
6. `TestInvoke_RetryAfterOverridesSchedule` — failure carries `RetryAfter: 60 * time.Second`: recorded delay is 60s.
7. `TestInvoke_ContextCancelDuringWait` — `after` returns a never-firing channel; cancel the context from the test; Invoke returns promptly with `context.Canceled`; inner called once.

**Step 2: Run, expect FAIL.**

**Step 3: Implement**

```go
// exhausted.go

// ExhaustedError reports that every permitted attempt failed. Unwrap exposes
// the final attempt's failure so typed inspection (errors.As on
// *failure.APIError / *failure.NetworkError) keeps working.
type ExhaustedError struct {
	Attempts int
	Cause    error
}

func (e *ExhaustedError) Error() string {
	return fmt.Sprintf("retry: %d attempts exhausted: %v", e.Attempts, e.Cause)
}
func (e *ExhaustedError) Unwrap() error { return e.Cause }
```

```go
func (c *Client) Invoke(ctx context.Context, req inference.Request) (*inference.Response, error) {
	resp, attempts, err := c.attempt(ctx, func() (*inference.Response, error) {
		return c.inner.Invoke(ctx, req)
	})
	if err != nil {
		return nil, err
	}
	resp.Attempts = attempts
	return resp, nil
}

// attempt drives the shared ladder for both entry points. T is the
// per-attempt success value (*inference.Response, or the stream reader).
func attemptLoop[T any](ctx context.Context, c *Client, call func() (T, error)) (T, int, error) {
	var zero T
	var lastErr error
	for attempt := 1; ; attempt++ {
		v, err := call()
		if err == nil {
			return v, attempt, nil
		}
		lastErr = err
		if !Retryable(err) {
			return zero, attempt, err // surface verbatim: typed inspection intact
		}
		if attempt >= c.policy.MaxAttempts {
			return zero, attempt, &ExhaustedError{Attempts: attempt, Cause: lastErr}
		}
		select {
		case <-c.after(nextDelay(c.policy, attempt, err, c.randFloat())):
		case <-ctx.Done():
			return zero, attempt, ctx.Err()
		}
	}
}
```

(Method `attempt` can't be generic on the receiver in Go — make `attemptLoop` a package-level generic function taking `c *Client`, as shown; `Invoke` calls `attemptLoop[*inference.Response](ctx, c, ...)`.)

Also add `Attempts int` to `inference.Response` (`inference/client.go:140`) with comment:

```go
	// Attempts is how many attempts produced this response when served
	// through a retrying decorator; 0 means the serving client does not
	// count attempts, 1 means first-try success.
	Attempts int
```

**Step 4: Run** `go test ./retry/ ./... -v` → PASS (module-wide catches the Response change; existing literals use field names, so nothing breaks).

**Step 5: Commit**: `git commit -m "feat(retry): Invoke retry loop with backoff, Retry-After, exhaustion error"`

---

### Task 5: `stream.StreamResult` gains Attempts

**Files:**
- Modify: `inference/stream/result.go:13`
- Test: extend an existing `stream` package result test file (find with `ls inference/stream/*_test.go`) or add `result_attempts_test.go`

**Step 1: Failing test** — construct a reader via `NewStreamReaderWithResult` whose producer returns `StreamResult{Attempts: 3}`; drain to EOF; assert `Result()` carries `Attempts == 3`. (Trivial, but pins the field into the copy semantics of `Result()`.)

**Step 2: FAIL** (unknown field). **Step 3:** add to `StreamResult`:

```go
	// Attempts is how many establishment attempts a retrying decorator made
	// before this stream opened; 0 when the producer does not count.
	Attempts int
```

**Step 4: PASS**, module-wide `go test ./...`.

**Step 5: Commit**: `git commit -m "feat(stream): Attempts field on StreamResult"`

---

### Task 6: `Stream` establishment retry

**Files:**
- Modify: `inference/retry/retry.go`
- Test: `inference/retry/stream_test.go`

**Step 1: Failing tests** (scripted fake gains `streamOutcomes`; a successful outcome builds a real `stream.NewStreamReaderWithResult` over a fixed chunk slice):

1. `TestStream_EstablishmentRetries` — 429, 429, then a reader yielding 2 chunks: 3 `Stream` calls, delays `[2s, 2s]`, returned reader yields both chunks then EOF, and `Result().Attempts == 3`.
2. `TestStream_FirstTryResultAttempts` — success first try: `Result().Attempts == 1`; chunks and Usage from the inner producer pass through unchanged.
3. `TestStream_InnerResultAbsent` — inner reader built with nil producer: wrapper's `Result()` still reports `ok == true` with `Attempts` set but zero Usage/Model? — NO. Decision: when the inner producer has no authoritative result, the wrapper must not fabricate one; it reports absent exactly like the inner reader. Attempts is then observable only on the retrying path via… nothing. Accept: Attempts is best-effort, carried only when the inner stream produces a result. Test asserts `ok == false`.
4. `TestStream_MidStreamErrorNotRetried` — reader whose `Next` fails with `&failure.APIError{Status: 500}` after one chunk: the error surfaces from `Next`, inner `Stream` called exactly once — no re-establishment.
5. `TestStream_NonRetryableFailsFast` / `TestStream_Exhaustion` — mirror the Invoke tests.

**Step 2: Run, expect FAIL.**

**Step 3: Implement**

```go
func (c *Client) Stream(ctx context.Context, req inference.Request) (*stream.StreamReader[content.Chunk], error) {
	inner, attempts, err := attemptLoop[*stream.StreamReader[content.Chunk]](ctx, c, func() (*stream.StreamReader[content.Chunk], error) {
		return c.inner.Stream(ctx, req)
	})
	if err != nil {
		return nil, err
	}
	// Rewrap only to stamp Attempts into the terminal result; Next/Close
	// delegate, so mid-stream semantics are byte-identical to the inner reader.
	return stream.NewStreamReaderWithResult(inner.Next, inner.Close, func() (stream.StreamResult, bool, error) {
		result, ok := inner.Result()
		if !ok {
			return stream.StreamResult{}, false, nil
		}
		result.Attempts = attempts
		return result, true, nil
	}), nil
}
```

Check `Result()`'s actual signature in `inference/stream/stream.go` (bottom of file) before writing the producer — if it returns `(StreamResult, bool)` the above is right; adjust if it also returns an error. Delete the Task-2 temporary delegating stubs.

**Step 4: Run** `go test ./... && go vet ./...` in `inference/` → PASS.

**Step 5: Commit**: `git commit -m "feat(retry): Stream establishment retry with Attempts on the result trailer"`

---

### Task 7: inference module wrap-up

**Step 1:** `cd inference && gofmt -l . && go vet ./... && go test ./...` — all clean/green.
**Step 2:** Reread the design doc's non-goals; confirm nothing crept in (no mid-stream retry, no failover, no harness edits).
**Step 3:** Commit anything outstanding; no tag.

---

### Task 8: coderig wiring

**Files:**
- Modify: `coderig/internal/app/model.go` (the `loadProductionModels` factory closure, ~line 24)
- Test: `coderig/internal/app/model_retry_test.go` (new)

**Step 1: Failing test**

```go
package app

import (
	"testing"

	"github.com/looprig/inference/retry"
)

func TestDefaultRetryPolicy_Valid(t *testing.T) {
	t.Parallel()
	if err := defaultRetryPolicy.Validate(); err != nil {
		t.Fatalf("defaultRetryPolicy invalid: %v", err)
	}
	if defaultRetryPolicy.StableRetries != 3 || defaultRetryPolicy.StableDelay != 2*time.Second {
		t.Fatalf("agreed schedule drifted: %+v", defaultRetryPolicy)
	}
}

func TestNewProductionClient_Wrapped(t *testing.T) {
	t.Parallel()
	// Use any model/key combination auto.New accepts offline (no I/O happens
	// at construction — see auto.New docs). Pick a known-valid catalogue model,
	// e.g. the one existing model.go tests construct; then assert the returned
	// client is a *retry.Client.
	c, err := newProductionClient(validTestModel(t), "test-key")
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := c.(*retry.Client); !ok {
		t.Fatalf("production client not retry-wrapped: %T", c)
	}
}
```

(Reuse whatever helper existing tests in `coderig/internal/app` use to get a valid `model.Model`; check `model_test.go` / `models_config_test.go` first — do not invent a new fixture if one exists.)

**Step 2: Run, expect FAIL.** `cd coderig && go test ./internal/app/ -run 'Retry|Wrapped' -v`

**Step 3: Implement** in `model.go`:

```go
// defaultRetryPolicy is the session-wide inference retry schedule agreed in
// inference/docs/plans/2026-08-08-retry-client-design.md: three stable 2s
// retries, then exponential to a 30s cap, six attempts total.
var defaultRetryPolicy = retry.Policy{
	StableRetries: 3,
	StableDelay:   2 * time.Second,
	MaxAttempts:   6,
	MaxDelay:      30 * time.Second,
}

// newProductionClient builds the concrete provider client and decorates it
// with the default retry schedule.
func newProductionClient(selected model.Model, key auth.APIKey) (inference.Client, error) {
	client, err := auto.New(selected, key)
	if err != nil {
		return nil, err
	}
	return retry.New(client, defaultRetryPolicy)
}
```

and change the `loadProductionModels` closure to `return newProductionClient(selected, key)`.

If `go build` complains that the workspace `inference` lacks `retry`: verify `go.work` uses the local `inference` directory (it does — workspace covers all modules; remember the offline-replace gotcha only applies outside the workspace).

**Step 4: Run** `cd coderig && go test ./... && go vet ./...` → PASS.

**Step 5: Commit** (in `coderig/`): `git commit -m "feat(app): wrap production inference clients with default retry policy"`

---

### Task 9: final verification

1. `cd inference && go test ./... && go vet ./...` → green.
2. `cd coderig && go test ./... && go vet ./...` → green.
3. `cd /Users/ipotter/code/looprig && go build ./...` via `make` if the root Makefile has a build/test target (check `make -n`; otherwise per-module is enough).
4. Confirm commit list: 1 docs commit (already made), ~6 inference commits, 1 coderig commit; none tagged, none pushed unless asked.
