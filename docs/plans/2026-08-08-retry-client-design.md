# Retry client design

Date: 2026-08-08
Status: agreed in discussion; production retry budget revised to ten attempts

## Problem

Nothing in the stack retries a failed inference call. The transport contract
forbids retry at the HTTP layer ("retry policy is the caller's",
`transport/client.go`), provider clients do one attempt, the gateway Mux
explicitly rejects fallback, and the harness main loop turns any inference
error into a failed turn (`loopruntime/step.go`). The only retry anywhere is
hustleruntime's `RetryPolicyClassifiedOnce`: one immediate re-run, no delay.
A single 429 or transient 503 therefore kills an interactive turn.

## Goal

A prebuilt, provider-neutral retry decorator in the inference module that any
consumer can wrap around any `inference.Client`, giving every loop turn,
subagent, and hustle automatic retry with exponential backoff — with no change
to the `inference.Client` interface and no change to harness wiring.

## Explicit non-goals (punted by decision, in order discussed)

- **Fallback model chains / failover targets** — cut. One client, one model.
- **Sticky failover / circuit breaker** — moot without failover.
- **Served-by-model attribution** — moot without failover.
- **Mid-stream restart (reset chunk)** — punted. A stream that dies after
  delivering chunks stays terminal, exactly as today. The retry ladder covers
  `Invoke` and `Stream` establishment only. The reset-chunk design (new chunk
  in core's content vocabulary; chunkProcessor discards accumulated blocks)
  was discussed and is the agreed shape *if* mid-stream restart is ever
  wanted; it depends on the standing harness invariant that nothing acts on
  partial content before clean EOF (`step.go` executes tools only after
  `io.EOF`).

## Design

### Package

New leaf subpackage in the inference module: `inference/retry` (name not
final; no strong opinions were held). Depends only on `inference`,
`inference/failure`, `inference/stream`.

```go
func New(inner inference.Client, policy Policy) (inference.Client, error)
```

The returned value implements `inference.Client` unchanged. Harness binding
is untouched: `loop.WithInference(retryClient, model)`.

### Policy — production schedule: 3 stable retries of 2s, then exponential

```go
type Policy struct {
    StableRetries int           // retries at fixed StableDelay first (agreed: 3)
    StableDelay   time.Duration // agreed: 2s
    MaxAttempts   int           // total attempts incl. the first (production: 10)
    MaxDelay      time.Duration // cap on the exponential leg (production: 256s)
}
```

Timeline with the production values: attempt → 2s → attempt → 2s → attempt
→ 2s → attempt → 4s → attempt → 8s → attempt → 16s → attempt → 32s
→ attempt → 64s → attempt → 128s → attempt → give up. The exponential leg
starts at `StableDelay * 2` and doubles, capped at `MaxDelay`. Ten total
attempts reach a 128s wait; 256s is the cap if the attempt budget grows.

- **Jitter**: ±10–20% on every delay so parallel loops/subagents don't
  re-slam a provider in lockstep.
- **Retry-After**: when a 429/503 `*failure.APIError` carries a parseable
  Retry-After, use `max(scheduleDelay, retryAfter)`. (Requires the header to
  be reachable from the typed error — if it is not on `APIError` today,
  carrying it there is in scope.)
- **Cancellation**: every wait selects on `ctx.Done()`; user cancellation
  aborts the sleep immediately and returns `ctx.Err()`.
- Zero-value `Policy` is invalid; `New` validates (fail-closed, matching
  module style: typed config error, no silent defaults).

### Classification — reuse the existing taxonomy

Retryable, exactly the predicate hustleruntime already applies:

- `*failure.NetworkError`
- `*failure.APIError` with status 408, 429, or 5xx
- `context.Canceled` / `DeadlineExceeded` are never retried

Everything else (400/401/403, validation, codec errors) surfaces on the
first attempt. The predicate lives in `inference/retry` as an exported
`Retryable(error) bool` so hustleruntime can converge on it later instead of
keeping its private copy (convergence itself is out of scope).

### Semantics

- **Invoke**: run the ladder; each attempt re-encodes the request from
  scratch, so the transport's never-replay-a-body contract is untouched.
- **Stream**: the ladder wraps the `Stream()` call — connection and
  first-byte failures (where 429s/503s land) retry invisibly; the caller
  receives a reader only from a successfully established stream. Once the
  reader is handed out, the wrapper is out of the picture: a `Next()` error
  is terminal (non-goal above).
- **Exhaustion**: return a typed error wrapping the last attempt's failure
  and carrying the attempt count, so callers keep `errors.As` access to the
  underlying `*failure.APIError`/`*NetworkError`.

### Observability (minimal)

`inference.Response` and `stream.StreamResult` each gain an
`Attempts int` field (0/1 = no retry), filled by the wrapper. This is the
channel by which the harness *can* record "this turn needed N attempts";
actually surfacing it in harness events/TUI is a follow-up, not in scope.

### Wiring

carbon's model loader (`carbon/internal/app/model.go`) wraps the client it
gets from `auto.New` with `retry.New(client, defaultPolicy)`. No
`models.json` schema change: the policy is code-level default for now. No
`llm/auto` helper is needed — that only existed for multi-target chains.

## Testing

- Unit tests in `inference/retry` with a scripted fake `inference.Client`:
  ladder timing (fake clock), stable→exponential transition, MaxDelay cap,
  jitter bounds, Retry-After override, non-retryable fast-fail,
  context-cancel during sleep, exhaustion error unwrapping, Attempts fields.
- A `Stream` test proving establishment retries are invisible and that a
  post-establishment `Next()` error is not retried.
- carbon: loader test that constructed clients are wrapped and the production
  policy remains at ten total attempts with a 256s cap.

## Future work (recorded, not planned)

Failover chains + sticky cooldown + served-by attribution; mid-stream
restart via reset chunk; hustleruntime converging on `retry.Retryable`;
harness surfacing of `Attempts`.
