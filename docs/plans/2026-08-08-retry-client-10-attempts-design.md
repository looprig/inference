# Retry Client Ten-Attempt Policy Design

Date: 2026-08-08
Status: approved

## Goal

Increase carbon's production retry budget from six total attempts to ten,
while preserving the existing retry classification, jitter, cancellation,
and stream-establishment semantics.

## Design

Keep the generic `inference/retry.Policy` API unchanged. Update only the
production policy used by carbon's model loader:

- three stable retries at 2 seconds;
- exponential delays thereafter, starting at 4 seconds;
- ten total attempts, including the initial attempt;
- a 256-second maximum delay.

This produces waits of 2, 2, 2, 4, 8, 16, 32, 64, and 128 seconds before
the ten attempts. The 256-second value is the cap for a larger future
attempt budget; it is not reached with ten total attempts.

## Testing

Extend the carbon production-policy test to assert `MaxAttempts == 10` and
`MaxDelay == 256*time.Second`. Update the retry design documentation to keep
the recorded schedule and its total-attempt semantics accurate. Run the
affected package tests, then the full inference and carbon test, vet, and
build suites.
