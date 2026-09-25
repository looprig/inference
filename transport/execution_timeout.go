package transport

import (
	"context"
	"net/http"
)

// noExecutionTimeoutKey is the unexported context key for WithoutExecutionTimeout.
type noExecutionTimeoutKey struct{}

// WithoutExecutionTimeout returns a copy of ctx that marks every Invoke and
// Stream made with it (or with a context derived from it) as having no
// transport-imposed execution ceiling: neither Invoke's whole-request timeout
// (default 5 minutes, WithInvokeTimeout) nor Stream's 60-second response-header
// timeout applies. The call runs on a dedicated HTTP client built alongside the
// normal ones, so the Client's TLS roots, caller-owned RoundTripper, dial and
// TLS-handshake limits, redirect refusal, authorization and response-size
// limits are unchanged. The marker is process-local: nothing is sent on the
// wire.
//
// Cancellation becomes the caller's job. ctx's own cancellation and deadline
// still end the call; a peer that stops responding on a half-open connection
// is detected only by TCP keepalive.
func WithoutExecutionTimeout(ctx context.Context) context.Context {
	return context.WithValue(ctx, noExecutionTimeoutKey{}, true)
}

// executionHTTPClient returns the unlimited client for a marked ctx and normal
// otherwise.
func (c *Client) executionHTTPClient(ctx context.Context, normal *http.Client) *http.Client {
	if unlimited, _ := ctx.Value(noExecutionTimeoutKey{}).(bool); unlimited {
		return c.hcUnlimited
	}
	return normal
}
