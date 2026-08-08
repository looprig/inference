// Package transport is a generic, connection-bound HTTP client for the inference seam.
// It composes an transport.Endpoint (connection identity), an route.Router (request
// routing), an codec.Codec (request encoding + response decoding), an OPTIONAL
// codec.StreamDecoder (streaming), and an auth.Authenticator (credentials)
// into an inference.Client. It owns HTTP mechanics only: it does not hardcode a method,
// a chat path, a Content-Type, a streaming Accept header, or a stream framing such as
// SSE — the Router supplies method+URL+route headers, the encoder supplies the body and
// its headers, and the StreamDecoder owns wire framing.
package transport

import (
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	auth "github.com/looprig/inference/auth"
	codec "github.com/looprig/inference/codec"
	failure "github.com/looprig/inference/failure"
	"github.com/looprig/inference/model"

	route "github.com/looprig/inference/route"
	stream "github.com/looprig/inference/stream"
)

// Client is a connection-bound inference.Client: one Codec x one Endpoint x one Router
// x one Authenticator, with an optional StreamDecoder. It performs the same ordered
// pre-I/O guards for both Invoke and Stream — binding check, then Model.Validate — then
// routes, encodes, authorizes, and executes, mapping transport failures to
// *failure.NetworkError and non-2xx responses to *failure.APIError.
type Client struct {
	ep     Endpoint
	router route.Router
	enc    codec.RequestEncoder
	dec    codec.ResponseDecoder
	stream codec.StreamDecoder // nil ⇒ streaming unsupported
	auth   auth.Authenticator
	// hcInvoke and hcStream are deliberately separate http.Clients: Invoke's
	// response is atomic (read whole via io.ReadAll, nothing partial to protect),
	// so it is safe to bound with a real whole-request Timeout. Stream must never
	// carry a whole-request Timeout — that would abort a long-lived body
	// mid-flight — so it keeps only the connect/TLS/response-header budget below.
	hcInvoke *http.Client
	hcStream *http.Client
}

// Compile-time proof that Client honors the inference.Client contract.
var _ inference.Client = (*Client)(nil)

// RequestBuildError is a failure to CONSTRUCT the outbound request — a router that
// cannot build a route, or a malformed method/URL. It is a request-configuration error,
// kept strictly distinct from *failure.NetworkError (reserved for hc.Do transport
// failures) so errors.As never misclassifies a config bug as a transport fault. Unwrap
// exposes the underlying cause (e.g. a route.MissingModelError or a net/http error).
type RequestBuildError struct {
	Err error
}

func (e *RequestBuildError) Error() string { return "transport: build request: " + e.Err.Error() }
func (e *RequestBuildError) Unwrap() error { return e.Err }

// UnsupportedStreamingError is returned by Stream, before any I/O, when the Client has
// no StreamDecoder (the codec is a plain Codec, not a StreamingCodec, and no decoder was
// injected). Fail-closed and typed so callers can errors.As it. APIFormat records the
// bound endpoint's format for diagnosis.
type UnsupportedStreamingError struct {
	APIFormat model.APIFormat
}

func (e *UnsupportedStreamingError) Error() string {
	return fmt.Sprintf("transport: streaming unsupported for API format %q: no StreamDecoder (codec is not a StreamingCodec and none was injected)", e.APIFormat)
}

// Timeout budget for the risky connection-setup phases of a request, shared by
// both Invoke and Stream. The per-request deadline for the body is the caller's
// context (every I/O call takes ctx) on top of whichever budget below applies.
const (
	dialTimeout           = 10 * time.Second
	tlsHandshakeTimeout   = 10 * time.Second
	expectContinueTimeout = 1 * time.Second
	idleConnTimeout       = 90 * time.Second

	// streamResponseHeaderTimeout bounds only time-to-first-byte for Stream: a
	// stream should start promptly even under load, and once headers arrive the
	// body is free to run long — there is no whole-request Timeout to fight with.
	streamResponseHeaderTimeout = 60 * time.Second

	// defaultInvokeTimeout is Invoke's whole-request ceiling. Unlike Stream,
	// Invoke's response is atomic (read in full before it is usable), so bounding
	// the entire call — not just the header wait — is safe and correct: there is
	// no partial progress a longer wait would risk losing. 60s (Stream's header-only
	// budget) is too short for a loaded local model or a slow cloud minute; this is
	// deliberately more generous, and overridable per Client via WithInvokeTimeout.
	defaultInvokeTimeout = 5 * time.Minute
)

// Option customizes a Client at construction.
type Option func(*Client)

// WithStreamDecoder sets or overrides the Client's StreamDecoder. Use it to enable
// streaming for a plain Codec, or to route streaming through a caller-supplied decoder
// (e.g. an NDJSON- or custom-framer-backed decoder) different from the codec's own.
func WithStreamDecoder(sd codec.StreamDecoder) Option {
	return func(c *Client) { c.stream = sd }
}

// WithInvokeTimeout overrides Invoke's whole-request timeout (default
// defaultInvokeTimeout). It has no effect on Stream, which never carries a
// whole-request timeout. A non-positive d is a no-op — the default (or a prior
// WithInvokeTimeout) is left in place — since a zero or negative timeout would
// mean "never time out" or "always time out", neither of which this option is
// meant to express.
func WithInvokeTimeout(d time.Duration) Option {
	return func(c *Client) {
		if d <= 0 {
			return
		}
		c.hcInvoke = newInvokeHTTPClient(d)
	}
}

// New constructs a Client bound to ep, routing with router, encoding/decoding with
// codec, authenticating with auth. If codec also satisfies codec.StreamDecoder
// (i.e. is a StreamingCodec), it becomes the stream decoder automatically; otherwise
// streaming fails before I/O with *UnsupportedStreamingError unless WithStreamDecoder
// supplies one. router, codec, and auth are required: a nil is a programmer error (the
// explicit "no credentials" value is auth.None()), so New panics rather than sending
// unrouted, unencoded, or silently unauthenticated requests.
func New(ep Endpoint, router route.Router, cdc codec.Codec, authenticator auth.Authenticator, opts ...Option) *Client {
	if router == nil {
		panic("transport.New: router must not be nil")
	}
	if cdc == nil {
		panic("transport.New: codec must not be nil")
	}
	if authenticator == nil {
		panic("transport.New: auth must not be nil; pass auth.None() for no credentials")
	}
	c := &Client{
		ep:       ep,
		router:   router,
		enc:      cdc,
		dec:      cdc,
		auth:     authenticator,
		hcInvoke: newInvokeHTTPClient(defaultInvokeTimeout),
		hcStream: newStreamHTTPClient(),
	}
	// Optional streaming: a StreamingCodec is its own StreamDecoder.
	if sd, ok := cdc.(codec.StreamDecoder); ok {
		c.stream = sd
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// baseTransport builds the http.Transport settings shared by both the Invoke and
// Stream clients: connect/TLS timeout budget, a TLS 1.2 floor, and connection
// pooling. responseHeaderTimeout is the one setting callers vary between the two
// use cases.
func baseTransport(responseHeaderTimeout time.Duration) *http.Transport {
	return &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		DialContext: (&net.Dialer{
			Timeout:   dialTimeout,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		// Default TLS with an explicit floor of 1.2 and no InsecureSkipVerify —
		// server certificates are verified.
		TLSClientConfig:       &tls.Config{MinVersion: tls.VersionTLS12},
		TLSHandshakeTimeout:   tlsHandshakeTimeout,
		ResponseHeaderTimeout: responseHeaderTimeout,
		ExpectContinueTimeout: expectContinueTimeout,
		IdleConnTimeout:       idleConnTimeout,
		ForceAttemptHTTP2:     true,
	}
}

// noRedirect is shared by both clients: a 3xx would replay the single-shot
// EncodedRequest.Body, and the transport must never retry or replay a body — a
// redirect is surfaced as its 3xx response (mapped to *failure.APIError) instead.
func noRedirect(*http.Request, []*http.Request) error {
	return http.ErrUseLastResponse
}

// parseRetryAfter reads Retry-After's integer-seconds form. The HTTP-date
// form is deliberately unsupported (needs a clock; no provider we bind uses it).
func parseRetryAfter(h http.Header) time.Duration {
	secs, err := strconv.Atoi(strings.TrimSpace(h.Get("Retry-After")))
	if err != nil || secs <= 0 {
		return 0
	}
	return time.Duration(secs) * time.Second
}

// newStreamHTTPClient builds the http.Client used by Stream: no whole-request
// Timeout (it would abort a long-lived streaming body mid-flight), just the
// connect/TLS/header timeout budget on the Transport. The body itself is bounded
// only by the caller's context.
func newStreamHTTPClient() *http.Client {
	return &http.Client{
		CheckRedirect: noRedirect,
		Transport:     baseTransport(streamResponseHeaderTimeout),
	}
}

// newInvokeHTTPClient builds the http.Client used by Invoke, bounded by a real
// whole-request Timeout of d. This is safe (unlike for Stream) because Invoke's
// response is atomic: it is read in full via io.ReadAll before it is usable, so
// there is no partial progress a longer wait could risk losing — a flat deadline
// on the entire call is the correct tool, not just a header-arrival guard.
// ResponseHeaderTimeout is left at 0 (unbounded) on this Transport: the outer
// Client.Timeout already covers header wait plus body read, so a second,
// shorter-or-equal timer here would be redundant.
func newInvokeHTTPClient(d time.Duration) *http.Client {
	return &http.Client{
		Timeout:       d,
		CheckRedirect: noRedirect,
		Transport:     baseTransport(0),
	}
}

// Invoke sends a non-streaming request and returns the complete response. Ordered, all
// pre-I/O guards first: (1) binding check, (2) Model.Validate, (3) route + encode +
// build, (4) authorize, (5) do + map errors, (6) decode.
func (c *Client) Invoke(ctx context.Context, req inference.Request) (*inference.Response, error) {
	if err := c.checkBinding(req.Model); err != nil {
		return nil, err
	}
	if err := req.Model.Validate(); err != nil {
		return nil, err
	}
	httpReq, err := c.buildRequest(ctx, req, codec.RequestModeInvoke)
	if err != nil {
		return nil, err
	}
	if err := c.auth.Authorize(ctx, httpReq); err != nil {
		return nil, err
	}
	httpResp, err := c.hcInvoke.Do(httpReq)
	if err != nil {
		return nil, &failure.NetworkError{Err: err}
	}
	defer httpResp.Body.Close()
	body, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, &failure.NetworkError{Err: err}
	}
	// Non-2xx is mapped to an APIError BEFORE the decoder is invoked; the body is
	// drained and closed (deferred) here because it is normal JSON/text, not a stream.
	if httpResp.StatusCode/100 != 2 {
		return nil, &failure.APIError{Status: httpResp.StatusCode, Message: string(body), Body: body, RetryAfter: parseRetryAfter(httpResp.Header)}
	}
	return c.dec.DecodeResponse(body)
}

// Stream sends a streaming request and returns a StreamReader over the decoded chunks.
// Same ordered guards as Invoke, plus: if there is no StreamDecoder it fails before any
// I/O with *UnsupportedStreamingError. On a 2xx it hands the still-open response to the
// StreamDecoder (which owns resp.Body); on a non-2xx it drains and closes the error body
// (error bodies are normal JSON, not a stream) and returns *failure.APIError.
func (c *Client) Stream(ctx context.Context, req inference.Request) (*stream.StreamReader[content.Chunk], error) {
	if err := c.checkBinding(req.Model); err != nil {
		return nil, err
	}
	if err := req.Model.Validate(); err != nil {
		return nil, err
	}
	if c.stream == nil {
		return nil, &UnsupportedStreamingError{APIFormat: c.ep.APIFormat}
	}
	httpReq, err := c.buildRequest(ctx, req, codec.RequestModeStream)
	if err != nil {
		return nil, err
	}
	if err := c.auth.Authorize(ctx, httpReq); err != nil {
		return nil, err
	}
	httpResp, err := c.hcStream.Do(httpReq)
	if err != nil {
		return nil, &failure.NetworkError{Err: err}
	}
	if httpResp.StatusCode/100 != 2 {
		defer httpResp.Body.Close()
		body, readErr := io.ReadAll(httpResp.Body)
		if readErr != nil {
			return nil, &failure.NetworkError{Err: fmt.Errorf("transport: reading error body (status %d): %w", httpResp.StatusCode, readErr)}
		}
		return nil, &failure.APIError{Status: httpResp.StatusCode, Message: string(body), Body: body, RetryAfter: parseRetryAfter(httpResp.Header)}
	}
	reader, err := c.stream.DecodeStream(httpResp)
	if err != nil {
		// Backstop the StreamDecoder body-ownership contract: a compliant decoder already
		// closed resp.Body before returning an error, but a contract-violating third-party
		// decoder might not. A double Close on an http response body is harmless, so close
		// here to guarantee the connection is never leaked on the highest-risk seam.
		_ = httpResp.Body.Close()
		return nil, err
	}
	return reader, nil
}

// checkBinding fails closed when the request's Model names a provider, endpoint, or API
// format that conflicts with the bound Endpoint, before any I/O. Empty request fields
// are wildcards ("use the bound value"), not claims, so only a non-empty field that
// disagrees is a cross-wiring error.
func (c *Client) checkBinding(m model.Model) error {
	providerConflict := m.Provider != "" && m.Provider != c.ep.Provider
	endpointConflict := m.BaseURL != "" && m.BaseURL != c.ep.BaseURL
	formatConflict := m.APIFormat != "" && m.APIFormat != c.ep.APIFormat
	if providerConflict || endpointConflict || formatConflict {
		return &failure.ModelMismatchError{
			BoundProvider:    c.ep.Provider,
			RequestProvider:  m.Provider,
			BoundEndpoint:    c.ep.BaseURL,
			RequestEndpoint:  m.BaseURL,
			BoundAPIFormat:   c.ep.APIFormat,
			RequestAPIFormat: m.APIFormat,
		}
	}
	return nil
}

// buildRequest routes and encodes req into a ctx-bound *http.Request, applying headers
// in order route → encoder (later overrides earlier). Auth headers are applied by the
// caller (Invoke/Stream) after this, so the full precedence is route → encoder → auth.
func (c *Client) buildRequest(ctx context.Context, req inference.Request, mode codec.RequestMode) (*http.Request, error) {
	route, err := c.router.BuildRoute(c.ep.BaseURL, req, mode)
	if err != nil {
		return nil, &RequestBuildError{Err: err}
	}
	enc, err := c.enc.EncodeRequest(req, mode)
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, route.Method, route.URL, enc.Body)
	if err != nil {
		return nil, &RequestBuildError{Err: err}
	}
	// EncodedRequest.Body is single-shot: clear the GetBody that NewRequestWithContext
	// auto-populates for a bytes/strings body so net/http can never rewind and replay it
	// on a connection-reuse retry (which it would for an idempotent method a custom Router
	// might return). The transport must never replay a body; retry policy is the caller's.
	httpReq.GetBody = nil
	applyHeaders(httpReq.Header, route.Header) // route headers first
	applyHeaders(httpReq.Header, enc.Header)   // encoder headers override route
	return httpReq, nil
}

// applyHeaders copies every key from src into dst, replacing any existing values for
// that key so a later layer overrides an earlier one while preserving multi-valued
// headers. A nil src is a no-op.
func applyHeaders(dst, src http.Header) {
	for k, vals := range src {
		dst.Del(k)
		for _, v := range vals {
			dst.Add(k, v)
		}
	}
}
