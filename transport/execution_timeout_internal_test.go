package transport

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"net/http"
	"testing"
	"time"

	"github.com/looprig/inference/auth"
	"github.com/looprig/inference/codec/openaiapi"
	"github.com/looprig/inference/route"
)

func newSelectionTestClient(opts ...Option) *Client {
	return New(Endpoint{BaseURL: "http://127.0.0.1:1/v1"}, route.StaticChat("/chat/completions"), openaiapi.Codec{}, auth.None(), opts...)
}

func TestExecutionHTTPClientSelection(t *testing.T) {
	t.Parallel()
	c := newSelectionTestClient()

	if got := c.executionHTTPClient(context.Background(), c.hcInvoke); got != c.hcInvoke {
		t.Fatal("unmarked Invoke did not keep hcInvoke")
	}
	if got := c.executionHTTPClient(context.Background(), c.hcStream); got != c.hcStream {
		t.Fatal("unmarked Stream did not keep hcStream")
	}
	if c.hcInvoke.Timeout != defaultInvokeTimeout {
		t.Fatalf("hcInvoke.Timeout = %v, want %v", c.hcInvoke.Timeout, defaultInvokeTimeout)
	}
	if got := c.hcStream.Transport.(*http.Transport).ResponseHeaderTimeout; got != streamResponseHeaderTimeout {
		t.Fatalf("hcStream ResponseHeaderTimeout = %v, want %v", got, streamResponseHeaderTimeout)
	}

	ctx, cancel := context.WithCancel(WithoutExecutionTimeout(context.Background()))
	defer cancel()
	for name, normal := range map[string]*http.Client{"invoke": c.hcInvoke, "stream": c.hcStream} {
		if got := c.executionHTTPClient(ctx, normal); got != c.hcUnlimited {
			t.Fatalf("marked %s did not select hcUnlimited", name)
		}
	}

	u := c.hcUnlimited
	tr, ok := u.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("hcUnlimited.Transport = %T, want *http.Transport", u.Transport)
	}
	if u.Timeout != 0 || tr.ResponseHeaderTimeout != 0 {
		t.Fatalf("hcUnlimited keeps an execution deadline: Timeout=%v ResponseHeaderTimeout=%v", u.Timeout, tr.ResponseHeaderTimeout)
	}
	if tr.TLSHandshakeTimeout != tlsHandshakeTimeout ||
		tr.ExpectContinueTimeout != expectContinueTimeout ||
		tr.IdleConnTimeout != idleConnTimeout ||
		tr.TLSClientConfig == nil || tr.TLSClientConfig.MinVersion != tls.VersionTLS12 ||
		tr.TLSClientConfig.InsecureSkipVerify {
		t.Fatal("hcUnlimited lost a connection-setup or TLS safeguard")
	}
	if u.CheckRedirect == nil {
		t.Fatal("hcUnlimited lost redirect refusal")
	}
}

type identityRoundTripper struct{}

func (*identityRoundTripper) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, http.ErrNotSupported
}

func TestUnlimitedHTTPClientFollowsOptions(t *testing.T) {
	t.Parallel()
	roots := x509.NewCertPool()
	roots.AddCert(&x509.Certificate{Raw: []byte{1}, RawSubject: []byte{1}})

	withRoots := newSelectionTestClient(WithTLSRootCAs(roots))
	tr := withRoots.hcUnlimited.Transport.(*http.Transport)
	if tr.TLSClientConfig.RootCAs == nil || !tr.TLSClientConfig.RootCAs.Equal(roots) {
		t.Fatal("WithTLSRootCAs did not reach hcUnlimited")
	}

	rt := &identityRoundTripper{}
	for name, opts := range map[string][]Option{
		"roundtripper then roots": {WithRoundTripper(rt), WithTLSRootCAs(roots)},
		"roots then roundtripper": {WithTLSRootCAs(roots), WithRoundTripper(rt)},
	} {
		if got := newSelectionTestClient(opts...).hcUnlimited.Transport; got != rt {
			t.Fatalf("%s: hcUnlimited.Transport = %T, want the caller's RoundTripper", name, got)
		}
	}

	if got := newSelectionTestClient(WithInvokeTimeout(time.Second)).hcUnlimited.Timeout; got != 0 {
		t.Fatalf("WithInvokeTimeout bounded hcUnlimited: Timeout = %v", got)
	}
}
