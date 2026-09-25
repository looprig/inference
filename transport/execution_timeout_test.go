package transport_test

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/inference/auth"
	failure "github.com/looprig/inference/failure"
	"github.com/looprig/inference/transport"
)

const (
	shortInvokeCeiling = 50 * time.Millisecond
	lateBy             = 300 * time.Millisecond
)

func lateHandler(delay time.Duration) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		select {
		case <-time.After(delay):
		case <-r.Context().Done():
			return
		}
		switch r.URL.Path {
		case "/invoke":
			_, _ = io.WriteString(w, `{"answer":"late"}`)
		case "/stream":
			_, _ = io.WriteString(w, `{"text":"late-stream"}`+"\n")
		default:
			http.NotFound(w, r)
		}
	}
}

func unlimited() context.Context {
	return transport.WithoutExecutionTimeout(context.Background())
}

func requireCeilingTimeout(t *testing.T, err error) {
	t.Helper()
	var networkErr *failure.NetworkError
	if !errors.As(err, &networkErr) {
		t.Fatalf("err = %v (%T), want *failure.NetworkError", err, err)
	}
	var timeout net.Error
	if !errors.As(err, &timeout) || !timeout.Timeout() {
		t.Fatalf("err = %v, want a client timeout", err)
	}
}

func requireStreamText(t *testing.T, ctx context.Context, c *transport.Client, want string) {
	t.Helper()
	reader, err := c.Stream(ctx, req("late"))
	if err != nil {
		t.Fatalf("Stream error: %v", err)
	}
	defer reader.Close()
	chunk, err := reader.Next()
	if err != nil {
		t.Fatalf("Stream.Next error: %v", err)
	}
	if text, ok := chunk.(*content.TextChunk); !ok || text.Text != want {
		t.Fatalf("chunk = %#v, want TextChunk %q", chunk, want)
	}
}

func TestWithoutExecutionTimeoutOutlastsInvokeCeiling(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(lateHandler(lateBy))
	defer srv.Close()
	c := transport.New(
		transport.Endpoint{BaseURL: srv.URL}, dualPathRouter{}, customCodec{body: `{}`}, auth.None(),
		transport.WithInvokeTimeout(shortInvokeCeiling),
		transport.WithStreamDecoder(ndjsonTextDecoder{}),
	)

	_, err := c.Invoke(context.Background(), req("late"))
	requireCeilingTimeout(t, err)

	resp, err := c.Invoke(unlimited(), req("late"))
	if err != nil {
		t.Fatalf("marked Invoke error: %v", err)
	}
	if got := firstText(t, resp); got != `{"answer":"late"}` {
		t.Fatalf("marked Invoke text = %q", got)
	}
	requireStreamText(t, unlimited(), c, "late-stream")
}

func TestWithoutExecutionTimeoutKeepsTLSRootCAs(t *testing.T) {
	t.Parallel()
	srv := httptest.NewTLSServer(lateHandler(lateBy))
	defer srv.Close()

	roots := x509.NewCertPool()
	roots.AddCert(srv.Certificate())
	trusted := transport.New(
		transport.Endpoint{BaseURL: srv.URL}, dualPathRouter{}, customCodec{body: `{}`}, auth.None(),
		transport.WithInvokeTimeout(shortInvokeCeiling),
		transport.WithTLSRootCAs(roots),
		transport.WithStreamDecoder(ndjsonTextDecoder{}),
	)
	_, err := trusted.Invoke(context.Background(), req("late"))
	requireCeilingTimeout(t, err)
	if _, err := trusted.Invoke(unlimited(), req("late")); err != nil {
		t.Fatalf("marked Invoke over trusted TLS error: %v", err)
	}
	requireStreamText(t, unlimited(), trusted, "late-stream")

	unrelated := x509.NewCertPool()
	unrelated.AddCert(&x509.Certificate{Raw: []byte{1}, RawSubject: []byte{1}})
	untrusted := transport.New(
		transport.Endpoint{BaseURL: srv.URL}, dualPathRouter{}, customCodec{body: `{}`}, auth.None(),
		transport.WithTLSRootCAs(unrelated),
	)
	_, err = untrusted.Invoke(unlimited(), req("late"))
	var verifyErr *tls.CertificateVerificationError
	if !errors.As(err, &verifyErr) {
		t.Fatalf("marked Invoke against an untrusted certificate: err = %v, want *tls.CertificateVerificationError", err)
	}
}

func TestWithoutExecutionTimeoutKeepsCustomRoundTripper(t *testing.T) {
	t.Parallel()
	srv := httptest.NewTLSServer(lateHandler(lateBy))
	defer srv.Close()

	var calls atomic.Int32
	base := srv.Client().Transport
	rt := roundTripperFunc(func(r *http.Request) (*http.Response, error) {
		calls.Add(1)
		return base.RoundTrip(r)
	})
	unrelated := x509.NewCertPool()
	unrelated.AddCert(&x509.Certificate{Raw: []byte{1}, RawSubject: []byte{1}})
	c := transport.New(
		transport.Endpoint{BaseURL: srv.URL}, dualPathRouter{}, customCodec{body: `{}`}, auth.None(),
		transport.WithInvokeTimeout(shortInvokeCeiling),
		transport.WithRoundTripper(rt),
		transport.WithTLSRootCAs(unrelated),
		transport.WithStreamDecoder(ndjsonTextDecoder{}),
	)

	_, err := c.Invoke(context.Background(), req("late"))
	requireCeilingTimeout(t, err)
	if _, err := c.Invoke(unlimited(), req("late")); err != nil {
		t.Fatalf("marked Invoke through caller RoundTripper error: %v", err)
	}
	requireStreamText(t, unlimited(), c, "late-stream")
	if got := calls.Load(); got != 3 {
		t.Fatalf("caller RoundTripper calls = %d, want 3", got)
	}
}

func TestWithoutExecutionTimeoutKeepsCallerCancellation(t *testing.T) {
	t.Parallel()
	started := make(chan struct{}, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		select {
		case started <- struct{}{}:
		default:
		}
		<-r.Context().Done()
	}))
	defer srv.Close()
	c := transport.New(
		transport.Endpoint{BaseURL: srv.URL}, dualPathRouter{}, customCodec{body: `{}`}, auth.None(),
		transport.WithStreamDecoder(ndjsonTextDecoder{}),
	)
	call := func(ctx context.Context, streaming bool) error {
		if streaming {
			reader, err := c.Stream(ctx, req("held"))
			if err == nil {
				_ = reader.Close()
			}
			return err
		}
		_, err := c.Invoke(ctx, req("held"))
		return err
	}

	for _, streaming := range []bool{false, true} {
		name := map[bool]string{false: "invoke", true: "stream"}[streaming]
		t.Run(name+" cancel", func(t *testing.T) {
			ctx, cancel := context.WithCancel(unlimited())
			defer cancel()
			done := make(chan error, 1)
			go func() { done <- call(ctx, streaming) }()
			select {
			case <-started:
			case err := <-done:
				t.Fatalf("returned before reaching the server: %v", err)
			case <-time.After(5 * time.Second):
				t.Fatal("request never reached the server")
			}
			cancel()
			select {
			case err := <-done:
				if !errors.Is(err, context.Canceled) {
					t.Fatalf("err = %v, want context.Canceled", err)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("marked call ignored cancellation")
			}
		})
	}

	t.Run("invoke caller deadline", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(unlimited(), shortInvokeCeiling)
		defer cancel()
		if err := call(ctx, false); !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("err = %v, want context.DeadlineExceeded", err)
		}
	})
}
