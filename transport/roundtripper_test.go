package transport_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/looprig/inference/auth"
	"github.com/looprig/inference/transport"
)

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestWithRoundTripper_UsedForInvokeAndStream(t *testing.T) {
	t.Parallel()

	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/invoke":
			_, _ = io.WriteString(w, `{"answer":"secure"}`)
		case "/stream":
			_, _ = io.WriteString(w, `{"text":"secure-stream"}`+"\n")
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	var calls atomic.Int32
	base := srv.Client().Transport
	rt := roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		calls.Add(1)
		return base.RoundTrip(req)
	})
	client := transport.New(
		transport.Endpoint{BaseURL: srv.URL},
		dualPathRouter{},
		customCodec{body: `{}`},
		auth.None(),
		transport.WithRoundTripper(rt),
		transport.WithStreamDecoder(ndjsonTextDecoder{}),
	)

	if _, err := client.Invoke(context.Background(), req("secure")); err != nil {
		t.Fatalf("Invoke error: %v", err)
	}
	reader, err := client.Stream(context.Background(), req("secure"))
	if err != nil {
		t.Fatalf("Stream error: %v", err)
	}
	defer reader.Close()
	if _, err := reader.Next(); err != nil {
		t.Fatalf("Stream.Next error: %v", err)
	}
	if got := calls.Load(); got != 2 {
		t.Fatalf("custom RoundTripper calls = %d, want 2", got)
	}
}

func TestWithRoundTripperRejectsNil(t *testing.T) {
	t.Parallel()
	defer func() {
		if recover() == nil {
			t.Fatal("WithRoundTripper(nil) did not panic")
		}
	}()
	transport.WithRoundTripper(nil)
}
