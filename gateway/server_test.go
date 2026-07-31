package gateway_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/looprig/inference"
	"github.com/looprig/inference/codec"
	"github.com/looprig/inference/codec/anthropicapi"
	"github.com/looprig/inference/gateway"
	"github.com/looprig/inference/model"
)

// --- ACP-owned ModelProxy structural contract -------------------------------
//
// modelProxy mirrors the ACP-owned ModelProxy contract (see
// acp/docs/connectors/inference-gateway.md's ProxyBinding/ModelProxy
// sketch), defined purely locally, in this test file, as a compile-time
// structural-satisfaction check. inference/gateway must never import
// anything ACP-related -- this interface exists only so a build failure
// here would immediately flag a signature drift between gateway.Server and
// what ACP's connector package expects to consume structurally.
type modelProxy interface {
	Start(context.Context) error
	Binding() (string, string, bool)
	Close(context.Context) error
}

var _ modelProxy = (*gateway.Server)(nil)

// --- shared test doubles and helpers -----------------------------------------

// okHandler is a trivial http.Handler that always responds 200 OK, used by
// every test in this file that only cares about Server's own lifecycle,
// binding, and auth-wrapping behavior -- not about anything a real
// gateway.Handler would add.
func okHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
}

// allowAllAuthenticator is a permissive gateway.Authenticator double used
// only by the routing-isolation test below, so that test exercises Server's
// own token-wrapping and a real gateway.Handler's routing without also
// having to thread the Server's randomly generated token through as the
// inner Handler's configured static token (Config.Authenticate is
// independent of, and unaware of, whatever Server wraps it with).
type allowAllAuthenticator struct{}

func (allowAllAuthenticator) Authenticate(*http.Request) error { return nil }

// newTestServer builds a *gateway.Server wrapping h with test defaults and
// registers a t.Cleanup that closes it -- idempotently safe even if the
// test itself already called Close.
func newTestServer(t *testing.T, h http.Handler) *gateway.Server {
	t.Helper()
	srv, err := gateway.NewServer(gateway.ServerConfig{Handler: h})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	t.Cleanup(func() { _ = srv.Close(context.Background()) })
	return srv
}

func mustStart(t *testing.T, srv *gateway.Server) {
	t.Helper()
	if err := srv.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
}

// doGet issues an unauthenticated-or-authenticated GET / against baseURL,
// returning the response status code. An empty token omits the
// Authorization header entirely.
func doGet(t *testing.T, baseURL, token string) int {
	t.Helper()
	req, err := http.NewRequest(http.MethodGet, baseURL+"/", nil)
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)
	return resp.StatusCode
}

// newRoutingHandler builds a real *gateway.Handler with exactly one route
// (Anthropic ingress, alias "primary") to client/upstreamModel, and a
// permissive inner Authenticator (see allowAllAuthenticator) -- reused by
// the two-Server distinct-routing and shared-handler tests below.
func newRoutingHandler(t *testing.T, upstreamModel model.Model, client inference.Client) *gateway.Handler {
	t.Helper()
	target := gateway.Target{ID: "t", Client: client, Model: upstreamModel}
	resolver, err := gateway.NewMux(gateway.Mux{
		Routes: map[gateway.RouteKey]gateway.Target{
			{Ingress: model.APIFormatAnthropic, Model: "primary"}: target,
		},
	})
	if err != nil {
		t.Fatalf("NewMux: %v", err)
	}
	h, err := gateway.New(gateway.Config{
		Resolver:     resolver,
		Codecs:       map[model.APIFormat]codec.ServerCodec{model.APIFormatAnthropic: anthropicapi.Codec{}},
		Authenticate: allowAllAuthenticator{},
	})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	return h
}

// postMessages POSTs validMessagesBody (defined in handler_test.go, same
// gateway_test package) to baseURL+"/v1/messages" with the given bearer
// token, failing the test unless the response is 200 OK.
func postMessages(t *testing.T, baseURL, token string) {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, baseURL+"/v1/messages", strings.NewReader(validMessagesBody))
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200; body: %s", resp.StatusCode, body)
	}
}

// --- NewServer construction validation ---------------------------------------

func TestNewServer_NilHandler_ConfigError(t *testing.T) {
	t.Parallel()
	_, err := gateway.NewServer(gateway.ServerConfig{})
	var cfgErr *gateway.ConfigError
	if !errors.As(err, &cfgErr) {
		t.Fatalf("NewServer(nil Handler) error = %v (%T), want *ConfigError", err, err)
	}
}

func TestNewServer_NegativeShutdownTimeout_ConfigError(t *testing.T) {
	t.Parallel()
	_, err := gateway.NewServer(gateway.ServerConfig{Handler: okHandler(), ShutdownTimeout: -time.Second})
	var cfgErr *gateway.ConfigError
	if !errors.As(err, &cfgErr) {
		t.Fatalf("NewServer(negative ShutdownTimeout) error = %v (%T), want *ConfigError", err, err)
	}
}

// --- Binding readiness boundary ----------------------------------------------

func TestServer_Binding_NotReadyBeforeStart(t *testing.T) {
	t.Parallel()
	srv := newTestServer(t, okHandler())
	baseURL, token, ready := srv.Binding()
	if ready {
		t.Fatal("ready = true before Start")
	}
	if baseURL != "" || token != "" {
		t.Fatalf("Binding() before Start = (%q, %q, %v), want zero values", baseURL, token, ready)
	}
}

func TestServer_Binding_NotReadyAfterClose(t *testing.T) {
	t.Parallel()
	srv := newTestServer(t, okHandler())
	mustStart(t, srv)
	if err := srv.Close(context.Background()); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, _, ready := srv.Binding(); ready {
		t.Fatal("ready = true after Close")
	}
}

// --- Start / Binding / auth happy path ---------------------------------------

func TestServer_StartBindingAndGeneratedTokenAuth(t *testing.T) {
	assertNoGoroutineLeak(t)
	t.Parallel()

	srv := newTestServer(t, okHandler())
	mustStart(t, srv)

	baseURL, token, ready := srv.Binding()
	if !ready {
		t.Fatal("ready = false after Start")
	}
	if !strings.HasPrefix(baseURL, "http://127.0.0.1:") {
		t.Fatalf("baseURL = %q, want http://127.0.0.1:<port>", baseURL)
	}
	if len(token) < 20 {
		t.Fatalf("token = %q, suspiciously short for a crypto/rand-generated secret", token)
	}

	if code := doGet(t, baseURL, token); code != http.StatusOK {
		t.Fatalf("valid token: status = %d, want 200", code)
	}
	if code := doGet(t, baseURL, ""); code != http.StatusUnauthorized {
		t.Fatalf("missing token: status = %d, want 401", code)
	}
	if code := doGet(t, baseURL, "not-the-right-token-at-all-000000000000"); code != http.StatusUnauthorized {
		t.Fatalf("wrong token: status = %d, want 401", code)
	}
}

// --- Two-server isolation -----------------------------------------------------

func TestServer_TwoServers_UniqueURLsTokensAndIsolation(t *testing.T) {
	assertNoGoroutineLeak(t)
	t.Parallel()

	srvA := newTestServer(t, okHandler())
	srvB := newTestServer(t, okHandler())
	mustStart(t, srvA)
	mustStart(t, srvB)

	urlA, tokenA, _ := srvA.Binding()
	urlB, tokenB, _ := srvB.Binding()

	if urlA == urlB {
		t.Fatalf("expected distinct base URLs, both = %q", urlA)
	}
	if tokenA == tokenB {
		t.Fatal("expected distinct tokens across two independent servers")
	}

	// Each server's own token works against itself.
	if code := doGet(t, urlA, tokenA); code != http.StatusOK {
		t.Fatalf("server A with its own token: status = %d, want 200", code)
	}
	if code := doGet(t, urlB, tokenB); code != http.StatusOK {
		t.Fatalf("server B with its own token: status = %d, want 200", code)
	}

	// Cross-token rejection: A's token must not authenticate against B, and
	// vice versa.
	if code := doGet(t, urlB, tokenA); code != http.StatusUnauthorized {
		t.Fatalf("server A's token against server B: status = %d, want 401", code)
	}
	if code := doGet(t, urlA, tokenB); code != http.StatusUnauthorized {
		t.Fatalf("server B's token against server A: status = %d, want 401", code)
	}

	// Closing A must not affect B (no cross-server state).
	if err := srvA.Close(context.Background()); err != nil {
		t.Fatalf("Close server A: %v", err)
	}
	if _, _, ready := srvA.Binding(); ready {
		t.Fatal("server A still ready after its own Close")
	}
	if code := doGet(t, urlB, tokenB); code != http.StatusOK {
		t.Fatalf("server B unreachable after closing server A: status = %d, want 200", code)
	}
}

// --- Distinct model routing per server (real gateway.Handlers) --------------

func TestServer_DistinctModelRouting_TwoGatewayHandlers(t *testing.T) {
	assertNoGoroutineLeak(t)
	t.Parallel()

	clientA := &recordingClient{}
	clientB := &recordingClient{}
	handlerA := newRoutingHandler(t, anthropicModel("model-a"), clientA)
	handlerB := newRoutingHandler(t, anthropicModel("model-b"), clientB)

	srvA := newTestServer(t, handlerA)
	srvB := newTestServer(t, handlerB)
	mustStart(t, srvA)
	mustStart(t, srvB)

	urlA, tokenA, _ := srvA.Binding()
	urlB, tokenB, _ := srvB.Binding()

	postMessages(t, urlA, tokenA)
	if got := clientA.callCount(); got != 1 {
		t.Fatalf("clientA calls = %d, want 1", got)
	}
	if got := clientB.callCount(); got != 0 {
		t.Fatalf("clientB calls = %d, want 0: a request to server A must never reach server B's target", got)
	}

	postMessages(t, urlB, tokenB)
	if got := clientB.callCount(); got != 1 {
		t.Fatalf("clientB calls = %d, want 1", got)
	}
	if got := clientA.callCount(); got != 1 {
		t.Fatalf("clientA calls = %d, want still 1 (unaffected by server B's request)", got)
	}
}

// --- Shared inner handler across multiple Servers ----------------------------

func TestServer_SharedHandler_MultipleServersWrapOneInnerHandler(t *testing.T) {
	assertNoGoroutineLeak(t)
	t.Parallel()

	var calls int32
	shared := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		w.WriteHeader(http.StatusOK)
	})

	srvA := newTestServer(t, shared)
	srvB := newTestServer(t, shared)
	mustStart(t, srvA)
	mustStart(t, srvB)

	urlA, tokenA, _ := srvA.Binding()
	urlB, tokenB, _ := srvB.Binding()

	if code := doGet(t, urlA, tokenA); code != http.StatusOK {
		t.Fatalf("server A: status = %d, want 200", code)
	}
	if code := doGet(t, urlB, tokenB); code != http.StatusOK {
		t.Fatalf("server B: status = %d, want 200", code)
	}
	if got := atomic.LoadInt32(&calls); got != 2 {
		t.Fatalf("shared handler invocation count = %d, want 2", got)
	}

	// Even though the inner http.Handler instance is shared, each Server
	// still enforces its own independently generated token.
	if code := doGet(t, urlA, tokenB); code != http.StatusUnauthorized {
		t.Fatalf("server A with server B's token: status = %d, want 401", code)
	}
}

// --- Close idempotency and lifecycle state errors ----------------------------

func TestServer_Close_Idempotent(t *testing.T) {
	assertNoGoroutineLeak(t)
	t.Parallel()

	srv := newTestServer(t, okHandler())
	mustStart(t, srv)

	err1 := srv.Close(context.Background())
	err2 := srv.Close(context.Background())
	if err1 != nil {
		t.Fatalf("first Close: %v", err1)
	}
	if err2 != nil {
		t.Fatalf("second Close: %v, want nil (idempotent no-op)", err2)
	}
}

func TestServer_CloseUnstartedServer_NoError(t *testing.T) {
	t.Parallel()
	srv, err := gateway.NewServer(gateway.ServerConfig{Handler: okHandler()})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	if err := srv.Close(context.Background()); err != nil {
		t.Fatalf("Close on never-started server = %v, want nil", err)
	}
	if _, _, ready := srv.Binding(); ready {
		t.Fatal("ready = true after closing a never-started server")
	}
}

func TestServer_Start_WhileRunning_ReturnsTypedStateError(t *testing.T) {
	t.Parallel()
	srv := newTestServer(t, okHandler())
	mustStart(t, srv)

	err := srv.Start(context.Background())
	var stateErr *gateway.ServerStateError
	if !errors.As(err, &stateErr) {
		t.Fatalf("second Start error = %v (%T), want *ServerStateError", err, err)
	}
	// Still running: the second Start must not have torn anything down.
	if _, _, ready := srv.Binding(); !ready {
		t.Fatal("server not ready after a rejected concurrent Start attempt")
	}
}

func TestServer_Start_AfterClose_ReturnsTypedStateError(t *testing.T) {
	t.Parallel()
	srv := newTestServer(t, okHandler())
	mustStart(t, srv)
	if err := srv.Close(context.Background()); err != nil {
		t.Fatalf("Close: %v", err)
	}

	err := srv.Start(context.Background())
	var stateErr *gateway.ServerStateError
	if !errors.As(err, &stateErr) {
		t.Fatalf("Start after Close error = %v (%T), want *ServerStateError", err, err)
	}
}
