package gateway_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec"
	"github.com/looprig/inference/codec/anthropicapi"
	"github.com/looprig/inference/contextcount"
	"github.com/looprig/inference/gateway"
	"github.com/looprig/inference/model"
	"github.com/looprig/inference/stream"
)

// --- shared test doubles and helpers, used by handler_test.go, auth_test.go, and limits_test.go ---

// recordingClient is a minimal inference.Client double that records the
// request it received (in particular, the resolved Model actually sent
// upstream) and returns a canned response or error. If block is non-nil,
// Invoke waits on it (or the request context) before returning, so tests can
// deliberately saturate concurrency admission.
type recordingClient struct {
	mu    sync.Mutex
	Got   inference.Request
	Calls int

	resp  *inference.Response
	err   error
	block <-chan struct{}
}

func (c *recordingClient) Invoke(ctx context.Context, req inference.Request) (*inference.Response, error) {
	c.mu.Lock()
	c.Got = req
	c.Calls++
	c.mu.Unlock()

	if c.block != nil {
		select {
		case <-c.block:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if c.err != nil {
		return nil, c.err
	}
	if c.resp != nil {
		return c.resp, nil
	}
	return &inference.Response{
		Message: &content.AIMessage{
			Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{&content.TextBlock{Text: "ok"}}},
		},
		FinishReason: stream.FinishReasonStop,
	}, nil
}

func (c *recordingClient) Stream(ctx context.Context, req inference.Request) (*stream.StreamReader[content.Chunk], error) {
	return nil, errors.New("recordingClient.Stream: not implemented")
}

func (c *recordingClient) lastRequest() inference.Request {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.Got
}

func (c *recordingClient) callCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.Calls
}

// anthropicModel builds a valid, Anthropic-dialect model.Model for tests.
func anthropicModel(name string, opts ...model.ModelOption) model.Model {
	return model.CustomModel(model.ProviderName("test-provider"), model.APIFormatAnthropic, "", name, opts...)
}

// newHandler builds a ready Handler for tests with sensible defaults: token
// "test-token", one route (Anthropic ingress, alias "primary") to a fresh
// *recordingClient resolving to upstreamModel, and anthropicapi.Codec{} as
// the sole configured codec.
func newHandler(t *testing.T, upstreamModel model.Model, client inference.Client) (*gateway.Handler, gateway.Target) {
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
		Authenticate: gateway.StaticToken("test-token"),
	})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	return h, target
}

func messagesRequest(t *testing.T, token, body string) *http.Request {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	return req
}

const validMessagesBody = `{"model":"primary","max_tokens":16,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`

// --- Step 1: vertical slice -------------------------------------------------

// TestHandler_VerticalSlice_ReplacesModelAndReportsAlias proves the full
// happy path: the harness-supplied alias ("primary") is never sent upstream
// -- Target.Model is sent instead -- and the harness-facing response reports
// the alias back, not the resolved upstream model name.
func TestHandler_VerticalSlice_ReplacesModelAndReportsAlias(t *testing.T) {
	t.Parallel()
	client := &recordingClient{}
	upstream := anthropicModel("kimi-k2")
	h, _ := newHandler(t, upstream, client)

	req := messagesRequest(t, "test-token", validMessagesBody)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rr.Code, rr.Body.String())
	}
	if got := client.lastRequest().Model.Name; got != upstream.Name {
		t.Errorf("upstream received Model.Name = %q, want %q", got, upstream.Name)
	}
	if got := client.lastRequest().Model.Name; got == "primary" {
		t.Errorf("upstream received the harness alias %q as the model name -- it must never be sent upstream", got)
	}

	var wire struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &wire); err != nil {
		t.Fatalf("decoding response JSON: %v (body: %s)", err, rr.Body.String())
	}
	if wire.Model != "primary" {
		t.Errorf("response model field = %q, want the harness alias %q", wire.Model, "primary")
	}
}

// --- Step 2: failure categories ---------------------------------------------

func TestHandler_WrongMethod_405(t *testing.T) {
	t.Parallel()
	h, _ := newHandler(t, anthropicModel("kimi-k2"), &recordingClient{})
	req := httptest.NewRequest(http.MethodGet, "/v1/messages", nil)
	req.Header.Set("Authorization", "Bearer test-token")
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want %d; body: %s", rr.Code, http.StatusMethodNotAllowed, rr.Body.String())
	}
}

func TestHandler_WrongPath_404(t *testing.T) {
	t.Parallel()
	h, _ := newHandler(t, anthropicModel("kimi-k2"), &recordingClient{})
	req := messagesRequest(t, "test-token", validMessagesBody)
	req.URL.Path = "/v1/nonexistent"
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusNotFound {
		t.Errorf("status = %d, want %d; body: %s", rr.Code, http.StatusNotFound, rr.Body.String())
	}
}

func TestHandler_WrongContentType_415(t *testing.T) {
	t.Parallel()
	h, _ := newHandler(t, anthropicModel("kimi-k2"), &recordingClient{})
	req := messagesRequest(t, "test-token", validMessagesBody)
	req.Header.Set("Content-Type", "text/plain")
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusUnsupportedMediaType {
		t.Errorf("status = %d, want %d; body: %s", rr.Code, http.StatusUnsupportedMediaType, rr.Body.String())
	}
}

func TestHandler_MissingToken_401(t *testing.T) {
	t.Parallel()
	h, _ := newHandler(t, anthropicModel("kimi-k2"), &recordingClient{})
	req := messagesRequest(t, "", validMessagesBody)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusUnauthorized {
		t.Errorf("status = %d, want %d; body: %s", rr.Code, http.StatusUnauthorized, rr.Body.String())
	}
}

func TestHandler_WrongToken_401(t *testing.T) {
	t.Parallel()
	h, _ := newHandler(t, anthropicModel("kimi-k2"), &recordingClient{})
	req := messagesRequest(t, "not-the-token", validMessagesBody)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusUnauthorized {
		t.Errorf("status = %d, want %d; body: %s", rr.Code, http.StatusUnauthorized, rr.Body.String())
	}
}

// TestHandler_AmbiguousCodecMatch_500 constructs a deliberately pathological
// two-codec configuration where both codecs' MatchRequest return true for
// every request, proving the request-time ambiguous-match check (500).
func TestHandler_AmbiguousCodecMatch_500(t *testing.T) {
	t.Parallel()
	resolver, err := gateway.Fixed(&recordingClient{}, anthropicModel("kimi-k2"))
	if err != nil {
		t.Fatalf("Fixed: %v", err)
	}
	h, err := gateway.New(gateway.Config{
		Resolver: resolver,
		Codecs: map[model.APIFormat]codec.ServerCodec{
			model.APIFormatAnthropic: anthropicapi.Codec{},
			model.APIFormatOpenAI:    matchAnythingCodec{},
		},
		Authenticate: gateway.StaticToken("test-token"),
	})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	req := messagesRequest(t, "test-token", validMessagesBody)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusInternalServerError {
		t.Errorf("status = %d, want %d; body: %s", rr.Code, http.StatusInternalServerError, rr.Body.String())
	}
}

// matchAnythingCodec is a pathological codec.ServerCodec test double whose
// MatchRequest always returns true, used only to trigger ambiguous-match
// detection. Its other methods are never expected to be called.
type matchAnythingCodec struct{}

func (matchAnythingCodec) MatchRequest(*http.Request) bool { return true }
func (matchAnythingCodec) DecodeRequest(*http.Request) (codec.DecodedRequest, error) {
	return codec.DecodedRequest{}, errors.New("matchAnythingCodec.DecodeRequest: not implemented")
}
func (matchAnythingCodec) WriteResponse(http.ResponseWriter, *inference.Response) error {
	return errors.New("matchAnythingCodec.WriteResponse: not implemented")
}
func (matchAnythingCodec) OpenStream(http.ResponseWriter) (codec.StreamEncoder, error) {
	return nil, errors.New("matchAnythingCodec.OpenStream: not implemented")
}
func (matchAnythingCodec) WriteError(http.ResponseWriter, error) {}

func TestHandler_NoRouteForRequestedModel_404(t *testing.T) {
	t.Parallel()
	h, _ := newHandler(t, anthropicModel("kimi-k2"), &recordingClient{})
	body := `{"model":"no-such-alias","max_tokens":16,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`
	req := messagesRequest(t, "test-token", body)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusNotFound {
		t.Errorf("status = %d, want %d; body: %s", rr.Code, http.StatusNotFound, rr.Body.String())
	}
}

// TestHandler_UnsupportedFeature_400 sends a request whose feature set
// (an image block) does not match the resolved Target.Model.Caps
// (AcceptsImages == false), proving step 8's post-replacement validation.
func TestHandler_UnsupportedFeature_400(t *testing.T) {
	t.Parallel()
	// upstream model does NOT advertise AcceptsImages.
	h, _ := newHandler(t, anthropicModel("kimi-k2"), &recordingClient{})
	body := `{"model":"primary","max_tokens":16,"messages":[{"role":"user","content":[` +
		`{"type":"image","source":{"type":"url","media_type":"image/png","url":"https://example.test/x.png"}}` +
		`]}]}`
	req := messagesRequest(t, "test-token", body)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want %d; body: %s", rr.Code, http.StatusBadRequest, rr.Body.String())
	}
}

// TestHandler_BodyTooLarge_413 configures a tiny MaxRequestBody and proves a
// larger body is rejected before ever reaching codec decode.
func TestHandler_BodyTooLarge_413(t *testing.T) {
	t.Parallel()
	target := gateway.Target{ID: "t", Client: &recordingClient{}, Model: anthropicModel("kimi-k2")}
	resolver, err := gateway.NewMux(gateway.Mux{
		Routes: map[gateway.RouteKey]gateway.Target{{Ingress: model.APIFormatAnthropic, Model: "primary"}: target},
	})
	if err != nil {
		t.Fatalf("NewMux: %v", err)
	}
	h, err := gateway.New(gateway.Config{
		Resolver:       resolver,
		Codecs:         map[model.APIFormat]codec.ServerCodec{model.APIFormatAnthropic: anthropicapi.Codec{}},
		Authenticate:   gateway.StaticToken("test-token"),
		MaxRequestBody: 16, // far smaller than validMessagesBody
	})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	req := messagesRequest(t, "test-token", validMessagesBody)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusRequestEntityTooLarge {
		t.Errorf("status = %d, want %d; body: %s", rr.Code, http.StatusRequestEntityTooLarge, rr.Body.String())
	}
}

// TestHandler_ConcurrencyAdmission_429 saturates MaxConcurrent with a
// blocked in-flight request, proves a further request is rejected with 429
// while the first is still in flight, and proves admission is released
// (a subsequent request after the first completes succeeds).
func TestHandler_ConcurrencyAdmission_429(t *testing.T) {
	t.Parallel()
	release := make(chan struct{})
	client := &recordingClient{block: release}
	target := gateway.Target{ID: "t", Client: client, Model: anthropicModel("kimi-k2")}
	resolver, err := gateway.NewMux(gateway.Mux{
		Routes: map[gateway.RouteKey]gateway.Target{{Ingress: model.APIFormatAnthropic, Model: "primary"}: target},
	})
	if err != nil {
		t.Fatalf("NewMux: %v", err)
	}
	h, err := gateway.New(gateway.Config{
		Resolver:      resolver,
		Codecs:        map[model.APIFormat]codec.ServerCodec{model.APIFormatAnthropic: anthropicapi.Codec{}},
		Authenticate:  gateway.StaticToken("test-token"),
		MaxConcurrent: 1,
	})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}

	done := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		rr := httptest.NewRecorder()
		h.ServeHTTP(rr, messagesRequest(t, "test-token", validMessagesBody))
		done <- rr
	}()

	// Wait for the first request to actually be in-flight (blocked in Invoke)
	// before firing the second.
	deadline := time.After(2 * time.Second)
	for client.callCount() == 0 {
		select {
		case <-deadline:
			t.Fatal("first request never reached Invoke")
		case <-time.After(time.Millisecond):
		}
	}

	rr2 := httptest.NewRecorder()
	h.ServeHTTP(rr2, messagesRequest(t, "test-token", validMessagesBody))
	if rr2.Code != http.StatusTooManyRequests {
		t.Errorf("second (saturated) request status = %d, want %d; body: %s", rr2.Code, http.StatusTooManyRequests, rr2.Body.String())
	}

	close(release)
	rr1 := <-done
	if rr1.Code != http.StatusOK {
		t.Errorf("first (in-flight) request status = %d, want %d; body: %s", rr1.Code, http.StatusOK, rr1.Body.String())
	}

	// Admission was released: a third request now succeeds instead of 429.
	rr3 := httptest.NewRecorder()
	h.ServeHTTP(rr3, messagesRequest(t, "test-token", validMessagesBody))
	if rr3.Code != http.StatusOK {
		t.Errorf("post-release request status = %d, want %d (admission not released?); body: %s", rr3.Code, http.StatusOK, rr3.Body.String())
	}
}

// TestHandler_UpstreamInvocationError_502 proves an upstream Invoke error is
// classified as a 502 by default, and that the upstream error's own message
// (which may contain provider-secret material this gateway does not control)
// never appears in the HTTP response body.
func TestHandler_UpstreamInvocationError_502(t *testing.T) {
	t.Parallel()
	const secretSubstring = "sk-upstream-secret-should-never-leak"
	client := &recordingClient{err: errors.New("upstream failed: " + secretSubstring)}
	h, _ := newHandler(t, anthropicModel("kimi-k2"), client)

	req := messagesRequest(t, "test-token", validMessagesBody)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusBadGateway {
		t.Errorf("status = %d, want %d; body: %s", rr.Code, http.StatusBadGateway, rr.Body.String())
	}
	if strings.Contains(rr.Body.String(), secretSubstring) {
		t.Errorf("response body leaked upstream error detail: %s", rr.Body.String())
	}
}

// TestHandler_UpstreamDeadlineExceeded_504 proves a context deadline
// exceeded while invoking Target.Client classifies as 504.
func TestHandler_UpstreamDeadlineExceeded_504(t *testing.T) {
	t.Parallel()
	block := make(chan struct{}) // never closed: Invoke blocks until ctx times out
	client := &recordingClient{block: block}
	h, _ := newHandler(t, anthropicModel("kimi-k2"), client)

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	req := messagesRequest(t, "test-token", validMessagesBody).WithContext(ctx)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusGatewayTimeout {
		t.Errorf("status = %d, want %d; body: %s", rr.Code, http.StatusGatewayTimeout, rr.Body.String())
	}
}

// TestHandler_RedactedDiagnostics_AuthFailureBodyOmitsSuppliedToken proves an
// auth failure response never echoes the caller-supplied (wrong) token back.
func TestHandler_RedactedDiagnostics_AuthFailureBodyOmitsSuppliedToken(t *testing.T) {
	t.Parallel()
	h, _ := newHandler(t, anthropicModel("kimi-k2"), &recordingClient{})
	const suppliedWrongToken = "totally-wrong-token-value-xyz"
	req := messagesRequest(t, suppliedWrongToken, validMessagesBody)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want %d", rr.Code, http.StatusUnauthorized)
	}
	if strings.Contains(rr.Body.String(), suppliedWrongToken) {
		t.Errorf("response body echoed the supplied wrong token: %s", rr.Body.String())
	}
	if strings.Contains(rr.Body.String(), "test-token") {
		t.Errorf("response body leaked the configured real token: %s", rr.Body.String())
	}
}

// --- count_tokens auxiliary route -------------------------------------------

// countingResolverHandler builds a Handler configured with counter as its
// ContextCounter (nil is a valid, explicit "unavailable" configuration).
func countingResolverHandler(t *testing.T, client *recordingClient, counter contextcount.ContextCounter) *gateway.Handler {
	t.Helper()
	target := gateway.Target{ID: "t", Client: client, Model: anthropicModel("kimi-k2")}
	resolver, err := gateway.NewMux(gateway.Mux{
		Routes: map[gateway.RouteKey]gateway.Target{{Ingress: model.APIFormatAnthropic, Model: "primary"}: target},
	})
	if err != nil {
		t.Fatalf("NewMux: %v", err)
	}
	h, err := gateway.New(gateway.Config{
		Resolver:       resolver,
		Codecs:         map[model.APIFormat]codec.ServerCodec{model.APIFormatAnthropic: anthropicapi.Codec{}},
		Authenticate:   gateway.StaticToken("test-token"),
		ContextCounter: counter,
	})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	return h
}

// TestHandlerCountTokens_Success proves the count_tokens route resolves and
// replaces the model exactly like normal inference, calls the configured
// ContextCounter (never Target.Client), and writes Anthropic's native
// {"input_tokens": N} response shape.
func TestHandlerCountTokens_Success(t *testing.T) {
	t.Parallel()
	client := &recordingClient{}
	var gotModelName string
	counter := contextcount.ContextCounterFunc{
		Count: func(ctx context.Context, req inference.Request) (contextcount.ContextCount, error) {
			gotModelName = req.Model.Name
			return contextcount.ContextCount{
				Model:       req.Model.Key(),
				InputTokens: 42,
				Quality:     contextcount.CountQualityHeuristicEstimate,
			}, nil
		},
		Capability: contextcount.CounterCapability{
			Transport: contextcount.CounterTransportLocal,
			Quality:   contextcount.CountQualityHeuristicEstimate,
		},
	}
	h := countingResolverHandler(t, client, counter)

	req := httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", strings.NewReader(validMessagesBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer test-token")
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body: %s", rr.Code, rr.Body.String())
	}
	var wire struct {
		InputTokens int `json:"input_tokens"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &wire); err != nil {
		t.Fatalf("decoding response JSON: %v (body: %s)", err, rr.Body.String())
	}
	if wire.InputTokens != 42 {
		t.Errorf("input_tokens = %d, want 42", wire.InputTokens)
	}
	if gotModelName != "kimi-k2" {
		t.Errorf("ContextCounter received Model.Name = %q, want resolved upstream name %q (not the alias)", gotModelName, "kimi-k2")
	}
	if calls := client.callCount(); calls != 0 {
		t.Errorf("Target.Client.Invoke was called %d times for a count_tokens request; it must never be called", calls)
	}
}

// TestHandlerCountTokens_NilContextCounter_503 proves that a nil
// Config.ContextCounter fails cleanly (no panic) with a typed 503, rather
// than panicking on a nil interface call.
func TestHandlerCountTokens_NilContextCounter_503(t *testing.T) {
	t.Parallel()
	h := countingResolverHandler(t, &recordingClient{}, nil)

	req := httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", strings.NewReader(validMessagesBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer test-token")
	rr := httptest.NewRecorder()

	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("ServeHTTP panicked with a nil ContextCounter: %v", r)
			}
		}()
		h.ServeHTTP(rr, req)
	}()

	if rr.Code != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want %d; body: %s", rr.Code, http.StatusServiceUnavailable, rr.Body.String())
	}
}

// --- misc sanity -------------------------------------------------------

func TestHandler_ImplementsHTTPHandler(t *testing.T) {
	t.Parallel()
	var _ http.Handler = (*gateway.Handler)(nil)
}
