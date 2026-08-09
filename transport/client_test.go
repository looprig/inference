package transport_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/credentials/httpauth"
	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	failure "github.com/looprig/inference/failure"

	codec "github.com/looprig/inference/codec"
	"github.com/looprig/inference/codec/anthropicapi"
	"github.com/looprig/inference/codec/geminiapi"
	"github.com/looprig/inference/codec/openaiapi"
	model "github.com/looprig/inference/model"
	"github.com/looprig/inference/route"

	stream "github.com/looprig/inference/stream"
	"github.com/looprig/inference/transport"

	"github.com/looprig/inference/wire/ndjson"
)

// ---- shared test fixtures ----------------------------------------------------

// recorder captures the last request a handler saw, mutex-guarded so -race sees a
// happens-before between the server goroutine's write and the test goroutine's read.
type recorder struct {
	mu     sync.Mutex
	method string
	path   string
	query  string
	header http.Header
}

func (r *recorder) record(req *http.Request) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.method = req.Method
	r.path = req.URL.Path
	r.query = req.URL.RawQuery
	r.header = req.Header.Clone()
}

func (r *recorder) snapshot() (method, path, query string, header http.Header) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.method, r.path, r.query, r.header
}

func req(name string) inference.Request {
	return inference.Request{Model: model.Model{Name: name}}
}

func firstText(t *testing.T, resp *inference.Response) string {
	t.Helper()
	if resp == nil || resp.Message == nil {
		t.Fatalf("response or message is nil: %+v", resp)
	}
	for _, b := range resp.Message.Blocks {
		if tb, ok := b.(*content.TextBlock); ok {
			return tb.Text
		}
	}
	t.Fatalf("no text block in response: %+v", resp.Message.Blocks)
	return ""
}

// staticRouter is a caller-supplied Router returning a fixed method/path plus optional
// route headers — proves the transport routes only via the injected Router.
type staticRouter struct {
	method string
	path   string
	header http.Header
}

func (r staticRouter) BuildRoute(base string, _ inference.Request, _ codec.RequestMode) (route.Route, error) {
	return route.Route{Method: r.method, URL: strings.TrimRight(base, "/") + r.path, Header: r.header}, nil
}

// customCodec is a caller-supplied Codec (RequestEncoder + ResponseDecoder) that does
// NOT implement StreamDecoder — used for the custom-API, header-precedence, optional-
// streaming, and non-2xx tests.
type customCodec struct {
	body      string      // request body to send
	encHeader http.Header // encoder headers to apply
	decode    func([]byte) (*inference.Response, error)
}

func (c customCodec) EncodeRequest(_ inference.Request, _ codec.RequestMode) (codec.EncodedRequest, error) {
	h := http.Header{}
	h.Set("Content-Type", "application/json")
	for k, vals := range c.encHeader {
		h.Del(k)
		for _, v := range vals {
			h.Add(k, v)
		}
	}
	return codec.EncodedRequest{Header: h, Body: strings.NewReader(c.body)}, nil
}

func (c customCodec) DecodeResponse(body []byte) (*inference.Response, error) {
	if c.decode != nil {
		return c.decode(body)
	}
	return &inference.Response{
		Model: "custom",
		Message: &content.AIMessage{Message: content.Message{
			Role:   content.RoleAssistant,
			Blocks: []content.Block{&content.TextBlock{Text: string(body)}},
		}},
	}, nil
}

// setHeaderAuth is a caller-supplied Authenticator that sets one header — used to prove
// auth is applied last in the header precedence chain.
type setHeaderAuth struct{ name, value string }

func (a setHeaderAuth) Authorize(_ context.Context, r *http.Request) error {
	r.Header.Set(a.name, a.value)
	return nil
}

// ndjsonTextDecoder is a caller-supplied StreamDecoder that frames a body as NDJSON and
// maps each {"text":...} line to a TextChunk — proves the transport does not assume SSE.
type ndjsonTextDecoder struct{}

func (ndjsonTextDecoder) DecodeStream(resp *http.Response) (*stream.StreamReader[content.Chunk], error) {
	frames, err := ndjson.DecodeStreamFrames(resp.Body)
	if err != nil {
		return nil, err
	}
	return stream.FramesToChunks(frames, func(f stream.StreamFrame) ([]content.Chunk, error) {
		var m struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal(f.Data, &m); err != nil || m.Text == "" {
			return nil, nil
		}
		return []content.Chunk{&content.TextChunk{Text: m.Text}}, nil
	}), nil
}

// ---- 1. Routing --------------------------------------------------------------

// TestRouting_InjectedRouter proves the transport routes solely via the injected Router:
// an OpenAI static router and a Gemini model-in-path router both work through the SAME
// transport type, each hitting the right URL and method.
func TestRouting_InjectedRouter(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		router    route.Router
		codec     codec.Codec
		modelName string
		respBody  string
		wantPath  string
		wantQuery string
	}{
		{
			name:      "openai static chat route",
			router:    route.StaticChat("/chat/completions"),
			codec:     openaiapi.Codec{},
			modelName: "gpt-x",
			respBody:  `{"id":"c1","model":"gpt-x","choices":[{"message":{"role":"assistant","content":"ok"}}]}`,
			wantPath:  "/chat/completions",
			wantQuery: "",
		},
		{
			name:      "gemini model-in-path route",
			router:    route.GeminiGenerateContent(),
			codec:     geminiapi.Codec{},
			modelName: "gemini-1.5",
			respBody:  `{"candidates":[{"content":{"parts":[{"text":"ok"}],"role":"model"}}]}`,
			wantPath:  "/models/gemini-1.5:generateContent",
			wantQuery: "",
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			rec := &recorder{}
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				rec.record(r)
				_, _ = io.WriteString(w, tc.respBody)
			}))
			defer srv.Close()

			c := transport.New(transport.Endpoint{BaseURL: srv.URL}, tc.router, tc.codec, auth.None())
			if _, err := c.Invoke(context.Background(), req(tc.modelName)); err != nil {
				t.Fatalf("Invoke error: %v", err)
			}
			method, path, query, _ := rec.snapshot()
			if method != http.MethodPost {
				t.Errorf("method = %q, want POST", method)
			}
			if path != tc.wantPath {
				t.Errorf("path = %q, want %q", path, tc.wantPath)
			}
			if query != tc.wantQuery {
				t.Errorf("query = %q, want %q", query, tc.wantQuery)
			}
		})
	}
}

// TestRouting_GeminiStreamQuery proves the Gemini router's stream mode is honored end to
// end: the transport hits :streamGenerateContent?alt=sse.
func TestRouting_GeminiStreamQuery(t *testing.T) {
	t.Parallel()
	rec := &recorder{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec.record(r)
		_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hi\"}],\"role\":\"model\"}}]}\n\n")
	}))
	defer srv.Close()

	c := transport.New(transport.Endpoint{BaseURL: srv.URL}, route.GeminiGenerateContent(), geminiapi.Codec{}, auth.None())
	stream, err := c.Stream(context.Background(), req("gemini-1.5"))
	if err != nil {
		t.Fatalf("Stream error: %v", err)
	}
	defer stream.Close()
	if _, err := stream.Next(); err != nil {
		t.Fatalf("Next error: %v", err)
	}
	_, path, query, _ := rec.snapshot()
	if path != "/models/gemini-1.5:streamGenerateContent" {
		t.Errorf("path = %q, want streamGenerateContent", path)
	}
	if query != "alt=sse" {
		t.Errorf("query = %q, want alt=sse", query)
	}
}

// ---- 2. Header precedence ----------------------------------------------------

// TestHeaderPrecedence proves headers are applied route → encoder → auth, with the last
// layer winning: all three set X-Prec, and the server must observe the auth value.
func TestHeaderPrecedence(t *testing.T) {
	t.Parallel()

	rec := &recorder{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec.record(r)
		_, _ = io.WriteString(w, "{}")
	}))
	defer srv.Close()

	routeHdr := http.Header{}
	routeHdr.Set("X-Prec", "route")
	routeHdr.Set("X-Route-Only", "r")

	encHdr := http.Header{}
	encHdr.Set("X-Prec", "encoder")
	encHdr.Set("X-Enc-Only", "e")

	codec := customCodec{body: "{}", encHeader: encHdr}
	router := staticRouter{method: http.MethodPost, path: "/x", header: routeHdr}

	c := transport.New(transport.Endpoint{BaseURL: srv.URL}, router, codec, setHeaderAuth{name: "X-Prec", value: "auth"})
	if _, err := c.Invoke(context.Background(), req("m")); err != nil {
		t.Fatalf("Invoke error: %v", err)
	}

	_, _, _, hdr := rec.snapshot()
	if got := hdr.Get("X-Prec"); got != "auth" {
		t.Errorf("X-Prec = %q, want auth (route→encoder→auth, auth wins)", got)
	}
	if got := hdr.Get("X-Route-Only"); got != "r" {
		t.Errorf("X-Route-Only = %q, want r (route headers pass through)", got)
	}
	if got := hdr.Get("X-Enc-Only"); got != "e" {
		t.Errorf("X-Enc-Only = %q, want e (encoder headers pass through)", got)
	}
}

func TestCallScopedAuthorizationUsesFreshAuthorizerPerRequest(t *testing.T) {
	t.Parallel()
	var seen []string
	var mu sync.Mutex
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		seen = append(seen, r.Header.Get("Authorization"))
		mu.Unlock()
		_, _ = io.WriteString(w, `{}`)
	}))
	defer srv.Close()

	c := transport.NewWithAuth(transport.Endpoint{BaseURL: srv.URL}, staticRouter{method: http.MethodPost, path: "/x"}, customCodec{body: `{}`})
	first := setHeaderAuth{name: "Authorization", value: "Bearer first"}
	second := setHeaderAuth{name: "Authorization", value: "Bearer second"}
	var _ httpauth.Authorizer = first
	if _, err := c.InvokeWithAuth(context.Background(), req("m"), first); err != nil {
		t.Fatalf("first InvokeWithAuth: %v", err)
	}
	if _, err := c.InvokeWithAuth(context.Background(), req("m"), second); err != nil {
		t.Fatalf("second InvokeWithAuth: %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if want := []string{"Bearer first", "Bearer second"}; !reflect.DeepEqual(seen, want) {
		t.Fatalf("authorization values = %#v, want %#v", seen, want)
	}
}

// ---- 3. Non-2xx mapped before decode -----------------------------------------

// TestNon2xxBeforeDecode proves a 500 JSON error body maps to *failure.APIError BEFORE
// the response/stream decoder is invoked, for both Invoke and Stream. The bounded body is
// parsed transiently and is never retained in APIError.
func TestNon2xxBeforeDecode(t *testing.T) {
	t.Parallel()

	const errBody = `{"error":{"type":"invalid_request_error","message":"provider-secret-boom"}}`
	newSrv := func() *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("X-Request-ID", "req-safe-123")
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = io.WriteString(w, errBody)
		}))
	}

	t.Run("invoke", func(t *testing.T) {
		t.Parallel()
		srv := newSrv()
		defer srv.Close()
		decodeCalled := false
		codec := customCodec{body: "{}", decode: func([]byte) (*inference.Response, error) {
			decodeCalled = true
			return &inference.Response{Model: "x"}, nil
		}}
		c := transport.New(transport.Endpoint{BaseURL: srv.URL}, staticRouter{method: http.MethodPost, path: "/x"}, codec, auth.None())
		_, err := c.Invoke(context.Background(), req("m"))
		var apiErr *failure.APIError
		if !errors.As(err, &apiErr) {
			t.Fatalf("err = %T (%v), want *failure.APIError", err, err)
		}
		if apiErr.Status != http.StatusInternalServerError {
			t.Errorf("APIError.Status = %d, want 500", apiErr.Status)
		}
		if apiErr.Code != "invalid_request_error" {
			t.Errorf("APIError.Code = %q, want invalid_request_error", apiErr.Code)
		}
		if apiErr.RequestID != "req-safe-123" {
			t.Errorf("APIError.RequestID = %q, want req-safe-123", apiErr.RequestID)
		}
		if len(apiErr.Body) != 0 {
			t.Errorf("APIError.Body = %q, want nil (provider body must not be retained)", apiErr.Body)
		}
		if strings.Contains(apiErr.Error(), errBody) {
			t.Errorf("APIError.Error retained raw provider body: %q", apiErr.Error())
		}
		if decodeCalled {
			t.Error("DecodeResponse must NOT be called on a non-2xx response")
		}
	})

	t.Run("stream", func(t *testing.T) {
		t.Parallel()
		srv := newSrv()
		defer srv.Close()
		streamCalled := false
		spy := spyStreamDecoder{called: &streamCalled}
		c := transport.New(transport.Endpoint{BaseURL: srv.URL}, staticRouter{method: http.MethodPost, path: "/x"}, customCodec{body: "{}"}, auth.None(), transport.WithStreamDecoder(spy))
		_, err := c.Stream(context.Background(), req("m"))
		var apiErr *failure.APIError
		if !errors.As(err, &apiErr) {
			t.Fatalf("err = %T (%v), want *failure.APIError", err, err)
		}
		if apiErr.Status != http.StatusInternalServerError {
			t.Errorf("APIError.Status = %d, want 500", apiErr.Status)
		}
		if apiErr.Code != "invalid_request_error" {
			t.Errorf("APIError.Code = %q, want invalid_request_error", apiErr.Code)
		}
		if apiErr.RequestID != "req-safe-123" {
			t.Errorf("APIError.RequestID = %q, want req-safe-123", apiErr.RequestID)
		}
		if len(apiErr.Body) != 0 {
			t.Errorf("APIError.Body = %q, want nil (provider body must not be retained)", apiErr.Body)
		}
		if strings.Contains(apiErr.Error(), errBody) {
			t.Errorf("APIError.Error retained raw provider body: %q", apiErr.Error())
		}
		if streamCalled {
			t.Error("StreamDecoder must NOT be called on a non-2xx response")
		}
	})
}

// spyStreamDecoder records whether DecodeStream was invoked.
type spyStreamDecoder struct{ called *bool }

func (s spyStreamDecoder) DecodeStream(resp *http.Response) (*stream.StreamReader[content.Chunk], error) {
	*s.called = true
	resp.Body.Close()
	return stream.NewStreamReader(func() (content.Chunk, error) { return nil, io.EOF }, nil), nil
}

// ---- 4. Optional streaming ---------------------------------------------------

// TestOptionalStreaming_NoDecoder proves a Codec with no StreamDecoder fails Stream
// before any I/O with *transport.UnsupportedStreamingError — the httptest server fails
// the test if it is ever hit.
func TestOptionalStreaming_NoDecoder(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Error("server must not be hit: streaming should fail before any I/O")
	}))
	defer srv.Close()

	c := transport.New(transport.Endpoint{BaseURL: srv.URL, APIFormat: model.APIFormat("custom")}, staticRouter{method: http.MethodPost, path: "/x"}, customCodec{body: "{}"}, auth.None())
	_, err := c.Stream(context.Background(), req("m"))
	var use *transport.UnsupportedStreamingError
	if !errors.As(err, &use) {
		t.Fatalf("err = %T (%v), want *transport.UnsupportedStreamingError", err, err)
	}
	if use.APIFormat != model.APIFormat("custom") {
		t.Errorf("UnsupportedStreamingError.APIFormat = %q, want custom", use.APIFormat)
	}
}

// ---- 5. No replay ------------------------------------------------------------

// eofCountReader wraps a reader and counts how many times it reports io.EOF, i.e. how
// many times it was fully consumed.
type eofCountReader struct {
	r    io.Reader
	eofs *int
}

func (e eofCountReader) Read(p []byte) (int, error) {
	n, err := e.r.Read(p)
	if errors.Is(err, io.EOF) {
		*e.eofs++
	}
	return n, err
}

// replayCodec returns an opaque (non-replayable) body wrapped to count consumption.
type replayCodec struct{ eofs *int }

func (c replayCodec) EncodeRequest(_ inference.Request, _ codec.RequestMode) (codec.EncodedRequest, error) {
	h := http.Header{}
	h.Set("Content-Type", "application/json")
	return codec.EncodedRequest{Header: h, Body: eofCountReader{r: strings.NewReader(`{"x":1}`), eofs: c.eofs}}, nil
}

func (replayCodec) DecodeResponse([]byte) (*inference.Response, error) {
	return &inference.Response{Model: "x"}, nil
}

func TestNoReplay_BodyConsumedOnce(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		_, _ = io.WriteString(w, "{}")
	}))
	defer srv.Close()

	eofs := 0
	c := transport.New(transport.Endpoint{BaseURL: srv.URL}, staticRouter{method: http.MethodPost, path: "/x"}, replayCodec{eofs: &eofs}, auth.None())
	if _, err := c.Invoke(context.Background(), req("m")); err != nil {
		t.Fatalf("Invoke error: %v", err)
	}
	if eofs != 1 {
		t.Errorf("request body consumed %d times, want exactly 1 (no replay)", eofs)
	}
}

// TestNoReplay_RedirectNotFollowed proves the transport does not follow a redirect (which
// would replay the body): a 307 is surfaced as its 3xx APIError and the redirect target
// is never hit.
func TestNoReplay_RedirectNotFollowed(t *testing.T) {
	t.Parallel()
	targetHit := false
	mux := http.NewServeMux()
	mux.HandleFunc("/start", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/target", http.StatusTemporaryRedirect)
	})
	mux.HandleFunc("/target", func(w http.ResponseWriter, r *http.Request) {
		targetHit = true
		_, _ = io.WriteString(w, "{}")
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	c := transport.New(transport.Endpoint{BaseURL: srv.URL}, staticRouter{method: http.MethodPost, path: "/start"}, customCodec{body: "{}"}, auth.None())
	_, err := c.Invoke(context.Background(), req("m"))
	var apiErr *failure.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("err = %T (%v), want *failure.APIError (redirect not followed)", err, err)
	}
	if apiErr.Status != http.StatusTemporaryRedirect {
		t.Errorf("APIError.Status = %d, want 307", apiErr.Status)
	}
	if targetHit {
		t.Error("redirect target was hit: transport must not follow redirects (no body replay)")
	}
}

// ---- 6. Binding mismatch (through Invoke) ------------------------------------

func TestBindingMismatch_Invoke(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"id":"c1","model":"m","choices":[{"message":{"role":"assistant","content":"ok"}}]}`)
	}))
	defer srv.Close()

	ep := transport.Endpoint{Provider: "openrouter", BaseURL: srv.URL, APIFormat: model.APIFormatOpenAI}
	c := transport.New(ep, route.StaticChat("/chat/completions"), openaiapi.Codec{}, auth.None())

	cases := []struct {
		name     string
		model    model.Model
		wantMism bool
	}{
		{name: "all wildcards ok", model: model.Model{Name: "m"}, wantMism: false},
		{name: "matching identity ok", model: model.Model{Name: "m", Provider: "openrouter", BaseURL: srv.URL, APIFormat: model.APIFormatOpenAI}, wantMism: false},
		{name: "conflicting provider", model: model.Model{Name: "m", Provider: "chutes"}, wantMism: true},
		{name: "conflicting base url", model: model.Model{Name: "m", BaseURL: "https://evil.example"}, wantMism: true},
		{name: "conflicting api format", model: model.Model{Name: "m", APIFormat: model.APIFormatAnthropic}, wantMism: true},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := c.Invoke(context.Background(), inference.Request{Model: tc.model})
			var mism *failure.ModelMismatchError
			gotMism := errors.As(err, &mism)
			if gotMism != tc.wantMism {
				t.Fatalf("mismatch=%v (err=%v), want mismatch=%v", gotMism, err, tc.wantMism)
			}
		})
	}
}

// ---- 7. Custom API end-to-end ------------------------------------------------

// TestCustomAPI_EndToEnd proves a caller-supplied encoder + decoder + router hit a custom
// httptest route with NO transport change, including an unknown/custom APIFormat.
func TestCustomAPI_EndToEnd(t *testing.T) {
	t.Parallel()

	rec := &recorder{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec.record(r)
		_, _ = io.WriteString(w, `{"answer":"forty-two"}`)
	}))
	defer srv.Close()

	codec := customCodec{
		body: `{"ask":"meaning"}`,
		decode: func(body []byte) (*inference.Response, error) {
			var m struct {
				Answer string `json:"answer"`
			}
			if err := json.Unmarshal(body, &m); err != nil {
				return nil, err
			}
			return &inference.Response{
				Model: "custom-1",
				Message: &content.AIMessage{Message: content.Message{
					Role:   content.RoleAssistant,
					Blocks: []content.Block{&content.TextBlock{Text: m.Answer}},
				}},
			}, nil
		},
	}
	ep := transport.Endpoint{BaseURL: srv.URL, APIFormat: model.APIFormat("my-custom-dialect")}
	c := transport.New(ep, staticRouter{method: http.MethodPost, path: "/v9/answer"}, codec, auth.None())

	resp, err := c.Invoke(context.Background(), req("custom-model"))
	if err != nil {
		t.Fatalf("Invoke error: %v", err)
	}
	if got := firstText(t, resp); got != "forty-two" {
		t.Errorf("answer = %q, want forty-two", got)
	}
	_, path, _, _ := rec.snapshot()
	if path != "/v9/answer" {
		t.Errorf("path = %q, want /v9/answer", path)
	}
}

// ---- 8. Non-SSE streaming ----------------------------------------------------

// TestNonSSEStreaming proves the transport streams via a caller-supplied NDJSON-backed
// StreamDecoder — the transport makes no SSE assumption.
func TestNonSSEStreaming(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, "{\"text\":\"one\"}\n{\"text\":\"two\"}\n{\"text\":\"three\"}\n")
	}))
	defer srv.Close()

	c := transport.New(transport.Endpoint{BaseURL: srv.URL}, staticRouter{method: http.MethodPost, path: "/stream"}, customCodec{body: "{}"}, auth.None(), transport.WithStreamDecoder(ndjsonTextDecoder{}))
	stream, err := c.Stream(context.Background(), req("m"))
	if err != nil {
		t.Fatalf("Stream error: %v", err)
	}
	defer stream.Close()

	var got []string
	for {
		chunk, err := stream.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Next error: %v", err)
		}
		tx, ok := chunk.(*content.TextChunk)
		if !ok {
			t.Fatalf("chunk type = %T, want *content.TextChunk", chunk)
		}
		got = append(got, tx.Text)
	}
	want := []string{"one", "two", "three"}
	if len(got) != len(want) {
		t.Fatalf("chunks = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("chunk[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

// ---- 9. Bundled codec round-trips --------------------------------------------

func TestBundledCodecs_Invoke(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name     string
		router   route.Router
		codec    codec.Codec
		respBody string
		wantText string
	}{
		{
			name:     "openai",
			router:   route.StaticChat("/chat/completions"),
			codec:    openaiapi.Codec{},
			respBody: `{"id":"c1","model":"gpt","choices":[{"message":{"role":"assistant","content":"hello world"}}],"usage":{"prompt_tokens":1,"completion_tokens":2}}`,
			wantText: "hello world",
		},
		{
			name:     "anthropic",
			router:   route.StaticChat("/messages"),
			codec:    anthropicapi.Codec{},
			respBody: `{"id":"m1","type":"message","role":"assistant","model":"claude","content":[{"type":"text","text":"hi there"}],"usage":{"input_tokens":1,"output_tokens":2}}`,
			wantText: "hi there",
		},
		{
			name:     "gemini",
			router:   route.GeminiGenerateContent(),
			codec:    geminiapi.Codec{},
			respBody: `{"candidates":[{"content":{"parts":[{"text":"gday"}],"role":"model"}}],"modelVersion":"gemini-2.5","usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2}}`,
			wantText: "gday",
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				_, _ = io.WriteString(w, tc.respBody)
			}))
			defer srv.Close()

			c := transport.New(transport.Endpoint{BaseURL: srv.URL}, tc.router, tc.codec, auth.None())
			resp, err := c.Invoke(context.Background(), req("model-"+tc.name))
			if err != nil {
				t.Fatalf("Invoke error: %v", err)
			}
			if got := firstText(t, resp); got != tc.wantText {
				t.Errorf("text = %q, want %q", got, tc.wantText)
			}
		})
	}
}

// TestBundledCodecs_OpenAIStream proves an end-to-end streaming round trip through the
// OpenAI StreamingCodec: SSE deltas decode to chunks and the [DONE] sentinel ends the
// stream with io.EOF.
func TestBundledCodecs_OpenAIStream(t *testing.T) {
	t.Parallel()

	body := "data: {\"choices\":[{\"delta\":{\"content\":\"str\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"eam\"}}]}\n\n" +
		"data: [DONE]\n\n"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, body)
	}))
	defer srv.Close()

	c := transport.New(transport.Endpoint{BaseURL: srv.URL}, route.StaticChat("/chat/completions"), openaiapi.Codec{}, auth.None())
	stream, err := c.Stream(context.Background(), req("gpt"))
	if err != nil {
		t.Fatalf("Stream error: %v", err)
	}
	defer stream.Close()

	var text string
	for {
		chunk, err := stream.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Next error: %v", err)
		}
		tx, ok := chunk.(*content.TextChunk)
		if !ok {
			t.Fatalf("chunk type = %T, want *content.TextChunk", chunk)
		}
		text += tx.Text
	}
	if text != "stream" {
		t.Errorf("accumulated text = %q, want %q", text, "stream")
	}
}
