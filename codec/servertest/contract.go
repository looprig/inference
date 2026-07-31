// Package servertest provides a reusable contract suite for codec.ServerCodec
// implementations. The same suite is meant to run, unmodified beyond its Config,
// against every dialect's real server codec (anthropicapi, openairesponses,
// openaiapi, geminiapi) as well as against hand-written test fakes.
package servertest

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec"
	"github.com/looprig/inference/stream"
)

// Factory builds a fresh, ready-to-use ServerCodec for the suite to exercise. Per
// codec.ServerCodec's own contract, implementations must be stateless and safe for
// concurrent use, so a Factory may legally return the same value on every call; it
// exists so the suite never assumes anything about how a codec is constructed.
type Factory func() codec.ServerCodec

// Config parameterizes Run so the identical suite can pin every dialect's real
// ServerCodec as well as a hand-written fake. Every field is required; Run fails
// fast with a clear message if one is missing, rather than silently skipping part
// of the contract.
type Config struct {
	// NewCodec builds the ServerCodec under test.
	NewCodec Factory

	// Method and Path are a request line this codec's dialect owns; MatchRequest
	// must report true for it.
	Method string
	Path   string

	// ContentType is the header value a well-formed request for Method/Path
	// carries. Defaults to "application/json" when empty.
	ContentType string

	// ValidBody is a well-formed native request body for Method/Path; served with
	// ContentType, DecodeRequest must accept it without error.
	ValidBody []byte

	// UnmatchedMethod and UnmatchedPath each describe, independently, a request
	// line this codec's dialect does not own; MatchRequest must report false for
	// UnmatchedMethod paired with Path, and for Method paired with UnmatchedPath.
	UnmatchedMethod string
	UnmatchedPath   string

	// WrongContentType is a Content-Type value this codec's dialect does not
	// accept for Method/Path (for example "text/plain"). DecodeRequest must
	// reject ValidBody served with this header instead of the configured
	// ContentType.
	WrongContentType string

	// MalformedBody is a syntactically invalid body (for example truncated JSON)
	// for Method/Path. DecodeRequest must reject it with an error and must never
	// panic.
	MalformedBody []byte

	// SampleResponse is fed to WriteResponse for the clean non-streaming path.
	SampleResponse *inference.Response

	// SampleChunks are fed to StreamEncoder.WriteChunk, in order, during the
	// clean streaming path. At least one chunk is required so the streaming
	// flush assertion is meaningful.
	SampleChunks []content.Chunk

	// SampleResult is fed to StreamEncoder.Finish to end the clean streaming
	// path.
	SampleResult stream.StreamResult

	// SampleError is fed to WriteError and to StreamEncoder.Fail.
	SampleError error

	// ForeignProviderStateResponse, if set, is fed to WriteResponse to prove
	// this codec never forwards a content.ThinkingBlock.ProviderState whose
	// ProviderStateFormat does not match this codec's own dialect label — the
	// written response bytes must not contain ForeignProviderStateMarker.
	// Optional: a codec with no ProviderState-based opaque-replay mechanism
	// leaves both fields at their zero value, and RejectsForeignProviderState
	// is skipped rather than failing.
	ForeignProviderStateResponse *inference.Response
	ForeignProviderStateMarker   string
}

// Run exercises cfg.NewCodec against the shared server-codec contract: exactly one
// route match, content-type behavior, malformed-body rejection, native error
// shape, streaming flush, clean finish, cancellation, and single ownership of
// request/stream closure. Call it from a dialect package's own tests with that
// dialect's real codec and fixtures; codec/servertest's own contract_test.go calls
// it against a fake to prove the suite itself is exercised.
func Run(t *testing.T, cfg Config) {
	t.Helper()
	validateConfig(t, cfg)

	t.Run("MatchesOwnRoute", func(t *testing.T) { testMatchesOwnRoute(t, cfg) })
	t.Run("RejectsOtherRoutes", func(t *testing.T) { testRejectsOtherRoutes(t, cfg) })
	t.Run("RejectsWrongContentType", func(t *testing.T) { testRejectsWrongContentType(t, cfg) })
	t.Run("RejectsMalformedBody", func(t *testing.T) { testRejectsMalformedBody(t, cfg) })
	t.Run("DecodeIsDeterministic", func(t *testing.T) { testDecodeIsDeterministic(t, cfg) })
	t.Run("WriteResponse", func(t *testing.T) { testWriteResponse(t, cfg) })
	t.Run("WriteErrorShape", func(t *testing.T) { testWriteErrorShape(t, cfg) })
	t.Run("StreamFlushesAndFinishesCleanly", func(t *testing.T) { testStreamFlushesAndFinishesCleanly(t, cfg) })
	t.Run("StreamFail", func(t *testing.T) { testStreamFail(t, cfg) })
	t.Run("StreamSingleTermination", func(t *testing.T) { testStreamSingleTermination(t, cfg) })
	t.Run("DecodeReturnsPromptlyOnCanceledContext", func(t *testing.T) { testDecodeReturnsPromptlyOnCanceledContext(t, cfg) })
	t.Run("RejectsForeignProviderState", func(t *testing.T) { testRejectsForeignProviderState(t, cfg) })
}

func validateConfig(t *testing.T, cfg Config) {
	t.Helper()
	switch {
	case cfg.NewCodec == nil:
		t.Fatal("servertest.Config.NewCodec is required")
	case cfg.Method == "":
		t.Fatal("servertest.Config.Method is required")
	case cfg.Path == "":
		t.Fatal("servertest.Config.Path is required")
	case cfg.ValidBody == nil:
		t.Fatal("servertest.Config.ValidBody is required")
	case cfg.UnmatchedMethod == "":
		t.Fatal("servertest.Config.UnmatchedMethod is required")
	case cfg.UnmatchedPath == "":
		t.Fatal("servertest.Config.UnmatchedPath is required")
	case cfg.WrongContentType == "":
		t.Fatal("servertest.Config.WrongContentType is required")
	case cfg.MalformedBody == nil:
		t.Fatal("servertest.Config.MalformedBody is required")
	case cfg.SampleResponse == nil:
		t.Fatal("servertest.Config.SampleResponse is required")
	case len(cfg.SampleChunks) == 0:
		t.Fatal("servertest.Config.SampleChunks must contain at least one chunk")
	case cfg.SampleError == nil:
		t.Fatal("servertest.Config.SampleError is required")
	}
}

func requestContentType(cfg Config) string {
	if cfg.ContentType != "" {
		return cfg.ContentType
	}
	return "application/json"
}

func newRequest(method, path, contentType string, body []byte) *http.Request {
	req := httptest.NewRequest(method, path, bytes.NewReader(body))
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	return req
}

func testMatchesOwnRoute(t *testing.T, cfg Config) {
	t.Helper()
	c := cfg.NewCodec()
	req := newRequest(cfg.Method, cfg.Path, requestContentType(cfg), cfg.ValidBody)
	if !c.MatchRequest(req) {
		t.Fatalf("MatchRequest(%s %s) = false, want true", cfg.Method, cfg.Path)
	}
}

func testRejectsOtherRoutes(t *testing.T, cfg Config) {
	t.Helper()
	c := cfg.NewCodec()

	wrongMethod := newRequest(cfg.UnmatchedMethod, cfg.Path, requestContentType(cfg), cfg.ValidBody)
	if c.MatchRequest(wrongMethod) {
		t.Errorf("MatchRequest(%s %s) = true, want false (wrong method)", cfg.UnmatchedMethod, cfg.Path)
	}

	wrongPath := newRequest(cfg.Method, cfg.UnmatchedPath, requestContentType(cfg), cfg.ValidBody)
	if c.MatchRequest(wrongPath) {
		t.Errorf("MatchRequest(%s %s) = true, want false (wrong path)", cfg.Method, cfg.UnmatchedPath)
	}
}

func testRejectsWrongContentType(t *testing.T, cfg Config) {
	t.Helper()
	c := cfg.NewCodec()
	req := newRequest(cfg.Method, cfg.Path, cfg.WrongContentType, cfg.ValidBody)
	if _, err := c.DecodeRequest(req); err == nil {
		t.Errorf("DecodeRequest with Content-Type %q: got nil error, want rejection", cfg.WrongContentType)
	}
}

func testRejectsMalformedBody(t *testing.T, cfg Config) {
	t.Helper()
	c := cfg.NewCodec()
	req := newRequest(cfg.Method, cfg.Path, requestContentType(cfg), cfg.MalformedBody)

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("DecodeRequest panicked on a malformed body: %v", r)
		}
	}()
	if _, err := c.DecodeRequest(req); err == nil {
		t.Error("DecodeRequest(malformed body): got nil error, want error")
	}
}

func testDecodeIsDeterministic(t *testing.T, cfg Config) {
	t.Helper()
	c := cfg.NewCodec()

	req1 := newRequest(cfg.Method, cfg.Path, requestContentType(cfg), cfg.ValidBody)
	got1, err1 := c.DecodeRequest(req1)
	if err1 != nil {
		t.Fatalf("first DecodeRequest: %v", err1)
	}

	req2 := newRequest(cfg.Method, cfg.Path, requestContentType(cfg), cfg.ValidBody)
	got2, err2 := c.DecodeRequest(req2)
	if err2 != nil {
		t.Fatalf("second DecodeRequest: %v", err2)
	}

	if !reflect.DeepEqual(got1, got2) {
		t.Errorf("DecodeRequest is not deterministic for identical input:\n  first:  %+v\n  second: %+v", got1, got2)
	}
}

func testWriteResponse(t *testing.T, cfg Config) {
	t.Helper()
	c := cfg.NewCodec()
	rec := httptest.NewRecorder()

	if err := c.WriteResponse(rec, cfg.SampleResponse); err != nil {
		t.Fatalf("WriteResponse: %v", err)
	}
	if rec.Code < 200 || rec.Code >= 300 {
		t.Errorf("WriteResponse status = %d, want 2xx", rec.Code)
	}
	if rec.Body.Len() == 0 {
		t.Error("WriteResponse wrote an empty body")
	}
}

func testWriteErrorShape(t *testing.T, cfg Config) {
	t.Helper()
	c := cfg.NewCodec()
	rec := httptest.NewRecorder()

	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("WriteError panicked: %v", r)
			}
		}()
		c.WriteError(rec, cfg.SampleError)
	}()

	if rec.Code < 400 || rec.Code >= 600 {
		t.Errorf("WriteError status = %d, want 4xx or 5xx", rec.Code)
	}
	if rec.Body.Len() == 0 {
		t.Error("WriteError wrote an empty body")
	}
}

func testStreamFlushesAndFinishesCleanly(t *testing.T, cfg Config) {
	t.Helper()
	c := cfg.NewCodec()
	rec := httptest.NewRecorder()

	enc, err := c.OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream: %v", err)
	}

	for i, chunk := range cfg.SampleChunks {
		if err := enc.WriteChunk(chunk); err != nil {
			t.Fatalf("WriteChunk[%d]: %v", i, err)
		}
	}
	if rec.Body.Len() == 0 {
		t.Error("WriteChunk wrote no bytes to the response before Finish; streaming must flush progressively")
	}

	if err := enc.Finish(cfg.SampleResult); err != nil {
		t.Fatalf("Finish: %v", err)
	}
	if rec.Body.Len() == 0 {
		t.Error("Finish left the response body empty")
	}
}

func testStreamFail(t *testing.T, cfg Config) {
	t.Helper()
	c := cfg.NewCodec()
	rec := httptest.NewRecorder()

	enc, err := c.OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream: %v", err)
	}

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Fail panicked: %v", r)
		}
	}()
	if err := enc.Fail(cfg.SampleError); err != nil {
		t.Fatalf("Fail: %v", err)
	}
}

// testStreamSingleTermination pins single ownership of stream closure: exactly one
// of Finish or Fail may terminate a StreamEncoder. Once terminated, every further
// call must be rejected with an error rather than writing again to the
// conceptually closed response, and must never panic.
func testStreamSingleTermination(t *testing.T, cfg Config) {
	t.Helper()
	c := cfg.NewCodec()
	rec := httptest.NewRecorder()

	enc, err := c.OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream: %v", err)
	}
	if err := enc.Finish(cfg.SampleResult); err != nil {
		t.Fatalf("first Finish: %v", err)
	}

	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("second Finish panicked: %v", r)
			}
		}()
		if err := enc.Finish(cfg.SampleResult); err == nil {
			t.Error("second Finish: got nil error, want error (already terminated)")
		}
	}()

	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("Fail-after-Finish panicked: %v", r)
			}
		}()
		if err := enc.Fail(cfg.SampleError); err == nil {
			t.Error("Fail after Finish: got nil error, want error (already terminated)")
		}
	}()

	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("WriteChunk-after-Finish panicked: %v", r)
			}
		}()
		if err := enc.WriteChunk(cfg.SampleChunks[0]); err == nil {
			t.Error("WriteChunk after Finish: got nil error, want error (already terminated)")
		}
	}()
}

// testDecodeReturnsPromptlyOnCanceledContext pins cancellation ownership at the
// codec boundary: DecodeRequest sees cancellation through the request's own
// context (the gateway owns cancellation policy, but a codec must never hang
// forever on I/O once the caller has walked away) and must return, not block
// indefinitely or panic.
func testDecodeReturnsPromptlyOnCanceledContext(t *testing.T, cfg Config) {
	t.Helper()
	c := cfg.NewCodec()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	req := newRequest(cfg.Method, cfg.Path, requestContentType(cfg), cfg.ValidBody).WithContext(ctx)

	type outcome struct{ panicVal any }
	done := make(chan outcome, 1)
	go func() {
		defer func() { done <- outcome{panicVal: recover()} }()
		_, _ = c.DecodeRequest(req)
	}()

	select {
	case o := <-done:
		if o.panicVal != nil {
			t.Fatalf("DecodeRequest panicked on a canceled request context: %v", o.panicVal)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("DecodeRequest did not return promptly for a canceled request context")
	}
}

// testRejectsForeignProviderState structurally guards the opaque-replay
// invariant documented on content.ThinkingBlock.ProviderStateFormat: a codec
// must never forward a ThinkingBlock.ProviderState tagged with a DIFFERENT
// dialect's format label toward a wire field it owns. Every dialect that
// carries an opaque-replay mechanism (currently geminiapi, openairesponses)
// must wire ForeignProviderStateResponse/ForeignProviderStateMarker so this
// runs for real; a dialect with no such mechanism (anthropicapi's
// Signature-only approach, openaiapi's lack of one at all) legitimately
// leaves both fields unset and this sub-test skips.
func testRejectsForeignProviderState(t *testing.T, cfg Config) {
	t.Helper()
	if cfg.ForeignProviderStateResponse == nil {
		t.Skip("codec does not use ThinkingBlock.ProviderState; nothing to guard")
	}
	c := cfg.NewCodec()
	rec := httptest.NewRecorder()
	if err := c.WriteResponse(rec, cfg.ForeignProviderStateResponse); err != nil {
		t.Fatalf("WriteResponse: %v", err)
	}
	if strings.Contains(rec.Body.String(), cfg.ForeignProviderStateMarker) {
		t.Errorf("WriteResponse forwarded foreign provider-opaque state (marker %q present in response body) -- a codec must only replay ThinkingBlock.ProviderState tagged with its own ProviderStateFormat", cfg.ForeignProviderStateMarker)
	}
}
