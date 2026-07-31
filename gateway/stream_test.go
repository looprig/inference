package gateway_test

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec"
	"github.com/looprig/inference/codec/anthropicapi"
	"github.com/looprig/inference/gateway"
	"github.com/looprig/inference/model"
	"github.com/looprig/inference/stream"
)

// --- shared streaming test doubles, used by stream_test.go, cancel_test.go, and stream_leak_test.go ---

const streamingMessagesBody = `{"model":"primary","max_tokens":16,"stream":true,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`

// streamStep is one controllable event fed to a controlledStreamClient's
// Next() call: either a chunk or a non-EOF error. Clean exhaustion is
// signaled by closing the steps channel instead of sending a step.
type streamStep struct {
	chunk content.Chunk
	err   error
}

// errStreamClosedDuringNext is what a blocked next() call observes when the
// stream is interrupted via Close (whether from a normal deferred Close or
// from the gateway's context-cancellation watcher) rather than exhausted
// cleanly or failed by the upstream itself.
var errStreamClosedDuringNext = errors.New("controlledStreamClient: Next interrupted by Close")

// controlledStreamClient is a minimal inference.Client double whose Stream
// method returns a *stream.StreamReader[content.Chunk] entirely driven by
// the test: each Next() call blocks on the unbuffered steps channel until
// the test sends a streamStep (or closes it for clean EOF), so a test can
// observe exactly what the handler wrote between every chunk (an unbuffered
// channel makes send-happens-before-receive an ordering guarantee: by the
// time a send for chunk N+1 returns, Next() must have already returned
// chunk N and the handler's pull loop must have already finished writing it,
// since Next() calls are strictly sequential within that single goroutine).
//
// closed is next()'s ONLY interruption path -- it deliberately does NOT also
// select on ctx.Done(), so a cancellation test exercises the gateway's own
// context-cancels-Close watcher rather than a shortcut built into this fake,
// mirroring how a real inference.Client's Stream implementation has no
// per-Next context to observe (see stream.StreamReader.Next's doc comment).
type controlledStreamClient struct {
	mu        sync.Mutex
	gotReq    inference.Request
	calls     int
	streamErr error
	// blockStream, if non-nil, makes Stream itself block on this channel (or
	// ctx.Done()) before proceeding -- mirrors recordingClient's `block`
	// field in handler_test.go, used only by the pre-header deadline test.
	blockStream chan struct{}

	result    stream.StreamResult
	hasResult bool

	steps      chan streamStep
	closed     chan struct{}
	closeOnce  sync.Once
	closeCount int32
}

func newControlledStreamClient() *controlledStreamClient {
	return &controlledStreamClient{
		steps:  make(chan streamStep),
		closed: make(chan struct{}),
	}
}

func (c *controlledStreamClient) Invoke(context.Context, inference.Request) (*inference.Response, error) {
	return nil, errors.New("controlledStreamClient.Invoke: not implemented")
}

func (c *controlledStreamClient) Stream(ctx context.Context, req inference.Request) (*stream.StreamReader[content.Chunk], error) {
	c.mu.Lock()
	c.gotReq = req
	c.calls++
	streamErr := c.streamErr
	block := c.blockStream
	c.mu.Unlock()

	if block != nil {
		select {
		case <-block:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if streamErr != nil {
		return nil, streamErr
	}

	next := func() (content.Chunk, error) {
		select {
		case step, ok := <-c.steps:
			if !ok {
				return nil, io.EOF
			}
			return step.chunk, step.err
		case <-c.closed:
			return nil, errStreamClosedDuringNext
		}
	}
	closer := func() error {
		c.closeOnce.Do(func() { close(c.closed) })
		atomic.AddInt32(&c.closeCount, 1)
		return nil
	}
	producer := func() (stream.StreamResult, bool, error) {
		c.mu.Lock()
		defer c.mu.Unlock()
		return c.result, c.hasResult, nil
	}
	return stream.NewStreamReaderWithResult(next, closer, producer), nil
}

func (c *controlledStreamClient) lastRequest() inference.Request {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.gotReq
}

func (c *controlledStreamClient) callCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.calls
}

// send delivers one chunk to the in-flight Next() call, blocking until it is
// received.
func (c *controlledStreamClient) send(chunk content.Chunk) {
	c.steps <- streamStep{chunk: chunk}
}

// fail delivers a non-EOF error to the in-flight Next() call, blocking until
// it is received.
func (c *controlledStreamClient) fail(err error) {
	c.steps <- streamStep{err: err}
}

// finishClean records result as the terminal metadata Result() will report,
// then closes steps so the next Next() call observes a clean io.EOF.
func (c *controlledStreamClient) finishClean(result stream.StreamResult) {
	c.mu.Lock()
	c.result = result
	c.hasResult = true
	c.mu.Unlock()
	close(c.steps)
}

// finishCleanNoResult closes steps for a clean io.EOF without ever recording
// a producer result -- the defensive "Result() returns false" case.
func (c *controlledStreamClient) finishCleanNoResult() {
	close(c.steps)
}

func (c *controlledStreamClient) closeCalls() int {
	return int(atomic.LoadInt32(&c.closeCount))
}

// --- spy ServerCodec: a thin decorator over the real anthropicapi.Codec ----
//
// Every actual encode/decode/framing operation is still performed by the
// real, already-tested anthropicapi.Codec{} -- this only adds observation
// points (which StreamResult Finish received, how many times Fail was
// called, whether OpenStream was reached at all) that the native Anthropic
// wire format does not otherwise expose to a black-box HTTP test (its
// streaming events carry no model field at all -- see server_stream.go's
// sseStartMessage/sseMessageDelta -- so "the terminal Model reported is the
// harness alias" cannot be asserted purely from wire bytes).

type spyStreamCodec struct {
	anthropicapi.Codec

	mu            sync.Mutex
	openStreamN   int
	finishResults []stream.StreamResult
	failErrs      []error
}

func (s *spyStreamCodec) OpenStream(w http.ResponseWriter) (codec.StreamEncoder, error) {
	s.mu.Lock()
	s.openStreamN++
	s.mu.Unlock()
	enc, err := s.Codec.OpenStream(w)
	if err != nil {
		return nil, err
	}
	return &spyStreamEncoder{StreamEncoder: enc, spy: s}, nil
}

func (s *spyStreamCodec) openStreamCalls() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.openStreamN
}

func (s *spyStreamCodec) finishCalls() []stream.StreamResult {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]stream.StreamResult(nil), s.finishResults...)
}

func (s *spyStreamCodec) failCalls() []error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]error(nil), s.failErrs...)
}

type spyStreamEncoder struct {
	codec.StreamEncoder
	spy *spyStreamCodec
}

func (e *spyStreamEncoder) Finish(result stream.StreamResult) error {
	e.spy.mu.Lock()
	e.spy.finishResults = append(e.spy.finishResults, result)
	e.spy.mu.Unlock()
	return e.StreamEncoder.Finish(result)
}

func (e *spyStreamEncoder) Fail(err error) error {
	e.spy.mu.Lock()
	e.spy.failErrs = append(e.spy.failErrs, err)
	e.spy.mu.Unlock()
	return e.StreamEncoder.Fail(err)
}

// newStreamHandler builds a Handler wired to a fresh *spyStreamCodec (which
// delegates all real work to anthropicapi.Codec{}) and the given client,
// with one route (Anthropic ingress, alias "primary"). maxConcurrent == 0
// means gateway.DefaultMaxConcurrent.
func newStreamHandler(t *testing.T, client inference.Client, maxConcurrent int) (*gateway.Handler, *spyStreamCodec) {
	t.Helper()
	target := gateway.Target{ID: "t", Client: client, Model: anthropicModel("kimi-k2")}
	resolver, err := gateway.NewMux(gateway.Mux{
		Routes: map[gateway.RouteKey]gateway.Target{
			{Ingress: model.APIFormatAnthropic, Model: "primary"}: target,
		},
	})
	if err != nil {
		t.Fatalf("NewMux: %v", err)
	}
	spy := &spyStreamCodec{}
	h, err := gateway.New(gateway.Config{
		Resolver:      resolver,
		Codecs:        map[model.APIFormat]codec.ServerCodec{model.APIFormatAnthropic: spy},
		Authenticate:  gateway.StaticToken("test-token"),
		MaxConcurrent: maxConcurrent,
	})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	return h, spy
}

// --- SSE parsing helper ------------------------------------------------------

type sseEvent struct {
	Name string
	Data map[string]any
}

// parseSSEEvents splits a complete native Anthropic SSE response body into
// its individual "event: ...\ndata: ...\n\n" frames.
func parseSSEEvents(t *testing.T, body string) []sseEvent {
	t.Helper()
	var events []sseEvent
	for _, frame := range strings.Split(strings.TrimRight(body, "\n"), "\n\n") {
		if strings.TrimSpace(frame) == "" {
			continue
		}
		var name, dataLine string
		for _, line := range strings.Split(frame, "\n") {
			switch {
			case strings.HasPrefix(line, "event: "):
				name = strings.TrimPrefix(line, "event: ")
			case strings.HasPrefix(line, "data: "):
				dataLine = strings.TrimPrefix(line, "data: ")
			}
		}
		var data map[string]any
		if dataLine != "" {
			if err := json.Unmarshal([]byte(dataLine), &data); err != nil {
				t.Fatalf("parsing SSE data for event %q: %v (raw: %s)", name, err, dataLine)
			}
		}
		events = append(events, sseEvent{Name: name, Data: data})
	}
	return events
}

func eventNames(events []sseEvent) []string {
	names := make([]string, len(events))
	for i, e := range events {
		names[i] = e.Name
	}
	return names
}

// --- Chunk order / tool-call index preservation ------------------------------

// TestServeStreaming_ChunkOrderAndToolIndexesPreserved proves the pull loop
// writes chunks to the StreamEncoder in exactly the order Next() returned
// them, and that interleaved content.ToolUseChunk values at different
// neutral Index values never bleed into each other's wire content_block.
// The exact wire event sequence below follows directly from the real
// anthropicapi encoder's documented ensureBlock behavior (server_stream.go):
// only one content_block is ever open at a time, so a return to a
// previously-active neutral Index re-opens a fresh wire block rather than
// resuming the old one -- this test pins down that this re-serialization
// still preserves chunk order and never attributes a delta to the wrong
// tool call.
func TestServeStreaming_ChunkOrderAndToolIndexesPreserved(t *testing.T) {
	assertNoGoroutineLeak(t)
	t.Parallel()
	client := newControlledStreamClient()
	h, _ := newStreamHandler(t, client, 0)

	rr := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		h.ServeHTTP(rr, messagesRequest(t, "test-token", streamingMessagesBody))
		close(done)
	}()

	client.send(&content.TextChunk{Text: "pre"})
	client.send(&content.ToolUseChunk{Index: 0, ID: "id0", Name: "alpha"})
	client.send(&content.ToolUseChunk{Index: 1, ID: "id1", Name: "beta"})
	client.send(&content.ToolUseChunk{Index: 0, InputJSON: `{"x":1}`})
	client.send(&content.ToolUseChunk{Index: 1, InputJSON: `{"y":2}`})
	client.finishClean(stream.StreamResult{FinishReason: stream.FinishReasonToolUse})

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("ServeHTTP did not return")
	}

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body: %s", rr.Code, rr.Body.String())
	}

	events := parseSSEEvents(t, rr.Body.String())
	wantNames := []string{
		"message_start",
		"content_block_start", // wire0: text
		"content_block_delta", // "pre"
		"content_block_stop",  // close wire0
		"content_block_start", // wire1: tool_use id0/alpha
		"content_block_stop",  // close wire1 (index switches to 1)
		"content_block_start", // wire2: tool_use id1/beta
		"content_block_stop",  // close wire2 (index switches back to 0)
		"content_block_start", // wire3: tool_use (fresh id, re-opened for index 0)
		"content_block_delta", // {"x":1}
		"content_block_stop",  // close wire3 (index switches to 1)
		"content_block_start", // wire4: tool_use (fresh id, re-opened for index 1)
		"content_block_delta", // {"y":2}
		"content_block_stop",  // Finish closes the still-open block
		"message_delta",
		"message_stop",
	}
	if got := eventNames(events); !equalStrings(got, wantNames) {
		t.Fatalf("event sequence =\n%v\nwant\n%v", got, wantNames)
	}

	// "pre" delta.
	assertDeltaText(t, events[2], "pre")
	// wire1 content_block_start: tool_use id0/alpha.
	assertToolUseStart(t, events[4], "id0", "alpha")
	// wire2 content_block_start: tool_use id1/beta.
	assertToolUseStart(t, events[6], "id1", "beta")
	// wire3 delta: partial_json for index 0's second occurrence.
	assertDeltaPartialJSON(t, events[9], `{"x":1}`)
	// wire4 delta: partial_json for index 1's second occurrence.
	assertDeltaPartialJSON(t, events[12], `{"y":2}`)

	if calls := client.closeCalls(); calls != 1 {
		t.Errorf("upstream StreamReader Close called %d times, want 1", calls)
	}
}

func assertDeltaText(t *testing.T, e sseEvent, want string) {
	t.Helper()
	delta, _ := e.Data["delta"].(map[string]any)
	if got, _ := delta["text"].(string); got != want {
		t.Errorf("content_block_delta text = %q, want %q (event: %+v)", got, want, e)
	}
}

func assertDeltaPartialJSON(t *testing.T, e sseEvent, want string) {
	t.Helper()
	delta, _ := e.Data["delta"].(map[string]any)
	if got, _ := delta["partial_json"].(string); got != want {
		t.Errorf("content_block_delta partial_json = %q, want %q (event: %+v)", got, want, e)
	}
}

func assertToolUseStart(t *testing.T, e sseEvent, wantID, wantName string) {
	t.Helper()
	block, _ := e.Data["content_block"].(map[string]any)
	if got, _ := block["type"].(string); got != "tool_use" {
		t.Errorf("content_block_start type = %q, want tool_use (event: %+v)", got, e)
	}
	if got, _ := block["id"].(string); got != wantID {
		t.Errorf("content_block_start id = %q, want %q (event: %+v)", got, wantID, e)
	}
	if got, _ := block["name"].(string); got != wantName {
		t.Errorf("content_block_start name = %q, want %q (event: %+v)", got, wantName, e)
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// --- Clean completion ---------------------------------------------------

// TestServeStreaming_CleanEOF_FinishReceivesAliasModel proves Finish is
// called exactly once, only after clean EOF, with the reader's authoritative
// terminal metadata except Model, which is overwritten with the harness
// alias ("primary") rather than the resolved upstream model name
// ("kimi-k2") -- mirroring the non-streaming path's resp.Model assignment.
// Fail must never be called on this path, and the upstream reader must be
// closed exactly once.
func TestServeStreaming_CleanEOF_FinishReceivesAliasModel(t *testing.T) {
	assertNoGoroutineLeak(t)
	t.Parallel()
	client := newControlledStreamClient()
	h, spy := newStreamHandler(t, client, 0)

	rr := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		h.ServeHTTP(rr, messagesRequest(t, "test-token", streamingMessagesBody))
		close(done)
	}()

	client.send(&content.TextChunk{Text: "hi"})
	usage := &content.Usage{OutputTokens: 7}
	client.finishClean(stream.StreamResult{
		Model:        "kimi-k2", // the resolved upstream name -- must be overwritten
		FinishReason: stream.FinishReasonStop,
		Usage:        usage,
	})

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("ServeHTTP did not return")
	}

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body: %s", rr.Code, rr.Body.String())
	}

	finishes := spy.finishCalls()
	if len(finishes) != 1 {
		t.Fatalf("Finish called %d times, want 1 (fails: %v)", len(finishes), spy.failCalls())
	}
	if got := finishes[0].Model; got != "primary" {
		t.Errorf("Finish's StreamResult.Model = %q, want the harness alias %q", got, "primary")
	}
	if finishes[0].FinishReason != stream.FinishReasonStop {
		t.Errorf("Finish's StreamResult.FinishReason = %q, want %q", finishes[0].FinishReason, stream.FinishReasonStop)
	}
	if finishes[0].Usage == nil || finishes[0].Usage.OutputTokens != 7 {
		t.Errorf("Finish's StreamResult.Usage = %+v, want OutputTokens=7", finishes[0].Usage)
	}
	if fails := spy.failCalls(); len(fails) != 0 {
		t.Errorf("Fail called %d times on a clean-EOF path, want 0: %v", len(fails), fails)
	}
	if calls := client.closeCalls(); calls != 1 {
		t.Errorf("upstream StreamReader Close called %d times, want 1", calls)
	}
}

// TestServeStreaming_DefensiveNoResult_StillFinishes proves that when
// Result() returns ok=false at clean EOF (a defensive case -- not expected
// from a well-behaved client, but must not panic or hang), the pull loop
// still calls Finish exactly once, with a zero-value StreamResult carrying
// only the harness alias as Model.
func TestServeStreaming_DefensiveNoResult_StillFinishes(t *testing.T) {
	assertNoGoroutineLeak(t)
	t.Parallel()
	client := newControlledStreamClient()
	h, spy := newStreamHandler(t, client, 0)

	rr := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		h.ServeHTTP(rr, messagesRequest(t, "test-token", streamingMessagesBody))
		close(done)
	}()

	client.finishCleanNoResult()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("ServeHTTP did not return")
	}

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body: %s", rr.Code, rr.Body.String())
	}
	finishes := spy.finishCalls()
	if len(finishes) != 1 {
		t.Fatalf("Finish called %d times, want 1", len(finishes))
	}
	if got := finishes[0].Model; got != "primary" {
		t.Errorf("Finish's StreamResult.Model = %q, want %q", got, "primary")
	}
	if finishes[0].FinishReason != stream.FinishReasonUnknown || finishes[0].Usage != nil {
		t.Errorf("Finish's StreamResult = %+v, want zero-value apart from Model", finishes[0])
	}
}

// --- In-stream (post-header) failure -----------------------------------

// TestServeStreaming_UpstreamNextError_CallsFailNotFinish proves a non-EOF
// error from Next after some chunks have already been written is a
// post-header failure: it terminates via the codec's native error event
// (Fail), never Finish, and the response body shows the real native SSE
// error frame (an `event: error` frame), not a fresh JSON error envelope
// written through WriteError.
func TestServeStreaming_UpstreamNextError_CallsFailNotFinish(t *testing.T) {
	assertNoGoroutineLeak(t)
	t.Parallel()
	client := newControlledStreamClient()
	h, spy := newStreamHandler(t, client, 0)

	rr := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		h.ServeHTTP(rr, messagesRequest(t, "test-token", streamingMessagesBody))
		close(done)
	}()

	client.send(&content.TextChunk{Text: "partial"})
	client.fail(errors.New("upstream broke mid-stream"))

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("ServeHTTP did not return")
	}

	// Headers/status were already committed by OpenStream before the
	// failure -- the recorder still reports whatever OpenStream wrote
	// (200), not a fresh error status.
	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (headers already committed by OpenStream); body: %s", rr.Code, rr.Body.String())
	}
	if spy.openStreamCalls() != 1 {
		t.Fatalf("OpenStream called %d times, want 1", spy.openStreamCalls())
	}
	if fails := spy.failCalls(); len(fails) != 1 {
		t.Fatalf("Fail called %d times, want 1", len(fails))
	}
	if finishes := spy.finishCalls(); len(finishes) != 0 {
		t.Errorf("Finish called %d times on an in-stream failure path, want 0", len(finishes))
	}

	events := parseSSEEvents(t, rr.Body.String())
	if len(events) == 0 || events[len(events)-1].Name != "error" {
		t.Fatalf("last SSE event = %+v, want a native `event: error` frame", events)
	}
	if calls := client.closeCalls(); calls != 1 {
		t.Errorf("upstream StreamReader Close called %d times, want 1", calls)
	}
}

// --- Pre-header (Stream call itself) failure -----------------------------

// TestServeStreaming_PreHeaderStreamError_UsesWriteErrorNotOpenStream proves
// that when target.Client.Stream itself fails, the gateway never reaches
// OpenStream at all: it classifies and writes the error exactly like a
// normal pre-invoke pipeline error (a JSON error envelope via WriteError,
// wrapped in the same *UpstreamInvocationError type the non-streaming
// Invoke path uses), not a native in-stream error event.
func TestServeStreaming_PreHeaderStreamError_UsesWriteErrorNotOpenStream(t *testing.T) {
	assertNoGoroutineLeak(t)
	t.Parallel()
	client := newControlledStreamClient()
	client.streamErr = errors.New("upstream: connection refused")
	h, spy := newStreamHandler(t, client, 0)

	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, messagesRequest(t, "test-token", streamingMessagesBody))

	if rr.Code != http.StatusBadGateway {
		t.Errorf("status = %d, want %d; body: %s", rr.Code, http.StatusBadGateway, rr.Body.String())
	}
	if spy.openStreamCalls() != 0 {
		t.Errorf("OpenStream called %d times, want 0 (Stream itself failed before header commit)", spy.openStreamCalls())
	}
	if strings.Contains(rr.Body.String(), "event:") {
		t.Errorf("response body looks like an SSE frame, want a plain JSON error envelope: %s", rr.Body.String())
	}
	var wire struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &wire); err != nil {
		t.Fatalf("decoding response JSON: %v (body: %s)", err, rr.Body.String())
	}
	if wire.Type != "error" {
		t.Errorf("response JSON type = %q, want %q", wire.Type, "error")
	}
}

// TestServeStreaming_PreHeaderDeadlineExceeded_504 proves a context deadline
// exceeded while calling target.Client.Stream itself (before any native
// stream has opened) classifies as 504, mirroring the non-streaming
// TestHandler_UpstreamDeadlineExceeded_504.
func TestServeStreaming_PreHeaderDeadlineExceeded_504(t *testing.T) {
	assertNoGoroutineLeak(t)
	t.Parallel()
	client := newControlledStreamClient()
	client.blockStream = make(chan struct{}) // never closed: Stream blocks until ctx times out
	h, spy := newStreamHandler(t, client, 0)

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	req := messagesRequest(t, "test-token", streamingMessagesBody).WithContext(ctx)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusGatewayTimeout {
		t.Errorf("status = %d, want %d; body: %s", rr.Code, http.StatusGatewayTimeout, rr.Body.String())
	}
	if spy.openStreamCalls() != 0 {
		t.Errorf("OpenStream called %d times, want 0", spy.openStreamCalls())
	}
}

// --- Admission release ----------------------------------------------------

// TestServeStreaming_AdmissionReleasedAfterCompletion proves the
// concurrency-admission slot held around serveStreaming (by serveInference,
// unchanged by this task) is released exactly once a streaming request
// completes: a saturating request rejected mid-stream succeeds once the
// first request finishes, following the same pattern as
// TestHandler_ConcurrencyAdmission_429 in handler_test.go.
func TestServeStreaming_AdmissionReleasedAfterCompletion(t *testing.T) {
	assertNoGoroutineLeak(t)
	t.Parallel()
	client := newControlledStreamClient()
	h, _ := newStreamHandler(t, client, 1)

	done := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		rr := httptest.NewRecorder()
		h.ServeHTTP(rr, messagesRequest(t, "test-token", streamingMessagesBody))
		done <- rr
	}()

	deadline := time.After(2 * time.Second)
	for client.callCount() == 0 {
		select {
		case <-deadline:
			t.Fatal("first request never reached Stream")
		case <-time.After(time.Millisecond):
		}
	}

	rr2 := httptest.NewRecorder()
	h.ServeHTTP(rr2, messagesRequest(t, "test-token", streamingMessagesBody))
	if rr2.Code != http.StatusTooManyRequests {
		t.Errorf("second (saturated) request status = %d, want %d; body: %s", rr2.Code, http.StatusTooManyRequests, rr2.Body.String())
	}

	client.finishClean(stream.StreamResult{FinishReason: stream.FinishReasonStop})
	rr1 := <-done
	if rr1.Code != http.StatusOK {
		t.Errorf("first (in-flight) request status = %d, want %d; body: %s", rr1.Code, http.StatusOK, rr1.Body.String())
	}

	rr3 := httptest.NewRecorder()
	h.ServeHTTP(rr3, messagesRequest(t, "test-token", streamingMessagesBody))
	if rr3.Code != http.StatusOK {
		t.Errorf("post-release request status = %d, want %d (admission not released?); body: %s", rr3.Code, http.StatusOK, rr3.Body.String())
	}
}

// --- Flush visibility (real server) --------------------------------------

// TestServeStreaming_FlushBeforeNextChunkReleased proves each native SSE
// event reaches an independently-reading client before the next chunk is
// released upstream. This deliberately uses a real httptest.Server and a
// real http.Client reading the response body incrementally, NOT an
// httptest.ResponseRecorder: a ResponseRecorder buffers every Write directly
// into an in-memory bytes.Buffer with no separate "flushed to the wire"
// signal a second, independent reader could observe -- so it cannot
// distinguish "the handler goroutine wrote and flushed this chunk" from "the
// handler goroutine merely appended to a buffer nobody has read yet". A real
// connection can: bufio.Reader.ReadString blocks on the client side until
// bytes are actually available, so successfully reading frame N+1 is only
// possible once the server has actually flushed it -- there is nothing else
// that could produce those bytes. (The other streaming tests in this file
// use ResponseRecorder plus the controlledStreamClient's unbuffered-channel
// rendezvous instead, which is sufficient there because they only need
// same-goroutine sequencing, not cross-process wire visibility.)
func TestServeStreaming_FlushBeforeNextChunkReleased(t *testing.T) {
	assertNoGoroutineLeak(t)
	t.Parallel()
	client := newControlledStreamClient()
	h, _ := newStreamHandler(t, client, 0)

	srv := httptest.NewServer(h)
	defer srv.Close()

	req, err := http.NewRequest(http.MethodPost, srv.URL+"/v1/messages", strings.NewReader(streamingMessagesBody))
	if err != nil {
		t.Fatalf("building request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer test-token")

	respCh := make(chan *http.Response, 1)
	errCh := make(chan error, 1)
	go func() {
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			errCh <- err
			return
		}
		respCh <- resp
	}()

	var resp *http.Response
	select {
	case resp = <-respCh:
	case err := <-errCh:
		t.Fatalf("request failed: %v", err)
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for response headers")
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	br := bufio.NewReader(resp.Body)
	readFrame := func() string {
		t.Helper()
		type result struct {
			frame string
			err   error
		}
		out := make(chan result, 1)
		go func() {
			var sb strings.Builder
			for {
				line, err := br.ReadString('\n')
				sb.WriteString(line)
				if err != nil {
					out <- result{sb.String(), err}
					return
				}
				if line == "\n" {
					out <- result{sb.String(), nil}
					return
				}
			}
		}()
		select {
		case r := <-out:
			if r.err != nil {
				t.Fatalf("reading SSE frame: %v", r.err)
			}
			return r.frame
		case <-time.After(2 * time.Second):
			t.Fatal("timed out reading SSE frame -- flush did not reach the client")
			return ""
		}
	}

	if frame := readFrame(); !strings.Contains(frame, "message_start") {
		t.Fatalf("first frame = %q, want message_start", frame)
	}

	client.send(&content.TextChunk{Text: "hello"})
	if frame := readFrame(); !strings.Contains(frame, "content_block_start") {
		t.Fatalf("frame after 1st chunk = %q, want content_block_start", frame)
	}
	if frame := readFrame(); !strings.Contains(frame, `"text":"hello"`) {
		t.Fatalf("frame after 1st chunk delta = %q, want text delta \"hello\"", frame)
	}

	client.finishClean(stream.StreamResult{FinishReason: stream.FinishReasonStop})
	if frame := readFrame(); !strings.Contains(frame, "content_block_stop") {
		t.Fatalf("frame after finish = %q, want content_block_stop", frame)
	}
	if frame := readFrame(); !strings.Contains(frame, "message_delta") {
		t.Fatalf("frame after finish = %q, want message_delta", frame)
	}
	if frame := readFrame(); !strings.Contains(frame, "message_stop") {
		t.Fatalf("frame after finish = %q, want message_stop", frame)
	}
}
