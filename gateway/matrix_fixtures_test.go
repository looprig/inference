package gateway_test

// This file provides the shared machinery for the Task 14 dialect-matrix
// tests (matrix_test.go) and the concurrency stress tests
// (concurrency_test.go): a per-dialect adapter table that knows how to speak
// each of the four bundled wire dialects both as an ingress (native
// harness-facing HTTP) and as a fake outbound target (a real httptest.Server
// driven by that SAME dialect's server-side codec.ServerCodec methods), plus
// one portable-subset fixture request/response pair reused across most of
// the 4x4 matrix.
//
// The "clever reuse" this leans on (see the plan): every bundled dialect's
// Codec value implements BOTH codec.Codec (client-side: EncodeRequest /
// DecodeResponse -- "I am a harness talking to a real target") AND
// codec.ServerCodec (server-side: DecodeRequest / WriteResponse / OpenStream
// -- "I am a target receiving a harness's request"). So:
//
//   - a NATIVE ingress request is built with a dialect's own client-side
//     EncodeRequest -- the same code a real harness for that dialect would
//     run -- rather than hand-written JSON;
//   - a NATIVE ingress response is read back with that same dialect's
//     client-side DecodeResponse, rather than hand-rolled JSON assertions;
//   - a fake target server for a dialect is just an http.HandlerFunc that
//     calls that dialect's own server-side DecodeRequest/WriteResponse/
//     OpenStream, exactly like the real gateway.Handler does for ingress.
import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	"github.com/looprig/inference/codec"
	"github.com/looprig/inference/codec/anthropicapi"
	"github.com/looprig/inference/codec/geminiapi"
	"github.com/looprig/inference/codec/openaiapi"
	"github.com/looprig/inference/codec/openairesponses"
	"github.com/looprig/inference/gateway"
	"github.com/looprig/inference/model"
	"github.com/looprig/inference/route"
	"github.com/looprig/inference/stream"
	"github.com/looprig/inference/transport"
)

// matrixToken is the gateway-local bearer token used by every matrix and
// concurrency test's Handler.
const matrixToken = "matrix-test-token"

// matrixAlias is the harness-requested model alias used by every matrix and
// concurrency test's single configured route.
const matrixAlias = "primary"

// fullDialectCodec is the intersection every bundled dialect's Codec{}
// value satisfies: both the client-side (outbound) and server-side
// (inbound) surfaces. Storing dialectAdapter.codec at this type lets the
// SAME value serve as both the ingress ServerCodec and, via transport.New,
// the outbound codec.Codec for a fake target -- with streaming automatically
// enabled on the transport.Client, since every bundled Codec also satisfies
// codec.StreamDecoder.
type fullDialectCodec interface {
	codec.ServerCodec
	codec.Codec
}

// dialectAdapter bundles everything the matrix/concurrency tests need to
// drive one API dialect both as an ingress and as an outbound target.
type dialectAdapter struct {
	format model.APIFormat
	codec  fullDialectCodec

	// ingressPath returns the native HTTP path a harness speaking this
	// dialect would POST alias's request to. Every dialect except Gemini
	// ignores alias/streaming (the model travels in the body); Gemini's
	// model-in-path convention uses both.
	ingressPath func(alias string, streaming bool) string

	// encodeRequest builds this dialect's native wire body for a neutral
	// request using the SAME client-side encoder a real harness would use.
	encodeRequest func(req inference.Request, streaming bool) ([]byte, error)

	// decodeResponse parses this dialect's native non-streaming response
	// body back into a neutral inference.Response using the SAME
	// client-side decoder a real harness would use.
	decodeResponse func(body []byte) (*inference.Response, error)

	// targetEndpoint builds the (BaseURL, Router) pair a transport.Client
	// needs to reach a fake target server of this dialect running at srv.
	targetEndpoint func(srv *httptest.Server) (baseURL string, router route.Router)
}

// matrixDialects is the 4-entry adapter table the matrix loop ranges over.
var matrixDialects = map[model.APIFormat]dialectAdapter{
	model.APIFormatAnthropic: {
		format: model.APIFormatAnthropic,
		codec:  anthropicapi.Codec{},
		ingressPath: func(string, bool) string {
			return "/v1/messages"
		},
		encodeRequest:  anthropicapi.EncodeRequest,
		decodeResponse: anthropicapi.DecodeResponse,
		targetEndpoint: func(srv *httptest.Server) (string, route.Router) {
			return srv.URL, route.StaticChat("/v1/messages")
		},
	},
	model.APIFormatOpenAI: {
		format: model.APIFormatOpenAI,
		codec:  openaiapi.Codec{},
		ingressPath: func(string, bool) string {
			return "/v1/chat/completions"
		},
		encodeRequest:  openaiapi.EncodeRequest,
		decodeResponse: openaiapi.DecodeResponse,
		targetEndpoint: func(srv *httptest.Server) (string, route.Router) {
			return srv.URL, route.StaticChat("/v1/chat/completions")
		},
	},
	model.APIFormatOpenAIResponses: {
		format: model.APIFormatOpenAIResponses,
		codec:  openairesponses.Codec{},
		ingressPath: func(string, bool) string {
			return "/v1/responses"
		},
		encodeRequest:  openairesponses.EncodeRequest,
		decodeResponse: openairesponses.DecodeResponse,
		targetEndpoint: func(srv *httptest.Server) (string, route.Router) {
			return srv.URL, route.StaticChat("/v1/responses")
		},
	},
	model.APIFormatGemini: {
		format: model.APIFormatGemini,
		codec:  geminiapi.Codec{},
		ingressPath: func(alias string, streaming bool) string {
			if streaming {
				return "/v1beta/models/" + alias + ":streamGenerateContent?alt=sse"
			}
			return "/v1beta/models/" + alias + ":generateContent"
		},
		encodeRequest: func(req inference.Request, _ bool) ([]byte, error) {
			// Gemini's generateContent/streamGenerateContent bodies are
			// identical -- streaming is chosen by path, not a body flag.
			return geminiapi.EncodeRequest(req)
		},
		decodeResponse: geminiapi.DecodeResponse,
		targetEndpoint: func(srv *httptest.Server) (string, route.Router) {
			// geminiapi's server-side MatchRequest requires the literal
			// "/v1beta/models/{model}:generateContent" path prefix
			// (modelPathPrefix); route.GeminiGenerateContent only appends
			// "/models/{model}:generateContent" to the base, so "/v1beta"
			// must already be part of BaseURL, exactly as a real caller
			// pointed at Google's actual endpoint would supply it.
			return srv.URL + "/v1beta", route.GeminiGenerateContent()
		},
	},
}

// matrixFormats lists the 4 dialects in a fixed order for deterministic
// test iteration/naming.
var matrixFormats = []model.APIFormat{
	model.APIFormatAnthropic,
	model.APIFormatOpenAIResponses,
	model.APIFormatOpenAI,
	model.APIFormatGemini,
}

// --- fake target server ------------------------------------------------

// fakeTarget is a minimal stand-in target server for one dialect, built
// entirely from that dialect's own server-side codec.ServerCodec methods
// (see the package doc comment above). It records every codec.DecodedRequest
// it decodes so a test can assert what the target actually received --
// system instructions, message thread, tool calls, opaque thinking state --
// not just that the round trip didn't error.
type fakeTarget struct {
	sc codec.ServerCodec

	mu        sync.Mutex
	requests  []codec.DecodedRequest
	rawBodies [][]byte

	respond   func(codec.DecodedRequest) (*inference.Response, error)
	respondSC func(codec.DecodedRequest) ([]content.Chunk, stream.StreamResult, error)
}

// newFakeTarget starts an httptest.Server for dialect sc and returns it
// alongside the fakeTarget controlling its canned behavior. The server is
// closed automatically via t.Cleanup.
func newFakeTarget(t *testing.T, sc codec.ServerCodec) (*httptest.Server, *fakeTarget) {
	t.Helper()
	ft := &fakeTarget{sc: sc}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Capture the raw wire bytes BEFORE decoding (and restore r.Body for
		// DecodeRequest) so a test can assert byte-level absence of a
		// cross-dialect opaque secret, not just absence from the semantic
		// DecodedRequest a well-behaved decoder happens to produce.
		var raw []byte
		if r.Body != nil {
			raw, _ = io.ReadAll(r.Body)
			r.Body = io.NopCloser(bytes.NewReader(raw))
		}

		decoded, err := sc.DecodeRequest(r)
		if err != nil {
			sc.WriteError(w, err)
			return
		}
		ft.mu.Lock()
		ft.requests = append(ft.requests, decoded)
		ft.rawBodies = append(ft.rawBodies, raw)
		respond, respondSC := ft.respond, ft.respondSC
		ft.mu.Unlock()

		if decoded.Streaming {
			if respondSC == nil {
				sc.WriteError(w, errors.New("faketarget: no streaming responder configured"))
				return
			}
			chunks, result, err := respondSC(decoded)
			if err != nil {
				sc.WriteError(w, err)
				return
			}
			enc, err := sc.OpenStream(w)
			if err != nil {
				return
			}
			for _, c := range chunks {
				if err := enc.WriteChunk(c); err != nil {
					return
				}
			}
			_ = enc.Finish(result)
			return
		}

		if respond == nil {
			sc.WriteError(w, errors.New("faketarget: no responder configured"))
			return
		}
		resp, err := respond(decoded)
		if err != nil {
			sc.WriteError(w, err)
			return
		}
		_ = sc.WriteResponse(w, resp)
	}))
	t.Cleanup(srv.Close)
	return srv, ft
}

// setResponse configures a single canned non-streaming response returned
// for every subsequent request.
func (ft *fakeTarget) setResponse(resp *inference.Response) {
	ft.mu.Lock()
	defer ft.mu.Unlock()
	ft.respond = func(codec.DecodedRequest) (*inference.Response, error) { return resp, nil }
}

// setResponseFunc configures a per-request responder, letting a test
// compute the canned response from what the target actually decoded (e.g.
// echoing the caller's text back so concurrent requests can be verified
// against their own input rather than a single shared canned value).
func (ft *fakeTarget) setResponseFunc(fn func(codec.DecodedRequest) (*inference.Response, error)) {
	ft.mu.Lock()
	defer ft.mu.Unlock()
	ft.respond = fn
}

// setStreamResponse configures the canned streaming chunks/terminal result
// returned for every subsequent streaming request.
func (ft *fakeTarget) setStreamResponse(chunks []content.Chunk, result stream.StreamResult) {
	ft.mu.Lock()
	defer ft.mu.Unlock()
	ft.respondSC = func(codec.DecodedRequest) ([]content.Chunk, stream.StreamResult, error) {
		return chunks, result, nil
	}
}

// requestsSnapshot returns a defensive copy of every request captured so far.
func (ft *fakeTarget) requestsSnapshot() []codec.DecodedRequest {
	ft.mu.Lock()
	defer ft.mu.Unlock()
	return append([]codec.DecodedRequest(nil), ft.requests...)
}

// lastRequest returns the most recently captured request, failing the test
// if none has arrived yet.
func (ft *fakeTarget) lastRequest(t *testing.T) codec.DecodedRequest {
	t.Helper()
	reqs := ft.requestsSnapshot()
	if len(reqs) == 0 {
		t.Fatal("faketarget: no request captured")
	}
	return reqs[len(reqs)-1]
}

// callCount reports how many requests this target has decoded so far.
func (ft *fakeTarget) callCount() int {
	ft.mu.Lock()
	defer ft.mu.Unlock()
	return len(ft.requests)
}

// lastRawBody returns the raw wire bytes of the most recently received
// request, failing the test if none has arrived yet.
func (ft *fakeTarget) lastRawBody(t *testing.T) []byte {
	t.Helper()
	ft.mu.Lock()
	defer ft.mu.Unlock()
	if len(ft.rawBodies) == 0 {
		t.Fatal("faketarget: no request captured")
	}
	return ft.rawBodies[len(ft.rawBodies)-1]
}

// --- search helpers for opaque-thinking-state assertions -------------------

// findFirstThinkingBlock returns the first *content.ThinkingBlock in blocks,
// or nil.
func findFirstThinkingBlock(blocks []content.Block) *content.ThinkingBlock {
	for _, b := range blocks {
		if tb, ok := b.(*content.ThinkingBlock); ok {
			return tb
		}
	}
	return nil
}

// findFirstThinkingBlockInMessages returns the first *content.ThinkingBlock
// found in any AIMessage across msgs, or nil.
func findFirstThinkingBlockInMessages(msgs content.AgenticMessages) *content.ThinkingBlock {
	for _, m := range msgs {
		ai, ok := m.(*content.AIMessage)
		if !ok {
			continue
		}
		if tb := findFirstThinkingBlock(ai.Blocks); tb != nil {
			return tb
		}
	}
	return nil
}

// thinkingOpaqueSubstringPresent reports whether ANY ThinkingBlock across
// msgs carries substr in its Signature or ProviderState -- used to prove a
// cross-dialect opaque secret was NOT forwarded into a differently-dialected
// target's decoded request.
func thinkingOpaqueSubstringPresent(msgs content.AgenticMessages, substr string) bool {
	for _, m := range msgs {
		ai, ok := m.(*content.AIMessage)
		if !ok {
			continue
		}
		for _, b := range ai.Blocks {
			tb, ok := b.(*content.ThinkingBlock)
			if !ok {
				continue
			}
			if strings.Contains(tb.Signature, substr) {
				return true
			}
			if strings.Contains(string(tb.ProviderState), substr) {
				return true
			}
		}
	}
	return false
}

// --- wiring helpers ------------------------------------------------------

// buildMatrixTarget builds a gateway.Target reaching the fake target server
// srv for dialect d, using a REAL transport.Client (transport.New) bound
// with that dialect's own Router/Codec -- exactly the composition a real
// deployment would use, just pointed at a fake httptest.Server instead of a
// real provider.
func buildMatrixTarget(t *testing.T, d dialectAdapter, srv *httptest.Server, modelName string, opts ...model.ModelOption) gateway.Target {
	t.Helper()
	baseURL, router := d.targetEndpoint(srv)
	provider := model.ProviderName("faketarget-" + string(d.format))
	ep := transport.Endpoint{BaseURL: baseURL, Provider: provider, APIFormat: d.format}
	client := transport.New(ep, router, d.codec, auth.None())
	m := model.CustomModel(provider, d.format, baseURL, modelName, opts...)
	return gateway.Target{ID: "target-" + string(d.format), Client: client, Model: m}
}

// buildMatrixHandler builds a gateway.Handler configured with exactly one
// ingress dialect (ingress) and exactly one route: (ingress.format,
// matrixAlias) -> target.
func buildMatrixHandler(t *testing.T, ingress dialectAdapter, target gateway.Target) *gateway.Handler {
	t.Helper()
	resolver, err := gateway.NewMux(gateway.Mux{
		Routes: map[gateway.RouteKey]gateway.Target{
			{Ingress: ingress.format, Model: matrixAlias}: target,
		},
	})
	if err != nil {
		t.Fatalf("NewMux: %v", err)
	}
	h, err := gateway.New(gateway.Config{
		Resolver:     resolver,
		Codecs:       map[model.APIFormat]codec.ServerCodec{ingress.format: ingress.codec},
		Authenticate: gateway.StaticToken(matrixToken),
	})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	return h
}

// sendMatrixInvoke drives one full non-streaming ingress round trip against
// the matrix's single fixed alias (matrixAlias). See sendIngressInvoke for
// the general form used by the concurrency tests, which need to vary the
// alias per call.
func sendMatrixInvoke(t *testing.T, h *gateway.Handler, ingress dialectAdapter, req inference.Request) (*httptest.ResponseRecorder, *inference.Response) {
	t.Helper()
	return sendIngressInvoke(t, h, ingress, matrixAlias, req, false)
}

// sendIngressInvoke drives one full ingress round trip: it encodes req as
// ingress's native wire body (via that dialect's own client-side
// EncodeRequest, addressed to alias), POSTs it to h at ingress's native path
// for streaming/non-streaming, and -- on a 200, non-streaming response --
// decodes the native response body back to a neutral inference.Response via
// ingress's own client-side DecodeResponse. It always returns the raw
// recorder so a caller can inspect non-200 outcomes, or a streaming
// response's raw SSE body, too; the returned *inference.Response is nil for
// a non-200 status OR a streaming request (whose body isn't a plain
// single-shot JSON response DecodeResponse can parse).
func sendIngressInvoke(t *testing.T, h *gateway.Handler, ingress dialectAdapter, alias string, req inference.Request, streaming bool) (*httptest.ResponseRecorder, *inference.Response) {
	t.Helper()
	req.Model.Name = alias
	body, err := ingress.encodeRequest(req, streaming)
	if err != nil {
		t.Fatalf("%s EncodeRequest: %v", ingress.format, err)
	}
	httpReq := httptest.NewRequest(http.MethodPost, ingress.ingressPath(alias, streaming), bytes.NewReader(body))
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+matrixToken)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, httpReq)
	if rr.Code != http.StatusOK || streaming {
		return rr, nil
	}
	resp, err := ingress.decodeResponse(rr.Body.Bytes())
	if err != nil {
		t.Fatalf("%s DecodeResponse: %v (body=%s)", ingress.format, err, rr.Body.String())
	}
	return rr, resp
}

// buildMultiRouteHandler builds a gateway.Handler configured with one
// ingress dialect and one route per (alias -> target) entry in routes --
// used by the concurrency tests, which need more than the matrix's single
// fixed route.
func buildMultiRouteHandler(t *testing.T, ingress dialectAdapter, routes map[string]gateway.Target) *gateway.Handler {
	t.Helper()
	muxRoutes := make(map[gateway.RouteKey]gateway.Target, len(routes))
	for alias, target := range routes {
		muxRoutes[gateway.RouteKey{Ingress: ingress.format, Model: alias}] = target
	}
	resolver, err := gateway.NewMux(gateway.Mux{Routes: muxRoutes})
	if err != nil {
		t.Fatalf("NewMux: %v", err)
	}
	h, err := gateway.New(gateway.Config{
		Resolver:     resolver,
		Codecs:       map[model.APIFormat]codec.ServerCodec{ingress.format: ingress.codec},
		Authenticate: gateway.StaticToken(matrixToken),
		// Generous: the concurrency suite is proving isolation/no-crosstalk
		// under load, not admission-limit behavior (see limits_test.go for
		// that).
		MaxConcurrent: 512,
	})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	return h
}

// textResponse builds a minimal canned non-streaming response carrying only
// text and a stop finish reason -- used by the concurrency tests, which care
// about isolation, not the fuller portable-subset fidelity the matrix tests
// already cover.
func textResponse(text string) *inference.Response {
	return &inference.Response{
		Message: &content.AIMessage{Message: content.Message{
			Role:   content.RoleAssistant,
			Blocks: []content.Block{&content.TextBlock{Text: text}},
		}},
		FinishReason: stream.FinishReasonStop,
	}
}

// textRequest builds a minimal single-user-turn request carrying text.
func textRequest(text string) inference.Request {
	return inference.Request{
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{
				Role:   content.RoleUser,
				Blocks: []content.Block{&content.TextBlock{Text: text}},
			}},
		},
	}
}

// broadCaps opts every fake target's Model into every gating capability this
// matrix might exercise (images, thinking, tools, structured output). Caps
// are local gating data only (never sent on the wire, per model.Model's doc
// comment), and this suite is about dialect translation fidelity, not
// capability gating -- so every target is opted in broadly to avoid spurious
// *ImageInputUnsupportedError-shaped failures unrelated to what a given cell
// actually tests.
func broadCaps() []model.ModelOption {
	return []model.ModelOption{
		model.WithImages(),
		// A declared dialect, not a bare WithThinking: the Anthropic encoder
		// fails closed on an undeclared one, and this matrix measures dialect
		// translation rather than catalogue completeness.
		model.WithThinkingDialect(model.ThinkingDialectAdaptive),
		model.WithTools(),
	}
}

// --- portable-subset fixture ----------------------------------------------
//
// portableFixtureRequest is the ONE representative fixture reused across
// most of the 4x4 matrix: system instructions, a user turn, an assistant
// turn with a tool call, a tool result, and a final user turn -- the
// portable subset every bundled dialect can express losslessly.

func portableFixtureRequest() inference.Request {
	return inference.Request{
		System: "You are a terse weather assistant.",
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{
				Role:   content.RoleUser,
				Blocks: []content.Block{&content.TextBlock{Text: "What's the weather in nyc?"}},
			}},
			&content.AIMessage{Message: content.Message{
				Role: content.RoleAssistant,
				Blocks: []content.Block{
					&content.TextBlock{Text: "Let me check."},
					&content.ToolUseBlock{ID: "call_1", Name: "get_weather", Input: json.RawMessage(`{"city":"nyc"}`)},
				},
			}},
			&content.ToolResultMessage{
				Message:   content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "sunny, 65F"}}},
				ToolUseID: "call_1",
			},
			&content.UserMessage{Message: content.Message{
				Role:   content.RoleUser,
				Blocks: []content.Block{&content.TextBlock{Text: "Great, thanks!"}},
			}},
		},
		Tools: []inference.Tool{
			{
				Name:        "get_weather",
				Description: "Get the current weather for a city",
				Schema:      json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
			},
		},
	}
}

// portableCannedTextResponse is the fake target's canned reply to the
// portable fixture: plain text, usage, and a stop finish reason.
func portableCannedTextResponse() *inference.Response {
	return &inference.Response{
		Message: &content.AIMessage{Message: content.Message{
			Role:   content.RoleAssistant,
			Blocks: []content.Block{&content.TextBlock{Text: "The weather in nyc is sunny, 65F."}},
		}},
		Usage:        &content.Usage{InputTokens: 123, OutputTokens: 45},
		FinishReason: stream.FinishReasonStop,
	}
}

// cannedToolCallResponse is a fake target's canned reply carrying a single
// tool call and a tool_use finish reason.
func cannedToolCallResponse() *inference.Response {
	return &inference.Response{
		Message: &content.AIMessage{Message: content.Message{
			Role:   content.RoleAssistant,
			Blocks: []content.Block{&content.ToolUseBlock{ID: "call_9", Name: "get_time", Input: json.RawMessage(`{"tz":"utc"}`)}},
		}},
		Usage:        &content.Usage{InputTokens: 50, OutputTokens: 20},
		FinishReason: stream.FinishReasonToolUse,
	}
}

// cannedParallelToolCallResponse is a fake target's canned reply carrying
// TWO parallel tool calls with distinct IDs, proving index/ID association
// survives translation.
func cannedParallelToolCallResponse() *inference.Response {
	return &inference.Response{
		Message: &content.AIMessage{Message: content.Message{
			Role: content.RoleAssistant,
			Blocks: []content.Block{
				&content.ToolUseBlock{ID: "call_a", Name: "get_weather", Input: json.RawMessage(`{"city":"nyc"}`)},
				&content.ToolUseBlock{ID: "call_b", Name: "get_time", Input: json.RawMessage(`{"tz":"utc"}`)},
			},
		}},
		Usage:        &content.Usage{InputTokens: 60, OutputTokens: 30},
		FinishReason: stream.FinishReasonToolUse,
	}
}

// cannedThinkingResponse is a fake target's canned reply carrying VISIBLE
// reasoning text (no opaque provider state) followed by the final answer,
// proving visible thinking survives translation as visible text.
func cannedThinkingResponse() *inference.Response {
	return &inference.Response{
		Message: &content.AIMessage{Message: content.Message{
			Role: content.RoleAssistant,
			Blocks: []content.Block{
				&content.ThinkingBlock{Thinking: "Step 1: recall nyc weather. Step 2: answer."},
				&content.TextBlock{Text: "42"},
			},
		}},
		Usage:        &content.Usage{InputTokens: 70, OutputTokens: 35},
		FinishReason: stream.FinishReasonStop,
	}
}

// tinyPNGBytes is a minimal (not necessarily valid past the signature)
// inline image payload used to exercise image translation without needing a
// real decodable PNG -- no codec in this suite parses image pixel content,
// only carries the bytes.
var tinyPNGBytes = []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x01, 0x02, 0x03}

// imageFixtureRequest is a minimal request whose sole user turn carries text
// plus one inline image. Model.Caps.AcceptsImages is set on the request
// itself (mirroring what a real image-capable harness would know about its
// own configured model) so this fixture's own client-side ingress
// EncodeRequest -- which fail-safe validates image support via
// inference.ValidateRequestFeatures before ever reaching the gateway --
// does not reject it; the gateway then independently re-validates against
// the resolved TARGET model's own Caps (see broadCaps).
func imageFixtureRequest() inference.Request {
	return inference.Request{
		Model: model.Model{Caps: model.Capabilities{AcceptsImages: true}},
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{
				Role: content.RoleUser,
				Blocks: []content.Block{
					&content.TextBlock{Text: "What is in this image?"},
					&content.ImageBlock{MediaType: content.MediaTypeImagePNG, Source: content.ImageSource{Data: tinyPNGBytes}},
				},
			}},
		},
	}
}

// dialectName is a small helper for collision-free subtest names.
func dialectName(f model.APIFormat) string { return string(f) }
