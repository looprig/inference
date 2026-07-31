package geminiapi_test

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/looprig/inference/codec/geminiapi"
)

// FuzzDecode ensures the two untrusted-input parsers — DecodeResponse (a full
// generateContent body) and Codec.DecodeEvent (one de-framed streamGenerateContent
// chunk) — never panic on arbitrary bytes. Both are fed each input because either
// can receive hostile or truncated provider data. A single target keeps
// `-fuzz=Fuzz` matching exactly one test (Go refuses to fuzz when more than one
// matches).
//
// Run: GOWORK=off go test -run '^$' -fuzz=Fuzz -fuzztime=30s ./pkg/llm/codec/gemini/
func FuzzDecode(f *testing.F) {
	// Response / chunk-shaped seeds (identical shape for generateContent + stream).
	f.Add([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2},"modelVersion":"gemini-2.5-flash"}`))
	f.Add([]byte(`{"candidates":[{"content":{"parts":[{"functionCall":{"id":"c1","name":"run","args":{"x":1}}}]}}]}`))
	f.Add([]byte(`{"candidates":[{"content":{"parts":[{"text":"thought","thought":true},{"text":"answer"}]}}]}`))
	f.Add([]byte(`{"candidates":[]}`))
	// Degenerate seeds.
	f.Add([]byte(`{}`))
	f.Add([]byte(`invalid json`))
	f.Add([]byte(``))

	f.Fuzz(func(t *testing.T, data []byte) {
		// Must not panic regardless of input; error returns are expected and ignored.
		_, _ = geminiapi.DecodeResponse(data)
		_, _ = geminiapi.Codec{}.DecodeEvent(data)
	})
}

// FuzzDecodeServerRequest exercises DecodeRequest — the untrusted-client-input
// entry point — against arbitrary bytes as a POST
// /v1beta/models/gemini-test:generateContent body. It must never panic; any
// error it returns must be non-nil (Go's fuzz harness itself treats a panic
// as a failure, so this target's job is just to keep feeding realistic-shaped
// and degenerate seeds through the real HTTP decode path, including the
// duplicate-key scanner and the strict DisallowUnknownFields decode).
//
// Run: GOWORK=off go test ./codec/geminiapi -run '^$' -fuzz=FuzzDecodeServerRequest -fuzztime=30s
func FuzzDecodeServerRequest(f *testing.F) {
	// Well-formed seeds, covering the shapes server_decode_test.go exercises.
	f.Add([]byte(`{"systemInstruction":{"parts":[{"text":"be terse"}]},"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`))
	f.Add([]byte(`{"contents":[{"role":"user","parts":[{"inlineData":{"mimeType":"image/png","data":"aGk="}}]}]}`))
	f.Add([]byte(`{"contents":[{"role":"user","parts":[{"fileData":{"mimeType":"image/png","fileUri":"https://x/y.png"}}]}]}`))
	f.Add([]byte(`{"contents":[{"role":"model","parts":[{"text":"planning","thought":true,"thoughtSignature":"sig"},{"text":"ans"},{"functionCall":{"id":"c","name":"f","args":{"x":1}}}]}]}`))
	f.Add([]byte(`{"contents":[{"role":"user","parts":[{"functionResponse":{"id":"c","name":"f","response":{"result":"ok"}}}]}]}`))
	f.Add([]byte(`{"contents":[],"tools":[{"functionDeclarations":[{"name":"f","parameters":{"type":"object"}}]}],"toolConfig":{"functionCallingConfig":{"mode":"ANY"}}}`))
	f.Add([]byte(`{"contents":[],"generationConfig":{"thinkingConfig":{"thinkingBudget":4096,"includeThoughts":true}}}`))
	f.Add([]byte(`{"contents":[],"generationConfig":{"responseMimeType":"application/json","responseJsonSchema":{"type":"object"}}}`))
	f.Add([]byte(`{"contents":[],"candidateCount":2}`))
	// Degenerate / hostile seeds.
	f.Add([]byte(`{}`))
	f.Add([]byte(`{"contents":[],"contents":[]}`)) // duplicate key
	f.Add([]byte(`{"contents":`))
	f.Add([]byte(`not json at all`))
	f.Add([]byte(``))
	f.Add([]byte(`null`))
	f.Add([]byte(`[]`))
	f.Add([]byte(`{"contents":[{"role":"weird","parts":[{"text":"x"}]}]}`))
	f.Add([]byte(`{"contents":[{"role":"user","parts":[{"functionResponse":{"name":"f","response":"not an object"}}]}]}`))

	f.Fuzz(func(t *testing.T, data []byte) {
		req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-test:generateContent", bytes.NewReader(data))
		req.Header.Set("Content-Type", "application/json")

		_, err := (geminiapi.Codec{}).DecodeRequest(req)
		// Any returned error must be non-nil (trivially true here) and the
		// call must not have panicked to reach this point. Unlike the other
		// three dialects, RequestedModel here comes from the URL path (fixed
		// in this fuzz target), not the body, so a successful decode's
		// RequestedModel is always "gemini-test" regardless of data — there
		// is no body-derived invariant analogous to their empty-RequestedModel
		// check to assert here.
		_ = err
	})
}
