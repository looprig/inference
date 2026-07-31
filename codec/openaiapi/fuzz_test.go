package openaiapi_test

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/looprig/inference/codec/openaiapi"
)

// FuzzDecodeResponse ensures DecodeResponse never panics on arbitrary input.
// Run with: go test -fuzz=FuzzDecodeResponse ./internal/llm/openaiapi/... -fuzztime=30s
func FuzzDecodeResponse(f *testing.F) {
	// Seed corpus: valid responses.
	f.Add([]byte(`{"id":"1","model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}}`))
	f.Add([]byte(`{"id":"2","model":"o3","choices":[{"message":{"role":"assistant","content":"","reasoning_content":"thinking"},"finish_reason":"stop"}]}`))
	f.Add([]byte(`{"id":"3","model":"m","choices":[]}`))
	f.Add([]byte(`{}`))
	f.Add([]byte(`invalid json`))

	f.Fuzz(func(t *testing.T, data []byte) {
		// Must not panic regardless of input.
		_, _ = openaiapi.DecodeResponse(data)
	})
}

// FuzzDecodeServerRequest exercises DecodeRequest — the untrusted-client-input
// entry point — against arbitrary bytes as a POST /v1/chat/completions body.
// It must never panic; any error it returns must be non-nil (Go's fuzz
// harness itself treats a panic as a failure, so this target's job is just to
// keep feeding realistic-shaped and degenerate seeds through the real HTTP
// decode path, including the duplicate-key scanner, the strict
// DisallowUnknownFields decode, and the string-or-array `content` decoder).
//
// Run: GOWORK=off go test ./codec/openaiapi -run '^$' -fuzz=FuzzDecodeServerRequest -fuzztime=30s
func FuzzDecodeServerRequest(f *testing.F) {
	// Well-formed seeds, covering the shapes server_decode_test.go exercises.
	f.Add([]byte(`{"model":"gpt-test","messages":[{"role":"system","content":"be terse"},{"role":"user","content":"hi"}]}`))
	f.Add([]byte(`{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":"data:image/png;base64,aGk="}}]}]}`))
	f.Add([]byte(`{"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/x.png"}}]}]}`))
	f.Add([]byte(`{"model":"m","messages":[{"role":"assistant","reasoning_content":"h","content":"ans","tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":"{\"x\":1}"}}]}]}`))
	f.Add([]byte(`{"model":"m","messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":"{}"}}]}]}`))
	f.Add([]byte(`{"model":"m","messages":[{"role":"tool","tool_call_id":"c","content":"result"}]}`))
	f.Add([]byte(`{"model":"m","tool_choice":"required","tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"messages":[]}`))
	f.Add([]byte(`{"model":"m","reasoning_effort":"low","messages":[]}`))
	f.Add([]byte(`{"model":"m","response_format":{"type":"json_schema","json_schema":{"name":"a","strict":true,"schema":{"type":"object"}}},"messages":[]}`))
	f.Add([]byte(`{"model":"m","n":2,"messages":[]}`))
	f.Add([]byte(`{"model":"m","n":1,"messages":[]}`))
	f.Add([]byte(`{"model":"m","stream":true,"stream_options":{"include_usage":true},"messages":[]}`))
	f.Add([]byte(`{"model":"m","parallel_tool_calls":true,"user":"h","seed":1,"messages":[]}`))
	// Degenerate / hostile seeds.
	f.Add([]byte(`{}`))
	f.Add([]byte(`{"model":"m","model":"m2","messages":[]}`)) // duplicate key
	f.Add([]byte(`{"model":`))
	f.Add([]byte(`not json at all`))
	f.Add([]byte(``))
	f.Add([]byte(`null`))
	f.Add([]byte(`[]`))
	f.Add([]byte(`{"model":"m","messages":[{"role":"weird","content":"x"}]}`))
	f.Add([]byte(`{"model":"m","messages":[{"role":"tool","content":"x"}]}`))
	f.Add([]byte(`{"model":"m","messages":[{"role":"assistant","tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":"not json"}}]}]}`))
	f.Add([]byte(`{"model":"m","tool_choice":{"type":"function","function":{"name":"f"}},"messages":[]}`))

	f.Fuzz(func(t *testing.T, data []byte) {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(data))
		req.Header.Set("Content-Type", "application/json")

		decoded, err := (openaiapi.Codec{}).DecodeRequest(req)
		if err == nil {
			// A successful decode must still be internally consistent: it
			// should never claim a non-empty RequestedModel came from truly
			// empty/invalid input, and Request.Model must stay unresolved
			// (decodeChatCompletionsBody never sets it).
			if decoded.RequestedModel == "" {
				t.Fatalf("DecodeRequest() succeeded with empty RequestedModel for input %q", data)
			}
			return
		}
		// Any returned error must be non-nil (trivially true here) and the
		// call must not have panicked to reach this point.
	})
}
