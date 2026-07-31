package openairesponses_test

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/looprig/inference/codec/openairesponses"
)

// FuzzDecodeServerRequest exercises DecodeRequest — the untrusted-client-input
// entry point — against arbitrary bytes as a POST /v1/responses body. It must
// never panic; any error it returns must be non-nil (Go's fuzz harness itself
// treats a panic as a failure, so this target's job is just to keep feeding
// realistic-shaped and degenerate seeds through the real HTTP decode path,
// including the duplicate-key scanner and the strict DisallowUnknownFields
// decode).
//
// Run: GOWORK=off go test ./codec/openairesponses -run '^$' -fuzz=FuzzDecodeServerRequest -fuzztime=30s
func FuzzDecodeServerRequest(f *testing.F) {
	// Well-formed seeds, covering the shapes server_decode_test.go exercises.
	f.Add([]byte(`{"model":"gpt-test","instructions":"be terse","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}`))
	f.Add([]byte(`{"model":"m","input":[{"type":"message","role":"system","content":[{"type":"input_text","text":"sys"}]}]}`))
	f.Add([]byte(`{"model":"m","metadata":{"user_id":"x"},"input":[]}`))
	f.Add([]byte(`{"model":"m","tool_choice":"required","tools":[{"type":"function","name":"f","parameters":{"type":"object"}}],"input":[]}`))
	f.Add([]byte(`{"model":"m","reasoning":{"effort":"low","summary":"auto"},"input":[]}`))
	f.Add([]byte(`{"model":"m","input":[{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"h"}],"encrypted_content":"s"},{"type":"function_call","call_id":"call_1","name":"f","arguments":"{}"}]}`))
	f.Add([]byte(`{"model":"m","input":[{"type":"function_call_output","call_id":"call_1","output":"ok"}]}`))
	f.Add([]byte(`{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,aGk=","detail":"auto"}]}]}`))
	f.Add([]byte(`{"model":"m","text":{"format":{"type":"json_schema","name":"a","schema":{"type":"object","properties":{},"required":[],"additionalProperties":false}}},"input":[]}`))
	f.Add([]byte(`{"model":"m","store":true,"input":[]}`))
	f.Add([]byte(`{"model":"m","previous_response_id":"resp_1","input":[]}`))
	// Degenerate / hostile seeds.
	f.Add([]byte(`{}`))
	f.Add([]byte(`{"model":"m","model":"m2","input":[]}`)) // duplicate key
	f.Add([]byte(`{"model":`))
	f.Add([]byte(`not json at all`))
	f.Add([]byte(``))
	f.Add([]byte(`null`))
	f.Add([]byte(`[]`))
	f.Add([]byte(`{"model":"m","input":[{"type":"unknown_item"}]}`))
	f.Add([]byte(`{"model":"m","input":[{"type":"function_call","call_id":"c","name":"f","arguments":"not json"}]}`))

	f.Fuzz(func(t *testing.T, data []byte) {
		req := httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(data))
		req.Header.Set("Content-Type", "application/json")

		decoded, err := (openairesponses.Codec{}).DecodeRequest(req)
		if err == nil {
			// A successful decode must still be internally consistent: it
			// should never claim a non-empty RequestedModel came from truly
			// empty/invalid input, and Request.Model must stay unresolved
			// (decodeResponsesBody never sets it).
			if decoded.RequestedModel == "" {
				t.Fatalf("DecodeRequest() succeeded with empty RequestedModel for input %q", data)
			}
			return
		}
		// Any returned error must be non-nil (trivially true here) and the
		// call must not have panicked to reach this point.
	})
}

// FuzzDecodeResponse ensures the two untrusted-input parsers — DecodeResponse
// (a full Responses response body) and Codec.DecodeEvent (one de-framed SSE
// event) — never panic on arbitrary bytes.
//
// Run: GOWORK=off go test -run '^$' -fuzz=FuzzDecodeResponse -fuzztime=30s ./codec/openairesponses/
func FuzzDecodeResponse(f *testing.F) {
	f.Add([]byte(`{"id":"1","status":"completed","model":"gpt-test","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}]}`))
	f.Add([]byte(`{"id":"1","status":"failed","model":"gpt-test","output":[],"error":{"code":"server_error","message":"boom"}}`))
	f.Add([]byte(`{"id":"1","status":"completed","model":"gpt-test","output":[{"type":"function_call","call_id":"c","name":"f","arguments":"{}"}]}`))
	f.Add([]byte(`{"id":"1","status":"completed","model":"gpt-test","output":[{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"h"}],"encrypted_content":"s"}]}`))
	f.Add([]byte(`{"output":[]}`))
	// Event-shaped seeds.
	f.Add([]byte(`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"c","name":"f"}}`))
	f.Add([]byte(`{"type":"response.output_text.delta","output_index":0,"delta":"hi"}`))
	f.Add([]byte(`{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"x\":"}`))
	f.Add([]byte(`{"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"hm"}`))
	f.Add([]byte(`{"type":"response.completed"}`))
	f.Add([]byte(`{"type":"response.failed","response":{"error":{"code":"e","message":"m"}}}`))
	// Degenerate seeds.
	f.Add([]byte(`{}`))
	f.Add([]byte(`invalid json`))
	f.Add([]byte(``))

	f.Fuzz(func(t *testing.T, data []byte) {
		// Must not panic regardless of input; error returns are expected and
		// ignored.
		_, _ = openairesponses.DecodeResponse(data)
		_, _ = openairesponses.Codec{}.DecodeEvent(data)
	})
}
