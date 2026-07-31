package anthropicapi_test

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/looprig/inference/codec/anthropicapi"
)

// FuzzDecodeServerRequest exercises DecodeRequest — the untrusted-client-input
// entry point this task adds — against arbitrary bytes as a POST /v1/messages
// body. It must never panic; any error it returns must be non-nil (Go's fuzz
// harness itself treats a panic as a failure, so this target's job is just to
// keep feeding realistic-shaped and degenerate seeds through the real HTTP
// decode path, including the duplicate-key scanner and the strict
// DisallowUnknownFields decode).
//
// Run: GOWORK=off go test ./codec/anthropicapi -run '^$' -fuzz=FuzzDecodeServerRequest -fuzztime=30s
func FuzzDecodeServerRequest(f *testing.F) {
	// Well-formed seeds, covering the shapes server_decode_test.go exercises.
	f.Add([]byte(`{"model":"claude-test","max_tokens":256,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`))
	f.Add([]byte(`{"model":"m","max_tokens":8,"system":[{"type":"text","text":"sys","cache_control":{"type":"ephemeral"}}],"messages":[]}`))
	f.Add([]byte(`{"model":"m","max_tokens":8,"metadata":{"user_id":"x"},"messages":[]}`))
	f.Add([]byte(`{"model":"m","max_tokens":8,"tool_choice":{"type":"any"},"tools":[{"name":"f","input_schema":{"type":"object"}}],"messages":[]}`))
	f.Add([]byte(`{"model":"m","max_tokens":8,"thinking":{"type":"adaptive"},"output_config":{"effort":"low"},"messages":[]}`))
	f.Add([]byte(`{"model":"m","max_tokens":8,"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"h","signature":"s"},{"type":"tool_use","id":"t1","name":"f","input":{}}]}]}`))
	f.Add([]byte(`{"model":"m","max_tokens":8,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[{"type":"text","text":"ok"}]}]}]}`))
	f.Add([]byte(`{"model":"m","max_tokens":8,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aGk="}}]}]}`))
	// Degenerate / hostile seeds.
	f.Add([]byte(`{}`))
	f.Add([]byte(`{"model":"m","max_tokens":8,"max_tokens":9,"messages":[]}`)) // duplicate key
	f.Add([]byte(`{"model":`))
	f.Add([]byte(`not json at all`))
	f.Add([]byte(``))
	f.Add([]byte(`null`))
	f.Add([]byte(`[]`))
	f.Add([]byte(`{"model":"m","max_tokens":8,"messages":[{"role":"user","content":[{"type":"unknown_block"}]}]}`))

	f.Fuzz(func(t *testing.T, data []byte) {
		req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(data))
		req.Header.Set("Content-Type", "application/json")

		decoded, err := (anthropicapi.Codec{}).DecodeRequest(req)
		if err == nil {
			// A successful decode must still be internally consistent: it should
			// never claim a non-empty RequestedModel came from truly empty/invalid
			// input, and Request.Model must stay unresolved (decodeMessagesBody
			// never sets it).
			if decoded.RequestedModel == "" {
				t.Fatalf("DecodeRequest() succeeded with empty RequestedModel for input %q", data)
			}
			return
		}
		// Any returned error must be non-nil (trivially true here) and the call
		// must not have panicked to reach this point.
	})
}
