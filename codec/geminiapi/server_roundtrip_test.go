package geminiapi_test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec"
	"github.com/looprig/inference/codec/geminiapi"
	"github.com/looprig/inference/codec/servertest"
	model "github.com/looprig/inference/model"
	stream "github.com/looprig/inference/stream"
	usage "github.com/looprig/inference/usage"
)

// httpRequestFor builds an httptest POST request for path carrying body,
// with the application/json content type. Unlike newDecodeRequest
// (server_decode_test.go, fixed to the :generateContent suffix), this takes
// an arbitrary path so tests can exercise the sibling
// :streamGenerateContent route too.
func httpRequestFor(t *testing.T, path, body string) *http.Request {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")
	return req
}

// TestGeminiServerCodec_SatisfiesContract runs the reusable codec/servertest
// suite against the real geminiapi.Codec, per the plan: every dialect's real
// ServerCodec is expected to call servertest.Run against itself the same way
// anthropicapi, openairesponses, and openaiapi do. Gemini's Codec legitimately
// owns TWO distinct routes; per the task's own guidance, the shared suite is
// pinned to the non-streaming :generateContent route (Method/Path), and
// UnmatchedPath is a route neither variant matches (NOT the sibling
// :streamGenerateContent route, which this codec correctly DOES match — see
// TestGeminiServerCodec_StreamingRouteMatchesAndSetsStreaming for dedicated
// coverage of that route).
func TestGeminiServerCodec_SatisfiesContract(t *testing.T) {
	servertest.Run(t, servertest.Config{
		NewCodec: func() codec.ServerCodec { return geminiapi.Codec{} },
		Method:   http.MethodPost,
		Path:     "/v1beta/models/gemini-test:generateContent",

		ValidBody: []byte(`{
			"systemInstruction": {"parts": [{"text": "be terse"}]},
			"contents": [
				{"role":"user","parts":[{"text":"what's the weather in nyc and the time in utc?"}]},
				{"role":"model","parts":[
					{"functionCall":{"id":"call_1","name":"get_weather","args":{"city":"nyc"}}},
					{"functionCall":{"id":"call_2","name":"get_time","args":{"tz":"utc"}}}
				]},
				{"role":"user","parts":[{"functionResponse":{"id":"call_1","name":"get_weather","response":{"result":"sunny"}}}]},
				{"role":"user","parts":[{"functionResponse":{"id":"call_2","name":"get_time","response":{"result":"12:00"}}}]}
			],
			"tools": [{"functionDeclarations":[{"name":"get_weather","description":"gets weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}]}]
		}`),

		UnmatchedMethod:  http.MethodGet,
		UnmatchedPath:    "/v1beta/models/gemini-test",
		WrongContentType: "text/plain",
		MalformedBody:    []byte(`{"contents":`),

		SampleResponse: &inference.Response{
			Message: &content.AIMessage{
				Message: content.Message{
					Role:   content.RoleAssistant,
					Blocks: []content.Block{&content.TextBlock{Text: "the weather is sunny and it is 12:00 utc"}},
				},
			},
			Model:        "gemini-test",
			Usage:        &usage.Usage{InputTokens: 42, OutputTokens: 11},
			FinishReason: stream.FinishReasonStop,
		},

		SampleChunks: []content.Chunk{
			&content.TextChunk{Text: "the weather is "},
			&content.TextChunk{Text: "sunny"},
			&content.ToolUseChunk{Index: 0, ID: "call_1", Name: "get_weather", InputJSON: `{"city":"nyc"}`},
		},

		SampleResult: stream.StreamResult{
			Model:        "gemini-test",
			FinishReason: stream.FinishReasonToolUse,
			Usage:        &usage.Usage{InputTokens: 42, OutputTokens: 11},
		},

		SampleError: &geminiapi.ServerDecodeError{Reason: "missing_model"},
	})
}

// TestGeminiServerCodec_StreamingRouteMatchesAndSetsStreaming proves the
// second, streaming-only route this codec owns is also matched and decoded
// correctly — the dialect-specific coverage the shared suite's single
// Method/Path pair cannot express.
func TestGeminiServerCodec_StreamingRouteMatchesAndSetsStreaming(t *testing.T) {
	t.Parallel()
	c := geminiapi.Codec{}
	// newDecodeRequest (server_decode_test.go) always builds
	// /v1beta/models/{model}:generateContent; build the streaming variant
	// directly here instead, since the suffix itself differs.
	req := httpRequestFor(t, "/v1beta/models/gemini-test:streamGenerateContent?alt=sse", `{"contents":[]}`)

	if !c.MatchRequest(req) {
		t.Fatal("MatchRequest(:streamGenerateContent) = false, want true")
	}
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if decoded.RequestedModel != "gemini-test" {
		t.Errorf("RequestedModel = %q, want gemini-test", decoded.RequestedModel)
	}
	if !decoded.Streaming {
		t.Error("Streaming = false, want true")
	}
}

// TestGeminiServerCodec_SameDialectRoundTrip encodes a neutral request with
// the existing client-direction EncodeRequest, decodes it back with the new
// server-direction DecodeRequest, and checks the result is semantically
// equivalent to the original — including the thoughtSignature round trip
// through ThinkingBlock.ProviderState, the concrete scenario the design
// calls out for opaque per-provider reasoning state.
func TestGeminiServerCodec_SameDialectRoundTrip(t *testing.T) {
	t.Parallel()

	providerState := json.RawMessage(`"opaque-thought-sig"`)
	original := inference.Request{
		Model: model.Model{Name: "gemini-test"},
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{
				Role:   content.RoleUser,
				Blocks: []content.Block{&content.TextBlock{Text: "solve this"}},
			}},
			&content.AIMessage{Message: content.Message{
				Role: content.RoleAssistant,
				Blocks: []content.Block{
					content.NewThinkingBlock("step by step", "", providerState, "gemini"),
					&content.ToolUseBlock{ID: "call_1", Name: "calc", Input: json.RawMessage(`{"x":1}`)},
				},
			}},
			&content.ToolResultMessage{
				Message:   content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "42"}}},
				ToolUseID: "call_1",
			},
		},
	}

	body, err := geminiapi.EncodeRequest(original)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}

	req := httpRequestFor(t, "/v1beta/models/gemini-test:generateContent", string(body))
	decoded, err := (geminiapi.Codec{}).DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v (body=%s)", err, body)
	}

	got := decoded.Request
	if len(got.Messages) != len(original.Messages) {
		t.Fatalf("Messages len = %d, want %d", len(got.Messages), len(original.Messages))
	}

	ai, ok := got.Messages[1].(*content.AIMessage)
	if !ok {
		t.Fatalf("Messages[1] type = %T, want *content.AIMessage", got.Messages[1])
	}
	tb, ok := ai.Blocks[0].(*content.ThinkingBlock)
	if !ok {
		t.Fatalf("Blocks[0] type = %T, want *content.ThinkingBlock", ai.Blocks[0])
	}
	if tb.Thinking != "step by step" {
		t.Errorf("Thinking = %q", tb.Thinking)
	}
	var opaque string
	if err := json.Unmarshal(tb.ProviderState, &opaque); err != nil {
		t.Fatalf("ProviderState not a JSON string: %v", err)
	}
	if opaque != "opaque-thought-sig" {
		t.Errorf("ProviderState = %q, want opaque-thought-sig (byte-for-byte replay)", opaque)
	}

	tu, ok := ai.Blocks[1].(*content.ToolUseBlock)
	if !ok || tu.ID != "call_1" || tu.Name != "calc" {
		t.Errorf("Blocks[1] = %#v", ai.Blocks[1])
	}

	tr, ok := got.Messages[2].(*content.ToolResultMessage)
	if !ok || tr.ToolUseID != "call_1" {
		t.Errorf("Messages[2] = %#v", got.Messages[2])
	}
}
