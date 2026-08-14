package openairesponses_test

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference/codec/openairesponses"
)

// The two tests below pin OPPOSITE answers for the same decoder
// (wireItemContent.UnmarshalJSON, which decodes every item's `content` array
// on all four of this codec's paths), because the two directions have opposite
// failure modes. The reasoning lives on wireItemContent in types.go; these
// tests are what stop either half from being "simplified" into the other.

// TestServerDecode_UnknownContentPartMemberIsRejectedBecauseIngressMustNotSilentlyDropCallerIntent
// covers the gateway's request ingress. Every other level of that decode is
// strict (DisallowUnknownFields on the body, which reaches the item level
// because wireItem has no hand-written UnmarshalJSON), and the content part
// was the one level that was not: a client sending a misspelled member had the
// member — and with it the text — dropped, and got a 200 for a prompt that was
// never delivered.
func TestServerDecode_UnknownContentPartMemberIsRejectedBecauseIngressMustNotSilentlyDropCallerIntent(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		body string
	}{
		{
			// The measured case: `txet` is dropped and the request decodes to
			// an EMPTY text block.
			name: "misspelled text member loses the whole prompt",
			body: `{"model":"gpt-5","input":[{"type":"message","role":"user",` +
				`"content":[{"type":"input_text","txet":"hello"}]}]}`,
		},
		{
			name: "unmodelled member alongside a good one",
			body: `{"model":"gpt-5","input":[{"type":"message","role":"user",` +
				`"content":[{"type":"input_text","text":"hi","bogus":1}]}]}`,
		},
		{
			// input_file's file_url is a real spec member this codec does not
			// model. Failing closed names it; the semantic check that fires
			// today reports only that file_data is missing.
			name: "spec member this codec does not model",
			body: `{"model":"gpt-5","input":[{"type":"message","role":"user",` +
				`"content":[{"type":"input_file","file_url":"https://example.test/y.pdf"}]}]}`,
		},
		{
			name: "assistant replay part",
			body: `{"model":"gpt-5","input":[{"type":"message","role":"assistant",` +
				`"content":[{"type":"output_text","text":"hi","annotations":[],"logprobs":[],"bogus":1}]}]}`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			req := httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader([]byte(tc.body)))
			req.Header.Set("Content-Type", "application/json")
			decoded, err := (openairesponses.Codec{}).DecodeRequest(req)
			if err == nil {
				t.Fatalf("DecodeRequest accepted an unknown content-part member; decoded %d turns from %s",
					len(decoded.Request.Messages), tc.body)
			}
			if !strings.Contains(err.Error(), "content_part") {
				t.Errorf("error = %v, want one naming the content part", err)
			}
		})
	}
}

// TestServerDecode_BenignContentPartMembersAreAcceptedAndIgnored keeps the
// strictness above from becoming a 400 for a member that costs the caller
// nothing to lose. prompt_cache_breakpoint is a provider-side performance
// hint, not content; this gateway cannot honour it and dropping it degrades
// nothing the caller can observe in the reply. It is the content-part
// counterpart of the request-level benign fields (parallel_tool_calls,
// include, metadata) that wireDecodeRequest already accepts and drops.
func TestServerDecode_BenignContentPartMembersAreAcceptedAndIgnored(t *testing.T) {
	t.Parallel()

	body := `{"model":"gpt-5","input":[{"type":"message","role":"user",` +
		`"content":[{"type":"input_text","text":"hi","prompt_cache_breakpoint":{"type":"auto"}}]}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")
	decoded, err := (openairesponses.Codec{}).DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest rejected a benign cache hint: %v", err)
	}
	if len(decoded.Request.Messages) != 1 {
		t.Fatalf("decoded %d turns, want 1", len(decoded.Request.Messages))
	}
	user, ok := decoded.Request.Messages[0].(*content.UserMessage)
	if !ok {
		t.Fatalf("turn is %T, want *content.UserMessage", decoded.Request.Messages[0])
	}
	if len(user.Blocks) != 1 {
		t.Fatalf("decoded %d blocks, want 1", len(user.Blocks))
	}
	text, ok := user.Blocks[0].(*content.TextBlock)
	if !ok || text.Text != "hi" {
		t.Errorf("block = %#v, want the text intact", user.Blocks[0])
	}
}

// TestDecodeResponse_UnknownContentPartMemberIsAcceptedBecauseAProviderAdditionMustNotBreakInference
// covers the opposite direction through the same decoder. OpenAI adds members
// to Responses content parts — OutputTextContent gained logprobs, which this
// codec had to grow a field for — and a strict client decode turns the next
// such addition into a hard inference failure on a response that is perfectly
// usable. Content already received is never discarded over a member we do not
// model.
func TestDecodeResponse_UnknownContentPartMemberIsAcceptedBecauseAProviderAdditionMustNotBreakInference(t *testing.T) {
	t.Parallel()

	body := []byte(`{"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"gpt-5",` +
		`"output":[{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[` +
		`{"type":"output_text","text":"hello","annotations":[],"logprobs":[],` +
		`"a_member_openai_adds_next_year":{"nested":true}}]}]}`)

	resp, err := openairesponses.DecodeResponse(body)
	if err != nil {
		t.Fatalf("DecodeResponse rejected a response over an unmodelled content-part member: %v", err)
	}
	if resp.Message == nil || len(resp.Message.Blocks) != 1 {
		t.Fatalf("decoded %#v, want one block", resp.Message)
	}
	text, ok := resp.Message.Blocks[0].(*content.TextBlock)
	if !ok || text.Text != "hello" {
		t.Errorf("block = %#v, want the text intact", resp.Message.Blocks[0])
	}
}
