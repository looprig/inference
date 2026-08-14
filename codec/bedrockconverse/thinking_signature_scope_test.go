package bedrockconverse

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/model"
)

// This file is the Bedrock half of the reasoning-signature provenance rule; see
// anthropicapi/thinking_signature_scope_test.go for the live evidence that the
// issuing API verifies the signature cryptographically and rejects both a
// tampered one and an absent one.
//
// Bedrock Converse is the case that makes the rule necessary rather than
// theoretical. It serves the same Claude models as the Anthropic Messages API,
// its reasoningText block is structurally the same block, and its signature is
// not portable to that other endpoint. Nothing but a provenance label can tell
// the two apart.

// foreignSignatureFormat stands for a dialect that is not this one — here, the
// direct Anthropic endpoint for the very same models.
const foreignSignatureFormat = "anthropic"

func signedAssistantRequest(block *content.ThinkingBlock) inference.Request {
	return inference.Request{
		Model: model.Model{Name: "anthropic.claude-sonnet-5", APIFormat: model.APIFormatBedrockConverse},
		Messages: content.AgenticMessages{&content.AIMessage{Message: content.Message{
			Role:   content.RoleAssistant,
			Blocks: []content.Block{block},
		}}},
	}
}

// wireReasoning returns the single reasoningContent block of an encoded body.
func wireReasoning(t *testing.T, body []byte) map[string]json.RawMessage {
	t.Helper()
	var decoded struct {
		Messages []struct {
			Content []map[string]json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	if len(decoded.Messages) != 1 || len(decoded.Messages[0].Content) != 1 {
		t.Fatalf("body = %s, want one message with one block", body)
	}
	var reasoning struct {
		ReasoningText map[string]json.RawMessage `json:"reasoningText"`
	}
	if err := json.Unmarshal(decoded.Messages[0].Content[0]["reasoningContent"], &reasoning); err != nil {
		t.Fatalf("unmarshal reasoningContent: %v", err)
	}
	return reasoning.ReasoningText
}

// TestEncodeRequest_RefusesAReasoningSignatureMintedByAnotherDialect is the
// mirror of the decisive Anthropic test: an Anthropic-minted signature replayed
// toward Bedrock is refused rather than sent.
func TestEncodeRequest_RefusesAReasoningSignatureMintedByAnotherDialect(t *testing.T) {
	t.Parallel()

	block := content.NewSignedThinkingBlock(
		"reasoning that Anthropic signed", "anthropic-minted-signature", foreignSignatureFormat, nil, "")

	body, err := EncodeRequest(signedAssistantRequest(block))
	if err == nil {
		t.Fatalf("EncodeRequest() encoded a foreign-minted signature into %s, want a refusal", body)
	}
	var foreign *ForeignReasoningSignatureError
	if !errors.As(err, &foreign) {
		t.Fatalf("EncodeRequest() error = %v (%T), want *ForeignReasoningSignatureError", err, err)
	}
	if foreign.Format != foreignSignatureFormat {
		t.Errorf("ForeignReasoningSignatureError.Format = %q, want %q", foreign.Format, foreignSignatureFormat)
	}
	if !strings.Contains(err.Error(), foreignSignatureFormat) {
		t.Errorf("error %q does not name the foreign dialect", err)
	}
}

// TestEncodeRequest_RefusesAnUntaggedReasoningSignature: an unlabelled
// signature has no provable issuer, so it fails closed like a foreign one.
func TestEncodeRequest_RefusesAnUntaggedReasoningSignature(t *testing.T) {
	t.Parallel()

	block := &content.ThinkingBlock{Thinking: "reasoning", Signature: "signature-of-unknown-origin"}

	body, err := EncodeRequest(signedAssistantRequest(block))
	if err == nil {
		t.Fatalf("EncodeRequest() encoded an untagged signature into %s, want a refusal", body)
	}
	var foreign *ForeignReasoningSignatureError
	if !errors.As(err, &foreign) {
		t.Fatalf("EncodeRequest() error = %v (%T), want *ForeignReasoningSignatureError", err, err)
	}
}

// TestEncodeRequest_ReplaysItsOwnSignatureVerbatim is the same-dialect control.
func TestEncodeRequest_ReplaysItsOwnSignatureVerbatim(t *testing.T) {
	t.Parallel()

	const signature = "EpUBCkYIBRgCKkC+opaque/signature+with==padding"
	block := content.NewSignedThinkingBlock("reasoning", signature, signatureFormatBedrockConverse, nil, "")

	body, err := EncodeRequest(signedAssistantRequest(block))
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	var got string
	if err := json.Unmarshal(wireReasoning(t, body)["signature"], &got); err != nil {
		t.Fatalf("unmarshal wire signature: %v", err)
	}
	if got != signature {
		t.Errorf("wire signature = %q, want %q byte-for-byte", got, signature)
	}
}

// TestEncodeRequest_UnsignedReasoningIsStillEncodable keeps the refusal narrow.
func TestEncodeRequest_UnsignedReasoningIsStillEncodable(t *testing.T) {
	t.Parallel()

	body, err := EncodeRequest(signedAssistantRequest(&content.ThinkingBlock{Thinking: "reasoning"}))
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	if _, ok := wireReasoning(t, body)["text"]; !ok {
		t.Errorf("body = %s, want a reasoningText block", body)
	}
}

// TestDecodeResponse_StampsThisDialectOnTheSignature proves the producing half.
func TestDecodeResponse_StampsThisDialectOnTheSignature(t *testing.T) {
	t.Parallel()

	resp, err := DecodeResponse([]byte(`{"output":{"message":{"role":"assistant","content":[` +
		`{"reasoningContent":{"reasoningText":{"text":"reasoning","signature":"sig-abc"}}}]}},` +
		`"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`))
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	tb, ok := resp.Message.Blocks[0].(*content.ThinkingBlock)
	if !ok {
		t.Fatalf("Blocks[0] type = %T, want *content.ThinkingBlock", resp.Message.Blocks[0])
	}
	got, replayable := tb.SignatureReplayableAs(signatureFormatBedrockConverse)
	if !replayable || got != "sig-abc" {
		t.Fatalf("SignatureReplayableAs(%q) = (%q, %v), want (%q, true)",
			signatureFormatBedrockConverse, got, replayable, "sig-abc")
	}
	if _, replayable := tb.SignatureReplayableAs(foreignSignatureFormat); replayable {
		t.Errorf("a Bedrock-minted signature reports itself replayable as %q", foreignSignatureFormat)
	}
}

// TestDecodeResponse_RedactedReasoningCarriesNoSignatureLabel guards the
// coexistence rule: redacted reasoning has an opaque payload and no signature,
// so the signature label stays empty.
func TestDecodeResponse_RedactedReasoningCarriesNoSignatureLabel(t *testing.T) {
	t.Parallel()

	resp, err := DecodeResponse([]byte(`{"output":{"message":{"role":"assistant","content":[` +
		`{"reasoningContent":{"redactedContent":"b3BhcXVl"}}]}},` +
		`"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`))
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	tb := resp.Message.Blocks[0].(*content.ThinkingBlock)
	if tb.Signature != "" || tb.SignatureFormat != "" {
		t.Errorf("redacted reasoning decoded with Signature=%q SignatureFormat=%q, want both empty",
			tb.Signature, tb.SignatureFormat)
	}
	if !tb.ReplayableAs(providerStateFormatBedrockRedacted) {
		t.Error("redacted reasoning lost its opaque provider state")
	}
}

// TestDecodeStream_StampsThisDialectOnASignatureDelta is the streaming
// producer: streaming must reconstruct the same continuation state, provenance
// included, as the non-streaming decoder.
func TestDecodeStream_StampsThisDialectOnASignatureDelta(t *testing.T) {
	t.Parallel()

	collector := &streamResultCollector{
		active: make(map[int]streamBlockKind),
		closed: make(map[int]struct{}),
	}
	if err := collector.messageStart([]byte(`{"role":"assistant"}`)); err != nil {
		t.Fatalf("messageStart() error = %v", err)
	}
	if _, err := collector.contentBlockDelta([]byte(
		`{"contentBlockIndex":0,"delta":{"reasoningContent":{"text":"reasoning"}}}`)); err != nil {
		t.Fatalf("contentBlockDelta(text) error = %v", err)
	}
	chunks, err := collector.contentBlockDelta([]byte(
		`{"contentBlockIndex":0,"delta":{"reasoningContent":{"signature":"sig-stream"}}}`))
	if err != nil {
		t.Fatalf("contentBlockDelta(signature) error = %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("DecodeEvent() returned %d chunks, want 1", len(chunks))
	}
	tc := chunks[0].(*content.ThinkingChunk)
	if got, ok := tc.SignatureReplayableAs(signatureFormatBedrockConverse); !ok || got != "sig-stream" {
		t.Fatalf("SignatureReplayableAs(%q) = (%q, %v), want (%q, true)",
			signatureFormatBedrockConverse, got, ok, "sig-stream")
	}
}
