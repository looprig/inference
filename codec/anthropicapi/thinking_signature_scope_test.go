package anthropicapi

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/model"
)

// This file pins the scoping of ThinkingBlock.Signature to the dialect that
// MINTED it.
//
// A reasoning signature is cryptographically validated by its issuer. Probed
// against api.anthropic.com with claude-haiku-4-5 on 2026-08-13: a real
// signature replayed verbatim is accepted (HTTP 200); the same signature with
// eight characters changed is rejected with
// `messages.1.content.0: Invalid "signature" in "thinking" block` (HTTP 400);
// and an EMPTY signature draws the identical 400. Those three results are the
// whole argument for what follows.
//
// Bedrock Converse and the Anthropic Messages API serve the SAME Claude model
// family. Their thinking blocks are structurally identical and their
// signatures are not interchangeable, so the dangerous case is invisible to
// every check except a provenance label. And because the empty-signature probe
// fails too, dropping a foreign signature is NOT a safe degrade: it converts a
// request that would be rejected for the wrong signature into a request
// rejected for a missing one. The only correct behaviour is to fail closed
// locally, where the diagnostic can name the foreign dialect.

// foreignSignatureFormat stands for any dialect that is not this one. Bedrock
// Converse is the realistic attacker-free case, not a hypothetical: it is the
// other endpoint for these very models.
const foreignSignatureFormat = "bedrock-converse"

func signedAssistantRequest(block *content.ThinkingBlock) inference.Request {
	return inference.Request{
		Model: model.Model{Name: "claude-sonnet-5", APIFormat: model.APIFormatAnthropic},
		Messages: content.AgenticMessages{&content.AIMessage{Message: content.Message{
			Role:   content.RoleAssistant,
			Blocks: []content.Block{block},
		}}},
	}
}

// wireThinkingBlock returns the single content block of the single message in
// an encoded request body.
func wireThinkingBlock(t *testing.T, body []byte) map[string]json.RawMessage {
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
	return decoded.Messages[0].Content[0]
}

// TestEncodeRequest_RefusesAThinkingSignatureMintedByAnotherDialect is the
// decisive test of this change. It replays a Bedrock-tagged signature through
// the Anthropic request encoder and requires a refusal.
//
// Before the fix the encoder copied Signature unconditionally, so this body
// went to api.anthropic.com and came back 400 — after the request had already
// been billed for and after the loop had committed to the turn.
func TestEncodeRequest_RefusesAThinkingSignatureMintedByAnotherDialect(t *testing.T) {
	t.Parallel()

	block := content.NewSignedThinkingBlock(
		"reasoning that Bedrock signed", "bedrock-minted-signature", foreignSignatureFormat, nil, "")

	body, err := EncodeRequest(signedAssistantRequest(block), false)
	if err == nil {
		t.Fatalf("EncodeRequest() encoded a foreign-minted signature into %s, want a refusal", body)
	}
	var foreign *ForeignThinkingSignatureError
	if !errors.As(err, &foreign) {
		t.Fatalf("EncodeRequest() error = %v (%T), want *ForeignThinkingSignatureError", err, err)
	}
	if foreign.Format != foreignSignatureFormat {
		t.Errorf("ForeignThinkingSignatureError.Format = %q, want %q", foreign.Format, foreignSignatureFormat)
	}
	if !strings.Contains(err.Error(), foreignSignatureFormat) {
		t.Errorf("error %q does not name the foreign dialect; the diagnostic is the whole value of failing here", err)
	}
}

// TestEncodeRequest_RefusesAnUntaggedThinkingSignature covers the other half of
// "cannot prove I minted it". An unlabelled signature has no provable issuer,
// so it is refused for the same reason a foreign one is. Treating it as
// trustworthy would leave the defect wide open behind a decoder that simply
// forgot to tag.
func TestEncodeRequest_RefusesAnUntaggedThinkingSignature(t *testing.T) {
	t.Parallel()

	block := &content.ThinkingBlock{Thinking: "reasoning", Signature: "signature-of-unknown-origin"}

	body, err := EncodeRequest(signedAssistantRequest(block), false)
	if err == nil {
		t.Fatalf("EncodeRequest() encoded an untagged signature into %s, want a refusal", body)
	}
	var foreign *ForeignThinkingSignatureError
	if !errors.As(err, &foreign) {
		t.Fatalf("EncodeRequest() error = %v (%T), want *ForeignThinkingSignatureError", err, err)
	}
	if foreign.Format != "" {
		t.Errorf("ForeignThinkingSignatureError.Format = %q, want the empty label of an untagged signature", foreign.Format)
	}
}

// TestEncodeRequest_ReplaysItsOwnSignatureVerbatim is the same-dialect control.
// The refusal above is only worth anything if the legitimate path still works,
// and "works" for a cryptographic seal means byte-for-byte: the live probe
// showed an eight-character edit is rejected.
func TestEncodeRequest_ReplaysItsOwnSignatureVerbatim(t *testing.T) {
	t.Parallel()

	const signature = "ErUBCkYIBRgCKkC+opaque/signature+with==padding"
	block := content.NewSignedThinkingBlock("reasoning", signature, signatureFormatAnthropic, nil, "")

	body, err := EncodeRequest(signedAssistantRequest(block), false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	wire := wireThinkingBlock(t, body)
	var got string
	if err := json.Unmarshal(wire["signature"], &got); err != nil {
		t.Fatalf("unmarshal wire signature: %v", err)
	}
	if got != signature {
		t.Errorf("wire signature = %q, want %q byte-for-byte", got, signature)
	}
}

// TestEncodeRequest_UnsignedThinkingIsStillEncodable keeps the refusal narrow.
// A block with no signature at all is not a provenance failure — it is a
// reasoning block from a dialect that seals nothing, or a partial one — and it
// must keep encoding exactly as before.
func TestEncodeRequest_UnsignedThinkingIsStillEncodable(t *testing.T) {
	t.Parallel()

	body, err := EncodeRequest(signedAssistantRequest(&content.ThinkingBlock{Thinking: "reasoning"}), false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	if _, ok := wireThinkingBlock(t, body)["thinking"]; !ok {
		t.Errorf("body = %s, want a thinking block", body)
	}
}

// TestDecodeResponse_StampsThisDialectOnTheSignature proves the producing half.
// A refusal at the encoder is only correct if this dialect's own decoder labels
// what it mints; otherwise the fix would refuse every real Anthropic turn.
func TestDecodeResponse_StampsThisDialectOnTheSignature(t *testing.T) {
	t.Parallel()

	resp, err := DecodeResponse([]byte(`{"id":"msg_1","model":"claude-sonnet-5","role":"assistant",` +
		`"content":[{"type":"thinking","thinking":"reasoning","signature":"sig-abc"}],` +
		`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	tb, ok := resp.Message.Blocks[0].(*content.ThinkingBlock)
	if !ok {
		t.Fatalf("Blocks[0] type = %T, want *content.ThinkingBlock", resp.Message.Blocks[0])
	}
	got, replayable := tb.SignatureReplayableAs(signatureFormatAnthropic)
	if !replayable || got != "sig-abc" {
		t.Fatalf("SignatureReplayableAs(%q) = (%q, %v), want (%q, true)",
			signatureFormatAnthropic, got, replayable, "sig-abc")
	}
	if _, replayable := tb.SignatureReplayableAs(foreignSignatureFormat); replayable {
		t.Errorf("an Anthropic-minted signature reports itself replayable as %q", foreignSignatureFormat)
	}
}

// TestDecodeResponse_RedactedThinkingCarriesNoSignatureLabel guards the
// coexistence rule. Redacted thinking has an opaque payload and NO signature,
// so the signature label must stay empty rather than being stamped on for
// symmetry — a label on nothing is the invalid state the constructors exist to
// prevent.
func TestDecodeResponse_RedactedThinkingCarriesNoSignatureLabel(t *testing.T) {
	t.Parallel()

	resp, err := DecodeResponse([]byte(`{"id":"msg_1","model":"claude-sonnet-5","role":"assistant",` +
		`"content":[{"type":"redacted_thinking","data":"opaque+/=payload"}],` +
		`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	tb := resp.Message.Blocks[0].(*content.ThinkingBlock)
	if tb.Signature != "" || tb.SignatureFormat != "" {
		t.Errorf("redacted thinking decoded with Signature=%q SignatureFormat=%q, want both empty",
			tb.Signature, tb.SignatureFormat)
	}
	if !tb.ReplayableAs(providerStateFormatAnthropicRedacted) {
		t.Errorf("redacted thinking lost its opaque provider state; the signature change must not disturb it")
	}
}

// TestDecodeEvent_StampsThisDialectOnASignatureDelta is the streaming producer.
// The module rule is that streaming reconstructs the SAME continuation state as
// the non-streaming decoder; provenance is part of that state, so a signature
// delta has to arrive labelled or the streamed turn becomes unreplayable while
// the non-streamed one stays fine.
func TestDecodeEvent_StampsThisDialectOnASignatureDelta(t *testing.T) {
	t.Parallel()

	chunks, err := decodeEvent([]byte(
		`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-stream"}}`))
	if err != nil {
		t.Fatalf("decodeEvent() error = %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("decodeEvent() returned %d chunks, want 1", len(chunks))
	}
	tc, ok := chunks[0].(*content.ThinkingChunk)
	if !ok {
		t.Fatalf("chunk type = %T, want *content.ThinkingChunk", chunks[0])
	}
	got, replayable := tc.SignatureReplayableAs(signatureFormatAnthropic)
	if !replayable || got != "sig-stream" {
		t.Fatalf("SignatureReplayableAs(%q) = (%q, %v), want (%q, true)",
			signatureFormatAnthropic, got, replayable, "sig-stream")
	}
}

// TestDecodeRequestBlock_StampsThisDialectOnAReplayedSignature covers the
// gateway's ingress. A client replaying a prior assistant turn to this gateway
// is replaying an ANTHROPIC-minted signature, so that is the label it earns —
// and it is the label the outbound Anthropic encoder will then accept.
func TestDecodeRequestBlock_StampsThisDialectOnAReplayedSignature(t *testing.T) {
	t.Parallel()

	block, err := decodeRequestBlock(anthropicBlock{
		Type: blockTypeThinking, Thinking: "reasoning", Signature: "sig-replayed",
	}, false)
	if err != nil {
		t.Fatalf("decodeRequestBlock() error = %v", err)
	}
	tb := block.(*content.ThinkingBlock)
	if got, ok := tb.SignatureReplayableAs(signatureFormatAnthropic); !ok || got != "sig-replayed" {
		t.Fatalf("SignatureReplayableAs(%q) = (%q, %v), want (%q, true)",
			signatureFormatAnthropic, got, ok, "sig-replayed")
	}
}

// TestEncodeResponseBlock_RefusesAForeignMintedSignature is the gateway's
// egress. Serving another dialect's signature to a client is strictly worse
// than refusing: the client stores it, replays it on the next turn, and the
// failure surfaces one turn later against a different component.
func TestEncodeResponseBlock_RefusesAForeignMintedSignature(t *testing.T) {
	t.Parallel()

	block := content.NewSignedThinkingBlock("reasoning", "sig", foreignSignatureFormat, nil, "")
	if _, err := encodeResponseBlock(block, func() string { return "id" }); err == nil {
		t.Fatal("encodeResponseBlock() served a foreign-minted signature, want a refusal")
	}

	own := content.NewSignedThinkingBlock("reasoning", "sig", signatureFormatAnthropic, nil, "")
	wire, err := encodeResponseBlock(own, func() string { return "id" })
	if err != nil {
		t.Fatalf("encodeResponseBlock() error = %v on this dialect's own signature", err)
	}
	if wire.Signature != "sig" {
		t.Errorf("served signature = %q, want %q", wire.Signature, "sig")
	}
}

// TestWriteChunk_RefusesAForeignMintedSignature is the streaming counterpart of
// the test above. The three egress paths — request encode, response encode and
// stream encode — must agree, or the same turn is refused or forwarded
// depending only on whether the client asked to stream.
func TestWriteChunk_RefusesAForeignMintedSignature(t *testing.T) {
	t.Parallel()

	foreign := &content.ThinkingChunk{
		Thinking: "reasoning", Signature: "sig", SignatureFormat: foreignSignatureFormat,
	}
	if err := writeChunkTo(httptest.NewRecorder(), foreign); err == nil {
		t.Fatal("WriteChunk() streamed a foreign-minted signature, want a refusal")
	}

	own := &content.ThinkingChunk{
		Thinking: "reasoning", Signature: "sig", SignatureFormat: signatureFormatAnthropic,
	}
	rec := httptest.NewRecorder()
	if err := writeChunkTo(rec, own); err != nil {
		t.Fatalf("WriteChunk() error = %v on this dialect's own signature", err)
	}
	if !strings.Contains(rec.Body.String(), `"signature_delta"`) {
		t.Errorf("stream body = %s, want a signature_delta", rec.Body.String())
	}
}

// writeChunkTo opens a server stream over w and writes one chunk to it.
func writeChunkTo(w http.ResponseWriter, chunk content.Chunk) error {
	encoder, err := openMessagesStream(w)
	if err != nil {
		return err
	}
	return encoder.WriteChunk(chunk)
}
