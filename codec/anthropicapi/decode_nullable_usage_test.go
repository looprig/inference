package anthropicapi

import (
	"errors"
	"testing"

	"github.com/looprig/inference/usage"
)

// Anthropic declares cache_read_input_tokens and cache_creation_input_tokens as
// anyOf[{integer, minimum 0}, {null}] and lists both as required members. Null
// is therefore a schema-valid uncached value, not a malformed count. Rejecting
// it discards an otherwise complete response over an accounting field, so both
// normalize to zero.
func TestDecodeResponse_NullCacheTokensNormalizeToZero(t *testing.T) {
	body := []byte(`{
		"id": "msg_1",
		"type": "message",
		"role": "assistant",
		"model": "claude-sonnet-5",
		"content": [{"type": "text", "text": "hello"}],
		"stop_reason": "end_turn",
		"usage": {
			"input_tokens": 12,
			"output_tokens": 7,
			"cache_read_input_tokens": null,
			"cache_creation_input_tokens": null
		}
	}`)

	resp, err := DecodeResponse(body)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if got := resp.Usage.CacheReadTokens; got != 0 {
		t.Errorf("CacheReadTokens = %d, want 0", got)
	}
	if got := resp.Usage.CacheCreationTokens; got != 0 {
		t.Errorf("CacheCreationTokens = %d, want 0", got)
	}
	if got := resp.Usage.InputTokens; got != 12 {
		t.Errorf("InputTokens = %d, want 12", got)
	}
	if got := resp.Usage.OutputTokens; got != 7 {
		t.Errorf("OutputTokens = %d, want 7", got)
	}
}

// A present, non-null cache count keeps the strict numeric validation: relaxing
// null must not relax everything else about the field.
func TestDecodeResponse_NonNullCacheTokensStayStrict(t *testing.T) {
	body := []byte(`{
		"id": "msg_1",
		"type": "message",
		"role": "assistant",
		"model": "claude-sonnet-5",
		"content": [{"type": "text", "text": "hello"}],
		"stop_reason": "end_turn",
		"usage": {
			"input_tokens": 12,
			"output_tokens": 7,
			"cache_read_input_tokens": 2051,
			"cache_creation_input_tokens": "nope"
		}
	}`)

	_, err := DecodeResponse(body)
	var normalizationErr *usage.UsageNormalizationError
	if !errors.As(err, &normalizationErr) {
		t.Fatalf("DecodeResponse() error = %T %v, want *usage.UsageNormalizationError", err, err)
	}
	if normalizationErr.Field != usage.UsageNormalizationFieldCacheCreationTokens ||
		normalizationErr.Reason != usage.UsageNormalizationReasonInvalidType {
		t.Errorf("normalization error = %#v, want CacheCreationTokens/invalid type", normalizationErr)
	}
}
