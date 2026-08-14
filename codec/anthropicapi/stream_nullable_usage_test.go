package anthropicapi_test

import (
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference/codec/anthropicapi"
	"github.com/looprig/inference/usage"
)

// Anthropic publishes two usage shapes. Usage (carried by message_start) types
// input_tokens and output_tokens as plain non-negative integers. MessageDeltaUsage
// (carried by message_delta) types input_tokens, cache_read_input_tokens and
// cache_creation_input_tokens as anyOf[{integer, minimum 0}, {null}] while
// keeping output_tokens a plain integer. All four are required members.
//
// A null in the delta therefore means "not reported in this event", not zero and
// not a protocol violation. Two defects followed from treating it as a value:
// the strict numeric conversion failed a stream that had ALREADY emitted its
// content, and the merge overwrote the authoritative input count that arrived in
// message_start. This is the streaming twin of
// TestDecodeResponse_NullCacheTokensNormalizeToZero — an accounting field must
// never discard a completed generation.
func TestAnthropicStream_NullDeltaUsageKeepsMessageStartCounts(t *testing.T) {
	t.Parallel()

	body := anthropicEvent("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":25,"output_tokens":1,"cache_creation_input_tokens":6,"cache_read_input_tokens":4}}}`) +
		anthropicEvent("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`) +
		anthropicEvent("ping", `{"type":"ping"}`) +
		anthropicEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`) +
		anthropicEvent("content_block_stop", `{"type":"content_block_stop","index":0}`) +
		anthropicEvent("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":null,"cache_creation_input_tokens":null,"cache_read_input_tokens":null,"output_tokens":15,"output_tokens_details":{"thinking_tokens":12}}}`) +
		anthropicEvent("message_stop", `{"type":"message_stop"}`)

	reader, err := (anthropicapi.Codec{}).DecodeStream(&http.Response{Body: io.NopCloser(strings.NewReader(body))})
	if err != nil {
		t.Fatalf("DecodeStream() error = %v", err)
	}
	defer reader.Close()

	var text strings.Builder
	for {
		chunk, err := reader.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Next() error = %T %v; a null accounting field must not fail a stream that already emitted content", err, err)
		}
		if delta, ok := chunk.(*content.TextChunk); ok {
			text.WriteString(delta.Text)
		}
	}
	if got := text.String(); got != "Hello" {
		t.Errorf("streamed text = %q, want %q", got, "Hello")
	}

	result, ok := reader.Result()
	if !ok {
		t.Fatal("Result() ok = false, want the terminal trailer after message_stop")
	}
	want := content.Usage{InputTokens: 25, OutputTokens: 15, CacheReadTokens: 4, CacheCreationTokens: 6, ReasoningTokens: 12}
	if result.Usage == nil || *result.Usage != want {
		t.Errorf("Result.Usage = %+v, want %+v (a null delta count must not clobber the message_start value)", result.Usage, want)
	}
}

// A stream whose only usage object is a delta carrying nulls still completes:
// null is "not reported", so the counts stay at their unreported zero rather
// than failing the turn.
func TestAnthropicStream_NullDeltaUsageWithoutStartCountsCompletes(t *testing.T) {
	t.Parallel()

	body := anthropicEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}`) +
		anthropicEvent("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":null,"cache_creation_input_tokens":null,"cache_read_input_tokens":null,"output_tokens":3}}`) +
		anthropicEvent("message_stop", `{"type":"message_stop"}`)

	reader, err := (anthropicapi.Codec{}).DecodeStream(&http.Response{Body: io.NopCloser(strings.NewReader(body))})
	if err != nil {
		t.Fatalf("DecodeStream() error = %v", err)
	}
	defer reader.Close()

	for {
		_, err := reader.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Next() error = %T %v, want a clean stream", err, err)
		}
	}
	result, ok := reader.Result()
	if !ok {
		t.Fatal("Result() ok = false, want the terminal trailer after message_stop")
	}
	want := content.Usage{OutputTokens: 3}
	if result.Usage == nil || *result.Usage != want {
		t.Errorf("Result.Usage = %+v, want %+v", result.Usage, want)
	}
}

// Relaxing null must not relax anything else: a present, non-null delta count
// keeps the strict numeric validation, and output_tokens — which
// MessageDeltaUsage types as a plain integer, not a nullable one — stays strict
// against null as well.
func TestAnthropicStream_NonNullDeltaCountsStayStrict(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name       string
		body       string
		wantField  usage.UsageNormalizationField
		wantReason usage.UsageNormalizationReason
	}{
		{
			name: "non-numeric nullable count",
			body: anthropicEvent("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":"many","output_tokens":15}}`) +
				anthropicEvent("message_stop", `{"type":"message_stop"}`),
			wantField:  usage.UsageNormalizationFieldInputTokens,
			wantReason: usage.UsageNormalizationReasonInvalidType,
		},
		{
			name: "negative nullable count",
			body: anthropicEvent("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"cache_read_input_tokens":-1,"output_tokens":15}}`) +
				anthropicEvent("message_stop", `{"type":"message_stop"}`),
			wantField:  usage.UsageNormalizationFieldCacheReadTokens,
			wantReason: usage.UsageNormalizationReasonNegative,
		},
		{
			name: "null in the non-nullable output count",
			body: anthropicEvent("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":null,"output_tokens":null}}`) +
				anthropicEvent("message_stop", `{"type":"message_stop"}`),
			wantField:  usage.UsageNormalizationFieldOutputTokens,
			wantReason: usage.UsageNormalizationReasonNull,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			reader, err := (anthropicapi.Codec{}).DecodeStream(&http.Response{Body: io.NopCloser(strings.NewReader(tc.body))})
			if err != nil {
				t.Fatalf("DecodeStream() error = %v", err)
			}
			defer reader.Close()

			for err == nil {
				_, err = reader.Next()
			}
			if errors.Is(err, io.EOF) {
				t.Fatal("stream ended cleanly, want a usage normalization failure")
			}
			var normalizationErr *usage.UsageNormalizationError
			if !errors.As(err, &normalizationErr) {
				t.Fatalf("Next() error = %T %v, want *usage.UsageNormalizationError", err, err)
			}
			if normalizationErr.Field != tc.wantField || normalizationErr.Reason != tc.wantReason {
				t.Errorf("normalization error = %#v, want %s/%s", normalizationErr, tc.wantField, tc.wantReason)
			}
			if _, ok := reader.Result(); ok {
				t.Error("Result() available after a usage normalization failure")
			}
		})
	}
}
