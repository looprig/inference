package anthropicapi

import (
	"encoding/json"
	"errors"
	"net/http"

	"github.com/looprig/core/content"
	codec "github.com/looprig/inference/codec"
	"github.com/looprig/inference/internal/usagenorm"
	stream "github.com/looprig/inference/stream"
	usage "github.com/looprig/inference/usage"
	"github.com/looprig/inference/wire/sse"
)

// Compile-time proof that Codec is a full codec.StreamingCodec.
var _ codec.StreamingCodec = Codec{}

// DecodeStream frames a successful Anthropic Messages streaming response with wire/sse
// and maps each frame through the codec's per-event decode logic. The message_stop
// marker authorizes the terminal result but yields no chunk; the body's natural
// EOF ends the transport stream. Because that EOF is indistinguishable from a
// dropped connection, a body that reaches it without message_stop fails with a
// *StreamDecodeError rather than reporting a clean, truncated success. It owns
// resp.Body: the returned reader's Close closes it.
func (Codec) DecodeStream(resp *http.Response) (*stream.StreamReader[content.Chunk], error) {
	frames, err := sse.DecodeStreamFrames(resp.Body)
	if err != nil {
		return nil, err
	}
	collector := &streamResultCollector{}
	return stream.FramesToChunksWithResult(frames, collector.mapFrame, collector.result), nil
}

// mapFrame decodes one raw SSE frame's Data via the shared per-event decoder. The
// Anthropic event type lives inside the JSON payload (decodeEvent reads it), so the
// SSE event Name on the frame is not needed here.
func mapFrame(f stream.StreamFrame) ([]content.Chunk, error) {
	return decodeEvent(f.Data)
}

type streamResultCollector struct {
	wireUsage       messageUsage
	usageSeen       bool
	messageStopSeen bool
	resultValue     stream.StreamResult
}

// mapFrame decodes one frame twice: once into the collector's trailer view and
// once through the shared per-event chunk decoder. A frame whose JSON does not
// parse aborts the stream with a *StreamEventDecodeError instead of being
// ignored — the collector's result trailer is only authoritative if every frame
// that reached it was actually understood.
func (c *streamResultCollector) mapFrame(frame stream.StreamFrame) ([]content.Chunk, error) {
	var event streamEvent
	if err := json.Unmarshal(frame.Data, &event); err != nil {
		return nil, &StreamEventDecodeError{Err: err}
	}
	if err := c.collect(event); err != nil {
		return nil, err
	}
	return mapFrame(frame)
}

func (c *streamResultCollector) collect(event streamEvent) error {
	if event.Type == responseTypeError {
		streamErr := &StreamAPIError{}
		if event.Error != nil {
			streamErr.Type = event.Error.Type
			streamErr.Message = event.Error.Message
		}
		return streamErr
	}
	if event.Type == eventMessageStop {
		c.messageStopSeen = true
	}
	if event.Type == eventMessageStart && event.Message != nil {
		if event.Message.Model != "" {
			c.resultValue.Model = event.Message.Model
		}
		if err := c.mergeUsage(event.Message.Usage, startUsageNullable); err != nil {
			return err
		}
	}
	if event.Type == eventMessageDelta {
		if event.Delta != nil && event.Delta.StopReason != "" {
			c.resultValue.FinishReason = mapFinishReason(event.Delta.StopReason)
		}
		if err := c.mergeUsage(event.Usage, deltaUsageNullable); err != nil {
			return err
		}
	}
	return nil
}

// nullableCounts names the members of one usage object the event's OWN schema
// permits to be null. Anthropic publishes two usage shapes and they differ:
// Usage (message_start) types input_tokens and output_tokens as plain
// non-negative integers, while MessageDeltaUsage (message_delta) types
// input_tokens as anyOf[{integer, minimum 0}, {null}] with "default": null.
// Both type the two cache counts that way. Every one of these is a REQUIRED
// member whose documented default is null, so a conforming stream carries nulls
// routinely.
type nullableCounts struct {
	input bool
	cache bool
}

var (
	// startUsageNullable follows components.schemas.Usage.
	startUsageNullable = nullableCounts{cache: true}
	// deltaUsageNullable follows components.schemas.MessageDeltaUsage, which
	// additionally makes input_tokens nullable. output_tokens stays a plain
	// integer in both, so a null there remains an error.
	deltaUsageNullable = nullableCounts{input: true, cache: true}
)

// mergeUsage folds one event's usage object into the stream-scoped view. Both
// shapes report CUMULATIVE totals, so the last known value of each count wins.
func (c *streamResultCollector) mergeUsage(update *messageUsage, nullable nullableCounts) error {
	if update == nil {
		return nil
	}
	c.usageSeen = true
	merges := []struct {
		into     *usagenorm.Count
		from     usagenorm.Count
		field    usagenorm.Field
		nullable bool
	}{
		{&c.wireUsage.InputTokens, update.InputTokens, usagenorm.FieldInputTokens, nullable.input},
		{&c.wireUsage.OutputTokens, update.OutputTokens, usagenorm.FieldOutputTokens, false},
		{&c.wireUsage.CacheReadTokens, update.CacheReadTokens, usagenorm.FieldCacheReadTokens, nullable.cache},
		{&c.wireUsage.CacheCreationTokens, update.CacheCreationTokens, usagenorm.FieldCacheCreationTokens, nullable.cache},
	}
	for _, merge := range merges {
		if err := mergeCount(merge.into, merge.from, merge.field, merge.nullable); err != nil {
			return err
		}
	}
	if update.OutputTokensDetails != nil {
		if c.wireUsage.OutputTokensDetails == nil {
			c.wireUsage.OutputTokensDetails = &messageOutputTokensDetails{}
		}
		if err := mergeCount(
			&c.wireUsage.OutputTokensDetails.ThinkingTokens,
			update.OutputTokensDetails.ThinkingTokens,
			usagenorm.FieldReasoningTokens,
			false,
		); err != nil {
			return err
		}
	}
	usage, err := normalizeUsage(&c.wireUsage)
	if err != nil {
		return err
	}
	c.resultValue.Usage = usage
	return nil
}

// mergeCount overwrites into with from only when from carries a KNOWN value.
//
// A null in a field its own schema declares nullable means "not reported in
// this event", which is neither zero nor a protocol violation. Merging it did
// two kinds of damage: normalizeUsage's strict conversion then failed a stream
// that had already emitted its content — an accounting field discarding a
// completed generation, the defect the non-streaming decoder was fixed for —
// and, because usagenorm.Count.Present() is true for an explicit null, the
// merge overwrote the authoritative input count that arrived in message_start
// with nothing. Skipping it keeps the last value actually reported.
//
// Only null is forgiven, and only where the schema allows it. Every other
// malformed count (a string, a fraction, a negative, an overflow, or a null in
// a member typed as a plain integer) is still returned as the usage
// normalization error it has always been, so relaxing null relaxes nothing
// else. usagenorm.Count keeps its raw token private, so TokenCount's
// UsageNormalizationReasonNull — the one diagnosis reserved for an explicit
// null — is what identifies the case.
func mergeCount(into *usagenorm.Count, from usagenorm.Count, field usagenorm.Field, nullable bool) error {
	if !from.Present() {
		return nil
	}
	if _, err := from.TokenCount(field); err != nil {
		var normalizationErr *usage.UsageNormalizationError
		if nullable && errors.As(err, &normalizationErr) && normalizationErr.Reason == usage.UsageNormalizationReasonNull {
			return nil
		}
		return err
	}
	*into = from
	return nil
}

// result authorizes the terminal trailer. message_stop is the Messages
// stream's only end-of-generation marker — the transport's own EOF is exactly
// what a dropped connection produces too — so a body that ends without it is a
// truncated answer, not a short one, and must fail rather than present partial
// content as a completed turn. Mirrors the gates in codec/geminiapi and
// codec/bedrockconverse.
func (c *streamResultCollector) result() (stream.StreamResult, bool, error) {
	if !c.messageStopSeen {
		return stream.StreamResult{}, false, &StreamDecodeError{Reason: "ended before message_stop"}
	}
	if !c.usageSeen {
		c.resultValue.Usage = nil
	}
	return c.resultValue, true, nil
}

// StreamDecodeError reports a Messages stream that is framed and parseable but
// structurally wrong — currently only a body that reaches EOF without the
// message_stop event the MessageStreamEvent union ends with, which means the
// answer was truncated in flight. It never includes the raw provider payload in
// its diagnostic. Named and shaped after the equivalent type in
// codec/bedrockconverse and codec/geminiapi. It lives here rather than in
// errors.go because the gate it serves is the only thing that raises it.
type StreamDecodeError struct {
	Reason string
	Err    error
}

func (e *StreamDecodeError) Error() string {
	if e.Err != nil {
		return "anthropicapi: stream " + e.Reason + ": " + e.Err.Error()
	}
	return "anthropicapi: stream " + e.Reason
}

func (e *StreamDecodeError) Unwrap() error { return e.Err }

func mapFinishReason(reason string) stream.FinishReason {
	switch reason {
	case "end_turn", "stop_sequence", "pause_turn":
		return stream.FinishReasonStop
	case "max_tokens":
		return stream.FinishReasonLength
	case "tool_use":
		return stream.FinishReasonToolUse
	case "refusal":
		return stream.FinishReasonContentFilter
	default:
		return stream.FinishReasonUnknown
	}
}
