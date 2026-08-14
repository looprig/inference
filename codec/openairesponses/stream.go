package openairesponses

import (
	"encoding/json"
	"net/http"

	"github.com/looprig/core/content"
	codec "github.com/looprig/inference/codec"
	stream "github.com/looprig/inference/stream"
	"github.com/looprig/inference/wire/sse"
)

// Compile-time proof that Codec is a full codec.StreamingCodec.
var _ codec.StreamingCodec = Codec{}

// DecodeStream frames a successful Responses streaming response with
// wire/sse and maps each frame through the codec's per-event decode logic.
// Unlike Chat Completions there is no explicit [DONE] sentinel: the stream
// ends when response.completed's terminal metadata has been observed and the
// body reaches natural EOF, matching how anthropicapi relies on
// message_stop plus EOF rather than a sentinel. An EOF that arrives without a
// terminal response event fails with a *StreamDecodeError rather than
// reporting a clean, truncated success. It owns resp.Body: the returned
// reader's Close closes it.
func (Codec) DecodeStream(resp *http.Response) (*stream.StreamReader[content.Chunk], error) {
	frames, err := sse.DecodeStreamFrames(resp.Body)
	if err != nil {
		return nil, err
	}
	collector := &streamResultCollector{}
	return stream.FramesToChunksWithResult(frames, collector.mapFrame, collector.result), nil
}

// DecodeEvent decodes one already-de-framed SSE event payload into the
// chunk(s) it yields. It is stateless: unknown valid event types are skipped,
// while malformed JSON is an error. Cross-event assembly (concatenating a tool call's
// argument fragments, or a reasoning summary's text fragments) happens
// downstream in the stream accumulator, not here.
func (Codec) DecodeEvent(event []byte) ([]content.Chunk, error) {
	return decodeEvent(event)
}

// sseEnvelope is the union view of one de-framed Responses SSE event this
// codec cares about. Content-delta fields feed decodeEvent; Response feeds
// the stream result collector (model, usage, finish reason) without entering
// the content chunk vocabulary. Code and Message are ResponseErrorEvent's
// top-level members (the `type:"error"` event carries them directly rather
// than inside a `response` object); Code is nullable in the spec and simply
// decodes to "".
type sseEnvelope struct {
	Type        string        `json:"type"`
	OutputIndex int           `json:"output_index"`
	ItemID      string        `json:"item_id"`
	Delta       string        `json:"delta"`
	Item        *wireItem     `json:"item"`
	Response    *wireResponse `json:"response"`
	Code        string        `json:"code"`
	Message     string        `json:"message"`
}

// decodeEvent maps one Responses SSE event to the chunk(s) it yields, per the
// design's streaming section:
//   - response.output_item.added(function_call) -> one ToolUseChunk carrying
//     Index/ID/Name (the fragment that seeds the accumulator with the tool
//     call id + name), analogous to Anthropic's content_block_start(tool_use).
//   - response.output_text.delta       -> one TextChunk.
//   - response.function_call_arguments.delta -> one ToolUseChunk arg
//     fragment (Index + InputJSON), emitted verbatim (even when empty, like
//     Anthropic's input_json_delta) for the accumulator to concatenate.
//   - response.reasoning_summary_text.delta  -> one ThinkingChunk (Index +
//     the summary fragment).
//   - response.output_item.done(reasoning)   -> one ThinkingChunk (Index +
//     the item's replayable id/encrypted_content as provider state).
//   - everything else                  -> (nil, nil), a tolerant skip.
//
// INDEX SEMANTICS. output_index maps directly to content.Chunk's Index field:
// it is a stable, response-scoped counter over the response's SINGLE `output`
// array — one space shared by reasoning, message and function_call items. So a
// gap in the reasoning accumulator's indexes is a message or a tool call, not a
// lost reasoning item, and it is exactly what parallel-tool-call indexing
// needs.
//
// Carrying it on the reasoning chunks is load-bearing, not cosmetic. A turn may
// hold SEVERAL reasoning items, each with its own id and encrypted_content that
// the next request must replay item-for-item; folding them onto one index bound
// the last blob to every summary and silently rewrote the continuation state.
// The two reasoning events for one item share an output_index, which is what
// re-unites a summary with the encrypted blob that belongs to it.
func decodeEvent(payload []byte) ([]content.Chunk, error) {
	var ev sseEnvelope
	if err := json.Unmarshal(payload, &ev); err != nil {
		return nil, &StreamEventDecodeError{Err: err}
	}
	return decodeEnvelope(ev)
}

func decodeEnvelope(ev sseEnvelope) ([]content.Chunk, error) {
	switch ev.Type {
	case eventOutputItemAdded:
		if ev.Item != nil && ev.Item.Type == itemTypeFunctionCall {
			return []content.Chunk{&content.ToolUseChunk{
				Index: ev.OutputIndex,
				ID:    ev.Item.CallID,
				Name:  ev.Item.Name,
			}}, nil
		}
		return nil, nil
	case eventOutputTextDelta:
		if ev.Delta == "" {
			return nil, nil
		}
		return []content.Chunk{&content.TextChunk{Text: ev.Delta}}, nil
	case eventFunctionCallArgsDelta:
		return []content.Chunk{&content.ToolUseChunk{Index: ev.OutputIndex, InputJSON: ev.Delta}}, nil
	case eventReasoningSummaryDelta:
		if ev.Delta == "" {
			return nil, nil
		}
		return []content.Chunk{&content.ThinkingChunk{Index: ev.OutputIndex, Thinking: ev.Delta}}, nil
	case eventRefusalDelta:
		// A refusal fragment becomes a RefusalChunk, which folds into the same
		// *content.RefusalBlock the non-streaming decoder builds from a refusal
		// content part (streamaccumulator.Refusal), so the two paths
		// reconstruct the same blocks for the same response.
		//
		// Unlike output_text.delta, an empty delta is NOT skipped: this event
		// only ever occurs on a refused turn, and its presence is the signal
		// (see refusalBlocks). Skipping it would let an explanation-free
		// refusal stream as a clean empty reply.
		return []content.Chunk{&content.RefusalChunk{Text: ev.Delta}}, nil
	case eventOutputItemDone:
		// The reasoning item's id is as load-bearing as its encrypted content:
		// ReasoningItem.required lists it, so an item known only by its
		// encrypted blob cannot be replayed. Either member alone is worth
		// carrying forward.
		if ev.Item == nil || ev.Item.Type != itemTypeReasoning {
			return nil, nil
		}
		if ev.Item.ID == "" && ev.Item.EncryptedContent == "" {
			return nil, nil
		}
		return []content.Chunk{&content.ThinkingChunk{
			Index:               ev.OutputIndex,
			ProviderState:       opaqueStateFromWire(ev.Item.ID, ev.Item.EncryptedContent),
			ProviderStateFormat: providerStateFormatOpenAIResponses,
		}}, nil
	default:
		// response.created, response.content_part.added/done,
		// non-reasoning response.output_item.done, response.completed,
		// response.function_call_arguments.done,
		// response.reasoning_summary_text.done, response.refusal.done (which
		// repeats the whole refusal the deltas already delivered), and any
		// unknown event type: no chunk (response.failed and the top-level
		// error event are handled by the collector, not here).
		return nil, nil
	}
}

// streamResultCollector accumulates terminal stream metadata (model, usage,
// finish reason) from response.completed, and surfaces response.failed as a
// hard stream error — mirroring anthropicapi's streamResultCollector.
// It does NOT track refusals: those travel as RefusalChunks in the content
// stream, which is where the signal belongs, and the terminal finish reason is
// derived from the response envelope exactly as the non-streaming decoder
// derives it — see deriveFinishReason (decode.go).
type streamResultCollector struct {
	completedSeen bool
	resultValue   stream.StreamResult
}

func (c *streamResultCollector) mapFrame(frame stream.StreamFrame) ([]content.Chunk, error) {
	var ev sseEnvelope
	if err := json.Unmarshal(frame.Data, &ev); err != nil {
		return nil, &StreamEventDecodeError{Err: err}
	}
	if err := c.collect(ev); err != nil {
		return nil, err
	}
	return decodeEnvelope(ev)
}

func (c *streamResultCollector) collect(ev sseEnvelope) error {
	switch ev.Type {
	case eventError:
		// ResponseErrorEvent: a first-class failure event that is not wrapped
		// in a `response` object, so it never reaches the response.failed arm.
		// Skipping it as an unknown type would end the stream at natural EOF
		// and report a truncated answer as a success.
		return &StreamAPIError{Code: ev.Code, Message: ev.Message}
	case eventResponseFailed:
		streamErr := &StreamAPIError{}
		if ev.Response != nil && ev.Response.Error != nil {
			streamErr.Code = ev.Response.Error.Code
			streamErr.Message = ev.Response.Error.Message
		}
		return streamErr
	case eventResponseCompleted, eventResponseIncomplete:
		c.completedSeen = true
		if ev.Response != nil {
			c.resultValue.Model = ev.Response.Model
			c.resultValue.FinishReason = deriveFinishReason(*ev.Response)
			u, err := normalizeUsage(ev.Response.Usage)
			if err != nil {
				return err
			}
			c.resultValue.Usage = u
		}
	}
	return nil
}

// result authorizes the terminal trailer. The Responses stream has no [DONE]
// sentinel, so its only end-of-generation markers are the union's terminal
// response events: response.completed and response.incomplete, which set
// completedSeen, plus response.failed and the top-level error event, which
// abort the stream with a *StreamAPIError before ever reaching here. A body
// that ends without one of them is a truncated answer, not a short one, and
// must fail rather than present partial content as a completed turn. Mirrors
// the gates in codec/geminiapi and codec/bedrockconverse.
func (c *streamResultCollector) result() (stream.StreamResult, bool, error) {
	if !c.completedSeen {
		return stream.StreamResult{}, false, &StreamDecodeError{Reason: "ended before a terminal response event"}
	}
	return c.resultValue, true, nil
}

// StreamDecodeError reports a Responses stream that is framed and parseable but
// structurally wrong — currently only a body that reaches EOF without one of
// the union's terminal response events (response.completed or
// response.incomplete; response.failed and the top-level error event abort with
// a *StreamAPIError instead), which means the answer was truncated in flight.
// It never includes the raw provider payload in its diagnostic. Named and
// shaped after the equivalent type in codec/bedrockconverse and
// codec/geminiapi. It lives here rather than in errors.go because the gate it
// serves is the only thing that raises it.
type StreamDecodeError struct {
	Reason string
	Err    error
}

func (e *StreamDecodeError) Error() string {
	if e.Err != nil {
		return "openairesponses: stream " + e.Reason + ": " + e.Err.Error()
	}
	return "openairesponses: stream " + e.Reason
}

func (e *StreamDecodeError) Unwrap() error { return e.Err }
