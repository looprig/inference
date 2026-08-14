package geminiapi

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

// DecodeStream frames a successful Gemini streamGenerateContent response (served as SSE
// via ?alt=sse) with wire/sse and maps each frame through the codec's per-event decode
// logic. Gemini has no terminal payload sentinel — the body simply ends — so the
// terminal signal is a candidate carrying a finishReason, and a body that reaches
// EOF without one fails with a *StreamDecodeError rather than reporting a clean,
// truncated success. It owns resp.Body: the returned reader's Close closes it.
//
// Unlike Codec.DecodeEvent, this path is stream-scoped, so it can number
// reasoning blocks across events (streamResultCollector.thoughtBase) rather than
// restarting at 0 in each one — see the INDEX SEMANTICS note on decodeEvent.
func (Codec) DecodeStream(resp *http.Response) (*stream.StreamReader[content.Chunk], error) {
	frames, err := sse.DecodeStreamFrames(resp.Body)
	if err != nil {
		return nil, err
	}
	collector := &streamResultCollector{}
	return stream.FramesToChunksWithResult(frames, collector.mapFrame, collector.result), nil
}

// mapFrame decodes one raw SSE frame's Data (a partial GenerateContentResponse) via the
// shared per-event decoder.
func mapFrame(f stream.StreamFrame) ([]content.Chunk, error) {
	return decodeEvent(f.Data)
}

// streamResultCollector accumulates terminal stream metadata (model, usage,
// finish reason) while frames are mapped, and enforces that the stream actually
// terminated. Gemini has no [DONE] sentinel, so sawTerminal — set by the first
// candidate carrying a non-empty finishReason — is the only evidence generation
// completed; without it, an EOF is a truncated answer, not a short one. This
// mirrors codec/bedrockconverse's stopped gate, the module's strictest
// streaming dialect.
type streamResultCollector struct {
	resultValue      stream.StreamResult
	functionCallSeen bool
	sawTerminal      bool
	// functionCallBase makes the per-event positional indexes produced by
	// decodeEvent unique across the whole stream. Gemini emits each functionCall
	// as a complete part, but parallel calls may be split across SSE frames.
	functionCallBase int
	// thoughtBase is the stream-scoped position of the reasoning block the NEXT
	// event's first thought part continues. Gemini puts no index on the wire and
	// decodeEvent is stateless, so its thought indexes restart at 0 every event;
	// without a base, a turn's second reasoning block would collide with its
	// first and the two thoughtSignatures would fuse onto one block. See
	// advanceThoughtBase for what advances it.
	thoughtBase int
}

// mapFrame decodes one frame, then rebases the event's per-event thought
// indexes onto the stream-scoped sequence. Chunks are rewritten in place: they
// were just allocated by decodeEvent and are not shared.
func (c *streamResultCollector) mapFrame(frame stream.StreamFrame) ([]content.Chunk, error) {
	var event GenerateContentResponse
	if err := json.Unmarshal(frame.Data, &event); err != nil {
		return nil, &StreamEventDecodeError{Err: err}
	}
	if err := c.collect(event); err != nil {
		return nil, err
	}
	chunks, err := mapFrame(frame)
	if err != nil {
		return nil, err
	}
	for _, chunk := range chunks {
		switch typed := chunk.(type) {
		case *content.ThinkingChunk:
			typed.Index += c.thoughtBase
		case *content.ToolUseChunk:
			typed.Index += c.functionCallBase
			typed.ID = rebaseSyntheticToolCallID(typed.ID, typed.Index)
		}
	}
	c.advanceThoughtBase(event)
	c.advanceFunctionCallBase(event)
	return chunks, nil
}

func (c *streamResultCollector) advanceFunctionCallBase(event GenerateContentResponse) {
	if len(event.Candidates) == 0 {
		return
	}
	for _, part := range event.Candidates[0].Content.Parts {
		if part.FunctionCall != nil {
			c.functionCallBase++
		}
	}
}

// advanceThoughtBase moves the base past every reasoning block this event
// SEALED. Gemini marks a thought block complete by attaching its
// `thoughtSignature` to the block's last part, so a signed thought part ends a
// block and an unsigned one continues into the next event. A model that emits
// thinking without signatures never advances the base — correct, since there is
// then no per-block state to keep apart and every fragment belongs to one
// reasoning block.
func (c *streamResultCollector) advanceThoughtBase(event GenerateContentResponse) {
	if len(event.Candidates) == 0 {
		return
	}
	for _, part := range event.Candidates[0].Content.Parts {
		if part.FunctionCall == nil && part.Thought && part.ThoughtSignature != "" {
			c.thoughtBase++
		}
	}
}

func (c *streamResultCollector) collect(event GenerateContentResponse) error {
	if event.ModelVersion != "" {
		c.resultValue.Model = event.ModelVersion
	}
	if len(event.Candidates) > 0 {
		candidate := event.Candidates[0]
		if hasFunctionCall(candidate.Content.Parts) {
			c.functionCallSeen = true
		}
		if candidate.FinishReason != "" {
			c.sawTerminal = true
			c.resultValue.FinishReason = mapFinishReason(candidate.FinishReason)
		}
		if c.functionCallSeen && (candidate.FinishReason == "" || candidate.FinishReason == "STOP") {
			c.resultValue.FinishReason = stream.FinishReasonToolUse
		}
	}
	if event.UsageMetadata == nil {
		return nil
	}
	usage, err := normalizeUsage(event.UsageMetadata)
	if err != nil {
		return err
	}
	c.resultValue.Usage = usage
	return nil
}

func hasFunctionCall(parts []geminiPart) bool {
	for _, part := range parts {
		if part.FunctionCall != nil {
			return true
		}
	}
	return false
}

func (c *streamResultCollector) result() (stream.StreamResult, bool, error) {
	if !c.sawTerminal {
		return stream.StreamResult{}, false, &StreamDecodeError{Reason: "ended before a candidate reported a finishReason"}
	}
	return c.resultValue, true, nil
}

func mapFinishReason(reason string) stream.FinishReason {
	switch reason {
	case "STOP":
		return stream.FinishReasonStop
	case "MAX_TOKENS":
		return stream.FinishReasonLength
	case "SAFETY", "RECITATION", "LANGUAGE", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION":
		return stream.FinishReasonContentFilter
	default:
		return stream.FinishReasonUnknown
	}
}
