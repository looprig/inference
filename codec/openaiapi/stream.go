package openaiapi

import (
	"encoding/json"
	"io"
	"net/http"

	"github.com/looprig/core/content"
	codec "github.com/looprig/inference/codec"
	stream "github.com/looprig/inference/stream"
	"github.com/looprig/inference/wire/sse"
)

// doneSentinel is OpenAI's terminal SSE payload. The wire/sse framer emits it as an
// ordinary frame (it does not interpret sentinels); the codec owns treating it as
// end-of-stream, mapping it to io.EOF.
const doneSentinel = "[DONE]"

// Compile-time proof that Codec is a full codec.StreamingCodec.
var _ codec.StreamingCodec = Codec{}

// DecodeStream frames a successful OpenAI streaming response with wire/sse and maps
// each frame through the codec's per-event decode logic. A body that ends without
// either end-of-generation signal — a choice's finish_reason or the [DONE]
// sentinel — fails with a *StreamDecodeError rather than reporting a clean,
// truncated success. It owns resp.Body: the returned reader's Close closes it
// (and DecodeStreamFrames closes it if it errors before returning a reader).
func (Codec) DecodeStream(resp *http.Response) (*stream.StreamReader[content.Chunk], error) {
	frames, err := sse.DecodeStreamFrames(resp.Body)
	if err != nil {
		return nil, err
	}
	collector := &streamResultCollector{}
	return stream.FramesToChunksWithResult(frames, collector.mapFrame, collector.result), nil
}

// NewStream adapts a raw OpenAI SSE body into a chunk stream. Exposed for provider
// extensions and dialect tests that drive a body directly; the transport uses
// DecodeStream. The caller must Close the returned reader when done.
func NewStream(body io.ReadCloser) *stream.StreamReader[content.Chunk] {
	frames, err := sse.DecodeStreamFrames(body)
	if err != nil {
		// Do not discard the framer error (e.g. a nil body): return a reader that surfaces
		// it on first Next rather than dereferencing a nil frames reader and panicking.
		return stream.NewStreamReader(
			func() (content.Chunk, error) { return nil, err },
			func() error { return nil },
		)
	}
	collector := &streamResultCollector{}
	return stream.FramesToChunksWithResult(frames, collector.mapFrame, collector.result)
}

// mapFrame maps one raw SSE frame to chunk(s): the [DONE] sentinel ends the stream
// (io.EOF), everything else runs through the shared per-event decoder.
func mapFrame(f stream.StreamFrame) ([]content.Chunk, error) {
	if string(f.Data) == doneSentinel {
		return nil, io.EOF
	}
	return decodeEvent(f.Data)
}

// streamResultCollector accumulates terminal stream metadata. It does NOT
// track refusals: the refusal deltas become RefusalChunks in the content
// stream, which is where the signal belongs, and the terminal finish reason is
// reported exactly as the wire sent it — see refusalBlocks (decode.go), whose
// non-streaming decision this reproduces.
type streamResultCollector struct {
	resultValue stream.StreamResult
	// doneSeen and finishReasonSeen are the two end-of-generation signals this
	// format defines; either one authorizes the trailer. See result.
	doneSeen         bool
	finishReasonSeen bool
}

func (c *streamResultCollector) mapFrame(frame stream.StreamFrame) ([]content.Chunk, error) {
	if string(frame.Data) == doneSentinel {
		c.doneSeen = true
		return mapFrame(frame)
	}
	var event sseChunk
	if err := json.Unmarshal(frame.Data, &event); err != nil {
		return nil, &StreamEventDecodeError{Err: err}
	}
	if err := c.collect(event); err != nil {
		return nil, err
	}
	return mapFrame(frame)
}

func (c *streamResultCollector) collect(event sseChunk) error {
	// A well-formed error envelope inside a 200 stream is a hard failure, not
	// a skippable event: transport's status-based APIError never saw it, so
	// dropping it here would report a truncated stream as a clean success.
	// Checked before anything else so a trailing finish_reason/usage on the
	// same frame cannot overwrite the result with success metadata.
	if event.Error != nil {
		code := event.Error.codeString()
		if code == "" {
			code = event.Error.Type
		}
		return &StreamAPIError{Code: code, Message: event.Error.Message}
	}
	if event.Model != "" {
		c.resultValue.Model = event.Model
	}
	if len(event.Choices) > 0 && event.Choices[0].FinishReason != "" {
		c.finishReasonSeen = true
		c.resultValue.FinishReason = mapFinishReason(event.Choices[0].FinishReason)
	}
	if event.Usage == nil {
		return nil
	}
	usage, err := normalizeUsage(event.Usage)
	if err != nil {
		return err
	}
	c.resultValue.Usage = usage
	return nil
}

// result authorizes the terminal trailer. Chat Completions defines two
// end-of-generation signals and either one is sufficient evidence the model
// finished: the [DONE] sentinel the request schema documents the stream as
// being "terminated by", and the non-null finish_reason a choice reports on the
// chunk where it stops. Both are accepted because they have different
// authority — finish_reason is a required, schema-modelled member of every
// streamed choice, while [DONE] is a transport convention carried in prose
// only, and OpenAI-compatible gateways omit one or the other. A body that
// carries NEITHER is a truncated answer, not a short one, and must fail rather
// than present partial content as a completed turn. Mirrors the gates in
// codec/geminiapi and codec/bedrockconverse.
func (c *streamResultCollector) result() (stream.StreamResult, bool, error) {
	if !c.doneSeen && !c.finishReasonSeen {
		return stream.StreamResult{}, false, &StreamDecodeError{Reason: "ended before a finish_reason or the [DONE] sentinel"}
	}
	return c.resultValue, true, nil
}

// StreamDecodeError reports a Chat Completions stream that is framed and
// parseable but structurally wrong — currently only a body that reaches EOF
// without either end-of-generation signal the format defines (a choice's
// non-null finish_reason, or the [DONE] sentinel), which means the answer was
// truncated in flight. It never includes the raw provider payload in its
// diagnostic. Named and shaped after the equivalent type in
// codec/bedrockconverse and codec/geminiapi. It lives here rather than in
// errors.go because the gate it serves is the only thing that raises it.
type StreamDecodeError struct {
	Reason string
	Err    error
}

func (e *StreamDecodeError) Error() string {
	if e.Err != nil {
		return "openaiapi: stream " + e.Reason + ": " + e.Err.Error()
	}
	return "openaiapi: stream " + e.Reason
}

func (e *StreamDecodeError) Unwrap() error { return e.Err }

func mapFinishReason(reason string) stream.FinishReason {
	switch reason {
	case "stop":
		return stream.FinishReasonStop
	case "length":
		return stream.FinishReasonLength
	case "tool_calls", "function_call":
		return stream.FinishReasonToolUse
	case "content_filter":
		return stream.FinishReasonContentFilter
	case "error":
		// Not in OpenAI's own finish_reason enum, but emitted by compatible
		// gateways that report an upstream failure inside a 200 response. The
		// neutral vocabulary has no error finish reason — the failure itself
		// travels as the typed error built from the body's `error` member —
		// so this is listed explicitly to document that it must never be
		// confused with a clean stop, not merely fall through the default.
		return stream.FinishReasonUnknown
	default:
		return stream.FinishReasonUnknown
	}
}
