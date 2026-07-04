package openaiapi

import (
	"io"
	"net/http"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/wire/sse"
)

// doneSentinel is OpenAI's terminal SSE payload. The wire/sse framer emits it as an
// ordinary frame (it does not interpret sentinels); the codec owns treating it as
// end-of-stream, mapping it to io.EOF.
const doneSentinel = "[DONE]"

// Compile-time proof that Codec is a full inference.StreamingCodec.
var _ inference.StreamingCodec = Codec{}

// DecodeStream frames a successful OpenAI streaming response with wire/sse and maps
// each frame through the codec's per-event decode logic. It owns resp.Body: the
// returned reader's Close closes it (and DecodeStreamFrames closes it if it errors
// before returning a reader).
func (Codec) DecodeStream(resp *http.Response) (*inference.StreamReader[content.Chunk], error) {
	frames, err := sse.DecodeStreamFrames(resp.Body)
	if err != nil {
		return nil, err
	}
	return inference.FramesToChunks(frames, mapFrame), nil
}

// NewStream adapts a raw OpenAI SSE body into a chunk stream. Exposed for provider
// extensions and dialect tests that drive a body directly; the transport uses
// DecodeStream. The caller must Close the returned reader when done.
func NewStream(body io.ReadCloser) *inference.StreamReader[content.Chunk] {
	frames, err := sse.DecodeStreamFrames(body)
	if err != nil {
		// Do not discard the framer error (e.g. a nil body): return a reader that surfaces
		// it on first Next rather than dereferencing a nil frames reader and panicking.
		return inference.NewStreamReader(
			func() (content.Chunk, error) { return nil, err },
			func() error { return nil },
		)
	}
	return inference.FramesToChunks(frames, mapFrame)
}

// mapFrame maps one raw SSE frame to chunk(s): the [DONE] sentinel ends the stream
// (io.EOF), everything else runs through the shared per-event decoder.
func mapFrame(f inference.StreamFrame) ([]content.Chunk, error) {
	if string(f.Data) == doneSentinel {
		return nil, io.EOF
	}
	return decodeEvent(f.Data)
}
