package anthropicapi

import (
	"net/http"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/wire/sse"
)

// Compile-time proof that Codec is a full inference.StreamingCodec.
var _ inference.StreamingCodec = Codec{}

// DecodeStream frames a successful Anthropic Messages streaming response with wire/sse
// and maps each frame through the codec's per-event decode logic. Anthropic has no
// terminal payload sentinel: message_stop yields no chunk and the stream ends on the
// body's natural EOF. It owns resp.Body: the returned reader's Close closes it.
func (Codec) DecodeStream(resp *http.Response) (*inference.StreamReader[content.Chunk], error) {
	frames, err := sse.DecodeStreamFrames(resp.Body)
	if err != nil {
		return nil, err
	}
	return inference.FramesToChunks(frames, mapFrame), nil
}

// mapFrame decodes one raw SSE frame's Data via the shared per-event decoder. The
// Anthropic event type lives inside the JSON payload (decodeEvent reads it), so the
// SSE event Name on the frame is not needed here.
func mapFrame(f inference.StreamFrame) ([]content.Chunk, error) {
	return decodeEvent(f.Data)
}
