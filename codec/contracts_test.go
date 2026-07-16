package codec_test

import (
	"bytes"
	"io"
	"net/http"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	codec "github.com/looprig/inference/codec"
	stream "github.com/looprig/inference/stream"
)

// nonStreamingFake implements ONLY RequestEncoder + ResponseDecoder. It deliberately does not
// implement DecodeStream, proving that Codec does not force a streaming stub (no LSP violation).
type nonStreamingFake struct{}

func (nonStreamingFake) EncodeRequest(inference.Request, codec.RequestMode) (codec.EncodedRequest, error) {
	return codec.EncodedRequest{Header: http.Header{}, Body: bytes.NewReader([]byte("{}"))}, nil
}

func (nonStreamingFake) DecodeResponse([]byte) (*inference.Response, error) {
	return &inference.Response{}, nil
}

// Compile-time assertion: a codec with no DecodeStream method still satisfies Codec.
var _ codec.Codec = nonStreamingFake{}

// streamingFake additionally implements DecodeStream, satisfying StreamingCodec.
type streamingFake struct{}

func (streamingFake) EncodeRequest(inference.Request, codec.RequestMode) (codec.EncodedRequest, error) {
	return codec.EncodedRequest{Header: http.Header{}, Body: bytes.NewReader([]byte("{}"))}, nil
}

func (streamingFake) DecodeResponse([]byte) (*inference.Response, error) {
	return &inference.Response{}, nil
}

func (streamingFake) DecodeStream(resp *http.Response) (*stream.StreamReader[content.Chunk], error) {
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	return stream.NewStreamReader(func() (content.Chunk, error) { return nil, io.EOF }, nil), nil
}

// Compile-time assertions: streamingFake satisfies both Codec and StreamingCodec.
var (
	_ codec.Codec          = streamingFake{}
	_ codec.StreamingCodec = streamingFake{}
)

// framerFake implements StreamFramer over a raw body.
type framerFake struct{}

func (framerFake) DecodeStreamFrames(body io.ReadCloser) (*stream.StreamReader[stream.StreamFrame], error) {
	return stream.NewStreamReader(
		func() (stream.StreamFrame, error) { return stream.StreamFrame{}, io.EOF },
		func() error { return body.Close() },
	), nil
}

var _ codec.StreamFramer = framerFake{}

// TestCodec_StreamingIsOptional confirms the segregation: a non-streaming codec satisfies
// Codec at compile time (asserted above) yet is NOT a StreamDecoder at runtime — so the
// streaming path can require an explicit StreamDecoder/StreamingCodec rather than a stub.
func TestCodec_StreamingIsOptional(t *testing.T) {
	t.Parallel()

	var c codec.Codec = nonStreamingFake{}
	if _, ok := c.(codec.StreamDecoder); ok {
		t.Error("nonStreamingFake unexpectedly satisfies StreamDecoder; Codec must not imply streaming")
	}

	var sc codec.Codec = streamingFake{}
	if _, ok := sc.(codec.StreamDecoder); !ok {
		t.Error("streamingFake should satisfy StreamDecoder")
	}
	if _, ok := sc.(codec.StreamingCodec); !ok {
		t.Error("streamingFake should satisfy StreamingCodec")
	}
}
