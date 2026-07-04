package inference_test

import (
	"bytes"
	"io"
	"net/http"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
)

// nonStreamingFake implements ONLY RequestEncoder + ResponseDecoder. It deliberately does not
// implement DecodeStream, proving that Codec does not force a streaming stub (no LSP violation).
type nonStreamingFake struct{}

func (nonStreamingFake) EncodeRequest(inference.Request, inference.RequestMode) (inference.EncodedRequest, error) {
	return inference.EncodedRequest{Header: http.Header{}, Body: bytes.NewReader([]byte("{}"))}, nil
}

func (nonStreamingFake) DecodeResponse([]byte) (*inference.Response, error) {
	return &inference.Response{}, nil
}

// Compile-time assertion: a codec with no DecodeStream method still satisfies Codec.
var _ inference.Codec = nonStreamingFake{}

// streamingFake additionally implements DecodeStream, satisfying StreamingCodec.
type streamingFake struct{}

func (streamingFake) EncodeRequest(inference.Request, inference.RequestMode) (inference.EncodedRequest, error) {
	return inference.EncodedRequest{Header: http.Header{}, Body: bytes.NewReader([]byte("{}"))}, nil
}

func (streamingFake) DecodeResponse([]byte) (*inference.Response, error) {
	return &inference.Response{}, nil
}

func (streamingFake) DecodeStream(resp *http.Response) (*inference.StreamReader[content.Chunk], error) {
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	return inference.NewStreamReader(func() (content.Chunk, error) { return nil, io.EOF }, nil), nil
}

// Compile-time assertions: streamingFake satisfies both Codec and StreamingCodec.
var (
	_ inference.Codec          = streamingFake{}
	_ inference.StreamingCodec = streamingFake{}
)

// framerFake implements StreamFramer over a raw body.
type framerFake struct{}

func (framerFake) DecodeStreamFrames(body io.ReadCloser) (*inference.StreamReader[inference.StreamFrame], error) {
	return inference.NewStreamReader(
		func() (inference.StreamFrame, error) { return inference.StreamFrame{}, io.EOF },
		func() error { return body.Close() },
	), nil
}

var _ inference.StreamFramer = framerFake{}

// TestCodec_StreamingIsOptional confirms the segregation: a non-streaming codec satisfies
// Codec at compile time (asserted above) yet is NOT a StreamDecoder at runtime — so the
// streaming path can require an explicit StreamDecoder/StreamingCodec rather than a stub.
func TestCodec_StreamingIsOptional(t *testing.T) {
	t.Parallel()

	var c inference.Codec = nonStreamingFake{}
	if _, ok := c.(inference.StreamDecoder); ok {
		t.Error("nonStreamingFake unexpectedly satisfies StreamDecoder; Codec must not imply streaming")
	}

	var sc inference.Codec = streamingFake{}
	if _, ok := sc.(inference.StreamDecoder); !ok {
		t.Error("streamingFake should satisfy StreamDecoder")
	}
	if _, ok := sc.(inference.StreamingCodec); !ok {
		t.Error("streamingFake should satisfy StreamingCodec")
	}
}
