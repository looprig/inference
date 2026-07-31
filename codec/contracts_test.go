package codec_test

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
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

// serverFake implements ServerCodec against a trivial single-route JSON dialect. It exists
// only to pin the compile-time shape of ServerCodec and StreamEncoder here; behavior is
// exercised end-to-end by the reusable suite in codec/servertest against a richer fake.
type serverFake struct{}

func (serverFake) MatchRequest(req *http.Request) bool {
	return req.Method == http.MethodPost && req.URL.Path == "/v1/fake"
}

func (serverFake) DecodeRequest(req *http.Request) (codec.DecodedRequest, error) {
	return codec.DecodedRequest{
		Request:        inference.Request{},
		RequestedModel: "fake-model",
		Streaming:      false,
	}, nil
}

func (serverFake) WriteResponse(w http.ResponseWriter, resp *inference.Response) error {
	w.WriteHeader(http.StatusOK)
	return nil
}

func (serverFake) OpenStream(w http.ResponseWriter) (codec.StreamEncoder, error) {
	w.WriteHeader(http.StatusOK)
	return serverStreamFake{}, nil
}

func (serverFake) WriteError(w http.ResponseWriter, err error) {
	w.WriteHeader(http.StatusBadRequest)
}

// Compile-time assertion: serverFake satisfies the new server-side contract.
var _ codec.ServerCodec = serverFake{}

// serverStreamFake implements StreamEncoder, the request-scoped counterpart returned by
// ServerCodec.OpenStream.
type serverStreamFake struct{}

func (serverStreamFake) WriteChunk(content.Chunk) error   { return nil }
func (serverStreamFake) Finish(stream.StreamResult) error { return nil }
func (serverStreamFake) Fail(error) error                 { return nil }

// Compile-time assertion: serverStreamFake satisfies StreamEncoder.
var _ codec.StreamEncoder = serverStreamFake{}

// TestServerCodec_MatchAndDecode exercises serverFake minimally beyond the compile-time
// assertions above, pinning the DecodedRequest field shape (Request, RequestedModel,
// Streaming) against real net/http types.
func TestServerCodec_MatchAndDecode(t *testing.T) {
	t.Parallel()

	var c codec.ServerCodec = serverFake{}

	req := httptest.NewRequest(http.MethodPost, "/v1/fake", strings.NewReader("{}"))
	if !c.MatchRequest(req) {
		t.Fatal("serverFake did not match its own route")
	}

	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest: %v", err)
	}
	if decoded.RequestedModel != "fake-model" {
		t.Errorf("RequestedModel = %q, want %q", decoded.RequestedModel, "fake-model")
	}
	if decoded.Streaming {
		t.Error("Streaming = true, want false")
	}

	other := httptest.NewRequest(http.MethodGet, "/v1/other", nil)
	if c.MatchRequest(other) {
		t.Error("serverFake matched an unrelated route")
	}
}
