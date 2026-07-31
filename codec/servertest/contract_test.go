package servertest_test

import (
	"errors"
	"io"
	"net/http"
	"testing"

	"encoding/json"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec"
	"github.com/looprig/inference/codec/servertest"
	"github.com/looprig/inference/stream"
)

const fakeContentType = "application/json"

// fakeServerCodec is a minimal, correct ServerCodec. It exists to prove that
// servertest.Run genuinely exercises whatever codec.ServerCodec its Factory
// produces, rather than being hardcoded to one implementation: every dialect
// codec built in later tasks (anthropicapi, openairesponses, openaiapi,
// geminiapi) is expected to call servertest.Run against itself the same way
// this test does.
type fakeServerCodec struct{}

func (fakeServerCodec) MatchRequest(req *http.Request) bool {
	return req.Method == http.MethodPost && req.URL.Path == "/v1/fake/messages"
}

type fakeRequestBody struct {
	Model  string `json:"model"`
	Stream bool   `json:"stream"`
}

func (fakeServerCodec) DecodeRequest(req *http.Request) (codec.DecodedRequest, error) {
	if ct := req.Header.Get("Content-Type"); ct != fakeContentType {
		return codec.DecodedRequest{}, errors.New("fakeServerCodec: unsupported content type " + ct)
	}
	body, err := io.ReadAll(req.Body)
	if err != nil {
		return codec.DecodedRequest{}, err
	}
	var decoded fakeRequestBody
	if err := json.Unmarshal(body, &decoded); err != nil {
		return codec.DecodedRequest{}, err
	}
	if decoded.Model == "" {
		return codec.DecodedRequest{}, errors.New("fakeServerCodec: missing model")
	}
	return codec.DecodedRequest{
		Request:        inference.Request{},
		RequestedModel: decoded.Model,
		Streaming:      decoded.Stream,
	}, nil
}

func (fakeServerCodec) WriteResponse(w http.ResponseWriter, resp *inference.Response) error {
	w.Header().Set("Content-Type", fakeContentType)
	w.WriteHeader(http.StatusOK)
	_, err := w.Write([]byte(`{"status":"ok"}`))
	return err
}

func (fakeServerCodec) WriteError(w http.ResponseWriter, err error) {
	w.Header().Set("Content-Type", fakeContentType)
	w.WriteHeader(http.StatusBadRequest)
	msg := "unknown error"
	if err != nil {
		msg = err.Error()
	}
	_, _ = w.Write([]byte(`{"error":"` + msg + `"}`))
}

func (fakeServerCodec) OpenStream(w http.ResponseWriter) (codec.StreamEncoder, error) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)
	return &fakeStreamEncoder{w: w, flusher: flusher}, nil
}

// Compile-time assertion, mirroring codec/contracts_test.go's style.
var _ codec.ServerCodec = fakeServerCodec{}

// fakeStreamEncoder is request-scoped: OpenStream returns a fresh instance per
// call, and it owns terminal-state enforcement (single ownership of stream
// closure) for that one stream.
type fakeStreamEncoder struct {
	w       http.ResponseWriter
	flusher http.Flusher
	done    bool
}

func (e *fakeStreamEncoder) WriteChunk(chunk content.Chunk) error {
	if e.done {
		return errors.New("fakeStreamEncoder: WriteChunk after stream termination")
	}
	var text string
	if tc, ok := chunk.(*content.TextChunk); ok {
		text = tc.Text
	}
	if _, err := e.w.Write([]byte("event: chunk\ndata: " + text + "\n\n")); err != nil {
		return err
	}
	if e.flusher != nil {
		e.flusher.Flush()
	}
	return nil
}

func (e *fakeStreamEncoder) Finish(result stream.StreamResult) error {
	if e.done {
		return errors.New("fakeStreamEncoder: Finish after stream termination")
	}
	e.done = true
	_, err := e.w.Write([]byte("event: done\ndata: {}\n\n"))
	return err
}

func (e *fakeStreamEncoder) Fail(err error) error {
	if e.done {
		return errors.New("fakeStreamEncoder: Fail after stream termination")
	}
	e.done = true
	msg := "unknown error"
	if err != nil {
		msg = err.Error()
	}
	_, writeErr := e.w.Write([]byte("event: error\ndata: " + msg + "\n\n"))
	return writeErr
}

// Compile-time assertion.
var _ codec.StreamEncoder = (*fakeStreamEncoder)(nil)

// TestFakeServerCodec_SatisfiesContract runs the reusable suite against
// fakeServerCodec, the only concrete codec.ServerCodec available at this point
// in the plan (the real dialect codecs land in later tasks). It both exercises
// servertest.Run's Config-driven design and pins fakeServerCodec itself as a
// worked example future codec tests can model.
func TestFakeServerCodec_SatisfiesContract(t *testing.T) {
	servertest.Run(t, servertest.Config{
		NewCodec:         func() codec.ServerCodec { return fakeServerCodec{} },
		Method:           http.MethodPost,
		Path:             "/v1/fake/messages",
		ValidBody:        []byte(`{"model":"fake-model-1","stream":false}`),
		UnmatchedMethod:  http.MethodGet,
		UnmatchedPath:    "/v1/other",
		WrongContentType: "text/plain",
		MalformedBody:    []byte(`{"model":`),
		SampleResponse:   &inference.Response{},
		SampleChunks:     []content.Chunk{&content.TextChunk{Text: "hello"}},
		SampleResult:     stream.StreamResult{},
		SampleError:      errors.New("boom"),
	})
}
