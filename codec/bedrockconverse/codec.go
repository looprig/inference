package bedrockconverse

import (
	"bytes"
	"encoding/json"
	"net/http"

	"github.com/looprig/inference"
	codec "github.com/looprig/inference/codec"
	"github.com/looprig/inference/wire/jsonbody"
)

// Codec is Amazon Bedrock's native Converse API dialect. The request body is
// shared by Converse and ConverseStream; streaming is selected by the provider
// route and response content type, not by a JSON flag.
type Codec struct{}

var _ codec.Codec = Codec{}

// EncodeRequest builds the native Converse JSON request. The model ID is
// intentionally absent: Bedrock identifies it in the URL path.
func EncodeRequest(req inference.Request) ([]byte, error) {
	r, err := buildRequest(req)
	if err != nil {
		return nil, err
	}
	return marshalRequest(r)
}

// EncodeCountTokensInput emits the Converse union member accepted by Bedrock's
// CountTokens API. Inference controls, output formatting, response-field paths,
// guardrails, and service tiers are intentionally excluded because CountTokens
// accepts only the conversation, system, tool configuration, and additional
// model-request fields.
func EncodeCountTokensInput(req inference.Request) ([]byte, error) {
	r, err := buildRequest(req)
	if err != nil {
		return nil, err
	}
	return json.Marshal(converseCountTokensRequest{
		Messages:   r.Messages,
		System:     r.System,
		ToolConfig: r.ToolConfig,
	})
}

// EncodeRequest implements codec.RequestEncoder. Converse and ConverseStream
// use the same body, so mode is intentionally ignored.
func (Codec) EncodeRequest(req inference.Request, _ codec.RequestMode) (codec.EncodedRequest, error) {
	body, err := EncodeRequest(req)
	if err != nil {
		return codec.EncodedRequest{}, err
	}
	header := http.Header{}
	header.Set("Content-Type", jsonbody.ContentType)
	return codec.EncodedRequest{Header: header, Body: bytes.NewReader(body)}, nil
}

// DecodeResponse is completed in decode.go; keeping the method here makes the
// codec's public shape explicit while request encoding is developed separately.
func (Codec) DecodeResponse(body []byte) (*inference.Response, error) {
	return DecodeResponse(body)
}
