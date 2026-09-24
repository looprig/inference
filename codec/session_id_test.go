package codec_test

import (
	"bytes"
	"errors"
	"io"
	"reflect"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec"
	"github.com/looprig/inference/codec/anthropicapi"
	"github.com/looprig/inference/codec/bedrockconverse"
	"github.com/looprig/inference/codec/geminiapi"
	"github.com/looprig/inference/codec/openaiapi"
	"github.com/looprig/inference/codec/openairesponses"
	"github.com/looprig/inference/model"
)

// TestEveryCodecIgnoresButValidatesSessionID pins the two halves of the
// Request.SessionID contract at the encoders, in both request modes:
//
//   - A valid SessionID changes nothing a codec returns: not the body and not
//     the header map. The identity is carried, where a provider documents it,
//     as a header added by that PROVIDER's transport; no neutral codec may leak
//     it onto the wire, where an upstream that does not know it would receive
//     caller data it never asked for.
//   - An unsendable SessionID is refused by every encoder with the typed
//     *inference.InvalidSessionIDError, including by the codecs whose providers
//     never read it — so a request valid for one provider stays valid when a
//     conversation switches to another.
func TestEveryCodecIgnoresButValidatesSessionID(t *testing.T) {
	t.Parallel()

	codecs := []struct {
		name    string
		encoder codec.RequestEncoder
	}{
		{name: "anthropic", encoder: anthropicapi.Codec{}},
		{name: "openai", encoder: openaiapi.Codec{}},
		{name: "openai-responses", encoder: openairesponses.Codec{}},
		{name: "gemini", encoder: geminiapi.Codec{}},
		{name: "bedrock-converse", encoder: bedrockconverse.Codec{}},
	}
	modes := []struct {
		name string
		mode codec.RequestMode
	}{
		{name: "invoke", mode: codec.RequestModeInvoke},
		{name: "stream", mode: codec.RequestModeStream},
	}

	const sessionID = "session-marker-6f1e2d3c"
	base := func() inference.Request {
		return inference.Request{
			Model: model.Model{Name: "m"},
			Messages: content.AgenticMessages{
				&content.UserMessage{Message: content.Message{
					Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "hello"}},
				}},
			},
		}
	}
	encode := func(t *testing.T, encoder codec.RequestEncoder, req inference.Request, mode codec.RequestMode) (codec.EncodedRequest, []byte, error) {
		t.Helper()
		encoded, err := encoder.EncodeRequest(req, mode)
		if err != nil || encoded.Body == nil {
			return encoded, nil, err
		}
		body, readErr := io.ReadAll(encoded.Body)
		if readErr != nil {
			t.Fatalf("read encoded body: %v", readErr)
		}
		return encoded, body, nil
	}

	for _, c := range codecs {
		for _, m := range modes {
			t.Run(c.name+"/"+m.name, func(t *testing.T) {
				t.Parallel()

				without, withoutBody, err := encode(t, c.encoder, base(), m.mode)
				if err != nil {
					t.Fatalf("encode without SessionID: %v", err)
				}
				withID := base()
				withID.SessionID = sessionID
				with, withBody, err := encode(t, c.encoder, withID, m.mode)
				if err != nil {
					t.Fatalf("encode with SessionID: %v", err)
				}

				if !bytes.Equal(withoutBody, withBody) {
					t.Errorf("SessionID changed the encoded body:\nwithout: %s\nwith:    %s", withoutBody, withBody)
				}
				if strings.Contains(string(withBody), sessionID) {
					t.Errorf("encoded body carries the SessionID: %s", withBody)
				}
				if !reflect.DeepEqual(without.Header, with.Header) {
					t.Errorf("SessionID changed the encoded header map:\nwithout: %v\nwith:    %v", without.Header, with.Header)
				}
				for name, values := range with.Header {
					if strings.Contains(name, sessionID) {
						t.Errorf("encoded header name %q carries the SessionID", name)
					}
					for _, value := range values {
						if strings.Contains(value, sessionID) {
							t.Errorf("encoded header %s = %q carries the SessionID", name, value)
						}
					}
				}

				invalid := base()
				invalid.SessionID = "id\r\nX-Injected: yes"
				_, _, err = encode(t, c.encoder, invalid, m.mode)
				var sessionErr *inference.InvalidSessionIDError
				if !errors.As(err, &sessionErr) {
					t.Fatalf("encode with unsendable SessionID: err = %T %v, want *inference.InvalidSessionIDError", err, err)
				}
			})
		}
	}
}
