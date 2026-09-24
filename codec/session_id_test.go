package codec_test

import (
	"bytes"
	"errors"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/anthropicapi"
	"github.com/looprig/inference/codec/bedrockconverse"
	"github.com/looprig/inference/codec/geminiapi"
	"github.com/looprig/inference/codec/openaiapi"
	"github.com/looprig/inference/codec/openairesponses"
	"github.com/looprig/inference/model"
)

// TestEveryCodecIgnoresButValidatesSessionID pins the two halves of the
// Request.SessionID contract at the encoders:
//
//   - A valid SessionID changes no wire byte. The identity is carried, where a
//     provider documents it, as a HEADER added by that provider's transport; no
//     neutral codec may leak it into a request body, where an upstream that
//     does not know it would reject the unknown member or log it.
//   - An unsendable SessionID is refused by every encoder with the typed
//     *inference.InvalidSessionIDError, before any body is produced, including
//     by the codecs whose providers never read it — so a request valid for one
//     provider stays valid when a conversation switches to another.
func TestEveryCodecIgnoresButValidatesSessionID(t *testing.T) {
	t.Parallel()

	codecs := []struct {
		name   string
		encode func(inference.Request) ([]byte, error)
	}{
		{name: "anthropic", encode: func(req inference.Request) ([]byte, error) { return anthropicapi.EncodeRequest(req, false) }},
		{name: "openai", encode: func(req inference.Request) ([]byte, error) { return openaiapi.EncodeRequest(req, false) }},
		{name: "openai-responses", encode: func(req inference.Request) ([]byte, error) { return openairesponses.EncodeRequest(req, false) }},
		{name: "gemini", encode: geminiapi.EncodeRequest},
		{name: "bedrock-converse", encode: bedrockconverse.EncodeRequest},
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

	for _, codec := range codecs {
		t.Run(codec.name, func(t *testing.T) {
			t.Parallel()

			without, err := codec.encode(base())
			if err != nil {
				t.Fatalf("encode without SessionID: %v", err)
			}
			withID := base()
			withID.SessionID = sessionID
			with, err := codec.encode(withID)
			if err != nil {
				t.Fatalf("encode with SessionID: %v", err)
			}
			if !bytes.Equal(without, with) {
				t.Errorf("SessionID changed the encoded body:\nwithout: %s\nwith:    %s", without, with)
			}
			if strings.Contains(string(with), sessionID) {
				t.Errorf("encoded body carries the SessionID: %s", with)
			}

			invalid := base()
			invalid.SessionID = "id\r\nX-Injected: yes"
			body, err := codec.encode(invalid)
			var sessionErr *inference.InvalidSessionIDError
			if !errors.As(err, &sessionErr) {
				t.Fatalf("encode with unsendable SessionID: err = %T %v, want *inference.InvalidSessionIDError", err, err)
			}
			if body != nil {
				t.Errorf("encode with unsendable SessionID returned a body: %s", body)
			}
		})
	}
}
