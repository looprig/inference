package bedrockconverse_test

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/bedrockconverse"
)

// Converse's ContentBlock union carries an `audio` member that this codec had
// never mapped, even though mcp/pkg/harness constructs a content.AudioBlock
// from an MCP audio tool result and the harness persists it. The Smithy model
// declares:
//
//	AudioBlock   required [format, source], format is the AudioFormat enum
//	AudioSource  union of bytes (@length min 1) and s3Location
//
// AudioFormat has fifteen members, most of which the shared content.MediaType
// vocabulary has no name for, so the mapping is an allowlist in both
// directions: a format AWS adds later, or a media type Looprig adds later,
// fails closed instead of travelling as an invented enum value.

var mp3Bytes = []byte{0x49, 0x44, 0x33, 0x04}

func TestEncodeRequest_AudioBlock(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name       string
		mediaType  content.MediaType
		wantFormat string
	}{
		{name: "mpeg", mediaType: content.MediaTypeAudioMPEG, wantFormat: "mp3"},
		{name: "wav", mediaType: content.MediaTypeAudioWAV, wantFormat: "wav"},
		{name: "ogg", mediaType: content.MediaTypeAudioOGG, wantFormat: "ogg"},
		{name: "flac", mediaType: content.MediaTypeAudioFLAC, wantFormat: "flac"},
		{name: "aac", mediaType: content.MediaTypeAudioAAC, wantFormat: "aac"},
		{name: "mp4", mediaType: content.MediaTypeAudioMP4, wantFormat: "mp4"},
		{name: "webm", mediaType: content.MediaTypeAudioWebM, wantFormat: "webm"},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			raw, err := bedrockconverse.EncodeRequest(inference.Request{
				Model: baseModel(),
				Messages: content.AgenticMessages{userMessage(
					&content.AudioBlock{MediaType: tc.mediaType, Data: mp3Bytes},
					&content.TextBlock{Text: "Transcribe this."},
				)},
			})
			if err != nil {
				t.Fatalf("EncodeRequest() error = %v", err)
			}
			blocks := decodeArray(t, decodeArray(t, decodeObject(t, raw)["messages"])[0]["content"])
			audio := decodeObject(t, blocks[0]["audio"])
			if got := asString(t, audio["format"]); got != tc.wantFormat {
				t.Errorf("audio.format = %q, want %q", got, tc.wantFormat)
			}
			source := decodeObject(t, audio["source"])
			if got := asString(t, source["bytes"]); got != base64.StdEncoding.EncodeToString(mp3Bytes) {
				t.Errorf("audio.source.bytes = %q, want the base64 payload", got)
			}
			if _, present := source["s3Location"]; present {
				t.Errorf("audio source carries s3Location alongside bytes, which breaks the union's arity: %s", audio["source"])
			}
		})
	}
}

func TestEncodeRequest_AudioBlockRejections(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name     string
		messages content.AgenticMessages
		want     string
	}{
		{
			// AudioFormat is an enum: a media type with no member has no legal
			// `format` value, and Converse answers an illegal one with a
			// ValidationException that names neither the block nor the field.
			name:     "media type outside the AudioFormat enum",
			messages: content.AgenticMessages{userMessage(&content.AudioBlock{MediaType: content.MediaType("audio/amr"), Data: mp3Bytes})},
			want:     "format",
		},
		{
			// AudioSource.bytes declares @length min 1.
			name:     "empty audio payload",
			messages: content.AgenticMessages{userMessage(&content.AudioBlock{MediaType: content.MediaTypeAudioWAV})},
			want:     "empty",
		},
		{
			// ToolResultContentBlock is a DIFFERENT union from ContentBlock and
			// has no audio member at all: json, text, image, document, video and
			// searchResult only. This is exactly the shape an MCP audio tool
			// result produces, so the error has to say why it cannot travel.
			name: "audio inside a tool result",
			messages: content.AgenticMessages{
				userMessage(&content.TextBlock{Text: "listen"}),
				assistantMessage(content.NewToolUseBlock("tooluse_listen", "listen", json.RawMessage(`{}`), nil, "")),
				&content.ToolResultMessage{
					ToolUseID: "tooluse_listen",
					Message:   content.Message{Role: content.RoleTool, Blocks: []content.Block{&content.AudioBlock{MediaType: content.MediaTypeAudioMPEG, Data: mp3Bytes}}},
				},
			},
			want: "toolResult",
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := bedrockconverse.EncodeRequest(inference.Request{Model: baseModel(), Messages: tc.messages})
			if err == nil {
				t.Fatalf("expected an error, got nil")
			}
			var unsupported *bedrockconverse.UnsupportedBlockError
			if !errors.As(err, &unsupported) {
				t.Fatalf("error = %v (%T), want *UnsupportedBlockError", err, err)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error = %q, want it to name %q", err, tc.want)
			}
		})
	}
}
