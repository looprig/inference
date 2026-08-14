package bedrockconverse_test

import (
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference/codec/bedrockconverse"
)

// ContentBlock is one union shared by Converse's request and response
// directions, so an audio block that can be encoded must also be decodable.
// The reverse format map is the same allowlist read backwards: an AudioFormat
// member with no content.MediaType name is refused rather than turned into an
// invented "audio/<format>" media type, which is what would happen if the
// decoder mirrored the image path's string concatenation.

func TestDecodeResponse_AudioBlock(t *testing.T) {
	t.Parallel()

	cases := []struct {
		format    string
		wantMedia content.MediaType
	}{
		{format: "mp3", wantMedia: content.MediaTypeAudioMPEG},
		{format: "mpeg", wantMedia: content.MediaTypeAudioMPEG},
		{format: "wav", wantMedia: content.MediaTypeAudioWAV},
		{format: "ogg", wantMedia: content.MediaTypeAudioOGG},
		{format: "flac", wantMedia: content.MediaTypeAudioFLAC},
		{format: "aac", wantMedia: content.MediaTypeAudioAAC},
		{format: "mp4", wantMedia: content.MediaTypeAudioMP4},
		{format: "m4a", wantMedia: content.MediaTypeAudioMP4},
		{format: "webm", wantMedia: content.MediaTypeAudioWebM},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.format, func(t *testing.T) {
			t.Parallel()
			body := []byte(`{"output":{"message":{"role":"assistant","content":[
				{"audio":{"format":"` + tc.format + `","source":{"bytes":"AQI="}}}
			]}},"stopReason":"end_turn"}`)
			response, err := bedrockconverse.DecodeResponse(body)
			if err != nil {
				t.Fatalf("DecodeResponse() error = %v", err)
			}
			audio, ok := response.Message.Blocks[0].(*content.AudioBlock)
			if !ok {
				t.Fatalf("Blocks[0] type = %T, want *content.AudioBlock", response.Message.Blocks[0])
			}
			if audio.MediaType != tc.wantMedia {
				t.Errorf("MediaType = %q, want %q", audio.MediaType, tc.wantMedia)
			}
			if string(audio.Data) != string([]byte{1, 2}) {
				t.Errorf("Data = %v, want the decoded bytes", audio.Data)
			}
		})
	}
}

// TestDecodeResponse_RejectsUnrepresentableSources covers the union members
// that are legal on the wire and have no neutral counterpart. Each one is
// refused by name; decoding any of them into an empty block would report a
// successful decode of content that had vanished.
func TestDecodeResponse_RejectsUnrepresentableSources(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name  string
		block string
		want  string
	}{
		{
			name:  "audio s3 location",
			block: `{"audio":{"format":"mp3","source":{"s3Location":{"uri":"s3://bucket/clip.mp3"}}}}`,
			want:  "s3Location",
		},
		{
			name:  "audio format outside the neutral vocabulary",
			block: `{"audio":{"format":"pcm","source":{"bytes":"AQI="}}}`,
			want:  "pcm",
		},
		{
			name:  "audio format outside the enum entirely",
			block: `{"audio":{"format":"amr","source":{"bytes":"AQI="}}}`,
			want:  "amr",
		},
		{
			name:  "audio with two source members",
			block: `{"audio":{"format":"mp3","source":{"bytes":"AQI=","s3Location":{"uri":"s3://bucket/clip.mp3"}}}}`,
			want:  "source",
		},
		{
			name:  "audio with an empty payload",
			block: `{"audio":{"format":"mp3","source":{}}}`,
			want:  "source",
		},
		{
			name:  "document s3 location",
			block: `{"document":{"format":"pdf","name":"filing","source":{"s3Location":{"uri":"s3://bucket/filing.pdf"}}}}`,
			want:  "s3Location",
		},
		{
			name:  "document content blocks",
			block: `{"document":{"format":"txt","name":"notes","source":{"content":[{"text":"chunk"}]}}}`,
			want:  "content",
		},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			body := []byte(`{"output":{"message":{"role":"assistant","content":[` + tc.block + `]}},"stopReason":"end_turn"}`)
			_, err := bedrockconverse.DecodeResponse(body)
			if err == nil {
				t.Fatalf("DecodeResponse() error = nil, want an error naming %q", tc.want)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error = %q, want it to name %q", err, tc.want)
			}
		})
	}
}
