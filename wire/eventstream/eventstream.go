// Package eventstream frames AWS Event Stream messages into raw inference
// stream frames. Bedrock ConverseStream uses this binary protocol rather than
// Server-Sent Events.
package eventstream

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
	"unicode/utf8"

	codec "github.com/looprig/inference/codec"
	stream "github.com/looprig/inference/stream"
)

const (
	preludeBytes    = 12 // total length, headers length, and prelude CRC
	messageCRCBytes = 4
	minFrameBytes   = preludeBytes + messageCRCBytes
	maxFrameBytes   = 16 << 20

	headerTypeString = 7
)

// FramerError identifies a malformed AWS Event Stream frame or a read failure
// encountered while reading one. Err remains available through errors.Is/As.
type FramerError struct {
	Reason string
	Err    error
}

func (e *FramerError) Error() string {
	if e.Err != nil {
		return "eventstream: " + e.Reason + ": " + e.Err.Error()
	}
	return "eventstream: " + e.Reason
}

func (e *FramerError) Unwrap() error { return e.Err }

var _ codec.StreamFramer = framer{}

type framer struct{}

func (framer) DecodeStreamFrames(body io.ReadCloser) (*stream.StreamReader[stream.StreamFrame], error) {
	return DecodeStreamFrames(body)
}

// Framer returns the package's StreamFramer as an injectable value.
func Framer() codec.StreamFramer { return framer{} }

// DecodeStreamFrames lazily decodes one AWS Event Stream message per Next call.
// The returned reader owns body and closes it when Close is called. A clean EOF
// is returned only when no bytes remain at the start of a new frame.
func DecodeStreamFrames(body io.ReadCloser) (*stream.StreamReader[stream.StreamFrame], error) {
	if body == nil {
		return nil, &FramerError{Reason: "nil body"}
	}

	next := func() (stream.StreamFrame, error) {
		var prelude [preludeBytes]byte
		n, err := io.ReadFull(body, prelude[:])
		if err != nil {
			if err == io.EOF && n == 0 {
				return stream.StreamFrame{}, io.EOF
			}
			return stream.StreamFrame{}, &FramerError{Reason: "read prelude", Err: err}
		}

		totalLength := binary.BigEndian.Uint32(prelude[0:4])
		headerLength := binary.BigEndian.Uint32(prelude[4:8])
		if got, want := binary.BigEndian.Uint32(prelude[8:12]), crc32.ChecksumIEEE(prelude[:8]); got != want {
			return stream.StreamFrame{}, &FramerError{Reason: fmt.Sprintf("invalid prelude crc (got %08x, want %08x)", got, want)}
		}
		if totalLength < minFrameBytes {
			return stream.StreamFrame{}, &FramerError{Reason: fmt.Sprintf("invalid total length %d", totalLength)}
		}
		if totalLength > maxFrameBytes {
			return stream.StreamFrame{}, &FramerError{Reason: fmt.Sprintf("frame exceeds %d bytes", maxFrameBytes)}
		}
		if uint64(headerLength) > uint64(totalLength)-minFrameBytes {
			return stream.StreamFrame{}, &FramerError{Reason: fmt.Sprintf("invalid headers length %d for frame length %d", headerLength, totalLength)}
		}

		frame := make([]byte, int(totalLength))
		copy(frame, prelude[:])
		if _, err := io.ReadFull(body, frame[preludeBytes:]); err != nil {
			return stream.StreamFrame{}, &FramerError{Reason: "read frame", Err: err}
		}
		if got, want := binary.BigEndian.Uint32(frame[len(frame)-messageCRCBytes:]), crc32.ChecksumIEEE(frame[:len(frame)-messageCRCBytes]); got != want {
			return stream.StreamFrame{}, &FramerError{Reason: fmt.Sprintf("invalid message crc (got %08x, want %08x)", got, want)}
		}

		headerEnd := preludeBytes + int(headerLength)
		metadata, err := decodeHeaders(frame[preludeBytes:headerEnd])
		if err != nil {
			return stream.StreamFrame{}, err
		}
		payloadEnd := len(frame) - messageCRCBytes
		payload := make([]byte, payloadEnd-headerEnd)
		copy(payload, frame[headerEnd:payloadEnd])
		name := metadata[":event-type"]
		if name == "" {
			name = metadata[":message-type"]
		}
		return stream.StreamFrame{Name: name, Metadata: metadata, Data: payload}, nil
	}

	return stream.NewStreamReader(next, body.Close), nil
}

func decodeHeaders(encoded []byte) (map[string]string, error) {
	if len(encoded) == 0 {
		return nil, nil
	}
	metadata := make(map[string]string)
	for offset := 0; offset < len(encoded); {
		nameLength := int(encoded[offset])
		offset++
		if nameLength == 0 {
			return nil, &FramerError{Reason: "empty header name"}
		}
		if nameLength > len(encoded)-offset {
			return nil, &FramerError{Reason: "truncated header name"}
		}
		name := string(encoded[offset : offset+nameLength])
		offset += nameLength
		if !utf8.ValidString(name) {
			return nil, &FramerError{Reason: "invalid UTF-8 header name"}
		}
		if offset >= len(encoded) {
			return nil, &FramerError{Reason: "missing header value type"}
		}
		typ := encoded[offset]
		offset++
		if typ != headerTypeString {
			return nil, &FramerError{Reason: fmt.Sprintf("unsupported header value type %d for %q", typ, name)}
		}
		if len(encoded)-offset < 2 {
			return nil, &FramerError{Reason: fmt.Sprintf("truncated string header %q", name)}
		}
		valueLength := int(binary.BigEndian.Uint16(encoded[offset : offset+2]))
		offset += 2
		if valueLength > len(encoded)-offset {
			return nil, &FramerError{Reason: fmt.Sprintf("truncated string header value %q", name)}
		}
		value := string(encoded[offset : offset+valueLength])
		offset += valueLength
		if _, exists := metadata[name]; exists {
			return nil, &FramerError{Reason: fmt.Sprintf("duplicate header %q", name)}
		}
		metadata[name] = value
	}
	return metadata, nil
}
