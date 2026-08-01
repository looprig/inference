package eventstream_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"io"
	"strings"
	"testing"

	codec "github.com/looprig/inference/codec"
	stream "github.com/looprig/inference/stream"
	"github.com/looprig/inference/wire/eventstream"
)

const (
	headerTypeString = 7
	headerTypeByte   = 2
)

type testHeader struct {
	name  string
	typ   byte
	value string
}

func buildFrame(headers []testHeader, payload []byte) []byte {
	var encodedHeaders bytes.Buffer
	for _, header := range headers {
		encodedHeaders.WriteByte(byte(len(header.name)))
		encodedHeaders.WriteString(header.name)
		encodedHeaders.WriteByte(header.typ)
		switch header.typ {
		case headerTypeString:
			var length [2]byte
			binary.BigEndian.PutUint16(length[:], uint16(len(header.value)))
			encodedHeaders.Write(length[:])
			encodedHeaders.WriteString(header.value)
		case headerTypeByte:
			encodedHeaders.WriteByte(header.value[0])
		}
	}

	totalLength := 16 + encodedHeaders.Len() + len(payload)
	frame := make([]byte, totalLength)
	binary.BigEndian.PutUint32(frame[0:4], uint32(totalLength))
	binary.BigEndian.PutUint32(frame[4:8], uint32(encodedHeaders.Len()))
	binary.BigEndian.PutUint32(frame[8:12], crc32.ChecksumIEEE(frame[:8]))
	copy(frame[12:], encodedHeaders.Bytes())
	copy(frame[12+encodedHeaders.Len():], payload)
	binary.BigEndian.PutUint32(frame[totalLength-4:], crc32.ChecksumIEEE(frame[:totalLength-4]))
	return frame
}

func corruptPreludeCRC(frame []byte) []byte {
	out := append([]byte(nil), frame...)
	out[8]++
	return out
}

func corruptMessageCRC(frame []byte) []byte {
	out := append([]byte(nil), frame...)
	out[len(out)-1]++
	return out
}

type fragmentedReader struct {
	data []byte
	pos  int
}

func (r *fragmentedReader) Read(p []byte) (int, error) {
	if r.pos == len(r.data) {
		return 0, io.EOF
	}
	p[0] = r.data[r.pos]
	r.pos++
	return 1, nil
}

func (r *fragmentedReader) Close() error { return nil }

type closingReader struct {
	io.Reader
	closed bool
}

func (r *closingReader) Close() error {
	r.closed = true
	return nil
}

type failingReader struct {
	prefix []byte
	err    error
	pos    int
}

func (r *failingReader) Read(p []byte) (int, error) {
	if r.pos == len(r.prefix) {
		return 0, r.err
	}
	n := copy(p, r.prefix[r.pos:])
	r.pos += n
	return n, nil
}

func (r *failingReader) Close() error { return nil }

func collect(r *stream.StreamReader[stream.StreamFrame]) ([]stream.StreamFrame, error) {
	var frames []stream.StreamFrame
	for {
		frame, err := r.Next()
		if err != nil {
			return frames, err
		}
		frames = append(frames, frame)
	}
}

func TestDecodeStreamFrames_ValidFrame(t *testing.T) {
	t.Parallel()

	body := buildFrame([]testHeader{{name: ":message-type", typ: headerTypeString, value: "event"}, {name: "event-type", typ: headerTypeString, value: "chunk"}}, []byte(`{"messageStart":{}}`))
	r, err := eventstream.DecodeStreamFrames(io.NopCloser(bytes.NewReader(body)))
	if err != nil {
		t.Fatalf("DecodeStreamFrames() error = %v", err)
	}
	defer r.Close()

	frames, terminalErr := collect(r)
	if !errors.Is(terminalErr, io.EOF) {
		t.Fatalf("terminal error = %v, want io.EOF", terminalErr)
	}
	if len(frames) != 1 {
		t.Fatalf("frame count = %d, want 1", len(frames))
	}
	if got, want := string(frames[0].Data), `{"messageStart":{}}`; got != want {
		t.Errorf("Data = %q, want %q", got, want)
	}
	if got, want := frames[0].Metadata[":message-type"], "event"; got != want {
		t.Errorf("Metadata[:message-type] = %q, want %q", got, want)
	}
	if got, want := frames[0].Metadata["event-type"], "chunk"; got != want {
		t.Errorf("Metadata[event-type] = %q, want %q", got, want)
	}
}

func TestDecodeStreamFrames_FragmentedAndMultiple(t *testing.T) {
	t.Parallel()

	first := buildFrame([]testHeader{{name: "event-type", typ: headerTypeString, value: "chunk"}}, []byte(`{"a":1}`))
	second := buildFrame(nil, []byte(`{"b":2}`))
	data := append(first, second...)
	r, err := eventstream.DecodeStreamFrames(&fragmentedReader{data: data})
	if err != nil {
		t.Fatalf("DecodeStreamFrames() error = %v", err)
	}
	defer r.Close()

	frames, terminalErr := collect(r)
	if !errors.Is(terminalErr, io.EOF) {
		t.Fatalf("terminal error = %v, want io.EOF", terminalErr)
	}
	if len(frames) != 2 || string(frames[0].Data) != `{"a":1}` || string(frames[1].Data) != `{"b":2}` {
		t.Fatalf("frames = %#v, want two JSON payloads", frames)
	}
}

func TestDecodeStreamFrames_CRCValidation(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name  string
		data  []byte
		match string
	}{
		{name: "prelude crc", data: corruptPreludeCRC(buildFrame(nil, []byte("x"))), match: "prelude crc"},
		{name: "message crc", data: corruptMessageCRC(buildFrame(nil, []byte("x"))), match: "message crc"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			r, err := eventstream.DecodeStreamFrames(io.NopCloser(bytes.NewReader(tc.data)))
			if err != nil {
				t.Fatalf("DecodeStreamFrames() error = %v", err)
			}
			defer r.Close()
			_, err = r.Next()
			var frameErr *eventstream.FramerError
			if !errors.As(err, &frameErr) || !strings.Contains(err.Error(), tc.match) {
				t.Fatalf("Next() error = %T (%v), want FramerError containing %q", err, err, tc.match)
			}
		})
	}
}

func TestDecodeStreamFrames_InvalidAndTruncated(t *testing.T) {
	t.Parallel()

	valid := buildFrame(nil, []byte("payload"))
	tooShort := make([]byte, 12)
	binary.BigEndian.PutUint32(tooShort[:4], 15)
	binary.BigEndian.PutUint32(tooShort[4:8], 0)
	binary.BigEndian.PutUint32(tooShort[8:12], crc32.ChecksumIEEE(tooShort[:8]))

	impossibleHeaders := make([]byte, 16)
	binary.BigEndian.PutUint32(impossibleHeaders[:4], 16)
	binary.BigEndian.PutUint32(impossibleHeaders[4:8], 1)
	binary.BigEndian.PutUint32(impossibleHeaders[8:12], crc32.ChecksumIEEE(impossibleHeaders[:8]))
	binary.BigEndian.PutUint32(impossibleHeaders[12:], crc32.ChecksumIEEE(impossibleHeaders[:12]))

	overSize := make([]byte, 12)
	binary.BigEndian.PutUint32(overSize[:4], 17<<20)
	binary.BigEndian.PutUint32(overSize[4:8], 0)
	binary.BigEndian.PutUint32(overSize[8:12], crc32.ChecksumIEEE(overSize[:8]))

	cases := []struct {
		name string
		data []byte
	}{
		{name: "truncated prelude", data: []byte{0, 0, 0}},
		{name: "truncated frame", data: valid[:len(valid)-1]},
		{name: "total length below minimum", data: tooShort},
		{name: "header length exceeds frame", data: impossibleHeaders},
		{name: "oversized frame", data: overSize},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			r, err := eventstream.DecodeStreamFrames(io.NopCloser(bytes.NewReader(tc.data)))
			if err != nil {
				t.Fatalf("DecodeStreamFrames() error = %v", err)
			}
			defer r.Close()
			_, err = r.Next()
			var frameErr *eventstream.FramerError
			if !errors.As(err, &frameErr) {
				t.Fatalf("Next() error = %T (%v), want *FramerError", err, err)
			}
		})
	}
}

func TestDecodeStreamFrames_HeaderValidation(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		headers []testHeader
	}{
		{name: "empty header name", headers: []testHeader{{name: "", typ: headerTypeString, value: "x"}}},
		{name: "unsupported header type", headers: []testHeader{{name: "flag", typ: headerTypeByte, value: "x"}}},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			r, err := eventstream.DecodeStreamFrames(io.NopCloser(bytes.NewReader(buildFrame(tc.headers, nil))))
			if err != nil {
				t.Fatalf("DecodeStreamFrames() error = %v", err)
			}
			defer r.Close()
			_, err = r.Next()
			var frameErr *eventstream.FramerError
			if !errors.As(err, &frameErr) {
				t.Fatalf("Next() error = %T (%v), want *FramerError", err, err)
			}
		})
	}
}

func TestDecodeStreamFrames_EOFAndClose(t *testing.T) {
	t.Parallel()

	empty, err := eventstream.DecodeStreamFrames(io.NopCloser(strings.NewReader("")))
	if err != nil {
		t.Fatalf("DecodeStreamFrames(empty) error = %v", err)
	}
	if _, err := empty.Next(); !errors.Is(err, io.EOF) {
		t.Fatalf("empty Next() error = %v, want io.EOF", err)
	}
	_ = empty.Close()

	spy := &closingReader{Reader: strings.NewReader("")}
	r, err := eventstream.DecodeStreamFrames(spy)
	if err != nil {
		t.Fatalf("DecodeStreamFrames() error = %v", err)
	}
	if err := r.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if !spy.closed {
		t.Fatal("Close() did not close the body")
	}
	if err := r.Close(); err != nil {
		t.Fatalf("second Close() error = %v", err)
	}
}

func TestDecodeStreamFrames_ReadErrorAndNilBody(t *testing.T) {
	t.Parallel()

	boom := errors.New("connection reset")
	valid := buildFrame(nil, []byte("payload"))
	r, err := eventstream.DecodeStreamFrames(&failingReader{prefix: valid[:5], err: boom})
	if err != nil {
		t.Fatalf("DecodeStreamFrames() error = %v", err)
	}
	defer r.Close()
	_, err = r.Next()
	if !errors.Is(err, boom) {
		t.Fatalf("Next() error = %v, want %v", err, boom)
	}
	var frameErr *eventstream.FramerError
	if !errors.As(err, &frameErr) {
		t.Fatalf("Next() error = %T (%v), want *FramerError", err, err)
	}

	_, err = eventstream.DecodeStreamFrames(nil)
	if !errors.As(err, &frameErr) {
		t.Fatalf("nil body error = %T (%v), want *FramerError", err, err)
	}
}

func TestFramer(t *testing.T) {
	t.Parallel()
	var f codec.StreamFramer = eventstream.Framer()
	r, err := f.DecodeStreamFrames(io.NopCloser(bytes.NewReader(buildFrame(nil, []byte("ok")))))
	if err != nil {
		t.Fatalf("DecodeStreamFrames() error = %v", err)
	}
	defer r.Close()
	frame, err := r.Next()
	if err != nil || string(frame.Data) != "ok" {
		t.Fatalf("first frame = %#v, error = %v, want Data=ok", frame, err)
	}
}
