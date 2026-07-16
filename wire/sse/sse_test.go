package sse_test

import (
	"errors"
	"io"
	"strings"
	"testing"

	codec "github.com/looprig/inference/codec"
	stream "github.com/looprig/inference/stream"
	"github.com/looprig/inference/wire/sse"
)

// closerSpy records whether Close was called on the framed body.
type closerSpy struct {
	io.Reader
	closed bool
}

func (c *closerSpy) Close() error {
	c.closed = true
	return nil
}

// errorReader emits a fixed prefix then fails, to exercise the read-error path.
type errorReader struct {
	prefix []byte
	pos    int
	err    error
}

func (e *errorReader) Read(p []byte) (int, error) {
	if e.pos >= len(e.prefix) {
		return 0, e.err
	}
	n := copy(p, e.prefix[e.pos:])
	e.pos += n
	return n, nil
}
func (e *errorReader) Close() error { return nil }

// collect drains a frame reader into a slice plus the terminal error.
func collect(r *stream.StreamReader[stream.StreamFrame]) ([]stream.StreamFrame, error) {
	var frames []stream.StreamFrame
	for {
		f, err := r.Next()
		if err != nil {
			return frames, err
		}
		frames = append(frames, f)
	}
}

func TestDecodeStreamFrames(t *testing.T) {
	t.Parallel()

	type want struct {
		name string
		data string
	}
	cases := []struct {
		name string
		body string
		want []want
	}{
		{
			name: "single data frame",
			body: "data: {\"a\":1}\n\n",
			want: []want{{data: `{"a":1}`}},
		},
		{
			name: "two frames blank-line separated",
			body: "data: {\"a\":1}\n\ndata: {\"b\":2}\n\n",
			want: []want{{data: `{"a":1}`}, {data: `{"b":2}`}},
		},
		{
			name: "multi-line data joined with newline",
			body: "data: line1\ndata: line2\ndata: line3\n\n",
			want: []want{{data: "line1\nline2\nline3"}},
		},
		{
			name: "comment lines ignored",
			body: ": keep-alive\n\ndata: hello\n\n: another comment\n\ndata: world\n\n",
			want: []want{{data: "hello"}, {data: "world"}},
		},
		{
			name: "event name preserved on frame",
			body: "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
			want: []want{{name: "message_stop", data: `{"type":"message_stop"}`}},
		},
		{
			name: "DONE returned as an ordinary frame, not swallowed",
			body: "data: {\"choices\":[]}\n\ndata: [DONE]\n\n",
			want: []want{{data: `{"choices":[]}`}, {data: "[DONE]"}},
		},
		{
			name: "unknown fields ignored, id ignored",
			body: "id: 42\nretry: 1000\ndata: payload\n\n",
			want: []want{{data: "payload"}},
		},
		{
			name: "leading space after colon stripped once",
			body: "data:  two-spaces\n\n",
			want: []want{{data: " two-spaces"}},
		},
		{
			name: "CRLF line endings framed",
			body: "data: {\"x\":1}\r\n\r\n",
			want: []want{{data: `{"x":1}`}},
		},
		{
			name: "event with no data dispatches nothing",
			body: "event: ping\n\ndata: real\n\n",
			want: []want{{data: "real"}},
		},
		{
			name: "trailing event without blank line is flushed at EOF",
			body: "data: last",
			want: []want{{data: "last"}},
		},
		{
			name: "empty stream yields no frames",
			body: "",
			want: nil,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			r, err := sse.DecodeStreamFrames(io.NopCloser(strings.NewReader(tc.body)))
			if err != nil {
				t.Fatalf("DecodeStreamFrames() error = %v", err)
			}
			defer r.Close()
			got, termErr := collect(r)
			if !errors.Is(termErr, io.EOF) {
				t.Fatalf("terminal error = %v, want io.EOF", termErr)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("frame count = %d (%v), want %d", len(got), got, len(tc.want))
			}
			for i, w := range tc.want {
				if got[i].Name != w.name {
					t.Errorf("frame[%d].Name = %q, want %q", i, got[i].Name, w.name)
				}
				if string(got[i].Data) != w.data {
					t.Errorf("frame[%d].Data = %q, want %q", i, got[i].Data, w.data)
				}
			}
		})
	}
}

func TestDecodeStreamFrames_ClosesBody(t *testing.T) {
	t.Parallel()
	spy := &closerSpy{Reader: strings.NewReader("data: x\n\n")}
	r, err := sse.DecodeStreamFrames(spy)
	if err != nil {
		t.Fatalf("DecodeStreamFrames() error = %v", err)
	}
	if err := r.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if !spy.closed {
		t.Error("expected body closed after Close()")
	}
}

func TestDecodeStreamFrames_ReadError(t *testing.T) {
	t.Parallel()
	boom := errors.New("connection reset")
	er := &errorReader{prefix: []byte("data: partial"), err: boom}
	r, err := sse.DecodeStreamFrames(er)
	if err != nil {
		t.Fatalf("DecodeStreamFrames() error = %v", err)
	}
	defer r.Close()
	_, termErr := collect(r)
	var fe *sse.FramerError
	if !errors.As(termErr, &fe) {
		t.Fatalf("terminal error = %T (%v), want *sse.FramerError", termErr, termErr)
	}
	if !errors.Is(termErr, boom) {
		t.Errorf("FramerError does not unwrap to the read error: %v", termErr)
	}
}

func TestDecodeStreamFrames_NilBody(t *testing.T) {
	t.Parallel()
	_, err := sse.DecodeStreamFrames(nil)
	var fe *sse.FramerError
	if !errors.As(err, &fe) {
		t.Fatalf("err = %T (%v), want *sse.FramerError", err, err)
	}
}

// TestFramer confirms the package satisfies codec.StreamFramer as an injectable value.
func TestFramer(t *testing.T) {
	t.Parallel()
	var f codec.StreamFramer = sse.Framer()
	r, err := f.DecodeStreamFrames(io.NopCloser(strings.NewReader("data: ok\n\n")))
	if err != nil {
		t.Fatalf("DecodeStreamFrames() error = %v", err)
	}
	defer r.Close()
	frames, termErr := collect(r)
	if !errors.Is(termErr, io.EOF) {
		t.Fatalf("terminal error = %v, want io.EOF", termErr)
	}
	if len(frames) != 1 || string(frames[0].Data) != "ok" {
		t.Fatalf("frames = %v, want one frame with Data=ok", frames)
	}
}

func FuzzDecodeStreamFrames(f *testing.F) {
	f.Add("data: {\"a\":1}\n\ndata: [DONE]\n\n")
	f.Add(": comment\n\nevent: x\ndata: y\n\n")
	f.Add("data: a\ndata: b\n\n")
	f.Add("")
	f.Add("data: no-terminator")
	f.Fuzz(func(t *testing.T, body string) {
		r, err := sse.DecodeStreamFrames(io.NopCloser(strings.NewReader(body)))
		if err != nil {
			return
		}
		defer r.Close()
		for {
			if _, err := r.Next(); err != nil {
				return
			}
		}
	})
}

// TestDecodeStreamFrames_BareCR verifies the WHATWG line-splitting rule that a bare CR
// (no following LF) terminates a line. bufio.ScanLines does not split on a lone CR, which
// would merge the whole body into one unrecognized line and silently drop every frame.
func TestDecodeStreamFrames_BareCR(t *testing.T) {
	t.Parallel()
	// Two CR-terminated data lines then a CR-terminated blank line: the two data fields
	// join into one event dispatched at the blank line.
	r, err := sse.DecodeStreamFrames(io.NopCloser(strings.NewReader("data: a\rdata: b\r\r")))
	if err != nil {
		t.Fatalf("DecodeStreamFrames: %v", err)
	}
	defer r.Close()
	f, err := r.Next()
	if err != nil {
		t.Fatalf("Next: %v", err)
	}
	if got := string(f.Data); got != "a\nb" {
		t.Errorf("frame data = %q, want %q", got, "a\nb")
	}
}
