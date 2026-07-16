package ndjson_test

import (
	"errors"
	"io"
	"strings"
	"testing"

	stream "github.com/looprig/inference/stream"
	"github.com/looprig/inference/wire/ndjson"
)

type closerSpy struct {
	io.Reader
	closed bool
}

func (c *closerSpy) Close() error {
	c.closed = true
	return nil
}

func collect(r *stream.StreamReader[stream.StreamFrame]) ([]string, error) {
	var out []string
	for {
		f, err := r.Next()
		if err != nil {
			return out, err
		}
		out = append(out, string(f.Data))
	}
}

func TestDecodeStreamFrames(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		body string
		want []string
	}{
		{
			name: "multi-line",
			body: "{\"a\":1}\n{\"b\":2}\n{\"c\":3}\n",
			want: []string{`{"a":1}`, `{"b":2}`, `{"c":3}`},
		},
		{
			name: "no trailing newline",
			body: "{\"a\":1}\n{\"b\":2}",
			want: []string{`{"a":1}`, `{"b":2}`},
		},
		{
			name: "trailing newline",
			body: "{\"only\":true}\n",
			want: []string{`{"only":true}`},
		},
		{
			name: "blank lines skipped",
			body: "{\"a\":1}\n\n\n{\"b\":2}\n",
			want: []string{`{"a":1}`, `{"b":2}`},
		},
		{
			name: "empty body",
			body: "",
			want: nil,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			r, err := ndjson.DecodeStreamFrames(io.NopCloser(strings.NewReader(tc.body)))
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
			for i := range tc.want {
				if got[i] != tc.want[i] {
					t.Errorf("frame[%d] = %q, want %q", i, got[i], tc.want[i])
				}
			}
		})
	}
}

func TestDecodeStreamFrames_ClosesBody(t *testing.T) {
	t.Parallel()
	spy := &closerSpy{Reader: strings.NewReader("{}\n")}
	r, err := ndjson.DecodeStreamFrames(spy)
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

func TestDecodeStreamFrames_NilBody(t *testing.T) {
	t.Parallel()
	_, err := ndjson.DecodeStreamFrames(nil)
	var fe *ndjson.FramerError
	if !errors.As(err, &fe) {
		t.Fatalf("err = %T (%v), want *ndjson.FramerError", err, err)
	}
}
