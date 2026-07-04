package inference_test

import (
	"errors"
	"io"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
)

// frameSource builds a StreamReader over a fixed set of frames, tracking Close.
func frameSource(frames []inference.StreamFrame, closed *bool) *inference.StreamReader[inference.StreamFrame] {
	i := 0
	return inference.NewStreamReader(func() (inference.StreamFrame, error) {
		if i >= len(frames) {
			return inference.StreamFrame{}, io.EOF
		}
		f := frames[i]
		i++
		return f, nil
	}, func() error {
		*closed = true
		return nil
	})
}

func TestFramesToChunks(t *testing.T) {
	t.Parallel()

	// mapFrame: "[DONE]" is terminal (io.EOF), "skip" yields nothing, "a,b" yields two
	// text chunks, anything else yields one text chunk with the frame's data as text.
	mapFrame := func(f inference.StreamFrame) ([]content.Chunk, error) {
		switch string(f.Data) {
		case "[DONE]":
			return nil, io.EOF
		case "skip":
			return nil, nil
		case "a,b":
			return []content.Chunk{&content.TextChunk{Text: "a"}, &content.TextChunk{Text: "b"}}, nil
		default:
			return []content.Chunk{&content.TextChunk{Text: string(f.Data)}}, nil
		}
	}

	cases := []struct {
		name  string
		datas []string
		want  []string
	}{
		{name: "one frame one chunk", datas: []string{"hi"}, want: []string{"hi"}},
		{name: "terminal sentinel ends stream", datas: []string{"x", "[DONE]", "y"}, want: []string{"x"}},
		{name: "skips are transparent", datas: []string{"skip", "z", "skip"}, want: []string{"z"}},
		{name: "multi-chunk frame drains all", datas: []string{"a,b", "c"}, want: []string{"a", "b", "c"}},
		{name: "empty", datas: nil, want: nil},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			frames := make([]inference.StreamFrame, 0, len(tc.datas))
			for _, d := range tc.datas {
				frames = append(frames, inference.StreamFrame{Data: []byte(d)})
			}
			closed := false
			r := inference.FramesToChunks(frameSource(frames, &closed), mapFrame)

			var got []string
			for {
				c, err := r.Next()
				if errors.Is(err, io.EOF) {
					break
				}
				if err != nil {
					t.Fatalf("Next() error = %v", err)
				}
				tx, ok := c.(*content.TextChunk)
				if !ok {
					t.Fatalf("chunk type = %T, want *content.TextChunk", c)
				}
				got = append(got, tx.Text)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("chunks = %v, want %v", got, tc.want)
			}
			for i := range tc.want {
				if got[i] != tc.want[i] {
					t.Errorf("chunk[%d] = %q, want %q", i, got[i], tc.want[i])
				}
			}
			if err := r.Close(); err != nil {
				t.Fatalf("Close() error = %v", err)
			}
			if !closed {
				t.Error("expected underlying frame reader closed via Close()")
			}
		})
	}
}

// TestFramesToChunks_MapError surfaces a non-EOF mapper error to the caller.
func TestFramesToChunks_MapError(t *testing.T) {
	t.Parallel()
	boom := errors.New("decode boom")
	closed := false
	frames := frameSource([]inference.StreamFrame{{Data: []byte("bad")}}, &closed)
	r := inference.FramesToChunks(frames, func(inference.StreamFrame) ([]content.Chunk, error) {
		return nil, boom
	})
	_, err := r.Next()
	if !errors.Is(err, boom) {
		t.Fatalf("Next() error = %v, want %v", err, boom)
	}
}
