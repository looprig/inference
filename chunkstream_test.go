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

func TestFramesToChunks_Result(t *testing.T) {
	t.Parallel()

	errMap := errors.New("map failed")
	tests := []struct {
		name           string
		frames         []inference.StreamFrame
		frameResult    inference.StreamResult
		frameResultOK  bool
		adapterResult  inference.StreamResultProducer
		mapFrame       func(inference.StreamFrame) ([]content.Chunk, error)
		wantTexts      []string
		wantErr        error
		wantResult     inference.StreamResult
		wantResultOK   bool
		wantFrameReads int
	}{
		{
			name:          "natural frame EOF propagates source metadata",
			frames:        []inference.StreamFrame{{Data: []byte("one")}},
			frameResult:   inference.StreamResult{Model: "source-model", FinishReason: inference.FinishReasonStop},
			frameResultOK: true,
			mapFrame: func(frame inference.StreamFrame) ([]content.Chunk, error) {
				return []content.Chunk{&content.TextChunk{Text: string(frame.Data)}}, nil
			},
			wantTexts:      []string{"one"},
			wantResult:     inference.StreamResult{Model: "source-model", FinishReason: inference.FinishReasonStop},
			wantResultOK:   true,
			wantFrameReads: 2,
		},
		{
			name:           "no source or adapter metadata stays unavailable",
			frames:         nil,
			mapFrame:       func(inference.StreamFrame) ([]content.Chunk, error) { return nil, nil },
			wantFrameReads: 1,
		},
		{
			name:   "all pending chunks drain before natural EOF authorizes metadata",
			frames: []inference.StreamFrame{{Data: []byte("many")}},
			frameResult: inference.StreamResult{
				Usage:        &content.Usage{InputTokens: 4, OutputTokens: 2},
				FinishReason: inference.FinishReasonLength,
			},
			frameResultOK: true,
			mapFrame: func(inference.StreamFrame) ([]content.Chunk, error) {
				return []content.Chunk{
					&content.TextChunk{Text: "a"},
					&content.TextChunk{Text: "b"},
					&content.TextChunk{Text: "c"},
				}, nil
			},
			wantTexts: []string{"a", "b", "c"},
			wantResult: inference.StreamResult{
				Usage:        &content.Usage{InputTokens: 4, OutputTokens: 2},
				FinishReason: inference.FinishReasonLength,
			},
			wantResultOK:   true,
			wantFrameReads: 2,
		},
		{
			name:   "mapper EOF drains returned chunks then authorizes adapter metadata",
			frames: []inference.StreamFrame{{Data: []byte("terminal")}, {Data: []byte("unread")}},
			adapterResult: func() (inference.StreamResult, bool, error) {
				return inference.StreamResult{Model: "collector-model", FinishReason: inference.FinishReasonToolUse}, true, nil
			},
			mapFrame: func(inference.StreamFrame) ([]content.Chunk, error) {
				return []content.Chunk{&content.TextChunk{Text: "last-a"}, &content.TextChunk{Text: "last-b"}}, io.EOF
			},
			wantTexts:      []string{"last-a", "last-b"},
			wantResult:     inference.StreamResult{Model: "collector-model", FinishReason: inference.FinishReasonToolUse},
			wantResultOK:   true,
			wantFrameReads: 1,
		},
		{
			name:   "mapper non EOF error leaks neither chunks nor result",
			frames: []inference.StreamFrame{{Data: []byte("bad")}},
			adapterResult: func() (inference.StreamResult, bool, error) {
				return inference.StreamResult{Model: "must-not-appear"}, true, nil
			},
			mapFrame: func(inference.StreamFrame) ([]content.Chunk, error) {
				return []content.Chunk{&content.TextChunk{Text: "must-not-appear"}}, errMap
			},
			wantErr:        errMap,
			wantFrameReads: 1,
		},
		{
			name:          "absent adapter result falls back to source metadata",
			frames:        nil,
			frameResult:   inference.StreamResult{Model: "fallback"},
			frameResultOK: true,
			adapterResult: func() (inference.StreamResult, bool, error) {
				return inference.StreamResult{}, false, nil
			},
			mapFrame:       func(inference.StreamFrame) ([]content.Chunk, error) { return nil, nil },
			wantResult:     inference.StreamResult{Model: "fallback"},
			wantResultOK:   true,
			wantFrameReads: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			frameIndex := 0
			frameReads := 0
			frames := inference.NewStreamReaderWithResult(
				func() (inference.StreamFrame, error) {
					frameReads++
					if frameIndex >= len(tt.frames) {
						return inference.StreamFrame{}, io.EOF
					}
					frame := tt.frames[frameIndex]
					frameIndex++
					return frame, nil
				},
				nil,
				func() (inference.StreamResult, bool, error) {
					return tt.frameResult, tt.frameResultOK, nil
				},
			)
			reader := inference.FramesToChunksWithResult(frames, tt.mapFrame, tt.adapterResult)
			if _, ok := reader.Result(); ok {
				t.Fatal("Result() available before downstream EOF")
			}

			var gotTexts []string
			var finalErr error
			for {
				chunk, err := reader.Next()
				if err != nil {
					finalErr = err
					break
				}
				if _, ok := reader.Result(); ok {
					t.Fatal("Result() available while chunks remain")
				}
				text, ok := chunk.(*content.TextChunk)
				if !ok {
					t.Fatalf("chunk type = %T, want *content.TextChunk", chunk)
				}
				gotTexts = append(gotTexts, text.Text)
			}
			if tt.wantErr == nil {
				if !errors.Is(finalErr, io.EOF) {
					t.Fatalf("final error = %v, want io.EOF", finalErr)
				}
			} else if !errors.Is(finalErr, tt.wantErr) {
				t.Fatalf("final error = %v, want %v", finalErr, tt.wantErr)
			}
			if len(gotTexts) != len(tt.wantTexts) {
				t.Fatalf("texts = %v, want %v", gotTexts, tt.wantTexts)
			}
			for i := range tt.wantTexts {
				if gotTexts[i] != tt.wantTexts[i] {
					t.Errorf("text[%d] = %q, want %q", i, gotTexts[i], tt.wantTexts[i])
				}
			}
			gotResult, gotOK := reader.Result()
			if gotOK != tt.wantResultOK {
				t.Fatalf("Result() ok = %v, want %v (result %+v)", gotOK, tt.wantResultOK, gotResult)
			}
			if gotOK {
				if gotResult.Model != tt.wantResult.Model || gotResult.FinishReason != tt.wantResult.FinishReason {
					t.Errorf("Result() = %+v, want %+v", gotResult, tt.wantResult)
				}
				if (gotResult.Usage == nil) != (tt.wantResult.Usage == nil) {
					t.Fatalf("Result().Usage = %+v, want %+v", gotResult.Usage, tt.wantResult.Usage)
				}
				if gotResult.Usage != nil && *gotResult.Usage != *tt.wantResult.Usage {
					t.Errorf("Result().Usage = %+v, want %+v", gotResult.Usage, tt.wantResult.Usage)
				}
			}
			if frameReads != tt.wantFrameReads {
				t.Errorf("frame reads = %d, want %d", frameReads, tt.wantFrameReads)
			}
		})
	}
}

func TestFramesToChunks_InvalidBoundary(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		frames      *inference.StreamReader[inference.StreamFrame]
		mapFrame    func(inference.StreamFrame) ([]content.Chunk, error)
		wantFailure inference.StreamReaderFailure
	}{
		{
			name:        "nil frame reader fails safely",
			frames:      nil,
			mapFrame:    func(inference.StreamFrame) ([]content.Chunk, error) { return nil, nil },
			wantFailure: inference.StreamReaderFailureNilReceiver,
		},
		{
			name:        "nil mapper fails safely",
			frames:      inference.NewStreamReader(func() (inference.StreamFrame, error) { return inference.StreamFrame{}, nil }, nil),
			mapFrame:    nil,
			wantFailure: inference.StreamReaderFailureMissingFrameMapper,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			reader := inference.FramesToChunks(tt.frames, tt.mapFrame)
			_, err := reader.Next()
			var streamErr *inference.StreamReaderError
			if !errors.As(err, &streamErr) {
				t.Fatalf("Next() error = %T %v, want *inference.StreamReaderError", err, err)
			}
			if streamErr.Failure != tt.wantFailure {
				t.Errorf("failure = %q, want %q", streamErr.Failure, tt.wantFailure)
			}
			if _, ok := reader.Result(); ok {
				t.Error("invalid adapter unexpectedly authorized a result")
			}
		})
	}
}
