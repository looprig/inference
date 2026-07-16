package stream_test

import (
	"errors"
	"io"
	"testing"

	"github.com/looprig/core/content"
	stream "github.com/looprig/inference/stream"
)

// frameSource builds a StreamReader over a fixed set of frames, tracking Close.
func frameSource(frames []stream.StreamFrame, closed *bool) *stream.StreamReader[stream.StreamFrame] {
	i := 0
	return stream.NewStreamReader(func() (stream.StreamFrame, error) {
		if i >= len(frames) {
			return stream.StreamFrame{}, io.EOF
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
	mapFrame := func(f stream.StreamFrame) ([]content.Chunk, error) {
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
			frames := make([]stream.StreamFrame, 0, len(tc.datas))
			for _, d := range tc.datas {
				frames = append(frames, stream.StreamFrame{Data: []byte(d)})
			}
			closed := false
			r := stream.FramesToChunks(frameSource(frames, &closed), mapFrame)

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
	frames := frameSource([]stream.StreamFrame{{Data: []byte("bad")}}, &closed)
	r := stream.FramesToChunks(frames, func(stream.StreamFrame) ([]content.Chunk, error) {
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
		frames         []stream.StreamFrame
		frameResult    stream.StreamResult
		frameResultOK  bool
		adapterResult  stream.StreamResultProducer
		mapFrame       func(stream.StreamFrame) ([]content.Chunk, error)
		wantTexts      []string
		wantErr        error
		wantResult     stream.StreamResult
		wantResultOK   bool
		wantFrameReads int
	}{
		{
			name:          "natural frame EOF propagates source metadata",
			frames:        []stream.StreamFrame{{Data: []byte("one")}},
			frameResult:   stream.StreamResult{Model: "source-model", FinishReason: stream.FinishReasonStop},
			frameResultOK: true,
			mapFrame: func(frame stream.StreamFrame) ([]content.Chunk, error) {
				return []content.Chunk{&content.TextChunk{Text: string(frame.Data)}}, nil
			},
			wantTexts:      []string{"one"},
			wantResult:     stream.StreamResult{Model: "source-model", FinishReason: stream.FinishReasonStop},
			wantResultOK:   true,
			wantFrameReads: 2,
		},
		{
			name:           "no source or adapter metadata stays unavailable",
			frames:         nil,
			mapFrame:       func(stream.StreamFrame) ([]content.Chunk, error) { return nil, nil },
			wantFrameReads: 1,
		},
		{
			name:   "all pending chunks drain before natural EOF authorizes metadata",
			frames: []stream.StreamFrame{{Data: []byte("many")}},
			frameResult: stream.StreamResult{
				Usage:        &content.Usage{InputTokens: 4, OutputTokens: 2},
				FinishReason: stream.FinishReasonLength,
			},
			frameResultOK: true,
			mapFrame: func(stream.StreamFrame) ([]content.Chunk, error) {
				return []content.Chunk{
					&content.TextChunk{Text: "a"},
					&content.TextChunk{Text: "b"},
					&content.TextChunk{Text: "c"},
				}, nil
			},
			wantTexts: []string{"a", "b", "c"},
			wantResult: stream.StreamResult{
				Usage:        &content.Usage{InputTokens: 4, OutputTokens: 2},
				FinishReason: stream.FinishReasonLength,
			},
			wantResultOK:   true,
			wantFrameReads: 2,
		},
		{
			name:   "mapper EOF drains returned chunks then authorizes adapter metadata",
			frames: []stream.StreamFrame{{Data: []byte("terminal")}, {Data: []byte("unread")}},
			adapterResult: func() (stream.StreamResult, bool, error) {
				return stream.StreamResult{Model: "collector-model", FinishReason: stream.FinishReasonToolUse}, true, nil
			},
			mapFrame: func(stream.StreamFrame) ([]content.Chunk, error) {
				return []content.Chunk{&content.TextChunk{Text: "last-a"}, &content.TextChunk{Text: "last-b"}}, io.EOF
			},
			wantTexts:      []string{"last-a", "last-b"},
			wantResult:     stream.StreamResult{Model: "collector-model", FinishReason: stream.FinishReasonToolUse},
			wantResultOK:   true,
			wantFrameReads: 1,
		},
		{
			name:   "mapper non EOF error leaks neither chunks nor result",
			frames: []stream.StreamFrame{{Data: []byte("bad")}},
			adapterResult: func() (stream.StreamResult, bool, error) {
				return stream.StreamResult{Model: "must-not-appear"}, true, nil
			},
			mapFrame: func(stream.StreamFrame) ([]content.Chunk, error) {
				return []content.Chunk{&content.TextChunk{Text: "must-not-appear"}}, errMap
			},
			wantErr:        errMap,
			wantFrameReads: 1,
		},
		{
			name:          "absent adapter result falls back to source metadata",
			frames:        nil,
			frameResult:   stream.StreamResult{Model: "fallback"},
			frameResultOK: true,
			adapterResult: func() (stream.StreamResult, bool, error) {
				return stream.StreamResult{}, false, nil
			},
			mapFrame:       func(stream.StreamFrame) ([]content.Chunk, error) { return nil, nil },
			wantResult:     stream.StreamResult{Model: "fallback"},
			wantResultOK:   true,
			wantFrameReads: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			frameIndex := 0
			frameReads := 0
			frames := stream.NewStreamReaderWithResult(
				func() (stream.StreamFrame, error) {
					frameReads++
					if frameIndex >= len(tt.frames) {
						return stream.StreamFrame{}, io.EOF
					}
					frame := tt.frames[frameIndex]
					frameIndex++
					return frame, nil
				},
				nil,
				func() (stream.StreamResult, bool, error) {
					return tt.frameResult, tt.frameResultOK, nil
				},
			)
			reader := stream.FramesToChunksWithResult(frames, tt.mapFrame, tt.adapterResult)
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
		name          string
		frames        func(reads *int, producerCalls *int) *stream.StreamReader[stream.StreamFrame]
		mapFrame      func(stream.StreamFrame) ([]content.Chunk, error)
		wantOperation stream.StreamOperation
		wantFailure   stream.StreamReaderFailure
		wantReads     int
		wantProducers int
	}{
		{
			name:          "nil frame reader fails safely",
			frames:        func(*int, *int) *stream.StreamReader[stream.StreamFrame] { return nil },
			mapFrame:      func(stream.StreamFrame) ([]content.Chunk, error) { return nil, nil },
			wantOperation: stream.StreamOperationNext,
			wantFailure:   stream.StreamReaderFailureNilReceiver,
		},
		{
			name: "nil mapper fails before reading a nonempty upstream",
			frames: func(reads *int, _ *int) *stream.StreamReader[stream.StreamFrame] {
				return stream.NewStreamReader(func() (stream.StreamFrame, error) {
					(*reads)++
					return stream.StreamFrame{}, nil
				}, nil)
			},
			wantOperation: stream.StreamOperationNext,
			wantFailure:   stream.StreamReaderFailureMissingFrameMapper,
		},
		{
			name: "nil mapper cannot authorize metadata from an empty upstream",
			frames: func(reads *int, producerCalls *int) *stream.StreamReader[stream.StreamFrame] {
				return stream.NewStreamReaderWithResult(
					func() (stream.StreamFrame, error) { (*reads)++; return stream.StreamFrame{}, io.EOF },
					nil,
					func() (stream.StreamResult, bool, error) {
						(*producerCalls)++
						return stream.StreamResult{Model: "must-not-authorize"}, true, nil
					},
				)
			},
			wantOperation: stream.StreamOperationNext,
			wantFailure:   stream.StreamReaderFailureMissingFrameMapper,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			reads := 0
			producerCalls := 0
			reader := stream.FramesToChunks(tt.frames(&reads, &producerCalls), tt.mapFrame)
			_, err := reader.Next()
			var streamErr *stream.StreamReaderError
			if !errors.As(err, &streamErr) {
				t.Fatalf("Next() error = %T %v, want *stream.StreamReaderError", err, err)
			}
			if streamErr.Operation != tt.wantOperation || streamErr.Failure != tt.wantFailure {
				t.Errorf("error = operation %q failure %q, want %q/%q", streamErr.Operation, streamErr.Failure, tt.wantOperation, tt.wantFailure)
			}
			if reads != tt.wantReads || producerCalls != tt.wantProducers {
				t.Errorf("upstream reads/producers = %d/%d, want %d/%d", reads, producerCalls, tt.wantReads, tt.wantProducers)
			}
			if _, ok := reader.Result(); ok {
				t.Error("invalid adapter unexpectedly authorized a result")
			}
		})
	}
}

func TestFramesToChunks_ResultPrecedence(t *testing.T) {
	t.Parallel()

	errSource := errors.New("source trailer failed")
	errSemantic := errors.New("semantic trailer failed")
	tests := []struct {
		name             string
		sourceError      error
		semanticProducer stream.StreamResultProducer
		wantResult       stream.StreamResult
		wantOK           bool
		wantCause        error
	}{
		{
			name:        "valid semantic result overrides failing source trailer",
			sourceError: errSource,
			semanticProducer: func() (stream.StreamResult, bool, error) {
				return stream.StreamResult{Model: "semantic", FinishReason: stream.FinishReasonStop}, true, nil
			},
			wantResult: stream.StreamResult{Model: "semantic", FinishReason: stream.FinishReasonStop},
			wantOK:     true,
		},
		{
			name:        "semantic error remains authoritative over source trailer error",
			sourceError: errSource,
			semanticProducer: func() (stream.StreamResult, bool, error) {
				return stream.StreamResult{}, false, errSemantic
			},
			wantCause: errSemantic,
		},
		{
			name:        "absent semantic result preserves source trailer error",
			sourceError: errSource,
			semanticProducer: func() (stream.StreamResult, bool, error) {
				return stream.StreamResult{}, false, nil
			},
			wantCause: errSource,
		},
		{
			name: "semantic error fails after source clean EOF",
			semanticProducer: func() (stream.StreamResult, bool, error) {
				return stream.StreamResult{}, false, errSemantic
			},
			wantCause: errSemantic,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			sourceCalls := 0
			semanticCalls := 0
			frames := stream.NewStreamReaderWithResult(
				func() (stream.StreamFrame, error) { return stream.StreamFrame{}, io.EOF },
				nil,
				func() (stream.StreamResult, bool, error) {
					sourceCalls++
					return stream.StreamResult{}, false, tt.sourceError
				},
			)
			semantic := func() (stream.StreamResult, bool, error) {
				semanticCalls++
				return tt.semanticProducer()
			}
			reader := stream.FramesToChunksWithResult(
				frames,
				func(stream.StreamFrame) ([]content.Chunk, error) {
					t.Fatal("mapper called after empty upstream")
					return nil, nil
				},
				semantic,
			)

			_, err := reader.Next()
			if tt.wantOK {
				if !errors.Is(err, io.EOF) {
					t.Fatalf("Next() error = %v, want clean io.EOF", err)
				}
				got, ok := reader.Result()
				if !ok || got != tt.wantResult {
					t.Fatalf("Result() = %+v, %v; want %+v, true", got, ok, tt.wantResult)
				}
			} else {
				if errors.Is(err, io.EOF) {
					t.Fatalf("Next() error = %v matches io.EOF; want terminal metadata failure", err)
				}
				var resultErr *stream.StreamResultError
				if !errors.As(err, &resultErr) {
					t.Fatalf("Next() error = %T %v, want *StreamResultError", err, err)
				}
				if resultErr.Cause != tt.wantCause {
					t.Errorf("result cause = %v, want exact %v", resultErr.Cause, tt.wantCause)
				}
				if _, ok := reader.Result(); ok {
					t.Error("terminal metadata failure unexpectedly authorized a result")
				}
			}
			if sourceCalls != 1 || semanticCalls != 1 {
				t.Errorf("source/semantic producer calls = %d/%d, want 1/1", sourceCalls, semanticCalls)
			}
		})
	}
}
