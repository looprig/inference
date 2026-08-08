package retry

import (
	"context"
	"errors"
	"io"
	"reflect"
	"testing"
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/failure"
	"github.com/looprig/inference/stream"
)

func scriptedStream(chunks []content.Chunk, result stream.StreamResult) *stream.StreamReader[content.Chunk] {
	index := 0
	next := func() (content.Chunk, error) {
		if index >= len(chunks) {
			return nil, io.EOF
		}
		chunk := chunks[index]
		index++
		return chunk, nil
	}
	return stream.NewStreamReaderWithResult(next, nil, func() (stream.StreamResult, bool, error) {
		return result, true, nil
	})
}

func scriptedStreamWithoutResult(chunks []content.Chunk) *stream.StreamReader[content.Chunk] {
	index := 0
	return stream.NewStreamReader(func() (content.Chunk, error) {
		if index >= len(chunks) {
			return nil, io.EOF
		}
		chunk := chunks[index]
		index++
		return chunk, nil
	}, nil)
}

func drainChunks(t *testing.T, reader *stream.StreamReader[content.Chunk]) []content.Chunk {
	t.Helper()
	var chunks []content.Chunk
	for {
		chunk, err := reader.Next()
		if errors.Is(err, io.EOF) {
			return chunks
		}
		if err != nil {
			t.Fatalf("Next() error = %v", err)
		}
		chunks = append(chunks, chunk)
	}
}

func TestStream_EstablishmentRetries(t *testing.T) {
	wantChunks := []content.Chunk{&content.TextChunk{Text: "one"}, &content.TextChunk{Text: "two"}}
	inner := &scriptedClient{streamOutcomes: []outcome{
		{err: &failure.APIError{Status: 429}},
		{err: &failure.APIError{Status: 429}},
		{reader: scriptedStream(wantChunks, stream.StreamResult{Model: "model"})},
	}}
	c, delays := newTestClient(t, inner)

	reader, err := c.Stream(context.Background(), inference.Request{})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	gotChunks := drainChunks(t, reader)
	if !reflect.DeepEqual(gotChunks, wantChunks) {
		t.Fatalf("chunks = %#v, want %#v", gotChunks, wantChunks)
	}
	result, ok := reader.Result()
	if !ok {
		t.Fatal("Result() reported no terminal metadata")
	}
	if result.Attempts != 3 {
		t.Fatalf("Attempts = %d, want 3", result.Attempts)
	}
	calls, _ := inner.callCounts()
	if calls != 0 {
		t.Fatalf("Invoke calls = %d, want 0", calls)
	}
	_, streamCalls := inner.callCounts()
	if streamCalls != 3 {
		t.Fatalf("Stream calls = %d, want 3", streamCalls)
	}
	if want := []time.Duration{2 * time.Second, 2 * time.Second}; !reflect.DeepEqual(*delays, want) {
		t.Fatalf("delays = %v, want %v", *delays, want)
	}
}

func TestStream_FirstTryResultAttempts(t *testing.T) {
	wantChunks := []content.Chunk{&content.TextChunk{Text: "one"}}
	wantUsage := &content.Usage{InputTokens: 2, OutputTokens: 3}
	inner := &scriptedClient{streamOutcomes: []outcome{
		{reader: scriptedStream(wantChunks, stream.StreamResult{Usage: wantUsage, Model: "model", FinishReason: stream.FinishReasonStop})},
	}}
	c, _ := newTestClient(t, inner)

	reader, err := c.Stream(context.Background(), inference.Request{})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	gotChunks := drainChunks(t, reader)
	if !reflect.DeepEqual(gotChunks, wantChunks) {
		t.Fatalf("chunks = %#v, want %#v", gotChunks, wantChunks)
	}
	result, ok := reader.Result()
	if !ok {
		t.Fatal("Result() reported no terminal metadata")
	}
	wantResult := stream.StreamResult{Usage: wantUsage, Model: "model", FinishReason: stream.FinishReasonStop, Attempts: 1}
	if !reflect.DeepEqual(result, wantResult) {
		t.Fatalf("result = %#v, want %#v", result, wantResult)
	}
}

func TestStream_InnerResultAbsent(t *testing.T) {
	inner := &scriptedClient{streamOutcomes: []outcome{
		{reader: scriptedStreamWithoutResult([]content.Chunk{&content.TextChunk{Text: "one"}})},
	}}
	c, _ := newTestClient(t, inner)

	reader, err := c.Stream(context.Background(), inference.Request{})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	_ = drainChunks(t, reader)
	if _, ok := reader.Result(); ok {
		t.Fatal("Result() reported metadata when the inner reader had none")
	}
}

func TestStream_MidStreamErrorNotRetried(t *testing.T) {
	streamErr := &failure.APIError{Status: 500}
	sent := false
	innerReader := stream.NewStreamReader(func() (content.Chunk, error) {
		if !sent {
			sent = true
			return &content.TextChunk{Text: "one"}, nil
		}
		return nil, streamErr
	}, nil)
	inner := &scriptedClient{streamOutcomes: []outcome{{reader: innerReader}}}
	c, _ := newTestClient(t, inner)

	reader, err := c.Stream(context.Background(), inference.Request{})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	if _, err := reader.Next(); err != nil {
		t.Fatalf("first Next() error = %v", err)
	}
	if _, err := reader.Next(); err != streamErr {
		t.Fatalf("second Next() error = %v, want original %v", err, streamErr)
	}
	_, streamCalls := inner.callCounts()
	if streamCalls != 1 {
		t.Fatalf("Stream calls = %d, want 1", streamCalls)
	}
}

func TestStream_NonRetryableFailsFast(t *testing.T) {
	wantErr := &failure.APIError{Status: 400}
	inner := &scriptedClient{streamOutcomes: []outcome{{err: wantErr}}}
	c, delays := newTestClient(t, inner)

	_, err := c.Stream(context.Background(), inference.Request{})
	if err != wantErr {
		t.Fatalf("error = %v (%T), want original %v", err, err, wantErr)
	}
	_, streamCalls := inner.callCounts()
	if streamCalls != 1 {
		t.Fatalf("Stream calls = %d, want 1", streamCalls)
	}
	if len(*delays) != 0 {
		t.Fatalf("delays = %v, want none", *delays)
	}
}

func TestStream_Exhaustion(t *testing.T) {
	lastErr := &failure.APIError{Status: 503, Message: "last"}
	inner := &scriptedClient{streamOutcomes: []outcome{
		{err: &failure.APIError{Status: 503}},
		{err: &failure.APIError{Status: 503}},
		{err: &failure.APIError{Status: 503}},
		{err: &failure.APIError{Status: 503}},
		{err: &failure.APIError{Status: 503}},
		{err: lastErr},
	}}
	c, _ := newTestClient(t, inner)

	_, err := c.Stream(context.Background(), inference.Request{})
	var exhausted *ExhaustedError
	if !errors.As(err, &exhausted) {
		t.Fatalf("error = %T, want *ExhaustedError", err)
	}
	if exhausted.Attempts != 6 {
		t.Fatalf("Attempts = %d, want 6", exhausted.Attempts)
	}
	var apiErr *failure.APIError
	if !errors.As(err, &apiErr) || apiErr != lastErr {
		t.Fatalf("error chain does not expose final API error: %v", err)
	}
	_, streamCalls := inner.callCounts()
	if streamCalls != 6 {
		t.Fatalf("Stream calls = %d, want 6", streamCalls)
	}
}

func TestStream_ClosesReaderReturnedWithError(t *testing.T) {
	closed := 0
	readerWithError := stream.NewStreamReader(func() (content.Chunk, error) {
		return nil, io.EOF
	}, func() error {
		closed++
		return nil
	})
	inner := &scriptedClient{streamOutcomes: []outcome{
		{reader: readerWithError, err: &failure.APIError{Status: 429}},
		{reader: scriptedStream(nil, stream.StreamResult{Model: "model"})},
	}}
	c, _ := newTestClient(t, inner)

	reader, err := c.Stream(context.Background(), inference.Request{})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	if err := reader.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if closed != 1 {
		t.Fatalf("errored reader close count = %d, want 1", closed)
	}
}
