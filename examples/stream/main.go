package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/stream"
)

type client struct{}

func (client) Invoke(context.Context, inference.Request) (*inference.Response, error) {
	return nil, errors.New("invoke is not used in this example")
}

func (client) Stream(context.Context, inference.Request) (*stream.StreamReader[content.Chunk], error) {
	chunks := []content.Chunk{
		&content.TextChunk{Text: "hello "},
		&content.TextChunk{Text: "stream"},
	}
	index := 0
	return stream.NewStreamReaderWithResult(
		func() (content.Chunk, error) {
			if index == len(chunks) {
				return nil, io.EOF
			}
			chunk := chunks[index]
			index++
			return chunk, nil
		},
		nil,
		func() (stream.StreamResult, bool, error) {
			return stream.StreamResult{Model: "demo-model", FinishReason: stream.FinishReasonStop}, true, nil
		},
	), nil
}

func main() {
	var provider inference.Client = client{}
	reader, err := provider.Stream(context.Background(), inference.Request{})
	if err != nil {
		panic(err)
	}
	defer reader.Close()

	var text strings.Builder
	for {
		chunk, err := reader.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			panic(err)
		}
		text.WriteString(chunk.(*content.TextChunk).Text)
	}
	result, ok := reader.Result()
	if !ok {
		panic("stream completed without terminal metadata")
	}
	fmt.Printf("chunks: %s\nfinish: %s\n", text.String(), result.FinishReason)
}
