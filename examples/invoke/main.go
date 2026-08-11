package main

import (
	"context"
	"errors"
	"fmt"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/model"
	"github.com/looprig/inference/stream"
)

type client struct{}

func (client) Invoke(_ context.Context, req inference.Request) (*inference.Response, error) {
	return &inference.Response{
		Message: &content.AIMessage{Message: content.Message{
			Role:   content.RoleAssistant,
			Blocks: []content.Block{&content.TextBlock{Text: "hello from inference"}},
		}},
		Model:        req.Model.Name,
		FinishReason: stream.FinishReasonStop,
	}, nil
}

func (client) Stream(context.Context, inference.Request) (*stream.StreamReader[content.Chunk], error) {
	return nil, errors.New("streaming is not used in this example")
}

func main() {
	modelConfig := model.CustomModel("example", model.APIFormatAnthropic, "", "demo-model")
	request := inference.Request{
		Model:  modelConfig,
		System: "Answer briefly.",
		Messages: content.AgenticMessages{&content.UserMessage{Message: content.Message{
			Role:   content.RoleUser,
			Blocks: []content.Block{&content.TextBlock{Text: "Say hello."}},
		}}},
	}

	var provider inference.Client = client{}
	response, err := provider.Invoke(context.Background(), request)
	if err != nil {
		panic(err)
	}
	text := response.Message.Blocks[0].(*content.TextBlock)
	fmt.Printf("model: %s\nreply: %s\n", response.Model, text.Text)
}
