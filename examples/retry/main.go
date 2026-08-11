package main

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/failure"
	"github.com/looprig/inference/retry"
	"github.com/looprig/inference/stream"
)

type client struct {
	mu       sync.Mutex
	attempts int
}

func (c *client) Invoke(context.Context, inference.Request) (*inference.Response, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.attempts++
	if c.attempts < 3 {
		return nil, &failure.APIError{Status: 429}
	}
	return &inference.Response{Message: &content.AIMessage{Message: content.Message{
		Role:   content.RoleAssistant,
		Blocks: []content.Block{&content.TextBlock{Text: "recovered"}},
	}}}, nil
}

func (*client) Stream(context.Context, inference.Request) (*stream.StreamReader[content.Chunk], error) {
	return nil, errors.New("streaming is not used in this example")
}

func main() {
	retrying, err := retry.New(&client{}, retry.Policy{
		StableRetries: 2,
		StableDelay:   time.Nanosecond,
		MaxAttempts:   3,
		MaxDelay:      time.Nanosecond,
	})
	if err != nil {
		panic(err)
	}
	response, err := retrying.Invoke(context.Background(), inference.Request{})
	if err != nil {
		panic(err)
	}
	text := response.Message.Blocks[0].(*content.TextBlock)
	fmt.Printf("attempts: %d\nreply: %s\n", response.Attempts, text.Text)
}
