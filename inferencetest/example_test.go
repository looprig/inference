package inferencetest_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/looprig/core/content"
	"github.com/looprig/core/content/streamaccumulator"
	"github.com/looprig/inference"
	"github.com/looprig/inference/failure"
	"github.com/looprig/inference/inferencetest"
)

func userMessage(text string) *content.UserMessage {
	return &content.UserMessage{Message: content.Message{
		Role:   content.RoleUser,
		Blocks: []content.Block{&content.TextBlock{Text: text}},
	}}
}

// A text reply, streamed word by word.
func ExampleText() {
	client := inferencetest.New(inferencetest.Text("Hello from a scripted model."))

	reader, err := client.Stream(context.Background(), inference.Request{
		Model:    inferencetest.Model(),
		Messages: content.AgenticMessages{userMessage("Say hello.")},
	})
	if err != nil {
		panic(err)
	}
	defer reader.Close()
	for {
		chunk, err := reader.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			panic(err)
		}
		fmt.Printf("%q\n", chunk.(*content.TextChunk).Text)
	}
	result, _ := reader.Result()
	fmt.Println("finish:", result.FinishReason)
	// Output:
	// "Hello "
	// "from "
	// "a "
	// "scripted "
	// "model."
	// finish: stop
}

// A tool-call round trip driven by a minimal agent loop: stream a reply,
// assemble it, run any tool it calls, send the results back, and repeat until
// the model stops calling tools. The second step checks that the tool result
// reached the model.
func Example_toolRoundTrip() {
	client := inferencetest.New(
		inferencetest.Text("Let me add those.").ToolCall("add", `{"a": 2, "b": 3}`),
		inferencetest.Text("2 + 3 = 5.").Expect(func(req inference.Request) error {
			last, ok := req.Messages[len(req.Messages)-1].(*content.ToolResultMessage)
			if !ok || last.ToolUseID != "call_1_0" {
				return errors.New("the model expected the add result")
			}
			return nil
		}),
	)
	tools := map[string]func(json.RawMessage) string{
		"add": func(input json.RawMessage) string {
			var args struct{ A, B int }
			if err := json.Unmarshal(input, &args); err != nil {
				return err.Error()
			}
			return fmt.Sprint(args.A + args.B)
		},
	}

	req := inference.Request{
		Model:    inferencetest.Model(),
		Messages: content.AgenticMessages{userMessage("What is 2 + 3?")},
		Tools:    []inference.Tool{{Name: "add", Schema: json.RawMessage(`{"type":"object"}`)}},
	}
	for {
		reply, calls := streamTurn(client, req)
		req.Messages = append(req.Messages, reply)
		fmt.Println("model:", reply.Blocks[0].(*content.TextBlock).Text)
		if len(calls) == 0 {
			break
		}
		for _, call := range calls {
			result := tools[call.Name](call.Input)
			fmt.Printf("tool %s(%s) = %s\n", call.Name, call.Input, result)
			req.Messages = append(req.Messages, &content.ToolResultMessage{
				Message: content.Message{Role: content.RoleTool, Blocks: []content.Block{
					&content.ToolResultBlock{ToolUseID: call.ID, Content: []content.Block{&content.TextBlock{Text: result}}},
				}},
				ToolUseID: call.ID,
			})
		}
	}
	fmt.Println("script error:", client.Err())
	// Output:
	// model: Let me add those.
	// tool add({"a": 2, "b": 3}) = 5
	// model: 2 + 3 = 5.
	// script error: <nil>
}

// streamTurn streams one reply and assembles it the way an agent loop does.
func streamTurn(client inference.Client, req inference.Request) (*content.AIMessage, []content.ToolUseBlock) {
	reader, err := client.Stream(context.Background(), req)
	if err != nil {
		panic(err)
	}
	defer reader.Close()
	var text streamaccumulator.Text
	var calls streamaccumulator.ToolUses
	for {
		chunk, err := reader.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			panic(err)
		}
		switch c := chunk.(type) {
		case *content.TextChunk:
			text.Add(c)
		case *content.ToolUseChunk:
			calls.Add(c)
		}
	}
	msg := &content.AIMessage{Message: content.Message{Role: content.RoleAssistant}}
	if block := text.Block(); block != nil {
		msg.Blocks = append(msg.Blocks, block)
	}
	for _, call := range calls.Blocks() {
		msg.Blocks = append(msg.Blocks, &call)
	}
	return msg, calls.Blocks()
}

// A provider error, such as a rate limit, surfaces from the call itself.
func ExampleFail() {
	client := inferencetest.New(
		inferencetest.Fail(failure.NewAPIError(429, "rate_limit_exceeded", "", 0)),
		inferencetest.Text("Recovered."),
	)
	req := inference.Request{Model: inferencetest.Model(), Messages: content.AgenticMessages{userMessage("Hi")}}

	_, err := client.Invoke(context.Background(), req)
	var apiErr *failure.APIError
	fmt.Println("rate limited:", errors.As(err, &apiErr) && apiErr.Status == 429)

	resp, err := client.Invoke(context.Background(), req)
	if err != nil {
		panic(err)
	}
	fmt.Println("retry:", resp.Message.Blocks[0].(*content.TextBlock).Text)
	// Output:
	// rate limited: true
	// retry: Recovered.
}

// Echo with Repeat answers anything, forever: a keyless demo model.
func ExampleEcho() {
	client := inferencetest.New(inferencetest.Echo().Repeat())
	for _, question := range []string{"ping", "is anyone there?"} {
		resp, err := client.Invoke(context.Background(), inference.Request{
			Model:    inferencetest.Model(),
			Messages: content.AgenticMessages{userMessage(question)},
		})
		if err != nil {
			panic(err)
		}
		fmt.Println(resp.Message.Blocks[0].(*content.TextBlock).Text)
	}
	// Output:
	// ping
	// is anyone there?
}

// Requests records what the model saw, for assertions after the fact.
func ExampleClient_Requests() {
	client := inferencetest.New(inferencetest.Text("ok"))
	_, _ = client.Invoke(context.Background(), inference.Request{
		Model:    inferencetest.Model(),
		System:   "Be brief.",
		Messages: content.AgenticMessages{userMessage("Summarize the plan.")},
	})
	for _, req := range client.Requests() {
		fmt.Println(strings.Join([]string{req.System, inferencetest.LastUserText(req)}, " | "))
	}
	// Output:
	// Be brief. | Summarize the plan.
}
