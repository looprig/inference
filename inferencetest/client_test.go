package inferencetest_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/core/content/streamaccumulator"
	"github.com/looprig/inference"
	"github.com/looprig/inference/inferencetest"
	"github.com/looprig/inference/model"
	"github.com/looprig/inference/stream"
)

func request(text string) inference.Request {
	return inference.Request{
		Model: inferencetest.Model(),
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{
				Role:   content.RoleUser,
				Blocks: []content.Block{&content.TextBlock{Text: text}},
			}},
		},
	}
}

// drain reads a stream to its end, returning every chunk, the terminal error
// (nil on clean EOF) and the reader's Result.
func drain(t *testing.T, sr *stream.StreamReader[content.Chunk]) ([]content.Chunk, *stream.StreamResult, error) {
	t.Helper()
	defer sr.Close()
	var chunks []content.Chunk
	for {
		chunk, err := sr.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return chunks, nil, err
		}
		chunks = append(chunks, chunk)
	}
	result, ok := sr.Result()
	if !ok {
		return chunks, nil, nil
	}
	return chunks, &result, nil
}

// fold assembles chunks the way an agent loop does: reasoning, text, then tool
// calls, each kind in ascending index order.
func fold(chunks []content.Chunk) []content.Block {
	var thinking streamaccumulator.Thinking
	var text streamaccumulator.Text
	var tools streamaccumulator.ToolUses
	for _, chunk := range chunks {
		switch c := chunk.(type) {
		case *content.ThinkingChunk:
			thinking.Add(c)
		case *content.TextChunk:
			text.Add(c)
		case *content.ToolUseChunk:
			tools.Add(c)
		}
	}
	blocks := []content.Block{}
	for _, b := range thinking.Blocks() {
		blocks = append(blocks, &b)
	}
	if b := text.Block(); b != nil {
		blocks = append(blocks, b)
	}
	for _, b := range tools.Blocks() {
		blocks = append(blocks, &b)
	}
	return blocks
}

func TestInvokeText(t *testing.T) {
	t.Parallel()
	client := inferencetest.New(inferencetest.Text("hello there"))
	resp, err := client.Invoke(context.Background(), request("hi"))
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	want := []content.Block{&content.TextBlock{Text: "hello there"}}
	if !reflect.DeepEqual(resp.Message.Blocks, want) {
		t.Fatalf("blocks = %#v, want %#v", resp.Message.Blocks, want)
	}
	if resp.Message.Role != content.RoleAssistant {
		t.Fatalf("role = %q", resp.Message.Role)
	}
	if resp.FinishReason != stream.FinishReasonStop {
		t.Fatalf("finish = %q", resp.FinishReason)
	}
	if resp.Model != inferencetest.ModelName {
		t.Fatalf("model = %q", resp.Model)
	}
	if resp.Usage == nil || resp.Usage.InputTokens == 0 || resp.Usage.OutputTokens == 0 {
		t.Fatalf("usage = %#v, want a nonzero estimate", resp.Usage)
	}
	if resp.Message.Usage == resp.Usage || !reflect.DeepEqual(resp.Message.Usage, resp.Usage) {
		t.Fatalf("message usage must equal but not alias response usage")
	}
}

func TestStreamMatchesInvoke(t *testing.T) {
	t.Parallel()
	steps := map[string]inferencetest.Step{
		"text":        inferencetest.Text("one two  three\nfour"),
		"tool":        inferencetest.ToolCall("read", `{"path": "a b.txt"}`),
		"mixed":       inferencetest.Thinking("first think").Thinking("then more").Text("Let me look.").ToolCall("read", `{"path":"x"}`).ToolCallWithID("fixed", "list", ""),
		"runes":       inferencetest.Text("héllo wörld").ToolCall("t", `{"k":"vvvvv"}`).ChunkSize(3),
		"whole":       inferencetest.Thinking("t").Text("single delta").ChunkSize(-1),
		"length":      inferencetest.Text("cut off").Finish(stream.FinishReasonLength),
		"usage":       inferencetest.Text("u").Usage(content.Usage{InputTokens: 7, OutputTokens: 3}),
		"empty":       {},
		"emptyThinks": inferencetest.Thinking("").Text("x"),
	}
	for name, step := range steps {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			client := inferencetest.New(step, step)
			resp, err := client.Invoke(context.Background(), request("go"))
			if err != nil {
				t.Fatalf("Invoke: %v", err)
			}
			sr, err := client.Stream(context.Background(), request("go"))
			if err != nil {
				t.Fatalf("Stream: %v", err)
			}
			chunks, result, err := drain(t, sr)
			if err != nil {
				t.Fatalf("stream: %v", err)
			}
			if result == nil {
				t.Fatal("stream has no Result")
			}
			// Tool IDs embed the call number; normalise the stream's to call 1.
			got := fold(chunks)
			for _, b := range got {
				if tu, ok := b.(*content.ToolUseBlock); ok && tu.ID != "fixed" {
					tu.ID = "call_1_" + tu.ID[len("call_2_"):]
				}
			}
			if len(got) == 0 && len(resp.Message.Blocks) == 0 {
				got = resp.Message.Blocks
			}
			if !reflect.DeepEqual(got, resp.Message.Blocks) {
				t.Fatalf("folded stream = %s\ninvoke = %s", dump(got), dump(resp.Message.Blocks))
			}
			if result.FinishReason != resp.FinishReason || result.Model != resp.Model || !reflect.DeepEqual(result.Usage, resp.Usage) {
				t.Fatalf("result = %+v, response finish=%q model=%q usage=%+v", result, resp.FinishReason, resp.Model, resp.Usage)
			}
		})
	}
}

func dump(blocks []content.Block) string {
	s := "["
	for _, b := range blocks {
		s += fmt.Sprintf("%#v ", b)
	}
	return s + "]"
}

func TestStreamChunkShapes(t *testing.T) {
	t.Parallel()
	client := inferencetest.New(
		inferencetest.Thinking("a b").Text("hello brave world").ToolCall("read", `{"p": 1}`),
		inferencetest.Text("abcdefg").ChunkSize(3),
		inferencetest.Text("keep it whole").ChunkSize(-1),
	)
	sr, err := client.Stream(context.Background(), request("x"))
	if err != nil {
		t.Fatal(err)
	}
	chunks, result, err := drain(t, sr)
	if err != nil {
		t.Fatal(err)
	}
	want := []content.Chunk{
		&content.ThinkingChunk{Index: 0, Thinking: "a "},
		&content.ThinkingChunk{Index: 0, Thinking: "b"},
		&content.TextChunk{Text: "hello "},
		&content.TextChunk{Text: "brave "},
		&content.TextChunk{Text: "world"},
		&content.ToolUseChunk{Index: 2, ID: "call_1_0", Name: "read"},
		&content.ToolUseChunk{Index: 2, InputJSON: `{"p": `},
		&content.ToolUseChunk{Index: 2, InputJSON: `1}`},
	}
	if !reflect.DeepEqual(chunks, want) {
		t.Fatalf("chunks:\n%#v\nwant:\n%#v", chunks, want)
	}
	if result.FinishReason != stream.FinishReasonToolUse {
		t.Fatalf("finish = %q, want tool_use", result.FinishReason)
	}

	for _, want := range [][]content.Chunk{
		{&content.TextChunk{Text: "abc"}, &content.TextChunk{Text: "def"}, &content.TextChunk{Text: "g"}},
		{&content.TextChunk{Text: "keep it whole"}},
	} {
		sr, err := client.Stream(context.Background(), request("x"))
		if err != nil {
			t.Fatal(err)
		}
		chunks, _, err := drain(t, sr)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(chunks, want) {
			t.Fatalf("chunks = %#v, want %#v", chunks, want)
		}
	}
}

func TestToolCallDefaults(t *testing.T) {
	t.Parallel()
	client := inferencetest.New(
		inferencetest.Text("x"),
		inferencetest.ToolCall("a", "").ToolCall("b", `{"q":1}`),
	)
	if _, err := client.Invoke(context.Background(), request("1")); err != nil {
		t.Fatal(err)
	}
	resp, err := client.Invoke(context.Background(), request("2"))
	if err != nil {
		t.Fatal(err)
	}
	want := []content.Block{
		content.NewToolUseBlock("call_2_0", "a", []byte(`{}`), nil, ""),
		content.NewToolUseBlock("call_2_1", "b", []byte(`{"q":1}`), nil, ""),
	}
	if !reflect.DeepEqual(resp.Message.Blocks, want) {
		t.Fatalf("blocks = %s", dump(resp.Message.Blocks))
	}
	if resp.FinishReason != stream.FinishReasonToolUse {
		t.Fatalf("finish = %q", resp.FinishReason)
	}
}

func TestFail(t *testing.T) {
	t.Parallel()
	boom := errors.New("boom")
	client := inferencetest.New(inferencetest.Fail(boom).Text("never"), inferencetest.Fail(boom))
	if resp, err := client.Invoke(context.Background(), request("x")); !errors.Is(err, boom) || resp != nil {
		t.Fatalf("Invoke = %v, %v; want nil, boom", resp, err)
	}
	if sr, err := client.Stream(context.Background(), request("x")); !errors.Is(err, boom) || sr != nil {
		t.Fatalf("Stream = %v, %v; want nil, boom", sr, err)
	}
	if client.Err() != nil {
		t.Fatalf("a scripted failure is not a script error: %v", client.Err())
	}
}

func TestStreamError(t *testing.T) {
	t.Parallel()
	cut := errors.New("connection reset")
	step := inferencetest.Text("partial answer").StreamError(cut)
	client := inferencetest.New(step, step)
	sr, err := client.Stream(context.Background(), request("x"))
	if err != nil {
		t.Fatal(err)
	}
	chunks, _, err := drain(t, sr)
	if !errors.Is(err, cut) {
		t.Fatalf("stream err = %v, want %v", err, cut)
	}
	if len(chunks) != 2 {
		t.Fatalf("got %d chunks before the error, want 2", len(chunks))
	}
	if _, ok := sr.Result(); ok {
		t.Fatal("a truncated stream must report no Result")
	}
	if _, err := client.Invoke(context.Background(), request("x")); !errors.Is(err, cut) {
		t.Fatalf("Invoke err = %v, want %v", err, cut)
	}
}

func TestExhausted(t *testing.T) {
	t.Parallel()
	client := inferencetest.New(inferencetest.Text("only"))
	if _, err := client.Invoke(context.Background(), request("1")); err != nil {
		t.Fatal(err)
	}
	_, err := client.Stream(context.Background(), request("2"))
	var exhausted *inferencetest.ExhaustedError
	if !errors.As(err, &exhausted) || exhausted.Call != 2 {
		t.Fatalf("err = %v, want ExhaustedError{Call: 2}", err)
	}
	if !errors.As(client.Err(), &exhausted) {
		t.Fatalf("Err() = %v", client.Err())
	}
	if n := len(client.Requests()); n != 2 {
		t.Fatalf("recorded %d requests, want 2", n)
	}
}

func TestExpect(t *testing.T) {
	t.Parallel()
	wrong := errors.New("wrong question")
	check := func(want string) func(inference.Request) error {
		return func(req inference.Request) error {
			if got := inferencetest.LastUserText(req); got != want {
				return fmt.Errorf("%w: %q", wrong, got)
			}
			return nil
		}
	}
	client := inferencetest.New(
		inferencetest.Text("yes").Expect(check("first")),
		inferencetest.Text("no").Expect(check("second")),
		inferencetest.Text("after"),
	)
	if resp, err := client.Invoke(context.Background(), request("first")); err != nil || resp.Message.Blocks[0].(*content.TextBlock).Text != "yes" {
		t.Fatalf("Invoke = %v, %v", resp, err)
	}
	_, err := client.Invoke(context.Background(), request("other"))
	var expectation *inferencetest.ExpectationError
	if !errors.As(err, &expectation) || expectation.Call != 2 || !errors.Is(err, wrong) {
		t.Fatalf("err = %v, want ExpectationError wrapping %v", err, wrong)
	}
	if !errors.Is(client.Err(), wrong) {
		t.Fatalf("Err() = %v", client.Err())
	}
	// The refused step was consumed.
	resp, err := client.Invoke(context.Background(), request("third"))
	if err != nil || resp.Message.Blocks[0].(*content.TextBlock).Text != "after" {
		t.Fatalf("Invoke = %v, %v", resp, err)
	}
}

func TestInvalidRequestConsumesNothing(t *testing.T) {
	t.Parallel()
	client := inferencetest.New(inferencetest.Text("x"))
	cases := map[string]inference.Request{
		"no model": {},
		"forced undeclared tool": func() inference.Request {
			r := request("x")
			r.ToolChoice = inference.ToolNamed("missing")
			return r
		}(),
	}
	for name, req := range cases {
		if _, err := client.Invoke(context.Background(), req); err == nil {
			t.Fatalf("%s: Invoke accepted an invalid request", name)
		}
		if _, err := client.Stream(context.Background(), req); err == nil {
			t.Fatalf("%s: Stream accepted an invalid request", name)
		}
	}
	if client.Remaining() != 1 || len(client.Requests()) != 0 || client.Err() != nil {
		t.Fatalf("remaining=%d requests=%d err=%v", client.Remaining(), len(client.Requests()), client.Err())
	}
}

func TestRequestsAreIsolatedCopies(t *testing.T) {
	t.Parallel()
	client := inferencetest.New(inferencetest.Text("x"))
	req := request("original")
	req.Tools = []inference.Tool{{Name: "t", Schema: []byte(`{"type":"object"}`)}}
	if _, err := client.Invoke(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	req.Messages[0].(*content.UserMessage).Blocks[0].(*content.TextBlock).Text = "mutated"
	req.Tools[0].Schema[0] = 'X'
	got := client.Requests()
	if text := inferencetest.LastUserText(got[0]); text != "original" {
		t.Fatalf("recorded text = %q, want original", text)
	}
	if string(got[0].Tools[0].Schema) != `{"type":"object"}` {
		t.Fatalf("recorded schema = %s", got[0].Tools[0].Schema)
	}
	got[0].Messages[0].(*content.UserMessage).Blocks[0].(*content.TextBlock).Text = "again"
	if text := inferencetest.LastUserText(client.Requests()[0]); text != "original" {
		t.Fatalf("Requests leaked its record: %q", text)
	}
}

func TestEchoRepeat(t *testing.T) {
	t.Parallel()
	client := inferencetest.New(inferencetest.Text("hi"), inferencetest.Echo().Repeat())
	for i, want := range []string{"hi", "one", "two", "three"} {
		in := want
		if i == 0 {
			in = "ignored"
		}
		resp, err := client.Invoke(context.Background(), request(in))
		if err != nil {
			t.Fatal(err)
		}
		if got := resp.Message.Blocks[0].(*content.TextBlock).Text; got != want {
			t.Fatalf("call %d = %q, want %q", i+1, got, want)
		}
	}
	if client.Remaining() != 1 {
		t.Fatalf("Remaining = %d, want 1", client.Remaining())
	}
}

func TestFunc(t *testing.T) {
	t.Parallel()
	client := inferencetest.New(inferencetest.Func(func(req inference.Request) inferencetest.Step {
		return inferencetest.ToolCall("count", fmt.Sprintf(`{"n":%d}`, len(req.Messages)))
	}))
	resp, err := client.Invoke(context.Background(), request("x"))
	if err != nil {
		t.Fatal(err)
	}
	if got := string(resp.Message.Blocks[0].(*content.ToolUseBlock).Input); got != `{"n":1}` {
		t.Fatalf("input = %s", got)
	}
}

func TestRepeatMustBeLast(t *testing.T) {
	t.Parallel()
	mustPanic := func(name string, fn func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Fatalf("%s did not panic", name)
			}
		}()
		fn()
	}
	mustPanic("New", func() { inferencetest.New(inferencetest.Echo().Repeat(), inferencetest.Text("x")) })
	client := inferencetest.New(inferencetest.Echo().Repeat())
	mustPanic("Append", func() { client.Append(inferencetest.Text("x")) })
}

func TestStepMethodsDoNotAlias(t *testing.T) {
	t.Parallel()
	// Three of each leaves spare capacity, so an append that shared base's
	// backing array would let b overwrite a.
	base := inferencetest.Thinking("t").Thinking("t").Thinking("t").
		ToolCall("base", "").ToolCall("base", "").ToolCall("base", "")
	a := base.Thinking("a").ToolCall("a", "")
	_ = base.Thinking("b").ToolCall("b", "")
	resp, err := inferencetest.New(a).Invoke(context.Background(), request("x"))
	if err != nil {
		t.Fatal(err)
	}
	want := []content.Block{
		content.NewSignedThinkingBlock("t", "", "", nil, ""),
		content.NewSignedThinkingBlock("t", "", "", nil, ""),
		content.NewSignedThinkingBlock("t", "", "", nil, ""),
		content.NewSignedThinkingBlock("a", "", "", nil, ""),
		content.NewToolUseBlock("call_1_0", "base", []byte("{}"), nil, ""),
		content.NewToolUseBlock("call_1_1", "base", []byte("{}"), nil, ""),
		content.NewToolUseBlock("call_1_2", "base", []byte("{}"), nil, ""),
		content.NewToolUseBlock("call_1_3", "a", []byte("{}"), nil, ""),
	}
	if !reflect.DeepEqual(resp.Message.Blocks, want) {
		t.Fatalf("blocks = %s", dump(resp.Message.Blocks))
	}
}

func TestConcurrentCallsConsumeEachStepOnce(t *testing.T) {
	t.Parallel()
	const n = 64
	steps := make([]inferencetest.Step, n)
	for i := range steps {
		steps[i] = inferencetest.Text(fmt.Sprint(i))
	}
	client := inferencetest.New(steps...)
	var (
		mu   sync.Mutex
		seen = map[string]bool{}
		wg   sync.WaitGroup
	)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			var text string
			if i%2 == 0 {
				resp, err := client.Invoke(context.Background(), request("x"))
				if err != nil {
					t.Error(err)
					return
				}
				text = resp.Message.Blocks[0].(*content.TextBlock).Text
			} else {
				sr, err := client.Stream(context.Background(), request("x"))
				if err != nil {
					t.Error(err)
					return
				}
				chunk, err := sr.Next()
				_ = sr.Close()
				if err != nil {
					t.Error(err)
					return
				}
				text = chunk.(*content.TextChunk).Text
			}
			mu.Lock()
			seen[text] = true
			mu.Unlock()
			_ = client.Requests()
			_ = client.Remaining()
		}(i)
	}
	wg.Wait()
	if len(seen) != n || client.Remaining() != 0 || len(client.Requests()) != n {
		t.Fatalf("distinct replies=%d remaining=%d requests=%d", len(seen), client.Remaining(), len(client.Requests()))
	}
}

func TestContext(t *testing.T) {
	t.Parallel()
	client := inferencetest.New(inferencetest.Text("a b c d").ChunkDelay(time.Hour))
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := client.Stream(cancelled, request("x")); !errors.Is(err, context.Canceled) {
		t.Fatalf("Stream on a cancelled context = %v", err)
	}
	if client.Remaining() != 1 {
		t.Fatal("a call on a cancelled context consumed a step")
	}

	ctx, cancel := context.WithCancel(context.Background())
	sr, err := client.Stream(ctx, request("x"))
	if err != nil {
		t.Fatal(err)
	}
	time.AfterFunc(10*time.Millisecond, cancel)
	if err := nextWithin(t, sr); !errors.Is(err, context.Canceled) {
		t.Fatalf("Next after cancel = %v, want context.Canceled", err)
	}

	client.Append(inferencetest.Text("a b").ChunkDelay(time.Hour))
	sr, err = client.Stream(context.Background(), request("x"))
	if err != nil {
		t.Fatal(err)
	}
	time.AfterFunc(10*time.Millisecond, func() { _ = sr.Close() })
	if err := nextWithin(t, sr); err == nil || errors.Is(err, io.EOF) {
		t.Fatalf("Next after Close = %v, want an error", err)
	}
}

// nextWithin calls Next and fails the test if it does not return promptly.
func nextWithin(t *testing.T, sr *stream.StreamReader[content.Chunk]) error {
	t.Helper()
	done := make(chan error, 1)
	go func() {
		_, err := sr.Next()
		done <- err
	}()
	select {
	case err := <-done:
		return err
	case <-time.After(5 * time.Second):
		t.Fatal("Next did not return after cancellation")
		return nil
	}
}

func TestModel(t *testing.T) {
	t.Parallel()
	m := inferencetest.Model()
	if err := m.Validate(); err != nil {
		t.Fatalf("Model does not validate: %v", err)
	}
	if !m.Caps.Tools || !m.Caps.StructuredOutputWithTools || !m.Caps.AcceptsImages || m.Limits.WindowTokens == 0 {
		t.Fatalf("caps = %+v limits = %+v", m.Caps, m.Limits)
	}
	if !inferencetest.Model(model.WithThinking()).Caps.Thinking {
		t.Fatal("options are not applied")
	}
}
