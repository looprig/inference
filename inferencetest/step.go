package inferencetest

import (
	"slices"
	"strings"
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/stream"
)

// Step is one scripted model reply. Build one with Text, Thinking, ToolCall,
// ToolCallWithID, Fail, Echo or Func, then refine it with its methods. Every
// method returns a modified copy, so a Step value can be shared and reused.
//
// A reply's content is laid out as reasoning, then text, then tool calls —
// the order an agent loop assembles a streamed message in — and within each
// kind in the order it was added. Successive Text calls concatenate into one
// text block; each Thinking call is its own reasoning block; each tool call is
// its own tool-use block.
//
// The zero Step is a reply with no content, which an agent loop treats as an
// empty response.
type Step struct {
	thinking  []string
	text      string
	toolCalls []toolCall

	finish     stream.FinishReason
	hasFinish  bool
	usage      *content.Usage
	fail       error
	streamErr  error
	chunkSize  int
	chunkDelay time.Duration
	expect     []func(inference.Request) error
	fn         func(inference.Request) Step
	repeat     bool
}

type toolCall struct {
	id   string
	name string
	args string
}

// Text returns a reply that says text. It finishes with
// stream.FinishReasonStop.
func Text(text string) Step { return Step{}.Text(text) }

// Thinking returns a reply that carries one reasoning block. Chain Text or
// ToolCall to give the reasoning something to lead to.
func Thinking(text string) Step { return Step{}.Thinking(text) }

// ToolCall returns a reply that calls tool name with argsJSON as its
// arguments. The call gets a deterministic ID, "call_<call>_<n>", where call is
// the 1-based number of the Client call it answers and n its position among
// the reply's tool calls. An empty argsJSON is sent as "{}". The arguments are
// sent verbatim, not validated: a model can emit malformed JSON, and a test may
// want to see how an agent copes. The reply finishes with
// stream.FinishReasonToolUse.
func ToolCall(name, argsJSON string) Step { return Step{}.ToolCall(name, argsJSON) }

// ToolCallWithID is ToolCall with a caller-chosen tool-use ID.
func ToolCallWithID(id, name, argsJSON string) Step {
	return Step{}.ToolCallWithID(id, name, argsJSON)
}

// Fail returns a step whose call fails with err before any output: Invoke
// returns (nil, err) and Stream returns (nil, err), the way a provider error
// such as failure.NewAPIError(429, ...) surfaces from a real client. Content
// added to a failing step is never delivered.
func Fail(err error) Step { return Step{fail: err} }

// Echo returns a step that replies with the text of the request's last user
// message (see LastUserText). Combine it with Repeat for a keyless demo agent
// that answers anything:
//
//	inferencetest.New(inferencetest.Echo().Repeat())
func Echo() Step {
	return Func(func(req inference.Request) Step { return Text(LastUserText(req)) })
}

// Func returns a step computed from the request it answers. fn's step is used
// in its place — its content, finish reason, usage, failure and delivery
// settings — except that its Repeat and Expect settings are ignored; set those
// on the step Func returns.
func Func(fn func(inference.Request) Step) Step { return Step{fn: fn} }

// Text appends text to the reply's text block.
func (s Step) Text(text string) Step {
	s = s.clone()
	s.text += text
	return s
}

// Thinking adds a reasoning block. It carries no signature, as a reasoning
// block from a model without signed reasoning would.
func (s Step) Thinking(text string) Step {
	s = s.clone()
	s.thinking = append(s.thinking, text)
	return s
}

// ToolCall adds a tool call; see the package-level ToolCall.
func (s Step) ToolCall(name, argsJSON string) Step {
	return s.ToolCallWithID("", name, argsJSON)
}

// ToolCallWithID adds a tool call with a caller-chosen ID. An empty id is
// replaced with the deterministic one ToolCall assigns.
func (s Step) ToolCallWithID(id, name, argsJSON string) Step {
	s = s.clone()
	s.toolCalls = append(s.toolCalls, toolCall{id: id, name: name, args: argsJSON})
	return s
}

// Finish overrides the reply's finish reason. By default a reply with tool
// calls finishes with stream.FinishReasonToolUse and any other reply with
// stream.FinishReasonStop; use Finish to script, for example, a reply cut off
// at the output limit (stream.FinishReasonLength).
func (s Step) Finish(reason stream.FinishReason) Step {
	s = s.clone()
	s.finish = reason
	s.hasFinish = true
	return s
}

// Usage sets the token usage the reply reports. By default a Client reports a
// deterministic estimate of about one token per four bytes of request and
// reply, which is enough to exercise usage accounting but is not a real
// tokenizer's count.
func (s Step) Usage(usage content.Usage) Step {
	s = s.clone()
	s.usage = &usage
	return s
}

// StreamError makes the reply fail part-way: Stream delivers all of the
// reply's chunks and then Next returns err instead of io.EOF, the way a
// dropped connection truncates a real stream, so the reader reports no
// Result. Invoke, which has no partial result to return, returns (nil, err).
func (s Step) StreamError(err error) Step {
	s = s.clone()
	s.streamErr = err
	return s
}

// ChunkSize sets how Stream splits text, reasoning and tool arguments into
// deltas. By default each delta is one word with its trailing whitespace; a
// positive n makes each delta at most n runes, and a negative n sends each
// block as a single delta.
func (s Step) ChunkSize(n int) Step {
	s = s.clone()
	s.chunkSize = n
	return s
}

// ChunkDelay makes Stream wait d before each delta, so a demo streams at a
// visible pace and a test can interrupt a stream part-way. Next stops waiting
// when the call's context is cancelled or the reader is closed.
func (s Step) ChunkDelay(d time.Duration) Step {
	s = s.clone()
	s.chunkDelay = d
	return s
}

// Expect adds a check on the request the step answers. When check returns an
// error the call fails with an *ExpectationError wrapping it, the step is
// consumed, and the Client's Err reports it — an agent loop usually turns a
// failed model call into a failed turn, so Err is how a test learns why.
// Several checks run in the order they were added.
func (s Step) Expect(check func(inference.Request) error) Step {
	s = s.clone()
	s.expect = append(s.expect, check)
	return s
}

// Repeat makes the step answer every remaining call instead of one: it is
// never consumed, so it must be the last step of a script.
func (s Step) Repeat() Step {
	s = s.clone()
	s.repeat = true
	return s
}

// clone returns a copy that shares no mutable state with s, so a method never
// alters a Step another variable still holds.
func (s Step) clone() Step {
	c := s
	c.thinking = slices.Clone(s.thinking)
	c.toolCalls = slices.Clone(s.toolCalls)
	c.expect = slices.Clone(s.expect)
	if s.usage != nil {
		usage := *s.usage
		c.usage = &usage
	}
	return c
}

// LastUserText returns the concatenated text blocks of the request's last user
// message, or "" when the request has none. It is the text Echo replies with.
func LastUserText(req inference.Request) string {
	for i := len(req.Messages) - 1; i >= 0; i-- {
		user, ok := req.Messages[i].(*content.UserMessage)
		if !ok || user == nil {
			continue
		}
		var text strings.Builder
		for _, block := range user.Blocks {
			if tb, ok := block.(*content.TextBlock); ok && tb != nil {
				text.WriteString(tb.Text)
			}
		}
		return text.String()
	}
	return ""
}
