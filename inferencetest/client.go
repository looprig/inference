package inferencetest

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"slices"
	"sync"
	"time"
	"unicode"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/stream"
)

// Client is a scripted inference.Client. Each call that passes request
// validation consumes the next Step of the script, in call order, whether it
// arrives through Invoke or Stream. Construct one with New; the zero value is
// an empty script. A Client is safe for concurrent use.
type Client struct {
	mu       sync.Mutex
	script   []Step
	calls    int
	requests []inference.Request
	err      error
}

var _ inference.Client = (*Client)(nil)

// New returns a Client that answers calls with steps, in order.
//
// New panics if a Repeat step is followed by another step, which could never
// be reached.
func New(steps ...Step) *Client {
	c := &Client{}
	c.Append(steps...)
	return c
}

// Append adds steps to the end of the script, for example to extend it between
// turns of a test. It panics if the script would contain a step after a Repeat
// step.
func (c *Client) Append(steps ...Step) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, step := range steps {
		if n := len(c.script); n > 0 && c.script[n-1].repeat {
			panic("inferencetest: a Repeat step must be the last step of a script")
		}
		c.script = append(c.script, step.clone())
	}
}

// Requests returns the requests that reached the script, in call order. Each
// is a copy whose message and tool slices the caller may keep and modify; the
// messages' blocks are copied too.
func (c *Client) Requests() []inference.Request {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]inference.Request, len(c.requests))
	for i, req := range c.requests {
		out[i] = cloneRequest(req)
	}
	return out
}

// Remaining reports how many steps have not been consumed. A Repeat step
// counts as one and is never consumed.
func (c *Client) Remaining() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.script)
}

// Err reports the first script failure: an *ExhaustedError when a call found
// no step left, or an *ExpectationError when a step's Expect check refused its
// request. The failing call also returned it, but an agent loop typically
// reports a failed model call only as a failed turn; Err keeps the cause. It
// is nil when every call so far was answered as scripted, including calls
// answered with a scripted Fail.
func (c *Client) Err() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.err
}

// ExhaustedError reports a call made after every step was consumed.
type ExhaustedError struct {
	// Call is the 1-based number of the call that found the script empty.
	Call int
}

func (e *ExhaustedError) Error() string {
	return fmt.Sprintf("inferencetest: call %d: script exhausted", e.Call)
}

// ExpectationError reports a request refused by a step's Expect check.
type ExpectationError struct {
	// Call is the 1-based number of the refused call.
	Call int
	// Err is the check's error.
	Err error
}

func (e *ExpectationError) Error() string {
	return fmt.Sprintf("inferencetest: call %d: unexpected request: %v", e.Call, e.Err)
}

func (e *ExpectationError) Unwrap() error { return e.Err }

// Invoke answers the next step with a complete response.
func (c *Client) Invoke(ctx context.Context, req inference.Request) (*inference.Response, error) {
	r, err := c.answer(ctx, req)
	if err != nil {
		return nil, err
	}
	if r.streamErr != nil {
		return nil, r.streamErr
	}
	return &inference.Response{
		Message: &content.AIMessage{
			Message: content.Message{Role: content.RoleAssistant, Blocks: r.blocks()},
			Usage:   cloneUsage(r.usage),
		},
		Usage:        cloneUsage(r.usage),
		Model:        r.model,
		FinishReason: r.finish,
	}, nil
}

// Stream answers the next step as a stream of content deltas.
func (c *Client) Stream(ctx context.Context, req inference.Request) (*stream.StreamReader[content.Chunk], error) {
	r, err := c.answer(ctx, req)
	if err != nil {
		return nil, err
	}
	chunks := r.chunks()
	closed := make(chan struct{})
	var closeOnce sync.Once
	next := 0
	return stream.NewStreamReaderWithResult(
		func() (content.Chunk, error) {
			if err := streamAlive(ctx, closed); err != nil {
				return nil, err
			}
			if next == len(chunks) {
				if r.streamErr != nil {
					return nil, r.streamErr
				}
				return nil, io.EOF
			}
			if r.delay > 0 {
				timer := time.NewTimer(r.delay)
				select {
				case <-timer.C:
				case <-ctx.Done():
					timer.Stop()
					return nil, ctx.Err()
				case <-closed:
					timer.Stop()
					return nil, errReaderClosed
				}
			}
			chunk := chunks[next]
			next++
			return chunk, nil
		},
		func() error {
			closeOnce.Do(func() { close(closed) })
			return nil
		},
		func() (stream.StreamResult, bool, error) {
			return stream.StreamResult{Usage: cloneUsage(r.usage), Model: r.model, FinishReason: r.finish}, true, nil
		},
	), nil
}

// errReaderClosed is returned by Next on a reader that was closed.
var errReaderClosed = fmt.Errorf("inferencetest: stream reader closed")

func streamAlive(ctx context.Context, closed <-chan struct{}) error {
	select {
	case <-closed:
		return errReaderClosed
	default:
	}
	return ctx.Err()
}

// answer validates req the way a real client does, consumes a step, runs its
// checks and resolves it into a reply.
func (c *Client) answer(ctx context.Context, req inference.Request) (reply, error) {
	if err := ctx.Err(); err != nil {
		return reply{}, err
	}
	if err := req.Model.Validate(); err != nil {
		return reply{}, err
	}
	if err := inference.ValidateRequestFeatures(req); err != nil {
		return reply{}, err
	}

	c.mu.Lock()
	c.calls++
	call := c.calls
	c.requests = append(c.requests, cloneRequest(req))
	if len(c.script) == 0 {
		err := &ExhaustedError{Call: call}
		c.recordLocked(err)
		c.mu.Unlock()
		return reply{}, err
	}
	step := c.script[0]
	if !step.repeat {
		c.script = c.script[1:]
	}
	c.mu.Unlock()

	// Checks and Func callbacks are caller code: run them without the lock so
	// one may inspect the Client.
	for _, check := range step.expect {
		if err := check(req); err != nil {
			err := &ExpectationError{Call: call, Err: err}
			c.mu.Lock()
			c.recordLocked(err)
			c.mu.Unlock()
			return reply{}, err
		}
	}
	for step.fn != nil {
		step = step.fn(req).clone()
	}
	if step.fail != nil {
		return reply{}, step.fail
	}
	return newReply(step, call, req), nil
}

func (c *Client) recordLocked(err error) {
	if c.err == nil {
		c.err = err
	}
}

// reply is a resolved step: the content one call delivers.
type reply struct {
	thinking  []string
	text      string
	toolCalls []toolCall
	finish    stream.FinishReason
	usage     *content.Usage
	model     string
	streamErr error
	chunkSize int
	delay     time.Duration
}

func newReply(s Step, call int, req inference.Request) reply {
	r := reply{
		text:      s.text,
		finish:    s.finish,
		model:     req.Model.Name,
		streamErr: s.streamErr,
		chunkSize: s.chunkSize,
		delay:     s.chunkDelay,
	}
	// Empty reasoning is dropped, as codecs drop empty deltas, so Invoke and
	// Stream agree on which blocks exist.
	for _, t := range s.thinking {
		if t != "" {
			r.thinking = append(r.thinking, t)
		}
	}
	for i, tc := range s.toolCalls {
		if tc.id == "" {
			tc.id = fmt.Sprintf("call_%d_%d", call, i)
		}
		if tc.args == "" {
			tc.args = "{}"
		}
		r.toolCalls = append(r.toolCalls, tc)
	}
	if !s.hasFinish {
		r.finish = stream.FinishReasonStop
		if len(r.toolCalls) > 0 {
			r.finish = stream.FinishReasonToolUse
		}
	}
	if s.usage != nil {
		usage := *s.usage
		r.usage = &usage
	} else {
		r.usage = r.estimateUsage(req)
	}
	return r
}

// blocks is the complete message content: reasoning, text, tool calls.
func (r reply) blocks() []content.Block {
	blocks := make([]content.Block, 0, len(r.thinking)+1+len(r.toolCalls))
	for _, t := range r.thinking {
		blocks = append(blocks, content.NewSignedThinkingBlock(t, "", "", nil, ""))
	}
	if r.text != "" {
		blocks = append(blocks, &content.TextBlock{Text: r.text})
	}
	for _, tc := range r.toolCalls {
		blocks = append(blocks, content.NewToolUseBlock(tc.id, tc.name, json.RawMessage(tc.args), nil, ""))
	}
	return blocks
}

// chunks is the same content as deltas. Like a Messages-style stream, every
// block takes the next position in one shared index sequence, so reasoning and
// tool-call indexes are sparse; stream accumulators key by index and never
// assume density. A tool call opens with its ID and Name and no arguments,
// then streams the arguments as InputJSON fragments.
func (r reply) chunks() []content.Chunk {
	var chunks []content.Chunk
	index := 0
	for _, t := range r.thinking {
		for _, part := range split(t, r.chunkSize) {
			chunks = append(chunks, &content.ThinkingChunk{Index: index, Thinking: part})
		}
		index++
	}
	if r.text != "" {
		for _, part := range split(r.text, r.chunkSize) {
			chunks = append(chunks, &content.TextChunk{Text: part})
		}
		index++
	}
	for _, tc := range r.toolCalls {
		chunks = append(chunks, &content.ToolUseChunk{Index: index, ID: tc.id, Name: tc.name})
		for _, part := range split(tc.args, r.chunkSize) {
			chunks = append(chunks, &content.ToolUseChunk{Index: index, InputJSON: part})
		}
		index++
	}
	return chunks
}

// split cuts s into deltas: words with their trailing whitespace when size is
// zero, runs of at most size runes when size is positive, and s whole when
// size is negative. It returns nothing for an empty s.
func split(s string, size int) []string {
	if s == "" {
		return nil
	}
	if size < 0 {
		return []string{s}
	}
	var parts []string
	start, runes := 0, 0
	prevSpace := false
	for i, ch := range s {
		space := unicode.IsSpace(ch)
		boundary := size > 0 && runes == size ||
			size == 0 && i > start && prevSpace && !space
		if boundary {
			parts = append(parts, s[start:i])
			start, runes = i, 0
		}
		runes++
		prevSpace = space
	}
	return append(parts, s[start:])
}

// estimateUsage approximates token counts at one token per four bytes.
func (r reply) estimateUsage(req inference.Request) *content.Usage {
	in := len(req.System)
	for _, msg := range req.Messages {
		in += messageBytes(msg)
	}
	for _, tool := range req.Tools {
		in += len(tool.Name) + len(tool.Description) + len(tool.Schema)
	}
	reasoning := 0
	for _, t := range r.thinking {
		reasoning += len(t)
	}
	out := reasoning + len(r.text)
	for _, tc := range r.toolCalls {
		out += len(tc.name) + len(tc.args)
	}
	return &content.Usage{
		InputTokens:     tokens(in),
		OutputTokens:    tokens(out),
		ReasoningTokens: tokens(reasoning),
	}
}

func tokens(bytes int) content.TokenCount {
	if bytes <= 0 {
		return 0
	}
	return content.TokenCount(uint(bytes)/4 + min(uint(bytes)%4, 1))
}

func messageBytes(msg content.Conversation) int {
	switch m := msg.(type) {
	case *content.SystemMessage:
		return blocksBytes(m.Blocks)
	case *content.UserMessage:
		return blocksBytes(m.Blocks)
	case *content.AIMessage:
		return blocksBytes(m.Blocks)
	case *content.ToolResultMessage:
		return blocksBytes(m.Blocks)
	}
	return 0
}

func blocksBytes(blocks []content.Block) int {
	n := 0
	for _, block := range blocks {
		switch b := block.(type) {
		case *content.TextBlock:
			n += len(b.Text)
		case *content.ThinkingBlock:
			n += len(b.Thinking)
		case *content.ToolUseBlock:
			n += len(b.Name) + len(b.Input)
		case *content.ToolResultBlock:
			n += blocksBytes(b.Content)
		}
	}
	return n
}

func cloneUsage(u *content.Usage) *content.Usage {
	if u == nil {
		return nil
	}
	c := *u
	return &c
}

// cloneRequest copies the request's slices and its messages' blocks, so a
// recorded request is unaffected by a caller that later reuses its buffers.
func cloneRequest(req inference.Request) inference.Request {
	out := req
	out.Model = req.Model.Clone()
	out.Tools = slices.Clone(req.Tools)
	for i := range out.Tools {
		out.Tools[i].Schema = slices.Clone(out.Tools[i].Schema)
	}
	if req.Output != nil {
		output := req.Output.Clone()
		out.Output = &output
	}
	if req.Override != nil {
		override := req.Override.Clone()
		out.Override = &override
	}
	if req.Messages != nil {
		out.Messages = make(content.AgenticMessages, len(req.Messages))
		for i, msg := range req.Messages {
			out.Messages[i] = cloneMessage(msg)
		}
	}
	return out
}

func cloneMessage(msg content.Conversation) content.Conversation {
	switch m := msg.(type) {
	case *content.SystemMessage:
		if m == nil {
			return m
		}
		c := *m
		c.Blocks = content.CloneBlocks(m.Blocks)
		return &c
	case *content.UserMessage:
		if m == nil {
			return m
		}
		c := *m
		c.Blocks = content.CloneBlocks(m.Blocks)
		return &c
	case *content.AIMessage:
		if m == nil {
			return m
		}
		c := *m
		c.Blocks = content.CloneBlocks(m.Blocks)
		c.Usage = cloneUsage(m.Usage)
		return &c
	case *content.ToolResultMessage:
		if m == nil {
			return m
		}
		c := *m
		c.Blocks = content.CloneBlocks(m.Blocks)
		return &c
	}
	return msg
}
