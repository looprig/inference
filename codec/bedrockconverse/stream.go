package bedrockconverse

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"net/http"

	"github.com/looprig/core/content"
	codec "github.com/looprig/inference/codec"
	"github.com/looprig/inference/stream"
	"github.com/looprig/inference/wire/eventstream"
)

var _ codec.StreamingCodec = Codec{}

// DecodeStream frames a successful ConverseStream response with AWS Event
// Stream and maps each event to shared content chunks. The returned reader owns
// resp.Body through the wire framer.
func (Codec) DecodeStream(resp *http.Response) (*stream.StreamReader[content.Chunk], error) {
	if resp == nil {
		return nil, &StreamDecodeError{Reason: "nil response"}
	}
	if resp.Body == nil {
		return nil, &StreamDecodeError{Reason: "nil response body"}
	}
	frames, err := eventstream.DecodeStreamFrames(resp.Body)
	if err != nil {
		return nil, err
	}
	collector := &streamResultCollector{
		active: make(map[int]streamBlockKind),
		closed: make(map[int]struct{}),
	}
	return stream.FramesToChunksWithResult(frames, collector.mapFrame, collector.result), nil
}

// streamResultCollector tracks the per-message state ConverseStream requires:
// which content blocks are open, which have closed, and the terminal metadata.
//
// INDEX SEMANTICS. `contentBlockIndex` is AWS's position in the message's
// SINGLE `content` array — one space shared by text, reasoningContent and
// toolUse blocks — so it is carried verbatim onto every chunk that has one.
// A gap in the reasoning accumulator's indexes is a text or tool-use block, not
// a lost reasoning block.
type streamResultCollector struct {
	started     bool
	stopped     bool
	active      map[int]streamBlockKind
	closed      map[int]struct{}
	resultValue stream.StreamResult
	usageSeen   bool
}

func (c *streamResultCollector) mapFrame(frame stream.StreamFrame) ([]content.Chunk, error) {
	messageType := frame.Metadata[":message-type"]
	if messageType == "exception" || messageType == "error" {
		return nil, c.decodeException(frame)
	}
	eventName := frame.Name
	if eventName == "" {
		eventName = frame.Metadata[":event-type"]
	}
	if eventName == "" {
		return nil, &StreamDecodeError{Reason: "event is missing :event-type"}
	}

	switch eventName {
	case "messageStart":
		return nil, c.messageStart(frame.Data)
	case "contentBlockStart":
		return c.contentBlockStart(frame.Data)
	case "contentBlockDelta":
		return c.contentBlockDelta(frame.Data)
	case "contentBlockStop":
		return nil, c.contentBlockStop(frame.Data)
	case "messageStop":
		return nil, c.messageStop(frame.Data)
	case "metadata":
		return nil, c.metadata(frame.Data)
	default:
		// AWS may add event types that have no shared content representation.
		// Preserve forward compatibility by ignoring unknown event payloads.
		return nil, nil
	}
}

func (c *streamResultCollector) messageStart(payload []byte) error {
	if c.started {
		return &StreamDecodeError{Reason: "duplicate messageStart"}
	}
	if c.stopped {
		return &StreamDecodeError{Reason: "messageStart after messageStop"}
	}
	var event struct {
		Role string `json:"role"`
	}
	if err := json.Unmarshal(payload, &event); err != nil {
		return &StreamDecodeError{Reason: "decode messageStart", Err: err}
	}
	if event.Role != roleAssistant {
		return &StreamDecodeError{Reason: "messageStart role is not assistant"}
	}
	c.started = true
	return nil
}

func (c *streamResultCollector) contentBlockStart(payload []byte) ([]content.Chunk, error) {
	if !c.started || c.stopped {
		return nil, &StreamDecodeError{Reason: "contentBlockStart outside active message"}
	}
	var event streamContentBlockStart
	if err := json.Unmarshal(payload, &event); err != nil {
		return nil, &StreamDecodeError{Reason: "decode contentBlockStart", Err: err}
	}
	if event.Index == nil || event.Start == nil {
		return nil, &StreamDecodeError{Reason: "contentBlockStart is missing contentBlockIndex or start"}
	}
	index := *event.Index
	if _, exists := c.active[index]; exists {
		return nil, &StreamDecodeError{Reason: "duplicate contentBlockStart"}
	}
	if _, exists := c.closed[index]; exists {
		return nil, &StreamDecodeError{Reason: "contentBlockStart reuses a closed index"}
	}
	if event.Start.ToolUse == nil {
		return nil, &StreamDecodeError{Reason: "contentBlockStart is only valid for tool use"}
	}
	if event.Start.ToolUse.ToolUseID == "" || event.Start.ToolUse.Name == "" {
		return nil, &StreamDecodeError{Reason: "toolUse start is missing toolUseId or name"}
	}
	c.active[index] = streamBlockKindToolUse
	return []content.Chunk{&content.ToolUseChunk{
		Index: index,
		ID:    event.Start.ToolUse.ToolUseID,
		Name:  event.Start.ToolUse.Name,
	}}, nil
}

func (c *streamResultCollector) contentBlockDelta(payload []byte) ([]content.Chunk, error) {
	if !c.started || c.stopped {
		return nil, &StreamDecodeError{Reason: "contentBlockDelta outside active message"}
	}
	var event streamContentBlockDelta
	if err := json.Unmarshal(payload, &event); err != nil {
		return nil, &StreamDecodeError{Reason: "decode contentBlockDelta", Err: err}
	}
	if event.Index == nil || event.Delta == nil {
		return nil, &StreamDecodeError{Reason: "contentBlockDelta is missing contentBlockIndex or delta"}
	}
	index := *event.Index
	if variants := streamBlockDeltaVariantCount(*event.Delta); variants != 1 {
		return nil, &StreamDecodeError{Reason: "content block delta must contain exactly one recognized variant"}
	}
	kind := event.Delta.kind()
	activeKind, active := c.active[index]
	if !active {
		if _, closed := c.closed[index]; closed {
			return nil, &StreamDecodeError{Reason: "contentBlockDelta received after stop"}
		}
		if kind == streamBlockKindToolUse {
			return nil, &StreamDecodeError{Reason: "toolUse delta received without contentBlockStart"}
		}
		// Bedrock emits contentBlockStart only for tool-use blocks. Text and
		// reasoning blocks become active with their first delta.
		c.active[index] = kind
	} else if activeKind != kind {
		return nil, &StreamDecodeError{Reason: "contentBlockDelta changes the content block variant"}
	}
	if event.Delta.Text != nil {
		return []content.Chunk{&content.TextChunk{Text: *event.Delta.Text}}, nil
	}
	if event.Delta.ReasoningContent != nil {
		reasoning := event.Delta.ReasoningContent
		reasoningVariants := 0
		if reasoning.Text != nil {
			reasoningVariants++
		}
		if reasoning.Signature != nil {
			reasoningVariants++
		}
		if len(reasoning.RedactedContent) > 0 {
			reasoningVariants++
		}
		if reasoningVariants != 1 {
			return nil, &StreamDecodeError{Reason: "reasoning content delta must contain exactly one recognized variant"}
		}
		// Every reasoning delta carries the event's own contentBlockIndex, the
		// same index the tool-use deltas below use. A Converse message may hold
		// SEVERAL reasoningContent blocks, each sealed by its own signature, and
		// the next request must replay them block-for-block; dropping the index
		// concatenated their text and rebound the last signature to all of it.
		if len(reasoning.RedactedContent) > 0 {
			encoded, _ := json.Marshal(base64.StdEncoding.EncodeToString(reasoning.RedactedContent))
			return []content.Chunk{&content.ThinkingChunk{
				Index:               index,
				ProviderState:       encoded,
				ProviderStateFormat: providerStateFormatBedrockRedacted,
			}}, nil
		}
		if reasoning.Text != nil {
			return []content.Chunk{&content.ThinkingChunk{Index: index, Thinking: *reasoning.Text}}, nil
		}
		// Labelled exactly as the non-streaming decoder labels it: streaming
		// must reconstruct the same continuation state, and provenance is part
		// of it.
		return []content.Chunk{&content.ThinkingChunk{
			Index:           index,
			Signature:       *reasoning.Signature,
			SignatureFormat: signatureFormatFor(*reasoning.Signature),
		}}, nil
	}
	if event.Delta.ToolUse != nil {
		if event.Delta.ToolUse.Input == "" {
			return nil, nil
		}
		return []content.Chunk{&content.ToolUseChunk{Index: index, InputJSON: event.Delta.ToolUse.Input}}, nil
	}
	return nil, nil
}

func streamBlockDeltaVariantCount(delta streamBlockDelta) int {
	count := 0
	if delta.Text != nil {
		count++
	}
	if delta.ReasoningContent != nil {
		count++
	}
	if delta.ToolUse != nil {
		count++
	}
	return count
}

func (c *streamResultCollector) contentBlockStop(payload []byte) error {
	if !c.started || c.stopped {
		return &StreamDecodeError{Reason: "contentBlockStop outside active message"}
	}
	var event streamContentBlockStop
	if err := json.Unmarshal(payload, &event); err != nil {
		return &StreamDecodeError{Reason: "decode contentBlockStop", Err: err}
	}
	if event.Index == nil {
		return &StreamDecodeError{Reason: "contentBlockStop is missing contentBlockIndex"}
	}
	index := *event.Index
	if _, exists := c.active[index]; !exists {
		return &StreamDecodeError{Reason: "contentBlockStop received without start"}
	}
	delete(c.active, index)
	c.closed[index] = struct{}{}
	return nil
}

func (c *streamResultCollector) messageStop(payload []byte) error {
	if !c.started {
		return &StreamDecodeError{Reason: "messageStop before messageStart"}
	}
	if c.stopped {
		return &StreamDecodeError{Reason: "duplicate messageStop"}
	}
	if len(c.active) != 0 {
		return &StreamDecodeError{Reason: "messageStop with open content block"}
	}
	var event streamMessageStop
	if err := json.Unmarshal(payload, &event); err != nil {
		return &StreamDecodeError{Reason: "decode messageStop", Err: err}
	}
	c.stopped = true
	c.resultValue.FinishReason = mapFinishReason(event.StopReason)
	return nil
}

func (c *streamResultCollector) metadata(payload []byte) error {
	if !c.started || !c.stopped {
		return &StreamDecodeError{Reason: "metadata before messageStop"}
	}
	var event streamMetadata
	if err := json.Unmarshal(payload, &event); err != nil {
		return &StreamDecodeError{Reason: "decode metadata", Err: err}
	}
	if event.Usage == nil {
		return nil
	}
	normalized, err := normalizeUsage(event.Usage)
	if err != nil {
		return err
	}
	c.resultValue.Usage = normalized
	c.usageSeen = true
	return nil
}

func (c *streamResultCollector) decodeException(frame stream.StreamFrame) error {
	var event struct {
		Message string `json:"message"`
	}
	if len(bytes.TrimSpace(frame.Data)) > 0 {
		if err := json.Unmarshal(frame.Data, &event); err != nil {
			return &StreamDecodeError{Reason: "decode exception event", Err: err}
		}
	}
	message := frame.Metadata[":error-message"]
	if message == "" {
		message = event.Message
	}
	if len(message) > 512 {
		message = message[:512]
	}
	typ := frame.Metadata[":exception-type"]
	if typ == "" {
		typ = frame.Metadata[":error-code"]
	}
	return &StreamAPIError{Type: typ, Message: message}
}

func (c *streamResultCollector) result() (stream.StreamResult, bool, error) {
	if !c.stopped {
		return stream.StreamResult{}, false, &StreamDecodeError{Reason: "stream ended before messageStop"}
	}
	if !c.usageSeen {
		c.resultValue.Usage = nil
	}
	return c.resultValue, true, nil
}

type streamContentBlockStart struct {
	Index *int              `json:"contentBlockIndex"`
	Start *streamBlockStart `json:"start"`
}

type streamBlockStart struct {
	ToolUse *streamToolUseStart `json:"toolUse"`
}

type streamToolUseStart struct {
	ToolUseID string `json:"toolUseId"`
	Name      string `json:"name"`
}

type streamContentBlockDelta struct {
	Index *int              `json:"contentBlockIndex"`
	Delta *streamBlockDelta `json:"delta"`
}

type streamBlockDelta struct {
	Text             *string               `json:"text"`
	ReasoningContent *streamReasoningDelta `json:"reasoningContent"`
	ToolUse          *streamToolUseDelta   `json:"toolUse"`
}

type streamBlockKind uint8

const (
	streamBlockKindText streamBlockKind = iota + 1
	streamBlockKindReasoning
	streamBlockKindToolUse
)

func (d streamBlockDelta) kind() streamBlockKind {
	if d.Text != nil {
		return streamBlockKindText
	}
	if d.ReasoningContent != nil {
		return streamBlockKindReasoning
	}
	return streamBlockKindToolUse
}

type streamReasoningDelta struct {
	Text            *string `json:"text"`
	Signature       *string `json:"signature"`
	RedactedContent []byte  `json:"redactedContent"`
}

type streamToolUseDelta struct {
	Input string `json:"input"`
}

type streamContentBlockStop struct {
	Index *int `json:"contentBlockIndex"`
}

type streamMessageStop struct {
	StopReason string `json:"stopReason"`
}

type streamMetadata struct {
	Usage *responseUsage `json:"usage"`
}
