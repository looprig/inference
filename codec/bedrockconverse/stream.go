package bedrockconverse

import (
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
		active: make(map[int]struct{}),
		closed: make(map[int]struct{}),
	}
	return stream.FramesToChunksWithResult(frames, collector.mapFrame, collector.result), nil
}

type streamResultCollector struct {
	started     bool
	stopped     bool
	active      map[int]struct{}
	closed      map[int]struct{}
	resultValue stream.StreamResult
	usageSeen   bool
}

func (c *streamResultCollector) mapFrame(frame stream.StreamFrame) ([]content.Chunk, error) {
	messageType := frame.Metadata[":message-type"]
	if messageType == "exception" {
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
	c.active[index] = struct{}{}
	if event.Start.ToolUse == nil {
		return nil, nil
	}
	if event.Start.ToolUse.ToolUseID == "" || event.Start.ToolUse.Name == "" {
		return nil, &StreamDecodeError{Reason: "toolUse start is missing toolUseId or name"}
	}
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
	if _, exists := c.active[index]; !exists {
		return nil, &StreamDecodeError{Reason: "contentBlockDelta received without start"}
	}
	if event.Delta.Text != nil {
		if *event.Delta.Text == "" {
			return nil, nil
		}
		return []content.Chunk{&content.TextChunk{Text: *event.Delta.Text}}, nil
	}
	if event.Delta.ReasoningContent != nil {
		if event.Delta.ReasoningContent.Text == "" {
			return nil, nil
		}
		return []content.Chunk{&content.ThinkingChunk{Thinking: event.Delta.ReasoningContent.Text}}, nil
	}
	if event.Delta.ToolUse != nil {
		if event.Delta.ToolUse.Input == "" {
			return nil, nil
		}
		return []content.Chunk{&content.ToolUseChunk{Index: index, InputJSON: event.Delta.ToolUse.Input}}, nil
	}
	return nil, nil
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
	if err := json.Unmarshal(frame.Data, &event); err != nil {
		return &StreamDecodeError{Reason: "decode exception event", Err: err}
	}
	message := event.Message
	if len(message) > 512 {
		message = message[:512]
	}
	return &StreamAPIError{Type: frame.Metadata[":exception-type"], Message: message}
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

type streamMessageStart struct {
	Role string `json:"role"`
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

type streamReasoningDelta struct {
	Text      string `json:"text"`
	Signature string `json:"signature"`
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
