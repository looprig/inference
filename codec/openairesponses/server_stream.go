package openairesponses

import (
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/looprig/core/content"
	codec "github.com/looprig/inference/codec"
	stream "github.com/looprig/inference/stream"
)

// itemKind identifies which native output item variant is currently open on
// the wire, so WriteChunk knows whether an incoming chunk continues the open
// item or must close it and start a new one.
type itemKind uint8

const (
	itemKindNone itemKind = iota
	itemKindText
	itemKindRefusal
	itemKindThinking
	itemKindTool
)

// openResponsesStream begins the native Responses streaming response: it
// commits the text/event-stream headers and the 200 status, then emits the
// leading response.created and response.in_progress events before returning
// the request-scoped encoder. The response id is generated here (not per the
// design's abbreviated example ids) since OpenStream receives no
// request/target context to derive one from.
//
// response.in_progress is emitted here, back-to-back with response.created,
// rather than lazily before the first item. Both frames carry the same
// Response snapshot with status in_progress, so there is nothing extra to
// learn by waiting, and emitting it at open means a stream that fails or
// completes without producing a single item still sends it — which is exactly
// the case where a client blocked on it would hang longest.
func openResponsesStream(w http.ResponseWriter) (codec.StreamEncoder, error) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)

	enc := &serverStreamEncoder{
		w:          w,
		flusher:    flusher,
		ids:        newToolIDGenerator(),
		responseID: "resp_" + randomHex(12),
		createdAt:  time.Now().Unix(),
	}
	if err := enc.writeResponseCreated(); err != nil {
		return nil, err
	}
	if err := enc.writeResponseInProgress(); err != nil {
		return nil, err
	}
	return enc, nil
}

// serverStreamEncoder is the request-scoped codec.StreamEncoder for one
// in-flight native Responses stream. It is never shared across requests: it
// holds per-stream state (the open item bookkeeping, accumulated text for
// the eventual *.done/output_item.done payloads, and the id generator) that
// OpenStream fills in fresh each call.
type serverStreamEncoder struct {
	w       http.ResponseWriter
	flusher http.Flusher
	ids     func() string

	responseID string
	done       bool

	// createdAt is stamped once, when the stream opens, and repeated on every
	// frame that carries a Response. A response has ONE creation time: letting
	// encodeWireResponse.MarshalJSON stamp each literal separately made
	// response.created and response.completed disagree whenever the generation
	// crossed a second boundary, which is most of them.
	createdAt int64
	// sequence is the next value of the SSE frames' `sequence_number`.
	// ResponseStreamEvent declares the member required on all 53 of its
	// members, and it is an ordering fact, not a constant: writeEvent owns the
	// increment so no event type can be added without one.
	sequence int

	nextOutputIndex int
	closedItems     []wireItem

	openKind        itemKind
	openOutputIndex int
	openItemID      string
	openToolIndex   int // valid only when openKind == itemKindTool
	openToolCallID  string
	openToolName    string

	textAccum     strings.Builder
	refusalAccum  strings.Builder
	thinkingAccum strings.Builder
	thinkingState string
	toolArgsAccum strings.Builder
}

var _ codec.StreamEncoder = (*serverStreamEncoder)(nil)

func (e *serverStreamEncoder) writeResponseCreated() error {
	return e.writeEvent(eventResponseCreated, &sseResponseEnvelope{
		Type:     eventResponseCreated,
		Response: encodeWireResponse{ID: e.responseID, CreatedAt: e.createdAt, Status: statusInProgress, Output: []wireItem{}},
	})
}

// writeResponseInProgress repeats the created snapshot under
// ResponseInProgressEvent. The id and created_at are the stream's own fields,
// not fresh values, so the two frames describe one response rather than two.
func (e *serverStreamEncoder) writeResponseInProgress() error {
	return e.writeEvent(eventResponseInProgress, &sseResponseEnvelope{
		Type:     eventResponseInProgress,
		Response: encodeWireResponse{ID: e.responseID, CreatedAt: e.createdAt, Status: statusInProgress, Output: []wireItem{}},
	})
}

// WriteChunk encodes and flushes one content.Chunk as native stream event(s):
// opening/closing output_item (and, for text, content_part) events as the
// active item kind (or, for tool calls, the active neutral Index) changes,
// and the appropriate delta event for the chunk's own payload.
func (e *serverStreamEncoder) WriteChunk(chunk content.Chunk) error {
	if e.done {
		return &StreamTerminatedError{}
	}

	switch c := chunk.(type) {
	case *content.TextChunk:
		if err := e.ensureItem(itemKindText, 0, "", "", ""); err != nil {
			return err
		}
		if c.Text == "" {
			return nil
		}
		e.textAccum.WriteString(c.Text)
		return e.writeEvent(eventOutputTextDelta, &sseOutputTextDelta{
			Type: eventOutputTextDelta, ItemID: e.openItemID, OutputIndex: e.openOutputIndex, ContentIndex: 0, Delta: c.Text,
			Logprobs: []json.RawMessage{},
		})
	case *content.RefusalChunk:
		// OutputContent's refusal member has its own content part and its own
		// delta/done events, so a proxied refusal keeps that channel instead of
		// being re-served as output_text. It opens its OWN message item rather
		// than sharing the text item's content array: the part index would
		// otherwise have to be tracked per item, and a refusal never coexists
		// with text in practice.
		if err := e.ensureItem(itemKindRefusal, 0, "", "", ""); err != nil {
			return err
		}
		e.refusalAccum.WriteString(c.Text)
		return e.writeEvent(eventRefusalDelta, &sseRefusalDelta{
			Type: eventRefusalDelta, ItemID: e.openItemID, OutputIndex: e.openOutputIndex, ContentIndex: 0, Delta: c.Text,
		})
	case *content.ThinkingChunk:
		// Both members of the upstream state are load-bearing, not just the
		// blob: ReasoningItem.required lists id, and blocksToItems (encode.go)
		// DROPS a reasoning item that has no real id rather than fabricate one
		// the provider would reject. Serving a gateway-minted id would
		// therefore make a client's replay lose the whole item — silently, and
		// exactly when store:false makes the encrypted content the only
		// continuation there is.
		var upstream reasoningState
		if c.ProviderStateFormat == providerStateFormatOpenAIResponses && len(c.ProviderState) > 0 {
			decoded, err := opaqueStateToWire(c.ProviderState)
			if err != nil {
				return err
			}
			upstream = decoded
		}
		if c.Thinking == "" && upstream.EncryptedContent == "" && upstream.ID == "" {
			return nil
		}
		if err := e.ensureItem(itemKindThinking, 0, "", "", upstream.ID); err != nil {
			return err
		}
		if upstream.EncryptedContent != "" {
			e.thinkingState = upstream.EncryptedContent
		}
		if upstream.ID != "" {
			// The upstream id normally arrives on its own terminal chunk,
			// AFTER the summary deltas opened the item (stream.go emits it from
			// response.output_item.done), so adopting it here — not only at
			// ensureItem — is what makes the served item's id the real one.
			// Events already written keep the provisional id; every later
			// event, including the output_item.done and response.completed
			// snapshots a client rebuilds the item from, carries the upstream
			// id.
			e.openItemID = upstream.ID
		}
		if c.Thinking == "" {
			return nil
		}
		e.thinkingAccum.WriteString(c.Thinking)
		return e.writeEvent(eventReasoningSummaryDelta, &sseReasoningSummaryDelta{
			Type: eventReasoningSummaryDelta, ItemID: e.openItemID, OutputIndex: e.openOutputIndex, SummaryIndex: 0, Delta: c.Thinking,
		})
	case *content.ToolUseChunk:
		if err := e.ensureItem(itemKindTool, c.Index, c.ID, c.Name, ""); err != nil {
			return err
		}
		if c.InputJSON == "" {
			return nil
		}
		e.toolArgsAccum.WriteString(c.InputJSON)
		return e.writeEvent(eventFunctionCallArgsDelta, &sseFunctionCallArgsDelta{
			Type: eventFunctionCallArgsDelta, ItemID: e.openItemID, OutputIndex: e.openOutputIndex, Delta: c.InputJSON,
		})
	default:
		return &UnsupportedChunkError{Chunk: unsupportedChunkTypeName(chunk)}
	}
}

// ensureItem makes kind (identified, for a tool call, by the neutral index)
// the active open item, closing whatever was open first if it differs.
// Responses' wire protocol only ever streams deltas for one open item's
// content at a time in a single-branch response; a target that genuinely
// interleaves multiple tool calls' deltas is re-serialized as a sequence of
// (possibly repeated) single-item add/done pairs rather than true
// interleaving, matching anthropicapi's identical trade-off for
// content_block_start/stop.
//
// upstreamItemID is the item id the upstream target issued, when the caller
// already knows it; it is used verbatim rather than minting one, so the served
// item is the one the provider will recognise on replay. Empty means "no
// upstream id", and a synthetic one is minted as before.
func (e *serverStreamEncoder) ensureItem(kind itemKind, toolIndex int, toolID, toolName, upstreamItemID string) error {
	if e.openKind == kind && (kind != itemKindTool || e.openToolIndex == toolIndex) {
		return nil
	}
	if err := e.closeOpenItem(); err != nil {
		return err
	}

	outputIndex := e.nextOutputIndex
	e.nextOutputIndex++
	itemID := upstreamItemID
	if itemID == "" {
		itemID = e.ids()
	}

	e.openKind = kind
	e.openOutputIndex = outputIndex
	e.openItemID = itemID
	e.openToolIndex = toolIndex
	e.textAccum.Reset()
	e.refusalAccum.Reset()
	e.thinkingAccum.Reset()
	e.thinkingState = ""
	e.toolArgsAccum.Reset()

	switch kind {
	case itemKindText:
		item := wireItem{Type: itemTypeMessage, ID: itemID, Role: roleAssistant, Content: partsContent(nil), Status: statusInProgress}
		if err := e.writeEvent(eventOutputItemAdded, &sseOutputItemAdded{Type: eventOutputItemAdded, OutputIndex: outputIndex, Item: item}); err != nil {
			return err
		}
		part := wireContentPart{Type: contentTypeOutputText, Text: "", Annotations: []json.RawMessage{}}
		return e.writeEvent(eventContentPartAdded, &sseContentPartAdded{
			Type: eventContentPartAdded, ItemID: itemID, OutputIndex: outputIndex, ContentIndex: 0, Part: part,
		})
	case itemKindRefusal:
		item := wireItem{Type: itemTypeMessage, ID: itemID, Role: roleAssistant, Content: partsContent(nil), Status: statusInProgress}
		if err := e.writeEvent(eventOutputItemAdded, &sseOutputItemAdded{Type: eventOutputItemAdded, OutputIndex: outputIndex, Item: item}); err != nil {
			return err
		}
		part := wireContentPart{Type: contentTypeRefusal, Refusal: ""}
		return e.writeEvent(eventContentPartAdded, &sseContentPartAdded{
			Type: eventContentPartAdded, ItemID: itemID, OutputIndex: outputIndex, ContentIndex: 0, Part: part,
		})
	case itemKindThinking:
		item := wireItem{Type: itemTypeReasoning, ID: itemID, Summary: []wireSummaryPart{}}
		return e.writeEvent(eventOutputItemAdded, &sseOutputItemAdded{Type: eventOutputItemAdded, OutputIndex: outputIndex, Item: item})
	case itemKindTool:
		id := toolID
		if id == "" {
			id = e.ids()
		}
		e.openToolCallID = id
		e.openToolName = toolName
		item := wireItem{Type: itemTypeFunctionCall, ID: itemID, CallID: id, Name: toolName, Arguments: ""}
		return e.writeEvent(eventOutputItemAdded, &sseOutputItemAdded{Type: eventOutputItemAdded, OutputIndex: outputIndex, Item: item})
	}
	return nil
}

// closeOpenItem emits the terminal *.done event(s) for the currently open
// item (if any) using its accumulated payload, records the finished item for
// the eventual response.completed snapshot, and clears the open-item state.
func (e *serverStreamEncoder) closeOpenItem() error {
	switch e.openKind {
	case itemKindNone:
		return nil
	case itemKindText:
		text := e.textAccum.String()
		// output_text.done first, then content_part.done: the channel terminal
		// precedes the terminal of the container it lives in, matching the
		// refusal branch below (refusal.done, then content_part.done) and the
		// order a real Responses stream uses. A client reconstructing text from
		// output_text.* alone sees its terminal here; without it the text
		// channel simply stopped, indistinguishable from a stalled stream.
		if err := e.writeEvent(eventOutputTextDone, &sseOutputTextDone{
			Type: eventOutputTextDone, ItemID: e.openItemID, OutputIndex: e.openOutputIndex, ContentIndex: 0,
			Text: text, Logprobs: []json.RawMessage{},
		}); err != nil {
			return err
		}
		part := wireContentPart{Type: contentTypeOutputText, Text: text, Annotations: []json.RawMessage{}}
		if err := e.writeEvent(eventContentPartDone, &sseContentPartDone{
			Type: eventContentPartDone, ItemID: e.openItemID, OutputIndex: e.openOutputIndex, ContentIndex: 0, Part: part,
		}); err != nil {
			return err
		}
		item := wireItem{Type: itemTypeMessage, ID: e.openItemID, Role: roleAssistant, Content: partsContent([]wireContentPart{part}), Status: statusCompleted}
		if err := e.writeEvent(eventOutputItemDone, &sseOutputItemDone{Type: eventOutputItemDone, OutputIndex: e.openOutputIndex, Item: item}); err != nil {
			return err
		}
		e.closedItems = append(e.closedItems, item)
	case itemKindRefusal:
		refusal := e.refusalAccum.String()
		if err := e.writeEvent(eventRefusalDone, &sseRefusalDone{
			Type: eventRefusalDone, ItemID: e.openItemID, OutputIndex: e.openOutputIndex, ContentIndex: 0, Refusal: refusal,
		}); err != nil {
			return err
		}
		part := wireContentPart{Type: contentTypeRefusal, Refusal: refusal}
		if err := e.writeEvent(eventContentPartDone, &sseContentPartDone{
			Type: eventContentPartDone, ItemID: e.openItemID, OutputIndex: e.openOutputIndex, ContentIndex: 0, Part: part,
		}); err != nil {
			return err
		}
		item := wireItem{Type: itemTypeMessage, ID: e.openItemID, Role: roleAssistant, Content: partsContent([]wireContentPart{part}), Status: statusCompleted}
		if err := e.writeEvent(eventOutputItemDone, &sseOutputItemDone{Type: eventOutputItemDone, OutputIndex: e.openOutputIndex, Item: item}); err != nil {
			return err
		}
		e.closedItems = append(e.closedItems, item)
	case itemKindThinking:
		text := e.thinkingAccum.String()
		if err := e.writeEvent(eventReasoningSummaryDone, &sseReasoningSummaryDone{
			Type: eventReasoningSummaryDone, ItemID: e.openItemID, OutputIndex: e.openOutputIndex, SummaryIndex: 0, Text: text,
		}); err != nil {
			return err
		}
		item := wireItem{Type: itemTypeReasoning, ID: e.openItemID, Summary: []wireSummaryPart{{Type: summaryTypeText, Text: text}}, EncryptedContent: e.thinkingState}
		if err := e.writeEvent(eventOutputItemDone, &sseOutputItemDone{Type: eventOutputItemDone, OutputIndex: e.openOutputIndex, Item: item}); err != nil {
			return err
		}
		e.closedItems = append(e.closedItems, item)
	case itemKindTool:
		args := e.toolArgsAccum.String()
		if args == "" {
			args = emptyObject
		}
		if err := e.writeEvent(eventFunctionCallArgsDone, &sseFunctionCallArgsDone{
			Type: eventFunctionCallArgsDone, ItemID: e.openItemID, OutputIndex: e.openOutputIndex, Name: e.openToolName, Arguments: args,
		}); err != nil {
			return err
		}
		item := wireItem{Type: itemTypeFunctionCall, ID: e.openItemID, CallID: e.openToolCallID, Name: e.openToolName, Arguments: args}
		if err := e.writeEvent(eventOutputItemDone, &sseOutputItemDone{Type: eventOutputItemDone, OutputIndex: e.openOutputIndex, Item: item}); err != nil {
			return err
		}
		e.closedItems = append(e.closedItems, item)
	}
	e.openKind = itemKindNone
	return nil
}

// Finish encodes the native stream-completion event — closing any still-open
// item, then response.completed carrying the authoritative terminal status,
// model, usage, and the full accumulated output array — from result.
func (e *serverStreamEncoder) Finish(result stream.StreamResult) error {
	if e.done {
		return &StreamTerminatedError{}
	}
	e.done = true

	if err := e.closeOpenItem(); err != nil {
		return err
	}

	status, incomplete := statusForFinishReason(result.FinishReason)
	// Response.output is an array-typed required field: a stream that produced
	// no item at all must still terminate with [], never null. writeResponseCreated
	// already starts from an explicit empty slice for the same reason.
	items := e.closedItems
	if items == nil {
		items = []wireItem{}
	}
	resp := encodeWireResponse{
		ID:                e.responseID,
		CreatedAt:         e.createdAt,
		Status:            status,
		Model:             result.Model,
		Output:            items,
		Usage:             encodeUsage(result.Usage),
		IncompleteDetails: incomplete,
	}
	eventType := eventResponseCompleted
	if status == statusIncomplete {
		eventType = eventResponseIncomplete
	}
	return e.writeEvent(eventType, &sseResponseEnvelope{Type: eventType, Response: resp})
}

// Fail encodes a native Responses `response.failed` event — the post-header
// counterpart to WriteError's pre-header error envelope, using the same
// classification — and terminates the stream. It does not attempt to close
// any still-open item first: an in-stream failure is abrupt, not a clean
// completion.
func (e *serverStreamEncoder) Fail(err error) error {
	if e.done {
		return &StreamTerminatedError{}
	}
	e.done = true

	_, code, message := classifyError(err)
	resp := encodeWireResponse{ID: e.responseID, CreatedAt: e.createdAt, Status: statusFailed, Error: &wireResponseError{Code: code, Message: message}}
	return e.writeEvent(eventResponseFailed, &sseResponseEnvelope{Type: eventResponseFailed, Response: resp})
}

// writeEvent stamps payload with the stream's next sequence_number, marshals
// it, and writes it as one SSE frame ("event: <name>\ndata: <json>\n\n"),
// flushing immediately so streaming progressively delivers bytes rather than
// buffering until the handler returns. wire/sse (this module) only frames the
// read side; this dialect's write-side SSE framing is small enough that it is
// not worth a shared package, so it lives here, local to this dialect —
// matching anthropicapi's identical local writeEvent.
//
// payload is a sequenced rather than an `any` deliberately: sequence_number is
// required on EVERY ResponseStreamEvent member, so a new event payload that
// does not embed sseSequence must fail to compile rather than ship a frame
// missing it. That is the whole reason the parameter is typed at all.
func (e *serverStreamEncoder) writeEvent(name string, payload sequenced) error {
	payload.setSequenceNumber(e.sequence)
	e.sequence++
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	if _, err := e.w.Write([]byte("event: " + name + "\ndata: " + string(body) + "\n\n")); err != nil {
		return err
	}
	if e.flusher != nil {
		e.flusher.Flush()
	}
	return nil
}

func unsupportedChunkTypeName(c content.Chunk) string {
	if c == nil {
		return "<nil>"
	}
	switch c.(type) {
	case *content.TextChunk:
		return "TextChunk"
	case *content.ThinkingChunk:
		return "ThinkingChunk"
	case *content.RefusalChunk:
		return "RefusalChunk"
	case *content.ToolUseChunk:
		return "ToolUseChunk"
	default:
		return "unknown"
	}
}

// --- server-encode-direction SSE event payloads -----------------------------

// sequenced is what writeEvent accepts: an event payload whose
// sequence_number it can fill in. Only *T satisfies it, so every call site
// passes a pointer and every payload type embeds sseSequence.
type sequenced interface {
	setSequenceNumber(int)
}

// sseSequence carries the member every ResponseStreamEvent requires. It is
// embedded rather than repeated so that adding an event type cannot forget it:
// without the embed the payload does not implement sequenced and writeEvent
// will not take it.
type sseSequence struct {
	SequenceNumber int `json:"sequence_number"`
}

func (s *sseSequence) setSequenceNumber(n int) { s.SequenceNumber = n }

type sseResponseEnvelope struct {
	sseSequence
	Type     string             `json:"type"`
	Response encodeWireResponse `json:"response"`
}

type sseOutputItemAdded struct {
	sseSequence
	Type        string   `json:"type"`
	OutputIndex int      `json:"output_index"`
	Item        wireItem `json:"item"`
}

type sseOutputItemDone struct {
	sseSequence
	Type        string   `json:"type"`
	OutputIndex int      `json:"output_index"`
	Item        wireItem `json:"item"`
}

type sseContentPartAdded struct {
	sseSequence
	Type         string          `json:"type"`
	ItemID       string          `json:"item_id"`
	OutputIndex  int             `json:"output_index"`
	ContentIndex int             `json:"content_index"`
	Part         wireContentPart `json:"part"`
}

type sseContentPartDone struct {
	sseSequence
	Type         string          `json:"type"`
	ItemID       string          `json:"item_id"`
	OutputIndex  int             `json:"output_index"`
	ContentIndex int             `json:"content_index"`
	Part         wireContentPart `json:"part"`
}

// sseOutputTextDelta carries `logprobs` without omitempty:
// ResponseTextDeltaEvent.required lists it and it admits no null. The gateway
// is never asked for logprobs and has none to forward, and the empty array is
// the only legal way to say so.
type sseOutputTextDelta struct {
	sseSequence
	Type         string            `json:"type"`
	ItemID       string            `json:"item_id"`
	OutputIndex  int               `json:"output_index"`
	ContentIndex int               `json:"content_index"`
	Delta        string            `json:"delta"`
	Logprobs     []json.RawMessage `json:"logprobs"`
}

// sseOutputTextDone carries `logprobs` without omitempty for the same reason
// sseOutputTextDelta does: ResponseTextDoneEvent.required lists it and it
// admits no null. `text` is the WHOLE accumulated text, not the last delta —
// the done frame is self-contained, so a client that missed a delta can still
// recover the final string from it.
type sseOutputTextDone struct {
	sseSequence
	Type         string            `json:"type"`
	ItemID       string            `json:"item_id"`
	OutputIndex  int               `json:"output_index"`
	ContentIndex int               `json:"content_index"`
	Text         string            `json:"text"`
	Logprobs     []json.RawMessage `json:"logprobs"`
}

type sseRefusalDelta struct {
	sseSequence
	Type         string `json:"type"`
	ItemID       string `json:"item_id"`
	OutputIndex  int    `json:"output_index"`
	ContentIndex int    `json:"content_index"`
	Delta        string `json:"delta"`
}

type sseRefusalDone struct {
	sseSequence
	Type         string `json:"type"`
	ItemID       string `json:"item_id"`
	OutputIndex  int    `json:"output_index"`
	ContentIndex int    `json:"content_index"`
	Refusal      string `json:"refusal"`
}

type sseFunctionCallArgsDelta struct {
	sseSequence
	Type        string `json:"type"`
	ItemID      string `json:"item_id"`
	OutputIndex int    `json:"output_index"`
	Delta       string `json:"delta"`
}

// sseFunctionCallArgsDone carries `name`, which
// ResponseFunctionCallArgumentsDoneEvent requires and the delta event does
// not: the done frame is self-contained enough for a client to act on without
// having kept the output_item.added that named the function.
type sseFunctionCallArgsDone struct {
	sseSequence
	Type        string `json:"type"`
	ItemID      string `json:"item_id"`
	OutputIndex int    `json:"output_index"`
	Name        string `json:"name"`
	Arguments   string `json:"arguments"`
}

type sseReasoningSummaryDelta struct {
	sseSequence
	Type         string `json:"type"`
	ItemID       string `json:"item_id"`
	OutputIndex  int    `json:"output_index"`
	SummaryIndex int    `json:"summary_index"`
	Delta        string `json:"delta"`
}

type sseReasoningSummaryDone struct {
	sseSequence
	Type         string `json:"type"`
	ItemID       string `json:"item_id"`
	OutputIndex  int    `json:"output_index"`
	SummaryIndex int    `json:"summary_index"`
	Text         string `json:"text"`
}
