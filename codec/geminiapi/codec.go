package geminiapi

import (
	"bytes"
	"encoding/json"
	"net/http"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	codec "github.com/looprig/inference/codec"
	"github.com/looprig/inference/wire/jsonbody"
)

// Codec is the Google Gemini generateContent wire dialect expressed as an
// codec.Codec (and, via DecodeStream, an codec.StreamingCodec). It is stateless
// (an empty struct with value-receiver methods), so one value is safely shared across
// goroutines: the transport owns HTTP mechanics, the Codec owns the JSON body,
// per-event semantics, and SSE stream decoding. Methods delegate to package-level free
// functions so the two surfaces cannot diverge.
type Codec struct{}

// Compile-time proof that Codec honors the codec.Codec contract.
var _ codec.Codec = Codec{}

// Compile-time proof that Codec also honors the ingress-side codec.ServerCodec
// contract: recognizing and decoding native
// POST /v1beta/models/{model}:generateContent and
// POST /v1beta/models/{model}:streamGenerateContent requests, and encoding
// results back into native generateContent responses/streams.
var _ codec.ServerCodec = Codec{}

// MatchRequest reports whether req is a POST to either of this codec's two
// owned routes: :generateContent (non-streaming) or :streamGenerateContent
// (streaming).
func (Codec) MatchRequest(req *http.Request) bool {
	return matchGenerateContentRequest(req)
}

// DecodeRequest decodes a matched request into a codec.DecodedRequest,
// delegating to the free decodeGenerateContentRequest. The {model} path
// segment becomes RequestedModel; which of the two routes matched sets
// Streaming.
func (Codec) DecodeRequest(req *http.Request) (codec.DecodedRequest, error) {
	return decodeGenerateContentRequest(req)
}

// WriteResponse encodes a complete inference.Response as the native
// generateContent non-streaming response, delegating to the free
// writeGenerateContentResponse.
func (Codec) WriteResponse(w http.ResponseWriter, resp *inference.Response) error {
	return writeGenerateContentResponse(w, resp)
}

// OpenStream begins the native streamGenerateContent streaming response and
// returns its request-scoped StreamEncoder, delegating to the free
// openGenerateContentStream.
func (Codec) OpenStream(w http.ResponseWriter) (codec.StreamEncoder, error) {
	return openGenerateContentStream(w)
}

// WriteError encodes err as the native generateContent error envelope,
// delegating to the free writeGenerateContentError.
func (Codec) WriteError(w http.ResponseWriter, err error) {
	writeGenerateContentError(w, err)
}

// EncodeRequest builds the Gemini generateContent request: a JSON body reader plus the
// application/json content type as an EncodedRequest. The RequestMode is intentionally
// ignored: Gemini's generateContent and streamGenerateContent bodies are identical —
// streaming is chosen by the transport via the route (`:streamGenerateContent?alt=sse`),
// not a body field — so Invoke and Stream produce the same bytes.
func (Codec) EncodeRequest(req inference.Request, _ codec.RequestMode) (codec.EncodedRequest, error) {
	body, err := EncodeRequest(req)
	if err != nil {
		return codec.EncodedRequest{}, err
	}
	h := http.Header{}
	h.Set("Content-Type", jsonbody.ContentType)
	return codec.EncodedRequest{Header: h, Body: bytes.NewReader(body)}, nil
}

// DecodeResponse parses a non-streaming Gemini generateContent body, delegating
// to the free DecodeResponse.
func (Codec) DecodeResponse(body []byte) (*inference.Response, error) {
	return DecodeResponse(body)
}

// DecodeEvent decodes one already-de-framed streamGenerateContent chunk (a
// partial GenerateContentResponse) into the chunk(s) it yields. Tolerance is
// scoped to SHAPE, not to validity: a well-formed chunk this dialect has no
// mapping for (no candidates, an unmodeled part type, a future top-level field)
// returns (nil, nil) so a new Gemini feature cannot break the stream, but
// malformed or truncated JSON is a *StreamEventDecodeError. Dropping it would
// let a half-delivered answer finish as a clean, complete-looking one.
// A frame reporting a blocked prompt (promptFeedback.blockReason) is the same
// kind of failure as that envelope: it is well-formed and candidate-less, so
// skipping it left the stream to end with no terminal event and be reported as
// truncated rather than refused. DecodeEvent is stateless; cross-event assembly
// happens downstream in the stream accumulator.
func (Codec) DecodeEvent(event []byte) ([]content.Chunk, error) {
	return decodeEvent(event)
}

// decodeEvent is the single per-event decoder. It maps candidates[0].content
// parts to chunks in order: a functionCall part -> ToolUseChunk, a thought-tagged
// text part -> ThinkingChunk, any other non-empty text part -> TextChunk. Empty
// and unknown parts are skipped, so an event with nothing to emit returns
// (nil, nil). JSON that does not parse at all is not "nothing to emit" — it is
// content of unknown size going missing — so it returns a
// *StreamEventDecodeError, matching openaiapi's decodeEvent. A frame carrying
// the `{"error":{...}}` envelope is likewise a failure, not a skip: it returns
// a *StreamAPIError so a mid-stream provider fault cannot pass as an
// uninteresting candidate-less chunk.
//
// INDEX SEMANTICS. Gemini puts NO index on the wire — a chunk is just an
// ordered `parts` array — so unlike Anthropic, Responses and Converse this
// codec's indexes are PER-CATEGORY positional counters it derives itself, not a
// shared content-block index. Tool-call Index and thought Index are therefore
// independent sequences that both start at 0, and a gap in either means
// nothing.
//
// Tool-call Index: Gemini sends a complete functionCall (full name + args) per
// part, so Index is the positional order of functionCall parts WITHIN this
// event. This is correct for the common case where a turn's (parallel) calls
// arrive together in one chunk; distinct calls split across separate chunks
// would collide on Index — a known stateless-decoder limitation. That same
// positional order also names an id-less call (toolCallID), so parallel calls
// the model sent without a wire `id` stay individually addressable once the
// accumulator folds them into ToolUseBlocks.
//
// Thought Index mirrors it, and the same limitation applies to this stateless
// entry point: it counts thought parts WITHIN this event. Unlike tool calls,
// though, thought blocks routinely span events — reasoning text streams in
// fragments, and Gemini seals a block by attaching its `thoughtSignature` to
// the block's LAST part. decodeEvent alone cannot see that boundary, so
// DecodeStream rebases these indexes across events (see thoughtBase in
// stream.go); the two compose as base + per-event position. A caller driving
// DecodeEvent directly, frame by frame, gets only the per-event half and must
// do its own rebasing.
func decodeEvent(payload []byte) ([]content.Chunk, error) {
	var ev GenerateContentResponse
	if err := json.Unmarshal(payload, &ev); err != nil {
		return nil, &StreamEventDecodeError{Err: err}
	}
	if ev.Error != nil {
		return nil, &StreamAPIError{Code: ev.Error.Code, Status: ev.Error.Status, Message: ev.Error.Message}
	}
	if blocked := promptBlockedError(ev); blocked != nil {
		return nil, blocked
	}
	if len(ev.Candidates) == 0 {
		return nil, nil
	}

	var out []content.Chunk
	fnIndex := 0
	thoughtIndex := 0
	for _, p := range ev.Candidates[0].Content.Parts {
		switch {
		case p.FunctionCall != nil:
			out = append(out, &content.ToolUseChunk{
				Index:               fnIndex,
				ID:                  toolCallID(p.FunctionCall.ID, fnIndex),
				Name:                p.FunctionCall.Name,
				InputJSON:           argsString(p.FunctionCall.Args),
				ProviderState:       providerStateFromThoughtSignature(p.ThoughtSignature),
				ProviderStateFormat: providerStateFormatFor(p.ThoughtSignature),
			})
			fnIndex++
		case p.Thought && (p.Text != "" || p.ThoughtSignature != ""):
			out = append(out, &content.ThinkingChunk{
				Index:               thoughtIndex,
				Thinking:            p.Text,
				ProviderState:       providerStateFromThoughtSignature(p.ThoughtSignature),
				ProviderStateFormat: providerStateFormatFor(p.ThoughtSignature),
			})
			// A thoughtSignature SEALS its block, so the next thought part in
			// this event belongs to a new one; an unsigned fragment continues
			// the block it is part of.
			if p.ThoughtSignature != "" {
				thoughtIndex++
			}
		case p.Text != "":
			out = append(out, &content.TextChunk{Text: p.Text})
		}
	}
	return out, nil
}

// argsString renders a streamed functionCall's arguments as the InputJSON string
// the accumulator concatenates. Gemini delivers the complete args object in one
// chunk, so this is the full JSON; an empty payload becomes "{}".
func argsString(raw json.RawMessage) string {
	if len(raw) == 0 {
		return "{}"
	}
	return string(raw)
}
