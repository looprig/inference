package anthropicapi

import (
	crand "crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	stream "github.com/looprig/inference/stream"
	"github.com/looprig/inference/wire/jsonbody"
)

// wireMessageResponse is the server-ENCODE-direction wire form of a
// non-streaming Messages API response. It deliberately does NOT reuse
// messageResponse/messageUsage (types.go): those back the existing
// client-DECODE direction and carry Usage as *messageUsage, whose
// usagenorm.Count fields hold only an unexported raw-JSON capture with no
// MarshalJSON — they can decode a real count but cannot encode one. Content
// blocks DO reuse anthropicBlock (through responseBlock, below), which is fully
// marshal-safe: encode.go already produces request blocks with it.
//
// Message.required is [id, type, role, content, model, stop_reason,
// stop_sequence, stop_details, usage, container], so NOT ONE of these fields
// may carry omitempty. The ones the gateway has no value for are nullable
// pointers, which marshal as an explicit null instead of vanishing. This is the
// response-direction instance of the required-field rule that produced the
// illegal thinking/redacted_thinking blocks: a required key erased by Go's zero
// value is a body the format's own schema rejects — and a gateway is a provider
// as far as its client is concerned.
//
// It is also the `message` of a message_start stream event (server_stream.go),
// which is a full Message on the wire — reused rather than re-declared so a
// required member can never be present on one path and missing on the other.
// StopReason is a pointer for that second use: at message_start no turn has
// ended yet, and stop_reason is nullable precisely to say so.
type wireMessageResponse struct {
	ID           string           `json:"id"`
	Type         string           `json:"type"`
	Role         string           `json:"role"`
	Model        string           `json:"model"`
	Content      []responseBlock  `json:"content"`
	StopReason   *string          `json:"stop_reason"`
	StopSequence *string          `json:"stop_sequence"`
	StopDetails  *json.RawMessage `json:"stop_details"`
	Container    *json.RawMessage `json:"container"`
	Usage        wireUsage        `json:"usage"`
}

// wireUsage is the encode-direction counterpart to messageUsage: plain
// exported fields that json.Marshal can actually serialize. Usage.required
// lists all nine of its properties, so none may be omitted; the five the
// gateway cannot know are nullable and travel as null rather than as a
// fabricated zero, which would be a false accounting claim.
type wireUsage struct {
	InputTokens         uint64               `json:"input_tokens"`
	OutputTokens        uint64               `json:"output_tokens"`
	CacheReadTokens     uint64               `json:"cache_read_input_tokens"`
	CacheCreationTokens uint64               `json:"cache_creation_input_tokens"`
	CacheCreation       *json.RawMessage     `json:"cache_creation"`
	InferenceGeo        *string              `json:"inference_geo"`
	OutputTokensDetails *wireOutputTokensDet `json:"output_tokens_details"`
	ServerToolUse       *json.RawMessage     `json:"server_tool_use"`
	ServiceTier         *string              `json:"service_tier"`
}

// wireOutputTokensDet is Usage.output_tokens_details, whose sole member
// thinking_tokens is required when the object is present.
type wireOutputTokensDet struct {
	ThinkingTokens uint64 `json:"thinking_tokens"`
}

// responseBlock marshals one content block in the RESPONSE direction. It is the
// seam for the response-only required members the shared request DTO cannot
// carry: the response schema requires two members the request schema merely
// permits — ResponseTextBlock.citations (nullable) and
// ResponseToolUseBlock.caller — and every other block variant CLOSES
// additionalProperties, so a shared struct that always emitted them would make
// the request direction illegal instead.
//
// The two directions were widened in the same change, and must stay that way:
// anthropicBlock (types.go) and decodeRequestBlock (server_decode.go) accept
// both members on the way in, so a client replaying an assistant turn served
// here — or served by real Anthropic, which always carries them — is decoded
// rather than answered with malformed_body/unknown field.
//
// The reasoning variants (thinking, redacted_thinking) declare no extra
// response members and keep reaching the wire through anthropicBlock's own
// per-variant marshallers.
type responseBlock struct {
	block anthropicBlock
}

// responseTextWireBlock is the marshal shape of ResponseTextBlock, required =
// [citations, text, type]. Neither key may carry omitempty: an empty assistant
// text block still owes `text`, and `citations` is required even when there are
// none. Citations is a nil json.RawMessage, which marshals as an explicit
// `null` — the schema's own "no citations" value. The gateway does not
// synthesise an empty array: `[]` would claim the model was asked to cite and
// found nothing, while null says the field does not apply.
type responseTextWireBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text"`
	Citations json.RawMessage `json:"citations"`
}

// responseToolUseWireBlock is the marshal shape of ResponseToolUseBlock,
// required = [caller, id, input, name, type].
//
// `caller` is NOT nullable — its union is DirectCaller | ServerToolCaller |
// ServerToolCaller_20260120 — so unlike citations there is no "unknown" value
// to emit, and one of the three must be chosen. DirectCaller is not a guess:
// the server-tool members describe a tool_use minted by Anthropic's own hosted
// tools, which arrive as a `server_tool_use` block this codec never produces.
// Every tool_use the gateway serves is a call it expects its CLIENT to execute,
// which is precisely what `{"type":"direct"}` states.
type responseToolUseWireBlock struct {
	Type   string           `json:"type"`
	ID     string           `json:"id"`
	Name   string           `json:"name"`
	Input  json.RawMessage  `json:"input"`
	Caller wireDirectCaller `json:"caller"`
}

// wireDirectCaller is DirectCaller, required = [type], additionalProperties =
// false.
type wireDirectCaller struct {
	Type string `json:"type"`
}

func (b responseBlock) MarshalJSON() ([]byte, error) {
	switch b.block.Type {
	case blockTypeText:
		return json.Marshal(responseTextWireBlock{Type: b.block.Type, Text: b.block.Text})
	case blockTypeToolUse:
		return json.Marshal(responseToolUseWireBlock{
			Type:   b.block.Type,
			ID:     b.block.ID,
			Name:   b.block.Name,
			Input:  inputOrEmpty(b.block.Input),
			Caller: wireDirectCaller{Type: callerTypeDirect},
		})
	}
	return json.Marshal(b.block)
}

// writeMessageResponse encodes a complete inference.Response as the native
// Anthropic Messages API non-streaming response and writes a 200 with it.
func writeMessageResponse(w http.ResponseWriter, resp *inference.Response) error {
	wire, err := buildWireMessageResponse(resp)
	if err != nil {
		return err
	}
	body, err := json.Marshal(wire)
	if err != nil {
		return err
	}
	w.Header().Set("Content-Type", jsonbody.ContentType)
	w.WriteHeader(http.StatusOK)
	_, err = w.Write(body)
	return err
}

func buildWireMessageResponse(resp *inference.Response) (wireMessageResponse, error) {
	if resp == nil {
		resp = &inference.Response{}
	}
	ids := newToolIDGenerator()

	// Never nil: `content` is a required array-typed field, and a nil slice
	// marshals as null, which the response schema rejects.
	blocks := []responseBlock{}
	var usage *content.Usage
	if resp.Message != nil {
		eb, err := encodeResponseBlocks(resp.Message.Blocks, ids)
		if err != nil {
			return wireMessageResponse{}, err
		}
		blocks = eb
	}
	usage = resp.Usage

	stopReason := encodeFinishReason(resp.FinishReason)
	return wireMessageResponse{
		ID:         "msg_" + randomHex(12),
		Type:       "message",
		Role:       roleAssistant,
		Model:      resp.Model,
		Content:    blocks,
		StopReason: &stopReason,
		Usage:      encodeUsage(usage),
	}, nil
}

// encodeResponseBlocks maps neutral response blocks to their Anthropic wire
// form. It mirrors encodeBlock (encode.go, the outbound-request direction) but
// additionally synthesizes a tool_use id when the neutral block arrived with
// none: Anthropic tool_use blocks always carry a provider-issued id, but a
// same-shape response assembled from a cross-dialect target might not.
//
// It also admits STRICTLY LESS than encodeBlock does, and deliberately. The
// request ContentBlock union has an image member; the RESPONSE ContentBlock
// union — text | thinking | redacted_thinking | tool_use | server_tool_use and
// the provider-hosted tool-result blocks — has none, because an image is an
// input to a model, never something a model emits. An upstream target that
// returns an ImageBlock therefore has no legal Anthropic response form at all,
// so it fails closed here (see the *content.ImageBlock case in
// encodeResponseBlock) instead of serving an `image` block no Anthropic client
// can parse.
func encodeResponseBlocks(blocks []content.Block, ids func() string) ([]responseBlock, error) {
	out := make([]responseBlock, 0, len(blocks))
	for _, b := range blocks {
		eb, err := encodeResponseBlock(b, ids)
		if err != nil {
			return nil, err
		}
		out = append(out, responseBlock{block: eb})
	}
	return out, nil
}

func encodeResponseBlock(b content.Block, ids func() string) (anthropicBlock, error) {
	switch b := b.(type) {
	case *content.TextBlock:
		return anthropicBlock{Type: blockTypeText, Text: b.Text}, nil
	case *content.ThinkingBlock:
		// A redacted block's opaque payload is the ONLY copy of that
		// reasoning: nothing else in the response carries it, and the gateway
		// is where it would be lost. Serving it as a `thinking` block would
		// emit {"type":"thinking","thinking":""} and destroy it, so the
		// redacted variant is detected here exactly as encodeBlock (encode.go,
		// the outbound-request direction) and WriteChunk (server_stream.go, the
		// streaming counterpart of this function) detect it — the three must
		// agree, or the same response serves differently depending on whether
		// the client asked to stream.
		if b.ReplayableAs(providerStateFormatAnthropicRedacted) {
			data, err := opaqueRedactedToWire(b.ProviderState)
			if err != nil {
				return anthropicBlock{}, err
			}
			return anthropicBlock{Type: blockTypeRedactedThinking, Data: data}, nil
		}
		// Serving another dialect's signature is strictly worse than refusing
		// it. The client stores what we serve and replays it next turn, so the
		// rejection would surface one turn later, against a different
		// component, with nothing left pointing at the gateway that mislabelled
		// it. Same reasoning as the redacted branch above: all three egress
		// paths must agree.
		signature, err := replayableSignature(b)
		if err != nil {
			return anthropicBlock{}, err
		}
		return anthropicBlock{Type: blockTypeThinking, Thinking: b.Thinking, Signature: signature}, nil
	case *content.ToolUseBlock:
		id := b.ID
		if id == "" {
			id = ids()
		}
		return anthropicBlock{Type: blockTypeToolUse, ID: id, Name: b.Name, Input: inputOrEmpty(b.Input)}, nil
	case *content.ImageBlock:
		// Unrepresentable, not merely unimplemented: the response ContentBlock
		// union has no image member. Emitting one (as this function did) puts a
		// block on the wire that the format's own response schema rejects and
		// that a real Anthropic client's decoder will not recognize, so the
		// caller learns about it here rather than downstream.
		return anthropicBlock{}, &UnsupportedBlockError{Block: fmt.Sprintf("%T", b)}
	default:
		return anthropicBlock{}, &UnsupportedBlockError{Block: fmt.Sprintf("%T", b)}
	}
}

// encodeFinishReason maps the neutral stream.FinishReason to its Anthropic wire
// stop_reason, inverting mapFinishReason (stream.go). FinishReasonUnknown (and
// any value mapFinishReason itself would never have produced) falls back to
// "end_turn": Anthropic responses always carry a non-empty stop_reason, and
// "end_turn" is its least presumptive value.
func encodeFinishReason(r stream.FinishReason) string {
	switch r {
	case stream.FinishReasonStop:
		return "end_turn"
	case stream.FinishReasonLength:
		return "max_tokens"
	case stream.FinishReasonToolUse:
		return "tool_use"
	case stream.FinishReasonContentFilter:
		return "refusal"
	default:
		return "end_turn"
	}
}

// encodeUsage builds the required `usage` object. A nil neutral Usage still
// produces one — the field is required, and an absent target-reported count is
// reported as zero rather than dropping the whole object and making the body
// illegal. The five members the gateway has no value for stay null: null is
// "unknown", whereas a zeroed CacheCreation or ServerToolUsage would be a
// positive claim that no caching and no server-tool calls occurred.
func encodeUsage(u *content.Usage) wireUsage {
	if u == nil {
		return wireUsage{}
	}
	return wireUsage{
		InputTokens:         uint64(u.InputTokens),
		OutputTokens:        uint64(u.OutputTokens),
		CacheReadTokens:     uint64(u.CacheReadTokens),
		CacheCreationTokens: uint64(u.CacheCreationTokens),
		OutputTokensDetails: &wireOutputTokensDet{ThinkingTokens: uint64(u.ReasoningTokens)},
	}
}

// --- count_tokens response encoding -----------------------------------------

// countTokensResponse is the native `{"input_tokens": N}` body Anthropic's
// count_tokens endpoint returns.
type countTokensResponse struct {
	InputTokens int `json:"input_tokens"`
}

// WriteCountTokensResponse writes Anthropic's count_tokens response shape given
// an already-computed token count. It does not compute the count itself: the
// gateway resolves the target model and calls a contextcount.ContextCounter,
// then calls this helper with the result.
func WriteCountTokensResponse(w http.ResponseWriter, inputTokens int) error {
	body, err := json.Marshal(countTokensResponse{InputTokens: inputTokens})
	if err != nil {
		return err
	}
	w.Header().Set("Content-Type", jsonbody.ContentType)
	w.WriteHeader(http.StatusOK)
	_, err = w.Write(body)
	return err
}

// --- id synthesis -------------------------------------------------------

// newToolIDGenerator returns a closure that yields fresh, collision-resistant,
// call-scoped synthetic tool_use ids. It combines a per-call random prefix
// (guarding against collision across different responses/streams) with a
// monotonic counter (guarding against collision within one response/stream,
// even with zero entropy available). Anthropic requires every tool_use block to
// carry a non-empty id; a cross-dialect upstream target might not supply one.
func newToolIDGenerator() func() string {
	prefix := randomHex(6)
	counter := 0
	return func() string {
		counter++
		return fmt.Sprintf("toolu_gw_%s_%d", prefix, counter)
	}
}

// randomHex returns n random bytes hex-encoded. On the practically-impossible
// event crypto/rand fails, it falls back to a fixed all-zero value rather than
// panicking or failing the response — a ropey synthetic id is far less harmful
// than an encoder crash.
func randomHex(n int) string {
	b := make([]byte, n)
	if _, err := crand.Read(b); err != nil {
		return strings.Repeat("0", n*2)
	}
	return hex.EncodeToString(b)
}
