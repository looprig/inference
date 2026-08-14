package openairesponses

import (
	crand "crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	stream "github.com/looprig/inference/stream"
	"github.com/looprig/inference/wire/jsonbody"
)

// encodeWireResponse is the server-ENCODE-direction wire form of a
// non-streaming Responses API response, and — through sseResponseEnvelope
// (server_stream.go) — of the whole Response object carried by every
// response.created / .in_progress / .completed / .incomplete / .failed stream
// frame. It deliberately does NOT reuse wireResponse (types.go): that backs the
// existing client-DECODE direction and carries Usage as *wireUsage, whose
// usagenorm.Count fields hold only an unexported raw-JSON capture with no
// MarshalJSON — they can decode a real count but cannot encode one. Output DOES
// reuse wireItem, which is fully marshal-safe.
//
// The struct holds only what the gateway KNOWS. Response.required lists
// fourteen members, most of which echo request parameters this encoder is never
// given; MarshalJSON below supplies them from responseWireBody, so that a caller
// assembling an encodeWireResponse literal (the streaming path does, four
// times) cannot produce a body missing half its required keys. It carries no
// struct tags for exactly that reason: responseWireBody owns the wire shape,
// and a second set here would be an unenforced duplicate of it.
type encodeWireResponse struct {
	ID     string
	Status string
	Model  string
	// CreatedAt is the response's unix creation time. Zero means "not stamped
	// by the caller", and MarshalJSON stamps the marshal time instead: the
	// member is required and non-nullable, so there is no way to say unknown.
	//
	// A CALLER THAT BUILDS MORE THAN ONE OF THESE FOR THE SAME RESPONSE MUST
	// STAMP IT. The streaming path used to leave it zero on each of the
	// literals it builds, so response.created and response.completed reported
	// different creation times for one response whenever the generation crossed
	// a second boundary; serverStreamEncoder.createdAt (server_stream.go) now
	// stamps every frame from one value taken when the stream opened. The
	// marshal-time fallback remains only for the single-shot non-streaming
	// body, where there is exactly one marshal and the two are the same thing.
	CreatedAt         int64
	Output            []wireItem
	Usage             *encodeWireUsage
	IncompleteDetails *wireIncompleteDetails
	Error             *wireResponseError
}

// responseWireBody is the marshal shape of a Response. Response.required is
// [id, object, created_at, error, incomplete_details, instructions, model,
// tools, output, parallel_tool_calls, metadata, tool_choice, temperature,
// top_p], so none of those keys may carry omitempty. `usage` and `status` are
// NOT required, and keep theirs.
//
// Four of the required members are nullable, and the gateway genuinely knows
// none of them, so all four travel as an explicit null: instructions, metadata,
// temperature and top_p are request parameters, and this encoder is handed only
// an inference.Response. Null says "not reported"; a fabricated 1.0 temperature
// would be a false statement about how the text was sampled.
//
// Three required members are NOT nullable, so null is not available and a value
// must be chosen. Each is the API's own documented default, and each is a
// KNOWN IMPRECISION rather than a fact, recorded here so nobody reads it as
// one:
//
//   - tools is an array with no null member; the gateway does not carry the
//     request's tool declarations into the encoder, so it emits [].
//   - tool_choice's union (ToolChoiceOptions | ToolChoiceAllowed | ...) has no
//     null member either; "auto" is the API default.
//   - parallel_tool_calls is a bare boolean; true is the API default.
//
// Threading the decoded request's tools/tool_choice through the gateway to this
// encoder is what would make the three exact.
type responseWireBody struct {
	ID                string                 `json:"id"`
	Object            string                 `json:"object"`
	CreatedAt         int64                  `json:"created_at"`
	Status            string                 `json:"status,omitempty"`
	Model             string                 `json:"model"`
	Output            []wireItem             `json:"output"`
	Error             *wireResponseError     `json:"error"`
	IncompleteDetails *wireIncompleteDetails `json:"incomplete_details"`
	Instructions      *string                `json:"instructions"`
	Tools             []json.RawMessage      `json:"tools"`
	ToolChoice        string                 `json:"tool_choice"`
	ParallelToolCalls bool                   `json:"parallel_tool_calls"`
	Metadata          *json.RawMessage       `json:"metadata"`
	Temperature       *float64               `json:"temperature"`
	TopP              *float64               `json:"top_p"`
	Usage             *encodeWireUsage       `json:"usage,omitempty"`
}

// objectResponse is the const value of Response.object.
const objectResponse = "response"

// defaultToolChoice and defaultParallelToolCalls are the Responses API's own
// defaults, emitted because their schema members admit no null. See
// responseWireBody.
const (
	defaultToolChoice        = "auto"
	defaultParallelToolCalls = true
)

func (r encodeWireResponse) MarshalJSON() ([]byte, error) {
	createdAt := r.CreatedAt
	if createdAt == 0 {
		createdAt = time.Now().Unix()
	}
	// `output` is an array-typed required member: a nil slice marshals as null,
	// which the schema rejects. This is the same defect the streaming path was
	// repaired for, and it survived here because the two paths built the value
	// separately; normalizing inside MarshalJSON means neither can regress.
	output := r.Output
	if output == nil {
		output = []wireItem{}
	}
	return json.Marshal(responseWireBody{
		ID:                r.ID,
		Object:            objectResponse,
		CreatedAt:         createdAt,
		Status:            r.Status,
		Model:             r.Model,
		Output:            output,
		Error:             responseError(r.Error),
		IncompleteDetails: r.IncompleteDetails,
		Tools:             []json.RawMessage{},
		ToolChoice:        defaultToolChoice,
		ParallelToolCalls: defaultParallelToolCalls,
		Usage:             r.Usage,
	})
}

// ResponseErrorCode members. The enum is short and mostly image-specific; these
// two are the only ones a gateway failure can honestly claim.
const (
	responseErrorCodeServerError       = "server_error"
	responseErrorCodeRateLimitExceeded = "rate_limit_exceeded"
	// classifiedRateLimit is the envelope code classifyError produces for a 429
	// (codeForStatus, server_error.go), the one classification with an exact
	// ResponseErrorCode counterpart.
	classifiedRateLimit = "rate_limit_error"
)

// responseError normalizes an error into a legal Response.error.
//
// Response.error is a DIFFERENT shape from the top-level error envelope
// writeResponsesError emits, even though both reuse wireResponseError: the
// envelope's `code` is free-form, while Response.error.code is the closed
// ResponseErrorCode enum. classifyError produces envelope codes
// ("api_error", "invalid_request_error", "authentication_error", …), none of
// which are enum members, so a response.failed frame carrying one is rejected
// by the schema — the anyOf falls through to `null` and reports "got object,
// want null".
//
// The classification is deliberately coarse rather than absent: `code` is a
// bucket and `message` carries the specifics, so mapping to server_error loses
// nothing a consumer could act on, whereas dropping the whole object (the other
// way to satisfy the schema, since error is nullable) would discard the failure
// itself.
func responseError(e *wireResponseError) *wireResponseError {
	if e == nil {
		return nil
	}
	code := responseErrorCodeServerError
	if e.Code == classifiedRateLimit || e.Code == responseErrorCodeRateLimitExceeded {
		code = responseErrorCodeRateLimitExceeded
	}
	return &wireResponseError{Code: code, Message: e.Message}
}

// encodeWireUsage is the encode-direction counterpart to wireUsage: plain
// exported uint64 fields that json.Marshal can actually serialize.
type encodeWireUsage struct {
	InputTokens         uint64                   `json:"input_tokens"`
	OutputTokens        uint64                   `json:"output_tokens"`
	TotalTokens         uint64                   `json:"total_tokens"`
	InputTokensDetails  encodeInputTokensDetail  `json:"input_tokens_details"`
	OutputTokensDetails encodeOutputTokensDetail `json:"output_tokens_details"`
}

// encodeInputTokensDetail is ResponseUsage.input_tokens_details, required =
// [cached_tokens, cache_write_tokens]. Both are plain integers with no null
// member, and — unlike the Response envelope's request echoes — both are values
// the neutral Usage actually carries, so neither is a guess.
type encodeInputTokensDetail struct {
	CachedTokens     uint64 `json:"cached_tokens"`
	CacheWriteTokens uint64 `json:"cache_write_tokens"`
}

type encodeOutputTokensDetail struct {
	ReasoningTokens uint64 `json:"reasoning_tokens"`
}

// writeResponsesResponse encodes a complete inference.Response as the native
// Responses API non-streaming response and writes a 200 with it.
func writeResponsesResponse(w http.ResponseWriter, resp *inference.Response) error {
	wire, err := buildWireResponse(resp)
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

func buildWireResponse(resp *inference.Response) (encodeWireResponse, error) {
	if resp == nil {
		resp = &inference.Response{}
	}
	ids := newToolIDGenerator()

	// Never nil: `output` is a required array-typed member, and a nil slice
	// marshals as null. MarshalJSON normalizes it too — belt and braces, since
	// the streaming path builds its own literals — but starting non-nil keeps
	// the invariant visible at the place the value is produced.
	output := []wireItem{}
	if resp.Message != nil {
		items, err := blocksToItems(resp.Message.Blocks, ids)
		if err != nil {
			return encodeWireResponse{}, err
		}
		if items != nil {
			output = items
		}
	}

	status, incomplete := statusForFinishReason(resp.FinishReason)

	return encodeWireResponse{
		ID:                "resp_" + randomHex(12),
		Status:            status,
		Model:             resp.Model,
		CreatedAt:         time.Now().Unix(),
		Output:            output,
		Usage:             encodeUsage(resp.Usage),
		IncompleteDetails: incomplete,
	}, nil
}

// statusForFinishReason maps the neutral stream.FinishReason to Responses'
// status (plus incomplete_details when applicable), inverting
// deriveFinishReason (decode.go)'s status/incomplete_details half. There is
// no wire status corresponding to FinishReasonContentFilter or
// FinishReasonUnknown, so both — like FinishReasonStop and
// FinishReasonToolUse — encode as "completed": Responses has no per-status
// content-filter distinction the way an Anthropic stop_reason does, and any
// response reaching this encoder (as opposed to WriteError/Fail) already
// succeeded, so "completed" is the least presumptive terminal status.
func statusForFinishReason(r stream.FinishReason) (string, *wireIncompleteDetails) {
	if r == stream.FinishReasonLength {
		return statusIncomplete, &wireIncompleteDetails{Reason: incompleteReasonMaxOutputTokens}
	}
	return statusCompleted, nil
}

// encodeUsage inverts normalizeUsage (decode.go): input_tokens is
// reconstructed as the gross prompt total (neutral InputTokens plus
// CacheReadTokens), matching how Responses reports it, and
// CacheCreationTokens — which Responses has no wire field for, since its
// caching is fully automatic — is folded into the gross total rather than
// silently dropped from the reported count.
func encodeUsage(u *content.Usage) *encodeWireUsage {
	if u == nil {
		return nil
	}
	grossInput := uint64(u.InputTokens) + uint64(u.CacheReadTokens) + uint64(u.CacheCreationTokens)
	total := grossInput + uint64(u.OutputTokens)
	return &encodeWireUsage{
		InputTokens:  grossInput,
		OutputTokens: uint64(u.OutputTokens),
		TotalTokens:  total,
		InputTokensDetails: encodeInputTokensDetail{
			CachedTokens:     uint64(u.CacheReadTokens),
			CacheWriteTokens: uint64(u.CacheCreationTokens),
		},
		OutputTokensDetails: encodeOutputTokensDetail{ReasoningTokens: uint64(u.ReasoningTokens)},
	}
}

// --- id synthesis -------------------------------------------------------

// newToolIDGenerator returns a closure that yields fresh, collision-resistant,
// call-scoped synthetic ids (used for both function_call ids and response/
// item ids). It combines a per-call random prefix (guarding against
// collision across different responses/streams) with a monotonic counter
// (guarding against collision within one response/stream, even with zero
// entropy available). A cross-dialect upstream target might not supply a
// tool-call id at all.
func newToolIDGenerator() func() string {
	prefix := randomHex(6)
	counter := 0
	return func() string {
		counter++
		return fmt.Sprintf("fc_gw_%s_%d", prefix, counter)
	}
}

// randomHex returns n random bytes hex-encoded. On the practically-impossible
// event crypto/rand fails, it falls back to a fixed all-zero value rather than
// panicking or failing the response.
func randomHex(n int) string {
	b := make([]byte, n)
	if _, err := crand.Read(b); err != nil {
		return strings.Repeat("0", n*2)
	}
	return hex.EncodeToString(b)
}
