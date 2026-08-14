package geminiapi

import (
	"encoding/base64"
	"strconv"
	"strings"
)

// syntheticToolCallIDPrefix marks a tool-call identity this codec invented
// rather than read off the wire. Gemini's FunctionCall.id is Optional ("Optional.
// Unique identifier of the function call. If populated, the client to execute
// the `function_call` and return the response with the matching `id`." — the
// v1beta discovery document's FunctionCall schema), and the Developer API
// routinely omits it, parallel calls included. The provider-neutral vocabulary,
// though, addresses a tool result by id alone (content.ToolResultMessage carries
// only ToolUseID), so two id-less parallel calls would both be addressed as ""
// and one call's output would silently answer the other. Decoding therefore
// gives each id-less call a per-turn ordinal under this prefix so the pair stays
// addressable in-process, and every wire-emitting site strips it again through
// wireToolCallID — a fabricated id must never be echoed to Gemini, which pairs a
// functionResponse on the Required `name` field, not on `id`.
const syntheticToolCallIDPrefix = "gemini-positional-call-"
const escapedWireToolCallIDPrefix = "gemini-wire-call-id-"

// toolCallID is the identity to carry for a decoded functionCall part: the
// model's own id when it supplied one, else a synthetic per-turn ordinal.
func toolCallID(wireID string, ordinal int) string {
	if wireID == "" {
		return syntheticToolCallIDPrefix + strconv.Itoa(ordinal)
	}
	if strings.HasPrefix(wireID, syntheticToolCallIDPrefix) || strings.HasPrefix(wireID, escapedWireToolCallIDPrefix) {
		return escapedWireToolCallIDPrefix + base64.RawURLEncoding.EncodeToString([]byte(wireID))
	}
	return wireID
}

// wireToolCallID is the id to write onto the Gemini wire: empty (so the
// omitempty `id` key disappears entirely) for a synthetic identity, otherwise
// the id unchanged.
func wireToolCallID(id string) string {
	if isSyntheticToolCallID(id) {
		return ""
	}
	if strings.HasPrefix(id, escapedWireToolCallIDPrefix) {
		decoded, err := base64.RawURLEncoding.DecodeString(strings.TrimPrefix(id, escapedWireToolCallIDPrefix))
		if err == nil {
			return string(decoded)
		}
	}
	return id
}

func isSyntheticToolCallID(id string) bool {
	suffix := strings.TrimPrefix(id, syntheticToolCallIDPrefix)
	if suffix == id || suffix == "" {
		return false
	}
	ordinal, err := strconv.Atoi(suffix)
	return err == nil && ordinal >= 0 && strconv.Itoa(ordinal) == suffix
}

func rebaseSyntheticToolCallID(id string, ordinal int) string {
	if !isSyntheticToolCallID(id) {
		return id
	}
	return syntheticToolCallIDPrefix + strconv.Itoa(ordinal)
}

// toolCallIndex resolves a ToolResultMessage back to the name of the
// functionCall it answers — the field Gemini actually matches on, since
// FunctionResponse.name is Required while both FunctionCall.id and
// FunctionResponse.id are Optional. byID serves calls that carry an id,
// including the synthetic ordinal the decode direction assigns. positional is
// the ordered queue of calls that still reached the encoder with no id at all
// (a cross-dialect thread, or a transcript recorded before the ordinal
// existed); their results are matched head-first in wire order, because "" is
// not a usable map key: keying every id-less call on it let the last call
// written claim every result.
type toolCallIndex struct {
	byID       map[string]string
	positional []string
}

// record notes a tool call's name under the identity a later result will use.
func (x *toolCallIndex) record(id, name string) {
	if id == "" {
		x.positional = append(x.positional, name)
		return
	}
	if x.byID == nil {
		x.byID = make(map[string]string)
	}
	x.byID[id] = name
}

// resolve names the call a tool result answers, consuming the head of the
// positional queue for an id-less result. An unrecognized id (or an exhausted
// queue) yields "": the paired call was not in this thread.
func (x *toolCallIndex) resolve(id string) string {
	if id != "" {
		return x.byID[id]
	}
	if len(x.positional) == 0 {
		return ""
	}
	name := x.positional[0]
	x.positional = x.positional[1:]
	return name
}
