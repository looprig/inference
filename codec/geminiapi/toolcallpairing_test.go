package geminiapi_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/geminiapi"
	model "github.com/looprig/inference/model"
)

func TestDecodeEncodeResponse_ProviderIDMatchingSyntheticNamespaceIsPreserved(t *testing.T) {
	t.Parallel()

	resp, err := geminiapi.DecodeResponse([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"id":"gemini-positional-call-0","name":"run","args":{}}}]},"finishReason":"STOP"}]}`))
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	body, err := geminiapi.EncodeRequest(inference.Request{
		Model:    model.Model{Name: "m"},
		Messages: content.AgenticMessages{resp.Message},
	})
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	if !strings.Contains(string(body), `"id":"gemini-positional-call-0"`) {
		t.Fatalf("provider-issued id was erased on replay: %s", body)
	}
}

// functionResponsesOf collects every functionResponse part of an encoded
// request body in wire order, as (name, id, result) triples. Ids are reported
// as "" when the key is absent, which is what an id-less pairing must produce.
func functionResponsesOf(t *testing.T, body []byte) []struct{ Name, ID, Result string } {
	t.Helper()
	var out []struct{ Name, ID, Result string }
	for _, entry := range contentsFromRaw(t, mustDecode(t, body)) {
		for _, part := range partsOf(t, entry) {
			frRaw, ok := part["functionResponse"]
			if !ok {
				continue
			}
			var fr map[string]json.RawMessage
			if err := json.Unmarshal(frRaw, &fr); err != nil {
				t.Fatalf("unmarshal functionResponse: %v", err)
			}
			var response map[string]string
			if err := json.Unmarshal(fr["response"], &response); err != nil {
				t.Fatalf("unmarshal functionResponse.response: %v", err)
			}
			got := struct{ Name, ID, Result string }{Name: strField(t, fr, "name"), Result: response["result"]}
			if _, has := fr["id"]; has {
				got.ID = strField(t, fr, "id")
			}
			out = append(out, got)
		}
	}
	return out
}

// functionCallIDsPresent reports, in wire order, whether each functionCall part
// of an encoded body carries an `id` key at all.
func functionCallIDsPresent(t *testing.T, body []byte) []bool {
	t.Helper()
	var out []bool
	for _, entry := range contentsFromRaw(t, mustDecode(t, body)) {
		for _, part := range partsOf(t, entry) {
			fcRaw, ok := part["functionCall"]
			if !ok {
				continue
			}
			var fc map[string]json.RawMessage
			if err := json.Unmarshal(fcRaw, &fc); err != nil {
				t.Fatalf("unmarshal functionCall: %v", err)
			}
			_, has := fc["id"]
			out = append(out, has)
		}
	}
	return out
}

// idLessParallelCalls is the real Developer API shape this pairing has to
// survive: two parallel functionCall parts with no `id` at all. The v1beta
// discovery document's FunctionCall schema documents `id` as "Optional.
// Unique identifier of the function call. If populated, the client to execute
// the `function_call` and return the response with the matching `id`." — an
// absent id is normal, not degenerate, and `name` is the Required field
// Gemini actually matches a functionResponse on.
const idLessParallelCalls = `{"candidates":[{"content":{"role":"model","parts":[` +
	`{"functionCall":{"name":"get_weather","args":{"city":"boston"}}},` +
	`{"functionCall":{"name":"get_time","args":{"zone":"utc"}}}` +
	`]},"finishReason":"STOP"}]}`

// TestDecodeResponse_IDLessParallelCallsGetDistinctIdentities proves the
// client-decode direction hands the harness two DISTINGUISHABLE tool calls.
// Both wire calls carry no id, so decoding them to two ToolUseBlocks that
// both have ID "" makes the pair unaddressable: the harness cannot say which
// result answers which call.
func TestDecodeResponse_IDLessParallelCallsGetDistinctIdentities(t *testing.T) {
	t.Parallel()

	resp, err := geminiapi.DecodeResponse([]byte(idLessParallelCalls))
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	uses := toolUsesOf(t, resp.Message.Blocks)
	if len(uses) != 2 {
		t.Fatalf("tool use blocks = %d, want 2", len(uses))
	}
	if uses[0].ID == "" || uses[1].ID == "" {
		t.Fatalf("id-less parallel calls decoded to empty ids (%q, %q); the pair is unaddressable", uses[0].ID, uses[1].ID)
	}
	if uses[0].ID == uses[1].ID {
		t.Fatalf("parallel calls share id %q; the pair is unaddressable", uses[0].ID)
	}
}

// TestEncodeRequest_IDLessParallelToolResultsKeepTheirNames is the end-to-end
// reproduction of the cross-attribution defect: decode a real id-less parallel
// call turn, answer both calls, re-encode, and require each functionResponse
// to name the function it actually answers. Keying the call-name lookup on the
// tool-use id alone made both calls key "" (last write wins), so get_time's
// name claimed get_weather's output and get_weather never got a response at
// all.
func TestEncodeRequest_IDLessParallelToolResultsKeepTheirNames(t *testing.T) {
	t.Parallel()

	resp, err := geminiapi.DecodeResponse([]byte(idLessParallelCalls))
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	uses := toolUsesOf(t, resp.Message.Blocks)
	if len(uses) != 2 {
		t.Fatalf("tool use blocks = %d, want 2", len(uses))
	}

	body, err := geminiapi.EncodeRequest(inference.Request{
		Model: model.Model{Name: "m"},
		Messages: content.AgenticMessages{
			userMsg(textBlock("weather and time please")),
			resp.Message,
			toolMsg(uses[0].ID, textBlock("sunny")),
			toolMsg(uses[1].ID, textBlock("12:00")),
		},
	})
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}

	got := functionResponsesOf(t, body)
	if len(got) != 2 {
		t.Fatalf("functionResponse parts = %d, want 2", len(got))
	}
	if got[0].Name != "get_weather" || got[0].Result != "sunny" {
		t.Errorf("first functionResponse = name %q result %q, want get_weather/sunny", got[0].Name, got[0].Result)
	}
	if got[1].Name != "get_time" || got[1].Result != "12:00" {
		t.Errorf("second functionResponse = name %q result %q, want get_time/12:00", got[1].Name, got[1].Result)
	}
}

// TestEncodeRequest_SynthesizedIDsNeverReachTheWire guards the other half of
// the fix: the identity this codec invents for an id-less call is INTERNAL.
// FunctionCall.id is Optional on the wire and Gemini pairs a functionResponse
// by `name`, so echoing a fabricated id back would assert an identity the
// model never issued.
func TestEncodeRequest_SynthesizedIDsNeverReachTheWire(t *testing.T) {
	t.Parallel()

	resp, err := geminiapi.DecodeResponse([]byte(idLessParallelCalls))
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	uses := toolUsesOf(t, resp.Message.Blocks)

	body, err := geminiapi.EncodeRequest(inference.Request{
		Model: model.Model{Name: "m"},
		Messages: content.AgenticMessages{
			resp.Message,
			toolMsg(uses[0].ID, textBlock("sunny")),
			toolMsg(uses[1].ID, textBlock("12:00")),
		},
	})
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}

	for i, present := range functionCallIDsPresent(t, body) {
		if present {
			t.Errorf("functionCall[%d] carries a fabricated id; wire id must stay absent", i)
		}
	}
	for i, fr := range functionResponsesOf(t, body) {
		if fr.ID != "" {
			t.Errorf("functionResponse[%d].id = %q, want absent", i, fr.ID)
		}
	}
}

// TestEncodeRequest_IDLessToolCallsFromAnySourcePairPositionally covers the
// blocks this codec did not itself decode (a cross-dialect thread, or a
// replayed transcript predating the synthesized identity): the ids really are
// empty, so the encoder falls back to pairing calls and results by position.
func TestEncodeRequest_IDLessToolCallsFromAnySourcePairPositionally(t *testing.T) {
	t.Parallel()

	body, err := geminiapi.EncodeRequest(inference.Request{
		Model: model.Model{Name: "m"},
		Messages: content.AgenticMessages{
			aiMsg(
				toolUseBlock("", "get_weather", json.RawMessage(`{"city":"boston"}`)),
				toolUseBlock("", "get_time", json.RawMessage(`{"zone":"utc"}`)),
			),
			toolMsg("", textBlock("sunny")),
			toolMsg("", textBlock("12:00")),
		},
	})
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}

	got := functionResponsesOf(t, body)
	if len(got) != 2 {
		t.Fatalf("functionResponse parts = %d, want 2", len(got))
	}
	if got[0].Name != "get_weather" || got[0].Result != "sunny" {
		t.Errorf("first functionResponse = name %q result %q, want get_weather/sunny", got[0].Name, got[0].Result)
	}
	if got[1].Name != "get_time" || got[1].Result != "12:00" {
		t.Errorf("second functionResponse = name %q result %q, want get_time/12:00", got[1].Name, got[1].Result)
	}
}

// TestServerDecode_IDLessParallelCallsPairWithTheirResponses covers the same
// defect on the ingress side. A native client may answer a call using only
// `name` (FunctionResponse.name is Required, `id` is Optional), and the
// decoder used to keep only the id — so an id-less parallel pair arrived in the
// neutral vocabulary as two results both addressed to "", indistinguishable
// from each other and from the calls they answer.
func TestServerDecode_IDLessParallelCallsPairWithTheirResponses(t *testing.T) {
	t.Parallel()

	c, req := decodeReq(t, "m", `{
		"contents": [
			{"role":"user","parts":[{"text":"weather and time please"}]},
			{"role":"model","parts":[
				{"functionCall":{"name":"get_weather","args":{"city":"boston"}}},
				{"functionCall":{"name":"get_time","args":{"zone":"utc"}}}
			]},
			{"role":"user","parts":[
				{"functionResponse":{"name":"get_weather","response":{"result":"sunny"}}},
				{"functionResponse":{"name":"get_time","response":{"result":"12:00"}}}
			]}
		]
	}`)
	decoded, err := c.DecodeRequest(req)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}

	ai, ok := decoded.Request.Messages[1].(*content.AIMessage)
	if !ok {
		t.Fatalf("Messages[1] = %T, want *content.AIMessage", decoded.Request.Messages[1])
	}
	uses := toolUsesOf(t, ai.Blocks)
	if len(uses) != 2 {
		t.Fatalf("tool use blocks = %d, want 2", len(uses))
	}
	if uses[0].ID == "" || uses[0].ID == uses[1].ID {
		t.Fatalf("decoded call ids = (%q, %q); an id-less parallel pair must stay distinguishable", uses[0].ID, uses[1].ID)
	}

	results := make([]*content.ToolResultMessage, 0, 2)
	for _, m := range decoded.Request.Messages[2:] {
		tr, ok := m.(*content.ToolResultMessage)
		if !ok {
			t.Fatalf("message = %T, want *content.ToolResultMessage", m)
		}
		results = append(results, tr)
	}
	if len(results) != 2 {
		t.Fatalf("tool result messages = %d, want 2", len(results))
	}
	if results[0].ToolUseID != uses[0].ID {
		t.Errorf("get_weather result addressed %q, want the get_weather call's %q", results[0].ToolUseID, uses[0].ID)
	}
	if results[1].ToolUseID != uses[1].ID {
		t.Errorf("get_time result addressed %q, want the get_time call's %q", results[1].ToolUseID, uses[1].ID)
	}
}

// toolUsesOf filters a decoded message's blocks down to its tool calls.
func toolUsesOf(t *testing.T, blocks []content.Block) []*content.ToolUseBlock {
	t.Helper()
	var out []*content.ToolUseBlock
	for _, b := range blocks {
		if use, ok := b.(*content.ToolUseBlock); ok {
			out = append(out, use)
		}
	}
	return out
}
