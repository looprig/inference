package geminiapi_test

import (
	"encoding/json"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/conformance"
	"github.com/looprig/inference/codec/geminiapi"
	model "github.com/looprig/inference/model"
)

// kindGenerateContentRequest is the gate's key for the GenerateContent request
// body.
const kindGenerateContentRequest = "generate_content_request"

// TestEncodeRequestHoldsAgainstTheOfficialRequestSchema is this codec's half of
// the module rule: "every encode path must hold its encoded body against the
// format's official request schema in tests".
//
// The rule was unsatisfiable here until the gate moved into this module. The
// schemas lived in llm, one tier up and behind an internal/, so the only tests
// that could reach them belonged to the provider clients that compose this
// codec — a tier too late to say which encoder produced a rejected body.
//
// State this gate's real strength rather than assuming it, and see
// TestTheRequestGateActuallyRejects below for the measurement it is stated
// from. Google's discovery document declares required properties on 1 of 49
// request shapes, contains no oneOf at all, and closes no object, so what is
// enforced here is types, nesting and declared enum members — nothing else. A
// two-member Part, a missing contents array, an undeclared property and an
// undeclared role all pass. Those constraints are carried by this package's
// encoder and by its own structural assertions, which is exactly why the gate
// is added alongside them and not instead of them.
func TestEncodeRequestHoldsAgainstTheOfficialRequestSchema(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		request inference.Request
	}{
		{
			name: "system instruction and a plain exchange",
			request: inference.Request{
				Model:  model.Model{Name: "gemini-2.5-pro"},
				System: "be brief",
				Messages: content.AgenticMessages{
					userMsg(textBlock("hello")),
					aiMsg(textBlock("hi")),
				},
			},
		},
		{
			name: "tool call and its result",
			request: inference.Request{
				Model: model.Model{Name: "gemini-2.5-pro"},
				Tools: []inference.Tool{{
					Name:        "lookup",
					Description: "Look up a value",
					Schema:      json.RawMessage(`{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}`),
				}},
				Messages: content.AgenticMessages{
					userMsg(textBlock("look it up")),
					aiMsg(content.NewToolUseBlock("call-1", "lookup", json.RawMessage(`{"key":"a"}`), nil, "")),
					toolMsg("call-1", textBlock("found")),
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			body, err := geminiapi.EncodeRequest(tt.request)
			if err != nil {
				t.Fatalf("EncodeRequest() error = %v", err)
			}
			conformance.MustValidateRequest(t, "gemini", kindGenerateContentRequest, body)
		})
	}
}

// TestTheRequestGateActuallyRejects measures what this gate really enforces
// instead of reasoning about it, and the measurement produced a surprise worth
// recording: the derived gemini request document does NOT reject an undeclared
// role such as {"role":"oracle"}, and does NOT reject an undeclared property
// such as {"txt":"hi"} — Google's discovery format types `role` as a plain
// string and closes nothing. Both were written here as expected rejections
// first, and both passed the gate.
//
// What the gate does catch, and therefore what the assertions above rest on,
// is types and declared enum members. The `want` column is the measurement, not
// an aspiration: a case that flips is a real change in the gate's strength and
// should be investigated rather than edited away.
func TestTheRequestGateActuallyRejects(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		body       string
		wantReject bool
	}{
		{
			name:       "contents typed as a string, not an array",
			body:       `{"contents":"hi"}`,
			wantReject: true,
		},
		{
			name:       "a part's text typed as a number",
			body:       `{"contents":[{"role":"user","parts":[{"text":5}]}]}`,
			wantReject: true,
		},
		{
			name:       "a safety category the enum does not declare",
			body:       `{"contents":[{"role":"user","parts":[{"text":"x"}]}],"safetySettings":[{"category":"NOPE","threshold":"BLOCK_NONE"}]}`,
			wantReject: true,
		},
		{
			name:       "a function-calling mode the enum does not declare",
			body:       `{"contents":[{"role":"user","parts":[{"text":"x"}]}],"toolConfig":{"functionCallingConfig":{"mode":"NOPE"}}}`,
			wantReject: true,
		},
		{
			// Measured, not assumed: discovery-derived documents close
			// nothing, so an invented top-level member passes. The encoder is
			// what keeps this out of a real body.
			name:       "an undeclared top-level property is NOT caught",
			body:       `{"contents":[{"role":"user","parts":[{"text":"x"}]}],"bogus":1}`,
			wantReject: false,
		},
		{
			// Measured, not assumed: `role` is a bare string in the discovery
			// document, so the union of legal roles is enforced by this
			// package's encoder alone.
			name:       "an undeclared role is NOT caught",
			body:       `{"contents":[{"role":"oracle","parts":[{"text":"x"}]}]}`,
			wantReject: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			err := conformance.Validate("gemini", kindGenerateContentRequest, []byte(tt.body))
			if (err != nil) != tt.wantReject {
				t.Fatalf("Validate(%s) error = %v, want rejected = %v", tt.body, err, tt.wantReject)
			}
		})
	}
}
