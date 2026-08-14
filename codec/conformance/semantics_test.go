package conformance

import (
	"encoding/json"
	"strings"
	"testing"
)

// chatRequestWithArguments builds a minimal, SCHEMA-VALID Chat Completions
// request whose one assistant tool call carries the given raw `arguments`
// member. Every case below therefore isolates the semantic check: the schema
// gate passes each of them, which is the point.
func chatRequestWithArguments(t *testing.T, arguments string) []byte {
	t.Helper()
	body := []byte(`{"model":"gpt-4.1","messages":[{"role":"assistant","tool_calls":[` +
		`{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":` + arguments + `}}]}]}`)
	if err := Validate("openai", "chat_completion_request", body); err != nil {
		t.Fatalf("case is not schema-valid, so it proves nothing about the semantic check: %v", err)
	}
	return body
}

// TestChatToolArgumentsSemanticCheck inverts the gate, per the rule that a gate
// which never fails is not a gate. `arguments` is spec-typed `string`, so a
// string wrapping another quoted string is schema-valid — the schema gate
// cannot see the double-encoding defect that shipped, and this check is what
// carries the constraint instead.
func TestChatToolArgumentsSemanticCheck(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		arguments string
		wantErr   string // empty means the body must be accepted
	}{
		{name: "object in a string is the contract", arguments: `"{\"city\":\"Paris\"}"`},
		{name: "no parameters", arguments: `"{}"`},
		{
			// The shipped defect, verbatim.
			name:      "double-encoded arguments are rejected",
			arguments: `"\"{\\\"city\\\": \\\"Paris\\\"}\""`,
			wantErr:   "double-encoded",
		},
		{name: "empty string is rejected", arguments: `""`, wantErr: "arguments is empty"},
		{name: "array is rejected", arguments: `"[1,2]"`, wantErr: "not an object"},
		{name: "number is rejected", arguments: `"7"`, wantErr: "not an object"},
		{
			// Preserved model output: not valid JSON, so not our encoding bug.
			// The tool dispatcher is the documented validation site.
			name:      "invalid model JSON passes through",
			arguments: `"{\"city\": "`,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			body := chatRequestWithArguments(t, tc.arguments)
			err := checkRequestSemantics("openai", "chat_completion_request", body)
			switch {
			case tc.wantErr == "" && err != nil:
				t.Fatalf("checkRequestSemantics() = %v, want accepted", err)
			case tc.wantErr == "":
			case err == nil:
				t.Fatalf("checkRequestSemantics() = nil, want a rejection mentioning %q", tc.wantErr)
			case !strings.Contains(err.Error(), tc.wantErr):
				t.Fatalf("checkRequestSemantics() = %v, want it to mention %q", err, tc.wantErr)
			default:
				// A rejection nobody can act on is barely better than none:
				// the diagnostic must locate the offending call.
				for _, want := range []string{"/messages/0/tool_calls/0/function/arguments", "get_weather", "call_1"} {
					if !strings.Contains(err.Error(), want) {
						t.Errorf("diagnostic %q does not mention %q", err, want)
					}
				}
			}
		})
	}
}

// TestBareObjectArgumentsAreRejectedByBothGates records which gate carries
// which half of the contract: the SCHEMA rejects a bare object (wrong type),
// the SEMANTIC check rejects a correctly-typed string with the wrong contents.
func TestBareObjectArgumentsAreRejectedByBothGates(t *testing.T) {
	t.Parallel()

	body := []byte(`{"model":"gpt-4.1","messages":[{"role":"assistant","tool_calls":[` +
		`{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":{"city":"Paris"}}}]}]}`)
	if err := Validate("openai", "chat_completion_request", body); err == nil {
		t.Error("schema accepted a bare-object arguments member")
	}
	err := checkRequestSemantics("openai", "chat_completion_request", body)
	if err == nil || !strings.Contains(err.Error(), "must be a JSON string") {
		t.Errorf("semantic check = %v, want a rejection naming the string requirement", err)
	}
}

func TestValidateRequestRunsSemanticChecks(t *testing.T) {
	t.Parallel()

	body := chatRequestWithArguments(t, `"\"{}\""`)
	err := ValidateRequest("openai", "chat_completion_request", body)
	if err == nil || !strings.Contains(err.Error(), "double-encoded") {
		t.Fatalf("ValidateRequest() = %v, want semantic rejection", err)
	}
}

func TestValidateRequestRejectsResponseKinds(t *testing.T) {
	t.Parallel()

	err := ValidateRequest("anthropic", "message", []byte(`{}`))
	if err == nil || !strings.Contains(err.Error(), "response kind") {
		t.Fatalf("ValidateRequest(response kind) = %v, want direction rejection", err)
	}
}

// TestSemanticCheckIgnoresOtherKinds keeps the check narrow: it must not fire
// on a format or kind it was not written for, or on a body it cannot parse
// (the schema gate owns that diagnostic).
func TestSemanticCheckIgnoresOtherKinds(t *testing.T) {
	t.Parallel()

	doubled := chatRequestWithArguments(t, `"\"{}\""`)
	if err := checkRequestSemantics("openai-responses", "create_response_request", doubled); err != nil {
		t.Errorf("checkRequestSemantics() on another format = %v, want nil", err)
	}
	if err := checkRequestSemantics("openai", "chat_completion_request", []byte(`{`)); err != nil {
		t.Errorf("checkRequestSemantics() on an unparsable body = %v, want nil", err)
	}
	var probe map[string]json.RawMessage
	if err := json.Unmarshal(doubled, &probe); err != nil {
		t.Fatalf("fixture is not JSON: %v", err)
	}
}
