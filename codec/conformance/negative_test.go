package conformance

import (
	"fmt"
	"strconv"
	"strings"
	"testing"
)

// A gate that never fails is not a gate. These tests are the load-bearing part
// of this package: each one takes a payload that is close enough to a real
// provider response to slip past review, and proves the gate rejects it AND
// says where. If any of them starts passing, every fixture assertion in the
// repository has quietly become worthless, so they assert on the diagnostic
// text too — a rejection that does not name the offending path is only
// marginally better than no rejection at all.

// rejection is one negative case: a checked-in payload that must not validate,
// together with the substrings its diagnostic has to contain.
type rejection struct {
	// name says which class of error this proves the gate catches.
	name string
	// format and kind select the schema.
	format, kind string
	// fixture is the checked-in payload.
	fixture string
	// wantPath is the instance location the diagnostic must point at.
	wantPath string
	// wantMessage is a distinctive fragment of the violation message.
	wantMessage string
	// maxViolations bounds how many violations the diagnostic may list. A
	// twenty-branch content-block union produces fifty violations unless the
	// wrong-variant branches are pruned, and a fifty-line rejection is one
	// nobody reads. Zero means unbounded.
	maxViolations int
}

// violationCount reads the count the diagnostic reports about itself.
func violationCount(t *testing.T, diagnostic string) int {
	t.Helper()
	for _, line := range strings.Split(diagnostic, "\n") {
		trimmed := strings.TrimSpace(line)
		rest, ok := strings.CutSuffix(trimmed, " violation(s):")
		if !ok {
			continue
		}
		n, err := strconv.Atoi(rest)
		if err != nil {
			t.Fatalf("unparsable violation count %q", trimmed)
		}
		return n
	}
	t.Fatalf("diagnostic states no violation count:\n%s", diagnostic)
	return 0
}

func TestGateRejectsPayloadsThatCouldNotHaveComeOffTheWire(t *testing.T) {
	t.Parallel()

	cases := []rejection{
		{
			name:        "a required property is missing",
			format:      "anthropic",
			kind:        "message",
			fixture:     "invalid_anthropic_message_missing_usage.json",
			wantPath:    "at (root)",
			wantMessage: "missing propert",
		},
		{
			name:        "an enum carries a value the provider never emits",
			format:      "openai",
			kind:        "chat_completion",
			fixture:     "invalid_openai_chat_completion_bad_finish_reason.json",
			wantPath:    "at /choices/0/finish_reason",
			wantMessage: "value must be one of",
		},
		{
			name:        "a Bedrock pattern constraint is violated",
			format:      "bedrock-converse",
			kind:        "converse_response",
			fixture:     "invalid_bedrock_converse_response_tool_use_id_pattern.json",
			wantPath:    "at /output/message/content/1/toolUse/toolUseId",
			wantMessage: "^[a-zA-Z0-9_.:-]+$",
		},
		{
			name:        "a Bedrock length constraint is violated",
			format:      "bedrock-converse",
			kind:        "converse_response",
			fixture:     "invalid_bedrock_converse_response_tool_use_id_length.json",
			wantPath:    "at /output/message/content/1/toolUse/toolUseId",
			wantMessage: "maxLength",
		},
		{
			name:        "a const-pinned property carries the wrong value",
			format:      "anthropic",
			kind:        "message",
			fixture:     "invalid_anthropic_message_wrong_role.json",
			wantPath:    "at /role",
			wantMessage: "assistant",
		},
		{
			name:        "a Smithy union sets two members at once",
			format:      "bedrock-converse",
			kind:        "converse_stream_output",
			fixture:     "invalid_bedrock_converse_stream_two_members.json",
			wantPath:    "at (root)",
			wantMessage: "oneOf",
		},
		{
			name:        "a Gemini enum carries an undefined value",
			format:      "gemini",
			kind:        "generate_content_response",
			fixture:     "invalid_gemini_generate_content_response_bad_finish_reason.json",
			wantPath:    "at /candidates/0/finishReason",
			wantMessage: "value must be one of",
		},
		{
			name:        "a number arrives as a string",
			format:      "openai",
			kind:        "chat_completion",
			fixture:     "invalid_openai_chat_completion_created_as_string.json",
			wantPath:    "at /created",
			wantMessage: "want integer",
		},
		{
			name:        "a stream event names a type the provider does not define",
			format:      "anthropic",
			kind:        "stream_event",
			fixture:     "invalid_anthropic_stream_unknown_event.json",
			wantPath:    "at (root)",
			wantMessage: "oneOf",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			err := Validate(tc.format, tc.kind, fixture(t, tc.fixture))
			if err == nil {
				t.Fatalf("Validate(%s/%s, %s) = nil; the gate accepted an illegal payload",
					tc.format, tc.kind, tc.fixture)
			}
			got := err.Error()
			if !strings.Contains(got, tc.wantPath) {
				t.Errorf("diagnostic does not point at %q:\n%s", tc.wantPath, got)
			}
			if !strings.Contains(got, tc.wantMessage) {
				t.Errorf("diagnostic does not mention %q:\n%s", tc.wantMessage, got)
			}
			if n := violationCount(t, got); tc.maxViolations > 0 && n > tc.maxViolations {
				t.Errorf("diagnostic lists %d violations, want at most %d; the union pruning has regressed:\n%s",
					n, tc.maxViolations, got)
			}
			// Recorded so a reviewer can read the diagnostics the gate
			// actually produces, not just that it produced one.
			t.Logf("rejection:\n%s", got)
		})
	}
}

// TestRequestGateRejectsEncoderBugs is the request half's reason for existing.
// Every case here is a defect that actually shipped in a Looprig encoder and had
// to be found by hand against a live API; each one is caught by the request gate
// before a byte leaves the process.
func TestRequestGateRejectsEncoderBugs(t *testing.T) {
	t.Parallel()

	cases := []rejection{
		{
			// The encoder tagged `thinking` omitempty and dropped it when the
			// text was empty. Anthropic requires it AND closes the block.
			name:          "an Anthropic thinking block is missing its thinking text",
			format:        "anthropic",
			kind:          "create_message_request",
			fixture:       "invalid_anthropic_create_message_request_thinking_missing_thinking.json",
			wantPath:      "at /messages/1/content/0",
			wantMessage:   "missing property 'thinking'",
			maxViolations: 4,
		},
		{
			// The Anthropic-shared encoder emitted a URL image source. Bedrock's
			// ImageSource union declares only bytes and s3Location.
			name:          "a Bedrock image source uses a member the union does not declare",
			format:        "bedrock-converse",
			kind:          "converse_request",
			fixture:       "invalid_bedrock_converse_request_image_source_url.json",
			wantPath:      "at /messages/0/content/1/image/source",
			wantMessage:   "oneOf",
			maxViolations: 4,
		},
		{
			name:        "a Bedrock request tool-use id violates the anchored pattern",
			format:      "bedrock-converse",
			kind:        "converse_request",
			fixture:     "invalid_bedrock_converse_request_tool_use_id_pattern.json",
			wantPath:    "at /messages/1/content/0/toolUse/toolUseId",
			wantMessage: "^[a-zA-Z0-9_.:-]+$",
		},
		{
			// Only reachable because the request specification closes the
			// object. The response schemas would accept this silently.
			name:          "a request carries a field the specification does not declare",
			format:        "anthropic",
			kind:          "create_message_request",
			fixture:       "invalid_anthropic_create_message_request_undeclared_field.json",
			wantPath:      "at /messages/0/content/1",
			wantMessage:   "additionalProperties",
			maxViolations: 4,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			err := Validate(tc.format, tc.kind, fixture(t, tc.fixture))
			if err == nil {
				t.Fatalf("Validate(%s/%s, %s) = nil; the gate accepted a request the provider would reject",
					tc.format, tc.kind, tc.fixture)
			}
			got := err.Error()
			if !strings.Contains(got, tc.wantPath) {
				t.Errorf("diagnostic does not point at %q:\n%s", tc.wantPath, got)
			}
			if !strings.Contains(got, tc.wantMessage) {
				t.Errorf("diagnostic does not mention %q:\n%s", tc.wantMessage, got)
			}
			if n := violationCount(t, got); tc.maxViolations > 0 && n > tc.maxViolations {
				t.Errorf("diagnostic lists %d violations, want at most %d; the union pruning has regressed:\n%s",
					n, tc.maxViolations, got)
			}
			t.Logf("rejection:\n%s", got)
		})
	}
}

// TestGateRejectsAStreamFrameThatDisagreesWithItsEventName covers a failure the
// per-payload schema cannot see: both halves of the frame are individually
// legal, but no provider would ever send them together.
func TestGateRejectsAStreamFrameThatDisagreesWithItsEventName(t *testing.T) {
	t.Parallel()

	stub := &stubTB{TB: t}
	stub.run(func() {
		MustValidateStream(stub, "anthropic", "stream_event",
			fixture(t, "invalid_anthropic_stream_event_name_mismatch.sse"))
	})
	if !stub.failed {
		t.Fatal("MustValidateStream accepted a frame whose event name contradicts its payload")
	}
	if !strings.Contains(stub.message, "disagrees with payload type") {
		t.Fatalf("diagnostic does not explain the disagreement:\n%s", stub.message)
	}
	t.Logf("rejection:\n%s", stub.message)
}

// TestMustValidateFailsTheTest proves the testing-facing entry point actually
// terminates a test rather than merely returning an error nobody reads.
func TestMustValidateFailsTheTest(t *testing.T) {
	t.Parallel()

	stub := &stubTB{TB: t}
	stub.run(func() {
		MustValidate(stub, "anthropic", "message", fixture(t, "invalid_anthropic_message_missing_usage.json"))
		t.Error("MustValidate returned instead of failing the test")
	})
	if !stub.failed {
		t.Fatal("MustValidate did not fail on an illegal payload")
	}
}

// TestGateRejectsAnUnknownFormatOrKind keeps a typo in a fixture test from
// silently validating nothing.
func TestGateRejectsAnUnknownFormatOrKind(t *testing.T) {
	t.Parallel()

	if err := Validate("openai-chat", "chat_completion", []byte(`{}`)); err == nil {
		t.Fatal("Validate with an unknown api-format = nil, want a rejection")
	}
	if err := Validate("openai", "completion", []byte(`{}`)); err == nil {
		t.Fatal("Validate with an unknown kind = nil, want a rejection")
	}
	if err := Validate("anthropic", "stream_event/message_middle", []byte(`{}`)); err == nil {
		t.Fatal("Validate with an unknown union member = nil, want a rejection")
	}
}

// TestGateRejectsMalformedJSON closes the last hole: bytes that are not JSON at
// all must fail here rather than at some decoder further down.
func TestGateRejectsMalformedJSON(t *testing.T) {
	t.Parallel()

	err := Validate("anthropic", "message", []byte(`{"id": "msg_1",}`))
	if err == nil {
		t.Fatal("Validate(malformed JSON) = nil, want a rejection")
	}
	if !strings.Contains(err.Error(), "not valid JSON") {
		t.Fatalf("diagnostic does not identify the payload as malformed: %v", err)
	}
}

// stubTB captures a fatal failure instead of ending the real test, so the
// negative tests can assert that MustValidate fails and inspect what it said.
type stubTB struct {
	testing.TB
	failed  bool
	message string
}

// abort unwinds a stubbed Fatalf, mirroring the goroutine exit the real
// testing.T performs.
type abort struct{}

func (s *stubTB) Helper() {}

func (s *stubTB) Fatalf(format string, args ...any) {
	s.failed = true
	s.message = fmt.Sprintf(format, args...)
	panic(abort{})
}

func (s *stubTB) Errorf(format string, args ...any) {
	s.failed = true
	s.message = fmt.Sprintf(format, args...)
}

// run executes fn, absorbing the unwind a stubbed Fatalf performs.
func (s *stubTB) run(fn func()) {
	defer func() {
		if r := recover(); r != nil {
			if _, ok := r.(abort); !ok {
				panic(r)
			}
		}
	}()
	fn()
}
