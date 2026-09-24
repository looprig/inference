package inference_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/looprig/inference"
)

// TestValidateRequestFeaturesSessionID pins the contract of the optional
// Request.SessionID: the zero value is always valid (every pre-existing caller
// sends it), a sendable identifier is accepted, and an identifier that could
// not arrive upstream byte-identical as an HTTP header field value fails closed
// with a typed error before any I/O. The error must not echo the value itself.
func TestValidateRequestFeaturesSessionID(t *testing.T) {
	t.Parallel()

	atLimit := strings.Repeat("a", inference.MaxSessionIDBytes)
	tests := []struct {
		name       string
		sessionID  string
		wantReason inference.SessionIDProblem
		wantIndex  int
	}{
		{name: "empty is absent and valid", sessionID: ""},
		{name: "uuid is valid", sessionID: "6f1e2d3c-4b5a-4978-8675-6453422110ff"},
		{name: "opaque ascii token is valid", sessionID: "conv_2026-09-10_a1b2"},
		{name: "every visible ascii byte is valid", sessionID: "!\"#$%&'()*+,-./09:;<=>?@AZ[\\]^_`az{|}~"},
		{name: "interior tab is valid", sessionID: "id\twith-tab"},
		{name: "interior space is valid", sessionID: "id with-space"},
		{name: "high bytes are valid obs-text", sessionID: "idé"},
		{name: "obs-text lower bound is valid", sessionID: "id\x80"},
		{name: "obs-text upper bound is valid", sessionID: "id\xff"},
		{name: "exactly the byte limit is valid", sessionID: atLimit},

		{name: "newline rejected", sessionID: "line1\nline2", wantReason: inference.SessionIDUnsendableByte, wantIndex: 5},
		{name: "carriage return rejected", sessionID: "line1\rline2", wantReason: inference.SessionIDUnsendableByte, wantIndex: 5},
		{name: "header-injection suffix rejected", sessionID: "id X-Injected: yes\r\n", wantReason: inference.SessionIDUnsendableByte, wantIndex: 18},
		{name: "nul byte rejected", sessionID: "id\x00", wantReason: inference.SessionIDUnsendableByte, wantIndex: 2},
		{name: "unit separator rejected", sessionID: "id\x1f", wantReason: inference.SessionIDUnsendableByte, wantIndex: 2},
		{name: "vertical tab rejected", sessionID: "id\x0b", wantReason: inference.SessionIDUnsendableByte, wantIndex: 2},
		{name: "DEL rejected", sessionID: "id\x7f", wantReason: inference.SessionIDUnsendableByte, wantIndex: 2},
		{name: "first offending byte is reported", sessionID: "a\x01b\x02", wantReason: inference.SessionIDUnsendableByte, wantIndex: 1},
		{name: "unsendable byte wins over surrounding whitespace", sessionID: " a\n", wantReason: inference.SessionIDUnsendableByte, wantIndex: 2},

		{name: "leading space rejected", sessionID: " id", wantReason: inference.SessionIDSurroundingWhitespace, wantIndex: 0},
		{name: "leading tab rejected", sessionID: "\tid", wantReason: inference.SessionIDSurroundingWhitespace, wantIndex: 0},
		{name: "trailing space rejected", sessionID: "id ", wantReason: inference.SessionIDSurroundingWhitespace, wantIndex: 2},
		{name: "trailing tab rejected", sessionID: "id\t", wantReason: inference.SessionIDSurroundingWhitespace, wantIndex: 2},
		{name: "whitespace only rejected", sessionID: "  ", wantReason: inference.SessionIDSurroundingWhitespace, wantIndex: 0},

		{name: "one byte over the limit rejected", sessionID: atLimit + "a", wantReason: inference.SessionIDTooLong, wantIndex: inference.MaxSessionIDBytes},
		{name: "too long wins over its bytes", sessionID: atLimit + "\n", wantReason: inference.SessionIDTooLong, wantIndex: inference.MaxSessionIDBytes},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			err := inference.ValidateRequestFeatures(inference.Request{SessionID: test.sessionID})
			if test.wantReason == "" {
				if err != nil {
					t.Fatalf("ValidateRequestFeatures() error = %v, want nil", err)
				}
				return
			}
			var invalid *inference.InvalidSessionIDError
			if !errors.As(err, &invalid) {
				t.Fatalf("ValidateRequestFeatures() error = %T %v, want *inference.InvalidSessionIDError", err, err)
			}
			if invalid.Reason != test.wantReason || invalid.Index != test.wantIndex {
				t.Errorf("InvalidSessionIDError = {Reason:%q Index:%d}, want {Reason:%q Index:%d}",
					invalid.Reason, invalid.Index, test.wantReason, test.wantIndex)
			}
			got := err.Error()
			if len(got) > 160 {
				t.Errorf("error message too long: %q", got)
			}
			if !strings.Contains(got, string(test.wantReason)) {
				t.Errorf("error message %q does not name the reason %q", got, test.wantReason)
			}
		})
	}
}

// TestValidateRequestFeaturesSessionIDOrdering pins where the SessionID check
// sits: after the message-shape invariant (a malformed thread is reported as
// such even when the identity is also bad), and before the tool-choice
// invariants, so an unsendable identity is reported even when the request
// would otherwise fail later.
func TestValidateRequestFeaturesSessionIDOrdering(t *testing.T) {
	t.Parallel()

	err := inference.ValidateRequestFeatures(inference.Request{SessionID: "id\n", TransientMessages: 1})
	var transient *inference.InvalidTransientMessagesError
	if !errors.As(err, &transient) {
		t.Fatalf("ValidateRequestFeatures() error = %T %v, want the transient-message error first", err, err)
	}

	err = inference.ValidateRequestFeatures(inference.Request{SessionID: "id\n", ToolChoice: inference.ToolRequired()})
	var invalid *inference.InvalidSessionIDError
	if !errors.As(err, &invalid) {
		t.Fatalf("ValidateRequestFeatures() error = %T %v, want *inference.InvalidSessionIDError before tool-choice checks", err, err)
	}
}

// TestInvalidSessionIDErrorDoesNotEchoValue pins that no refusal reason copies
// the caller's identifier into the error text, where it would reach logs.
func TestInvalidSessionIDErrorDoesNotEchoValue(t *testing.T) {
	t.Parallel()

	const marker = "conversation-marker"
	for _, sessionID := range []string{
		marker + "\n",
		" " + marker,
		marker + strings.Repeat("x", inference.MaxSessionIDBytes),
	} {
		err := inference.ValidateRequestFeatures(inference.Request{SessionID: sessionID})
		if err == nil {
			t.Fatalf("ValidateRequestFeatures(%q) = nil, want an error", sessionID)
		}
		if strings.Contains(err.Error(), marker) {
			t.Errorf("error %q echoes the caller's SessionID", err.Error())
		}
	}
}

// TestSessionIDProblemSpellings pins the reason codes. They appear in error
// text a caller may match or log, so a rename is a contract change.
func TestSessionIDProblemSpellings(t *testing.T) {
	t.Parallel()

	for got, want := range map[inference.SessionIDProblem]string{
		inference.SessionIDUnsendableByte:        "unsendable_byte",
		inference.SessionIDSurroundingWhitespace: "surrounding_whitespace",
		inference.SessionIDTooLong:               "too_long",
	} {
		if string(got) != want {
			t.Errorf("SessionIDProblem = %q, want %q", got, want)
		}
	}
	if inference.MaxSessionIDBytes != 256 {
		t.Errorf("MaxSessionIDBytes = %d, want 256 (Core sessionwire MaxIDBytes)", inference.MaxSessionIDBytes)
	}
}
