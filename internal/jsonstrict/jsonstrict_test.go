package jsonstrict_test

import (
	"errors"
	"testing"

	"github.com/looprig/inference/internal/jsonstrict"
)

func TestRejectDuplicateKeys_NoDuplicate(t *testing.T) {
	t.Parallel()

	err := jsonstrict.RejectDuplicateKeys([]byte(`{"a":1,"b":{"c":2},"d":[1,2,{"e":3}]}`))
	if err != nil {
		t.Fatalf("RejectDuplicateKeys() = %v, want nil", err)
	}
}

func TestRejectDuplicateKeys_TopLevelDuplicate(t *testing.T) {
	t.Parallel()

	err := jsonstrict.RejectDuplicateKeys([]byte(`{"a":1,"a":2}`))
	var dupErr *jsonstrict.DuplicateKeyError
	if !errors.As(err, &dupErr) {
		t.Fatalf("RejectDuplicateKeys() = %v (%T), want *DuplicateKeyError", err, err)
	}
	if dupErr.Key != "a" {
		t.Errorf("DuplicateKeyError.Key = %q, want %q", dupErr.Key, "a")
	}
}

func TestRejectDuplicateKeys_NestedDuplicate(t *testing.T) {
	t.Parallel()

	err := jsonstrict.RejectDuplicateKeys([]byte(`{"a":1,"nested":{"x":1,"y":2,"x":3}}`))
	var dupErr *jsonstrict.DuplicateKeyError
	if !errors.As(err, &dupErr) {
		t.Fatalf("RejectDuplicateKeys() = %v (%T), want *DuplicateKeyError", err, err)
	}
	if dupErr.Key != "x" {
		t.Errorf("DuplicateKeyError.Key = %q, want %q", dupErr.Key, "x")
	}
}

func TestRejectDuplicateKeys_DuplicateInArrayElement(t *testing.T) {
	t.Parallel()

	err := jsonstrict.RejectDuplicateKeys([]byte(`{"list":[{"k":1},{"k":2,"k":3}]}`))
	var dupErr *jsonstrict.DuplicateKeyError
	if !errors.As(err, &dupErr) {
		t.Fatalf("RejectDuplicateKeys() = %v (%T), want *DuplicateKeyError", err, err)
	}
	if dupErr.Key != "k" {
		t.Errorf("DuplicateKeyError.Key = %q, want %q", dupErr.Key, "k")
	}
}

func TestRejectDuplicateKeys_SameKeyDifferentObjectsIsFine(t *testing.T) {
	t.Parallel()

	// The same member name repeated across sibling objects is not a
	// duplicate; only repetition within the same object matters.
	err := jsonstrict.RejectDuplicateKeys([]byte(`[{"k":1},{"k":2}]`))
	if err != nil {
		t.Fatalf("RejectDuplicateKeys() = %v, want nil", err)
	}
}

func TestRejectDuplicateKeys_MalformedTruncatedString(t *testing.T) {
	t.Parallel()

	// An unterminated string is a genuine JSON syntax error the underlying
	// json.Decoder surfaces as a non-EOF error while tokenizing (unlike a
	// body that is merely truncated at an otherwise-valid token boundary,
	// which json.Decoder.Token reports as a clean io.EOF and this scan
	// correctly treats as "nothing more to walk" -- the subsequent full
	// unmarshal in each dialect's DecodeRequest is what rejects that case).
	err := jsonstrict.RejectDuplicateKeys([]byte(`{"a": "b`))
	var malformedErr *jsonstrict.MalformedError
	if !errors.As(err, &malformedErr) {
		t.Fatalf("RejectDuplicateKeys() = %v (%T), want *MalformedError", err, err)
	}
}

func TestRejectDuplicateKeys_MismatchedDelimiter(t *testing.T) {
	t.Parallel()

	err := jsonstrict.RejectDuplicateKeys([]byte(`{"a":1}}`))
	var malformedErr *jsonstrict.MalformedError
	if !errors.As(err, &malformedErr) {
		t.Fatalf("RejectDuplicateKeys() = %v (%T), want *MalformedError", err, err)
	}
}
