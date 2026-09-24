package inference

import "fmt"

// MaxSessionIDBytes is the largest permitted byte length of Request.SessionID.
// It matches Core's sessionwire MaxIDBytes, so any Looprig session identity is
// representable, while keeping the header a provider receives far below the
// request-header limits HTTP servers enforce (an oversized header draws an
// opaque 431 or a dropped connection rather than a diagnosable refusal).
const MaxSessionIDBytes = 256

// SessionIDProblem names why a Request.SessionID was refused.
type SessionIDProblem string

const (
	// SessionIDUnsendableByte: a byte that cannot appear in an HTTP header
	// field value — CR, LF, NUL, any other C0 control except HTAB, or DEL.
	SessionIDUnsendableByte SessionIDProblem = "unsendable_byte"
	// SessionIDSurroundingWhitespace: a leading or trailing SP or HTAB. HTTP
	// recipients strip optional whitespace around a field value (RFC 9110
	// §5.5), so the identity would arrive upstream as a DIFFERENT value — one
	// that can collide with another conversation's.
	SessionIDSurroundingWhitespace SessionIDProblem = "surrounding_whitespace"
	// SessionIDTooLong: longer than MaxSessionIDBytes bytes.
	SessionIDTooLong SessionIDProblem = "too_long"
)

// InvalidSessionIDError reports a Request.SessionID that could not arrive at a
// provider byte-identical as an HTTP header field value. A provider that carries
// the session identity in a request header (see Request.SessionID) must send it
// verbatim, and an identity that is altered or refused in transit is worse than
// one refused locally: net/http's Transport refuses a control byte with an
// untyped error naming only the header, a transport that writes through
// Request.Write rewrites CR and LF into spaces, and a recipient strips
// surrounding whitespace. A silently-altered affinity key is undiagnosable from
// the provider side, where it looks like a conversation that keeps changing
// sessions.
//
// The message deliberately does not echo the offending value: it is caller data
// and may contain arbitrary bytes. Reason and Index locate the problem instead.
type InvalidSessionIDError struct {
	// Reason classifies the refusal.
	Reason SessionIDProblem
	// Index is the byte offset of the first offending byte. For
	// SessionIDTooLong it is MaxSessionIDBytes, the offset of the first byte
	// past the limit.
	Index int
}

func (e *InvalidSessionIDError) Error() string {
	return fmt.Sprintf("inference: request SessionID is not sendable as an HTTP header value (%s at byte offset %d)", e.Reason, e.Index)
}

// validateSessionID reports whether id can be transmitted verbatim, and
// unaltered by the recipient, as an HTTP header field value. The checks run in
// a fixed order — length, then bytes, then surrounding whitespace — so exactly
// one reason is reported for a value with several problems.
//
// The accepted byte set is RFC 9110's field-value: visible ASCII, SP and HTAB,
// and obs-text (0x80-0xFF; RFC 9110 permits it and net/http transmits it). An
// empty id is valid: it means "absent", and providers omit the header entirely.
func validateSessionID(id string) error {
	if len(id) > MaxSessionIDBytes {
		return &InvalidSessionIDError{Reason: SessionIDTooLong, Index: MaxSessionIDBytes}
	}
	for i := 0; i < len(id); i++ {
		b := id[i]
		switch {
		case b == '\t':
		case b >= 0x20 && b <= 0x7E:
		case b >= 0x80:
		default:
			return &InvalidSessionIDError{Reason: SessionIDUnsendableByte, Index: i}
		}
	}
	if id == "" {
		return nil
	}
	if isHeaderWhitespace(id[0]) {
		return &InvalidSessionIDError{Reason: SessionIDSurroundingWhitespace, Index: 0}
	}
	if last := len(id) - 1; isHeaderWhitespace(id[last]) {
		return &InvalidSessionIDError{Reason: SessionIDSurroundingWhitespace, Index: last}
	}
	return nil
}

func isHeaderWhitespace(b byte) bool { return b == ' ' || b == '\t' }
