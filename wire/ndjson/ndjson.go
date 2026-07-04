// Package ndjson frames a newline-delimited JSON body into one raw stream frame per
// line: StreamFrame.Data is the line's bytes and Name is empty. Blank lines are
// skipped. It is byte-level wire framing only — it does not parse the JSON or know any
// LLM semantics; a semantic StreamDecoder interprets each line's Data.
package ndjson

import (
	"bufio"
	"io"

	"github.com/looprig/inference"
)

// maxLineBytes bounds a single NDJSON line so a buggy server cannot force unbounded
// buffering. 1 MiB comfortably exceeds any realistic streamed JSON object line.
const maxLineBytes = 1 << 20

// FramerError wraps a stream read failure surfaced while framing. Typed per the repo
// rule so callers can errors.As it and Unwrap the underlying I/O cause.
type FramerError struct {
	Reason string
	Err    error
}

func (e *FramerError) Error() string {
	if e.Err != nil {
		return "ndjson: " + e.Reason + ": " + e.Err.Error()
	}
	return "ndjson: " + e.Reason
}

func (e *FramerError) Unwrap() error { return e.Err }

// Compile-time proof that the package satisfies the framer contract.
var _ inference.StreamFramer = framer{}

type framer struct{}

func (framer) DecodeStreamFrames(body io.ReadCloser) (*inference.StreamReader[inference.StreamFrame], error) {
	return DecodeStreamFrames(body)
}

// Framer returns the package's StreamFramer as an interface value, for callers that
// inject an inference.StreamFramer.
func Framer() inference.StreamFramer { return framer{} }

// DecodeStreamFrames frames an NDJSON body into one StreamFrame per non-empty line. It
// owns body: the returned reader's Close closes body; on an early error (nil body)
// there is no body to close.
func DecodeStreamFrames(body io.ReadCloser) (*inference.StreamReader[inference.StreamFrame], error) {
	if body == nil {
		return nil, &FramerError{Reason: "nil body"}
	}
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, bufio.MaxScanTokenSize), maxLineBytes)

	next := func() (inference.StreamFrame, error) {
		for scanner.Scan() {
			line := scanner.Bytes()
			if len(line) == 0 {
				continue // skip blank lines
			}
			// scanner.Bytes() aliases the scan buffer; copy so the frame is stable.
			out := make([]byte, len(line))
			copy(out, line)
			return inference.StreamFrame{Data: out}, nil
		}
		if err := scanner.Err(); err != nil {
			return inference.StreamFrame{}, &FramerError{Reason: "read stream", Err: err}
		}
		return inference.StreamFrame{}, io.EOF
	}

	return inference.NewStreamReader(next, body.Close), nil
}
