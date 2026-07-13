package inference

import (
	"fmt"

	"github.com/looprig/core/content"
)

// StreamResult is authoritative terminal metadata for one cleanly completed
// provider stream. Usage is absent when the provider did not report it.
type StreamResult struct {
	Usage        *content.Usage
	Model        string
	FinishReason FinishReason
}

// StreamResultProducer supplies the terminal metadata accumulated while a
// stream is read. It is called once, and only after Next observes clean EOF.
// A false bool means the producer has no authoritative terminal metadata.
type StreamResultProducer func() (StreamResult, bool, error)

// StreamOperation identifies the public StreamReader operation that failed.
type StreamOperation string

const (
	StreamOperationNext  StreamOperation = "Next"
	StreamOperationClose StreamOperation = "Close"
)

// StreamReaderFailure identifies a structurally invalid StreamReader boundary.
type StreamReaderFailure string

const (
	StreamReaderFailureNilReceiver        StreamReaderFailure = "nil receiver"
	StreamReaderFailureMissingNext        StreamReaderFailure = "missing next function"
	StreamReaderFailureMissingFrameMapper StreamReaderFailure = "missing frame mapper"
)

// StreamReaderError reports an invalid public StreamReader boundary.
type StreamReaderError struct {
	Operation StreamOperation
	Failure   StreamReaderFailure
}

func (e *StreamReaderError) Error() string {
	return fmt.Sprintf("inference: stream %s failed: %s", e.Operation, e.Failure)
}

// StreamResultError reports terminal metadata that could not be authorized.
type StreamResultError struct {
	// Cause remains directly inspectable but is intentionally not exposed through
	// Unwrap: even an io.EOF cause is a metadata failure, never clean exhaustion.
	Cause error
}

func (e *StreamResultError) Error() string {
	if e.Cause == nil {
		return "inference: stream result failed: unknown cause"
	}
	return "inference: stream result failed: " + e.Cause.Error()
}

func cloneStreamResult(result StreamResult) StreamResult {
	if result.Usage == nil {
		return result
	}
	usage := *result.Usage
	result.Usage = &usage
	return result
}

func readStreamResult(producer StreamResultProducer) (StreamResult, bool, error) {
	if producer == nil {
		return StreamResult{}, false, nil
	}
	result, ok, err := producer()
	if err != nil {
		return StreamResult{}, false, &StreamResultError{Cause: err}
	}
	if !ok {
		return StreamResult{}, false, nil
	}
	result = cloneStreamResult(result)
	if result.Usage != nil {
		if err := result.Usage.Validate(); err != nil {
			return StreamResult{}, false, &StreamResultError{Cause: err}
		}
	}
	return result, true, nil
}
