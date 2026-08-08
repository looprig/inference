package stream

import (
	"io"
	"testing"
)

func TestStreamResult_PreservesAttempts(t *testing.T) {
	reader := NewStreamReaderWithResult(
		func() (int, error) { return 0, io.EOF },
		nil,
		func() (StreamResult, bool, error) {
			return StreamResult{Attempts: 3}, true, nil
		},
	)
	if _, err := reader.Next(); err != io.EOF {
		t.Fatalf("Next() error = %v, want io.EOF", err)
	}
	result, ok := reader.Result()
	if !ok {
		t.Fatal("Result() reported no terminal metadata")
	}
	if result.Attempts != 3 {
		t.Fatalf("Attempts = %d, want 3", result.Attempts)
	}
}
