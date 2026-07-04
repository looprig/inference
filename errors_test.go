package inference_test

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/looprig/inference"
)

// Compile-time assertions that all error types satisfy the error interface.
// These are intentionally outside the table-driven pattern — they carry no runtime behavior.
var (
	_ error = (*inference.NetworkError)(nil)
	_ error = (*inference.APIError)(nil)
	_ error = (*inference.ValidationError)(nil)
	_ error = (*inference.MissingCredentialsError)(nil)
	_ error = (*inference.ModelMismatchError)(nil)
)

func TestNetworkError(t *testing.T) {
	t.Parallel()

	inner := errors.New("connection refused")

	tests := []struct {
		name         string
		err          *inference.NetworkError
		wantContains string
		wantUnwrap   error
	}{
		{
			name:         "wraps inner error message in output",
			err:          &inference.NetworkError{Err: inner},
			wantContains: "connection refused",
			wantUnwrap:   inner,
		},
		{
			name:         "error string has inference prefix",
			err:          &inference.NetworkError{Err: inner},
			wantContains: "inference:",
			wantUnwrap:   inner,
		},
		{
			name:         "wraps a wrapped error and unwrap chain works",
			err:          &inference.NetworkError{Err: fmt.Errorf("dial tcp: %w", inner)},
			wantContains: "dial tcp",
			wantUnwrap:   nil, // we check errors.Is separately below
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := tt.err.Error()
			if got == "" {
				t.Fatalf("Error() returned empty string")
			}
			if !strings.Contains(got, tt.wantContains) {
				t.Errorf("Error() = %q, want it to contain %q", got, tt.wantContains)
			}
			if tt.wantUnwrap != nil {
				if !errors.Is(tt.err, tt.wantUnwrap) {
					t.Errorf("errors.Is(%v, %v) = false, want true", tt.err, tt.wantUnwrap)
				}
			}
		})
	}

	// Separate table for Unwrap chain through nested wrapping.
	t.Run("errors.Is traverses full unwrap chain", func(t *testing.T) {
		t.Parallel()

		wrapped := fmt.Errorf("dial tcp: %w", inner)
		ne := &inference.NetworkError{Err: wrapped}
		if !errors.Is(ne, inner) {
			t.Errorf("errors.Is(NetworkError{Err: fmt.Errorf(\"...: inner\")}, inner) = false, want true")
		}
	})
}

func TestNetworkError_NilErrPanics(t *testing.T) {
	t.Parallel()
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on NetworkError{Err: nil}.Error(), got none")
		}
	}()
	e := &inference.NetworkError{Err: nil}
	_ = e.Error() // must panic
}

func TestAPIError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		err          *inference.APIError
		wantContains []string
	}{
		{
			name:         "status 200 with message",
			err:          &inference.APIError{Status: 200, Message: "ok"},
			wantContains: []string{"200", "ok"},
		},
		{
			name:         "status 429 rate limited",
			err:          &inference.APIError{Status: 429, Message: "rate limited"},
			wantContains: []string{"429", "rate limited"},
		},
		{
			name:         "status 500 server error",
			err:          &inference.APIError{Status: 500, Message: "internal server error"},
			wantContains: []string{"500", "internal server error"},
		},
		{
			name:         "empty message boundary",
			err:          &inference.APIError{Status: 503, Message: ""},
			wantContains: []string{"503"},
		},
		{
			name:         "error string has inference prefix",
			err:          &inference.APIError{Status: 400, Message: "bad request"},
			wantContains: []string{"inference:"},
		},
		{
			name:         "nil body is accepted without panic",
			err:          &inference.APIError{Status: 404, Message: "not found", Body: nil},
			wantContains: []string{"404", "not found"},
		},
		{
			name:         "non-nil body does not affect Error() output",
			err:          &inference.APIError{Status: 401, Message: "unauthorized", Body: []byte(`{"error":"auth"}`)},
			wantContains: []string{"401", "unauthorized"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := tt.err.Error()
			if got == "" {
				t.Fatalf("Error() returned empty string")
			}
			for _, want := range tt.wantContains {
				if !strings.Contains(got, want) {
					t.Errorf("Error() = %q, want it to contain %q", got, want)
				}
			}
		})
	}
}

func TestValidationError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		err          *inference.ValidationError
		wantContains []string
	}{
		{
			name:         "field and reason both present",
			err:          &inference.ValidationError{Field: "temperature", Reason: "must be between 0 and 1"},
			wantContains: []string{"temperature", "must be between 0 and 1"},
		},
		{
			name:         "error string has inference prefix",
			err:          &inference.ValidationError{Field: "model", Reason: "unknown model"},
			wantContains: []string{"inference:"},
		},
		{
			name:         "empty field boundary",
			err:          &inference.ValidationError{Field: "", Reason: "missing required field"},
			wantContains: []string{"missing required field"},
		},
		{
			name:         "empty reason boundary",
			err:          &inference.ValidationError{Field: "max_tokens", Reason: ""},
			wantContains: []string{"max_tokens"},
		},
		{
			name:         "both empty boundary produces non-empty string",
			err:          &inference.ValidationError{Field: "", Reason: ""},
			wantContains: []string{"inference:"},
		},
		{
			name:         "messages field with long reason",
			err:          &inference.ValidationError{Field: "messages", Reason: "must not be empty"},
			wantContains: []string{"messages", "must not be empty"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := tt.err.Error()
			if got == "" {
				t.Fatalf("Error() returned empty string")
			}
			for _, want := range tt.wantContains {
				if !strings.Contains(got, want) {
					t.Errorf("Error() = %q, want it to contain %q", got, want)
				}
			}
		})
	}
}

func TestMissingCredentialsError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		err          *inference.MissingCredentialsError
		wantContains []string
	}{
		{
			name:         "names the missing credential",
			err:          &inference.MissingCredentialsError{Credential: "Authorization"},
			wantContains: []string{"inference:", "Authorization"},
		},
		{
			name:         "api key header",
			err:          &inference.MissingCredentialsError{Credential: "x-api-key"},
			wantContains: []string{"x-api-key"},
		},
		{
			name:         "empty credential boundary still produces prefixed string",
			err:          &inference.MissingCredentialsError{Credential: ""},
			wantContains: []string{"inference:", "missing credential"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := tt.err.Error()
			if got == "" {
				t.Fatalf("Error() returned empty string")
			}
			for _, want := range tt.wantContains {
				if !strings.Contains(got, want) {
					t.Errorf("Error() = %q, want it to contain %q", got, want)
				}
			}
		})
	}

	// The error is usable as a typed error via errors.As.
	t.Run("errors.As recovers the concrete type", func(t *testing.T) {
		t.Parallel()
		var wrapped error = fmt.Errorf("authorize failed: %w", &inference.MissingCredentialsError{Credential: "Authorization"})
		var mce *inference.MissingCredentialsError
		if !errors.As(wrapped, &mce) {
			t.Fatalf("errors.As failed to recover *MissingCredentialsError")
		}
		if mce.Credential != "Authorization" {
			t.Errorf("Credential = %q, want Authorization", mce.Credential)
		}
	})
}

func TestModelMismatchError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		err          *inference.ModelMismatchError
		wantContains []string
	}{
		{
			name: "provider mismatch",
			err: &inference.ModelMismatchError{
				BoundProvider:   inference.ProviderName("chutes"),
				RequestProvider: inference.ProviderName("phala"),
				BoundEndpoint:   "https://api.chutes.ai",
				RequestEndpoint: "https://api.phala.network",
			},
			wantContains: []string{"inference:", "chutes", "phala", "api.chutes.ai", "api.phala.network"},
		},
		{
			name: "empty request provider is a wildcard but still rendered",
			err: &inference.ModelMismatchError{
				BoundProvider:   inference.ProviderName("chutes"),
				RequestProvider: inference.ProviderName(""),
				BoundEndpoint:   "https://api.chutes.ai",
				RequestEndpoint: "https://api.chutes.ai",
			},
			wantContains: []string{"inference:", "chutes"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := tt.err.Error()
			if got == "" {
				t.Fatalf("Error() returned empty string")
			}
			for _, want := range tt.wantContains {
				if !strings.Contains(got, want) {
					t.Errorf("Error() = %q, want it to contain %q", got, want)
				}
			}
		})
	}
}
