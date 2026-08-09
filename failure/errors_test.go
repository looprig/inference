package failure_test

import (
	"bytes"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"testing"

	auth "github.com/looprig/inference/auth"
	contextcount "github.com/looprig/inference/contextcount"
	failure "github.com/looprig/inference/failure"
	model "github.com/looprig/inference/model"
)

// Compile-time assertions that all error types satisfy the error interface.
// These are intentionally outside the table-driven pattern — they carry no runtime behavior.
var (
	_ error = (*failure.NetworkError)(nil)
	_ error = (*failure.APIError)(nil)
	_ error = (*model.ValidationError)(nil)
	_ error = (*auth.MissingCredentialsError)(nil)
	_ error = (*failure.ModelMismatchError)(nil)
	_ error = (*contextcount.ContextCountError)(nil)
	_ error = (*contextcount.CapabilityValidationError)(nil)
	_ error = (*contextcount.CounterCompatibilityError)(nil)
)

func TestContextCounterErrors(t *testing.T) {
	t.Parallel()

	cause := &model.ModelKeyValidationError{Field: model.ModelKeyFieldModel, Reason: model.ModelKeyValidationReasonEmpty}
	tests := []struct {
		name         string
		err          error
		wantContains []string
		wantCause    error
	}{
		{name: "context count error names model quality and cause", err: &contextcount.ContextCountError{Model: model.ModelKey{Provider: "provider", Model: "model"}, Quality: contextcount.CountQualityExactLocal, Cause: cause}, wantContains: []string{"inference:", "provider", "model", "2", "must not be empty"}, wantCause: cause},
		{name: "capability validation error names metadata", err: &contextcount.CapabilityValidationError{Capability: contextcount.CapabilityKindCounter, Field: contextcount.CapabilityFieldQuality, Reason: contextcount.CapabilityValidationReasonUnknown}, wantContains: []string{"inference:", "counter", "Quality", "unknown"}},
		{name: "compatibility error names downgrade", err: &contextcount.CounterCompatibilityError{Reason: contextcount.CounterCompatibilityTransportDowngrade}, wantContains: []string{"inference:", "transport downgrade"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := tt.err.Error()
			for _, want := range tt.wantContains {
				if !strings.Contains(got, want) {
					t.Errorf("Error() = %q, want it to contain %q", got, want)
				}
			}
			if tt.wantCause != nil && !errors.Is(tt.err, tt.wantCause) {
				t.Errorf("errors.Is(%v, %v) = false, want true", tt.err, tt.wantCause)
			}
		})
	}
}

func TestNetworkError(t *testing.T) {
	t.Parallel()

	inner := errors.New("connection refused")

	tests := []struct {
		name         string
		err          *failure.NetworkError
		wantContains string
		wantUnwrap   error
	}{
		{
			name:         "wraps inner error message in output",
			err:          &failure.NetworkError{Err: inner},
			wantContains: "connection refused",
			wantUnwrap:   inner,
		},
		{
			name:         "error string has inference prefix",
			err:          &failure.NetworkError{Err: inner},
			wantContains: "inference:",
			wantUnwrap:   inner,
		},
		{
			name:         "wraps a wrapped error and unwrap chain works",
			err:          &failure.NetworkError{Err: fmt.Errorf("dial tcp: %w", inner)},
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
		ne := &failure.NetworkError{Err: wrapped}
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
	e := &failure.NetworkError{Err: nil}
	_ = e.Error() // must panic
}

func TestAPIError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		err          *failure.APIError
		wantContains []string
	}{
		{
			name:         "status 200 with message",
			err:          &failure.APIError{Status: 200, Message: "ok"},
			wantContains: []string{"200", "ok"},
		},
		{
			name:         "status 429 rate limited",
			err:          &failure.APIError{Status: 429, Message: "rate limited"},
			wantContains: []string{"429", "rate limited"},
		},
		{
			name:         "status 500 server error",
			err:          &failure.APIError{Status: 500, Message: "internal server error"},
			wantContains: []string{"500", "internal server error"},
		},
		{
			name:         "empty message boundary",
			err:          &failure.APIError{Status: 503, Message: ""},
			wantContains: []string{"503"},
		},
		{
			name:         "error string has inference prefix",
			err:          &failure.APIError{Status: 400, Message: "bad request"},
			wantContains: []string{"inference:"},
		},
		{
			name:         "nil body is accepted without panic",
			err:          &failure.APIError{Status: 404, Message: "not found", Body: nil},
			wantContains: []string{"404", "not found"},
		},
		{
			name:         "non-nil body does not affect Error() output",
			err:          &failure.APIError{Status: 401, Message: "unauthorized", Body: []byte(`{"error":"auth"}`)},
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

func TestAPIErrorRedactsRawProviderBodyAndHeaders(t *testing.T) {
	t.Parallel()
	const secretBody = `{"error":{"message":"provider-token-should-not-escape","code":"not-an-allowlisted-code"}}`
	err := &failure.APIError{
		Status:    500,
		Message:   secretBody,
		Body:      []byte(secretBody),
		Code:      "not-an-allowlisted-code",
		RequestID: "request-safe-123",
	}
	for _, format := range []string{"%s", "%v", "%+v", "%#v", "%q"} {
		got := fmt.Sprintf(format, err)
		if strings.Contains(got, "provider-token-should-not-escape") || strings.Contains(got, secretBody) {
			t.Errorf("format %q leaked provider body: %q", format, got)
		}
	}
	valueFormat := fmt.Sprintf("%#v", *err)
	if strings.Contains(valueFormat, "provider-token-should-not-escape") || strings.Contains(valueFormat, secretBody) {
		t.Fatalf("value formatting leaked provider body: %q", valueFormat)
	}
	var logs bytes.Buffer
	slog.New(slog.NewTextHandler(&logs, nil)).Info("failure", "error", err)
	if strings.Contains(logs.String(), "provider-token-should-not-escape") || strings.Contains(logs.String(), secretBody) {
		t.Fatalf("slog leaked provider body: %q", logs.String())
	}
}

func TestValidationError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		err          *model.ValidationError
		wantContains []string
	}{
		{
			name:         "field and reason both present",
			err:          &model.ValidationError{Field: "temperature", Reason: "must be between 0 and 1"},
			wantContains: []string{"temperature", "must be between 0 and 1"},
		},
		{
			name:         "error string has inference prefix",
			err:          &model.ValidationError{Field: "model", Reason: "unknown model"},
			wantContains: []string{"inference:"},
		},
		{
			name:         "empty field boundary",
			err:          &model.ValidationError{Field: "", Reason: "missing required field"},
			wantContains: []string{"missing required field"},
		},
		{
			name:         "empty reason boundary",
			err:          &model.ValidationError{Field: "max_tokens", Reason: ""},
			wantContains: []string{"max_tokens"},
		},
		{
			name:         "both empty boundary produces non-empty string",
			err:          &model.ValidationError{Field: "", Reason: ""},
			wantContains: []string{"inference:"},
		},
		{
			name:         "messages field with long reason",
			err:          &model.ValidationError{Field: "messages", Reason: "must not be empty"},
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
		err          *auth.MissingCredentialsError
		wantContains []string
	}{
		{
			name:         "names the missing credential",
			err:          &auth.MissingCredentialsError{Credential: "Authorization"},
			wantContains: []string{"inference:", "Authorization"},
		},
		{
			name:         "api key header",
			err:          &auth.MissingCredentialsError{Credential: "x-api-key"},
			wantContains: []string{"x-api-key"},
		},
		{
			name:         "empty credential boundary still produces prefixed string",
			err:          &auth.MissingCredentialsError{Credential: ""},
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
		var wrapped error = fmt.Errorf("authorize failed: %w", &auth.MissingCredentialsError{Credential: "Authorization"})
		var mce *auth.MissingCredentialsError
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
		err          *failure.ModelMismatchError
		wantContains []string
	}{
		{
			name: "provider mismatch",
			err: &failure.ModelMismatchError{
				BoundProvider:   model.ProviderName("chutes"),
				RequestProvider: model.ProviderName("phala"),
				BoundEndpoint:   "https://api.chutes.ai",
				RequestEndpoint: "https://api.phala.network",
			},
			wantContains: []string{"inference:", "chutes", "phala", "api.chutes.ai", "api.phala.network"},
		},
		{
			name: "empty request provider is a wildcard but still rendered",
			err: &failure.ModelMismatchError{
				BoundProvider:   model.ProviderName("chutes"),
				RequestProvider: model.ProviderName(""),
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
