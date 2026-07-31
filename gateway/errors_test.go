package gateway_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/looprig/inference/gateway"
	"github.com/looprig/inference/model"
)

// TestConfigError_ErrorAndUnwrap verifies ConfigError formats a readable
// message including its Location and Reason, and that a wrapped cause is
// reachable via errors.Is/errors.As (e.g. a *model.ValidationError produced
// by NewMux's Model validation step).
func TestConfigError_ErrorAndUnwrap(t *testing.T) {
	t.Parallel()
	cause := &model.ValidationError{Field: "Name", Reason: "model name must not be empty"}
	err := &gateway.ConfigError{Location: "Routes[anthropic/primary]", Reason: "invalid Model", Err: cause}

	msg := err.Error()
	for _, want := range []string{"Routes[anthropic/primary]", "invalid Model"} {
		if !strings.Contains(msg, want) {
			t.Errorf("ConfigError.Error() = %q, want substring %q", msg, want)
		}
	}

	var got *model.ValidationError
	if !errors.As(err, &got) {
		t.Fatalf("errors.As(err, *model.ValidationError) failed for %v", err)
	}
	if got != cause {
		t.Errorf("unwrapped cause = %v, want %v", got, cause)
	}
	if !errors.Is(err, cause) {
		t.Errorf("errors.Is(err, cause) = false, want true")
	}
}

// TestConfigError_ErrorWithoutWrappedCause verifies ConfigError still
// produces a sensible message when Err is nil (e.g. the nil-Client and
// duplicate-ID cases, which have no underlying error to wrap).
func TestConfigError_ErrorWithoutWrappedCause(t *testing.T) {
	t.Parallel()
	err := &gateway.ConfigError{Location: "Default", Reason: "Client must not be nil"}
	msg := err.Error()
	for _, want := range []string{"Default", "Client must not be nil"} {
		if !strings.Contains(msg, want) {
			t.Errorf("ConfigError.Error() = %q, want substring %q", msg, want)
		}
	}
}

// TestRouteNotFoundError_Error verifies the message identifies the ingress
// format and requested model alias that failed to resolve.
func TestRouteNotFoundError_Error(t *testing.T) {
	t.Parallel()
	err := &gateway.RouteNotFoundError{Ingress: model.APIFormatOpenAI, Model: "ghost"}
	msg := err.Error()
	for _, want := range []string{string(model.APIFormatOpenAI), "ghost"} {
		if !strings.Contains(msg, want) {
			t.Errorf("RouteNotFoundError.Error() = %q, want substring %q", msg, want)
		}
	}
}

// TestConfigError_And_RouteNotFoundError_AreDistinctTypes verifies the two
// typed error categories owned by this package (invalid configuration vs.
// route not found) are distinguishable via errors.As and never satisfy each
// other's type assertion.
func TestConfigError_And_RouteNotFoundError_AreDistinctTypes(t *testing.T) {
	t.Parallel()
	var routeErr error = &gateway.RouteNotFoundError{Ingress: model.APIFormatGemini, Model: "x"}
	var cfgErr *gateway.ConfigError
	if errors.As(routeErr, &cfgErr) {
		t.Error("RouteNotFoundError must not satisfy errors.As(*ConfigError)")
	}

	var cfg error = &gateway.ConfigError{Location: "x", Reason: "y"}
	var notFound *gateway.RouteNotFoundError
	if errors.As(cfg, &notFound) {
		t.Error("ConfigError must not satisfy errors.As(*RouteNotFoundError)")
	}
}
