package gateway_test

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/looprig/inference/codec"
	"github.com/looprig/inference/codec/anthropicapi"
	"github.com/looprig/inference/gateway"
	"github.com/looprig/inference/model"
)

// TestLimit_Defaults_AreConservativeAndFinite locks in the documented
// default values: a body limit of 10 MiB and a concurrency limit of 64.
func TestLimit_Defaults_AreConservativeAndFinite(t *testing.T) {
	t.Parallel()
	if gateway.DefaultMaxRequestBody != 10<<20 {
		t.Errorf("DefaultMaxRequestBody = %d, want %d (10 MiB)", gateway.DefaultMaxRequestBody, 10<<20)
	}
	if gateway.DefaultMaxConcurrent != 64 {
		t.Errorf("DefaultMaxConcurrent = %d, want 64", gateway.DefaultMaxConcurrent)
	}
	if gateway.DefaultMaxRequestBody <= 0 {
		t.Error("DefaultMaxRequestBody must be finite and positive")
	}
	if gateway.DefaultMaxConcurrent <= 0 {
		t.Error("DefaultMaxConcurrent must be finite and positive")
	}
}

// TestLimit_ZeroConfigFields_ApplyDefaults proves New accepts a Config with
// MaxRequestBody and MaxConcurrent left at their zero value -- the defaults
// are applied silently rather than construction failing or the limits being
// unbounded.
func TestLimit_ZeroConfigFields_ApplyDefaults(t *testing.T) {
	t.Parallel()
	resolver, err := gateway.Fixed(&recordingClient{}, anthropicModel("kimi-k2"))
	if err != nil {
		t.Fatalf("Fixed: %v", err)
	}
	h, err := gateway.New(gateway.Config{
		Resolver:     resolver,
		Codecs:       map[model.APIFormat]codec.ServerCodec{model.APIFormatAnthropic: anthropicapi.Codec{}},
		Authenticate: gateway.StaticToken("test-token"),
		// MaxRequestBody and MaxConcurrent both left zero.
	})
	if err != nil {
		t.Fatalf("New: unexpected error with zero limit fields: %v", err)
	}
	if h == nil {
		t.Fatal("New returned a nil Handler with no error")
	}
}

// TestLimit_NegativeMaxRequestBody_Rejected proves a negative
// Config.MaxRequestBody is a construction-time *ConfigError, not a silent
// unbounded-body configuration.
func TestLimit_NegativeMaxRequestBody_Rejected(t *testing.T) {
	t.Parallel()
	resolver, err := gateway.Fixed(&recordingClient{}, anthropicModel("kimi-k2"))
	if err != nil {
		t.Fatalf("Fixed: %v", err)
	}
	_, err = gateway.New(gateway.Config{
		Resolver:       resolver,
		Codecs:         map[model.APIFormat]codec.ServerCodec{model.APIFormatAnthropic: anthropicapi.Codec{}},
		Authenticate:   gateway.StaticToken("test-token"),
		MaxRequestBody: -1,
	})
	assertGatewayConfigError(t, err)
}

// TestLimit_NegativeMaxConcurrent_Rejected mirrors the above for
// MaxConcurrent.
func TestLimit_NegativeMaxConcurrent_Rejected(t *testing.T) {
	t.Parallel()
	resolver, err := gateway.Fixed(&recordingClient{}, anthropicModel("kimi-k2"))
	if err != nil {
		t.Fatalf("Fixed: %v", err)
	}
	_, err = gateway.New(gateway.Config{
		Resolver:      resolver,
		Codecs:        map[model.APIFormat]codec.ServerCodec{model.APIFormatAnthropic: anthropicapi.Codec{}},
		Authenticate:  gateway.StaticToken("test-token"),
		MaxConcurrent: -1,
	})
	assertGatewayConfigError(t, err)
}

// TestLimit_New_RequiresResolver proves the required-collaborator
// validation New performs beyond limits: a nil Resolver is rejected.
func TestLimit_New_RequiresResolver(t *testing.T) {
	t.Parallel()
	_, err := gateway.New(gateway.Config{
		Codecs:       map[model.APIFormat]codec.ServerCodec{model.APIFormatAnthropic: anthropicapi.Codec{}},
		Authenticate: gateway.StaticToken("test-token"),
	})
	assertGatewayConfigError(t, err)
}

// TestLimit_New_RequiresNonEmptyCodecs proves an empty (or nil) Codecs map
// is rejected.
func TestLimit_New_RequiresNonEmptyCodecs(t *testing.T) {
	t.Parallel()
	resolver, err := gateway.Fixed(&recordingClient{}, anthropicModel("kimi-k2"))
	if err != nil {
		t.Fatalf("Fixed: %v", err)
	}
	_, err = gateway.New(gateway.Config{
		Resolver:     resolver,
		Authenticate: gateway.StaticToken("test-token"),
	})
	assertGatewayConfigError(t, err)
}

// TestLimit_New_RequiresAuthenticate proves a nil Authenticate is rejected.
func TestLimit_New_RequiresAuthenticate(t *testing.T) {
	t.Parallel()
	resolver, err := gateway.Fixed(&recordingClient{}, anthropicModel("kimi-k2"))
	if err != nil {
		t.Fatalf("Fixed: %v", err)
	}
	_, err = gateway.New(gateway.Config{
		Resolver: resolver,
		Codecs:   map[model.APIFormat]codec.ServerCodec{model.APIFormatAnthropic: anthropicapi.Codec{}},
	})
	assertGatewayConfigError(t, err)
}

// TestLimit_New_DuplicateConcreteCodecType_Rejected proves the
// construction-time best-effort duplicate-codec check: registering the same
// concrete codec value under two different Codecs keys is rejected, even
// though a Go map cannot have a literal duplicate key.
func TestLimit_New_DuplicateConcreteCodecType_Rejected(t *testing.T) {
	t.Parallel()
	resolver, err := gateway.Fixed(&recordingClient{}, anthropicModel("kimi-k2"))
	if err != nil {
		t.Fatalf("Fixed: %v", err)
	}
	_, err = gateway.New(gateway.Config{
		Resolver: resolver,
		Codecs: map[model.APIFormat]codec.ServerCodec{
			model.APIFormatAnthropic:       anthropicapi.Codec{},
			model.APIFormatOpenAIResponses: anthropicapi.Codec{}, // same concrete type, different key: rejected
		},
		Authenticate: gateway.StaticToken("test-token"),
	})
	assertGatewayConfigError(t, err)
}

// TestLimit_MaxRequestBody_Enforced is a focused, small-body version of the
// 413 behavior (see handler_test.go's TestHandler_BodyTooLarge_413 for the
// full pipeline proof): a body one byte over an explicit tiny limit is
// rejected, and a body at or under the limit is not rejected for size
// (it may still fail decode for unrelated reasons, which is fine -- this
// test only asserts it is not classified 413).
func TestLimit_MaxRequestBody_Enforced(t *testing.T) {
	t.Parallel()
	const limit = 32
	target := gateway.Target{ID: "t", Client: &recordingClient{}, Model: anthropicModel("kimi-k2")}
	resolver, err := gateway.NewMux(gateway.Mux{
		Routes: map[gateway.RouteKey]gateway.Target{{Ingress: model.APIFormatAnthropic, Model: "primary"}: target},
	})
	if err != nil {
		t.Fatalf("NewMux: %v", err)
	}
	h, err := gateway.New(gateway.Config{
		Resolver:       resolver,
		Codecs:         map[model.APIFormat]codec.ServerCodec{model.APIFormatAnthropic: anthropicapi.Codec{}},
		Authenticate:   gateway.StaticToken("test-token"),
		MaxRequestBody: limit,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	over := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(strings.Repeat("x", limit+1)))
	over.Header.Set("Content-Type", "application/json")
	over.Header.Set("Authorization", "Bearer test-token")
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, over)
	if rr.Code != http.StatusRequestEntityTooLarge {
		t.Errorf("over-limit body: status = %d, want %d; body: %s", rr.Code, http.StatusRequestEntityTooLarge, rr.Body.String())
	}

	under := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(strings.Repeat("x", limit)))
	under.Header.Set("Content-Type", "application/json")
	under.Header.Set("Authorization", "Bearer test-token")
	rr2 := httptest.NewRecorder()
	h.ServeHTTP(rr2, under)
	if rr2.Code == http.StatusRequestEntityTooLarge {
		t.Errorf("at-limit body: status = %d, must not be classified as body-too-large", rr2.Code)
	}
}

func assertGatewayConfigError(t *testing.T, err error) {
	t.Helper()
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	var cfgErr *gateway.ConfigError
	if !errors.As(err, &cfgErr) {
		t.Fatalf("error = %v (%T), want *gateway.ConfigError", err, err)
	}
}
