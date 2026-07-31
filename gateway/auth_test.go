package gateway_test

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/looprig/inference/gateway"
)

// TestAuth_StaticToken_ValidBearerToken_Succeeds proves the happy path: a
// correctly formed and matching bearer token authenticates cleanly.
func TestAuth_StaticToken_ValidBearerToken_Succeeds(t *testing.T) {
	t.Parallel()
	authn := gateway.StaticToken("correct-token")
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
	req.Header.Set("Authorization", "Bearer correct-token")
	if err := authn.Authenticate(req); err != nil {
		t.Errorf("Authenticate: unexpected error: %v", err)
	}
}

// TestAuth_StaticToken_MissingHeader_Fails proves no Authorization header at
// all fails authentication.
func TestAuth_StaticToken_MissingHeader_Fails(t *testing.T) {
	t.Parallel()
	authn := gateway.StaticToken("correct-token")
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
	assertAuthenticationError(t, authn.Authenticate(req))
}

// TestAuth_StaticToken_NonBearerScheme_Fails proves a non-Bearer
// Authorization scheme (e.g. Basic) fails.
func TestAuth_StaticToken_NonBearerScheme_Fails(t *testing.T) {
	t.Parallel()
	authn := gateway.StaticToken("correct-token")
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
	req.Header.Set("Authorization", "Basic Y29ycmVjdC10b2tlbg==")
	assertAuthenticationError(t, authn.Authenticate(req))
}

// TestAuth_StaticToken_WrongToken_Fails proves a well-formed but wrong
// bearer token fails, at both a shorter and a longer length than the real
// token (both branches of the constant-time-compare length guard).
func TestAuth_StaticToken_WrongToken_Fails(t *testing.T) {
	t.Parallel()
	authn := gateway.StaticToken("correct-token")
	for _, tc := range []string{"wrong", "wrong-token-that-is-much-longer-than-correct", "correct-tokeX"} {
		req := httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
		req.Header.Set("Authorization", "Bearer "+tc)
		assertAuthenticationError(t, authn.Authenticate(req))
	}
}

// TestAuth_StaticToken_EmptyBearerValue_Fails proves "Bearer " with nothing
// after it fails (and does not, say, match a StaticToken("") configuration
// meant for a different purpose than "no auth" -- this package has no
// implicit no-auth value).
func TestAuth_StaticToken_EmptyBearerValue_Fails(t *testing.T) {
	t.Parallel()
	authn := gateway.StaticToken("correct-token")
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
	req.Header.Set("Authorization", "Bearer ")
	assertAuthenticationError(t, authn.Authenticate(req))
}

// TestAuth_StaticToken_OversizedHeader_Fails proves the header-bounds
// defense: an absurdly long Authorization header is rejected outright rather
// than being handed to a constant-time compare.
func TestAuth_StaticToken_OversizedHeader_Fails(t *testing.T) {
	t.Parallel()
	authn := gateway.StaticToken("correct-token")
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
	req.Header.Set("Authorization", "Bearer "+strings.Repeat("a", 1<<20))
	assertAuthenticationError(t, authn.Authenticate(req))
}

// TestAuth_StaticToken_AllFailureModesReportIdenticalError locks in the
// information-disclosure requirement: every distinct failure cause (missing
// header, wrong scheme, wrong token, oversized header) must produce an
// identical error message, so a response or log built from it can never
// leak *why* authentication failed.
func TestAuth_StaticToken_AllFailureModesReportIdenticalError(t *testing.T) {
	t.Parallel()
	authn := gateway.StaticToken("correct-token")

	missing := httptest.NewRequest(http.MethodPost, "/v1/messages", nil)

	wrongScheme := httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
	wrongScheme.Header.Set("Authorization", "Basic xyz")

	wrongToken := httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
	wrongToken.Header.Set("Authorization", "Bearer nope")

	oversized := httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
	oversized.Header.Set("Authorization", "Bearer "+strings.Repeat("a", 1<<20))

	errs := []error{
		authn.Authenticate(missing),
		authn.Authenticate(wrongScheme),
		authn.Authenticate(wrongToken),
		authn.Authenticate(oversized),
	}
	want := errs[0].Error()
	for i, err := range errs {
		if err == nil {
			t.Fatalf("errs[%d]: expected an error, got nil", i)
		}
		if err.Error() != want {
			t.Errorf("errs[%d].Error() = %q, want %q (all failure causes must be indistinguishable)", i, err.Error(), want)
		}
	}
}

// TestAuth_StaticToken_ImplementsAuthenticator is a compile-time-flavored
// sanity check that StaticToken's return value satisfies Authenticator.
func TestAuth_StaticToken_ImplementsAuthenticator(t *testing.T) {
	t.Parallel()
	var _ gateway.Authenticator = gateway.StaticToken("x")
}

func assertAuthenticationError(t *testing.T, err error) {
	t.Helper()
	if err == nil {
		t.Fatal("Authenticate: expected an error, got nil")
	}
	var authErr *gateway.AuthenticationError
	if !errors.As(err, &authErr) {
		t.Fatalf("Authenticate error = %v (%T), want *gateway.AuthenticationError", err, err)
	}
}
