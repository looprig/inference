// Package auth provides generic inference.Authenticator implementations: static
// header and API-key bearer auth, plus an explicit no-auth value. It imports inference
// for the interface; inference never imports auth. Provider-specific authenticators
// (e.g. cloud request-signing schemes) live in the llm module, not here.
package auth

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"

	"github.com/looprig/inference"
)

// APIKey is a bearer/API-key secret. A named type so a base URL cannot be passed where
// a key belongs, and so provider constructors can demand it at compile time.
type APIKey string

// headerAuth sets one request header to a fixed value. Its String/LogValue/GoString
// methods redact the value so the secret never leaks through any format verb or log.
type headerAuth struct{ name, value string }

func (h headerAuth) Authorize(_ context.Context, r *http.Request) error {
	r.Header.Set(h.name, h.value)
	return nil
}

// String redacts the credential so %v, %+v, and %s never expose the secret header value.
func (headerAuth) String() string { return "auth.headerAuth(REDACTED)" }

// LogValue redacts the credential for slog structured logging.
func (headerAuth) LogValue() slog.Value { return slog.StringValue("REDACTED") }

// GoString redacts the credential under the %#v verb (fmt.GoStringer) so even
// Go-syntax debug formatting never exposes the secret header value.
func (headerAuth) GoString() string { return "auth.headerAuth(REDACTED)" }

var (
	_ fmt.Stringer   = headerAuth{}
	_ fmt.GoStringer = headerAuth{}
	_ slog.LogValuer = headerAuth{}
)

// Key returns an Authenticator that sets "Authorization: Bearer <k>".
func Key(k APIKey) inference.Authenticator {
	return headerAuth{name: "Authorization", value: "Bearer " + string(k)}
}

// Header returns an Authenticator that sets an arbitrary header (e.g. "x-api-key") to
// the key value.
func Header(k APIKey, name string) inference.Authenticator {
	return headerAuth{name: name, value: string(k)}
}

type noneAuth struct{}

func (noneAuth) Authorize(context.Context, *http.Request) error { return nil }

// None returns an Authenticator that adds no credentials — the explicit, visible "no
// auth" value (never a zero-value default).
func None() inference.Authenticator { return noneAuth{} }
