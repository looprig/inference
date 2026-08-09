package auth_test

import (
	"context"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/looprig/credentials/httpauth"
	"github.com/looprig/inference/auth"
)

// Compile-time proof the constructors yield auth.Authenticator values.
var (
	_ auth.Authenticator  = auth.Key("k")
	_ auth.Authenticator  = auth.Header("k", "x-api-key")
	_ auth.Authenticator  = auth.None()
	_ auth.Authorizer     = auth.Key("k")
	_ httpauth.Authorizer = auth.Key("k")
)

func TestKeySetsBearer(t *testing.T) {
	t.Parallel()
	req, _ := http.NewRequest(http.MethodPost, "https://x.test", nil)
	if err := auth.Key("sekret").Authorize(context.Background(), req); err != nil {
		t.Fatalf("Authorize: %v", err)
	}
	if got := req.Header.Get("Authorization"); got != "Bearer sekret" {
		t.Errorf("Authorization = %q, want %q", got, "Bearer sekret")
	}
}

func TestHeaderSetsCustom(t *testing.T) {
	t.Parallel()
	req, _ := http.NewRequest(http.MethodPost, "https://x.test", nil)
	if err := auth.Header("sekret", "x-api-key").Authorize(context.Background(), req); err != nil {
		t.Fatalf("Authorize: %v", err)
	}
	if got := req.Header.Get("x-api-key"); got != "sekret" {
		t.Errorf("x-api-key = %q, want %q", got, "sekret")
	}
	if req.Header.Get("Authorization") != "" {
		t.Errorf("Header must not set Authorization")
	}
}

func TestHeaderReplacesCaseInsensitiveStaleValues(t *testing.T) {
	t.Parallel()
	req, _ := http.NewRequest(http.MethodPost, "https://x.test", nil)
	req.Header["X-Api-Key"] = []string{"stale"}
	req.Header["x-api-key"] = []string{"older"}
	if err := auth.Header("current", "X-API-KEY").Authorize(context.Background(), req); err != nil {
		t.Fatalf("Authorize: %v", err)
	}
	if got, want := req.Header.Values("X-Api-Key"), []string{"current"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("header values = %#v, want %#v", got, want)
	}
	for key, values := range req.Header {
		if strings.EqualFold(key, "X-Api-Key") && !reflect.DeepEqual(values, []string{"current"}) {
			t.Fatalf("stale key %q remained: %#v", key, values)
		}
	}
}

func TestNoneIsNoop(t *testing.T) {
	t.Parallel()
	req, _ := http.NewRequest(http.MethodPost, "https://x.test", nil)
	if err := auth.None().Authorize(context.Background(), req); err != nil {
		t.Fatalf("Authorize: %v", err)
	}
	if len(req.Header) != 0 {
		t.Errorf("None must not mutate headers, got %v", req.Header)
	}
}

func TestAuthenticatorRedactsSecret(t *testing.T) {
	t.Parallel()
	const secret = "supersecret-token"
	auths := []struct {
		name string
		a    auth.Authenticator
	}{
		{name: "Key", a: auth.Key(secret)},
		{name: "Header", a: auth.Header(secret, "x-api-key")},
	}
	for _, tt := range auths {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			for _, s := range []string{
				fmt.Sprintf("%v", tt.a),
				fmt.Sprintf("%+v", tt.a),
				fmt.Sprintf("%s", tt.a),
				fmt.Sprintf("%#v", tt.a),
			} {
				if strings.Contains(s, secret) {
					t.Errorf("formatted authenticator leaked secret: %q", s)
				}
			}
		})
	}
}
