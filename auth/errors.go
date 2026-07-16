package auth

import "fmt"

// AuthKind classifies the credential an authentication mechanism requires.
type AuthKind string

const (
	AuthNone   AuthKind = "none"
	AuthAPIKey AuthKind = "api_key"
)

// MissingCredentialsError reports a required credential that was not supplied.
type MissingCredentialsError struct {
	Credential string
}

func (e *MissingCredentialsError) Error() string {
	return fmt.Sprintf("inference: missing credential: %s", e.Credential)
}
