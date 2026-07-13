package inference

import (
	"errors"
	"fmt"
)

// Leaf causes used by ContextCountError when no lower-level error exists.
var (
	ErrContextCountFunctionMissing           = errors.New("context count function is missing")
	ErrContextCountQualityInvalid            = errors.New("context count quality is invalid")
	ErrContextCountModelMismatch             = errors.New("context count model does not match request model")
	ErrContextCountCapabilityQualityMismatch = errors.New("context count quality does not match declared capability")
)

// ContextCountError reports a structurally invalid count result or adapter.
type ContextCountError struct {
	Model   ModelKey
	Quality CountQuality
	Cause   error
}

func (e *ContextCountError) Error() string {
	cause := "unknown cause"
	if e.Cause != nil {
		cause = e.Cause.Error()
	}
	return fmt.Sprintf("inference: context count for model %q/%q with quality %d failed: %s", e.Model.Provider, e.Model.Model, e.Quality, cause)
}

func (e *ContextCountError) Unwrap() error { return e.Cause }

// CapabilityKind identifies which trust posture failed validation.
type CapabilityKind string

const (
	CapabilityKindCounter   CapabilityKind = "counter"
	CapabilityKindInference CapabilityKind = "inference"
)

// CapabilityField identifies a structurally invalid capability field.
type CapabilityField string

const (
	CapabilityFieldProvider          CapabilityField = "Provider"
	CapabilityFieldTransport         CapabilityField = "Transport"
	CapabilityFieldSecurityIdentity  CapabilityField = "SecurityIdentity"
	CapabilityFieldRetention         CapabilityField = "Retention"
	CapabilityFieldTokenizerRevision CapabilityField = "TokenizerRev"
	CapabilityFieldQuality           CapabilityField = "Quality"
)

// CapabilityValidationReason identifies why capability metadata is invalid.
type CapabilityValidationReason string

const (
	CapabilityValidationReasonUnknown    CapabilityValidationReason = "unknown"
	CapabilityValidationReasonOutOfRange CapabilityValidationReason = "out of range"
	CapabilityValidationReasonEmpty      CapabilityValidationReason = "must not be empty"
	CapabilityValidationReasonMustBeZero CapabilityValidationReason = "must be zero"
)

// CapabilityValidationError reports invalid counter or inference metadata.
type CapabilityValidationError struct {
	Capability CapabilityKind
	Field      CapabilityField
	Reason     CapabilityValidationReason
}

func (e *CapabilityValidationError) Error() string {
	return fmt.Sprintf("inference: invalid %s capability field %s: %s", e.Capability, e.Field, e.Reason)
}

// CounterCompatibilityReason identifies why a counter would weaken inference.
type CounterCompatibilityReason string

const (
	CounterCompatibilityInvalidInference   CounterCompatibilityReason = "invalid inference capability"
	CounterCompatibilityInvalidCounter     CounterCompatibilityReason = "invalid counter capability"
	CounterCompatibilityProviderMismatch   CounterCompatibilityReason = "provider mismatch"
	CounterCompatibilityIdentityMismatch   CounterCompatibilityReason = "security identity mismatch"
	CounterCompatibilityTransportDowngrade CounterCompatibilityReason = "transport downgrade"
	CounterCompatibilityRetentionDowngrade CounterCompatibilityReason = "retention downgrade"
)

// CounterCompatibilityError reports why a counter is unacceptable for an
// inference path and preserves both inputs for deterministic inspection.
type CounterCompatibilityError struct {
	Inference InferenceCapability
	Counter   CounterCapability
	Reason    CounterCompatibilityReason
	Cause     error
}

func (e *CounterCompatibilityError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("inference: incompatible counter: %s: %v", e.Reason, e.Cause)
	}
	return "inference: incompatible counter: " + string(e.Reason)
}

func (e *CounterCompatibilityError) Unwrap() error { return e.Cause }

// NetworkError wraps a transport-level failure (DNS, TCP, TLS).
// Err must not be nil; a nil Err will panic on Error() — callers are responsible for providing a cause.
type NetworkError struct {
	Err error
}

func (e *NetworkError) Error() string { return "inference: network error: " + e.Err.Error() }
func (e *NetworkError) Unwrap() error { return e.Err }

// APIError is a non-2xx response from the provider.
// Body holds the raw response payload for provider-specific error parsing and may be nil.
type APIError struct {
	Status  int
	Message string
	Body    []byte
}

func (e *APIError) Error() string {
	return fmt.Sprintf("inference: api error %d: %s", e.Status, e.Message)
}

// ValidationError is a self-contradictory request rejected before sending to the provider.
// Field identifies which request parameter is invalid; Reason explains the constraint violated.
type ValidationError struct {
	Field  string
	Reason string
}

func (e *ValidationError) Error() string {
	return fmt.Sprintf("inference: validation error: %s: %s", e.Field, e.Reason)
}

// AuthKind classifies the credential a generic auth mechanism requires. Provider-specific
// credential kinds (e.g. cloud request-signing) live in the llm module, not here.
type AuthKind string

const (
	AuthNone   AuthKind = "none"
	AuthAPIKey AuthKind = "api_key"
)

// MissingCredentialsError is returned by a generic auth mechanism when a required credential
// or header value is absent. It names the missing credential without depending on any provider
// policy table. Fail-closed: an authorize step returning this error must block the request.
// Carries no secret.
type MissingCredentialsError struct {
	Credential string
}

func (e *MissingCredentialsError) Error() string {
	return fmt.Sprintf("inference: missing credential: %s", e.Credential)
}

// ModelMismatchError is returned before any network I/O when a Request's model names a
// provider, endpoint, or API format that differs from the connection the client is bound
// to. Fail-closed. All three binding dimensions are carried so a caller doing errors.As
// can see exactly which one conflicted; an empty request-side field is a wildcard, not a
// claim, so it never triggers this error.
type ModelMismatchError struct {
	BoundProvider    ProviderName
	RequestProvider  ProviderName
	BoundEndpoint    string
	RequestEndpoint  string
	BoundAPIFormat   APIFormat
	RequestAPIFormat APIFormat
}

func (e *ModelMismatchError) Error() string {
	return fmt.Sprintf("inference: request model provider %q/endpoint %q/format %q does not match bound client %q/%q/%q",
		e.RequestProvider, e.RequestEndpoint, e.RequestAPIFormat,
		e.BoundProvider, e.BoundEndpoint, e.BoundAPIFormat)
}
