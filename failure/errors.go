// Package failure owns provider-neutral inference failures shared by codecs,
// transports, and provider integrations.
package failure

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/looprig/inference/model"
)

type NetworkError struct {
	Err error
}

func (e *NetworkError) Error() string { return "inference: network error: " + e.Err.Error() }
func (e *NetworkError) Unwrap() error { return e.Err }

type APIError struct {
	Status  int
	Message string
	// Code is a provider error code from the closed allowlist below. It is
	// intentionally not a free-form provider message.
	Code string
	// ProviderCode is an additive descriptive alias for Code. Both fields are
	// normalized before they are emitted; NewAPIError populates both.
	ProviderCode string
	// RequestID is copied only from known request-ID headers and only when it
	// is bounded and printable. Raw response headers are never retained.
	RequestID string
	// Body is retained as a deprecated source-compatibility field for callers
	// that still compile struct literals. Transport-created APIErrors leave it
	// nil; formatting and logging never include it.
	Body []byte
	// RetryAfter is the server-advertised wait from a Retry-After header,
	// integer-seconds form only. Zero means absent or unparseable.
	RetryAfter time.Duration
}

func (e *APIError) Error() string {
	if e == nil {
		return "inference: api error"
	}
	message := safeAPIMessage(e.Message)
	code := e.safeCode()
	if code != "" {
		message = code
	}
	if message == "" {
		message = http.StatusText(e.Status)
	}
	if message == "" {
		message = "request failed"
	}
	if requestID := safeRequestID(e.RequestID); requestID != "" {
		return fmt.Sprintf("inference: api error %d: %s (request id %s)", e.Status, message, requestID)
	}
	return fmt.Sprintf("inference: api error %d: %s", e.Status, message)
}

// Format prevents fmt's default struct formatting from exposing the
// deprecated Body field or any future unsafe fields.
func (e APIError) Format(state fmt.State, _ rune) {
	_, _ = state.Write([]byte((&e).Error()))
}

func (e APIError) GoString() string { return (&e).Error() }

// LogValue emits only the bounded, allowlisted diagnostics. In particular,
// it does not expose Body, headers, or provider response text.
func (e APIError) LogValue() slog.Value {
	attrs := []slog.Attr{slog.Int("status", e.Status)}
	if code := e.safeCode(); code != "" {
		attrs = append(attrs, slog.String("code", code))
	}
	if requestID := safeRequestID(e.RequestID); requestID != "" {
		attrs = append(attrs, slog.String("request_id", requestID))
	}
	if e.RetryAfter > 0 {
		attrs = append(attrs, slog.Duration("retry_after", e.RetryAfter))
	}
	return slog.GroupValue(attrs...)
}

func (e *APIError) safeCode() string {
	if e == nil {
		return ""
	}
	if code := safeProviderCode(e.Code); code != "" {
		return code
	}
	return safeProviderCode(e.ProviderCode)
}

// NewAPIError constructs an APIError from already-sanitized fields. It is
// useful to provider integrations that classify a response themselves.
func NewAPIError(status int, code, requestID string, retryAfter time.Duration) *APIError {
	return &APIError{
		Status:       status,
		Code:         safeProviderCode(code),
		ProviderCode: safeProviderCode(code),
		RequestID:    safeRequestID(requestID),
		RetryAfter:   retryAfter,
	}
}

const maxProviderCodeLength = 128
const maxRequestIDLength = 256

// APIErrorFromResponse extracts only allowlisted provider codes and request
// IDs. The body is parsed transiently and is never copied into the returned
// error. The caller must bound body before calling this helper; this function
// also truncates defensively for direct callers.
func APIErrorFromResponse(status int, body []byte, headers http.Header, retryAfter time.Duration) *APIError {
	if len(body) > maxErrorBodyBytes {
		body = body[:maxErrorBodyBytes]
	}
	code := providerCodeFromBody(body)
	requestID := requestIDFromHeaders(headers)
	return NewAPIError(status, code, requestID, retryAfter)
}

// MaxErrorBodyBytes is the maximum transient provider-error prefix parsed by
// APIErrorFromResponse. No bytes from that prefix are retained in APIError.
const MaxErrorBodyBytes = 64 << 10

const maxErrorBodyBytes = MaxErrorBodyBytes

var providerCodeAllowlist = map[string]struct{}{
	"aborted": {}, "api_error": {}, "authentication_error": {}, "bad_request": {},
	"billing_hard_limit_reached": {}, "content_policy_violation": {},
	"context_length_exceeded": {}, "deadline_exceeded": {}, "forbidden": {},
	"insufficient_quota": {}, "internal_server_error": {}, "invalid_argument": {},
	"invalid_api_key": {}, "invalid_request_error": {}, "model_not_found": {},
	"not_found": {}, "not_found_error": {}, "overloaded_error": {},
	"permission_denied": {}, "permission_error": {}, "quota_exceeded": {},
	"rate_limit_error": {}, "rate_limit_exceeded": {}, "resource_exhausted": {},
	"server_error": {}, "temporarily_unavailable": {}, "unauthenticated": {},
	"unauthorized": {}, "unavailable": {},
}

func safeProviderCode(value string) string {
	if len(value) == 0 || len(value) > maxProviderCodeLength {
		return ""
	}
	value = strings.ToLower(strings.TrimSpace(value))
	if _, ok := providerCodeAllowlist[value]; !ok {
		return ""
	}
	return value
}

func safeRequestID(value string) string {
	if len(value) == 0 || len(value) > maxRequestIDLength {
		return ""
	}
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	for _, c := range value {
		if c < 0x21 || c > 0x7e || strings.ContainsRune("\"'<>\\", c) {
			return ""
		}
	}
	return value
}

func safeAPIMessage(value string) string {
	// Message is a legacy field. Only retain a small, provider-neutral set of
	// diagnostics that existing callers used; arbitrary response text is not a
	// safe API error message. Gateway-owned messages are already bounded and
	// intentionally carry the requested alias, so retain that package-owned
	// prefix for compatibility with gateway error responses.
	value = strings.TrimSpace(value)
	if strings.HasPrefix(value, "gateway:") {
		if len(value) <= 512 && !strings.ContainsAny(value, "\r\n\t") {
			return value
		}
		return ""
	}
	if len(value) > 128 || strings.ContainsAny(value, "{}[]\r\n\t") {
		return ""
	}
	switch strings.ToLower(value) {
	case "ok", "rate limited", "internal server error", "bad request", "not found", "unauthorized":
		return value
	default:
		return ""
	}
}

func providerCodeFromBody(body []byte) string {
	if len(body) == 0 {
		return ""
	}
	var root map[string]json.RawMessage
	if json.Unmarshal(body, &root) != nil {
		return ""
	}
	for _, key := range []string{"code", "type", "status"} {
		if code := stringField(root[key]); code != "" {
			if safe := safeProviderCode(code); safe != "" {
				return safe
			}
		}
	}
	if raw := root["error"]; len(raw) > 0 {
		var nested map[string]json.RawMessage
		if json.Unmarshal(raw, &nested) == nil {
			for _, key := range []string{"code", "type", "status"} {
				if code := stringField(nested[key]); code != "" {
					if safe := safeProviderCode(code); safe != "" {
						return safe
					}
				}
			}
		}
	}
	return ""
}

func stringField(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var value string
	if json.Unmarshal(raw, &value) != nil {
		return ""
	}
	return value
}

func requestIDFromHeaders(headers http.Header) string {
	if headers == nil {
		return ""
	}
	known := []string{"X-Request-ID", "Request-ID", "X-Goog-Request-Id", "Anthropic-Request-Id"}
	for key, values := range headers {
		for _, candidate := range known {
			if !strings.EqualFold(key, candidate) || len(values) == 0 {
				continue
			}
			if value := safeRequestID(values[0]); value != "" {
				return value
			}
		}
	}
	return ""
}

// ResponseBodyTooLargeError reports a bounded response that cannot safely be
// decoded. It contains only the configured bound.
type ResponseBodyTooLargeError struct{ Limit int }

func (e *ResponseBodyTooLargeError) Error() string {
	if e == nil {
		return "inference: response body exceeds configured limit"
	}
	return "inference: response body exceeds configured limit of " + strconv.Itoa(e.Limit) + " bytes"
}

func (e *ResponseBodyTooLargeError) Format(state fmt.State, _ rune) {
	_, _ = state.Write([]byte(e.Error()))
}

func (e *ResponseBodyTooLargeError) GoString() string { return e.Error() }

type ModelMismatchError struct {
	BoundProvider    model.ProviderName
	RequestProvider  model.ProviderName
	BoundEndpoint    string
	RequestEndpoint  string
	BoundAPIFormat   model.APIFormat
	RequestAPIFormat model.APIFormat
}

func (e *ModelMismatchError) Error() string {
	return fmt.Sprintf("inference: request model provider %q/endpoint %q/format %q does not match bound client %q/%q/%q",
		e.RequestProvider, e.RequestEndpoint, e.RequestAPIFormat,
		e.BoundProvider, e.BoundEndpoint, e.BoundAPIFormat)
}
