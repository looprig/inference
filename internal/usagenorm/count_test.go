package usagenorm_test

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference/internal/usagenorm"
	usage "github.com/looprig/inference/usage"
)

func TestCountTokenCount(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		json       string
		want       content.TokenCount
		wantField  usage.UsageNormalizationField
		wantReason usage.UsageNormalizationReason
		wantValue  int64
	}{
		{name: "absent defaults to zero", json: `{}`, want: 0},
		{name: "explicit zero", json: `{"count":0}`, want: 0},
		{name: "maximum int64", json: `{"count":9223372036854775807}`, want: 9223372036854775807},
		{name: "negative", json: `{"count":-1}`, wantField: usage.UsageNormalizationFieldInputTokens, wantReason: usage.UsageNormalizationReasonNegative, wantValue: -1},
		{name: "null", json: `{"count":null}`, wantField: usage.UsageNormalizationFieldInputTokens, wantReason: usage.UsageNormalizationReasonNull},
		{name: "fraction", json: `{"count":1.5}`, wantField: usage.UsageNormalizationFieldInputTokens, wantReason: usage.UsageNormalizationReasonFractional},
		{name: "exponent", json: `{"count":1e3}`, wantField: usage.UsageNormalizationFieldInputTokens, wantReason: usage.UsageNormalizationReasonFractional},
		{name: "out of range positive", json: `{"count":9223372036854775808}`, wantField: usage.UsageNormalizationFieldInputTokens, wantReason: usage.UsageNormalizationReasonOutOfRange},
		{name: "out of range negative", json: `{"count":-9223372036854775809}`, wantField: usage.UsageNormalizationFieldInputTokens, wantReason: usage.UsageNormalizationReasonOutOfRange},
		{name: "string", json: `{"count":"1"}`, wantField: usage.UsageNormalizationFieldInputTokens, wantReason: usage.UsageNormalizationReasonInvalidType},
		{name: "boolean", json: `{"count":true}`, wantField: usage.UsageNormalizationFieldInputTokens, wantReason: usage.UsageNormalizationReasonInvalidType},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			var wire struct {
				Count usagenorm.Count `json:"count"`
			}
			if err := json.Unmarshal([]byte(tt.json), &wire); err != nil {
				t.Fatalf("json.Unmarshal() error = %v", err)
			}
			got, err := wire.Count.TokenCount(usagenorm.FieldInputTokens)
			if tt.wantReason == "" {
				if err != nil {
					t.Fatalf("TokenCount() error = %v", err)
				}
				if got != tt.want {
					t.Errorf("TokenCount() = %d, want %d", got, tt.want)
				}
				return
			}
			assertNormalizationError(t, err, tt.wantField, tt.wantReason, tt.wantValue, 0, 0)
			if !strings.Contains(err.Error(), string(tt.wantField)) || !strings.Contains(err.Error(), string(tt.wantReason)) {
				t.Errorf("Error() = %q, want field %q and reason %q", err, tt.wantField, tt.wantReason)
			}
		})
	}
}

func TestCountErrorDoesNotExposeWireValue(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		json   string
		secret string
	}{
		{name: "string value", json: `{"count":"super-secret-user-data"}`, secret: "super-secret-user-data"},
		{name: "large numeric value", json: `{"count":999999999999999999999999999999999999}`, secret: "999999999999999999999999999999999999"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			var wire struct {
				Count usagenorm.Count `json:"count"`
			}
			if err := json.Unmarshal([]byte(tt.json), &wire); err != nil {
				t.Fatalf("json.Unmarshal() error = %v", err)
			}
			_, err := wire.Count.TokenCount(usagenorm.FieldInputTokens)
			if err == nil {
				t.Fatal("TokenCount() error = nil, want typed failure")
			}
			if strings.Contains(err.Error(), tt.secret) {
				t.Errorf("TokenCount() error exposes wire value: %q", err)
			}
		})
	}
}

func TestCountPresent(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		json string
		want bool
	}{
		{name: "absent", json: `{}`, want: false},
		{name: "zero", json: `{"count":0}`, want: true},
		{name: "null", json: `{"count":null}`, want: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			var wire struct {
				Count usagenorm.Count `json:"count"`
			}
			if err := json.Unmarshal([]byte(tt.json), &wire); err != nil {
				t.Fatalf("json.Unmarshal() error = %v", err)
			}
			if got := wire.Count.Present(); got != tt.want {
				t.Errorf("Present() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCountRejectsUnknownField(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		field usagenorm.Field
	}{
		{name: "zero enum", field: usagenorm.Field(0)},
		{name: "large enum", field: usagenorm.Field(255)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			var count usagenorm.Count
			_, err := count.TokenCount(tt.field)
			assertNormalizationError(t, err, "", usage.UsageNormalizationReasonInvalidField, 0, 0, 0)
		})
	}
}

func assertNormalizationError(t *testing.T, err error, field usage.UsageNormalizationField, reason usage.UsageNormalizationReason, value int64, left, right content.TokenCount) {
	t.Helper()
	var normalizationErr *usage.UsageNormalizationError
	if !errors.As(err, &normalizationErr) {
		t.Fatalf("error = %T %v, want *UsageNormalizationError", err, err)
	}
	if normalizationErr.Field != field || normalizationErr.Reason != reason || normalizationErr.Value != value || normalizationErr.Left != left || normalizationErr.Right != right {
		t.Errorf("normalization error = %+v, want field=%q reason=%q value=%d left=%d right=%d", normalizationErr, field, reason, value, left, right)
	}
}
