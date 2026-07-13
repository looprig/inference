package inference_test

import (
	"context"
	"crypto/sha256"
	"errors"
	"testing"

	"github.com/looprig/inference"
)

func TestCounterCapabilityValidate(t *testing.T) {
	t.Parallel()

	identity := sha256.Sum256([]byte("counter.example.test"))
	valid := inference.CounterCapability{
		Provider:         "provider",
		Transport:        inference.CounterTransportSameEndpoint,
		SecurityIdentity: identity,
		Retention:        inference.RetentionNone,
		TokenizerRev:     "tokenizer-v1",
		Quality:          inference.CountQualityExactProvider,
	}
	tests := []struct {
		name       string
		capability inference.CounterCapability
		wantField  inference.CapabilityField
	}{
		{name: "same endpoint", capability: valid},
		{name: "local provider neutral", capability: inference.CounterCapability{Transport: inference.CounterTransportLocal, Retention: inference.RetentionNone, TokenizerRev: "estimate-v1", Quality: inference.CountQualityHeuristicEstimate}},
		{name: "local provider specific", capability: inference.CounterCapability{Provider: "provider", Transport: inference.CounterTransportLocal, Retention: inference.RetentionEphemeral, TokenizerRev: "local-v1", Quality: inference.CountQualityExactLocal}},
		{name: "separate endpoint", capability: inference.CounterCapability{Provider: "provider", Transport: inference.CounterTransportSeparateEndpoint, SecurityIdentity: identity, Retention: inference.RetentionLogged, TokenizerRev: "provider-v2", Quality: inference.CountQualityExactProvider}},
		{name: "unknown transport", capability: withCounterTransport(valid, inference.CounterTransportUnknown), wantField: inference.CapabilityFieldTransport},
		{name: "transport above range", capability: withCounterTransport(valid, inference.CounterTransport(255)), wantField: inference.CapabilityFieldTransport},
		{name: "unknown retention", capability: withCounterRetention(valid, inference.RetentionUnknown), wantField: inference.CapabilityFieldRetention},
		{name: "retention above range", capability: withCounterRetention(valid, inference.RetentionPosture(255)), wantField: inference.CapabilityFieldRetention},
		{name: "empty tokenizer revision", capability: withTokenizerRevision(valid, ""), wantField: inference.CapabilityFieldTokenizerRevision},
		{name: "unknown quality", capability: withCountQuality(valid, inference.CountQualityUnknown), wantField: inference.CapabilityFieldQuality},
		{name: "quality above range", capability: withCountQuality(valid, inference.CountQuality(255)), wantField: inference.CapabilityFieldQuality},
		{name: "local nonzero identity", capability: inference.CounterCapability{Transport: inference.CounterTransportLocal, SecurityIdentity: identity, Retention: inference.RetentionNone, TokenizerRev: "local-v1", Quality: inference.CountQualityExactLocal}, wantField: inference.CapabilityFieldSecurityIdentity},
		{name: "same endpoint empty provider", capability: withCounterProvider(valid, ""), wantField: inference.CapabilityFieldProvider},
		{name: "same endpoint zero identity", capability: withCounterIdentity(valid, inference.SecurityIdentity{}), wantField: inference.CapabilityFieldSecurityIdentity},
		{name: "separate endpoint empty provider", capability: inference.CounterCapability{Transport: inference.CounterTransportSeparateEndpoint, SecurityIdentity: identity, Retention: inference.RetentionNone, TokenizerRev: "provider-v1", Quality: inference.CountQualityExactProvider}, wantField: inference.CapabilityFieldProvider},
		{name: "separate endpoint zero identity", capability: inference.CounterCapability{Provider: "provider", Transport: inference.CounterTransportSeparateEndpoint, Retention: inference.RetentionNone, TokenizerRev: "provider-v1", Quality: inference.CountQualityExactProvider}, wantField: inference.CapabilityFieldSecurityIdentity},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			err := tt.capability.Validate()
			if tt.wantField == "" {
				if err != nil {
					t.Fatalf("Validate() error = %v, want nil", err)
				}
				return
			}
			var validationErr *inference.CapabilityValidationError
			if !errors.As(err, &validationErr) {
				t.Fatalf("Validate() error = %T, want *inference.CapabilityValidationError", err)
			}
			if validationErr.Capability != inference.CapabilityKindCounter || validationErr.Field != tt.wantField || validationErr.Reason == "" {
				t.Errorf("Validate() error = %+v, want counter field %q with named reason", validationErr, tt.wantField)
			}
		})
	}
}

func TestInferenceCapabilityValidate(t *testing.T) {
	t.Parallel()

	identity := sha256.Sum256([]byte("inference.example.test"))
	valid := inference.InferenceCapability{Provider: "provider", Transport: inference.InferenceTransportTLS, SecurityIdentity: identity, Retention: inference.RetentionEphemeral}
	tests := []struct {
		name       string
		capability inference.InferenceCapability
		wantField  inference.CapabilityField
	}{
		{name: "local", capability: inference.InferenceCapability{Transport: inference.InferenceTransportLocal, Retention: inference.RetentionNone}},
		{name: "tls", capability: valid},
		{name: "attested tls", capability: withInferenceTransport(valid, inference.InferenceTransportAttestedTLS)},
		{name: "end to end encrypted", capability: withInferenceTransport(valid, inference.InferenceTransportEndToEndEncrypted)},
		{name: "unknown retention is structurally valid", capability: withInferenceRetention(valid, inference.RetentionUnknown)},
		{name: "unknown transport", capability: withInferenceTransport(valid, inference.InferenceTransportUnknown), wantField: inference.CapabilityFieldTransport},
		{name: "transport above range", capability: withInferenceTransport(valid, inference.InferenceTransport(255)), wantField: inference.CapabilityFieldTransport},
		{name: "retention above range", capability: withInferenceRetention(valid, inference.RetentionPosture(255)), wantField: inference.CapabilityFieldRetention},
		{name: "local nonzero identity", capability: inference.InferenceCapability{Transport: inference.InferenceTransportLocal, SecurityIdentity: identity, Retention: inference.RetentionNone}, wantField: inference.CapabilityFieldSecurityIdentity},
		{name: "remote empty provider", capability: withInferenceProvider(valid, ""), wantField: inference.CapabilityFieldProvider},
		{name: "remote zero identity", capability: withInferenceIdentity(valid, inference.SecurityIdentity{}), wantField: inference.CapabilityFieldSecurityIdentity},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			err := tt.capability.Validate()
			if tt.wantField == "" {
				if err != nil {
					t.Fatalf("Validate() error = %v, want nil", err)
				}
				return
			}
			var validationErr *inference.CapabilityValidationError
			if !errors.As(err, &validationErr) {
				t.Fatalf("Validate() error = %T, want *inference.CapabilityValidationError", err)
			}
			if validationErr.Capability != inference.CapabilityKindInference || validationErr.Field != tt.wantField || validationErr.Reason == "" {
				t.Errorf("Validate() error = %+v, want inference field %q with named reason", validationErr, tt.wantField)
			}
		})
	}
}

func TestContextCounterFunc(t *testing.T) {
	t.Parallel()

	req := inference.Request{Model: inference.Model{Provider: "provider", Name: "model"}}
	validCount := inference.ContextCount{Model: inference.ModelKey{Provider: "provider", Model: "model"}, InputTokens: 0, Quality: inference.CountQualityExactLocal}
	validCapability := inference.CounterCapability{Quality: inference.CountQualityExactLocal}
	runtimeErr := &inference.ValidationError{Field: "request", Reason: "rejected"}
	tests := []struct {
		name        string
		counter     inference.ContextCounterFunc
		want        inference.ContextCount
		wantModel   inference.ModelKey
		wantQuality inference.CountQuality
		wantCause   error
		wantErr     error
	}{
		{name: "happy path and zero token boundary", counter: inference.ContextCounterFunc{Count: func(context.Context, inference.Request) (inference.ContextCount, error) { return validCount, nil }, Capability: validCapability}, want: validCount},
		{name: "nil count function", counter: inference.ContextCounterFunc{}, wantCause: inference.ErrContextCountFunctionMissing},
		{name: "invalid returned model", counter: inference.ContextCounterFunc{Count: func(context.Context, inference.Request) (inference.ContextCount, error) {
			count := validCount
			count.Model.Model = ""
			return count, nil
		}, Capability: validCapability}, wantModel: inference.ModelKey{Provider: "provider"}, wantQuality: inference.CountQualityExactLocal, wantCause: &inference.ModelKeyValidationError{}},
		{name: "unknown returned quality", counter: inference.ContextCounterFunc{Count: func(context.Context, inference.Request) (inference.ContextCount, error) {
			count := validCount
			count.Quality = inference.CountQualityUnknown
			return count, nil
		}, Capability: validCapability}, wantModel: validCount.Model, wantQuality: inference.CountQualityUnknown, wantCause: inference.ErrContextCountQualityInvalid},
		{name: "out of range returned quality", counter: inference.ContextCounterFunc{Count: func(context.Context, inference.Request) (inference.ContextCount, error) {
			count := validCount
			count.Quality = inference.CountQuality(255)
			return count, nil
		}, Capability: validCapability}, wantModel: validCount.Model, wantQuality: inference.CountQuality(255), wantCause: inference.ErrContextCountQualityInvalid},
		{name: "returned model differs from request", counter: inference.ContextCounterFunc{Count: func(context.Context, inference.Request) (inference.ContextCount, error) {
			count := validCount
			count.Model.Model = "other-model"
			return count, nil
		}, Capability: validCapability}, wantModel: inference.ModelKey{Provider: "provider", Model: "other-model"}, wantQuality: inference.CountQualityExactLocal, wantCause: inference.ErrContextCountModelMismatch},
		{name: "returned quality differs from declared capability", counter: inference.ContextCounterFunc{Count: func(context.Context, inference.Request) (inference.ContextCount, error) {
			count := validCount
			count.Quality = inference.CountQualityExactProvider
			return count, nil
		}, Capability: validCapability}, wantModel: validCount.Model, wantQuality: inference.CountQualityExactProvider, wantCause: inference.ErrContextCountCapabilityQualityMismatch},
		{name: "provider runtime error passes through", counter: inference.ContextCounterFunc{Count: func(context.Context, inference.Request) (inference.ContextCount, error) {
			return inference.ContextCount{}, runtimeErr
		}}, wantErr: runtimeErr},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := tt.counter.CountContext(context.Background(), req)
			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Fatalf("CountContext() error = %v, want original %v", err, tt.wantErr)
				}
				return
			}
			if tt.wantCause == nil {
				if err != nil {
					t.Fatalf("CountContext() error = %v, want nil", err)
				}
				if got != tt.want {
					t.Errorf("CountContext() = %+v, want %+v", got, tt.want)
				}
				return
			}
			var countErr *inference.ContextCountError
			if !errors.As(err, &countErr) {
				t.Fatalf("CountContext() error = %T, want *inference.ContextCountError", err)
			}
			if countErr.Model != tt.wantModel || countErr.Quality != tt.wantQuality {
				t.Errorf("CountContext() error = %+v, want model %+v quality %d", countErr, tt.wantModel, tt.wantQuality)
			}
			var wantModelErr *inference.ModelKeyValidationError
			if errors.As(tt.wantCause, &wantModelErr) {
				var modelErr *inference.ModelKeyValidationError
				if !errors.As(err, &modelErr) {
					t.Errorf("CountContext() error does not wrap *inference.ModelKeyValidationError")
				}
			} else if !errors.Is(err, tt.wantCause) {
				t.Errorf("CountContext() error = %v, want cause %v", err, tt.wantCause)
			}
		})
	}
}

func TestContextCounterFuncCapability(t *testing.T) {
	t.Parallel()

	want := inference.CounterCapability{Transport: inference.CounterTransportLocal, Retention: inference.RetentionNone, TokenizerRev: "estimate-v1", Quality: inference.CountQualityHeuristicEstimate}
	tests := []struct {
		name    string
		counter inference.ContextCounterFunc
		want    inference.CounterCapability
	}{
		{name: "returns declared metadata unchanged", counter: inference.ContextCounterFunc{Capability: want}, want: want},
		{name: "zero metadata boundary", counter: inference.ContextCounterFunc{}, want: inference.CounterCapability{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := tt.counter.CounterCapability(); got != tt.want {
				t.Errorf("CounterCapability() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestCompatibleCounter(t *testing.T) {
	t.Parallel()

	identity := sha256.Sum256([]byte("provider.example.test"))
	otherIdentity := sha256.Sum256([]byte("other.example.test"))
	baseInf := inference.InferenceCapability{Provider: "provider", Transport: inference.InferenceTransportTLS, SecurityIdentity: identity, Retention: inference.RetentionEphemeral}
	baseCounter := inference.CounterCapability{Provider: "provider", Transport: inference.CounterTransportSameEndpoint, SecurityIdentity: identity, Retention: inference.RetentionNone, TokenizerRev: "provider-v1", Quality: inference.CountQualityExactProvider}
	neutral := inference.CounterCapability{Transport: inference.CounterTransportLocal, Retention: inference.RetentionNone, TokenizerRev: "estimate-v1", Quality: inference.CountQualityHeuristicEstimate}
	tests := []struct {
		name       string
		inf        inference.InferenceCapability
		counter    inference.CounterCapability
		wantReason inference.CounterCompatibilityReason
		wantField  inference.CapabilityField
	}{
		{name: "provider neutral local with local inference", inf: inference.InferenceCapability{Transport: inference.InferenceTransportLocal, Retention: inference.RetentionUnknown}, counter: neutral},
		{name: "provider neutral local with tls", inf: withInferenceRetention(baseInf, inference.RetentionUnknown), counter: neutral},
		{name: "provider neutral local with attested tls", inf: withInferenceTransport(withInferenceRetention(baseInf, inference.RetentionUnknown), inference.InferenceTransportAttestedTLS), counter: neutral},
		{name: "provider neutral local with end to end encryption", inf: withInferenceTransport(withInferenceRetention(baseInf, inference.RetentionUnknown), inference.InferenceTransportEndToEndEncrypted), counter: neutral},
		{name: "provider specific local matching", inf: baseInf, counter: inference.CounterCapability{Provider: "provider", Transport: inference.CounterTransportLocal, Retention: inference.RetentionNone, TokenizerRev: "local-v1", Quality: inference.CountQualityExactLocal}},
		{name: "provider specific local mismatch", inf: baseInf, counter: inference.CounterCapability{Provider: "other", Transport: inference.CounterTransportLocal, Retention: inference.RetentionNone, TokenizerRev: "local-v1", Quality: inference.CountQualityExactLocal}, wantReason: inference.CounterCompatibilityProviderMismatch},
		{name: "provider specific local empty provider", inf: baseInf, counter: inference.CounterCapability{Transport: inference.CounterTransportLocal, Retention: inference.RetentionEphemeral, TokenizerRev: "local-v1", Quality: inference.CountQualityExactLocal}, wantReason: inference.CounterCompatibilityProviderMismatch},
		{name: "same endpoint exact match", inf: baseInf, counter: baseCounter},
		{name: "same endpoint provider mismatch", inf: baseInf, counter: withCounterProvider(baseCounter, "other"), wantReason: inference.CounterCompatibilityProviderMismatch},
		{name: "same endpoint identity mismatch", inf: baseInf, counter: withCounterIdentity(baseCounter, otherIdentity), wantReason: inference.CounterCompatibilityIdentityMismatch},
		{name: "same endpoint with local inference rejected", inf: inference.InferenceCapability{Transport: inference.InferenceTransportLocal, Retention: inference.RetentionNone}, counter: baseCounter, wantReason: inference.CounterCompatibilityTransportDowngrade},
		{name: "separate endpoint with tls", inf: baseInf, counter: withCounterTransport(baseCounter, inference.CounterTransportSeparateEndpoint)},
		{name: "separate endpoint provider mismatch", inf: baseInf, counter: withCounterProvider(withCounterTransport(baseCounter, inference.CounterTransportSeparateEndpoint), "other"), wantReason: inference.CounterCompatibilityProviderMismatch},
		{name: "separate endpoint with local rejected", inf: inference.InferenceCapability{Transport: inference.InferenceTransportLocal, Retention: inference.RetentionNone}, counter: withCounterTransport(baseCounter, inference.CounterTransportSeparateEndpoint), wantReason: inference.CounterCompatibilityTransportDowngrade},
		{name: "separate endpoint with attested rejected", inf: withInferenceTransport(baseInf, inference.InferenceTransportAttestedTLS), counter: withCounterTransport(baseCounter, inference.CounterTransportSeparateEndpoint), wantReason: inference.CounterCompatibilityTransportDowngrade},
		{name: "separate endpoint with end to end encryption rejected", inf: withInferenceTransport(baseInf, inference.InferenceTransportEndToEndEncrypted), counter: withCounterTransport(baseCounter, inference.CounterTransportSeparateEndpoint), wantReason: inference.CounterCompatibilityTransportDowngrade},
		{name: "none retention under none", inf: withInferenceRetention(baseInf, inference.RetentionNone), counter: withCounterRetention(baseCounter, inference.RetentionNone)},
		{name: "none retention under ephemeral", inf: withInferenceRetention(baseInf, inference.RetentionEphemeral), counter: withCounterRetention(baseCounter, inference.RetentionNone)},
		{name: "none retention under logged", inf: withInferenceRetention(baseInf, inference.RetentionLogged), counter: withCounterRetention(baseCounter, inference.RetentionNone)},
		{name: "ephemeral retention under none rejected", inf: withInferenceRetention(baseInf, inference.RetentionNone), counter: withCounterRetention(baseCounter, inference.RetentionEphemeral), wantReason: inference.CounterCompatibilityRetentionDowngrade},
		{name: "ephemeral retention under ephemeral", inf: withInferenceRetention(baseInf, inference.RetentionEphemeral), counter: withCounterRetention(baseCounter, inference.RetentionEphemeral)},
		{name: "ephemeral retention under logged", inf: withInferenceRetention(baseInf, inference.RetentionLogged), counter: withCounterRetention(baseCounter, inference.RetentionEphemeral)},
		{name: "logged retention under none rejected", inf: withInferenceRetention(baseInf, inference.RetentionNone), counter: withCounterRetention(baseCounter, inference.RetentionLogged), wantReason: inference.CounterCompatibilityRetentionDowngrade},
		{name: "logged retention under ephemeral rejected", inf: withInferenceRetention(baseInf, inference.RetentionEphemeral), counter: withCounterRetention(baseCounter, inference.RetentionLogged), wantReason: inference.CounterCompatibilityRetentionDowngrade},
		{name: "logged retention under logged", inf: withInferenceRetention(baseInf, inference.RetentionLogged), counter: withCounterRetention(baseCounter, inference.RetentionLogged)},
		{name: "unknown inference retention rejects non-neutral counter", inf: withInferenceRetention(baseInf, inference.RetentionUnknown), counter: baseCounter, wantReason: inference.CounterCompatibilityRetentionDowngrade},
		{name: "invalid inference capability", inf: withInferenceTransport(baseInf, inference.InferenceTransportUnknown), counter: neutral, wantReason: inference.CounterCompatibilityInvalidInference, wantField: inference.CapabilityFieldTransport},
		{name: "invalid counter capability", inf: baseInf, counter: withCounterRetention(baseCounter, inference.RetentionUnknown), wantReason: inference.CounterCompatibilityInvalidCounter, wantField: inference.CapabilityFieldRetention},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			err := inference.CompatibleCounter(tt.inf, tt.counter)
			if tt.wantReason == "" {
				if err != nil {
					t.Fatalf("CompatibleCounter() error = %v, want nil", err)
				}
				return
			}
			var compatibilityErr *inference.CounterCompatibilityError
			if !errors.As(err, &compatibilityErr) {
				t.Fatalf("CompatibleCounter() error = %T, want *inference.CounterCompatibilityError", err)
			}
			if compatibilityErr.Reason != tt.wantReason || compatibilityErr.Inference != tt.inf || compatibilityErr.Counter != tt.counter {
				t.Errorf("CompatibleCounter() error = %+v, want reason %q and original capabilities", compatibilityErr, tt.wantReason)
			}
			if tt.wantField != "" {
				var validationErr *inference.CapabilityValidationError
				if !errors.As(err, &validationErr) || validationErr.Field != tt.wantField {
					t.Errorf("CompatibleCounter() validation cause = %+v, want field %q", validationErr, tt.wantField)
				}
			}
		})
	}
}

func withCounterTransport(capability inference.CounterCapability, transport inference.CounterTransport) inference.CounterCapability {
	capability.Transport = transport
	return capability
}

func withCounterRetention(capability inference.CounterCapability, retention inference.RetentionPosture) inference.CounterCapability {
	capability.Retention = retention
	return capability
}

func withTokenizerRevision(capability inference.CounterCapability, revision inference.TokenizerRevision) inference.CounterCapability {
	capability.TokenizerRev = revision
	return capability
}

func withCountQuality(capability inference.CounterCapability, quality inference.CountQuality) inference.CounterCapability {
	capability.Quality = quality
	return capability
}

func withCounterProvider(capability inference.CounterCapability, provider inference.ProviderID) inference.CounterCapability {
	capability.Provider = provider
	return capability
}

func withCounterIdentity(capability inference.CounterCapability, identity inference.SecurityIdentity) inference.CounterCapability {
	capability.SecurityIdentity = identity
	return capability
}

func withInferenceTransport(capability inference.InferenceCapability, transport inference.InferenceTransport) inference.InferenceCapability {
	capability.Transport = transport
	return capability
}

func withInferenceRetention(capability inference.InferenceCapability, retention inference.RetentionPosture) inference.InferenceCapability {
	capability.Retention = retention
	return capability
}

func withInferenceProvider(capability inference.InferenceCapability, provider inference.ProviderID) inference.InferenceCapability {
	capability.Provider = provider
	return capability
}

func withInferenceIdentity(capability inference.InferenceCapability, identity inference.SecurityIdentity) inference.InferenceCapability {
	capability.SecurityIdentity = identity
	return capability
}
