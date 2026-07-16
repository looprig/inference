package contextcount_test

import (
	"context"
	"crypto/sha256"
	"errors"
	"testing"

	"github.com/looprig/inference"
	contextcount "github.com/looprig/inference/contextcount"
	model "github.com/looprig/inference/model"
)

func TestCounterCapabilityValidate(t *testing.T) {
	t.Parallel()

	identity := sha256.Sum256([]byte("counter.example.test"))
	valid := contextcount.CounterCapability{
		Provider:         "provider",
		Transport:        contextcount.CounterTransportSameEndpoint,
		SecurityIdentity: identity,
		Retention:        contextcount.RetentionNone,
		TokenizerRev:     "tokenizer-v1",
		Quality:          contextcount.CountQualityExactProvider,
	}
	tests := []struct {
		name       string
		capability contextcount.CounterCapability
		wantField  contextcount.CapabilityField
	}{
		{name: "same endpoint", capability: valid},
		{name: "local provider neutral", capability: contextcount.CounterCapability{Transport: contextcount.CounterTransportLocal, Retention: contextcount.RetentionNone, TokenizerRev: "estimate-v1", Quality: contextcount.CountQualityHeuristicEstimate}},
		{name: "local provider specific", capability: contextcount.CounterCapability{Provider: "provider", Transport: contextcount.CounterTransportLocal, Retention: contextcount.RetentionEphemeral, TokenizerRev: "local-v1", Quality: contextcount.CountQualityExactLocal}},
		{name: "separate endpoint", capability: contextcount.CounterCapability{Provider: "provider", Transport: contextcount.CounterTransportSeparateEndpoint, SecurityIdentity: identity, Retention: contextcount.RetentionLogged, TokenizerRev: "provider-v2", Quality: contextcount.CountQualityExactProvider}},
		{name: "unknown transport", capability: withCounterTransport(valid, contextcount.CounterTransportUnknown), wantField: contextcount.CapabilityFieldTransport},
		{name: "transport above range", capability: withCounterTransport(valid, contextcount.CounterTransport(255)), wantField: contextcount.CapabilityFieldTransport},
		{name: "unknown retention", capability: withCounterRetention(valid, contextcount.RetentionUnknown), wantField: contextcount.CapabilityFieldRetention},
		{name: "retention above range", capability: withCounterRetention(valid, contextcount.RetentionPosture(255)), wantField: contextcount.CapabilityFieldRetention},
		{name: "empty tokenizer revision", capability: withTokenizerRevision(valid, ""), wantField: contextcount.CapabilityFieldTokenizerRevision},
		{name: "unknown quality", capability: withCountQuality(valid, contextcount.CountQualityUnknown), wantField: contextcount.CapabilityFieldQuality},
		{name: "quality above range", capability: withCountQuality(valid, contextcount.CountQuality(255)), wantField: contextcount.CapabilityFieldQuality},
		{name: "local nonzero identity", capability: contextcount.CounterCapability{Transport: contextcount.CounterTransportLocal, SecurityIdentity: identity, Retention: contextcount.RetentionNone, TokenizerRev: "local-v1", Quality: contextcount.CountQualityExactLocal}, wantField: contextcount.CapabilityFieldSecurityIdentity},
		{name: "same endpoint empty provider", capability: withCounterProvider(valid, ""), wantField: contextcount.CapabilityFieldProvider},
		{name: "same endpoint zero identity", capability: withCounterIdentity(valid, contextcount.SecurityIdentity{}), wantField: contextcount.CapabilityFieldSecurityIdentity},
		{name: "separate endpoint empty provider", capability: contextcount.CounterCapability{Transport: contextcount.CounterTransportSeparateEndpoint, SecurityIdentity: identity, Retention: contextcount.RetentionNone, TokenizerRev: "provider-v1", Quality: contextcount.CountQualityExactProvider}, wantField: contextcount.CapabilityFieldProvider},
		{name: "separate endpoint zero identity", capability: contextcount.CounterCapability{Provider: "provider", Transport: contextcount.CounterTransportSeparateEndpoint, Retention: contextcount.RetentionNone, TokenizerRev: "provider-v1", Quality: contextcount.CountQualityExactProvider}, wantField: contextcount.CapabilityFieldSecurityIdentity},
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
			var validationErr *contextcount.CapabilityValidationError
			if !errors.As(err, &validationErr) {
				t.Fatalf("Validate() error = %T, want *contextcount.CapabilityValidationError", err)
			}
			if validationErr.Capability != contextcount.CapabilityKindCounter || validationErr.Field != tt.wantField || validationErr.Reason == "" {
				t.Errorf("Validate() error = %+v, want counter field %q with named reason", validationErr, tt.wantField)
			}
		})
	}
}

func TestInferenceCapabilityValidate(t *testing.T) {
	t.Parallel()

	identity := sha256.Sum256([]byte("inference.example.test"))
	valid := contextcount.InferenceCapability{Provider: "provider", Transport: contextcount.InferenceTransportTLS, SecurityIdentity: identity, Retention: contextcount.RetentionEphemeral}
	tests := []struct {
		name       string
		capability contextcount.InferenceCapability
		wantField  contextcount.CapabilityField
	}{
		{name: "local", capability: contextcount.InferenceCapability{Transport: contextcount.InferenceTransportLocal, Retention: contextcount.RetentionNone}},
		{name: "tls", capability: valid},
		{name: "attested tls", capability: withInferenceTransport(valid, contextcount.InferenceTransportAttestedTLS)},
		{name: "end to end encrypted", capability: withInferenceTransport(valid, contextcount.InferenceTransportEndToEndEncrypted)},
		{name: "unknown retention is structurally valid", capability: withInferenceRetention(valid, contextcount.RetentionUnknown)},
		{name: "unknown transport", capability: withInferenceTransport(valid, contextcount.InferenceTransportUnknown), wantField: contextcount.CapabilityFieldTransport},
		{name: "transport above range", capability: withInferenceTransport(valid, contextcount.InferenceTransport(255)), wantField: contextcount.CapabilityFieldTransport},
		{name: "retention above range", capability: withInferenceRetention(valid, contextcount.RetentionPosture(255)), wantField: contextcount.CapabilityFieldRetention},
		{name: "local nonzero identity", capability: contextcount.InferenceCapability{Transport: contextcount.InferenceTransportLocal, SecurityIdentity: identity, Retention: contextcount.RetentionNone}, wantField: contextcount.CapabilityFieldSecurityIdentity},
		{name: "remote empty provider", capability: withInferenceProvider(valid, ""), wantField: contextcount.CapabilityFieldProvider},
		{name: "remote zero identity", capability: withInferenceIdentity(valid, contextcount.SecurityIdentity{}), wantField: contextcount.CapabilityFieldSecurityIdentity},
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
			var validationErr *contextcount.CapabilityValidationError
			if !errors.As(err, &validationErr) {
				t.Fatalf("Validate() error = %T, want *contextcount.CapabilityValidationError", err)
			}
			if validationErr.Capability != contextcount.CapabilityKindInference || validationErr.Field != tt.wantField || validationErr.Reason == "" {
				t.Errorf("Validate() error = %+v, want inference field %q with named reason", validationErr, tt.wantField)
			}
		})
	}
}

func TestContextCounterFunc(t *testing.T) {
	t.Parallel()

	req := inference.Request{Model: model.Model{Provider: "provider", Name: "model"}}
	validCount := contextcount.ContextCount{Model: model.ModelKey{Provider: "provider", Model: "model"}, InputTokens: 0, Quality: contextcount.CountQualityExactLocal}
	validCapability := contextcount.CounterCapability{Quality: contextcount.CountQualityExactLocal}
	runtimeErr := &model.ValidationError{Field: "request", Reason: "rejected"}
	tests := []struct {
		name        string
		counter     contextcount.ContextCounterFunc
		want        contextcount.ContextCount
		wantModel   model.ModelKey
		wantQuality contextcount.CountQuality
		wantCause   error
		wantErr     error
	}{
		{name: "happy path and zero token boundary", counter: contextcount.ContextCounterFunc{Count: func(context.Context, inference.Request) (contextcount.ContextCount, error) { return validCount, nil }, Capability: validCapability}, want: validCount},
		{name: "nil count function", counter: contextcount.ContextCounterFunc{}, wantCause: contextcount.ErrContextCountFunctionMissing},
		{name: "invalid returned model", counter: contextcount.ContextCounterFunc{Count: func(context.Context, inference.Request) (contextcount.ContextCount, error) {
			count := validCount
			count.Model.Model = ""
			return count, nil
		}, Capability: validCapability}, wantModel: model.ModelKey{Provider: "provider"}, wantQuality: contextcount.CountQualityExactLocal, wantCause: &model.ModelKeyValidationError{}},
		{name: "unknown returned quality", counter: contextcount.ContextCounterFunc{Count: func(context.Context, inference.Request) (contextcount.ContextCount, error) {
			count := validCount
			count.Quality = contextcount.CountQualityUnknown
			return count, nil
		}, Capability: validCapability}, wantModel: validCount.Model, wantQuality: contextcount.CountQualityUnknown, wantCause: contextcount.ErrContextCountQualityInvalid},
		{name: "out of range returned quality", counter: contextcount.ContextCounterFunc{Count: func(context.Context, inference.Request) (contextcount.ContextCount, error) {
			count := validCount
			count.Quality = contextcount.CountQuality(255)
			return count, nil
		}, Capability: validCapability}, wantModel: validCount.Model, wantQuality: contextcount.CountQuality(255), wantCause: contextcount.ErrContextCountQualityInvalid},
		{name: "returned model differs from request", counter: contextcount.ContextCounterFunc{Count: func(context.Context, inference.Request) (contextcount.ContextCount, error) {
			count := validCount
			count.Model.Model = "other-model"
			return count, nil
		}, Capability: validCapability}, wantModel: model.ModelKey{Provider: "provider", Model: "other-model"}, wantQuality: contextcount.CountQualityExactLocal, wantCause: contextcount.ErrContextCountModelMismatch},
		{name: "returned quality differs from declared capability", counter: contextcount.ContextCounterFunc{Count: func(context.Context, inference.Request) (contextcount.ContextCount, error) {
			count := validCount
			count.Quality = contextcount.CountQualityExactProvider
			return count, nil
		}, Capability: validCapability}, wantModel: validCount.Model, wantQuality: contextcount.CountQualityExactProvider, wantCause: contextcount.ErrContextCountCapabilityQualityMismatch},
		{name: "provider runtime error passes through", counter: contextcount.ContextCounterFunc{Count: func(context.Context, inference.Request) (contextcount.ContextCount, error) {
			return contextcount.ContextCount{}, runtimeErr
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
			var countErr *contextcount.ContextCountError
			if !errors.As(err, &countErr) {
				t.Fatalf("CountContext() error = %T, want *contextcount.ContextCountError", err)
			}
			if countErr.Model != tt.wantModel || countErr.Quality != tt.wantQuality {
				t.Errorf("CountContext() error = %+v, want model %+v quality %d", countErr, tt.wantModel, tt.wantQuality)
			}
			var wantModelErr *model.ModelKeyValidationError
			if errors.As(tt.wantCause, &wantModelErr) {
				var modelErr *model.ModelKeyValidationError
				if !errors.As(err, &modelErr) {
					t.Errorf("CountContext() error does not wrap *model.ModelKeyValidationError")
				}
			} else if !errors.Is(err, tt.wantCause) {
				t.Errorf("CountContext() error = %v, want cause %v", err, tt.wantCause)
			}
		})
	}
}

func TestContextCounterFuncCapability(t *testing.T) {
	t.Parallel()

	want := contextcount.CounterCapability{Transport: contextcount.CounterTransportLocal, Retention: contextcount.RetentionNone, TokenizerRev: "estimate-v1", Quality: contextcount.CountQualityHeuristicEstimate}
	tests := []struct {
		name    string
		counter contextcount.ContextCounterFunc
		want    contextcount.CounterCapability
	}{
		{name: "returns declared metadata unchanged", counter: contextcount.ContextCounterFunc{Capability: want}, want: want},
		{name: "zero metadata boundary", counter: contextcount.ContextCounterFunc{}, want: contextcount.CounterCapability{}},
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
	baseInf := contextcount.InferenceCapability{Provider: "provider", Transport: contextcount.InferenceTransportTLS, SecurityIdentity: identity, Retention: contextcount.RetentionEphemeral}
	baseCounter := contextcount.CounterCapability{Provider: "provider", Transport: contextcount.CounterTransportSameEndpoint, SecurityIdentity: identity, Retention: contextcount.RetentionNone, TokenizerRev: "provider-v1", Quality: contextcount.CountQualityExactProvider}
	neutral := contextcount.CounterCapability{Transport: contextcount.CounterTransportLocal, Retention: contextcount.RetentionNone, TokenizerRev: "estimate-v1", Quality: contextcount.CountQualityHeuristicEstimate}
	tests := []struct {
		name       string
		inf        contextcount.InferenceCapability
		counter    contextcount.CounterCapability
		wantReason contextcount.CounterCompatibilityReason
		wantField  contextcount.CapabilityField
	}{
		{name: "provider neutral local with local inference", inf: contextcount.InferenceCapability{Transport: contextcount.InferenceTransportLocal, Retention: contextcount.RetentionUnknown}, counter: neutral},
		{name: "provider neutral local with tls", inf: withInferenceRetention(baseInf, contextcount.RetentionUnknown), counter: neutral},
		{name: "provider neutral local with attested tls", inf: withInferenceTransport(withInferenceRetention(baseInf, contextcount.RetentionUnknown), contextcount.InferenceTransportAttestedTLS), counter: neutral},
		{name: "provider neutral local with end to end encryption", inf: withInferenceTransport(withInferenceRetention(baseInf, contextcount.RetentionUnknown), contextcount.InferenceTransportEndToEndEncrypted), counter: neutral},
		{name: "provider specific local matching", inf: baseInf, counter: contextcount.CounterCapability{Provider: "provider", Transport: contextcount.CounterTransportLocal, Retention: contextcount.RetentionNone, TokenizerRev: "local-v1", Quality: contextcount.CountQualityExactLocal}},
		{name: "provider specific local mismatch", inf: baseInf, counter: contextcount.CounterCapability{Provider: "other", Transport: contextcount.CounterTransportLocal, Retention: contextcount.RetentionNone, TokenizerRev: "local-v1", Quality: contextcount.CountQualityExactLocal}, wantReason: contextcount.CounterCompatibilityProviderMismatch},
		{name: "provider specific local empty provider", inf: baseInf, counter: contextcount.CounterCapability{Transport: contextcount.CounterTransportLocal, Retention: contextcount.RetentionEphemeral, TokenizerRev: "local-v1", Quality: contextcount.CountQualityExactLocal}, wantReason: contextcount.CounterCompatibilityProviderMismatch},
		{name: "same endpoint exact match", inf: baseInf, counter: baseCounter},
		{name: "same endpoint provider mismatch", inf: baseInf, counter: withCounterProvider(baseCounter, "other"), wantReason: contextcount.CounterCompatibilityProviderMismatch},
		{name: "same endpoint identity mismatch", inf: baseInf, counter: withCounterIdentity(baseCounter, otherIdentity), wantReason: contextcount.CounterCompatibilityIdentityMismatch},
		{name: "same endpoint with local inference rejected", inf: contextcount.InferenceCapability{Transport: contextcount.InferenceTransportLocal, Retention: contextcount.RetentionNone}, counter: baseCounter, wantReason: contextcount.CounterCompatibilityTransportDowngrade},
		{name: "separate endpoint with tls", inf: baseInf, counter: withCounterTransport(baseCounter, contextcount.CounterTransportSeparateEndpoint)},
		{name: "separate endpoint provider mismatch", inf: baseInf, counter: withCounterProvider(withCounterTransport(baseCounter, contextcount.CounterTransportSeparateEndpoint), "other"), wantReason: contextcount.CounterCompatibilityProviderMismatch},
		{name: "separate endpoint with local rejected", inf: contextcount.InferenceCapability{Transport: contextcount.InferenceTransportLocal, Retention: contextcount.RetentionNone}, counter: withCounterTransport(baseCounter, contextcount.CounterTransportSeparateEndpoint), wantReason: contextcount.CounterCompatibilityTransportDowngrade},
		{name: "separate endpoint with attested rejected", inf: withInferenceTransport(baseInf, contextcount.InferenceTransportAttestedTLS), counter: withCounterTransport(baseCounter, contextcount.CounterTransportSeparateEndpoint), wantReason: contextcount.CounterCompatibilityTransportDowngrade},
		{name: "separate endpoint with end to end encryption rejected", inf: withInferenceTransport(baseInf, contextcount.InferenceTransportEndToEndEncrypted), counter: withCounterTransport(baseCounter, contextcount.CounterTransportSeparateEndpoint), wantReason: contextcount.CounterCompatibilityTransportDowngrade},
		{name: "none retention under none", inf: withInferenceRetention(baseInf, contextcount.RetentionNone), counter: withCounterRetention(baseCounter, contextcount.RetentionNone)},
		{name: "none retention under ephemeral", inf: withInferenceRetention(baseInf, contextcount.RetentionEphemeral), counter: withCounterRetention(baseCounter, contextcount.RetentionNone)},
		{name: "none retention under logged", inf: withInferenceRetention(baseInf, contextcount.RetentionLogged), counter: withCounterRetention(baseCounter, contextcount.RetentionNone)},
		{name: "ephemeral retention under none rejected", inf: withInferenceRetention(baseInf, contextcount.RetentionNone), counter: withCounterRetention(baseCounter, contextcount.RetentionEphemeral), wantReason: contextcount.CounterCompatibilityRetentionDowngrade},
		{name: "ephemeral retention under ephemeral", inf: withInferenceRetention(baseInf, contextcount.RetentionEphemeral), counter: withCounterRetention(baseCounter, contextcount.RetentionEphemeral)},
		{name: "ephemeral retention under logged", inf: withInferenceRetention(baseInf, contextcount.RetentionLogged), counter: withCounterRetention(baseCounter, contextcount.RetentionEphemeral)},
		{name: "logged retention under none rejected", inf: withInferenceRetention(baseInf, contextcount.RetentionNone), counter: withCounterRetention(baseCounter, contextcount.RetentionLogged), wantReason: contextcount.CounterCompatibilityRetentionDowngrade},
		{name: "logged retention under ephemeral rejected", inf: withInferenceRetention(baseInf, contextcount.RetentionEphemeral), counter: withCounterRetention(baseCounter, contextcount.RetentionLogged), wantReason: contextcount.CounterCompatibilityRetentionDowngrade},
		{name: "logged retention under logged", inf: withInferenceRetention(baseInf, contextcount.RetentionLogged), counter: withCounterRetention(baseCounter, contextcount.RetentionLogged)},
		{name: "unknown inference retention rejects non-neutral counter", inf: withInferenceRetention(baseInf, contextcount.RetentionUnknown), counter: baseCounter, wantReason: contextcount.CounterCompatibilityRetentionDowngrade},
		{name: "invalid inference capability", inf: withInferenceTransport(baseInf, contextcount.InferenceTransportUnknown), counter: neutral, wantReason: contextcount.CounterCompatibilityInvalidInference, wantField: contextcount.CapabilityFieldTransport},
		{name: "invalid counter capability", inf: baseInf, counter: withCounterRetention(baseCounter, contextcount.RetentionUnknown), wantReason: contextcount.CounterCompatibilityInvalidCounter, wantField: contextcount.CapabilityFieldRetention},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			err := contextcount.CompatibleCounter(tt.inf, tt.counter)
			if tt.wantReason == "" {
				if err != nil {
					t.Fatalf("CompatibleCounter() error = %v, want nil", err)
				}
				return
			}
			var compatibilityErr *contextcount.CounterCompatibilityError
			if !errors.As(err, &compatibilityErr) {
				t.Fatalf("CompatibleCounter() error = %T, want *contextcount.CounterCompatibilityError", err)
			}
			if compatibilityErr.Reason != tt.wantReason || compatibilityErr.Inference != tt.inf || compatibilityErr.Counter != tt.counter {
				t.Errorf("CompatibleCounter() error = %+v, want reason %q and original capabilities", compatibilityErr, tt.wantReason)
			}
			if tt.wantField != "" {
				var validationErr *contextcount.CapabilityValidationError
				if !errors.As(err, &validationErr) || validationErr.Field != tt.wantField {
					t.Errorf("CompatibleCounter() validation cause = %+v, want field %q", validationErr, tt.wantField)
				}
			}
		})
	}
}

func withCounterTransport(capability contextcount.CounterCapability, transport contextcount.CounterTransport) contextcount.CounterCapability {
	capability.Transport = transport
	return capability
}

func withCounterRetention(capability contextcount.CounterCapability, retention contextcount.RetentionPosture) contextcount.CounterCapability {
	capability.Retention = retention
	return capability
}

func withTokenizerRevision(capability contextcount.CounterCapability, revision contextcount.TokenizerRevision) contextcount.CounterCapability {
	capability.TokenizerRev = revision
	return capability
}

func withCountQuality(capability contextcount.CounterCapability, quality contextcount.CountQuality) contextcount.CounterCapability {
	capability.Quality = quality
	return capability
}

func withCounterProvider(capability contextcount.CounterCapability, provider contextcount.ProviderID) contextcount.CounterCapability {
	capability.Provider = provider
	return capability
}

func withCounterIdentity(capability contextcount.CounterCapability, identity contextcount.SecurityIdentity) contextcount.CounterCapability {
	capability.SecurityIdentity = identity
	return capability
}

func withInferenceTransport(capability contextcount.InferenceCapability, transport contextcount.InferenceTransport) contextcount.InferenceCapability {
	capability.Transport = transport
	return capability
}

func withInferenceRetention(capability contextcount.InferenceCapability, retention contextcount.RetentionPosture) contextcount.InferenceCapability {
	capability.Retention = retention
	return capability
}

func withInferenceProvider(capability contextcount.InferenceCapability, provider contextcount.ProviderID) contextcount.InferenceCapability {
	capability.Provider = provider
	return capability
}

func withInferenceIdentity(capability contextcount.InferenceCapability, identity contextcount.SecurityIdentity) contextcount.InferenceCapability {
	capability.SecurityIdentity = identity
	return capability
}
