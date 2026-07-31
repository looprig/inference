package gateway_test

import (
	"context"
	"errors"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/gateway"
	"github.com/looprig/inference/model"
	"github.com/looprig/inference/stream"
)

// fakeClient is a minimal inference.Client double used only to exercise
// gateway routing/resolution. It never gets invoked by these tests -- the
// gateway package under test never calls Invoke or Stream, it only routes to
// a Target that carries one.
type fakeClient struct {
	name string
}

func (f *fakeClient) Invoke(ctx context.Context, req inference.Request) (*inference.Response, error) {
	return nil, errors.New("fakeClient.Invoke: not implemented")
}

func (f *fakeClient) Stream(ctx context.Context, req inference.Request) (*stream.StreamReader[content.Chunk], error) {
	return nil, errors.New("fakeClient.Stream: not implemented")
}

func validModel(name string) model.Model {
	return model.Model{
		Provider:  model.ProviderName("test-provider"),
		APIFormat: model.APIFormatAnthropic,
		Name:      name,
	}
}

func f64ptr(v float64) *float64 { return &v }

// TestMux_Resolve_ExactRoute verifies the highest-precedence match: an exact
// (ingress format, requested model alias) pair present in Routes.
func TestMux_Resolve_ExactRoute(t *testing.T) {
	t.Parallel()
	clientA := &fakeClient{name: "a"}
	mux, err := gateway.NewMux(gateway.Mux{
		Routes: map[gateway.RouteKey]gateway.Target{
			{Ingress: model.APIFormatAnthropic, Model: "primary"}: {
				ID:     "kimi-on-phala",
				Client: clientA,
				Model:  validModel("kimi-k2"),
			},
		},
	})
	if err != nil {
		t.Fatalf("NewMux: unexpected error: %v", err)
	}

	target, err := mux.Resolve(context.Background(), model.APIFormatAnthropic, "primary")
	if err != nil {
		t.Fatalf("Resolve: unexpected error: %v", err)
	}
	if target.ID != "kimi-on-phala" {
		t.Errorf("target.ID = %q, want %q", target.ID, "kimi-on-phala")
	}
	if target.Client != clientA {
		t.Errorf("target.Client = %v, want %v", target.Client, clientA)
	}
}

// TestMux_Resolve_FormatDefault verifies that when no exact route matches,
// resolution falls back to the ingress format's default target.
func TestMux_Resolve_FormatDefault(t *testing.T) {
	t.Parallel()
	clientA := &fakeClient{name: "a"}
	mux, err := gateway.NewMux(gateway.Mux{
		FormatDefaults: map[model.APIFormat]gateway.Target{
			model.APIFormatAnthropic: {
				ID:     "format-default",
				Client: clientA,
				Model:  validModel("kimi-k2"),
			},
		},
	})
	if err != nil {
		t.Fatalf("NewMux: unexpected error: %v", err)
	}

	target, err := mux.Resolve(context.Background(), model.APIFormatAnthropic, "unregistered-alias")
	if err != nil {
		t.Fatalf("Resolve: unexpected error: %v", err)
	}
	if target.ID != "format-default" {
		t.Errorf("target.ID = %q, want %q", target.ID, "format-default")
	}
}

// TestMux_Resolve_ExactRouteBeatsFormatDefault verifies precedence: an exact
// route wins over the format default even when both are configured for the
// same ingress format and would otherwise both match.
func TestMux_Resolve_ExactRouteBeatsFormatDefault(t *testing.T) {
	t.Parallel()
	clientExact := &fakeClient{name: "exact"}
	clientDefault := &fakeClient{name: "default"}
	mux, err := gateway.NewMux(gateway.Mux{
		Routes: map[gateway.RouteKey]gateway.Target{
			{Ingress: model.APIFormatAnthropic, Model: "primary"}: {
				ID: "exact", Client: clientExact, Model: validModel("m-exact"),
			},
		},
		FormatDefaults: map[model.APIFormat]gateway.Target{
			model.APIFormatAnthropic: {
				ID: "default", Client: clientDefault, Model: validModel("m-default"),
			},
		},
	})
	if err != nil {
		t.Fatalf("NewMux: unexpected error: %v", err)
	}

	target, err := mux.Resolve(context.Background(), model.APIFormatAnthropic, "primary")
	if err != nil {
		t.Fatalf("Resolve: unexpected error: %v", err)
	}
	if target.ID != "exact" {
		t.Errorf("target.ID = %q, want %q (exact route must beat format default)", target.ID, "exact")
	}
}

// TestMux_Resolve_GlobalDefault verifies the third precedence tier: when
// neither an exact route nor a format default matches, the global Default
// target is used.
func TestMux_Resolve_GlobalDefault(t *testing.T) {
	t.Parallel()
	clientA := &fakeClient{name: "a"}
	def := gateway.Target{ID: "global-default", Client: clientA, Model: validModel("kimi-k2")}
	mux, err := gateway.NewMux(gateway.Mux{Default: &def})
	if err != nil {
		t.Fatalf("NewMux: unexpected error: %v", err)
	}

	target, err := mux.Resolve(context.Background(), model.APIFormatGemini, "anything")
	if err != nil {
		t.Fatalf("Resolve: unexpected error: %v", err)
	}
	if target.ID != "global-default" {
		t.Errorf("target.ID = %q, want %q", target.ID, "global-default")
	}
}

// TestMux_Resolve_FormatDefaultBeatsGlobalDefault verifies precedence: a
// format default wins over the global default.
func TestMux_Resolve_FormatDefaultBeatsGlobalDefault(t *testing.T) {
	t.Parallel()
	clientFmt := &fakeClient{name: "fmt"}
	clientGlobal := &fakeClient{name: "global"}
	def := gateway.Target{ID: "global", Client: clientGlobal, Model: validModel("m-global")}
	mux, err := gateway.NewMux(gateway.Mux{
		FormatDefaults: map[model.APIFormat]gateway.Target{
			model.APIFormatAnthropic: {ID: "fmt", Client: clientFmt, Model: validModel("m-fmt")},
		},
		Default: &def,
	})
	if err != nil {
		t.Fatalf("NewMux: unexpected error: %v", err)
	}

	target, err := mux.Resolve(context.Background(), model.APIFormatAnthropic, "whatever")
	if err != nil {
		t.Fatalf("Resolve: unexpected error: %v", err)
	}
	if target.ID != "fmt" {
		t.Errorf("target.ID = %q, want %q (format default must beat global default)", target.ID, "fmt")
	}
}

// TestMux_Resolve_MissingRoute verifies the fourth precedence tier: no exact
// route, no format default, no global default -> typed route-not-found error.
func TestMux_Resolve_MissingRoute(t *testing.T) {
	t.Parallel()
	mux, err := gateway.NewMux(gateway.Mux{})
	if err != nil {
		t.Fatalf("NewMux: unexpected error: %v", err)
	}

	_, err = mux.Resolve(context.Background(), model.APIFormatOpenAI, "nope")
	if err == nil {
		t.Fatal("Resolve: expected error, got nil")
	}
	var notFound *gateway.RouteNotFoundError
	if !errors.As(err, &notFound) {
		t.Fatalf("Resolve error = %v (%T), want *gateway.RouteNotFoundError", err, err)
	}
	if notFound.Ingress != model.APIFormatOpenAI || notFound.Model != "nope" {
		t.Errorf("RouteNotFoundError = %+v, want Ingress=%q Model=%q", notFound, model.APIFormatOpenAI, "nope")
	}
}

// TestMux_RouteKeyEmptyModel_IsOrdinaryNotReserved locks in the decision that
// RouteKey.Model == "" is NOT a reserved/special alias -- it is an ordinary
// exact key like any other, matching only a request that literally supplies
// an empty requested-model string. It must not be treated as equivalent to,
// or in conflict with, a FormatDefaults entry for the same ingress format.
func TestMux_RouteKeyEmptyModel_IsOrdinaryNotReserved(t *testing.T) {
	t.Parallel()
	clientExact := &fakeClient{name: "exact-empty"}
	clientFmt := &fakeClient{name: "fmt"}
	mux, err := gateway.NewMux(gateway.Mux{
		Routes: map[gateway.RouteKey]gateway.Target{
			{Ingress: model.APIFormatAnthropic, Model: ""}: {
				ID: "exact-empty", Client: clientExact, Model: validModel("m-exact-empty"),
			},
		},
		FormatDefaults: map[model.APIFormat]gateway.Target{
			model.APIFormatAnthropic: {ID: "fmt", Client: clientFmt, Model: validModel("m-fmt")},
		},
	})
	if err != nil {
		t.Fatalf("NewMux: unexpected error configuring an empty RouteKey.Model alongside a FormatDefaults entry: %v", err)
	}

	// A request with an empty model alias hits the exact "" route, not the format default.
	target, err := mux.Resolve(context.Background(), model.APIFormatAnthropic, "")
	if err != nil {
		t.Fatalf("Resolve: unexpected error: %v", err)
	}
	if target.ID != "exact-empty" {
		t.Errorf("target.ID = %q, want %q", target.ID, "exact-empty")
	}

	// A request with any other model alias still falls through to the format default.
	target, err = mux.Resolve(context.Background(), model.APIFormatAnthropic, "something-else")
	if err != nil {
		t.Fatalf("Resolve: unexpected error: %v", err)
	}
	if target.ID != "fmt" {
		t.Errorf("target.ID = %q, want %q", target.ID, "fmt")
	}
}

// TestNewMux_NilClient_Routes/FormatDefaults/Default verify that a nil
// Client anywhere in the configuration is rejected with a typed ConfigError.
func TestNewMux_NilClient_Routes(t *testing.T) {
	t.Parallel()
	_, err := gateway.NewMux(gateway.Mux{
		Routes: map[gateway.RouteKey]gateway.Target{
			{Ingress: model.APIFormatAnthropic, Model: "primary"}: {ID: "x", Client: nil, Model: validModel("m")},
		},
	})
	assertConfigError(t, err)
}

func TestNewMux_NilClient_FormatDefaults(t *testing.T) {
	t.Parallel()
	_, err := gateway.NewMux(gateway.Mux{
		FormatDefaults: map[model.APIFormat]gateway.Target{
			model.APIFormatAnthropic: {ID: "x", Client: nil, Model: validModel("m")},
		},
	})
	assertConfigError(t, err)
}

func TestNewMux_NilClient_Default(t *testing.T) {
	t.Parallel()
	def := gateway.Target{ID: "x", Client: nil, Model: validModel("m")}
	_, err := gateway.NewMux(gateway.Mux{Default: &def})
	assertConfigError(t, err)
}

// TestNewMux_InvalidModel verifies that a Target whose Model fails
// model.Model.Validate (e.g. an empty Name) is rejected with a typed
// ConfigError wrapping the underlying validation error.
func TestNewMux_InvalidModel(t *testing.T) {
	t.Parallel()
	clientA := &fakeClient{name: "a"}
	_, err := gateway.NewMux(gateway.Mux{
		Routes: map[gateway.RouteKey]gateway.Target{
			{Ingress: model.APIFormatAnthropic, Model: "primary"}: {
				ID: "x", Client: clientA, Model: model.Model{}, // empty Name -> invalid
			},
		},
	})
	assertConfigError(t, err)
	var modelErr *model.ValidationError
	if !errors.As(err, &modelErr) {
		t.Errorf("error chain does not contain *model.ValidationError: %v", err)
	}
}

// TestNewMux_DuplicateLogicalRoute locks in the decision for what counts as
// a "duplicate logical route" given this API shape: since Routes and
// FormatDefaults are Go maps, a literal duplicate key is structurally
// impossible. The real hazard is a non-empty Target.ID -- a stable
// diagnostic identity -- silently reused across two Targets that bind to
// different Model values, which would make the ID lie about what it
// identifies. That is rejected. Reusing the same ID for the literal same
// Model (e.g. one target reached through two different route keys) is NOT a
// conflict and must be allowed.
func TestNewMux_DuplicateLogicalRoute_ConflictingModelsRejected(t *testing.T) {
	t.Parallel()
	clientA := &fakeClient{name: "a"}
	clientB := &fakeClient{name: "b"}
	_, err := gateway.NewMux(gateway.Mux{
		Routes: map[gateway.RouteKey]gateway.Target{
			{Ingress: model.APIFormatAnthropic, Model: "primary"}: {
				ID: "shared-id", Client: clientA, Model: validModel("model-a"),
			},
		},
		FormatDefaults: map[model.APIFormat]gateway.Target{
			model.APIFormatAnthropic: {
				ID: "shared-id", Client: clientB, Model: validModel("model-b"),
			},
		},
	})
	assertConfigError(t, err)
}

func TestNewMux_DuplicateLogicalRoute_SameModelReuseAllowed(t *testing.T) {
	t.Parallel()
	clientA := &fakeClient{name: "a"}
	sharedModel := validModel("model-a")
	mux, err := gateway.NewMux(gateway.Mux{
		Routes: map[gateway.RouteKey]gateway.Target{
			{Ingress: model.APIFormatAnthropic, Model: "primary"}: {
				ID: "shared-id", Client: clientA, Model: sharedModel,
			},
		},
		FormatDefaults: map[model.APIFormat]gateway.Target{
			model.APIFormatAnthropic: {
				ID: "shared-id", Client: clientA, Model: sharedModel,
			},
		},
	})
	if err != nil {
		t.Fatalf("NewMux: reusing one ID for the identical Model must be allowed, got error: %v", err)
	}
	target, err := mux.Resolve(context.Background(), model.APIFormatAnthropic, "primary")
	if err != nil {
		t.Fatalf("Resolve: unexpected error: %v", err)
	}
	if target.ID != "shared-id" {
		t.Errorf("target.ID = %q, want %q", target.ID, "shared-id")
	}
}

// TestNewMux_EmptyTargetID_AllowedAndExemptFromDuplicateCheck locks in the
// decision that an empty Target.ID is a valid ("no diagnostic identity
// asserted") configuration, and multiple Targets with an empty ID never
// collide with each other under the duplicate-ID check -- "" is not a
// meaningful identity to begin with.
func TestNewMux_EmptyTargetID_AllowedAndExemptFromDuplicateCheck(t *testing.T) {
	t.Parallel()
	clientA := &fakeClient{name: "a"}
	clientB := &fakeClient{name: "b"}
	_, err := gateway.NewMux(gateway.Mux{
		Routes: map[gateway.RouteKey]gateway.Target{
			{Ingress: model.APIFormatAnthropic, Model: "one"}: {ID: "", Client: clientA, Model: validModel("model-a")},
			{Ingress: model.APIFormatAnthropic, Model: "two"}: {ID: "", Client: clientB, Model: validModel("model-b")},
		},
	})
	if err != nil {
		t.Fatalf("NewMux: two Targets with empty ID and different Models must be allowed, got error: %v", err)
	}
}

// TestNewMux_DefensiveCopy_CallerRoutesMapMutationHasNoEffect proves that
// mutating the caller's original Routes map after NewMux returns does not
// change the constructed Mux's resolution behavior.
func TestNewMux_DefensiveCopy_CallerRoutesMapMutationHasNoEffect(t *testing.T) {
	t.Parallel()
	clientA := &fakeClient{name: "a"}
	clientEvil := &fakeClient{name: "evil"}
	key := gateway.RouteKey{Ingress: model.APIFormatAnthropic, Model: "primary"}
	routes := map[gateway.RouteKey]gateway.Target{
		key: {ID: "original", Client: clientA, Model: validModel("m")},
	}
	mux, err := gateway.NewMux(gateway.Mux{Routes: routes})
	if err != nil {
		t.Fatalf("NewMux: unexpected error: %v", err)
	}

	// Mutate the caller's original map after construction.
	routes[key] = gateway.Target{ID: "evil", Client: clientEvil, Model: validModel("evil-model")}
	routes[gateway.RouteKey{Ingress: model.APIFormatOpenAI, Model: "new"}] = gateway.Target{
		ID: "injected", Client: clientEvil, Model: validModel("injected-model"),
	}

	target, err := mux.Resolve(context.Background(), model.APIFormatAnthropic, "primary")
	if err != nil {
		t.Fatalf("Resolve: unexpected error: %v", err)
	}
	if target.ID != "original" {
		t.Errorf("target.ID = %q after caller map mutation, want unchanged %q", target.ID, "original")
	}

	if _, err := mux.Resolve(context.Background(), model.APIFormatOpenAI, "new"); err == nil {
		t.Error("Resolve found a route injected into the mux via post-construction caller-map mutation, want route-not-found")
	}
}

// TestNewMux_DefensiveCopy_CallerFormatDefaultsMapMutationHasNoEffect mirrors
// the Routes case for FormatDefaults.
func TestNewMux_DefensiveCopy_CallerFormatDefaultsMapMutationHasNoEffect(t *testing.T) {
	t.Parallel()
	clientA := &fakeClient{name: "a"}
	clientEvil := &fakeClient{name: "evil"}
	fd := map[model.APIFormat]gateway.Target{
		model.APIFormatAnthropic: {ID: "original", Client: clientA, Model: validModel("m")},
	}
	mux, err := gateway.NewMux(gateway.Mux{FormatDefaults: fd})
	if err != nil {
		t.Fatalf("NewMux: unexpected error: %v", err)
	}

	fd[model.APIFormatAnthropic] = gateway.Target{ID: "evil", Client: clientEvil, Model: validModel("evil-model")}

	target, err := mux.Resolve(context.Background(), model.APIFormatAnthropic, "unregistered")
	if err != nil {
		t.Fatalf("Resolve: unexpected error: %v", err)
	}
	if target.ID != "original" {
		t.Errorf("target.ID = %q after caller map mutation, want unchanged %q", target.ID, "original")
	}
}

// TestNewMux_DefensiveCopy_CallerDefaultPointerMutationHasNoEffect proves
// that NewMux copies the pointed-to Default Target value rather than
// retaining the caller's pointer: mutating *cfg.Default (or reassigning the
// caller's local variable) after construction must not change what a
// subsequent Resolve returns.
func TestNewMux_DefensiveCopy_CallerDefaultPointerMutationHasNoEffect(t *testing.T) {
	t.Parallel()
	clientA := &fakeClient{name: "a"}
	def := gateway.Target{ID: "original", Client: clientA, Model: validModel("m")}
	mux, err := gateway.NewMux(gateway.Mux{Default: &def})
	if err != nil {
		t.Fatalf("NewMux: unexpected error: %v", err)
	}

	// Mutate through the caller's original pointer.
	def.ID = "evil"
	def.Model = validModel("evil-model")

	target, err := mux.Resolve(context.Background(), model.APIFormatGemini, "whatever")
	if err != nil {
		t.Fatalf("Resolve: unexpected error: %v", err)
	}
	if target.ID != "original" {
		t.Errorf("target.ID = %q after caller pointer mutation, want unchanged %q", target.ID, "original")
	}
}

// TestNewMux_DefensiveCopy_TargetModelMutationAfterConstructionHasNoEffect
// proves that mutating a Target.Model value obtained by the caller before
// handing it to NewMux (via a shared pointer inside Sampling) does not
// corrupt the Mux, and separately that mutating the Target returned from
// Resolve (including its Sampling pointer/slice fields) has no effect on the
// Mux's internal state or a subsequent Resolve call.
func TestNewMux_DefensiveCopy_TargetModelMutationAfterConstructionHasNoEffect(t *testing.T) {
	t.Parallel()
	clientA := &fakeClient{name: "a"}
	temp := f64ptr(0.5)
	m := validModel("m")
	m.Sampling = model.Sampling{
		Temperature: temp,
		Stop:        []string{"a", "b"},
	}
	mux, err := gateway.NewMux(gateway.Mux{
		Routes: map[gateway.RouteKey]gateway.Target{
			{Ingress: model.APIFormatAnthropic, Model: "primary"}: {ID: "t", Client: clientA, Model: m},
		},
	})
	if err != nil {
		t.Fatalf("NewMux: unexpected error: %v", err)
	}

	// Mutate the caller's original Model's shared pointer/slice state after construction.
	*temp = 999
	m.Sampling.Stop[0] = "corrupted"

	target, err := mux.Resolve(context.Background(), model.APIFormatAnthropic, "primary")
	if err != nil {
		t.Fatalf("Resolve: unexpected error: %v", err)
	}
	if target.Model.Sampling.Temperature == nil || *target.Model.Sampling.Temperature != 0.5 {
		t.Errorf("target.Model.Sampling.Temperature corrupted by caller-side pointer mutation after NewMux: %v", target.Model.Sampling.Temperature)
	}
	if target.Model.Sampling.Stop[0] != "a" {
		t.Errorf("target.Model.Sampling.Stop corrupted by caller-side slice mutation after NewMux: %v", target.Model.Sampling.Stop)
	}

	// Now mutate the Target *returned by Resolve* and prove a fresh Resolve
	// call is unaffected -- Resolve must not return aliases into Mux state.
	*target.Model.Sampling.Temperature = -1
	target.Model.Sampling.Stop[0] = "also-corrupted"
	target.ID = "renamed"

	target2, err := mux.Resolve(context.Background(), model.APIFormatAnthropic, "primary")
	if err != nil {
		t.Fatalf("Resolve: unexpected error: %v", err)
	}
	if target2.ID != "t" {
		t.Errorf("target2.ID = %q after mutating a previously-returned Target, want unchanged %q", target2.ID, "t")
	}
	if *target2.Model.Sampling.Temperature != 0.5 {
		t.Errorf("target2.Model.Sampling.Temperature = %v after mutating a previously-returned Target, want unchanged 0.5", *target2.Model.Sampling.Temperature)
	}
	if target2.Model.Sampling.Stop[0] != "a" {
		t.Errorf("target2.Model.Sampling.Stop = %v after mutating a previously-returned Target, want unchanged [a b]", target2.Model.Sampling.Stop)
	}
}

// TestFixed_RoutesEveryRequestToOneTarget verifies that Fixed ignores the
// requested model alias (and the ingress format) entirely and always
// resolves to the same, fully validated target.
func TestFixed_RoutesEveryRequestToOneTarget(t *testing.T) {
	t.Parallel()
	clientA := &fakeClient{name: "a"}
	resolver, err := gateway.Fixed(clientA, validModel("kimi-k2"))
	if err != nil {
		t.Fatalf("Fixed: unexpected error: %v", err)
	}

	for _, tc := range []struct {
		format model.APIFormat
		alias  string
	}{
		{model.APIFormatAnthropic, "primary"},
		{model.APIFormatOpenAI, "anything-at-all"},
		{model.APIFormatGemini, ""},
	} {
		target, err := resolver.Resolve(context.Background(), tc.format, tc.alias)
		if err != nil {
			t.Fatalf("Resolve(%q, %q): unexpected error: %v", tc.format, tc.alias, err)
		}
		if target.Model.Name != "kimi-k2" {
			t.Errorf("Resolve(%q, %q) target.Model.Name = %q, want %q", tc.format, tc.alias, target.Model.Name, "kimi-k2")
		}
		if target.Client != clientA {
			t.Errorf("Resolve(%q, %q) target.Client = %v, want %v", tc.format, tc.alias, target.Client, clientA)
		}
	}
}

// TestFixed_NilClient verifies Fixed validates its Client just as NewMux
// does -- routing being trivial (always the same target) must not skip
// validation.
func TestFixed_NilClient(t *testing.T) {
	t.Parallel()
	_, err := gateway.Fixed(nil, validModel("kimi-k2"))
	assertConfigError(t, err)
}

// TestFixed_InvalidModel verifies Fixed validates its Model just as NewMux
// does.
func TestFixed_InvalidModel(t *testing.T) {
	t.Parallel()
	clientA := &fakeClient{name: "a"}
	_, err := gateway.Fixed(clientA, model.Model{})
	assertConfigError(t, err)
}

func assertConfigError(t *testing.T, err error) {
	t.Helper()
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	var cfgErr *gateway.ConfigError
	if !errors.As(err, &cfgErr) {
		t.Fatalf("error = %v (%T), want *gateway.ConfigError", err, err)
	}
}
