# Inference Package Boundaries

## Goal

Organize the inference module by domain without changing request encoding, response decoding, streaming, routing, authentication, transport, counting, or error behavior.

## Package layout

- `inference` owns the provider-neutral invocation contract: `Client`, `Request`, `Response`, and `Tool`.
- `model` owns model identity, origin, API format, capabilities, context limits, sampling, effort, and model validation.
- `stream` owns frames, stream readers, stream results, finish reasons, and stream errors.
- `codec` owns request encoding and response and stream decoding contracts.
- `contextcount` owns context counting contracts, compatibility checks, and estimators.
- `usage` owns usage normalization errors and fields.
- `auth` owns the authenticator contract and concrete authenticators.
- `route` owns route selection contracts and concrete routers.
- `transport` owns endpoints and HTTP execution.
- `failure` owns provider-neutral API, network, and binding failures shared by transports and provider integrations.
- `wire` keeps protocol framing implementations.

The root package may depend on `model` and `stream`. Domain packages may depend on the root invocation contract where needed, but the root package must not depend on codec, auth, route, transport, or wire implementations.

## Migration

Move declarations and their tests to the package that owns them, then update Harness, LLM, TUI, and CodeRig imports in the same change. Do not retain broad aliases in the root package merely to preserve the old catch-all API.

The migration is structural. Existing values, validation rules, JSON behavior, error messages, and execution paths remain unchanged.

## Verification

Run formatting, dependency-boundary tests, and race suites across Inference and its source-level consumers. Vendored copies are regenerated only through the owning module's normal vendoring workflow.
