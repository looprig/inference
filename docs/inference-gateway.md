# Inference gateway

`inference/gateway` is a local, loopback-only HTTP compatibility layer. It
lets a harness speaking one model-API dialect (Anthropic Messages, OpenAI
Responses, OpenAI Chat Completions, or Gemini `generateContent`) reach any
injected `inference.Client`/`model.Model` target, including a target that
speaks a *different* dialect.

`inference` is provider-policy-free: it does not import `llm`, know provider
credentials, construct provider clients, or maintain a model catalog. Targets
are arbitrary caller-supplied `inference.Client`/`model.Model` pairs, already
bound to their own credentials and connection policy. This package never
sees a provider API key.

See `../../docs/plans/2026-07-31-inference-gateway-design.md` for the full design
rationale. This document is the practical "how do I use it" reference for the
implemented API.

## Request pipeline

For every request, `gateway.Handler.ServeHTTP` runs, in order:

1. Validate method, content type, and header bounds.
2. Authenticate the local gateway token (constant-time comparison).
3. Select exactly one ingress codec (`codec.ServerCodec.MatchRequest`).
4. Apply the bounded body reader before JSON decoding.
5. Decode the native request into a `codec.DecodedRequest`.
6. Resolve `(ingress format, requested model alias)` to a `gateway.Target`
   via the configured `Resolver`.
7. Replace `Request.Model` with `Target.Model`. The harness-supplied alias is
   never sent upstream as the target model name.
8. Validate request features against `Target.Model.Caps`
   (`inference.ValidateRequestFeatures`).
9. Apply global concurrency admission (reject-on-full, never queued).
10. Invoke `Target.Client.Invoke` or `Target.Client.Stream`. Never retried.
11. Encode the result back into the *ingress* dialect — the harness-facing
    response reports the requested alias, never the resolved upstream model
    name, wherever its dialect has a model field.
12. Close all upstream bodies/readers and release admission exactly once.

`POST /v1/messages/count_tokens` (Anthropic only) is a narrower auxiliary
route: it resolves and replaces the model exactly like steps 6–7 above, then
calls a configured `contextcount.ContextCounter` — it never calls
`Target.Client` or contacts any upstream.

## Composition

A shared gateway, multiple aliases, both same- and cross-dialect routing:

```go
targets, err := gateway.NewMux(gateway.Mux{
	Routes: map[gateway.RouteKey]gateway.Target{
		{Ingress: model.APIFormatAnthropic, Model: "primary"}: {
			ID: "primary-anthropic", Client: clientA, Model: modelA,
		},
		{Ingress: model.APIFormatAnthropic, Model: "small"}: {
			ID: "small-openai", Client: clientB, Model: modelB, // cross-dialect target
		},
		{Ingress: model.APIFormatOpenAIResponses, Model: "reviewer"}: {
			ID: "reviewer-gemini", Client: clientC, Model: modelC,
		},
	},
})
if err != nil {
	return err
}

// Server (below) generates and enforces the one harness-facing token. The
// inner Handler still requires an Authenticator, so a server-owned composition
// supplies a deliberately non-authorizing internal authenticator; Handler is
// never exposed without Server's bearer check.
type internalAuthenticator struct{}
func (internalAuthenticator) Authenticate(*http.Request) error { return nil }

handler, err := gateway.New(gateway.Config{
	Resolver: targets,
	Codecs: map[model.APIFormat]codec.ServerCodec{
		model.APIFormatAnthropic:       anthropicapi.Codec{},
		model.APIFormatOpenAIResponses: openairesponses.Codec{},
		model.APIFormatOpenAI:          openaiapi.Codec{},
		model.APIFormatGemini:          geminiapi.Codec{},
	},
	Authenticate:   internalAuthenticator{},
	ContextCounter: contextcount.NewEstimator(), // optional; omit to leave count_tokens unavailable
})
if err != nil {
	return err
}

server, err := gateway.NewServer(gateway.ServerConfig{Handler: handler})
if err != nil {
	return err
}
if err := server.Start(ctx); err != nil {
	return err
}
defer server.Close(ctx)

// server generated its own random token; this is the one a harness receives.
baseURL, harnessToken, ready := server.Binding()
```

Two independent gateway servers in one process — each gets its own listener,
port, and token, and closing one never affects the other:

```go
serverA, _ := gateway.NewServer(gateway.ServerConfig{Handler: handlerA})
serverB, _ := gateway.NewServer(gateway.ServerConfig{Handler: handlerB})
_ = serverA.Start(ctx)
_ = serverB.Start(ctx)
// serverA.Close(ctx) does not affect serverB, and vice versa.
```

A single `Handler` (and the `Server` that generated its own token wrapping
it) may also be shared: multiple `Server`s can wrap one already-built
`Handler`, or multiple ACP clients can borrow one running `Server`'s binding.

## Credential-backed targets

The gateway consumes an already-constructed `inference.Client`; it does not
look up, refresh, or select credentials. Keep the package boundary explicit:

| Package | Owns | Does not own |
|---|---|---|
| `secrets` | opaque secret values, references, and local secret storage | provider identity, OAuth policy, or model selection |
| `credentials` | safe descriptors/references, sources, leases, refresh state, and HTTP authorizers | login UI, provider transports, or inference codecs |
| `inference` | neutral requests/results, codecs, transports, retry boundaries, and this gateway | provider credentials, account catalogs, or provider policy |
| `llm` | provider/API-format policy, concrete transports, and credential-backed client adapters | account lifecycle UI and harness child process policy |
| CodeRig | model catalog/configuration, explicit list/login/logout lifecycle, and child composition | raw provider credentials in a child environment |

The canonical construction path is `llm/auto.NewWithAuth(model, source)`. The
`source` is bound to one exact provider, transport, scheme, and usage class;
each call acquires a lease and the adapter authorizes only that call. The
legacy `llm/auto.New(model, inference/auth.APIKey("..."))` wrapper remains
available for an API key supplied directly by the caller. It is a compatibility
path, not environment discovery. The `inference/auth` package is likewise a
legacy facade for static keys and explicit unauthenticated requests;
call-scoped sources use `credentials/httpauth` through the `llm` adapter.

For example, a caller may provide an API key explicitly (the value is never
part of `model.Model`):

```go
client, err := auto.New(selectedModel, auth.APIKey(apiKey))
```

Or CodeRig can resolve a safe reference such as
`credential://openai/personal` from its explicit catalog/store and pass the
resulting source to `auto.NewWithAuth`. The reference identifies an account;
it is not a URL, a filesystem path, an access token, or a routing rule.

### Local stores and platform limits

`secrets/local.New` and `credentials/catalog.New` require an explicit,
absolute, clean root. They do not consult `HOME`, ambient provider variables,
or a default profile. On Darwin and Linux they use owner-only directories and
files, descriptor-relative no-follow access, bounded records, atomic rename,
and directory synchronization. A `secrets.Secret` is limited to 1 MiB;
credential references and descriptors are separately bounded. Other operating
systems return the package's typed unsupported-platform error. These stores
are not an OS keychain, cloud vault, account pool, backup service, or cross-host
sync mechanism; an integrator must provide that boundary explicitly.

### Authorization, recovery, and failure propagation

For each wire attempt, the credential adapter clones the request body, acquires
one lease, and applies its authorizer. Only an explicitly classified,
recoverable authentication rejection may invalidate that generation and cause
one additional acquire/send attempt. Quota, rate, permission, malformed
request, and transport failures are not reclassified as authentication and do
not trigger refresh. The adapter never resets an outer inference retry budget.
Cancellation is passed through acquire, refresh, transport, and stream reader
closure. A gateway quota/error response therefore propagates to the harness;
it does not select another credential, account, model, or route.

### Model identity and capabilities

`gateway.Target.Model` is the real deployment-specific `model.Model` used by
the client. The ingress alias is only a harness-facing route key and is never
sent upstream as the deployment name. Capabilities (tools, thinking, images,
structured output, and streaming) come from that target model and are checked
before invocation. A credential does not add capabilities, normalize a model
name, or authorize a different deployment. Phase-one composition has no
automatic provider selection, account routing, quota failover, or model-name
normalization; configure one explicit route and one explicit credential
reference per target.

### Harness child isolation

When CodeRig starts Claude Code, Codex, or another ACP child through this
gateway, the child receives only the loopback gateway base URL and a unique
gateway bearer token returned by `Server.Binding()`. Provider API keys,
OAuth access/refresh tokens, credential-store roots, and account state remain
with the parent process and are not copied into child environment variables,
arguments, logs, or errors. A child deliberately using its native provider
login is a separate mode: it receives its own explicitly configured native
auth boundary and does not silently borrow the parent's gateway credential.

OpenAI and Anthropic subscription registration is a provider policy boundary,
not something the gateway fabricates. If the sanctioned registration adapter
is unavailable, construction fails closed with its typed gate error; tests may
skip that provider case with the error type and reason. No browser flow,
native client ID, or undocumented token exchange is invented by `inference`.
For current vendor terminology, see the [OpenAI authentication
documentation](https://learn.chatgpt.com/docs/auth), [OpenAI Codex plan
documentation](https://help.openai.com/en/articles/11369540-using-codex-with-your-chatgpt-plan),
[OpenAI API quickstart](https://platform.openai.com/docs/quickstart/make-your-first-api-request),
[Claude Code overview](https://code.claude.com/docs/en/overview), and
[Anthropic API overview](https://platform.claude.com/docs/en/api/overview).

## Authentication: two independent layers

There are two distinct, independently-enforced bearer-token checks in this
package, and understanding why matters for anyone embedding `Handler`
directly rather than through `Server`:

- **`Config.Authenticate`** (`gateway.StaticToken(token)`) is baked into a
  `*Handler` at `gateway.New` time, using whatever token the caller chose at
  that point. It is **required** — `gateway.New` rejects a nil
  `Authenticate` with a `*ConfigError`.
- **`gateway.Server`** additionally generates its *own* independent,
  cryptographically random 256-bit token (`crypto/rand`,
  `base64.RawURLEncoding`) and wraps whatever `http.Handler` it's given in
  its own constant-time bearer-check middleware, entirely unaware of and
  independent from `Config.Authenticate`. This is deliberate: the design's
  security posture describes *the local server* — not the `Handler` — as the
  thing that "generates a cryptographically random bearer token per server",
  and `Server` is meant to be usable in front of any `http.Handler`, not only
  one built via `gateway.New`.

In the server-owned composition above, the generated `Server.Binding()` token
is the only harness-facing check: the inner authenticator is intentionally
permissive and the `Handler` is not exposed directly. If you embed a
`Handler` standalone (for `httptest`, or behind an application-owned server),
use `gateway.StaticToken` or another real authenticator and treat that handler
token as a separate boundary. Do not put an arbitrary second bearer token in a
server-owned child contract.

**Known documentation gap, not a code defect:** the design doc's own worked
composition example omits `Authenticate` from its `gateway.Config` literal
and sketches `Config.Codecs` as `[]codec.ServerCodec` rather than the
implemented `map[model.APIFormat]codec.ServerCodec`. Both are implementation
decisions made during this milestone (`Authenticate` mandatory for
fail-closed construction; `Codecs` keyed by format because
`codec.ServerCodec` carries no `APIFormat()` accessor) documented in
`config.go`'s doc comments — the design doc's example predates them and
should be read as illustrative, not copy-pasteable.

## Security posture

- Loopback-only listener (`127.0.0.1:0`), ephemeral port, not configurable.
- Independent per-server cryptographically random token with constant-time
  comparison (`crypto/subtle.ConstantTimeCompare`). A standalone `Handler`
  may add a second application-owned authenticator; the server-owned example
  uses the permissive internal authenticator so the generated token is checked
  exactly once.
- Finite, conservative default body (`DefaultMaxRequestBody`, 10 MiB) and
  concurrency (`DefaultMaxConcurrent`, 64) limits; both are configurable and
  admission is reject-on-full, never queued.
- `ReadHeaderTimeout` set on the local `http.Server` (Slowloris defense in
  depth).
- No redirects, no retries upstream, ever.
- No ambient environment inheritance, no upstream credential forwarding —
  this package never sees one.
- Immutable routes for the server's lifetime: `gateway.NewMux` and
  `gateway.Fixed` defensively copy every map/`Target`/`model.Model` they're
  given, both at construction and at every `Resolve` return (`model.Sampling`
  carries pointer/slice fields, so a shallow copy would not be enough).
- Diagnostics are bounded and secret-free by construction: every typed error
  in `errors.go`/`http_errors.go` either carries no upstream detail at all
  (`*AuthenticationError` reports one identical message for every failure
  cause) or wraps an upstream error without exposing its message directly
  (`*UpstreamInvocationError.Error()` never includes the wrapped cause's
  text — only `Unwrap()` exposes it, for a caller that already trusts that
  error).

## Provider-opaque thinking state

Visible thinking/reasoning text is portable
(`content.ThinkingBlock.Thinking`). The *opaque* replay state a provider
needs to continue a thinking-plus-tool-use turn is not: an Anthropic
signature (`content.ThinkingBlock.Signature`) is only meaningful to an
Anthropic target; a Gemini `thoughtSignature` or an OpenAI Responses
encrypted-reasoning blob (both carried in
`content.ThinkingBlock.ProviderState`) are each only meaningful to a
same-dialect target.

`ProviderState` alone is not enough to prevent cross-dialect leakage: Gemini
and OpenAI Responses independently chose the same "bare JSON-marshaled
string" wire encoding for their opaque state, so a naive implementation could
silently forward a Gemini `thoughtSignature` to an OpenAI Responses target as
if it were `encrypted_content` (or vice versa) whenever a harness replayed
history against a *different* target than the one that originally produced
it. `content.ThinkingBlock.ProviderStateFormat` closes this: every codec that
uses `ProviderState` tags it with its own dialect label on construction
(`content.NewThinkingBlock(thinking, signature, providerState,
providerStateFormat)`) and checks `ThinkingBlock.ReplayableAs(format)` before
ever forwarding it — a non-matching or absent tag degrades to exactly the
same behavior as no opaque state at all, never a translation attempt. The
bundled `codec/servertest` contract suite structurally guards this for any
future dialect via its optional `ForeignProviderStateResponse`/
`ForeignProviderStateMarker` `Config` fields (see `codec/servertest`'s doc
comments) — a new dialect that reuses another dialect's opaque-state wire
shape without checking the tag fails that contract test, not just a hoped-for
manual review.

## Bundled dialects

| API format | Route | Codec |
|---|---|---|
| Anthropic Messages | `POST /v1/messages` (+ `/v1/messages/count_tokens`) | `codec/anthropicapi.Codec{}` |
| OpenAI Responses | `POST /v1/responses` | `codec/openairesponses.Codec{}` |
| OpenAI Chat Completions | `POST /v1/chat/completions` | `codec/openaiapi.Codec{}` |
| Gemini | `POST /v1beta/models/{model}:generateContent` (+ `:streamGenerateContent?alt=sse`) | `codec/geminiapi.Codec{}` |

Every bundled `Codec` is a stateless value type satisfying both the
client-side `codec.StreamingCodec` contract (used when that dialect is a
*target*: `EncodeRequest`/`DecodeResponse`/`DecodeStream`) and the
server-side `codec.ServerCodec` contract (used when that dialect is
*ingress*: `MatchRequest`/`DecodeRequest`/`WriteResponse`/`OpenStream`/
`WriteError`). Gemini is the one dialect whose model name and streaming mode
come from the URL path, not the request body — `MatchRequest`/`DecodeRequest`
parse `{model}` and the `:generateContent`/`:streamGenerateContent` suffix
directly from `r.URL.Path`; the `?alt=sse` query parameter is accepted but
never load-bearing for routing.

Portable coverage across all four: streaming and non-streaming; system,
user, and assistant messages; text; inline and URL-referenced images; tool
definitions, calls, results, and parallel calls; visible thinking; usage;
finish reasons; native error envelopes. Explicitly out of scope for this
milestone (fails closed with a typed error, never silently dropped, except
for the documented benign fields each codec accepts-and-ignores such as
`cache_control` and request `metadata`): citations, provider-hosted tools,
audio/document content, computer-use tool variants, cross-provider reasoning
translation, `previous_response_id`/server-stored Responses conversations,
Responses WebSockets, and live route-table mutation.

## Testing

`inference/codec/servertest` is a reusable contract suite
(`servertest.Run(t, servertest.Config{...})`) proven against every bundled
dialect's real `ServerCodec` — route matching, content-type/malformed-body
rejection, decode determinism, response/error/stream encoding, single stream
termination, and (for dialects that use `ProviderState`) the foreign-state
guard described above. `inference/gateway/matrix_test.go` exercises all 16
ingress×target dialect combinations end to end through a real `Handler`,
real codecs, and real `httptest`-based fake targets built from each
dialect's own server-side codec, including explicit assertions that
opaque thinking state is preserved same-dialect and never forwarded
cross-dialect. `inference/gateway/concurrency_test.go` stress-tests
concurrent requests to different targets, the same target, mixed
streaming/non-streaming, and route resolution under load, all run with
`-race`.
