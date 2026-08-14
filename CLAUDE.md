# CLAUDE.md — Inference Development Guidelines

This module owns the wire codecs: one package per **API format**, each translating
between the neutral transcript and a provider's published contract. The workspace
`AGENTS.md` remains canonical for the dependency graph and release policy.

## The schema is the contract

Every API format is defined by an **official, machine-readable specification**. That
document — not the prose documentation, not an SDK's README, not a community
reconstruction — is the authority for what a request may contain and what a response
may carry.

Use first-party sources. Where a provider publishes no first-party-hosted document,
record the chain of custody instead of quietly substituting a third party:

| Format | Source | Notes |
|---|---|---|
| `openai`, `openai-responses` | `github.com/openai/openai-openapi` | OpenAI's own repo, OpenAPI 3.1 |
| `anthropic` | the `openapi_spec_url` in `anthropics/anthropic-sdk-go/.stats.yml` | no first-party-hosted document exists; the pointer is first-party, the hosting is their SDK vendor's |
| `gemini` | `generativelanguage.googleapis.com/$discovery/rest?version=v1beta` | Google's own production endpoint |
| `bedrock-converse` | `aws/api-models-aws` | AWS's official model distribution. **Not** botocore's vendored copy, which strips regex anchors |

Resolved JSON Schema documents, their provenance, and the refresh path live in
this module, at `codec/conformance`. Regenerate with `make conformance-schemas`;
never hand-edit a generated schema.

The gate used to live one tier up, in `llm`, and behind an `internal/`, which put
it out of reach of the codec encode tests the rule below targets most directly.
It is exported from here so both this module and `llm` can hold a body against
the schema that governs it.

## Validate what we encode, not just what we decode

Response fixtures test our tolerance of what a provider sends. **Request validation
tests what we send**, and that is where our own bugs live. Every encode path must
hold its encoded body against the format's official *request* schema in tests.

Request schemas are materially stricter than response schemas — Anthropic closes 83
of 84 request object shapes and declares required properties on 81 of them, against
5 of 49 on the response side. That strictness is the point.

Know the gate's real strength per format and state it rather than assuming uniform
coverage. Bedrock's `ConverseRequest` requires nothing at the top level, because AWS
marks only `modelId` `@required` and it lives in the URI path. Gemini declares
required properties on 1 of 49 request shapes. Neither gate can see inside an
`any`-typed field such as `parametersJsonSchema` or `functionCall.args`.

**Measure a gate's strength; do not reason about it.** Feed it deliberately wrong
shapes and record which are rejected. A spec-derived document often enforces less than
its source appears to promise — Gemini's derived request schema contains no `oneOf` at
all, so it accepts a two-member `Part`, base64 that is not base64, and a mime type the
declared union does not admit. Every belief about a gate that was reasoned rather than
probed has so far turned out to be wrong. Where a gate is blind, hold the constraint
somewhere that is not blind — an encoder allowlist, an explicit assertion, a golden
comparison — and say in the comment which one carries it.

**Never weaken a fixture or a gate to make something pass.** If the gate rejects, the
encoder is wrong until proven otherwise; if the spec itself is ambiguous, fix the
converter and record why.

## Rules earned from real defects

**A field the schema marks `required` must never carry `omitempty`.** Go's zero value
erases the key, and the provider rejects the object. A tagged union whose variants
have different required sets needs a per-variant marshaller, not a shared struct with
optional tags. This single mistake produced illegal `thinking`, `redacted_thinking`,
`function_call_output.output`, `reasoning.id`, `tool_use.id` and `FunctionTool.strict`.

**An empty slice must encode as `[]`, never `null`,** for a field the schema types as
an array. `json.Marshal` writes `null` for a nil slice, and array-typed fields reject
it. This appeared independently in four codecs — always start from a non-nil slice.

**Transcribe the spec's constraints into encoder validation.** Patterns, length caps,
enum members and numeric ranges are part of the contract. Fail closed locally with a
typed error rather than sending a request you can prove will 400 — a local error names
the field; a provider 400 does not.

**Prefer an allowlist to a denylist** when validating a union member or an enum, so a
value the provider adds later fails closed instead of leaking through.

**Never silently drop caller intent.** If a schema cannot represent something the
caller asked for — a tool's `additionalProperties`, a document source, a sampling
value — either route it to a field that can hold it or fail loudly. A silently
degraded request is worse than a rejected one, because nobody learns about it.

**An accounting field must never discard a completed generation.** Usage mismatches,
nullable token counts and unmodelled detail fields are metrics; content is the
product. Clamp or report, do not fail the response.

## Streaming

Malformed JSON in a stream is an **error**, never a skipped frame — a skipped frame
becomes silent content loss reported as success. A well-formed provider **error
object** is likewise an error, including when it arrives with HTTP 200. Unknown but
structurally valid event types are ignored for forward compatibility.

A stream that ends without its terminal event is truncated, not complete. Where a
format defines a terminal signal — Gemini's non-empty `finishReason`, Anthropic's
`message_stop`, Responses' `response.completed` — gate the result on having seen it.

Streaming must reconstruct the **same continuation state** as the non-streaming
decoder for the same response, including per-block reasoning signatures. If the two
paths disagree, the streaming path is wrong.

## Provider-opaque state

Encrypted reasoning and vendor continuation blobs are carried in `ProviderState` with
a `ProviderStateFormat` tag, and replayed **only** to the dialect that issued them —
check `ReplayableAs` at every forwarding site. Preserve the bytes exactly; never
parse, reorder or regenerate them.

A native reasoning **signature** is the second, independent channel of the same kind
of state — `ThinkingBlock.Signature` with its own `SignatureFormat` label, read only
through `SignatureReplayableAs`. It is separate from `ProviderState` because a block
carries one or the other (a signed visible block, or an opaque redacted one), and one
label must never authorize the other's replay.

The signature's degrade rule is the opposite of `ProviderState`'s, and this is the
part to get right. `ReplayableAs` returning false means "treat as absent". A signature
you cannot claim must instead **fail the encode**: the issuer verifies it
cryptographically, so forwarding it draws a 400 — and so does dropping it, because an
unsigned `thinking` block is rejected too (Anthropic's own request schema marks
`signature` required, and a live probe returns the same 400 for an empty one). Both
degrades lose, so refuse locally where the diagnostic can name the foreign dialect.
An **unlabelled** signature is refused for the same reason: no provable issuer.
Bedrock Converse and the Messages API front the same Claude models with structurally
identical reasoning blocks and non-interchangeable signatures, so the label is the
only thing that tells them apart.

Any code that copies, serializes or sanitizes a content block must preserve these
fields. Do not hand-write a copy helper: `content.CloneBlock` / `content.CloneBlocks`
own that, they copy the struct whole rather than enumerating fields, and they are
faithful — nil-versus-empty and a half-set provider-state pair come back exactly as
they went in. The `core/content` constructors are for CONSTRUCTION, where normalizing
a half-set pair is correct; a copy is not construction.

Pin the guarantee with a reflection-driven test that populates every exported field, so
a new field fails loudly instead of vanishing. `core/content/blocktest` builds those
fixtures and checks its own variant list against the sealed union as content declares
it; `blocktest.AssertIndependent` reports any byte array or pointer a copy still shares.

## Testing

A test must never assert defective behaviour. Nine such tests were found in this
codebase — each one prevented a real fix from landing and had to be rewritten. If a
test's expectation contradicts the official schema, the test is wrong.

Compare JSON semantically, not byte-exactly, unless the bytes are the contract. Tool
arguments and provider state *are* the contract and must round-trip verbatim;
formatting elsewhere is not.

**Per-package green does not compose.** Codec changes shift byte counts in
`contextcount`, alter fixtures in `llm`, and change what `harness` materializes. Run
the affected repositories together before believing a result.
