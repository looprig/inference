# inference

`github.com/looprig/inference` is Looprig's provider-neutral inference seam. It
defines the `Client` / `Request` / `Response` contract over Core's `content`
transcript types. It also has one wire codec per model API format, each
translating between the neutral transcript and that format's published
contract, and a generic HTTP transport that combines a codec, a router and an
authorizer into a `Client`.

The module is provider-policy-free. It holds no provider default endpoints, no
model catalogue and no credential discovery. Those live in
[`llm`](https://github.com/looprig/llm).

## Status

Released. Implemented API formats (`model.APIFormat`):

| Format | Codec package |
|---|---|
| `openai` (Chat Completions) | `codec/openaiapi` |
| `openai-responses` | `codec/openairesponses` |
| `anthropic` (Messages) | `codec/anthropicapi` |
| `gemini` (`generateContent`) | `codec/geminiapi` |
| `bedrock-converse` | `codec/bedrockconverse` |

Also implemented:

- Request and stream encoding and decoding, with SSE, NDJSON and AWS Event
  Stream framing.
- Structured output (`Request.Output`, `DecodeOutput`) and tool choice.
- Normalized token usage and deterministic context counting.
- Bounded, classified retry through `retry`.
- The loopback `gateway`, which lets a client speaking one dialect reach any
  injected `inference.Client` target. See
  [docs/inference-gateway.md](docs/inference-gateway.md).

Each codec's encoded request bodies are checked in tests against the format's
official request schema. The schemas are in `codec/conformance` and are
regenerated with `make conformance-schemas`.

## Install

```sh
go get github.com/looprig/inference@latest
```

## Packages

| Package | Purpose |
|---|---|
| `inference` | `Client`, `Request`, `Response`, tools, tool choice, structured output, `ValidateRequestFeatures`. |
| `model` | Model identity, API format, capabilities, context limits, sampling and effort (`model.CustomModel`). |
| `codec` | Codec interfaces: client-side request encoders, response and stream decoders, server-side `ServerCodec`. |
| `codec/openaiapi`, `codec/openairesponses`, `codec/anthropicapi`, `codec/geminiapi`, `codec/bedrockconverse` | One wire codec per API format. |
| `codec/conformance` | Schema conformance gate for provider wire fixtures and encoded requests. `cmd/schemagen` regenerates its schemas. |
| `codec/servertest` | Reusable contract suite for `codec.ServerCodec` implementations. |
| `transport` | Connection-bound HTTP `Client` built from an `Endpoint`, a `route.Router`, a `codec.Codec` and an authorizer. |
| `route` | Routers for the bundled wire shapes (`StaticChat`, `GeminiGenerateContent`). |
| `stream` | Pull-based `StreamReader`, stream frames, finish reasons and stream results. |
| `retry` | `inference.Client` decorator with bounded, classified retry and exponential backoff. |
| `gateway` | Loopback HTTP compatibility layer (Anthropic Messages, OpenAI Responses, OpenAI Chat, Gemini ingress). |
| `contextcount` | Deterministic complete-request context counting. |
| `usage` | Normalized token usage. |
| `failure` | Provider-neutral API, network and binding failures. |
| `auth` | Legacy authorization facade over `credentials/httpauth`, kept for static API keys and `auth.None()`. |
| `wire/sse`, `wire/ndjson`, `wire/eventstream`, `wire/jsonbody` | Byte-level framing helpers with no LLM semantics. |

## Usage

Build a request and compose an HTTP client for an OpenAI-compatible endpoint:

```go
m := model.CustomModel("local", model.APIFormatOpenAI, "http://127.0.0.1:8080/v1", "demo-model")

client := transport.New(
	transport.Endpoint{BaseURL: m.BaseURL, Provider: m.Provider, APIFormat: m.APIFormat},
	route.StaticChat("/chat/completions"),
	openaiapi.Codec{},
	auth.None(), // or a credentials/httpauth Authorizer
)

resp, err := client.Invoke(ctx, inference.Request{
	Model:  m,
	System: "Answer briefly.",
	Messages: content.AgenticMessages{&content.UserMessage{Message: content.Message{
		Role:   content.RoleUser,
		Blocks: []content.Block{&content.TextBlock{Text: "Say hello."}},
	}}},
})
```

To use a hosted provider with its endpoint, auth policy and model catalogue
already configured, take the provider package from `llm`. Runnable programs are
in `examples/`: `invoke`, `stream`, `retry` and `gateway`.

## Consumer obligation: `Request.SessionID`

`Request.SessionID` is an optional, stable identifier for a conversation.
Providers that document a per-conversation header can use it. `""` means
absent, and no neutral codec encodes it.

`ValidateRequestFeatures`, which every codec runs, checks the id for every
provider and refuses one that could not be sent unchanged as an HTTP header
value. The reasons are control bytes, surrounding whitespace, or more than
`MaxSessionIDBytes` (256) bytes. The error is `*InvalidSessionIDError`.

A refusal fails the whole request, not just the header. If you wire a session
id, pre-check it with `inference.ValidateRequestFeatures(inference.Request{SessionID: id})`,
then omit or map a refused id rather than failing every turn.

## Where it sits

Tier 2 in the Looprig workspace. Direct Looprig dependencies: `core`,
`credentials`, `secrets`. Consumers include `llm`, `harness`, `eval`,
`classifiers`, `mcp`, `tui` and `pluto`.

## Development

The Go baseline is 1.26.8. The module does not vendor. Verify it standalone:

```sh
GOWORK=off go test ./...
make check                # gofmt check, vet, staticcheck, gosec, govulncheck, race tests, build
make conformance-schemas  # re-derive codec/conformance/schema from upstream specs (network)
```

Other targets: `fmt`, `fmt-check`, `vet`, `test`, `lint`, `vuln`, `secure`,
`check-staticcheck`, `check-gosec`, `check-vuln`, `build`. `make fuzz` only
prints how to run a fuzz target:

```sh
go test -fuzz=FuzzXxx ./path/to/pkg -fuzztime=30s
```

The lint and security tools are pinned by `tool` directives in `go.mod`. The
tests need no network, credentials or build tags.

## License

Apache License 2.0. See [LICENSE](LICENSE).
