# Effort Superset Design

## Goal

Represent the union of supported reasoning-effort controls across Looprig's
built-in inference formats without silently dropping, renaming, or clamping a
caller's selection. Carbon must accept the same neutral vocabulary for gateway
models and native ACP allowlists.

`ultra` is intentionally excluded because it is an orchestration mode rather
than a provider reasoning-effort value.

## Neutral vocabulary

The ordered neutral vocabulary is:

```text
none < minimal < low < medium < high < xhigh < max
```

`none` remains the zero value and means that no explicit reasoning control is
sent. Every other value is a distinct caller intent. `xhigh` and `max` must not
collapse into `high` when a wire format can represent them.

Carbon's `models.json` keeps a per-model allowlist. The parser accepts the
neutral superset, while an operator advertises only the subset supported by a
particular model. Native ACP still performs its existing lazy check against the
adapter's live advertised model/effort options.

## Format mappings

### OpenAI Chat Completions and Responses

The checked-in official schemas admit `none`, `minimal`, `low`, `medium`,
`high`, `xhigh`, and `max`. Both codecs pass through every non-`none` value
exactly. `none` omits the wire field. Server-side decoders invert the same
mapping and reject unknown values.

### Anthropic Messages

Adaptive thinking admits `low`, `medium`, `high`, `xhigh`, and `max` as
`output_config.effort`; `none` omits effort and thinking. `minimal` has no
Anthropic enum member and is rejected rather than clamped.

For the legacy budget dialect, Looprig retains its explicit proportional
translation because the wire accepts `budget_tokens`, not the complete neutral
ladder. The ordered policy becomes 10%, 25%, 50%, 75%, 85%, and 90% of
`max_tokens` for minimal through max, subject to the existing schema minimum
and strict-below-output-cap checks. This is a Looprig policy, not a claim that
Anthropic publishes those percentages.

### Gemini GenerateContent

Google documents `minimal` and `low` as 1,024 thinking tokens, `medium` as
8,192, and `high` as 24,576 for Gemini 2.5's budget control. Looprig uses those
published values. `none` omits thinking. Gemini defines neither `xhigh` nor
`max` in its level mapping, so those values fail locally with a typed error
instead of being clamped or omitted.

### Bedrock Converse

Converse has no model-neutral reasoning-effort member. Its
`additionalModelRequestFields` escape hatch is model-specific and is not
populated by Looprig's neutral request. Any non-`none` effort therefore fails
locally with a typed error. `none` continues to omit reasoning control.

## Carbon behavior

Carbon accepts and ranks the complete neutral vocabulary for gateway models,
native ACP model allowlists, runtime controls, catalogue fingerprints, and
tool schemas. The full fallback effort menu uses the same order. Existing
configurations retain their exact meaning and stable order.

The two local LM Studio models advertise all seven neutral values and default
to `xhigh`. Because their `api_format` is `openai`, the selected value is sent
as an exact `reasoning_effort` string.

## Verification

Tests first pin the neutral enum, every encoder and decoder mapping, explicit
unsupported-format failures, Carbon normalization/ranking, native ACP
pass-through, runtime choices, and configuration digest stability. Request
fixtures are checked against the bundled provider schemas where the schema is
strong enough to express the field.

Standalone verification runs with `GOWORK=off`, inference before Carbon. Carbon
cannot pass standalone against a new inference symbol until the inference
version exists remotely and Carbon's `go.mod` is updated, so that dependency
release boundary must be reported rather than hidden with a local `replace`.
