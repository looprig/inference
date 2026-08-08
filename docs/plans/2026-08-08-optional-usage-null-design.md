# Optional Usage Detail Null Compatibility Design

**Date:** 2026-08-08

**Status:** Approved

## Goal

Allow OpenAI-compatible providers to report optional usage-detail counters as
JSON `null` without failing an otherwise valid response, while preserving every
numeric cache and reasoning count and keeping required token totals strict.

## Problem

Some OpenAI-compatible Chat Completions providers include optional detail
fields with a JSON `null` value when the provider does not report that metric.
For example, OpenCode Go can return a numeric prompt-token total and cache-read
count alongside `prompt_tokens_details.cache_write_tokens: null`.

The shared `usagenorm.Count.TokenCount` operation deliberately distinguishes an
absent field from an explicit `null` and rejects the latter. The OpenAI codec
currently applies that strict operation to both required totals and optional
details. Consequently, a terminal usage event can fail the complete model turn
even though all required totals are valid.

## Semantics

Required totals remain strict:

- `prompt_tokens`
- `completion_tokens`

For these fields, an explicit JSON `null` remains a typed usage-normalization
error.

Optional detail counters accept either absence or JSON `null` as zero, meaning
"not reported":

- `prompt_tokens_details.cached_tokens`
- `prompt_tokens_details.cache_write_tokens`
- `completion_tokens_details.reasoning_tokens`

Numeric optional values retain the existing validation and normalization. A
numeric cache-write count is still recorded as `CacheCreationTokens`; a numeric
cache-read count is still recorded as `CacheReadTokens`; and a numeric reasoning
count is still recorded as `ReasoningTokens`. Negative, fractional, invalid, or
out-of-range values remain errors.

Zero is the neutral representation because the domain usage type has no
per-field unknown state. This does not assert that no caching happened; it means
the provider supplied no usable count for that optional metric.

## Implementation

Add a separate operation on `usagenorm.Count` for optional detail counters. It
returns zero when the field is absent or explicitly `null`, then delegates all
numeric validation to the existing strict operation. Keep `TokenCount` unchanged
so existing codecs and required fields do not silently weaken.

Use the optional operation only at the three OpenAI Chat Completions detail-field
call sites. Do not change Anthropic, Gemini, Bedrock, or Responses API semantics
as part of this compatibility fix.

## Testing

Add table cases proving that:

- optional cache-read, cache-write, and reasoning `null` values decode as zero;
- required prompt and completion `null` values still fail;
- numeric optional cache and reasoning values remain preserved;
- malformed non-null optional values remain rejected;
- terminal streaming usage with an optional `null` detail completes normally
  and exposes the remaining numeric usage.

Run the focused `openaiapi` codec tests and the complete inference module test
suite.
