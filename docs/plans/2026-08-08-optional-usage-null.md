# Optional Usage Detail Null Compatibility Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Decode JSON `null` as zero only for optional OpenAI-compatible usage-detail counters while preserving strict required totals and all numeric cache/reasoning accounting.

**Architecture:** Extend the internal count normalizer with an explicitly named optional-detail operation rather than weakening the existing strict operation. Apply it only to the three OpenAI Chat Completions detail fields, then cover both ordinary response decoding and terminal streaming usage.

**Tech Stack:** Go, `encoding/json`, table-driven tests, inference OpenAI codec and stream collector.

---

### Task 1: Specify optional-detail count semantics

**Files:**
- Modify: `codec/openaiapi/decode_test.go`
- Modify: `internal/usagenorm/count.go`

**Step 1: Write the failing decoder tests**

Add table cases to `TestDecodeResponseUsageNormalization`:

```go
{
	name:       "null optional details are unreported",
	usageField: `,"usage":{"prompt_tokens":10,"completion_tokens":6,"prompt_tokens_details":{"cached_tokens":null,"cache_write_tokens":null},"completion_tokens_details":{"reasoning_tokens":null}}`,
	want:       &content.Usage{InputTokens: 10, OutputTokens: 6},
},
{
	name:       "null completion remains invalid",
	usageField: `,"usage":{"completion_tokens":null}`,
	wantField:  usage.UsageNormalizationFieldOutputTokens,
	wantReason: usage.UsageNormalizationReasonNull,
},
```

Keep the existing numeric cache/read/reasoning case and strict null-prompt case unchanged; together they prove that caching data remains preserved and required totals remain strict.

**Step 2: Run the focused test to verify RED**

Run:

```bash
go test ./codec/openaiapi -run TestDecodeResponseUsageNormalization -count=1
```

Expected: FAIL for `null_optional_details_are_unreported` with `cannot normalize usage field CacheReadTokens: null`.

**Step 3: Add the minimal optional normalizer**

Add this method beside `TokenCount` in `internal/usagenorm/count.go`:

```go
// OptionalTokenCount returns zero when an optional usage-detail field is absent
// or explicitly null. Every present non-null value retains TokenCount's strict
// numeric validation.
func (c Count) OptionalTokenCount(field Field) (content.TokenCount, error) {
	if !c.present || bytes.Equal(bytes.TrimSpace(c.raw), []byte("null")) {
		return 0, nil
	}
	return c.TokenCount(field)
}
```

Do not change `TokenCount`.

**Step 4: Apply it only to OpenAI optional details**

In `codec/openaiapi/decode.go`, replace `TokenCount` with `OptionalTokenCount` for:

```go
wire.PromptTokensDetails.CachedTokens
wire.PromptTokensDetails.CacheWriteTokens
wire.CompletionTokensDetails.ReasoningTokens
```

Leave `PromptTokens` and `CompletionTokens` on strict `TokenCount`.

**Step 5: Run the focused test to verify GREEN**

Run:

```bash
go test ./codec/openaiapi -run TestDecodeResponseUsageNormalization -count=1
```

Expected: PASS.

**Step 6: Commit the first behavior slice**

```bash
git add internal/usagenorm/count.go codec/openaiapi/decode.go codec/openaiapi/decode_test.go
git commit -m "fix(openaiapi): accept null optional usage details"
```

### Task 2: Cover terminal streaming usage

**Files:**
- Modify: `codec/openaiapi/stream_test.go`

**Step 1: Write a streaming regression test**

Add a case to the stream usage-result table using a terminal event shaped like:

```json
{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":6,"prompt_tokens_details":{"cached_tokens":3,"cache_write_tokens":null},"completion_tokens_details":{"reasoning_tokens":2}}}
```

Assert that the result contains:

```go
&content.Usage{
	InputTokens:     7,
	OutputTokens:    6,
	CacheReadTokens: 3,
	ReasoningTokens: 2,
}
```

To prove the test is a real regression test after Task 1, temporarily switch the cache-write call site back to strict `TokenCount`.

**Step 2: Run the stream test to verify RED**

Run:

```bash
go test ./codec/openaiapi -run TestStreamUsageResult -count=1
```

Expected: FAIL with `cannot normalize usage field CacheCreationTokens: null`.

**Step 3: Restore the optional call site**

Restore `OptionalTokenCount` for `CacheWriteTokens`; make no other production change.

**Step 4: Run the stream test to verify GREEN**

Run:

```bash
go test ./codec/openaiapi -run TestStreamUsageResult -count=1
```

Expected: PASS.

**Step 5: Commit the stream regression**

```bash
git add codec/openaiapi/stream_test.go
git commit -m "test(openaiapi): cover null cache-write stream usage"
```

### Task 3: Verify the module

**Files:**
- Verify only

**Step 1: Format touched Go files**

```bash
gofmt -w internal/usagenorm/count.go codec/openaiapi/decode.go codec/openaiapi/decode_test.go codec/openaiapi/stream_test.go
```

**Step 2: Run focused codec tests**

```bash
go test ./codec/openaiapi -count=1
```

Expected: PASS.

**Step 3: Run the full module test suite**

```bash
go test ./... -count=1
```

Expected: PASS.

**Step 4: Inspect the final diff and repository state**

```bash
git diff --check HEAD~2..HEAD
git status --short
```

Expected: no whitespace errors and a clean working tree.
