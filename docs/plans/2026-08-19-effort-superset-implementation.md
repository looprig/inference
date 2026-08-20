# Effort Superset Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development and superpowers:verification-before-completion task-by-task.

**Goal:** Add the neutral effort superset and correct per-format mappings, then expose it through Carbon and the local model catalogue.

**Architecture:** `github.com/looprig/inference/model` owns the neutral enum. Each codec owns its exact wire mapping or typed unsupported-value error. Carbon consumes the enum for configuration, ordering, runtime selection, gateway targets, and native ACP pass-through.

**Tech Stack:** Go, JSON provider contracts, table-driven tests, Carbon `models.json`.

---

### Task 1: Extend the inference effort domain

**Files:**
- Modify: `model/effort.go`
- Modify: `model/effort_test.go`

1. Add failing cases proving `minimal` and `xhigh` are valid and unknown values remain invalid.
2. Run `GOWORK=off go test ./model` and confirm the new cases fail for the missing constants/validation.
3. Add `EffortMinimal` and `EffortXHigh` in semantic order and admit them in `Valid`.
4. Re-run the focused package test.

### Task 2: Preserve exact symbolic efforts

**Files:**
- Modify: `codec/openaiapi/encode.go`
- Modify: `codec/openaiapi/encode_test.go`
- Modify: `codec/openaiapi/server_decode.go`
- Modify: `codec/openaiapi/server_decode_test.go`
- Modify: `codec/openairesponses/encode.go`
- Modify: `codec/openairesponses/encode_test.go`
- Modify: `codec/openairesponses/server_decode.go`
- Modify: `codec/openairesponses/server_decode_test.go`
- Modify: `codec/anthropicapi/encode.go`
- Modify: `codec/anthropicapi/encode_test.go`
- Modify: `codec/anthropicapi/server_decode.go`
- Modify: `codec/anthropicapi/server_decode_test.go`

1. Add failing table cases for exact `minimal`, `xhigh`, and `max` OpenAI round trips and exact Anthropic `xhigh` round trips.
2. Add a failing Anthropic `minimal` rejection case.
3. Run the three codec package tests and confirm failures identify the old clamp/unknown paths.
4. Implement exact pass-through/decoding and a typed unsupported-effort error where the wire has no member.
5. Validate encoded fixtures against the bundled official schemas and re-run the focused tests.

### Task 3: Correct budget-format translations

**Files:**
- Modify: `codec/anthropicapi/encode.go`
- Modify: `codec/anthropicapi/encode_thinking_dialect_test.go`
- Modify: `codec/geminiapi/encode.go`
- Modify: `codec/geminiapi/encode_test.go`
- Modify: `codec/geminiapi/server_decode.go`
- Modify: `codec/geminiapi/server_decode_test.go`
- Modify: `codec/bedrockconverse/encode.go`
- Modify: `codec/bedrockconverse/encode_test.go`
- Modify: `codec/bedrockconverse/errors.go`

1. Add failing tests for the six-level Anthropic budget ordering, Google's published Gemini budgets, and explicit Gemini/Bedrock rejection of unsupported non-neutral efforts.
2. Run the affected package tests and confirm the intended failures.
3. Implement the minimal mapping and typed errors.
4. Re-run all affected package tests.

### Task 4: Extend Carbon's configuration and runtime catalogue

**Files:**
- Modify: `../carbon/internal/app/modelconfig_normalize.go`
- Modify: `../carbon/internal/app/modelconfig_validate_test.go`
- Modify: `../carbon/internal/app/modelconfig_native_test.go`
- Modify: `../carbon/internal/app/modelconfig_digest_test.go`
- Modify: `../carbon/internal/app/runtime_controls.go`
- Modify: `../carbon/internal/app/runtime_controls_test.go`
- Modify: relevant ACP catalogue and child tests under `../carbon/internal/app/`
- Modify: `../carbon/CONTRIBUTING.md`
- Modify: `../carbon/CLAUDE.md`

1. Replace the old test asserting `xhigh` is invalid with failing acceptance, ordering, digest, runtime-menu, gateway, and native ACP pass-through cases for `minimal` and `xhigh`.
2. Run focused Carbon tests under the workspace and confirm the failures.
3. Admit and rank the full superset, and include it in fallback runtime choices.
4. Update operator documentation with the neutral superset and per-model subset rule.
5. Re-run focused Carbon tests.

### Task 5: Update personal model configuration

**Files:**
- Modify: `/Users/ipotter/.looprig/carbon/models.json`

1. Change both LM Studio rows to `efforts: ["none", "minimal", "low", "medium", "high", "xhigh", "max"]` and `default_effort: "xhigh"`.
2. Validate JSON, uniqueness, exact model rows, and owner-only mode without printing credentials.

### Task 6: Verify repository boundaries

1. In `inference`, run formatting, native checks, and `GOWORK=off go test ./...`.
2. In `carbon`, run formatting and workspace tests against the local inference change.
3. Run `GOWORK=off go test ./...` in Carbon only after a published inference version is pinned; do not add a local `replace` or vendor tree.
4. Review `git diff --check`, repository-local diffs, and root workspace status without staging nested repositories.
5. Report the exact verification evidence and any release dependency still owed.
