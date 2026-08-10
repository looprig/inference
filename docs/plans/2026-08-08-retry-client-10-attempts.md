# Retry Client Ten-Attempt Policy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend carbon's production inference retry policy to ten total attempts with a 256-second exponential-delay cap.

**Architecture:** Keep `inference/retry.Policy` and the retry decorator unchanged; only update the production policy at carbon's model-loader boundary. Keep the existing three stable 2-second retries, retry classification, jitter, cancellation, and stream-establishment behavior.

**Tech Stack:** Go 1.26, `inference/retry`, carbon internal app tests, Go test/vet/build.

---

### Task 1: Lock the new production policy in a failing test

**Files:**
- Modify: `carbon/internal/app/model_retry_test.go`

**Step 1: Write the failing test**

Extend `TestDefaultRetryPolicy_Valid` to assert `defaultRetryPolicy.MaxAttempts == 10` and `defaultRetryPolicy.MaxDelay == 256*time.Second`.

**Step 2: Run test to verify it fails**

Run from the carbon worktree:

```bash
GOWORK=/private/tmp/looprig-retry-10-attempts.work go test ./internal/app -run TestDefaultRetryPolicy_Valid -count=1
```

Expected: FAIL because the current production policy still has six total attempts and a 30-second cap.

### Task 2: Update the production retry policy

**Files:**
- Modify: `carbon/internal/app/model.go`

**Step 1: Write minimal implementation**

Set `MaxAttempts` to `10`, set `MaxDelay` to `256*time.Second`, and update the policy comment to describe the ten-attempt, 256-second-cap schedule.

**Step 2: Run the focused test**

```bash
GOWORK=/private/tmp/looprig-retry-10-attempts.work go test ./internal/app -run TestDefaultRetryPolicy_Valid -count=1
```

Expected: PASS.

### Task 3: Update the retry design record

**Files:**
- Modify: `inference/docs/plans/2026-08-08-retry-client-design.md`
- Create: `inference/docs/plans/2026-08-08-retry-client-10-attempts-design.md`

**Step 1: Record the approved schedule**

Document ten total attempts, the nine waits through 128 seconds, and the 256-second cap. Make the total-attempt semantics explicit so the cap is not mistaken for an additional attempt.

**Step 2: Review the diff**

```bash
git diff --check
git diff -- docs/plans/2026-08-08-retry-client-design.md docs/plans/2026-08-08-retry-client-10-attempts-design.md
```

Expected: no whitespace errors and documentation matches the code and test.

### Task 4: Verify both modules

**Files:** None.

**Step 1: Run affected tests**

```bash
GOWORK=/private/tmp/looprig-retry-10-attempts.work go test ./retry
GOWORK=/private/tmp/looprig-retry-10-attempts.work go test ./internal/app
```

**Step 2: Run full verification**

```bash
GOWORK=/private/tmp/looprig-retry-10-attempts.work go test ./...
GOWORK=/private/tmp/looprig-retry-10-attempts.work go vet ./...
GOWORK=/private/tmp/looprig-retry-10-attempts.work go build ./...
```

Expected: all commands pass for the merged inference and carbon worktrees.

### Task 5: Commit the change

**Step 1: Review status and commit**

```bash
git status --short
git add internal/app/model.go internal/app/model_retry_test.go
git commit -m "feat(retry): extend production budget to ten attempts"
```

The inference worktree commits the design/documentation updates separately:

```bash
git add docs/plans/2026-08-08-retry-client-design.md docs/plans/2026-08-08-retry-client-10-attempts-design.md docs/plans/2026-08-08-retry-client-10-attempts.md
git commit -m "docs(retry): record ten-attempt production schedule"
```
