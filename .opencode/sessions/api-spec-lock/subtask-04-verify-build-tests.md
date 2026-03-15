# Subtask 04 — verify-build-tests

## Objective
Run the full build and test suite to verify all changes from subtasks 01–03 are correct and all 64 tests still pass.

## Scope

### HW-Direct
- `pixi run build dev`
- `pixi run test`

### Excluded
- No file edits in this subtask

## Constraints
- Expected result: build succeeds with no errors, all 64 tests pass
- If tests fail, apply the build-test-debug loop (circuit breaker: 3 consecutive failures)
- Do NOT skip any test failures — all 64 must pass

## Todolist
- [ ] Run `pixi run build dev`
- [ ] Run `pixi run test`
- [ ] Verify 64/64 pass

## Delegation
- **Agent:** HW-direct (no agent delegation)
- **Type:** Build and test verification
