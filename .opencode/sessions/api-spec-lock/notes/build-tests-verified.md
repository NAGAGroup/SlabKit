# Build and Test Verification — Subtask 04

## Date
2026-03-15

## Summary
All changes from the api-spec-lock session were verified by running the full build and test suite.

## Results
- `pixi run build dev`: ✅ 8/8 targets built, 0 errors, 0 warnings (acpp optimization notices are expected in dev builds)
- `pixi run test dev`: ✅ 64/64 tests passed

## Test invocation
Note: `pixi run test` without arguments defaults to the `default` preset (no tests built). Use `pixi run test dev` to run tests against the dev build.

## Changes verified
- Subtask 01: API surface spec written to `docs/design/api-surface-spec.md`
- Subtask 02: Bug fixes (cross-device assert, bounds-check ordering) and `[[nodiscard]]` annotations applied; Context made fully non-movable
- Subtask 03: `DeviceScratchBuffer` alias fully removed and all usages migrated to `DeviceScratchMemory`
