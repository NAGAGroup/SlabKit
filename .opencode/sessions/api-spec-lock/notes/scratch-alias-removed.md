# Note: DeviceScratchBuffer Alias Removed

**Date:** 2026-03-15
**Subtask:** 03 — remove-scratch-alias

## What was done

The deprecated `DeviceScratchBuffer` type alias (`using DeviceScratchBuffer = DeviceScratchMemory;`) was completely removed from `include/slabkit/scratch.hpp`.

## Files changed

- `include/slabkit/scratch.hpp` — alias line removed
- `tests/test_api.cpp` — 2 replacements (test name + variable declaration)
- `docs/api-reference.md` — deprecated section deleted entirely; `make_scratch` return type references updated
- `docs/anti-patterns.md` — "Anti-Pattern 8: Using the Deprecated DeviceScratchBuffer Alias" section deleted; remaining anti-patterns renumbered

## Files checked, no changes needed

- `tests/test_memory.cpp`, `test_context.cpp`, `test_array.cpp`, `test_context_array.cpp`, `test_transfers.cpp`
- `docs/patterns.md`, `docs/concepts.md`, `docs/getting-started.md`

## Verification

`grep -r DeviceScratchBuffer` across all `.hpp`, `.cpp`, `.md` files returns zero results.
