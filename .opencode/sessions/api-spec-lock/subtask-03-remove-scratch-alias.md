# Subtask 03 — remove-scratch-alias

## Objective
Remove the deprecated `DeviceScratchBuffer` type alias and migrate all uses in tests, patterns documentation, and the API reference to use `DeviceScratchMemory` directly.

## Scope

### Edit
- `include/slabkit/scratch.hpp` — remove the alias
- `tests/test_api.cpp` — replace `DeviceScratchBuffer` with `DeviceScratchMemory`
- `tests/test_memory.cpp` — replace if present
- `tests/test_context.cpp` — replace if present
- `tests/test_array.cpp` — replace if present
- `tests/test_context_array.cpp` — replace if present
- `tests/test_transfers.cpp` — replace if present
- `docs/patterns.md` — update Pattern 5 to use `DeviceScratchMemory`
- `docs/api-reference.md` — remove the deprecated alias entry

### Read
- All files being edited (read before editing)

### Excluded
- `docs/concepts.md` — check for mentions but likely no changes needed
- `docs/getting-started.md` — check for mentions but likely no changes needed
- `docs/anti-patterns.md` — check for mentions
- `include/slabkit/detail/device_stack.hpp` — no changes (internal class name stays `DeviceScratchMemory`)

## Constraints
- The alias `using DeviceScratchBuffer = DeviceScratchMemory;` must be completely removed from `scratch.hpp`
- In tests, all occurrences of `DeviceScratchBuffer` must be replaced with `DeviceScratchMemory` — no occurrences should remain
- In `docs/patterns.md`, Pattern 5 currently demonstrates scratch memory usage with `DeviceScratchBuffer` — update all variable type declarations and any mentions to use `DeviceScratchMemory`
- In `docs/api-reference.md`, remove or clearly strike through the `DeviceScratchBuffer` alias section; do not just add a note — fully remove the alias documentation since the type no longer exists
- Do NOT rename the internal `DeviceScratchMemory` class itself — it stays named `DeviceScratchMemory` in `device_stack.hpp`
- Do NOT change any function logic, only type name replacements
- Check `docs/concepts.md`, `docs/getting-started.md`, and `docs/anti-patterns.md` for any `DeviceScratchBuffer` mentions and update if found

## Todolist
- [ ] Read scratch.hpp and remove the DeviceScratchBuffer alias line
- [ ] Search all test files for DeviceScratchBuffer and replace with DeviceScratchMemory
- [ ] Update docs/patterns.md Pattern 5
- [ ] Update docs/api-reference.md to remove DeviceScratchBuffer entry
- [ ] Check docs/concepts.md, docs/getting-started.md, docs/anti-patterns.md for stray mentions and fix if found

## Delegation
- **Agent:** SlabKitImplAgent
- **Model:** PLACEHOLDER_MODEL_ID
- **Type:** Implementation (alias removal + migration)
