# Subtask 04 — Refactor `scratch.hpp`

## Delegation
**Agent:** @CodeWriter
**Model:** fast (haiku) — small file, minimal changes: rename SlabContext reference, add sycl::device overload, update comment about registry-backed make_scratch

---

## Objective

Refactor `include/slabkit/scratch.hpp` to:

1. **Update the `make_scratch(ctx, device_index, size)` signature** — change `std::shared_ptr<SlabContext>` → `std::shared_ptr<SlabKitContext>`.
2. **Add a `sycl::device` overload** to `make_scratch` that accepts `(ctx, dev, size)` and resolves `device_index` internally.
3. **Update the file-level comment** — the note at the bottom says "registry-backed make_scratch(dev, size) is defined in memory.hpp"; this note should be removed now that the registry is gone.
4. Update any other `SlabContext` references in comments or code.

This is a small file (78 lines). The `DeviceScratchBuffer` alias, `DeviceScratchView` struct, and all their members are unchanged.

---

## Todolist

### 1. Update make_scratch signature
- [ ] Change the parameter type in `make_scratch(std::shared_ptr<SlabContext> ctx, ...)`:
    - `std::shared_ptr<SlabContext> ctx` → `std::shared_ptr<SlabKitContext> ctx`
- [ ] Update the doc comment to reference `SlabKitContext` instead of `SlabContext`

### 2. Add sycl::device overload
- [ ] Add immediately after the existing `make_scratch(ctx, device_index, size)`:
    ```cpp
    /// Device overload: resolves device_index via ctx->device_index(dev).
    inline DeviceScratchBuffer make_scratch(std::shared_ptr<SlabKitContext> ctx,
                                             const sycl::device& dev,
                                             std::size_t size = 64 * 1024) {
        return DeviceScratchBuffer{&ctx->pool_for(dev), size};
    }
    ```
    Note: uses `pool_for(dev)` directly (equivalent to `pool_at(device_index(dev))` but more concise)

### 3. Remove stale registry comment
- [ ] Delete the comment: `// Note: registry-backed make_scratch(dev, size) is defined in memory.hpp`
- [ ] Update the file-level header comment block (`=====` banner) to remove any mention of "registry-backed" make_scratch

### 4. Update file banner comment
- [ ] Change the `// Exports:` list at the top of the file:
    - Remove: `//   - make_scratch(dev, size): registry-backed factory`
    - Keep: `//   - make_scratch(ctx, device_index, size): registry-free factory`
    - Add: `//   - make_scratch(ctx, dev, size): device-based registry-free factory`
    - Remove: `// Note: Factory implementations are in memory.hpp to avoid circular includes.`
      (This note only applied to the registry-backed version; both remaining factories are defined here)

### 5. Verify
- [ ] `make_scratch` with `std::shared_ptr<SlabKitContext>` compiles
- [ ] No `SlabContext` (old name) anywhere
- [ ] No `AllocatorRegistry` anywhere
- [ ] Two `make_scratch` overloads: `(ctx, device_index, size)` and `(ctx, dev, size)`
- [ ] `DeviceScratchBuffer`, `DeviceScratchView` unchanged

---

## Scope
- **Edit:** `include/slabkit/scratch.hpp`
- **Read:** `include/slabkit/context.hpp` (post-subtask-01, to confirm `SlabKitContext::pool_for(dev)` signature)
- **Write:** nothing new
- **Excluded:** `memory.hpp`, `array.hpp`, `detail/device_pool.hpp`, `detail/device_stack.hpp`, `error.hpp`, all test files

---

## Patterns

```
✅ GOOD — Updated make_scratch with device_index:
    inline DeviceScratchBuffer make_scratch(std::shared_ptr<SlabKitContext> ctx,
                                             std::size_t device_index,
                                             std::size_t size = 64 * 1024) {
        return DeviceScratchBuffer{&ctx->pool_at(device_index), size};
    }

✅ GOOD — New sycl::device overload:
    inline DeviceScratchBuffer make_scratch(std::shared_ptr<SlabKitContext> ctx,
                                             const sycl::device& dev,
                                             std::size_t size = 64 * 1024) {
        return DeviceScratchBuffer{&ctx->pool_for(dev), size};
    }

❌ BAD  — Leaving SlabContext in the parameter:
    inline DeviceScratchBuffer make_scratch(std::shared_ptr<SlabContext> ctx, ...)
    //                                                     ^^^^^^^^^^^^ old name

❌ BAD  — Adding a registry-backed overload:
    inline DeviceScratchBuffer make_scratch(const sycl::device& dev, ...) {
        // needs AllocatorRegistry — do not add
    }
```

---

## Constraints
- `DeviceScratchMemory` (from `detail/device_stack.hpp`) is still needed via `#include "slabkit/detail/device_stack.hpp"` — do not remove that include
- `DeviceScratchBuffer = DeviceScratchMemory` alias must remain
- `DeviceScratchView` struct and all its members unchanged
- `sycl/sycl.hpp` is not directly included in scratch.hpp — `sycl::device` is needed for the new overload. Add `#include <sycl/sycl.hpp>` if not already present.

## [🚫 GATE] G1 — API Shape Review

After this subtask completes, do NOT proceed to subtask-05 (test rewrite). Stop and surface the following to the user:

**Summary of work completed (subtasks 01–04):**
- `context.hpp`: Renamed `SlabContext` → `SlabKitContext`, `SlabContextBuilder` → `SlabKitContextBuilder`; builder now auto-creates `sycl::context` per platform group from devices only
- `memory.hpp`: `AllocatorRegistry` class deleted; `DeviceUSMPoolDeleter` updated; `malloc_device` now has `(count, ctx, device_index)` and `(count, ctx, dev)` overloads
- `array.hpp`: Registry-backed `make_device_array` and `MirroredArray(range, dev)` removed; `sycl::device` overloads added; all type refs updated
- `scratch.hpp`: Registry-backed `make_scratch(dev, size)` removed; `sycl::device` overload added; type refs updated

**What to review:**
- `SlabKitContext` and `SlabKitContextBuilder` in `context.hpp` — correct rename, auto-context build logic
- All factory free function signatures in `memory.hpp`, `array.hpp`, `scratch.hpp`
- Absence of any `AllocatorRegistry` references in all headers

**Approval needed:** Confirm the API shape is correct before the test files are rewritten (subtask-05 is a significant investment).

---

*At the end of this subtask, follow the checkpoint protocol in `~/.config/opencode/protocols/checkpoint.md`.*
