# Session: remove-allocator-registry-singleton

**Goal:** Remove the AllocatorRegistry singleton, rename SlabContext → SlabKitContext with an auto-context builder, add `sycl::device`-based factory methods, and enforce explicit context ownership throughout the library.

---

## Done Criteria

- [x] `AllocatorRegistry` class deleted entirely (no `get()`, no `Config`, no singleton)
- [x] `SlabContext` renamed to `Context` across all headers and tests
- [x] `SlabContextBuilder` renamed to `ContextBuilder`; builder accepts only `devices(...)` — sycl::context is auto-created per platform group
- [x] All public factory free functions require an explicit `Context` parameter — no registry-backed overloads remain
- [x] `sycl::device` overloads added to free factory functions (`malloc_device`, `make_device_array`, `make_scratch`)
- [x] Lazy pool creation for pre-registered devices preserved (unchanged behaviour, just confirmed)
- [x] All 6 test files updated and compiling
- [x] `pixi run test dev` passes with zero failures
- [x] Design doc updated to reflect new architecture

---

## Subtask Table

| # | Status | Description |
|---|--------|-------------|
| 01 | ✅ complete | Refactor `context.hpp`: rename SlabContext→SlabKitContext, update builder to auto-create contexts — **CodeWriter / standard** |
| 02 | ✅ complete | Refactor `memory.hpp`: delete AllocatorRegistry, update pool deleter, add `sycl::device` overload to `malloc_device` — **CodeWriter / standard** |
| 03 | ✅ complete | Refactor `array.hpp`: remove registry-backed factory overloads, add `sycl::device` overloads, update type refs — **CodeWriter / standard** |
| 04 | ✅ complete | Refactor `scratch.hpp`: remove registry-backed overloads, add `sycl::device` overloads, update type refs — **CodeWriter / standard** |
| G1 | ✅ complete | Review API shape before test rewrite |
| 05 | ✅ complete | Update all 6 test files for new API — **CodeWriter / standard** |
| 06 | ✅ complete | Build & test — **HeadWrench** |
| 07 | ✅ complete | Update `docs/design/slabkit-v0.1-design.md` — **DocWriter / fast** |

---

## Gates

### G1 — API Shape Review

**Stop condition:** After subtask-04 completes, pause before rewriting tests.

**What to review:**
- `SlabKitContext` public API (methods, construction via builder)
- `SlabKitContextBuilder` auto-context creation logic
- Free function signatures in `memory.hpp`, `array.hpp`, `scratch.hpp`
- Absence of all `AllocatorRegistry` references

**Approval needed:** Explicit user sign-off that the API shape is correct before the test rewrite investment.

---

## Current Focus

**Status:** ✅ Complete — all 7 subtasks done. Session `remove-allocator-registry-singleton` finished 2026-03-10.

**Final commit:** `docs: update slabkit-v0.1-design.md to reflect post-refactor architecture`

---

## Scope

### In scope
- `include/slabkit/context.hpp` — rename + builder refactor
- `include/slabkit/memory.hpp` — delete AllocatorRegistry, update pool deleter
- `include/slabkit/array.hpp` — remove registry overloads, add device overloads
- `include/slabkit/scratch.hpp` — remove registry overloads, add device overloads
- `include/slabkit/slabkit.hpp` — verify, no changes expected
- `tests/test_memory.cpp`, `test_array.cpp`, `test_context.cpp`, `test_context_array.cpp`, `test_api.cpp`, `test_transfers.cpp`
- `docs/design/slabkit-v0.1-design.md`

### Out of scope
- `include/slabkit/detail/device_pool.hpp` — no changes
- `include/slabkit/detail/device_stack.hpp` — no changes
- `include/slabkit/error.hpp` — no changes
- Build system files (`CMakeLists.txt`, `pixi.toml`, `CMakePresets.json`)
- New SYCL backends or device discovery strategies

---

## Patterns & Constraints

- **Header-only**: All implementations remain in `.hpp` files — no `.cpp` source files
- **weak_ptr lifetime safety**: `DeviceUSMPoolDeleter` holds `weak_ptr<SlabKitContext>` — pattern must be preserved
- **Thread-safe lazy pool init**: Double-checked locking in `SlabKitContext::pool_at()` must be preserved
- **DeviceMemoryPool internals**: `detail/device_pool.hpp` is off-limits — bitmap freelist unchanged
- **DeviceArray / View / SubView API**: Indexing and view types unchanged
- **No auto-discovery**: `SlabKitContextBuilder` takes explicit device lists — no `sycl::device::get_devices()` call
- **Lazy pool = pre-registered devices only**: No dynamic device registration after context construction
- **Build commands**: `pixi run configure dev`, `pixi run build dev`, `pixi run test dev`
- **Git**: Work directly on `main`; use conventional commit format (`refactor:`, `feat:`, `test:`, `docs:`, `wip:`)
- **Circuit breaker**: Stop after 3 consecutive build/test failures
