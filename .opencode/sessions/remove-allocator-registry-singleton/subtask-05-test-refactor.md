# Subtask 05 — Refactor all test files

## Delegation
- **Agent:** @CodeWriter
- **Model tier:** standard (claude-sonnet) — six files, significant structural rewrites, must reason about what registry-backed tests become, and what new sycl::device overload tests should cover
- **Reason:** High context breadth across six files with non-trivial transformation logic; standard tier needed to handle multi-file coherence and judgment calls on test preservation vs replacement

---

## Objective

All six test files use the old API (`AllocatorRegistry`, `SlabContext`, `SlabContextBuilder().contexts(...)`, registry-backed free function overloads). After subtasks 01–04 have removed all of those, the tests will fail to compile. This subtask rewrites them to use the new `SlabKitContext` / `SlabKitContextBuilder` API exclusively.

**Key transformations per file:**

- `test_context.cpp` — rename `SlabContext` → `SlabKitContext`, `SlabContextBuilder` → `SlabKitContextBuilder`; remove `.contexts({sycl_ctx})` from all builder calls (builder now auto-creates contexts); update the "throws if not covered" test to test the new guard condition (`devices({})` — empty device list)
- `test_memory.cpp` — the entire `AllocatorRegistry` block is **deleted**; replace the scratch tests with a proper context-backed setup
- `test_array.cpp` — update `make_test_context()` helper; no `.contexts()` call; remove `SlabContext` type references
- `test_context_array.cpp` — same as `test_array.cpp`; update the helper
- `test_api.cpp` — this file is almost entirely registry-backed tests; most of the file is **deleted and rewritten** as context-backed equivalents; see detailed section below
- `test_transfers.cpp` — update `make_test_context()` helper; no `.contexts()` call; the rest of this file is already context-backed and should need minimal changes

---

## Todolist

### 1. Update `test_context.cpp`
- [ ] In every `SlabContextBuilder()` call: remove `.contexts({sycl_ctx})` and remove the `sycl::context sycl_ctx{dev};` line that preceded it
- [ ] Rename all `slabkit::SlabContextBuilder` → `slabkit::SlabKitContextBuilder`
- [ ] Rename all `slabkit::SlabContext` → `slabkit::SlabKitContext`
- [ ] Update TEST_CASE names and section descriptions that say "SlabContext" → "SlabKitContext"
- [ ] Update the test `"SlabContextBuilder throws if device not covered by any context"`: the new guard is `devices({})` (empty device list) — change to:
  ```cpp
  REQUIRE_THROWS_AS(slabkit::SlabKitContextBuilder()
                        .devices({})
                        .slab_size_per_device(1024 * 1024)
                        .build(),
                    std::invalid_argument);
  ```
- [ ] Update the builder test for "builds valid context for CPU device" — it should now verify that `platform_contexts` is auto-created (1 context for 1 CPU device, no `.contexts()` call):
  ```cpp
  sycl::device dev{sycl::cpu_selector_v};
  auto ctx = slabkit::SlabKitContextBuilder()
                 .devices({dev})
                 .node_size(256)
                 .slab_size_per_device(1024 * 1024)
                 .build();
  REQUIRE(ctx->devices.size() == 1);
  REQUIRE(ctx->platform_contexts.size() == 1);
  REQUIRE(ctx->device_to_platform.size() == 1);
  REQUIRE(ctx->device_to_platform[0] == 0);
  ```

### 2. Update `test_memory.cpp`
- [ ] **Delete** the entire `AllocatorRegistry` block (all test cases in the `[memory][registry]` and `[memory][registry]` sections — lines 9–52 in the current file)
- [ ] **Delete** the `AllocatorRegistry::Config` test cases
- [ ] Replace the `make_scratch construction` test with a context-backed version:
  ```cpp
  TEST_CASE("make_scratch construction with context", "[memory][scratch]") {
      sycl::device dev{sycl::cpu_selector_v};
      auto ctx = slabkit::SlabKitContextBuilder()
                     .devices({dev})
                     .slab_size_per_device(1024 * 1024)
                     .build();
      REQUIRE_NOTHROW(slabkit::make_scratch(ctx, 0));
  }
  ```
- [ ] Update the `DeviceScratchMemory allocate and unwind` test to build a context explicitly:
  ```cpp
  TEST_CASE("DeviceScratchMemory allocate and unwind", "[memory][scratch]") {
      sycl::device dev{sycl::cpu_selector_v};
      auto ctx = slabkit::SlabKitContextBuilder()
                     .devices({dev})
                     .slab_size_per_device(1024 * 1024)
                     .build();
      auto scratch = slabkit::make_scratch(ctx, 0);
      // ... rest of the SECTION blocks unchanged ...
  }
  ```
- [ ] Add `#include "slabkit/context.hpp"` and `#include <sycl/sycl.hpp>` at the top if not already present

### 3. Update `test_array.cpp`
- [ ] Update `make_test_context()` helper: remove `sycl::context sycl_ctx{dev};` and `.contexts({sycl_ctx})`; change type to `std::shared_ptr<slabkit::SlabKitContext>`; change `slabkit::SlabContextBuilder()` → `slabkit::SlabKitContextBuilder()`
- [ ] Update the static inline function signature:
  ```cpp
  inline std::shared_ptr<slabkit::SlabKitContext> make_test_context() {
      sycl::device dev{sycl::cpu_selector_v};
      return slabkit::SlabKitContextBuilder()
              .devices({dev})
              .node_size(1024)
              .slab_size_per_device(4 * 1024 * 1024)
              .build();
  }
  ```
- [ ] No other changes needed in this file — the rest is already context-backed

### 4. Update `test_context_array.cpp`
- [ ] Same `make_test_context()` helper update as `test_array.cpp` (currently identical helper in both files)
- [ ] No other changes needed

### 5. Rewrite `test_api.cpp`
This file is almost entirely registry-backed. It must be rewritten. Keep the file structure (section headers, test coverage intent) but replace all registry usage with explicit `SlabKitContext` construction.

- [ ] Remove all includes referencing `AllocatorRegistry` (keep all existing includes; just remove any `AllocatorRegistry` usage from test bodies)
- [ ] **Delete** registry-backed queue factory tests (`make_in_order_queue registry-backed`, `make_out_of_order_queue registry-backed`, `make_queue_with_props registry-backed`) — replace with context-backed versions using `ctx->make_in_order_queue(dev)` (these are already tested in `test_context.cpp`, so these can simply be removed or replaced with a brief context-backed test)
- [ ] **Delete** `make_device_array registry-backed` test and replace with context-backed equivalent using `sycl::device` overload (to exercise the new device overload):
  ```cpp
  TEST_CASE("make_device_array with sycl::device overload", "[api][make_device_array]") {
      sycl::device dev{sycl::cpu_selector_v};
      auto ctx = slabkit::SlabKitContextBuilder()
                     .devices({dev})
                     .slab_size_per_device(4 * 1024 * 1024)
                     .build();
      auto arr = slabkit::make_device_array<int, 1>(sycl::range<1>{100}, ctx, dev);
      REQUIRE(arr.size() == 100);
      REQUIRE(arr.data() != nullptr);
  }
  ```
- [ ] **Delete** `malloc_device registry-backed` test; update `malloc_device registry-free` to use a locally-built context (not `AllocatorRegistry::context()`):
  ```cpp
  TEST_CASE("malloc_device with explicit context", "[api][malloc_device]") {
      sycl::device dev{sycl::cpu_selector_v};
      auto ctx = slabkit::SlabKitContextBuilder()
                     .devices({dev})
                     .slab_size_per_device(4 * 1024 * 1024)
                     .build();
      auto ptr = slabkit::malloc_device<int>(1, ctx, 0);
      REQUIRE(ptr != nullptr);
  }
  ```
- [ ] Update `AllocationError thrown on pool exhaustion` test: remove `AllocatorRegistry::get().device_by_index(0)` and replace with local `sycl::device dev{sycl::cpu_selector_v}`; remove `.contexts({sycl::context(dev)})` from the builder call
- [ ] **Delete** `AllocatorRegistry::context()` test cases entirely
- [ ] Update `make_scratch registry-backed` and `make_scratch registry-free` tests — replace registry path with context-backed version; add a test for the new `sycl::device` overload:
  ```cpp
  TEST_CASE("make_scratch with sycl::device overload", "[api][scratch]") {
      sycl::device dev{sycl::cpu_selector_v};
      auto ctx = slabkit::SlabKitContextBuilder()
                     .devices({dev})
                     .slab_size_per_device(4 * 1024 * 1024)
                     .build();
      auto scratch = slabkit::make_scratch(ctx, dev, 64 * 1024);
      REQUIRE(scratch.capacity() >= 64 * 1024);
  }
  ```
- [ ] Update `DeviceScratchBuffer is DeviceScratchMemory` test to use context-backed scratch
- [ ] Update `DeviceScratchView construction and access` test to use context-backed scratch
- [ ] **Delete** `MirroredArray registry-backed constructor` test; replace with context-backed version:
  ```cpp
  TEST_CASE("MirroredArray explicit context constructor", "[api][mirrored_array]") {
      sycl::device dev{sycl::cpu_selector_v};
      auto ctx = slabkit::SlabKitContextBuilder()
                     .devices({dev})
                     .slab_size_per_device(4 * 1024 * 1024)
                     .build();
      auto mirrored = slabkit::MirroredArray<int, 1>(sycl::range<1>{100}, ctx, 0);
      REQUIRE(mirrored.size() == 100);
      REQUIRE(mirrored.host_data().size() == 100);
  }
  ```
- [ ] Update the top-of-file comment block to remove references to "Registry Integration Tests" and replace with a description of what the file now tests (context-backed API paths)

### 6. Update `test_transfers.cpp`
- [ ] Update `make_test_context()` helper: same change as `test_array.cpp` (remove `.contexts(...)`, change `SlabContextBuilder` → `SlabKitContextBuilder`, type to `SlabKitContext`)
- [ ] No other changes needed — all tests in this file are already context-backed

### 7. Verify
- [ ] No `AllocatorRegistry` reference anywhere in any test file
- [ ] No `SlabContext` (old name) anywhere in any test file
- [ ] No `SlabContextBuilder` (old name) anywhere in any test file
- [ ] No `.contexts(...)` call anywhere in any test file
- [ ] Every builder call uses `.devices({...})` only (no `.contexts()`)
- [ ] `sycl::device` overloads are exercised in `test_api.cpp` for `make_device_array` and `make_scratch`

---

## Scope
- **Edit:** `tests/test_context.cpp`, `tests/test_memory.cpp`, `tests/test_array.cpp`, `tests/test_context_array.cpp`, `tests/test_api.cpp`, `tests/test_transfers.cpp`
- **Read:** `include/slabkit/context.hpp` (post-01), `include/slabkit/memory.hpp` (post-02), `include/slabkit/array.hpp` (post-03), `include/slabkit/scratch.hpp` (post-04) — to confirm exact new API signatures before writing test code
- **Write:** nothing new
- **Excluded:** all `.hpp` headers, build system files, docs

---

## Patterns

```
✅ GOOD — New builder pattern (no .contexts()):
    sycl::device dev{sycl::cpu_selector_v};
    auto ctx = slabkit::SlabKitContextBuilder()
                   .devices({dev})
                   .node_size(256)
                   .slab_size_per_device(4 * 1024 * 1024)
                   .build();

✅ GOOD — sycl::device overload for make_device_array:
    auto arr = slabkit::make_device_array<float, 1>(sycl::range<1>{8}, ctx, dev);

✅ GOOD — sycl::device overload for make_scratch:
    auto scratch = slabkit::make_scratch(ctx, dev, 64 * 1024);

❌ BAD  — Old builder with .contexts():
    sycl::context sycl_ctx{dev};
    slabkit::SlabContextBuilder().contexts({sycl_ctx}).devices({dev})...

❌ BAD  — Any AllocatorRegistry usage:
    slabkit::AllocatorRegistry::get().device_by_index(0)
    slabkit::AllocatorRegistry::context()

❌ BAD  — Old type names:
    std::shared_ptr<slabkit::SlabContext>
    slabkit::SlabContextBuilder()
```

---

## Constraints
- Do not remove tests for `AllocationError` (pool exhaustion) — just update to use context-backed construction
- Do not remove `DeviceScratchView`, `DeviceScratchBuffer` type alias tests — just update to use context-backed scratch
- Do not remove `DeviceViewLike` concept static asserts — they require no changes (not API-related)
- The `make_test_context()` helper exists in three files (`test_array.cpp`, `test_context_array.cpp`, `test_transfers.cpp`); update all three consistently
- `test_context.cpp` does not define `make_test_context()` — do not add one; it builds its contexts inline per test case
- Catch2 macros (`REQUIRE`, `REQUIRE_THROWS_AS`, `REQUIRE_NOTHROW`, `SECTION`) are unchanged
- Keep `[tag]` labels on test cases consistent with their existing categories

---

*At the end of this subtask, follow the checkpoint protocol in `~/.config/opencode/protocols/checkpoint.md`.*
