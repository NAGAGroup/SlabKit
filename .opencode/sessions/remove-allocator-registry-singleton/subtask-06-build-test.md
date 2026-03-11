# Subtask 06 — Build & Test

## Delegation
- **Agent:** HeadWrench (direct execution — not delegated to any subagent)
- **Model tier:** n/a
- **Reason:** Build and test steps are always HeadWrench-owned per session protocol. HeadWrench runs the commands directly, reads errors, and drives the debug loop.

---

## Objective

Run the full build and test suite after all code changes (subtasks 01–05) are complete. Drive the build-test-debug loop until `pixi run test dev` passes with zero failures. The circuit breaker is 3 consecutive failures.

---

## Todolist

### 1. Configure
- [ ] Run `pixi run configure dev`
- [ ] If it fails: read CMake output, delegate diagnosis to @ContextScout, fix, retry

### 2. Build
- [ ] Run `pixi run build dev`
- [ ] If it fails: read compiler errors, diagnose, form a hypothesis (write to `notes/`), delegate fix to @CodeWriter, rebuild
- [ ] Track consecutive failure count — stop at 3 and escalate to user

### 3. Test
- [ ] Run `pixi run test dev`
- [ ] If any tests fail: read test output, diagnose, form a hypothesis (write to `notes/`), delegate fix to @CodeWriter, rebuild and retest
- [ ] Track consecutive failure count — stop at 3 and escalate to user

### 4. Final verification
- [ ] All tests pass with zero failures
- [ ] Zero warnings treated as errors (if `-Werror` is active — check CMake flags)
- [ ] Commit all changes: `git add -A && git commit -m "refactor: remove AllocatorRegistry singleton, add SlabKitContext and sycl::device overloads"`

---

## Scope
- **Run:** `pixi run configure dev`, `pixi run build dev`, `pixi run test dev`
- **Read:** compiler errors, test output, `notes/` for prior hypotheses
- **Edit (via CodeWriter delegation):** any `.hpp` or test `.cpp` file that has a build or test failure
- **Excluded:** build system files (`CMakeLists.txt`, `pixi.toml`, `CMakePresets.json`) — do not modify unless a build failure is definitively caused by them

---

## Patterns

```
✅ GOOD — Debug loop: read error → ContextScout for location → write hypothesis note → CodeWriter fix → rebuild
✅ GOOD — One fix per failure: do not batch multiple hypotheses; verify each fix before moving on
✅ GOOD — Circuit breaker: if 3 consecutive build/test failures with no progress, stop and surface to user

❌ BAD  — Delegating build/test execution to CodeWriter or any subagent
❌ BAD  — Skipping the configure step if it was already run (always re-run after source changes)
❌ BAD  — Amending the commit if tests fail after it — create a new fix commit instead
```

---

## Constraints
- Build commands: `pixi run configure dev`, `pixi run build dev`, `pixi run test dev`
- Git: conventional commit format; work directly on `main`
- Circuit breaker threshold: 3 consecutive failures
- Do not modify `detail/device_pool.hpp` or `detail/device_stack.hpp` — off-limits for this session
- Do not modify `error.hpp` — off-limits for this session
- Commit only after all tests pass

---

## Known Build Risks (from prior session notes)

The following are known areas where the build may fail on first attempt:

1. **`sycl::context(vector<sycl::device>)` constructor** — AdaptiveCpp may have a different signature than the standard. Check the AdaptiveCpp headers if this constructor fails to resolve.
2. **`sycl::platform::get_backend()`** — Verify AdaptiveCpp exposes `get_backend()` on `sycl::platform` (not just `sycl::device`). If missing, the platform-grouping key in `SlabKitContextBuilder::build()` may need to use only `get_info<platform::name>()`.
3. **`sycl::device` overloads** — If `pool_for(dev)` is not yet on `SlabKitContext`, `make_scratch(ctx, dev, size)` and `make_device_array(range, ctx, dev)` will fail. Verify the context header exposes `pool_for(const sycl::device&)`.
4. **Block-alignment of slab sizes** — `DeviceSlabAllocator` asserts `total_size % block_size == 0`. Verify `SlabKitContextBuilder` rounds `slab_size_per_device` to a multiple of `block_size` (if it doesn't, add the rounding).
5. **`DeviceUSMPoolDeleter` type** — After subtask-02 renames the deleter, verify `malloc_device` return type uses the correct new deleter name throughout.

---

*At the end of this subtask, follow the checkpoint protocol in `~/.config/opencode/protocols/checkpoint.md`.*
