# Subtask 03 — Refactor `array.hpp`

## Delegation
**Agent:** @CodeWriter
**Model:** standard (claude-sonnet) — multiple interacting systems: updating SlabContext refs, removing registry-backed overloads, adding sycl::device overloads, and updating MirroredArray

---

## Objective

Refactor `include/slabkit/array.hpp` to:

1. **Replace all `SlabContext` references** with `SlabKitContext` throughout the file.
2. **Remove the registry-backed `make_device_array<T,N>(range, dev)` overload** (the one that calls `AllocatorRegistry::context()`).
3. **Add a `sycl::device` overload** to `make_device_array` that accepts `(range, ctx, dev)` and resolves `device_index` internally.
4. **Remove the registry-backed `MirroredArray` constructor** `MirroredArray(range, dev)` (the one that calls `AllocatorRegistry::context()`).
5. Update the `DeviceArray::from_pool` static factory and all internal uses of `SlabContext`.
6. Update `MirroredArray`'s `ctx_` member type.

The view and subview types (`DeviceView`, `DeviceSubView`), transfer utilities, and all kernel-safe types are **unchanged**.

---

## Todolist

### 1. Update SlabContext → SlabKitContext references
- [ ] `DeviceArray::from_pool` signature: `std::shared_ptr<SlabContext> ctx` → `std::shared_ptr<SlabKitContext> ctx`
- [ ] `DeviceArray::from_pool` body: `DeviceUSMPoolDeleter{std::weak_ptr<SlabContext>(ctx), ...}` → `DeviceUSMPoolDeleter{std::weak_ptr<SlabKitContext>(ctx), ...}`
- [ ] `make_device_array(range, ctx, device_index)` signature: `std::shared_ptr<SlabContext> ctx` → `std::shared_ptr<SlabKitContext> ctx`
- [ ] `MirroredArray` private member: `std::shared_ptr<SlabContext> ctx_` → `std::shared_ptr<SlabKitContext> ctx_`
- [ ] `MirroredArray` registry-free constructor: `std::shared_ptr<SlabContext> ctx` → `std::shared_ptr<SlabKitContext> ctx`
- [ ] All comments and doc-comments referencing `SlabContext` → `SlabKitContext`

### 2. Remove registry-backed make_device_array overload
- [ ] Delete the `make_device_array<T,N>(sycl::range<N> range, const sycl::device& dev)` template function that calls `AllocatorRegistry::context()`
- [ ] Delete its doc comment block (`/// Registry-backed factory: ...`)

### 3. Add sycl::device overload to make_device_array
- [ ] Add immediately after the existing `make_device_array(range, ctx, device_index)`:
    ```cpp
    /// Device overload: resolves device_index automatically via ctx->device_index(dev).
    template <typename T, int N>
    DeviceArray<T, N> make_device_array(sycl::range<N> range,
                                         std::shared_ptr<SlabKitContext> ctx,
                                         const sycl::device& dev) {
        return DeviceArray<T, N>::from_pool(range, ctx, ctx->device_index(dev));
    }
    ```

### 4. Remove registry-backed MirroredArray constructor
- [ ] Delete the `MirroredArray(sycl::range<N> range, const sycl::device& dev)` constructor (the one that calls `AllocatorRegistry::context()`)
- [ ] Delete its doc comment block (`/// Registry-backed constructor: ...`)

### 5. Remove AllocatorRegistry include dependency
- [ ] The `#include "slabkit/memory.hpp"` include is still needed (for `unique_device_ptr`, `DeviceUSMPoolDeleter`) — **keep it**
- [ ] Remove any comment that references `AllocatorRegistry` if present
- [ ] Ensure no `AllocatorRegistry` symbol is referenced anywhere in the file

### 6. Verify
- [ ] No `SlabContext` (old name) anywhere in the file
- [ ] No `AllocatorRegistry` anywhere in the file
- [ ] Two `make_device_array` overloads remain: `(range, ctx, device_index)` and `(range, ctx, dev)`
- [ ] One `MirroredArray` constructor: `(range, ctx, device_index)` — no `(range, dev)` overload
- [ ] `DeviceArray::from_pool`, `DeviceView`, `DeviceSubView`, `MirroredArray::clone()`, all transfer utilities unchanged

---

## Scope
- **Edit:** `include/slabkit/array.hpp`
- **Read:** `include/slabkit/memory.hpp` (post-subtask-02, to confirm `SlabKitContext` and `DeviceUSMPoolDeleter` types)
- **Write:** nothing new
- **Excluded:** `context.hpp`, `scratch.hpp`, `detail/device_pool.hpp`, `detail/device_stack.hpp`, `error.hpp`, all test files

---

## Patterns

```
✅ GOOD — Updated from_pool using SlabKitContext:
    static DeviceArray from_pool(sycl::range<N> range,
                                  std::shared_ptr<SlabKitContext> ctx,
                                  size_t device_index) {
        auto& pool = ctx->pool_at(device_index);
        ...
        DeviceUSMPoolDeleter del{std::weak_ptr<SlabKitContext>(ctx), device_index, node_count};
        ...
    }

✅ GOOD — Two make_device_array overloads:
    template <typename T, int N>
    DeviceArray<T, N> make_device_array(sycl::range<N> range,
                                         std::shared_ptr<SlabKitContext> ctx,
                                         size_t device_index) { ... }

    template <typename T, int N>
    DeviceArray<T, N> make_device_array(sycl::range<N> range,
                                         std::shared_ptr<SlabKitContext> ctx,
                                         const sycl::device& dev) {
        return DeviceArray<T, N>::from_pool(range, ctx, ctx->device_index(dev));
    }

✅ GOOD — MirroredArray single registry-free ctor:
    MirroredArray(sycl::range<N> range,
                  std::shared_ptr<SlabKitContext> ctx,
                  size_t device_index)
        : ctx_(ctx), device_idx_(device_index),
          device_(DeviceArray<T, N>::from_pool(range, ctx, device_index)),
          host_(range.size()) {}

❌ BAD  — Registry-backed make_device_array:
    template <typename T, int N>
    DeviceArray<T, N> make_device_array(sycl::range<N> range, const sycl::device& dev) {
        auto ctx = AllocatorRegistry::context();  // REMOVE
        ...
    }

❌ BAD  — Registry-backed MirroredArray ctor:
    MirroredArray(sycl::range<N> range, const sycl::device& dev)
        : MirroredArray(range, AllocatorRegistry::context(), ...)  // REMOVE
    {}
```

---

## Constraints
- `DeviceView<T,N>` and `DeviceSubView<T,N>` — no changes whatsoever; they are kernel-safe and trivially copyable
- All transfer utility functions (`copy_to_host`, `copy_to_device`, `copy`, `sync_to_host`) — no changes
- `MirroredArray::clone()` still uses `ctx_` (the `shared_ptr<SlabKitContext>` member) — this must compile after the rename
- `DeviceArray` is still move-only; copy ctor and assignment remain deleted
- `#include "slabkit/memory.hpp"` must remain — provides `DeviceUSMPoolDeleter`, `unique_device_ptr`
- `#include "slabkit/context.hpp"` must remain — provides `SlabKitContext`

---

*At the end of this subtask, follow the checkpoint protocol in `~/.config/opencode/protocols/checkpoint.md`.*
