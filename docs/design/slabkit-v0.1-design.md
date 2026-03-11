# SlabKit v0.1 — Design Document

**Status**: Complete  
**Date**: 2026-03-06  
**Last Updated**: 2026-03-10 (post-refactor)
**Mode**: D3 — Structured phases  

---

## 1. Problem Statement

SlabKit is a header-only C++20 library providing SYCL USM memory management for HPC applications. The original draft headers were design sketches — never compilable — and have been redesigned and substantially rewritten in this session.

---

## 2. Architecture Overview

```
User code
    │
    ├── DeviceArray<T,N>          (RAII owner, move-only via unique_ptr)
    ├── MirroredArray<T,N>        (paired device+host)
    ├── DeviceView<T,N>           (non-owning, contiguous, trivially copyable)
    └── DeviceSubView<T,N>        (non-owning, strided, trivially copyable)
            │
            └── make_device_array()
                    │
                    └── malloc_device()
                            │
                            └── Context (entry point, manages pools & contexts)
                                        │
                    ContextBuilder (groups devices by platform, auto-creates sycl::context)
                                        │
                        shared_ptr<DeviceMemoryPool>  (one per device, lazy)
                                        │
                        DeviceSlabAllocator  (BlockAllocator, owned by pool internally)
                                        │
                        sycl::malloc_device slab (fraction of device global_mem, allocated once)
```

**Invariants:**
- No `sycl::buffer` / accessor — USM exclusively
- All USM pointers are owned by `unique_ptr<T, DeviceUSMPoolDeleter>` with weak_ptr<Context> lifetime anchor
- `DeviceArray` is move-only (backed by `unique_ptr` with custom deleter)
- Views are trivially copyable, safe to pass into SYCL kernels
- Async transfers return `sycl::event` and accept `std::vector<sycl::event> deps`
- `sycl::malloc_device` is called exactly once per device (at allocator construction) — never again
- Queues are user-only; Context and allocator internals never construct or store queues
- `Context` is always heap-allocated via `shared_ptr` for `enable_shared_from_this` to work in lazy pool init

---

## 3. Key Design Decisions

### Allocator architecture — `DeviceSlabAllocator` as `BlockAllocator`

**Decision:** `DeviceSlabAllocator` implements the foonathan `BlockAllocator` concept, not `RawAllocator`.

Rationale: `memory_pool` and `memory_stack` accept `BlockOrRawAllocator`. As a `BlockAllocator`, `DeviceSlabAllocator` only needs to implement `allocate_block()`, `deallocate_block()`, `next_block_size()` — it hands out fixed-size blocks and foonathan's pool manages node-level allocation internally. This eliminates the `SubAllocator` / `FreeBlock` / coalescing complexity from the original draft entirely.

The slab allocator is deliberately dumb: linear scan over `std::vector<bool> block_map_` for a free block, return it. All allocation sophistication lives in the pool.

`DeviceSlabAllocator` must **not** be `final` — foonathan's `memory_arena` inherits from the `BlockAllocator` via EBO.

---

### `DeviceSlabAllocator` — copyable via `shared_ptr`

`DeviceSlabAllocator` is **copyable, not movable**. This is required because foonathan's pool constructor takes the `BlockAllocator` by value, creating a copy that lives inside the pool. The allocator instance owned by the pool and any external copies all share the same `shared_ptr<void>` slab.

Ownership chain:
- `slab_` is `shared_ptr<void>` with `SlabDeleter{shared_ptr<sycl::context>}`
- `sycl::context` is also `shared_ptr<sycl::context>` (one per platform, owned by `Context`)
- Copy of allocator = copy of shared_ptrs = shared slab and context ownership

---

### ContextBuilder — auto-context creation per platform group

**Decision:** `ContextBuilder` accepts only a `devices(vector<sycl::device>)` input — no explicit `sycl::context` construction required from the user. The builder automatically groups devices by `(backend, platform_name)` composite key and creates one `sycl::context` per platform group.

`build()` returns `shared_ptr<Context>` with:
- `Context::platform_contexts` — auto-populated, one `sycl::context` per platform group
- `Context::device_to_platform` — mapping from device index to platform context index
- All pools lazy-initialized on first access to `Context::pool_at(device_index)`

Rationale: No global state or singleton. User creates and owns the `Context` via `shared_ptr`. `enable_shared_from_this<Context>` enables safe lazy pool initialization with weak_ptr lifetime anchors. No queues are created or stored by `ContextBuilder` or `Context` — queues are entirely user-side.

```cpp
auto ctx = ContextBuilder()
    .devices(device_list)
    .node_size(1024)
    .slab_size_per_device(128 * 1024 * 1024)
    .build();

auto arr = make_device_array<float, 2>(range, ctx, device_index);
```

---

### USM pointer ownership — `DeviceUSMPoolDeleter`

**Decision:** USM pointers allocated from a `Context` pool are owned by `unique_ptr<T, DeviceUSMPoolDeleter>` with a weak_ptr<Context> lifetime anchor.

```cpp
struct DeviceUSMPoolDeleter {
    std::weak_ptr<Context> ctx;
    size_t device_index;
    size_t node_count;

    void operator()(void *ptr) const noexcept {
        if (auto c = ctx.lock())
            c->pool_at(device_index).deallocate_array(ptr, node_count);
        // if expired: Context destroyed, USM slab already freed — no-op is correct
    }
};

template <typename T>
using unique_device_ptr = std::unique_ptr<T, DeviceUSMPoolDeleter>;
```

The deleter holds a `weak_ptr<Context>` — it does not keep the context alive. If the context is destroyed before the pointer is deleted, deallocation is skipped silently (the slab was already freed when the context destroyed). This enables safe cleanup without worrying about destructor ordering.

**Rationale:** `weak_ptr` breaks the cycle that would arise if the context held strong `shared_ptr` to pools which held strong `shared_ptr` back to context. The weak anchor is sufficient because contexts outlive their allocations in typical usage — and if they don't, silent skip is correct (memory will be freed with the context's slab anyway).

---

### `DeviceArray` — move-only via `unique_ptr`

`DeviceArray<T,N>` stores `unique_ptr<T, DeviceUSMPoolDeleter>` backing and is **move-only** (copy constructor and assignment deleted). This is a departure from the copyable-via-shared_ptr ergonomics of `sycl::buffer`, but reflects the actual SYCL USM model: each unique allocation is owned by a single logical entity at a time.

`MirroredArray` is also move-only (because its device side is `DeviceArray`). Host `std::vector<T>` is cloned on `clone()` call, not on copy — intentional, two independent host buffers sharing one device allocation would be a footgun.

---

### View operators — fold-expression helpers in `slabkit::detail`

**Decision:** Replace hardcoded N≤3 `if constexpr` chains with fold-expression helpers.

```cpp
namespace slabkit::detail {

// row_major_offset(id, range): processes dims right-to-left via unary right fold
// For N=2: id[0]*range[1] + id[1]
template <int N, std::size_t... Is>
std::size_t row_major_offset_impl(sycl::id<N> id, sycl::range<N> range,
                                  std::index_sequence<Is...>) noexcept {
    std::size_t result = 0, stride = 1;
    ((void)(result += id[N-1-Is] * stride, stride *= range[N-1-Is]), ...);
    return result;
}
template <int N>
std::size_t row_major_offset(sycl::id<N> id, sycl::range<N> range) noexcept;

// strided_offset(id, strides): dot product
template <int N, std::size_t... Is>
std::size_t strided_offset_impl(sycl::id<N> id, sycl::range<N> strides,
                                std::index_sequence<Is...>) noexcept {
    return ((id[Is] * strides[Is]) + ...);
}
template <int N>
std::size_t strided_offset(sycl::id<N> id, sycl::range<N> strides) noexcept;

} // namespace slabkit::detail
```

**SYCL N constraint:** `sycl::id<N>` and `sycl::range<N>` are only defined for N∈{1,2,3} per SYCL 2020 §4.9.1. The fold-expression generalisation is architecturally correct for arbitrary N, but the SYCL type system caps practical use at N=3. If N≥4 is ever needed, both types must be replaced with `std::array<size_t,N>`.

**Evaluation order note:** `row_major_offset_impl` is a unary right fold over `,` — rightmost pack element evaluated first, giving correct right-to-left dimension processing.

---

### Pool node size — 1024 bytes

Default `node_size = 1024` in `Config`. Rationale: typical workloads are large persistent arrays (handled by pool node rounding) and small temporaries (handled by scratch memory stack). 1024B is a reasonable mid-point; pool exhaustion throws `std::bad_alloc`.

---

### `DeviceScratchMemory` — `memory_stack<DeviceMemoryPool>`

`DeviceScratchMemory = memory::memory_stack<DeviceMemoryPool>`. Stack requests blocks from the global pool. First block is allocated eagerly in `memory_stack` constructor — `make_scratch_memory()` will trigger lazy pool+allocator construction for that device if not already done.

---

### `DeviceSubView` transfers — implemented with staging buffers

Strided transfers (`copy_to_host` / `copy_to_device`) use device USM staging buffers (allocated from `DeviceScratchMemory`), not host staging. Pipeline:

**Gather (`copy_to_host`, strided device → host):**
1. Allocate contiguous staging on device from scratch memory
2. `parallel_for` kernel: `staging[linear_i] = src[strided_id]` (gather)
3. `memcpy(staging → host_ptr, {gather_event})`

**Scatter (`copy_to_device`, host → strided device):**
1. Allocate contiguous staging on device from scratch memory
2. `memcpy(host_ptr → staging, deps)`
3. `parallel_for` kernel: `dst[strided_id] = staging[linear_i]` (scatter)

Caller must keep `DeviceScratchMemory` alive until the returned `sycl::event` completes.

---

### Thread safety — current state

- Per-device pool init: double-checked locking in `Context::pool_at()` using `std::mutex pool_mutex_` — safe
- Per-device alloc/dealloc (`DeviceMemoryPool::allocate_array` / `deallocate_array`): **not thread-safe** — documented v0.1 contract: single-threaded allocation per device. Per-device mutex deferred to v0.2.
- `Context` construction is not thread-safe (user responsibility to construct once and share) — by design, once constructed, `Context` is immutable (no setters after `build()`).

---

## 4. Current File State

### `include/slabkit/memory.hpp` — DONE (83 lines)

Contains:
- `DeviceUSMPoolDeleter` — custom deleter for USM pointers allocated from Context pools
- `unique_device_ptr<T>` alias = `std::unique_ptr<T, DeviceUSMPoolDeleter>`
- Free functions: `malloc_device<T>(count, ctx, device_index)` and `malloc_device<T>(count, ctx, dev)` overloads

### `include/slabkit/array.hpp` — DONE (610 lines)

Contains:
- `slabkit::detail::row_major_offset` + `strided_offset` fold-expression helpers
- `DeviceViewLike` concept
- `DeviceView<T,N>` — contiguous, fold-expression operators
- `DeviceSubView<T,N>` — strided, fold-expression operators
- `DeviceArray<T,N>` — move-only via `unique_ptr<T, DeviceUSMPoolDeleter>`, `view()` + `subview()`
- `make_device_array<T,N>(range, ctx, device_index)` and `make_device_array<T,N>(range, ctx, dev)` overloads
- Transfer functions: `copy_to_host`, `copy_to_device`, `copy` for `DeviceArray`, `DeviceView`, and `DeviceSubView`
- `sync_to_host` blocking convenience wrappers
- `MirroredArray<T,N>` — paired device+host, device side move-only, host side deep-copy on `clone()`

### `include/slabkit/context.hpp` — DONE (255 lines)

Contains:
- `Context` — manages SYCL platform contexts, devices, and device-specific memory pools
- `ContextBuilder` — builder pattern, accepts `devices()`, auto-creates one `sycl::context` per platform group

### `include/slabkit/scratch.hpp` — DONE

Contains:
- `DeviceScratchMemory` / `DeviceScratchBuffer` = `memory::memory_stack<DeviceMemoryPool>`
- `DeviceScratchView<T,N>` — non-owning scratch view
- Free functions: `make_scratch(ctx, device_index, size)` and `make_scratch(ctx, dev, size)` overloads
- No registry-backed overloads — all factory functions require an explicit `Context`

### `tests/` — DONE (6 test files, 64 tests, 64/64 passing)

- `test_context.cpp` — `Context`, `ContextBuilder`, `pool_for`, `device_index`, queue creation
- `test_array.cpp` — `DeviceArray`, `DeviceView`, `DeviceSubView`, `make_device_array` (context + device overloads)
- `test_memory.cpp` — `malloc_device` (registry-free), `AllocationError`
- `test_scratch.cpp` — `make_scratch` (context, device_index, sycl::device overloads)
- `test_api.cpp` — `DeviceViewLike` concept checks
- `test_transfers.cpp` — `copy_to_device`, `copy_to_host`, sync, `MirroredArray`

All test files are context-backed; no registry tests remain.

---

## 5. Remaining Work

All planned phases for v0.1 are complete. `AllocatorRegistry` removed in session `remove-allocator-registry-singleton` (2026-03-10).

Potential v0.2 items (tracked separately):
- Per-device allocation thread safety — add per-device `std::mutex` in `DeviceMemoryPool`
- `ext_oneapi_memcpy2d` fast path for 2D SubView transfers (Intel-only extension)
- Custom pool injection into `make_device_array`

---

## 6. Out of Scope for v0.1

- Per-device allocation thread safety (v0.2 — add per-device `std::mutex` in `DeviceMemoryPool`)
- Global default context / convenience singleton — by design, users must construct `Context` explicitly
- Allocation statistics API
- `ext_oneapi_memcpy2d` fast path for 2D SubView transfers (Intel-only extension)
- N≥4 view support (blocked by SYCL spec `sycl::id<N>` / `sycl::range<N>` cap at N=3)
- Custom pool injection into `make_device_array` (architecture supports it, API not exposed yet)
- Direct `DeviceSlabAllocator` / `DeviceMemoryPool` instantiation by users (internals are all accessible, public API not documented/stabilised yet)
