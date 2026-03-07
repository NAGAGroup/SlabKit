# SlabKit v0.1 — Design Document

**Status**: In Progress  
**Date**: 2026-03-06  
**Mode**: D3 — Structured phases  

---

## 1. Problem Statement

SlabKit is a header-only C++20 library providing SYCL USM memory management for HPC applications. The original draft headers were design sketches — never compilable — and have been redesigned and substantially rewritten in this session.

---

## 2. Architecture Overview

```
User code
    │
    ├── DeviceArray<T,N>          (RAII owner, copyable via shared_ptr)
    ├── MirroredArray<T,N>        (paired device+host)
    ├── DeviceView<T,N>           (non-owning, contiguous, trivially copyable)
    └── DeviceSubView<T,N>        (non-owning, strided, trivially copyable)
            │
            └── make_device_array()
                    │
                    └── malloc_shared_device_ptr<T>()
                            │
                            └── AllocatorRegistry (Meyer's singleton, lazy per-device)
                                        │
                            shared_ptr<DeviceMemoryPool>  (one per device, lazy)
                                        │
                            DeviceSlabAllocator  (BlockAllocator, owned by pool internally)
                                        │
                            sycl::malloc_device slab (fraction of device global_mem, allocated once)
```

**Invariants:**
- No `sycl::buffer` / accessor — USM exclusively
- All USM pointers are owned by `shared_ptr` or `unique_ptr` with `DeviceUSMFromPoolDeleter`
- `DeviceArray` is copyable (shared ownership via `shared_ptr<T>`)
- Views are trivially copyable, safe to pass into SYCL kernels
- Async transfers return `sycl::event` and accept `std::vector<sycl::event> deps`
- `sycl::malloc_device` is called exactly once per device (at allocator construction) — never again
- Queues are user-only; registry and allocator internals never construct or store queues

---

## 3. Key Design Decisions

### Allocator architecture — `DeviceSlabAllocator` as `BlockAllocator`

**Decision:** `DeviceSlabAllocator` implements the foonathan `BlockAllocator` concept, not `RawAllocator`.

Rationale: `memory_pool` and `memory_stack` accept `BlockOrRawAllocator`. As a `BlockAllocator`, `DeviceSlabAllocator` only needs to implement `allocate_block()`, `deallocate_block()`, `next_block_size()` — it hands out fixed-size blocks and foonathan's pool manages node-level allocation internally. This eliminates the `SubAllocator` / `FreeBlock` / coalescing complexity from the original draft entirely.

The slab allocator is deliberately dumb: linear scan over `std::vector<bool> block_map_` for a free block, return it. All allocation sophistication lives in the pool.

`DeviceSlabAllocator` must **not** be `final` — foonathan's `memory_arena` inherits from the `BlockAllocator` via EBO.

---

### `DeviceSlabAllocator` — copyable via `shared_ptr`

`DeviceSlabAllocator` is **copyable, not movable**. This is required because foonathan's pool constructor takes the `BlockAllocator` by value, creating a copy that lives inside the pool. The registry's allocator instance (owned by the pool) and any external copies all share the same `shared_ptr<void>` slab.

Ownership chain:
- `slab_` is `shared_ptr<void>` with `SlabDeleter{shared_ptr<sycl::context>}`
- `sycl::context` is also `shared_ptr<sycl::context>` (one per platform, owned by registry)
- Copy of allocator = copy of shared_ptrs = shared slab and context ownership

---

### Registry — pools only, no allocator storage

**Decision:** `AllocatorRegistry` stores only `vector<shared_ptr<DeviceMemoryPool>>`. No separate allocator vector. The pool is the only external handle needed. The `DeviceSlabAllocator` is constructed inline during pool construction and lives inside foonathan's arena.

`pool(device_index)` returns `shared_ptr<DeviceMemoryPool>` — callers can hold a copy for lifetime safety.

---

### `sycl::context` — one per platform, no queues in registry

Registry groups devices by platform during `init()` using `get_backend() + get_info<platform::name>()` as a composite key (`.get()` is not standard). Constructs one `shared_ptr<sycl::context>` per platform via `sycl::context(vector<sycl::device>)`. No queues ever constructed or stored in registry internals.

---

### USM pointer ownership — `DeviceUSMFromPoolDeleter<T>`

**Decision:** USM pointers allocated from a pool are owned by `shared_ptr<T>` or `unique_device_ptr<T>`, both using `DeviceUSMFromPoolDeleter<T>`.

```cpp
template <typename T>
struct DeviceUSMFromPoolDeleter {
    size_t                          count;
    std::weak_ptr<DeviceMemoryPool> pool;   // weak — does not extend pool lifetime

    void operator()(T* ptr) const noexcept {
        if (auto p = pool.lock())
            p->deallocate_node(ptr, count * sizeof(T), alignof(T));
        // if expired: pool destroyed, USM freed with it — skip silently
    }
};

template <typename T>
using unique_device_ptr = std::unique_ptr<T, DeviceUSMFromPoolDeleter<T>>;
```

The deleter holds a `weak_ptr` — it does not keep the pool alive. `malloc_shared_device_ptr` captures the pool as `weak_ptr` in its lambda deleter. `malloc_unique_device_ptr` uses `DeviceUSMFromPoolDeleter` directly.

---

### `DeviceArray` — copyable via `shared_ptr`

`DeviceArray<T,N>` stores `shared_ptr<T>` (from `malloc_shared_device_ptr`). Copy/move all defaulted. This matches the ergonomics of `sycl::buffer` which SlabKit is intended to replace — no artificial move-only friction.

`MirroredArray` copy: device side shares ownership (shared_ptr), host `std::vector<T>` is deep-copied — intentional, two independent host buffers sharing one device allocation would be a footgun.

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

### `AllocatorRegistry::Config` — runtime configuration

**Decision:** Replace individual `set_*` methods with a `Config` struct accessed via a public static `config()` Meyer's singleton that always returns a mutable reference.

```cpp
struct Config {
    size_t block_size           = 64ull * 1024;  // arena block granularity (bytes)
    size_t node_size            = 1024;           // pool freelist node size (bytes)
    float  device_slab_fraction = 0.60f;          // fraction of global_mem_size for non-CPU devices
    float  cpu_slab_fraction    = 0.30f;          // fraction of global_mem_size for CPU devices
};

static Config& config();
```

`config()` is the only configuration interface. Users modify only the fields they care about. Writes after the first call to `get()` are silently ignored (config is read exactly once in `init()`).

**Slab sizing** is computed per-device in `init()` rather than stored as a fixed value:

```cpp
const size_t total_mem = dev.get_info<sycl::info::device::global_mem_size>();
const float  fraction  = dev.is_cpu() ? cfg.cpu_slab_fraction : cfg.device_slab_fraction;
const size_t computed  = static_cast<size_t>(total_mem * fraction);
device_slab_sizes_[dev_idx] = std::max(computed, 2 * cfg.block_size);
```

Rationale: SlabKit is intended to be the only USM allocation mechanism for an application. Claiming a fixed fraction of total `global_mem_size` (the only memory query available in SYCL — there is no free-memory equivalent) is the correct approach. 60% for GPU devices, 30% for CPU devices (OpenCL, OMP) which share system memory. Floor of `2 * block_size` ensures the slab always holds at least two blocks.

`pool_initial_blocks_` is removed entirely. `memory_pool`'s mandatory initial block allocation is cheap — `DeviceSlabAllocator::allocate_block()` is just a bitmap flip and pointer offset, no additional USM allocation.

**Note:** `device_slab_sizes_[dev_idx]` must be block-aligned for `DeviceSlabAllocator`'s assert (`total_size % block_size == 0`). This rounding is a known open item — see §5.

---

### Pool node size — 1024 bytes

Default `node_size = 1024` in `Config`. Rationale: typical workloads are large persistent arrays (handled by pool node rounding) and small temporaries (handled by scratch memory stack). 1024B is a reasonable mid-point; pool exhaustion throws `std::bad_alloc`.

---

### `DeviceScratchMemory` — `memory_stack<DeviceMemoryPool>`

`DeviceScratchMemory = memory::memory_stack<DeviceMemoryPool>`. Stack requests blocks from the global pool. First block is allocated eagerly in `memory_stack` constructor — `make_scratch_memory()` will trigger lazy pool+allocator construction for that device if not already done.

---

### `DeviceSubView` transfers — not yet implemented (Phase 5)

`copy_to_host` / `copy_to_device` for `DeviceSubView` are currently `static_assert(sizeof(T) == 0, "not yet implemented")` placeholders. They will not compile if instantiated, but do not block compilation of the rest of the file.

**Planned implementation (Phase 5):** `sycl::malloc_host` staging buffer + gather/scatter SYCL kernels. Event chain: `input_deps → gather_event → memcpy_event`. RAII staging wrapper with `event.wait()` in destructor before `sycl::free`.

---

### Thread safety — current state

- Registry-level init: `std::call_once(registry_init_, ...)` — safe
- Per-device pool init: `std::call_once(pool_flags_[i], ...)` + `std::mutex sycl_init_mutex_` — safe
- Per-device alloc/dealloc (`DeviceMemoryPool::allocate_node`): **not thread-safe** — documented v0.1 contract: single-threaded allocation per device. Per-device mutex deferred to v0.2.
- `AllocatorRegistry::config()`: not thread-safe vs concurrent init — must be called before first `AllocatorRegistry::get()`.

---

## 4. Current File State

### `include/slabkit/memory.hpp` — DONE (387 lines)

Contains:
- `DeviceSlabAllocator` — `BlockAllocator`, copyable, owns slab via `shared_ptr<void>` with `SlabDeleter`
- `DeviceMemoryPool` = `memory::memory_pool<memory::array_pool, DeviceSlabAllocator>`
- `DeviceScratchMemory` = `memory::memory_stack<DeviceMemoryPool>`
- `DeviceUSMFromPoolDeleter<T>` + `unique_device_ptr<T>` alias
- `AllocatorRegistry` — singleton, lazy per-device pool construction, one `shared_ptr<context>` per platform
  - `Config` struct + `config()` public static accessor
  - Per-device slab sizing via `global_mem_size * fraction` computed at `init()`
- Free functions: `malloc_shared_device_ptr<T>`, `malloc_unique_device_ptr<T>`, `make_scratch_memory`

### `include/slabkit/array.hpp` — DONE (340 lines)

Contains:
- `slabkit::detail::row_major_offset` + `strided_offset` fold-expression helpers
- `DeviceViewLike` concept
- `DeviceView<T,N>` — contiguous, fold-expression operators
- `DeviceSubView<T,N>` — strided, fold-expression operators
- `DeviceArray<T,N>` — copyable via `shared_ptr<T>`, `view()` + `subview()`
- `make_device_array<T,N>` — uses `malloc_shared_device_ptr`
- Transfer functions: `copy_to_host`, `copy_to_device`, `copy` for `DeviceArray` and `DeviceView`
- `DeviceSubView` transfers: `static_assert` stubs (Phase 5)
- `sync_to_host` blocking convenience
- `MirroredArray<T,N>` — paired device+host, device side shared, host side deep-copy on copy

### `include/slabkit/device_allocator.hpp` — DELETED

No longer needed. `array.hpp` includes `slabkit/memory.hpp` directly.

### `tests/test_memory.cpp` / `tests/test_array.cpp` — stubs

Includes uncommented, all test cases are `SUCCEED("not yet implemented")`. Phase 6.

---

## 5. Remaining Work

### Phase 4.5 — Block-align computed slab sizes (`memory.hpp`)

`device_slab_sizes_[dev_idx]` is currently `max(total_mem * fraction, 2 * block_size)` with no guarantee of block alignment. `DeviceSlabAllocator` asserts `total_size % block_size == 0`. Fix:

```cpp
const size_t raw = std::max(computed, 2 * cfg.block_size);
device_slab_sizes_[dev_idx] = (raw / cfg.block_size) * cfg.block_size;
```

### Phase 5 — DeviceSubView transfers (`array.hpp`)

Implement `copy_to_host` and `copy_to_device` for `DeviceSubView<T,N>`:

1. Allocate `sycl::malloc_host<T>(src.size(), q)` staging buffer — RAII wrapper with `event.wait()` + `sycl::free` in destructor
2. **Gather kernel** (`copy_to_host`):
   - `cgh.parallel_for(src.get_range(), [=](sycl::id<N> idx) { staging[row_major_offset(idx, extents)] = src[idx]; })`
   - Event chain: `input_deps → gather_event → memcpy(staging→dst, {gather_event})`
3. **Scatter kernel** (`copy_to_device`):
   - Symmetric: `memcpy(src→staging, deps) → scatter_event` where kernel writes `dst[idx] = staging[row_major_offset(idx, extents)]`
4. `strides_` in `DeviceSubView` is `sycl::range<N>` — device-copyable, can be captured by value in kernel lambda

### Phase 6 — Tests (`test_memory.cpp`, `test_array.cpp`)

`test_memory.cpp`:
- `AllocatorRegistry::get()` returns singleton
- Lazy init: pool is null before first access, constructed after
- `malloc_shared_device_ptr` / `malloc_unique_device_ptr` return non-null
- `make_scratch_memory` constructs without throwing
- `AllocatorRegistry::config()` modifications affect subsequent pool construction (when called before `get()`)

`test_array.cpp`:
- `DeviceView` satisfies `DeviceViewLike` concept
- `DeviceView::operator()` / `operator[]` correct row-major offsets (1D, 2D)
- `DeviceSubView::operator()` / `operator[]` correct strided offsets
- `DeviceArray` copyable — copy shares device memory
- `make_device_array` allocates non-null
- `DeviceArray::subview` bounds check fires on OOB
- `MirroredArray` round-trip: `sync_to_device` → kernel → `sync_to_host` → correct values
- `copy_to_host` / `copy_to_device` return valid `sycl::event`

### Final gate — single build + test run

```
pixi run configure
pixi run build
pixi run test
```

Known issues to watch at build gate:
- `sycl::context(vector<sycl::device>)` constructor — AdaptiveCpp signature may differ
- `sycl::platform::get_backend()` — verify AdaptiveCpp exposes this on `sycl::platform`
- `DeviceScratchMemory{pool}` — `memory_stack<DeviceMemoryPool>` takes pool by reference; verify foonathan doesn't copy it
- `row_major_offset_impl` right-fold evaluation order — verify correct strides for 2D/3D cases

---

## 6. Out of Scope for v0.1

- Per-device allocation thread safety (v0.2 — add per-device `std::mutex` in `DeviceMemoryPool`)
- Device reset / registry teardown
- Allocation statistics API
- `ext_oneapi_memcpy2d` fast path for 2D SubView transfers (Intel-only extension)
- N≥4 view support (blocked by SYCL spec `sycl::id<N>` / `sycl::range<N>` cap at N=3)
- Custom pool injection into `make_device_array` (architecture supports it, API not exposed yet)
- Direct `DeviceSlabAllocator` / `DeviceMemoryPool` instantiation by users (internals are all accessible, public API not documented/stabilised yet)
