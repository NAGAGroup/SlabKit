# SlabKit v0.1 — API Surface Specification

**Status:** Draft for review  
**Date:** 2026-03-15  
**Namespace:** `slabkit`  
**Include:** `#include "slabkit/slabkit.hpp"`  
**Language standard:** C++20  
**SYCL standard:** SYCL 2020

---

## Table of Contents

1. [Overview](#1-overview)
2. [Public Types](#2-public-types)
3. [Free Functions](#3-free-functions)
4. [Concepts](#4-concepts)
5. [Design Decisions](#5-design-decisions)
6. [Thread-Safety Model](#6-thread-safety-model)
7. [Migration Notes](#7-migration-notes)
8. [Known Limitations](#8-known-limitations)
9. [Future Additions](#9-future-additions)

---

## 1. Overview

SlabKit is a header-only C++20 / SYCL 2020 library for managing device USM (Unified Shared Memory) allocations on one or more SYCL devices. It provides:

- A fixed-node-size slab memory pool (`DeviceMemoryPool`, internal) per device, managed through an explicit `Context` object
- RAII owning types (`DeviceArray`, `MirroredArray`) backed by pool allocations
- Non-owning, trivially-copyable view types (`DeviceView`, `DeviceSubView`) safe for capture in SYCL kernels
- A four-tier family of data transfer functions covering contiguous and strided layouts, host↔device and device↔device, with async and blocking variants
- A bump-allocator scratch buffer (`DeviceScratchMemory`) for temporary device working memory within kernels

### Stability Annotations

Every symbol in this document carries one of three stability labels:

| Label | Meaning |
|-------|---------|
| `[STABLE]` | API is locked; no breaking changes before v2.0 |
| `[UNSTABLE]` | API may change; do not depend on it across releases |
| `[DEPRECATED]` | Scheduled for removal; migration path documented below |

---

## 2. Public Types

### 2.1 `AllocationError` `[STABLE]`

**Header:** `slabkit/error.hpp`

Exception thrown when a pool or scratch allocation fails. Derives from `std::exception`.

```cpp
class AllocationError : public std::exception {
public:
    enum class Reason { PoolExhausted, Fragmented, ScratchFull };

    explicit AllocationError(Reason r, const char* msg = "");

    Reason      reason() const noexcept;
    const char* what()   const noexcept override;
};
```

#### `AllocationError::Reason` `[STABLE]`

| Enumerator | When thrown |
|-----------|------------|
| `PoolExhausted` | Pool has no free slots; all memory is allocated |
| `Fragmented` | Enough total free slots exist but no contiguous run is available for the requested size |
| `ScratchFull` | Scratch bump pointer would exceed capacity |

#### Notes

- `reason()` and `what()` are both `noexcept`.
- `what()` returns a human-readable message combining the reason code and any message string provided at construction.
- Can be caught as `std::exception`.

---

### 2.2 `Context` `[STABLE]`

**Header:** `slabkit/context.hpp`

Manages one or more SYCL devices together with their associated memory pools and SYCL context objects. Must always be heap-allocated and held via `std::shared_ptr<Context>` — use `ContextBuilder::build()` to construct.

```cpp
struct Context : std::enable_shared_from_this<Context> {

    // Public data members
    std::vector<sycl::context>                             platform_contexts;
    std::vector<sycl::device>                              devices;
    std::vector<size_t>                                    device_to_platform;
    std::vector<size_t>                                    device_slab_sizes;
    mutable std::vector<std::unique_ptr<DeviceMemoryPool>> pools;
    size_t             node_size_;
    mutable std::mutex pool_mutex_;

    // Non-copyable
    Context(const Context&)            = delete;
    Context& operator=(const Context&) = delete;

    // Move semantics — [DECISION PENDING]
    // Currently: move constructor is enabled (noexcept), move assignment is deleted.
    // This asymmetry is incorrect: std::mutex is not movable, making Context
    // non-movable in practice. Resolution: delete both move constructor and move
    // assignment to make the non-movable contract explicit. See ADR-1 and ADR-2.
    Context(Context&&) noexcept;
    Context& operator=(Context&&) = delete;

    // Member functions
    size_t device_index(const sycl::device& dev) const;
    DeviceMemoryPool&       pool_at(size_t idx);
    const DeviceMemoryPool& pool_at(size_t idx) const;
    DeviceMemoryPool&       pool_for(const sycl::device& dev);
    const DeviceMemoryPool& pool_for(const sycl::device& dev) const;
    const sycl::context& context_for(size_t device_index_val) const;
    const sycl::context& context_for(const sycl::device& dev) const;
    sycl::queue make_in_order_queue(const sycl::device& dev) const;
    sycl::queue make_out_of_order_queue(const sycl::device& dev) const;
    sycl::queue make_queue_with_props(const sycl::device& dev,
                                      const sycl::property_list& props) const;
};
```

#### Public Data Members

| Member | Type | Description |
|--------|------|-------------|
| `platform_contexts` | `vector<sycl::context>` | One SYCL context per platform group; auto-created by `ContextBuilder` |
| `devices` | `vector<sycl::device>` | All devices managed by this `Context`, in construction order |
| `device_to_platform` | `vector<size_t>` | Maps `devices[i]` → index into `platform_contexts` |
| `device_slab_sizes` | `vector<size_t>` | Per-device pool slab size in bytes |
| `pools` | `vector<unique_ptr<DeviceMemoryPool>>` | Lazy-initialized; one per device; `mutable` to allow const `pool_at()` |
| `node_size_` | `size_t` | Pool slot granularity in bytes (default 1024) |
| `pool_mutex_` | `mutable mutex` | Guards lazy pool initialization |

#### Member Functions

| Function | Throws | Description |
|----------|--------|-------------|
| `device_index(dev)` | `std::invalid_argument` if device not found | Returns index of `dev` in `devices` |
| `pool_at(idx)` | `std::out_of_range` if idx ≥ devices.size() | Returns reference to pool for device `idx`; initializes lazily on first call |
| `pool_for(dev)` | `std::invalid_argument` | Looks up device index then delegates to `pool_at()` |
| `context_for(idx)` | `std::out_of_range` | Returns the SYCL context for device at index `idx` |
| `context_for(dev)` | `std::invalid_argument` | Looks up device index then delegates |
| `make_in_order_queue(dev)` | — | Creates a SYCL in-order queue for `dev` using its platform context |
| `make_out_of_order_queue(dev)` | — | Creates a SYCL out-of-order queue for `dev` |
| `make_queue_with_props(dev, props)` | — | Creates a SYCL queue with the given property list |

---

### 2.3 `ContextBuilder` `[STABLE]`

**Header:** `slabkit/context.hpp`

Builder for constructing a `std::shared_ptr<Context>`. SYCL context objects are auto-created — one per platform group derived from the provided device list. Callers never construct `sycl::context` manually.

```cpp
class ContextBuilder {
public:
    ContextBuilder& devices(std::vector<sycl::device> devs);
    ContextBuilder& node_size(size_t ns);
    ContextBuilder& slab_size_per_device(size_t sz);
    std::shared_ptr<Context> build();
};
```

#### Methods

| Method | Default | Description |
|--------|---------|-------------|
| `devices(devs)` | — | Sets the list of SYCL devices to manage |
| `node_size(ns)` | 1024 bytes | Pool allocation granularity |
| `slab_size_per_device(sz)` | 64 MB (when sz == 0) | Uniform pool slab size for all devices in bytes |
| `build()` | — | Constructs and returns the Context; throws `std::invalid_argument` if device list is empty |

#### Notes

- Devices are grouped by `(backend, platform_name)` to create one `sycl::context` per platform group. This avoids SYCL multi-platform errors.
- Pools are **not** created at build time — they are lazy-initialized on first access via `pool_at()`.
- The returned `shared_ptr` is constructed via `new Context(...)` (not `make_shared`) to ensure `enable_shared_from_this` registers the control block correctly. See ADR-2.

---

### 2.4 `DeviceUSMPoolDeleter` `[STABLE]`

**Header:** `slabkit/memory.hpp`

Custom deleter for `unique_ptr` holding device USM memory allocated from a `Context` pool. Returned as part of `unique_device_ptr<T>`; users rarely construct this directly.

```cpp
struct DeviceUSMPoolDeleter {
    std::weak_ptr<Context> ctx;
    size_t device_index;
    size_t node_count;

    void operator()(void* ptr) const noexcept;
};
```

#### Notes

- Holds a `weak_ptr<Context>` to avoid keeping the context alive. If the context has been destroyed when the deleter fires, `ctx.lock()` returns null and the deallocation is a no-op (the slab was already freed with the context). See ADR-7.
- `node_count` is the number of pool slots that were allocated (pre-computed at allocation time).
- `operator()` is `noexcept` — deallocation never throws.

---

### 2.5 `unique_device_ptr<T>` `[STABLE]`

**Header:** `slabkit/memory.hpp`

Type alias for `unique_ptr` with pool-returning deleter.

```cpp
template <typename T>
using unique_device_ptr = std::unique_ptr<T, DeviceUSMPoolDeleter>;
```

Use this type to hold raw device memory allocated via `malloc_device<T>()`. When the pointer goes out of scope, memory is returned to the pool automatically.

---

### 2.6 `DeviceScratchView` `[STABLE]`

**Header:** `slabkit/scratch.hpp`

Non-owning, trivially-copyable view into device USM scratch memory. Safe to capture by value in SYCL kernels. Does not manage ownership or lifetime.

```cpp
struct DeviceScratchView {
    std::byte*  ptr_;
    std::size_t size_;

    static DeviceScratchView from_allocation(void* ptr, std::size_t size) noexcept;

    void*       data()  noexcept;
    const void* data()  const noexcept;
    std::size_t size()  const noexcept;
    bool        empty() const noexcept;
};
```

#### Notes

- `ptr_` is a device pointer; it cannot be dereferenced from host code.
- All member functions are `noexcept`.
- Use `from_allocation()` to create a view from a raw pointer (e.g., when wrapping externally-managed device memory).
- For pool-allocated scratch memory, use `make_scratch()` instead to get a `DeviceScratchMemory` (which exposes a view via its interface).

---

### 2.7 `DeviceScratchBuffer` `[DEPRECATED]`

**Header:** `slabkit/scratch.hpp`

```cpp
using DeviceScratchBuffer = DeviceScratchMemory;  // DEPRECATED — remove before v1.0
```

This alias was introduced to provide a "buffer-like" name for the scratch allocator but is misleading — `DeviceScratchMemory` is a bump-stack allocator, not a simple buffer. **Replace all uses with `DeviceScratchMemory` directly.** The alias will be removed before v1.0. See [Migration Notes](#7-migration-notes).

---

### 2.8 `DeviceView<T, N>` `[STABLE]`

**Header:** `slabkit/array.hpp`

Non-owning, contiguous, row-major view into device memory. Trivially copyable — safe to capture by value in SYCL kernels.

```cpp
template <typename T, int N>
struct DeviceView {
    T*             ptr_;
    sycl::range<N> range_;

    // Construction
    DeviceView(T* ptr, sycl::range<N> range) noexcept;

    // Implicit const conversion: DeviceView<U,N> → DeviceView<const U,N>
    template <typename U> requires std::same_as<T, const U>
    DeviceView(DeviceView<U, N> other) noexcept;

    // Accessors
    T*             data()      const noexcept;
    sycl::range<N> get_range() const noexcept;
    size_t         size()      const noexcept;
    size_t         byte_size() const noexcept;

    // Indexing
    T& operator()(auto... idx) const noexcept;  // exactly N arguments, row-major
    T& operator[](sycl::id<N> id) const noexcept;
};
```

#### Notes

- **Row-major layout:** inner-most dimension varies fastest. `operator()` accepts exactly `N` arguments.
- **`operator[]` requires `sycl::id<N>`**, not a raw integer. For 1D views, use `v[sycl::id<1>{i}]` or `v(i)`. A natural `v[i]` overload is planned (see [Section 9](#9-future-additions)).
- The implicit `DeviceView<U,N>` → `DeviceView<const U,N>` conversion enables passing mutable views to const-expecting functions without a cast.
- `DeviceArray<T,N>` implicitly converts to `DeviceView<T,N>` — prefer passing `DeviceView` to functions to avoid depending on the owning type.

---

### 2.9 `DeviceSubView<T, N>` `[STABLE]`

**Header:** `slabkit/array.hpp`

Non-owning, **strided** view into device memory. Trivially copyable — safe to capture by value in SYCL kernels. Supports non-contiguous layouts.

```cpp
template <typename T, int N>
struct DeviceSubView {
    T*             ptr_;
    sycl::range<N> range_;
    sycl::range<N> strides_;  // element strides, NOT byte strides

    // Construction
    DeviceSubView(T* ptr, sycl::range<N> range, sycl::range<N> strides) noexcept;

    // Implicit const conversion
    template <typename U> requires std::same_as<T, const U>
    DeviceSubView(DeviceSubView<U, N> other) noexcept;

    // Accessors
    T*             data()      const noexcept;
    sycl::range<N> get_range() const noexcept;
    size_t         size()      const noexcept;  // product of range dimensions
    size_t         byte_size() const noexcept;  // size() * sizeof(T)

    // Indexing
    T& operator()(auto... idx) const noexcept;  // exactly N arguments, strided
    T& operator[](sycl::id<N> id) const noexcept;
};
```

#### Notes

- **`strides_` are element counts, not byte offsets.** A stride of `{8, 1}` means "advance 8 elements to move one row". This is consistent with the SYCL dimension model and avoids `sizeof(T)` ambiguity. Contrast with BLAS/numpy where strides are typically byte offsets.
- `data()` points to the first element of the sub-view; it does not point to contiguous memory in general.
- `byte_size()` returns `size() * sizeof(T)` (logical element count in bytes), not the physical footprint of the strided region.
- Obtain a `DeviceSubView` from `DeviceArray::subview()` or `MirroredArray::subview()`.

---

### 2.10 `DeviceArray<T, N>` `[STABLE]`

**Header:** `slabkit/array.hpp`

Owning, RAII, move-only device array. Backed by a pool allocation via `DeviceUSMPoolDeleter`. When destroyed (or moved from), memory is returned to the pool.

```cpp
template <typename T, int N>
class DeviceArray {
public:
    // Non-copyable, move-only
    DeviceArray(const DeviceArray&)            = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    DeviceArray(DeviceArray&&)                 = default;
    DeviceArray& operator=(DeviceArray&&)      = default;

    // Data access
    T*       data() noexcept;
    const T* data() const noexcept;

    // Dimensions
    sycl::range<N> get_range()     const noexcept;
    size_t         size()          const noexcept;
    size_t         byte_size()     const noexcept;
    size_t         device_index()  const noexcept;

    // Views
    DeviceView<T, N>         view()       noexcept;
    DeviceView<const T, N>   view()       const noexcept;
    DeviceView<const T, N>   const_view() const noexcept;

    // Implicit conversion to view (enables pass-by-view ergonomics)
    operator DeviceView<T, N>()       noexcept;
    operator DeviceView<const T, N>() const noexcept;

    // Sub-views
    DeviceSubView<T, N>       subview(std::array<size_t, N> offsets,
                                      sycl::range<N> sub_range);
    DeviceSubView<const T, N> const_subview(std::array<size_t, N> offsets,
                                             sycl::range<N> sub_range) const;

    // Factory
    static DeviceArray from_pool(sycl::range<N> range,
                                  std::shared_ptr<Context> ctx,
                                  size_t device_index);
};
```

#### Notes

- **Device memory is uninitialized at allocation.** There is no zero-initialization guarantee on the device side.
- `subview()` bounds-checks offsets and sub_range against the array dimensions. In debug builds, invalid arguments trigger an `assert`; in NDEBUG builds the check is stripped (see [Known Limitations](#8-known-limitations)).
- Prefer `make_device_array()` free function over `from_pool()` for typical usage. `from_pool()` is the canonical factory but `make_device_array()` provides the same interface with a more discoverable name.
- Capturing a `DeviceArray` in a kernel lambda will fail to compile (copy-deleted). Capture `view()` or use implicit conversion. See ADR-3.

---

### 2.11 `MirroredArray<T, N>` `[STABLE]`

**Header:** `slabkit/array.hpp`

Paired device (`DeviceArray`) + host (`std::vector`) storage with explicit synchronisation methods. Move-only.

```cpp
template <typename T, int N>
class MirroredArray {
public:
    // Construction
    MirroredArray(sycl::range<N> range,
                  std::shared_ptr<Context> ctx,
                  size_t device_index);

    // Non-copyable, move-only
    MirroredArray(const MirroredArray&)            = delete;
    MirroredArray& operator=(const MirroredArray&) = delete;
    MirroredArray(MirroredArray&&)                 = default;
    MirroredArray& operator=(MirroredArray&&)      = default;

    // Deep copy (host side only)
    MirroredArray clone() const;

    // Device views
    DeviceView<T, N>         view()       noexcept;
    DeviceView<const T, N>   view()       const noexcept;
    DeviceView<const T, N>   const_view() const noexcept;
    DeviceSubView<T, N>       subview(std::array<size_t, N> offsets,
                                      sycl::range<N> sub_range);
    DeviceSubView<const T, N> const_subview(std::array<size_t, N> offsets,
                                             sycl::range<N> sub_range) const;

    // Host access
    std::span<T>       host_data() noexcept;
    std::span<const T> host_data() const noexcept;

    // Dimensions
    sycl::range<N> get_range()    const noexcept;
    size_t         size()         const noexcept;
    size_t         byte_size()    const noexcept;
    size_t         device_index() const noexcept;

    // Async sync (return event; caller must wait before accessing destination)
    sycl::event sync_to_host(sycl::queue& q,
                              const std::vector<sycl::event>& deps = {});
    sycl::event sync_to_device(sycl::queue& q,
                                const std::vector<sycl::event>& deps = {});
};
```

#### Notes

- **Initialization asymmetry:** Host side (`std::vector<T>`) is zero-initialized at construction. Device side is **uninitialized**. Call `sync_to_device()` to push zeroes to the device if needed.
- `clone()` copies only the host data into a new `MirroredArray`; the device side of the clone is uninitialized. Call `clone().sync_to_device(q)` for a full device copy.
- `sync_to_host()` and `sync_to_device()` are asynchronous — they return a `sycl::event`. The caller must wait on the event (or pass it as a dependency) before reading the destination.

---

## 3. Free Functions

### 3.1 Memory Allocation

#### `malloc_device<T>` — by device index `[STABLE]`

```cpp
template <typename T>
unique_device_ptr<T> malloc_device(size_t count,
                                   std::shared_ptr<Context> ctx,
                                   size_t device_index);
```

Allocates `count` elements of type `T` from the pool for `device_index`.

| Parameter | Description |
|-----------|-------------|
| `count` | Number of elements to allocate |
| `ctx` | Shared owning context |
| `device_index` | Index into `ctx->devices` |

**Returns:** `unique_device_ptr<T>` — RAII owning pointer; returns memory to pool on destruction.  
**Throws:** `std::out_of_range` (invalid index), `AllocationError` (pool exhausted or fragmented).

---

#### `malloc_device<T>` — by device `[STABLE]`

```cpp
template <typename T>
unique_device_ptr<T> malloc_device(size_t count,
                                   std::shared_ptr<Context> ctx,
                                   const sycl::device& dev);
```

Same as above; resolves `device_index` via `ctx->device_index(dev)`.  
**Throws:** additionally `std::invalid_argument` if `dev` is not in `ctx->devices`.

---

#### `make_scratch` — by device index `[STABLE]`

```cpp
inline DeviceScratchMemory make_scratch(std::shared_ptr<Context> ctx,
                                        std::size_t device_index,
                                        std::size_t size = 64 * 1024);
```

Creates a bump-allocator scratch buffer of `size` bytes backed by the pool for `device_index`.

| Parameter | Description |
|-----------|-------------|
| `ctx` | Shared owning context |
| `device_index` | Index into `ctx->devices` |
| `size` | Scratch capacity in bytes (default 64 KB) |

**Returns:** `DeviceScratchMemory` — pool-owning scratch allocator.  
**Throws:** `std::out_of_range`, `AllocationError`.

---

#### `make_scratch` — by device `[STABLE]`

```cpp
inline DeviceScratchMemory make_scratch(std::shared_ptr<Context> ctx,
                                        const sycl::device& dev,
                                        std::size_t size = 64 * 1024);
```

Same as above; resolves device via `ctx->pool_for(dev)`.  
**Throws:** additionally `std::invalid_argument` if `dev` is not in `ctx->devices`.

---

### 3.2 Array Factory

#### `make_device_array<T, N>` — by device index `[STABLE]`

```cpp
template <typename T, int N>
DeviceArray<T, N> make_device_array(sycl::range<N> range,
                                    std::shared_ptr<Context> ctx,
                                    size_t device_index);
```

Allocates a `DeviceArray<T,N>` of the given shape from the pool for `device_index`. Delegates to `DeviceArray<T,N>::from_pool()`.

**Returns:** `DeviceArray<T,N>` — move-only RAII owning array.  
**Throws:** `std::out_of_range`, `AllocationError`.

---

#### `make_device_array<T, N>` — by device `[STABLE]`

```cpp
template <typename T, int N>
DeviceArray<T, N> make_device_array(sycl::range<N> range,
                                    std::shared_ptr<Context> ctx,
                                    const sycl::device& dev);
```

Same as above; resolves `device_index` via `ctx->device_index(dev)`.  
**Throws:** additionally `std::invalid_argument` if `dev` is not in `ctx->devices`.

---

### 3.3 Transfer Functions

Transfer functions are organised into four tiers:

| Tier | Layout | Direction | Staging | Return |
|------|--------|-----------|---------|--------|
| 1 | Contiguous | Host↔Device, Device↔Device | `q.memcpy` | `sycl::event` |
| 2 | Strided | Host↔Device | Device scratch buffer | `sycl::event` |
| 3 | Strided | Device↔Device | None (kernel) | `sycl::event` |
| 4 | Any | Host↔Device | Delegates to Tier 1/2 | `void` (blocks) |

All async functions (Tiers 1–3) accept an optional `deps` vector and return a `sycl::event`. The event must be waited on (or passed as a dependency to subsequent operations) before the destination memory is accessed.

---

#### Tier 1 — Contiguous Async

```cpp
// DeviceArray → host
template <typename T, int N>
sycl::event copy_to_host(const DeviceArray<T, N>& src, T* dst,
                          sycl::queue& q,
                          const std::vector<sycl::event>& deps = {});

// DeviceView → host (mutable view)
template <typename T, int N>
sycl::event copy_to_host(DeviceView<T, N> src, T* dst,
                          sycl::queue& q,
                          const std::vector<sycl::event>& deps = {});

// DeviceView<const T> → host
template <typename T, int N>
sycl::event copy_to_host(DeviceView<const T, N> src, T* dst,
                          sycl::queue& q,
                          const std::vector<sycl::event>& deps = {});

// host → DeviceArray
template <typename T, int N>
sycl::event copy_to_device(const T* src, DeviceArray<T, N>& dst,
                             sycl::queue& q,
                             const std::vector<sycl::event>& deps = {});

// host → DeviceView
template <typename T, int N>
sycl::event copy_to_device(const T* src, DeviceView<T, N> dst,
                             sycl::queue& q,
                             const std::vector<sycl::event>& deps = {});

// DeviceArray → DeviceArray (same device)
template <typename T, int N>
sycl::event copy(const DeviceArray<T, N>& src, DeviceArray<T, N>& dst,
                  sycl::queue& q,
                  const std::vector<sycl::event>& deps = {});
```

**Notes:**
- All Tier 1 functions use `q.memcpy()`. Source and destination must be on the same device (for device-to-device `copy`). Calling `copy` across different device indices is undefined behaviour — an assertion guards against this in debug builds.
- `dst` for `copy_to_host` must be a valid host pointer with at least `src.byte_size()` bytes available.

---

#### Tier 2 — Strided Host Async (with Staging Scratch)

```cpp
// DeviceSubView → host (gather via staging buffer)
template <typename T, int N>
sycl::event copy_to_host(DeviceSubView<T, N> src, T* dst,
                          sycl::queue& q,
                          DeviceScratchMemory& scratch,
                          const std::vector<sycl::event>& deps = {});

// DeviceSubView<const T> → host
template <typename T, int N>
sycl::event copy_to_host(DeviceSubView<const T, N> src, T* dst,
                          sycl::queue& q,
                          DeviceScratchMemory& scratch,
                          const std::vector<sycl::event>& deps = {});

// host → DeviceSubView (scatter via staging buffer)
template <typename T, int N>
sycl::event copy_to_device(const T* src, DeviceSubView<T, N> dst,
                             sycl::queue& q,
                             DeviceScratchMemory& scratch,
                             const std::vector<sycl::event>& deps = {});
```

**Notes:**
- A contiguous staging buffer is allocated from `scratch` on the device side, then a gather/scatter kernel runs to rearrange elements according to the view strides.
- **Lifetime:** `scratch` must remain alive until the returned event completes. Calling `scratch.unwind()` or destroying `scratch` before the event is waited on is undefined behaviour.
- Scratch capacity must be at least `src.byte_size()` bytes.

---

#### Tier 3 — Strided Device-to-Device Async

```cpp
// DeviceSubView → DeviceSubView
template <typename T, int N>
sycl::event copy(DeviceSubView<T, N> src, DeviceSubView<T, N> dst,
                  sycl::queue& q,
                  const std::vector<sycl::event>& deps = {});

// DeviceSubView → DeviceArray
template <typename T, int N>
sycl::event copy(DeviceSubView<T, N> src, DeviceArray<T, N>& dst,
                  sycl::queue& q,
                  const std::vector<sycl::event>& deps = {});

// DeviceArray → DeviceSubView
template <typename T, int N>
sycl::event copy(const DeviceArray<T, N>& src, DeviceSubView<T, N> dst,
                  sycl::queue& q,
                  const std::vector<sycl::event>& deps = {});
```

**Notes:**
- Implemented as a `q.parallel_for` kernel — no staging buffer required.
- Source and destination must be on the same SYCL device (same SYCL context).
- Range dimensions of `src` and `dst` must match.

---

#### Tier 4 — Blocking Wrappers

```cpp
// Contiguous blocking
template <typename T, int N>
void sync_to_host(const DeviceArray<T, N>& src, T* dst, sycl::queue& q);

template <typename T, int N>
void sync_to_host(DeviceView<const T, N> src, T* dst, sycl::queue& q);

// Strided blocking (requires scratch)
template <typename T, int N>
void sync_to_host(DeviceSubView<T, N> src, T* dst,
                   sycl::queue& q, DeviceScratchMemory& scratch);

template <typename T, int N>
void sync_to_host(DeviceSubView<const T, N> src, T* dst,
                   sycl::queue& q, DeviceScratchMemory& scratch);
```

**Notes:**
- Each `sync_to_host` overload calls the corresponding `copy_to_host` then calls `.wait()` on the event. Returns `void`.
- Use Tier 4 in sequential code where blocking is acceptable. For performance-sensitive pipelines, prefer Tier 1–3 and manage events explicitly.

---

## 4. Concepts

### `DeviceViewLike<V, T, N>` `[STABLE]`

**Header:** `slabkit/array.hpp`

A concept satisfied by any type that exposes the core view interface — range, size, and indexed element access. Satisfied by `DeviceView<T,N>`, `DeviceSubView<T,N>`, and `DeviceArray<T,N>`.

```cpp
template <typename V, typename T, int N>
concept DeviceViewLike = requires(V v, sycl::id<N> id) {
    { v.get_range() } -> std::same_as<sycl::range<N>>;
    { v.size()      } -> std::convertible_to<size_t>;
    { v.byte_size() } -> std::convertible_to<size_t>;
    { v[id]         } -> std::same_as<T&>;
};
```

#### Notes

- `data()` is intentionally **not** required — `DeviceSubView` does not guarantee a contiguous data pointer.
- The concept is used internally to constrain generic kernel helpers. Users may use it to constrain their own generic device code.

---

## 5. Design Decisions

### ADR-1: No Global Registry

**Context:** An earlier design used an `AllocatorRegistry` singleton that tracked all `sycl::context` and pool state globally.

**Decision:** Remove the singleton entirely. All API entry points require an explicit `std::shared_ptr<Context>`.

**Rationale:** The singleton made it impossible to have multiple independent device contexts in one process (e.g., for testing, multi-tenant workloads, or multiple GPU backends). Explicit context passing is slightly more verbose but completely predictable, composable, and testable.

---

### ADR-2: `Context` is Always `shared_ptr`-Managed

**Context:** `Context` uses `std::enable_shared_from_this<Context>` so that pool deleters can hold a `weak_ptr<Context>` to the owning context.

**Decision:** `Context` must be constructed on the heap and wrapped in `std::shared_ptr`. Specifically, `ContextBuilder::build()` uses `std::shared_ptr<Context>(new Context(...))` rather than `std::make_shared<Context>(...)`.

**Rationale:** `enable_shared_from_this` works correctly only when the object is managed by a `shared_ptr` whose control block was created at the same time as the object. Using `make_shared` with a move-constructed temporary does not register the control block correctly. This is a subtle but critical correctness requirement — do not "fix" this to `make_shared`.

---

### ADR-3: `DeviceArray` is Move-Only

**Context:** `DeviceArray` owns USM device memory via a `unique_ptr` with a custom pool-returning deleter.

**Decision:** Copy constructor and copy assignment are deleted. Move constructor and move assignment are defaulted.

**Rationale:** Device memory is expensive and non-trivially owned (it holds a reference to a pool slot). Allowing copies would require either deep-copying device memory (expensive, requires a queue) or shallow-copying ownership (incorrect — would double-free on destruction). Move-only semantics make the ownership model explicit and compile-time-enforced. To pass arrays to functions, use `.view()` or rely on the implicit `DeviceArray` → `DeviceView` conversion.

---

### ADR-4: Views are Trivially Copyable

**Context:** `DeviceView<T,N>` and `DeviceSubView<T,N>` need to be capturable in SYCL kernel lambdas.

**Decision:** Both view types are trivially copyable structs (no user-defined constructors, destructors, or copy operators beyond the implicit ones).

**Rationale:** SYCL requires that types captured in kernel lambdas be trivially copyable (to enable device-side instantiation). Non-trivial types cannot be safely passed through the SYCL runtime. Making views trivially copyable is the idiomatic SYCL approach for non-owning handles.

---

### ADR-5: Four-Tier Transfer Function Design

**Context:** Data movement between host and device spans a wide range of complexity: simple contiguous copies, strided subarray copies, device-to-device copies, and blocking vs. async variants.

**Decision:** Transfer functions are organised into four explicit tiers with different signatures rather than using a single overload set with optional parameters.

**Rationale:** A single overloaded set would produce ambiguous overload resolution for strided vs. contiguous cases and obscure the presence of a staging scratch buffer. The four-tier design makes the performance characteristics and requirements (staging buffer, kernel dispatch) explicit at the call site. Tier 1 is zero-overhead; Tier 2 requires a scratch argument; Tier 3 dispatches a kernel; Tier 4 adds a `.wait()` call.

---

### ADR-6: Element Strides in `DeviceSubView`

**Context:** `DeviceSubView` needs to express non-contiguous strides to address sub-regions of larger arrays.

**Decision:** `strides_` in `DeviceSubView` are **element counts**, not byte offsets.

**Rationale:** Using element strides is consistent with the SYCL range/id model, which works in units of elements rather than bytes. Byte strides would require dividing by `sizeof(T)` everywhere and would be error-prone for non-power-of-two element sizes. Users familiar with numpy/BLAS byte strides should note this distinction explicitly.

---

### ADR-7: `weak_ptr` Lifetime Anchoring in Deleters

**Context:** `DeviceUSMPoolDeleter` and `DeviceScratchMemory` (pool-owning mode) hold references back to the `Context` to perform deallocation.

**Decision:** These types hold `std::weak_ptr<Context>` (type-erased in some cases) rather than `shared_ptr`.

**Rationale:** If arrays or scratch buffers outlive their context (e.g., a static or thread-local `DeviceArray` that outlives the context), the deleter must not crash. Using `weak_ptr` allows the deleter to detect context destruction (`lock()` returns null) and skip deallocation gracefully — the slab is already freed with the context. This is a silent no-op by design.

---

### ADR-8: Pool-Level Mutex (Coarse-Grained Locking)

**Context:** `DeviceMemoryPool` must support concurrent allocation from multiple threads.

**Decision:** Each pool is protected by a single `std::mutex` that covers both allocation and deallocation.

**Rationale:** Coarse-grained locking is correct and simple to reason about. Fine-grained per-segment or lock-free allocation is deferred to v0.2. For the current use case (one or a few threads doing allocation at setup time, then kernels running), coarse locking is not a bottleneck. See [Known Limitations](#8-known-limitations).

---

## 6. Thread-Safety Model

| Component | Thread-safe? | Notes |
|-----------|-------------|-------|
| `ContextBuilder::build()` | **No** | One-time construction; call before spawning threads |
| `Context` construction | **No** | Construct and fully configure before sharing across threads |
| `Context::pool_at()` / `pool_for()` | **Yes** | Lazy-init guarded by `pool_mutex_`; safe to call concurrently after Context is constructed |
| `DeviceMemoryPool::allocate_*` / `deallocate_*` | **Yes** | Internal `std::mutex` protects all freelist operations |
| `DeviceScratchMemory::allocate()` | **No** | No internal synchronization; use per-thread instances or external locking |
| `DeviceScratchMemory::unwind()` / `reset()` | **No** | Same as above |
| Transfer functions (Tier 1–4) | **Yes** (with caveats) | Thread-safe if source and destination are not concurrently modified; the SYCL queue is the synchronization primitive |
| `DeviceView` / `DeviceSubView` access from device | Per SYCL rules | Device-side data races follow SYCL memory model |

**Key rule:** Construct your `Context` once in a single thread before any concurrent work begins. After construction, concurrent pool allocations are safe. Never share a single `DeviceScratchMemory` across threads.

---

## 7. Migration Notes

### `DeviceScratchBuffer` → `DeviceScratchMemory`

`DeviceScratchBuffer` is a deprecated alias defined in `slabkit/scratch.hpp`:

```cpp
using DeviceScratchBuffer = DeviceScratchMemory;  // DEPRECATED
```

**Migration:** Replace every use of `DeviceScratchBuffer` with `DeviceScratchMemory`. The interface is identical — no other code changes are required.

```cpp
// Before (deprecated)
slabkit::DeviceScratchBuffer scratch = slabkit::make_scratch(ctx, 0);

// After
slabkit::DeviceScratchMemory scratch = slabkit::make_scratch(ctx, 0);
```

**Why:** The `Buffer` suffix implies a simple contiguous memory region. `DeviceScratchMemory` is a bump-stack allocator with `allocate()`, `top()`, `unwind()`, and `reset()` — the `Memory` suffix better captures this semantics.

**Timeline:** The alias will be removed before v1.0. All documentation, tests, and patterns are being updated to use `DeviceScratchMemory` directly.

---

## 8. Known Limitations

1. **No fine-grained pool locking.** The entire `DeviceMemoryPool` is locked during each allocation and deallocation. Under high concurrency (many threads allocating simultaneously), this mutex becomes a bottleneck. Fine-grained locking or lock-free allocation is planned for v0.2.

2. **No automatic fragmentation compaction.** When a `Fragmented` allocation error is thrown, the pool cannot compact its free space. The only recovery is to free enough existing allocations to create a contiguous run, or to recreate the pool entirely. Fragmentation is most likely with many small, short-lived allocations of varying sizes.

3. **No `operator[](size_t)` for 1D views.** Accessing element `i` of a 1D `DeviceView` or `DeviceSubView` requires `v[sycl::id<1>{i}]` or `v(i)`. The more natural `v[i]` (with a raw integer) is not supported. This is a planned ergonomic improvement (see [Section 9](#9-future-additions)).

4. **No runtime bounds checking in release builds.** `DeviceArray::subview()` and `DeviceArray::const_subview()` validate offsets and sub_range sizes using `assert()`. In NDEBUG builds (typical release configuration), these checks are stripped and out-of-bounds subviews result in undefined behaviour.

5. **`DeviceScratchMemory` is not thread-safe.** The bump allocator has no internal synchronization. Use one `DeviceScratchMemory` per thread, or provide external locking, when scratch buffers are accessed from multiple threads.

---

## 9. Future Additions

The following improvements are planned but not yet scheduled for a specific release:

### `operator[](size_t)` for 1D Views

Add a size_t-indexed overload to `DeviceView<T,1>` and `DeviceSubView<T,1>`:

```cpp
T& operator[](size_t i) const noexcept;  // planned
```

This would allow natural 1D access (`v[i]`) without requiring `sycl::id<1>{i}` or `v(i)`.

### Fine-Grained Pool Locking (v0.2)

Replace the per-pool coarse `std::mutex` with per-segment locking or a lock-free bitmap implementation. This would allow multiple threads to allocate concurrently from the same pool without serializing on a single mutex.
