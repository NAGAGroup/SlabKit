# SlabKit API Reference

Complete reference documentation for SlabKit's public API. This page is for looking up specific types and functions—not a tutorial. Each entry includes the signature, purpose, parameters, return value, and important notes. Assume familiarity with C++, SYCL, and USM. For conceptual background and design rationale, see [Concepts](concepts.md).

## Quick Reference

| Type / Function | Header | Purpose |
|---|---|---|
| `AllocationError` | `slabkit/error.hpp` | Exception thrown on memory allocation failure |
| `ContextBuilder` | `slabkit/context.hpp` | Fluent builder for creating a `Context` |
| `Context` | `slabkit/context.hpp` | Root ownership object; manages devices, SYCL contexts, and pools |
| `DeviceUSMPoolDeleter` | `slabkit/memory.hpp` | Custom deleter for returning USM to pool |
| `unique_device_ptr<T>` | `slabkit/memory.hpp` | Owning smart pointer for USM device memory |
| `malloc_device<T>()` | `slabkit/memory.hpp` | Allocate raw USM memory from pool |
| `DeviceViewLike` | `slabkit/array.hpp` | Concept for trivially-copyable device views |
| `DeviceView<T, N>` | `slabkit/array.hpp` | Non-owning contiguous view of device memory |
| `DeviceSubView<T, N>` | `slabkit/array.hpp` | Non-owning strided view of device memory |
| `DeviceArray<T, N>` | `slabkit/array.hpp` | RAII-managed pool-backed device array |
| `make_device_array<T, N>()` | `slabkit/array.hpp` | Primary allocator for `DeviceArray` |
| `MirroredArray<T, N>` | `slabkit/array.hpp` | Host + device pair with sync methods |
| `copy_to_host()` | `slabkit/array.hpp` | Async transfer device → host |
| `copy_to_device()` | `slabkit/array.hpp` | Async transfer host → device |
| `copy()` | `slabkit/array.hpp` | Async transfer device ↔ device |
| `sync_to_host()` | `slabkit/array.hpp` | Blocking transfer device → host |
| `DeviceScratchMemory` | `slabkit/scratch.hpp` | Bump allocator for temporary device memory |
| `DeviceScratchView` | `slabkit/scratch.hpp` | Non-owning view of scratch allocation |
| `make_scratch()` | `slabkit/scratch.hpp` | Create pool-backed scratch memory |

---

## Error Types

### `AllocationError`

**Header:** `slabkit/error.hpp`  
**Include via:** `#include "slabkit/slabkit.hpp"`

Exception thrown when device memory allocation fails.

```cpp
class AllocationError : public std::exception {
public:
    enum class Reason { PoolExhausted, Fragmented, ScratchFull };
    
    explicit AllocationError(Reason r, const char* msg = "");
    
    Reason      reason() const noexcept;
    const char* what() const noexcept override;
};
```

**Description**  
Thrown by `malloc_device()`, `make_device_array()`, and `DeviceScratchMemory::allocate()` when the request cannot be satisfied.

**Parameters**
- `r` — The reason for failure (see `Reason` enum).
- `msg` — Optional human-readable detail (usually a short string, e.g., "device 0 slab full").

**Return Value**  
None (exception). Call `reason()` to query the cause; call `what()` for a descriptive message.

**Reasons**
- `PoolExhausted` — The device's USM slab has no free space.
- `Fragmented` — The slab is too fragmented to satisfy the allocation despite available free space.
- `ScratchFull` — The scratch memory bump allocator is out of capacity.

**Notes**
- Thread-safe to throw and catch across threads.
- Inherits from `std::exception`; handle as a standard C++ exception.

---

## Context Management

### `ContextBuilder`

**Header:** `slabkit/context.hpp`  
**Include via:** `#include "slabkit/slabkit.hpp"`

Fluent builder for constructing a `Context`.

```cpp
class ContextBuilder {
public:
    ContextBuilder& devices(std::vector<sycl::device> devs);
    ContextBuilder& node_size(size_t ns);
    ContextBuilder& slab_size_per_device(size_t sz);
    std::shared_ptr<Context> build();
};
```

**Description**  
Configures and constructs a `Context` using the builder pattern. Each configuration method returns `*this`, enabling chaining. `build()` creates the Context and groups devices by platform.

**Methods**

| Method | Default | Notes |
|---|---|---|
| `devices(vec)` | Required | List of `sycl::device` objects to manage. `build()` throws `std::invalid_argument` if empty. |
| `node_size(ns)` | 1024 bytes | Allocation granularity for the pool. All requests are rounded up to multiples of this. |
| `slab_size_per_device(sz)` | 64 MB | Total USM slab size per device. Once full, further allocations raise `AllocationError`. |
| `build()` | — | Constructs the Context; returns `std::shared_ptr<Context>`. Never returns null. |

**Return Value (build)**  
`std::shared_ptr<Context>` — never null. Always use the shared_ptr; never stack-allocate Context.

**Platform Grouping**  
`build()` automatically groups devices by platform (backend + platform name). For each unique platform group, it creates exactly one `sycl::context`. This minimizes SYCL resource usage and enables intra-platform device operations.

**Example**
```cpp
#include "slabkit/slabkit.hpp"

auto dev = sycl::device{sycl::gpu_selector_v};
auto ctx = slabkit::ContextBuilder()
    .devices({dev})
    .node_size(1024)
    .slab_size_per_device(256 * 1024 * 1024)
    .build();
```

**Exceptions**  
- `std::invalid_argument` — if `devices()` was not called or given an empty list.

---

### `Context`

**Header:** `slabkit/context.hpp`  
**Include via:** `#include "slabkit/slabkit.hpp"`

Root ownership object managing devices, SYCL contexts, and memory pools. Heap-allocated as `shared_ptr<Context>` via `ContextBuilder::build()`.

```cpp
struct Context : std::enable_shared_from_this<Context> {
    // Data members (public, read-only after construction)
    std::vector<sycl::context>  platform_contexts;
    std::vector<sycl::device>   devices;
    std::vector<size_t>         device_to_platform;
    std::vector<size_t>         device_slab_sizes;

    // Device lookup
    size_t device_index(const sycl::device& dev) const;

    // Pool access (lazy-init, mutex-protected)
    DeviceMemoryPool& pool_at(size_t idx);
    const DeviceMemoryPool& pool_at(size_t idx) const;
    DeviceMemoryPool& pool_for(const sycl::device& dev);
    const DeviceMemoryPool& pool_for(const sycl::device& dev) const;

    // SYCL context lookup
    const sycl::context& context_for(size_t device_index_val) const;
    const sycl::context& context_for(const sycl::device& dev) const;

    // Queue factories
    sycl::queue make_in_order_queue(const sycl::device& dev) const;
    sycl::queue make_out_of_order_queue(const sycl::device& dev) const;
    sycl::queue make_queue_with_props(const sycl::device& dev,
                                       const sycl::property_list& props) const;

    // Move-only
    Context(Context&&) noexcept;
    Context& operator=(Context&&) = delete;
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
};
```

**Description**  
Never construct directly. Always via `ContextBuilder::build()`. The Context owns all devices, SYCL contexts, and pools. It is the root of the ownership hierarchy.

**Data Members**  
- `platform_contexts` — One `sycl::context` per unique platform group.
- `devices` — All devices passed to the builder, in order.
- `device_to_platform` — Maps `devices[i]` to `platform_contexts[device_to_platform[i]]`.
- `device_slab_sizes` — Pool slab size for each device (currently uniform, but indexed per device).

All members are mutable but conceptually immutable after construction.

**Methods**

**`device_index(dev) → size_t`**  
Returns the index of `dev` in the `devices` vector.

| | |
|---|---|
| **Exceptions** | `std::invalid_argument` if `dev` not found. |
| **Notes** | Useful for resolving a device to a pool index before calling `pool_at()`. |

**`pool_at(idx) → DeviceMemoryPool&` / `const DeviceMemoryPool&`**  
Returns a reference to the pool for device `idx`.

| | |
|---|---|
| **Parameters** | `idx` — device index (0-based). |
| **Exceptions** | `std::out_of_range` if `idx >= devices.size()`. |
| **Thread Safety** | Lazy-initializes the pool once (mutex-protected). Subsequent calls are lock-free. |
| **Notes** | `DeviceMemoryPool` is an internal implementation detail; do not call its methods directly. |

**`pool_for(dev) → DeviceMemoryPool&` / `const DeviceMemoryPool&`**  
Convenience overload: resolves `device_index(dev)` and calls `pool_at()`.

| | |
|---|---|
| **Exceptions** | `std::invalid_argument` if `dev` not found; `std::out_of_range` if index out of range. |
| **Thread Safety** | Same as `pool_at()`. |

**`context_for(idx) → const sycl::context&`** / **`context_for(dev) → const sycl::context&`**  
Returns the SYCL context for a given device.

| | |
|---|---|
| **Parameters (idx)** | `idx` — device index. |
| **Exceptions** | `std::out_of_range` if index invalid. |
| **Parameters (dev)** | `dev` — a device object. |
| **Exceptions** | `std::invalid_argument` if device not found. |
| **Notes** | Returns the SYCL context associated with `dev`'s platform group. |

**Queue Factories**

**`make_in_order_queue(dev) → sycl::queue`**  
Creates an in-order queue for the device (guarantees sequential kernel execution).

| | |
|---|---|
| **Exceptions** | `std::invalid_argument` if `dev` not found. |
| **Notes** | Equivalent to `sycl::queue{context_for(dev), dev, sycl::property::queue::in_order{}}`. |

**`make_out_of_order_queue(dev) → sycl::queue`**  
Creates an out-of-order queue (kernels may execute in any order; faster for independent work).

| | |
|---|---|
| **Exceptions** | `std::invalid_argument` if `dev` not found. |

**`make_queue_with_props(dev, props) → sycl::queue`**  
Creates a queue with custom properties.

| | |
|---|---|
| **Parameters** | `dev` — device; `props` — `sycl::property_list` with desired properties. |
| **Exceptions** | `std::invalid_argument` if `dev` not found. |
| **Notes** | If you need fine control over queue properties, use this method. |

**Thread Safety**  
- Construction is **not** thread-safe. Call `ContextBuilder::build()` once before spawning threads.
- All access methods (`pool_at()`, `device_index()`, queue factories) are thread-safe after construction.
- Pool initialization is thread-safe (internal mutex).

**Ownership**  
- Always heap-allocated as `shared_ptr<Context>`.
- Move-constructible but not move-assignable; copy-deleted.
- Arrays hold `weak_ptr<Context>` in their deleters, ensuring Context can be destroyed independently.

---

## Raw Memory

### `DeviceUSMPoolDeleter`

**Header:** `slabkit/memory.hpp`  
**Include via:** `#include "slabkit/slabkit.hpp"`

Custom deleter for `std::unique_ptr` that returns USM memory to the pool instead of freeing it.

```cpp
struct DeviceUSMPoolDeleter {
    std::weak_ptr<Context> ctx;
    size_t                 device_index;
    size_t                 node_count;
    
    void operator()(void* ptr) const noexcept;
};
```

**Description**  
Do **not** construct this directly. It is used internally by `malloc_device()` and `DeviceArray`. When a `unique_ptr<T, DeviceUSMPoolDeleter>` is destroyed, the deleter returns nodes to the pool for reuse.

**Members**  
- `ctx` — weak reference to the `Context` owning the pool.
- `device_index` — which device's pool to return to.
- `node_count` — how many nodes to return.

**Behavior**  
When `operator()` is called on destruction:
- If the Context is still alive, nodes are returned to the pool.
- If the Context is destroyed, the `weak_ptr` fails silently; no return occurs. This is safe—the pool is gone anyway.

**Thread Safety**  
The deleter's destructor is safe to call from any thread, even if it races with Context destruction.

**Notes**  
This is an implementation detail. Users interact with it through `unique_device_ptr` and `malloc_device()`.

---

### `unique_device_ptr<T>`

**Header:** `slabkit/memory.hpp`  
**Include via:** `#include "slabkit/slabkit.hpp"`

Type alias for `std::unique_ptr<T, DeviceUSMPoolDeleter>`.

```cpp
template <typename T>
using unique_device_ptr = std::unique_ptr<T, DeviceUSMPoolDeleter>;
```

**Description**  
Owning smart pointer for USM device memory. Behaves like a standard `std::unique_ptr`: move-only, non-copyable, automatically calls the deleter on destruction. The deleter returns nodes to the pool instead of freeing.

**Example**
```cpp
auto ptr = slabkit::malloc_device<float>(1024, ctx, 0);  // unique_device_ptr<float>
// Use ptr[i] to access elements
ptr[512] = 3.14f;
// On destruction, returns 1024 nodes to device 0's pool
```

---

### `malloc_device<T>()`

**Header:** `slabkit/memory.hpp`  
**Include via:** `#include "slabkit/slabkit.hpp"`

Allocate raw USM device memory from the Context's pool.

```cpp
template <typename T>
unique_device_ptr<T> malloc_device(size_t count,
                                    std::shared_ptr<Context> ctx,
                                    size_t device_index);

template <typename T>
unique_device_ptr<T> malloc_device(size_t count,
                                    std::shared_ptr<Context> ctx,
                                    const sycl::device& dev);
```

**Description**  
Low-level allocation of `count` elements of type `T` from the pool. Returns an owning `unique_device_ptr<T>`. This is the foundation for `DeviceArray`; most users should use `make_device_array()` instead for proper RAII and rank safety.

**Parameters**

| Overload | Parameter | Notes |
|---|---|---|
| Both | `count` | Number of `T` elements to allocate. |
| Both | `ctx` | `shared_ptr<Context>`. |
| (1) | `device_index` | Index of the device (0-based). |
| (2) | `dev` | `sycl::device` object; resolved to index via `ctx->device_index(dev)`. |

**Return Value**  
`unique_device_ptr<T>` — owning smart pointer, safe to use as `ptr[i]` or `ptr.get()`.

**Exceptions**  
- `AllocationError` — if the pool's slab is exhausted or fragmented.
- `std::out_of_range` (overload 1) — if `device_index` is invalid.
- `std::invalid_argument` (overload 2) — if `dev` not found in Context.

**Example**
```cpp
// Allocate 1000 ints from device 0
auto ptr = slabkit::malloc_device<int>(1000, ctx, 0);

// Use in a kernel
sycl::queue q = ctx->make_in_order_queue(ctx->devices[0]);
int* raw = ptr.get();
q.parallel_for(sycl::range<1>{1000}, [=](sycl::id<1> i) {
    raw[i.get(0)] = i.get(0) * 2;
}).wait();

// Destroyed: nodes returned to pool
```

**Notes**  
- The returned pointer is uninitialized (unless `T` is trivially default-constructible and you use value initialization).
- No bounds checking; using `ptr[i]` with `i >= count` is undefined behavior.
- Use `DeviceArray<T, N>` for safer, dimension-aware allocation and transfer functions.

---

## Views

### `DeviceViewLike` Concept

**Header:** `slabkit/array.hpp`  
**Include via:** `#include "slabkit/slabkit.hpp"`

Concept for any type that provides device memory access.

```cpp
template <typename V, typename T, int N>
concept DeviceViewLike = requires(V v, sycl::id<N> id) {
    { v.get_range() } -> std::same_as<sycl::range<N>>;
    { v.size() }      -> std::convertible_to<size_t>;
    { v.byte_size() } -> std::convertible_to<size_t>;
    { v[id] }         -> std::same_as<T&>;
};
```

**Description**  
Refines any type that:
1. Has a shape (rank `N`)
2. Supports indexing by `sycl::id<N>`
3. Reports size in elements and bytes

Both `DeviceView<T, N>` and `DeviceSubView<T, N>` model this concept. The concept does **not** require `data()` to be contiguous.

**Usage**  
Use this concept in generic functions to accept either view type:

```cpp
template <typename V, typename T, int N>
requires slabkit::DeviceViewLike<V, T, N>
void kernel_launcher(V view, sycl::queue& q) {
    q.parallel_for(view.get_range(), [=](sycl::id<N> id) {
        view[id] = 42;
    }).wait();
}
```

---

### `DeviceView<T, N>`

**Header:** `slabkit/array.hpp`  
**Include via:** `#include "slabkit/slabkit.hpp"`

Non-owning, trivially-copyable view of contiguous device memory.

```cpp
template <typename T, int N>
struct DeviceView {
    T*             ptr_;
    sycl::range<N> range_;

    DeviceView(T* ptr, sycl::range<N> range) noexcept;

    // Implicit const conversion
    template <typename U> requires std::same_as<T, const U>
    DeviceView(DeviceView<U, N> other) noexcept;

    T*             data() const noexcept;
    sycl::range<N> get_range() const noexcept;
    size_t         size() const noexcept;
    size_t         byte_size() const noexcept;

    T& operator()(auto... idx) const noexcept;      // N indices, row-major
    T& operator[](sycl::id<N> id) const noexcept;
};
```

**Description**  
A trivially-copyable projection over a contiguous array. It holds only a pointer and shape, making it safe to capture by value in SYCL kernel lambdas.

**Constructor**  
`DeviceView(T* ptr, sycl::range<N> range)` — directly constructs a view. Usually obtained via `DeviceArray::view()` instead.

**Const Conversion**  
An implicit conversion from `DeviceView<T, N>` to `DeviceView<const T, N>` is available. Allows non-const views to implicitly convert to const.

**Methods**

| Method | Returns | Notes |
|---|---|---|
| `data()` | `T*` | Pointer to the first element. |
| `get_range()` | `sycl::range<N>` | The shape. |
| `size()` | `size_t` | Product of all range dimensions (in elements). |
| `byte_size()` | `size_t` | `size() * sizeof(T)`. |

**Indexing**

**`operator()(idx0, idx1, ...) → T&`** (row-major)  
Index by individual dimensions:
```cpp
auto elem = view(row, col);  // for N=2
```
Exactly `N` arguments required.

**`operator[](sycl::id<N>) → T&`**  
Index by `sycl::id<N>`:
```cpp
auto elem = view[sycl::id<2>{row, col}];
```

Both use row-major (C-order) indexing: `ptr_[sum(idx[d] * range_[d+1..N])]`.

**Triviality**  
`DeviceView` is trivially copyable and trivially destructible—it is **safe to capture by value** in kernel lambdas. Unlike `DeviceArray`, it does not prevent copies.

**Ownership**  
Non-owning; does not extend the lifetime of the backing array. The backing `DeviceArray` must outlive all kernels using this view.

**Example**
```cpp
auto view = arr.view();
sycl::queue q = ctx->make_in_order_queue(dev);

auto ev = q.parallel_for(view.get_range(), [=](sycl::id<2> id) {
    view[id] *= 2.0f;  // Capture by value, safe
});
ev.wait();
// arr must still be in scope; view does not own it
```

---

### `DeviceSubView<T, N>`

**Header:** `slabkit/array.hpp`  
**Include via:** `#include "slabkit/slabkit.hpp"`

Non-owning, trivially-copyable view of strided device memory.

```cpp
template <typename T, int N>
struct DeviceSubView {
    T*             ptr_;
    sycl::range<N> range_;
    sycl::range<N> strides_;

    DeviceSubView(T* ptr, sycl::range<N> range, sycl::range<N> strides) noexcept;

    // Implicit const conversion
    template <typename U> requires std::same_as<T, const U>
    DeviceSubView(DeviceSubView<U, N> other) noexcept;

    T*             data() const noexcept;
    sycl::range<N> get_range() const noexcept;
    size_t         size() const noexcept;
    size_t         byte_size() const noexcept;

    T& operator()(auto... idx) const noexcept;      // N indices, strided
    T& operator[](sycl::id<N> id) const noexcept;
};
```

**Description**  
A strided view over device memory. Indexing uses element strides, not byte strides. Trivially copyable and safe for kernels.

**Constructor**  
`DeviceSubView(T* ptr, sycl::range<N> range, sycl::range<N> strides)` — usually created via `DeviceArray::subview()` or transfer function overloads.

**Strides**  
The `strides_` are **element strides**, not bytes. For a 2D array with row-major layout and every other column, you might use `strides = {1, 2}`. Indexing computes: `ptr_[id[0] * strides_[0] + id[1] * strides_[1] + ...]`.

**Methods**  
Same as `DeviceView`, except:

- `data()` — Returns the base pointer. **Warning:** The view is strided; `data()` does **not** imply contiguity. Incrementing by 1 does not move to the next logical element.
- All other methods (`get_range()`, `size()`, `byte_size()`) describe the logical view, not the physical memory layout.

**Indexing**  
Same API as `DeviceView`, but strides apply. `operator()(idx0, idx1, ...)` computes `ptr_[sum(idx[d] * strides_[d])]`.

**Triviality**  
Trivially copyable; safe to capture in kernels.

**Example**
```cpp
// Create a subview with every other row and column
auto sub = arr.subview({0, 0}, {512, 512});
// Modify strides to skip: (1, 2) means stride 1 in rows, stride 2 in cols
// (In practice, subview() returns contiguous views; you'd construct this manually if needed)
```

---

## Arrays

### `DeviceArray<T, N>`

**Header:** `slabkit/array.hpp`  
**Include via:** `#include "slabkit/slabkit.hpp"`

RAII-managed, pool-backed multidimensional array. Move-only.

```cpp
template <typename T, int N>
class DeviceArray {
public:
    // Move-only
    DeviceArray(const DeviceArray&)            = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    DeviceArray(DeviceArray&&)                 = default;
    DeviceArray& operator=(DeviceArray&&)      = default;

    T*             data() noexcept;
    const T*       data() const noexcept;
    sycl::range<N> get_range() const noexcept;
    size_t         size() const noexcept;
    size_t         byte_size() const noexcept;
    size_t         device_index() const noexcept;

    DeviceView<T, N>       view() noexcept;
    DeviceView<const T, N> view() const noexcept;
    DeviceView<const T, N> const_view() const noexcept;

    operator DeviceView<T, N>() noexcept;
    operator DeviceView<const T, N>() const noexcept;

    DeviceSubView<T, N>       subview(std::array<size_t, N> offsets, sycl::range<N> sub_range);
    DeviceSubView<const T, N> const_subview(std::array<size_t, N> offsets, sycl::range<N> sub_range) const;

    static DeviceArray from_pool(sycl::range<N> range,
                                  std::shared_ptr<Context> ctx,
                                  size_t device_index);
};
```

**Description**  
The primary way to own device memory. Backed by `std::unique_ptr<T, DeviceUSMPoolDeleter>`. When destroyed, automatically returns nodes to the pool. Move-only to enforce clear ownership semantics.

**Construction**  
Do **not** construct directly. Use `make_device_array<T, N>()`. The static method `from_pool()` is for internal use.

**Methods**

| Method | Returns | Notes |
|---|---|---|
| `data()` | `T*` / `const T*` | Pointer to first element. |
| `get_range()` | `sycl::range<N>` | The shape (dimensions). |
| `size()` | `size_t` | Total number of elements. |
| `byte_size()` | `size_t` | `size() * sizeof(T)`. |
| `device_index()` | `size_t` | Which device owns this memory. |

**Views**

**`view() → DeviceView<T, N>` / `const_view() → DeviceView<const T, N>`**  
Returns a trivially-copyable view. Use this to access the array in kernels.

```cpp
auto view = arr.view();
q.parallel_for(view.get_range(), [=](sycl::id<N> id) {
    view[id] = compute(id);
}).wait();
```

**Implicit Conversion**  
`operator DeviceView<T, N>()` allows passing an array where a view is expected:
```cpp
void fill_kernel(slabkit::DeviceView<float, 2> view, sycl::queue& q) {
    q.parallel_for(view.get_range(), [=](sycl::id<2> id) {
        view[id] = 42.0f;
    }).wait();
}

auto arr = slabkit::make_device_array<float, 2>(range, ctx, 0);
fill_kernel(arr, q);  // arr implicitly converts to DeviceView<float, 2>
```

**Subviews**

**`subview(offsets, sub_range) → DeviceSubView<T, N>`**  
Creates a contiguous subview.

| | |
|---|---|
| **Parameters** | `offsets` — starting position in each dimension; `sub_range` — shape of subview. |
| **Precondition** | For all dimensions `d`: `offsets[d] + sub_range[d] <= range_[d]`. Asserted. |
| **Returns** | A `DeviceSubView<T, N>` with base pointer adjusted and strides = 1 in all dimensions. |
| **Notes** | The subview is contiguous within the backing array's layout. |

**Example**
```cpp
auto arr = slabkit::make_device_array<float, 2>(
    sycl::range<2>{1024, 1024},
    ctx,
    0
);
auto sub = arr.subview({0, 0}, {512, 512});  // Top-left quadrant
```

**Ownership**  
- Owns backing memory via `unique_ptr<T, DeviceUSMPoolDeleter>`.
- Holds `shared_ptr<Context>` in the deleter (via `weak_ptr`), ensuring the pool stays alive.
- Move-only; destruction returns nodes to the pool.

**Thread Safety**  
- Individual array instances are **not** thread-safe. Do not access the same array from multiple threads simultaneously.
- Different arrays from the same Context **are** thread-safe (they use different memory).

---

### `make_device_array<T, N>()`

**Header:** `slabkit/array.hpp`  
**Include via:** `#include "slabkit/slabkit.hpp"`

Primary allocator for `DeviceArray<T, N>`.

```cpp
template <typename T, int N>
DeviceArray<T, N> make_device_array(sycl::range<N> range,
                                     std::shared_ptr<Context> ctx,
                                     size_t device_index);

template <typename T, int N>
DeviceArray<T, N> make_device_array(sycl::range<N> range,
                                     std::shared_ptr<Context> ctx,
                                     const sycl::device& dev);
```

**Description**  
Allocates `T[size()]` where `size() = product(range)` from the Context's pool. Returns an owning `DeviceArray<T, N>`.

**Parameters**

| Overload | Parameter | Notes |
|---|---|---|
| Both | `range` | Shape of the array in each dimension. |
| Both | `ctx` | `shared_ptr<Context>`. |
| (1) | `device_index` | Device index (0-based). |
| (2) | `dev` | `sycl::device`; resolved via `ctx->device_index(dev)`. |

**Return Value**  
`DeviceArray<T, N>` — move-only, move-constructed (RVO applies).

**Exceptions**  
- `AllocationError` — slab exhausted, fragmented, or Context weak_ptr dead.
- `std::out_of_range` (overload 1) — invalid device index.
- `std::invalid_argument` (overload 2) — device not found in Context.

**Memory Initialization**  
The allocated memory is **uninitialized** (contains garbage). If you need zeroed memory, transfer zeros from host:

```cpp
std::vector<float> zeros(arr.size(), 0.0f);
slabkit::copy_to_device(zeros.data(), arr, q);
```

**Example**
```cpp
auto arr = slabkit::make_device_array<int, 2>(
    sycl::range<2>{256, 256},
    ctx,
    dev
);
// arr.size() == 256 * 256 == 65536
// arr.byte_size() == 65536 * sizeof(int) == 262144 bytes
```

---

### `MirroredArray<T, N>`

**Header:** `slabkit/array.hpp`  
**Include via:** `#include "slabkit/slabkit.hpp"`

Host + device pair with synchronized copies. Host side is a `std::vector<T>`; device side is a `DeviceArray<T, N>`.

```cpp
template <typename T, int N>
class MirroredArray {
public:
    MirroredArray(sycl::range<N> range,
                  std::shared_ptr<Context> ctx,
                  size_t device_index);

    // Move-only
    MirroredArray(const MirroredArray&)            = delete;
    MirroredArray& operator=(const MirroredArray&) = delete;
    MirroredArray(MirroredArray&&)                 = default;
    MirroredArray& operator=(MirroredArray&&)      = default;

    MirroredArray clone() const;

    DeviceView<T, N>       view() noexcept;
    DeviceView<const T, N> view() const noexcept;
    DeviceView<const T, N> const_view() const noexcept;

    DeviceSubView<T, N>       subview(std::array<size_t, N> offsets, sycl::range<N> sub_range);
    DeviceSubView<const T, N> const_subview(std::array<size_t, N> offsets, sycl::range<N> sub_range) const;

    std::span<T>       host_data() noexcept;
    std::span<const T> host_data() const noexcept;

    sycl::range<N> get_range() const noexcept;
    size_t         size() const noexcept;
    size_t         byte_size() const noexcept;
    size_t         device_index() const noexcept;

    sycl::event sync_to_host(sycl::queue& q, const std::vector<sycl::event>& deps = {});
    sycl::event sync_to_device(sycl::queue& q, const std::vector<sycl::event>& deps = {});
};
```

**Description**  
Owns both a host vector (zero-initialized on construction) and a device array (uninitialized). Provides methods to synchronize between them.

**Constructor**  
```cpp
MirroredArray<float, 2> arr(sycl::range<2>{512, 512}, ctx, dev_idx);
// Host side: std::vector<float> with 262144 zeros
// Device side: DeviceArray<float, 2>, uninitialized
```

**Methods**

| Method | Notes |
|---|---|
| `view()` / `const_view()` | Returns a view of the **device** array. Use in kernels. |
| `subview()` | Creates a subview of the **device** array. |
| `host_data()` | Returns a `std::span<T>` to the host vector. |
| `get_range()` | Shape of both host and device arrays. |
| `size()` | Total elements. |
| `byte_size()` | Bytes per array (both sides are the same size). |
| `device_index()` | Which device owns the device array. |
| `clone()` | Deep-copies the host vector; allocates a new uninitialized device array. |

**Synchronization**

**`sync_to_host(q, deps) → sycl::event`**  
Asynchronously copies device → host. Depends on `deps`.

**`sync_to_device(q, deps) → sycl::event`**  
Asynchronously copies host → device. Depends on `deps`.

Both return a `sycl::event` you can wait on or add as a dependency to future work.

**Example**
```cpp
// Use an in-order queue so sync and kernel execute sequentially
sycl::queue q = ctx->make_in_order_queue(ctx->devices[0]);
auto arr = slabkit::MirroredArray<float, 2>(sycl::range<2>{512, 512}, ctx, 0);

// Modify host side
std::fill(arr.host_data().begin(), arr.host_data().end(), 1.0f);

// Copy to device
arr.sync_to_device(q).wait();

// Run kernel (in-order queue ensures sync is complete)
auto view = arr.view();
q.parallel_for(view.get_range(), [=](sycl::id<2> id) {
    view[id] *= 2.0f;  // Each element becomes 2.0
}).wait();

// Copy back
arr.sync_to_host(q).wait();

// Read host side
std::cout << arr.host_data()[0] << "\n";  // Prints 2.0
```

**Notes**  
- Host side is zero-initialized; device side is not.
- `clone()` is useful for creating independent copies for different computations.
- Not suitable for large-scale GPU-only work; use plain `DeviceArray` if the host never needs the data.

---

## Transfer Functions

Transfer functions move data between host and device. They are organized into four tiers by capability and interface.

### Tier 1: Contiguous Transfers (Async)

Return a `sycl::event` for dependency tracking. No staging buffer required. Can transfer from/to `DeviceArray` or `DeviceView`.

**Device → Host**

```cpp
template <typename T, int N>
sycl::event copy_to_host(const DeviceArray<T, N>& src, T* dst, sycl::queue& q,
                          const std::vector<sycl::event>& deps = {});

template <typename T, int N>
sycl::event copy_to_host(DeviceView<T, N> src, T* dst, sycl::queue& q,
                          const std::vector<sycl::event>& deps = {});

template <typename T, int N>
sycl::event copy_to_host(DeviceView<const T, N> src, T* dst, sycl::queue& q,
                          const std::vector<sycl::event>& deps = {});
```

| | |
|---|---|
| **Description** | Copies contiguous device memory to a host buffer. |
| **Parameters** | `src` — source array or view; `dst` — host buffer (must be at least `src.size() * sizeof(T)` bytes); `q` — queue; `deps` — prerequisites. |
| **Returns** | `sycl::event` — wait on this to ensure transfer completes. |
| **Exceptions** | None from transfer itself; SYCL may throw on submission. |

**Host → Device**

```cpp
template <typename T, int N>
sycl::event copy_to_device(const T* src, DeviceArray<T, N>& dst, sycl::queue& q,
                             const std::vector<sycl::event>& deps = {});

template <typename T, int N>
sycl::event copy_to_device(const T* src, DeviceView<T, N> dst, sycl::queue& q,
                             const std::vector<sycl::event>& deps = {});
```

| | |
|---|---|
| **Description** | Copies from a host buffer to contiguous device memory. |
| **Parameters** | `src` — host buffer (at least `dst.size() * sizeof(T)` bytes); `dst` — destination array or view; `q` — queue; `deps` — prerequisites. |
| **Returns** | `sycl::event`. |

**Device ↔ Device**

```cpp
template <typename T, int N>
sycl::event copy(const DeviceArray<T, N>& src, DeviceArray<T, N>& dst, sycl::queue& q,
                  const std::vector<sycl::event>& deps = {});
```

| | |
|---|---|
| **Description** | Device-to-device copy (host-side queue only; both arrays must be on the same device). |
| **Parameters** | `src`, `dst` — arrays (same device); `q` — queue for the source device; `deps` — prerequisites. |
| **Returns** | `sycl::event`. |
| **Precondition** | `src.device_index() == dst.device_index()`. Not asserted; mismatching devices causes undefined behavior. |

**Example**
```cpp
std::vector<float> host_buf(1024);
auto dev_arr = slabkit::make_device_array<float, 1>(
    sycl::range<1>{1024},
    ctx,
    0
);

// Copy from device to host (async)
auto ev = slabkit::copy_to_host(dev_arr, host_buf.data(), q);
ev.wait();

// Use host_buf
std::cout << host_buf[0] << "\n";
```

---

### Tier 2: Strided Host Transfers (Async, Staging)

Require a `DeviceScratchMemory` staging buffer. Use for non-contiguous device memory (e.g., subviews).

```cpp
template <typename T, int N>
sycl::event copy_to_host(DeviceSubView<T, N> src, T* dst, sycl::queue& q,
                          DeviceScratchMemory& scratch,
                          const std::vector<sycl::event>& deps = {});

template <typename T, int N>
sycl::event copy_to_host(DeviceSubView<const T, N> src, T* dst, sycl::queue& q,
                          DeviceScratchMemory& scratch,
                          const std::vector<sycl::event>& deps = {});

template <typename T, int N>
sycl::event copy_to_device(const T* src, DeviceSubView<T, N> dst, sycl::queue& q,
                             DeviceScratchMemory& scratch,
                             const std::vector<sycl::event>& deps = {});
```

| | |
|---|---|
| **Description** | Gather (device → host) or scatter (host → device) using a staging buffer. |
| **Parameters** | `src`/`dst` — `DeviceSubView`; `dst` (host) — destination buffer; `q` — queue; `scratch` — temporary device memory; `deps` — prerequisites. |
| **Returns** | `sycl::event`. |
| **Precondition** | The scratch allocation must remain valid until the returned event completes. |
| **Notes** | Uses a gather/scatter kernel + intermediate copy. Slower than contiguous transfers but necessary for strided memory. |

**Example**
```cpp
// Gather every other element
auto sub = arr.subview({0}, {512});  // Actually contiguous; for strided use
auto scratch = slabkit::make_scratch(ctx, 0, 4096);

std::vector<float> out(512);
auto ev = slabkit::copy_to_host(sub, out.data(), q, scratch);
ev.wait();
// scratch can be reused or destroyed
```

---

### Tier 3: Strided Device-to-Device Transfers (Async, No Staging)

Device-to-device strided copies using a `parallel_for` kernel.

```cpp
template <typename T, int N>
sycl::event copy(DeviceSubView<T, N> src, DeviceSubView<T, N> dst, sycl::queue& q,
                  const std::vector<sycl::event>& deps = {});

template <typename T, int N>
sycl::event copy(DeviceSubView<T, N> src, DeviceArray<T, N>& dst, sycl::queue& q,
                  const std::vector<sycl::event>& deps = {});

template <typename T, int N>
sycl::event copy(const DeviceArray<T, N>& src, DeviceSubView<T, N> dst, sycl::queue& q,
                  const std::vector<sycl::event>& deps = {});
```

| | |
|---|---|
| **Description** | Device-to-device copy with arbitrary strides; no staging or synchronization required. |
| **Returns** | `sycl::event`. |
| **Precondition** | Source and destination on the same device. |
| **Notes** | Fastest for strided transfers; uses a kernel instead of staging. |

---

### Tier 4: Blocking Sync Wrappers (Sync)

Synchronous versions of Tier 1 and 2. Block until transfer completes. No return value.

```cpp
template <typename T, int N>
void sync_to_host(const DeviceArray<T, N>& src, T* dst, sycl::queue& q);

template <typename T, int N>
void sync_to_host(DeviceView<const T, N> src, T* dst, sycl::queue& q);

template <typename T, int N>
void sync_to_host(DeviceSubView<T, N> src, T* dst, sycl::queue& q, DeviceScratchMemory& scratch);

template <typename T, int N>
void sync_to_host(DeviceSubView<const T, N> src, T* dst, sycl::queue& q, DeviceScratchMemory& scratch);
```

| | |
|---|---|
| **Description** | Synchronous transfer (calls corresponding `copy_to_host(...).wait()` internally). |
| **Returns** | Void. Returns only when transfer is complete. |
| **Notes** | Convenience for simple, single-threaded code. For concurrent work, use async versions. |

**Example**
```cpp
std::vector<float> host_buf(arr.size());
slabkit::sync_to_host(arr, host_buf.data(), q);  // Blocks until done
// host_buf is now valid
```

---

## Scratch Memory

Temporary bump allocator for device memory. Allocations are carved from a slab without a free-list. Useful for short-lived temporaries in loops.

### `DeviceScratchMemory`

**Header:** `slabkit/detail/device_stack.hpp` (exported via `slabkit/scratch.hpp`)  
**Include via:** `#include "slabkit/slabkit.hpp"`

Bump allocator over a device USM slab.

```cpp
class DeviceScratchMemory {
public:
    using Marker = std::size_t;

    // Non-owning constructor (pre-allocated slab)
    DeviceScratchMemory(void* slab_base, std::size_t capacity) noexcept;

    // Pool-owning constructor (allocates from pool; freed on destruction)
    DeviceScratchMemory(DeviceMemoryPool* pool, std::size_t scratch_size);

    // Move-only
    DeviceScratchMemory(const DeviceScratchMemory&)            = delete;
    DeviceScratchMemory& operator=(const DeviceScratchMemory&) = delete;
    DeviceScratchMemory(DeviceScratchMemory&&) noexcept;
    DeviceScratchMemory& operator=(DeviceScratchMemory&&) noexcept;
    ~DeviceScratchMemory();

    // Allocation
    void* allocate(std::size_t size, std::size_t alignment = alignof(std::max_align_t));
    // throws AllocationError if capacity exceeded

    // Markers
    Marker top() const noexcept;
    void   unwind(Marker marker) noexcept;
    void   reset() noexcept;

    // Accessors
    std::size_t capacity() const noexcept;
    std::size_t used() const noexcept;
    std::size_t remaining() const noexcept;
    bool        full() const noexcept;
    bool        empty() const noexcept;
    const std::byte* slab_base() const noexcept;
};
```

**Description**  
A simple bump allocator. Allocation is O(1): increment the bump pointer and return it. No free-list, no fragmentation. Useful for temporary allocations within a scope.

**Constructors**

**Non-owning (pre-allocated slab):**
```cpp
DeviceScratchMemory(void* slab_base, std::size_t capacity) noexcept;
```
Creates a non-owning view over an external slab. Useful if you pre-allocated device memory and want to bump-allocate from it. On destruction, does nothing (does not free the slab).

**Pool-owning (allocates from Context pool):**
```cpp
DeviceScratchMemory(DeviceMemoryPool* pool, std::size_t scratch_size);
```
Allocates a slab of size `scratch_size` from the pool. On destruction, returns the slab to the pool. Usually created via `make_scratch()`.

**Methods**

| Method | Returns | Notes |
|---|---|---|
| `allocate(size, align)` | `void*` | Allocates `size` bytes with the given alignment (default: `std::max_align_t`). Throws `AllocationError` if capacity exceeded. |
| `top()` | `Marker` (size_t) | Saves the current bump position. |
| `unwind(marker)` | — | Rewinds to a previous marker. Precondition: `marker <= current offset`. Does NOT touch device memory. |
| `reset()` | — | Rewinds to 0 (full reset). |
| `capacity()` | `size_t` | Total slab size. |
| `used()` | `size_t` | Bytes currently bumped. |
| `remaining()` | `size_t` | `capacity() - used()`. |
| `full()` | `bool` | `remaining() == 0`. |
| `empty()` | `bool` | `used() == 0`. |
| `slab_base()` | `const std::byte*` | Pointer to the start of the slab. |

**Unwind Semantics**  
`unwind()` and `reset()` adjust only the host-side bump pointer; they do **not** touch device memory. Use them to reclaim space for a new allocation. The old allocation is not freed—it's simply overwritten by future bumps.

```cpp
auto scratch = slabkit::make_scratch(ctx, 0, 4096);
auto m1 = scratch.top();  // Mark: 0
auto p1 = scratch.allocate(1000);  // Bump to 1000
auto m2 = scratch.top();  // Mark: 1000
auto p2 = scratch.allocate(1000);  // Bump to 2000
scratch.unwind(m1);  // Rewind to 0; p2 is now stale, reusable
```

**Thread Safety**  
Not thread-safe. Use one `DeviceScratchMemory` per thread, or synchronize externally.

**Move Semantics**  
Move-only. Useful for returning scratch from functions:
```cpp
DeviceScratchMemory create_scratch() {
    return slabkit::make_scratch(ctx, 0, 8192);
}
```

---

### `DeviceScratchBuffer` (Deprecated)

**Header:** `slabkit/scratch.hpp`  
**Include via:** `#include "slabkit/slabkit.hpp"`

```cpp
using DeviceScratchBuffer = DeviceScratchMemory;
```

**Status:** Deprecated. Use `DeviceScratchMemory` directly. This type alias will be removed in a future version.

---

### `DeviceScratchView`

**Header:** `slabkit/scratch.hpp`  
**Include via:** `#include "slabkit/slabkit.hpp"`

Non-owning, trivially-copyable view of a scratch allocation.

```cpp
struct DeviceScratchView {
    std::byte*  ptr_;
    std::size_t size_;

    static DeviceScratchView from_allocation(void* ptr, std::size_t size) noexcept;

    void*       data() noexcept;
    const void* data() const noexcept;
    std::size_t size() const noexcept;
    bool        empty() const noexcept;
};
```

**Description**  
A trivially-copyable struct holding a pointer and size to a scratch allocation. Safe to capture in kernels. Does not own the backing memory; the `DeviceScratchMemory` must remain valid.

**Static Factory**  
```cpp
auto alloc_ptr = scratch.allocate(256);
auto view = slabkit::DeviceScratchView::from_allocation(alloc_ptr, 256);
```

**Usage in Kernels**  
```cpp
auto view = DeviceScratchView::from_allocation(scratch.allocate(512), 512);
q.parallel_for(1000, [=](size_t i) {
    auto ptr = static_cast<float*>(view.data());
    ptr[i % 256] = i;  // Write strided
}).wait();
```

---

### `make_scratch()`

**Header:** `slabkit/scratch.hpp`  
**Include via:** `#include "slabkit/slabkit.hpp"`

Factory function to create a pool-backed `DeviceScratchMemory`.

```cpp
DeviceScratchBuffer make_scratch(std::shared_ptr<Context> ctx,
                                  std::size_t device_index,
                                  std::size_t size = 64 * 1024);

DeviceScratchBuffer make_scratch(std::shared_ptr<Context> ctx,
                                  const sycl::device& dev,
                                  std::size_t size = 64 * 1024);
```

**Description**  
Allocates `size` bytes from the device's pool and wraps it in a `DeviceScratchMemory`. On destruction, the memory is returned to the pool.

**Parameters**

| Overload | Parameter | Notes |
|---|---|---|
| Both | `ctx` | `shared_ptr<Context>`. |
| (1) | `device_index` | Device index (0-based). |
| (2) | `dev` | `sycl::device`; resolved via `ctx->device_index(dev)`. |
| Both | `size` | Allocation size in bytes. Default: 64 KB. |

**Return Value**  
`DeviceScratchBuffer` (alias for `DeviceScratchMemory`) — move-only.

**Exceptions**  
- `AllocationError` — pool's slab is exhausted.
- `std::out_of_range` / `std::invalid_argument` — invalid device.

**Example**
```cpp
auto ctx = slabkit::ContextBuilder()
    .devices({dev})
    .build();

// 4 KB scratch buffer
auto scratch = slabkit::make_scratch(ctx, 0, 4096);

auto mark = scratch.top();
auto buf = scratch.allocate(2048);
// Use buf in a kernel...
scratch.unwind(mark);  // Reclaim; buf is stale
```

---

## See Also

- **[Getting Started](getting-started.md)** — Step-by-step walkthrough of your first SlabKit program.
- **[Concepts](concepts.md)** — Ownership model, RAII semantics, view safety, and design rationale.
- **[Patterns](patterns.md)** — Best-practice usage patterns with complete examples.
- **[Anti-Patterns](anti-patterns.md)** — Common pitfalls, how to diagnose them, and how to fix them.
