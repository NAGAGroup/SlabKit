# SlabKit Conceptual Guide

This document explains the *why* behind SlabKit's design: the ownership hierarchy, the role of each component, and how memory is managed throughout the lifecycle of a program. It's written for experienced C++/SYCL developers who want to understand the architecture before building non-trivial SlabKit applications.

## Why These Concepts Matter

SlabKit introduces several key abstractions—`Context`, `DeviceMemoryPool`, `DeviceArray`, `DeviceView`, and `DeviceScratchMemory`—each with a specific purpose in the ownership chain. Understanding their roles and the relationships between them is critical before:

- Building multi-device systems where each device needs its own pool
- Managing arrays with clear, non-copyable ownership semantics
- Using scratch memory safely in hot-loop optimisations
- Debugging lifetime issues related to kernel execution and memory deallocation

This guide walks you through the design philosophy and enforces a mental model of resource ownership that prevents common pitfalls.

## The Ownership Hierarchy

SlabKit's memory management follows a strict ownership chain. Here's the architecture:

```
                       Context (shared_ptr)
                            |
                  +----------+----------+
                  |                     |
           DeviceMemoryPool    DeviceMemoryPool    ...
           (per device)        (per device)
                  |                     |
          +-----------+         +-----------+
          |           |         |           |
       DeviceArray  DeviceArray  ...
      (unique_ptr)  (unique_ptr)
          |           |
          |           |
        View       View
      (raw ptr)  (raw ptr)
```

**The rule:** Each level owns the level below it. At the top, `Context` is heap-allocated as a `shared_ptr` and owns all pools. Each pool owns device USM slabs. Each `DeviceArray` allocates nodes from a pool and holds a `unique_ptr` backed by a custom deleter that returns nodes to the pool. `DeviceView` and `DeviceSubView` own nothing—they are non-owning projections over an array's data.

This design enforces three fundamental principles:

1. **Exclusive ownership:** `DeviceArray` is move-only, ensuring only one path in the code owns the underlying memory at any moment.
2. **Automatic resource return:** When an array is destroyed, its deleter automatically returns nodes to the pool. No explicit `deallocate()` call required—it's RAII.
3. **Safe weak references:** Arrays hold `weak_ptr<Context>` in their deleters, ensuring they don't extend the Context's lifetime. If the user destroys the Context, the pools and slabs disappear, but arrays are safe to destroy afterward (their deleters simply fail silently).

## Context: The Root of Ownership

### Heap Allocation and Shared Ownership

`Context` is **always** heap-allocated as a `std::shared_ptr<Context>`. It is never stack-allocated. This is because:

- The `Context` derives from `std::enable_shared_from_this<Context>`, enabling `shared_from_this()` in pool lazy-init to pass `shared_ptr` to its own pools without a separate retain.
- Arrays allocated from the Context need a stable `shared_ptr` to it, and the pool deleter holds a `weak_ptr` derived from that `shared_ptr`.
- SYCL contexts are expensive to create—sharing a single logical Context across multiple pools (one per device) deduplicates that cost.

You never construct a `Context` directly. Instead, use `ContextBuilder`:

```cpp
auto ctx = slabkit::ContextBuilder()
    .devices({dev1, dev2})
    .node_size(1024)
    .slab_size_per_device(64 * 1024 * 1024)
    .build();
```

### Platform Grouping and SYCL Context Creation

`ContextBuilder::build()` automatically groups the provided devices by platform (backend + platform name). For each unique platform group, it creates a single `sycl::context`. You do not create `sycl::context` objects yourself; the builder handles it.

**Why:** SYCL contexts are expensive resources tied to specific hardware backends. Grouping devices by platform ensures we create exactly one SYCL context per backend, reusing it for all devices in that group. This minimises SYCL context allocation and enables cross-device operations within the same platform.

### Configuration: `node_size` and `slab_size_per_device`

Two parameters control pool granularity:

- **`node_size`** (default: 1024 bytes) — the allocation granularity. Every allocation is rounded up to a multiple of this size. Set this based on the smallest meaningful allocation in your application. A tiny value (e.g., 64) wastes memory; a large value (e.g., 1 MB) fragments fragmentation. For kernels allocating many small arrays, use a small node size; for a few large tensors, a larger one is fine.

- **`slab_size_per_device`** (default: 64 MB) — the total size of the USM slab owned by the pool for each device. Once this slab is full, further allocations fail. Choose based on your device's capacity and working set size. Multiple devices can have different slab sizes if needed (though SlabKit currently uses a uniform size).

**Example:** If you have 8 GB of device memory and expect to use 4 GB concurrently, set `slab_size_per_device(4 * 1024 * 1024 * 1024)`. If your smallest allocations are arrays of 512 bytes, you might use `node_size(512)` or `node_size(1024)` to avoid over-fragmentation.

### No Global State

SlabKit has **no global singleton** or static Context. Multiple independent Contexts can coexist in the same program. Each Context manages its own set of devices and pools. This enables:

- Running different components with different device configurations
- Unit testing with mock contexts
- Graceful per-context resource cleanup when a Context is destroyed

### Pool Access

`Context` provides two ways to access a pool:

```cpp
// By device index (pre-resolved)
auto& pool = ctx->pool_at(0);

// By sycl::device object
auto& pool = ctx->pool_for(dev);
```

Internally, pools are lazily initialised on first access using double-checked locking (mutex-protected init, allocation itself is not thread-safe — see Thread Safety section).

### Queue Creation

`Context` provides helpers to create queues associated with its managed SYCL contexts:

```cpp
sycl::queue in_order = ctx->make_in_order_queue(dev);
sycl::queue out_of_order = ctx->make_out_of_order_queue(dev);
sycl::queue custom = ctx->make_queue_with_props(dev, property_list);
```

These return queues linked to the SYCL context that was created for the device's platform. You are not restricted to these helpers—you can create your own queues with a Context's `sycl::context` obtained via `ctx->context_for(dev)`—but using the provided helpers ensures consistency.

## DeviceArray: Move-Only Ownership and RAII

### Move-Only Semantics

`DeviceArray<T, N>` is move-only. You cannot copy it:

```cpp
auto arr1 = slabkit::make_device_array<float, 1>(range, ctx, 0);
// auto arr2 = arr1;                    // Compilation error: copy deleted
auto arr2 = std::move(arr1);            // OK: move allowed
// arr1 is now empty; arr2 owns the memory
```

**Why move-only?** A `DeviceArray` wraps a USM device pointer, which is the exclusive right to a contiguous chunk of device memory. Copying the pointer would create two independent owners of the same memory—a nightmare for RAII. The move-only design enforces that only one path owns the memory at any time. When you move an array, you're transferring that exclusive ownership; the source is left empty (null).

### Implementation: Unique Pointer with Custom Deleter

Under the hood, `DeviceArray` is backed by a `std::unique_ptr<T, DeviceUSMPoolDeleter>`. The deleter is the magic:

```cpp
struct DeviceUSMPoolDeleter {
    std::weak_ptr<Context> ctx;
    size_t device_index;
    size_t node_count;
    
    void operator()(void* ptr) const noexcept {
        if (auto c = ctx.lock())
            c->pool_at(device_index).deallocate_array(ptr, node_count);
        // if expired: Context destroyed, memory was freed with it — no-op
    }
};
```

When the array's `unique_ptr` is destroyed, the deleter is invoked. It locks the `weak_ptr<Context>`. If the Context is still alive, it returns the nodes to the pool. If the Context has already been destroyed, the lock fails silently—the memory was freed with the slab anyway.

**Why `weak_ptr` and not `shared_ptr`?** If the deleter held a `shared_ptr`, destroying the last array would keep the Context alive until the deleter ran, creating a reference cycle (Context owns pools, arrays hold Context). With `weak_ptr`, arrays do not extend the Context's lifetime. If all user code releases the Context, it is destroyed immediately; any existing arrays simply become inert (their deleters are no-ops). This is the correct ownership model: the user, not the arrays, controls Context lifetime.

### Lazy Allocation from Pools

`DeviceArray` does not allocate memory directly. Instead, use the factory function:

```cpp
auto arr = slabkit::make_device_array<T, N>(range, ctx, device_index);
```

This calls `DeviceArray::from_pool()`, which:

1. Looks up the pool via `ctx->pool_at(device_index)`
2. Computes how many nodes are needed (`ceil(count * sizeof(T) / node_size)`)
3. Calls `pool.allocate_array(node_count)` to get a contiguous raw pointer
4. Wraps the pointer in a `unique_ptr` with the custom deleter
5. Returns the constructed `DeviceArray`

There is no separate pool object returned to the user. The Context owns all pools; the factory just orchestrates the allocation.

### The RAII Principle

`DeviceArray` exemplifies RAII: memory is acquired in the constructor (implicit, via `from_pool`), and released in the destructor (via the deleter's pool return). The user allocates once, uses the array, and the memory is automatically returned when the array goes out of scope or is moved away. No manual `deallocate()` call required.

### Array Methods

Core accessors:

```cpp
T*             arr.data()        // raw device pointer
sycl::range<N> arr.get_range()   // array dimensions
size_t         arr.size()        // element count
size_t         arr.byte_size()   // byte count
size_t         arr.device_index() // which device owns this?
```

View projection:

```cpp
DeviceView<T, N>       arr.view()        // mutable view
DeviceView<const T, N> arr.const_view()  // const view

// Implicit conversion to view
DeviceView<T, N> v = arr;  // allowed; converts arr to view
```

Sub-views:

```cpp
// Strided sub-array: offset + sub-shape
auto subview = arr.subview(
    std::array<size_t, 2>{0, 10},    // start indices
    sycl::range<2>{4, 8}              // sub-shape
);  // DeviceSubView<T, 2> with strides computed
```

## Views: Non-Owning Projections

### DeviceView: Contiguous

`DeviceView<T, N>` is a lightweight, non-owning struct:

```cpp
template <typename T, int N>
struct DeviceView {
    T*             ptr_;     // raw pointer
    sycl::range<N> range_;   // shape
};
```

A `DeviceView` is trivially copyable and trivially constructible. You can capture it by value in SYCL kernel lambdas without any special machinery:

```cpp
auto view = arr.view();
auto ev = q.parallel_for(arr.get_range(), [=](sycl::id<N> id) {
    view[id] = compute_something();  // Works! view is captured by value.
});
```

**Row-major indexing:** `view[id]` and `view(i, j, ...)` use row-major (C-order) semantics. For a 2D array with shape (M, N), element at row `i`, column `j` is at linear offset `i*N + j`.

### DeviceSubView: Strided

`DeviceSubView<T, N>` extends `DeviceView` with explicit strides:

```cpp
template <typename T, int N>
struct DeviceSubView {
    T*             ptr_;      // base pointer
    sycl::range<N> range_;    // logical shape
    sycl::range<N> strides_;  // element strides (not byte strides!)
};
```

A sub-view can represent non-contiguous slices—for example, every other row of a 2D matrix. Like `DeviceView`, it's trivially copyable and safe to capture in kernels.

**Strided indexing:** `subview[id]` computes `ptr_[sum(id[i] * strides_[i])]`. The strides are in *elements*, not bytes, allowing the arithmetic to be dimension-agnostic.

### Views Do Not Extend Array Lifetime

**Critical invariant:** Views hold raw pointers. They do not hold a reference to the backing `DeviceArray`. If the array is destroyed while a view is in use by a kernel, the result is undefined behavior:

```cpp
auto arr = slabkit::make_device_array<float, 1>(range, ctx, 0);
auto view = arr.view();
std::unique_ptr<DeviceArray<float, 1>> arr_ptr(std::make_unique<DeviceArray<float, 1>>(std::move(arr)));

// arr is now an empty shell; arr_ptr owns the memory
// ... 
q.parallel_for(range, [=](sycl::id<1> id) { view[id] = 1.f; });  // safe
arr_ptr.reset();  // destroy array, return memory to pool

// Now the kernel might run with view pointing to freed memory — UB!
```

**The user is responsible** for ensuring the `DeviceArray` outlives all kernels using its views. This is a consequence of raw-pointer views: no lifetime extension, no overhead, but discipline required.

In practice, keep the array and all views in scope until all work completes:

```cpp
auto arr = slabkit::make_device_array<float, 1>(range, ctx, 0);
auto view = arr.view();
auto ev = q.parallel_for(range, [=](sycl::id<1> id) { view[id] = 1.f; });
ev.wait();  // wait before arr goes out of scope
```

### Implicit Conversion

`DeviceArray<T, N>` implicitly converts to `DeviceView<T, N>`:

```cpp
void kernel_wrapper(DeviceView<float, 1> view) { /* ... */ }
auto arr = slabkit::make_device_array<float, 1>(range, ctx, 0);
kernel_wrapper(arr);  // allowed! arr → DeviceView<float, 1>
```

This is safe because the conversion is just a projection; the array's lifetime is unaffected.

## MirroredArray: Paired Host and Device Storage

`MirroredArray<T, N>` owns both a host (`std::vector<T>`) and device (`DeviceArray<T, N>`) allocation for the same logical data:

```cpp
auto mirrored = slabkit::MirroredArray<float, 2>(
    sycl::range<2>{1024, 1024},
    ctx,
    0  // device index
);
```

### Ownership Model

`MirroredArray` is move-only (because `DeviceArray` is move-only). It owns:

- A `std::vector<T>` for host-side data (CPU-accessible, zero-initialized by default)
- A `DeviceArray<T, N>` on the device (uninitialized)

The host and device sides are **not automatically synchronised**. You must explicitly sync:

```cpp
// Host → Device
auto ev_h2d = mirrored.sync_to_device(queue, {dep1, dep2});

// Device → Host
auto ev_d2h = mirrored.sync_to_host(queue, {dep1, dep2});

ev_d2h.wait();  // wait for sync to complete
```

Both methods are async and return `sycl::event`.

### Host Access

Retrieve a `std::span` to the host side:

```cpp
auto host_span = mirrored.host_data();  // std::span<T>
for (size_t i = 0; i < host_span.size(); ++i)
    host_span[i] = i;
```

### Deep Copying

Clone a `MirroredArray`:

```cpp
auto original = slabkit::MirroredArray<float, 1>(range, ctx, 0);
// ... populate original.host_data() ...

auto copy = original.clone();  // deep copy of host vector, new device allocation
```

The clone's device side is **uninitialized**. Call `sync_to_device` to populate it:

```cpp
auto ev = copy.sync_to_device(queue);
ev.wait();
```

### When to Use

Use `MirroredArray` when:

- You have a natural host–device pair for an allocation
- You want a single owner for both sides
- You frequently synchronise between host and device

For device-only allocations, use `DeviceArray` directly—it's lighter and avoids unnecessary host storage.

## DeviceScratchMemory: Bump Allocation for Temporaries

`DeviceScratchMemory` is a bump allocator over a device USM slab. It's designed for functions called in hot loops that need temporary device buffers without the overhead of pool allocation/deallocation on every call.

### Creating Scratch Memory

```cpp
auto scratch = slabkit::make_scratch(ctx, device_index, 64 * 1024);  // 64 KB
```

This allocates a 64 KB slab from the Context's pool for the device. When the `DeviceScratchMemory` is destroyed, the slab is returned to the pool.

### Allocation and Unwinding

The core API is simple:

```cpp
// Bump-allocate size bytes, aligned to alignment
void* ptr = scratch.allocate(size, alignment);

// Save current position
auto marker = scratch.top();

// Rewind to saved position (does NOT touch device memory)
scratch.unwind(marker);

// Reset to beginning
scratch.reset();
```

**Key invariant:** `unwind()` and `reset()` only adjust the host-side bump pointer. They do NOT touch device memory. If a kernel is in-flight using a pointer allocated before unwinding, that pointer remains valid until the kernel completes—the memory is not freed, just logically reclaimed for future allocations.

### The Canonical Pattern

Save-unwind is the canonical pattern for hot-loop temporary memory:

```cpp
auto scratch = slabkit::make_scratch(ctx, 0, 8192);

for (int iter = 0; iter < 1000; ++iter) {
    auto mark = scratch.top();
    
    // Allocate temps
    float* tmp1 = static_cast<float*>(scratch.allocate(1024 * sizeof(float), alignof(float)));
    float* tmp2 = static_cast<float*>(scratch.allocate(512 * sizeof(float), alignof(float)));
    
    // Use in kernel
    q.parallel_for(range1, [=](sycl::id<1> id) { tmp1[id.get(0)] = compute(); });
    
    // Unwind: reclaim space for next iteration
    scratch.unwind(mark);
}
```

Inside nested function calls, reuse the same `DeviceScratchMemory`:

```cpp
void kernel_step(slabkit::DeviceScratchMemory& scratch, sycl::queue& q) {
    // Bump-allocate a temporary
    float* temp = static_cast<float*>(
        scratch.allocate(256 * sizeof(float), alignof(float))
    );
    
    // Use temp in kernel
    q.parallel_for(sycl::range<1>{256}, [=](sycl::id<1> id) {
        temp[id.get(0)] = id.get(0) * 2.f;
    });
    
    // No deallocation — caller unwinds when done
}

// Usage
auto scratch = slabkit::make_scratch(ctx, 0, 4096);
auto m1 = scratch.top();
kernel_step(scratch, q);
scratch.unwind(m1);  // reclaims all allocations inside kernel_step
```

### Move-Only

`DeviceScratchMemory` is move-only. You cannot copy it. When moved, ownership of the slab transfers to the target object; the source is left with no slab.

### Thread Safety

`DeviceScratchMemory` is **not thread-safe**. If multiple threads allocate and unwind concurrently, the bump pointer gets corrupted. Either:

- Create a separate `DeviceScratchMemory` per thread
- Use external synchronisation (mutex around allocations)

For single-threaded use, no synchronisation is needed.

## Thread Safety

Be explicit about what is and isn't thread-safe:

- **Context construction:** Creating or building a `Context` is not thread-safe. Build your Context once on the main thread before spawning threads. Accessing a `Context` after construction (pool lookups, queue creation) is safe via internal mutex protection.

- **Pool allocation:** `make_device_array` and `make_scratch` may trigger lazy pool initialisation on first call, which is mutex-protected (double-checked locking). Once a pool is initialised, all `allocate_array` and `deallocate_array` calls are **thread-safe** — `DeviceMemoryPool` holds an internal mutex that protects every allocation and deallocation.

- **DeviceArray / MirroredArray:** Accessing an array from multiple threads simultaneously (e.g., modifying host data) is not thread-safe. Arrays are not atomic; synchronise at a higher level.

- **DeviceView / DeviceSubView:** Views are plain structs with no mutable state. Read-only concurrent access (e.g., multiple threads reading the same view's pointer and shape) is safe. However, writing to the underlying device memory via the view from multiple threads/queues follows SYCL queue semantics, not SlabKit's—the user is responsible for queue synchronisation.

- **Multiple independent Contexts:** Different `Context` objects are fully independent. Using separate Contexts from different threads is safe.

**Future roadmap:** Fine-grained pool locking (per-pool-per-device mutexes instead of global pool mutex) is planned for v0.2 to reduce contention in multi-threaded allocation-heavy workloads.

## Design Rationale Summary

Why is SlabKit shaped this way?

1. **Hierarchy:** Ownership flows top-down (Context → pools → arrays → views). Each level owns what it needs; nothing owns up.

2. **Move-only semantics:** `DeviceArray` is move-only to enforce exclusive ownership. This makes lifetime issues explicit and prevents silent bugs from accidental copies.

3. **RAII and automatic pool return:** Arrays return memory to pools automatically on destruction. The pool reuses memory efficiently without expensive SYCL `free` calls.

4. **Weak references:** Pool deleters hold `weak_ptr<Context>`, so arrays don't extend the Context's lifetime. The user controls when the Context (and all pools) are destroyed.

5. **Trivially copyable views:** Views are plain structs that can be captured in kernel lambdas by value with zero overhead. They carry no ownership—just shape and a pointer.

6. **Bump allocation for temporaries:** `DeviceScratchMemory` avoids pool overhead for short-lived allocations in inner loops. Unwinding doesn't touch device memory, so in-flight kernels remain safe.

7. **No global state:** Multiple independent `Context` objects can coexist. This simplifies testing, composition, and resource cleanup.

These choices prioritise safety (move-only, RAII), performance (pool reuse, bump allocation), and clarity (explicit ownership hierarchy).

## See Also

- **[getting-started.md](getting-started.md)** — Tutorial introduction with code examples
- **[api-reference.md](api-reference.md)** — Complete API documentation
- **[patterns.md](patterns.md)** — Best practices and idiomatic usage
- **[anti-patterns.md](anti-patterns.md)** — Common mistakes to avoid
