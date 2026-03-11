# Getting Started with SlabKit

This guide walks you through the essentials of SlabKit: building your first Context, allocating memory, and running your first kernel. It assumes you're comfortable with C++, SYCL concepts like `sycl::queue` and USM, and kernel basics.

## Your First Context

A **Context** is SlabKit's central abstraction. It owns the SYCL devices you'll use, manages one or more SYCL contexts (one per platform), and—most importantly—owns the memory pools for allocation. When you allocate a `DeviceArray`, the array doesn't talk directly to SYCL; instead, it allocates from the Context's pool.

Build a Context using `ContextBuilder`:

```cpp
#include "slabkit/slabkit.hpp"

// Create a CPU device
sycl::device dev{sycl::cpu_selector_v};

// Build a Context for that device
auto ctx = slabkit::ContextBuilder()
    .devices({dev})
    .node_size(1024)                    // granularity of pool allocations
    .slab_size_per_device(64 * 1024 * 1024)  // 64 MB per device
    .build();
```

`build()` returns a `std::shared_ptr<Context>`. This shared ownership is intentional: the Context stays alive as long as any arrays allocated from it need their memory. When you create a `DeviceArray`, it holds a reference to the Context to ensure the pool (and thus the memory) remains valid.

## Allocating a DeviceArray

Once you have a Context, allocate a device array using `make_device_array`:

```cpp
// Allocate 1024 floats on device 0
auto arr = slabkit::make_device_array<float, 1>(
    sycl::range<1>{1024},
    ctx,
    0  // device index
);
```

This carves out a node (or chain of nodes) from the Context's pool for device 0, wrapping it in a `DeviceArray<float, 1>` that owns that memory via a `unique_ptr` with a custom deleter. The array is **move-only**: you cannot copy it, only move it. This enforces clear ownership semantics.

When the array goes out of scope, its destructor returns the node to the pool instead of freeing it. That's **RAII for pool memory**: allocate on construction, deallocate (return to pool) on destruction. The pool reuses that node for future allocations, avoiding expensive device memory frees.

For host + device pairs, use `MirroredArray<T, N>`, which owns both a `std::vector<T>` on the host and a `DeviceArray<T, N>` on the device, with helper methods to sync between them. (See [concepts.md](concepts.md) for details on when and how to use it.)

## Views and Kernels

To use an array in a kernel, get a `DeviceView` via `.view()`:

```cpp
auto view = arr.view();  // DeviceView<float, 1>
```

A **DeviceView** is non-owning, trivially copyable struct holding a raw pointer and shape. Because it's trivially copyable, you can capture it directly in a SYCL kernel lambda by value. Do **not** capture the `DeviceArray` by value—it holds a `unique_ptr` and is not trivially copyable.

With USM, device memory is just a pointer—no accessor machinery or buffer synchronisation. Here's a minimal kernel that squares each element:

```cpp
sycl::queue q = ctx->make_in_order_queue(dev);
auto view = arr.view();  // DeviceView<float, 1> — trivially copyable

// USM pointers are plain pointers — capture by value and use directly
auto ev = q.parallel_for(arr.get_range(), [=](sycl::id<1> id) {
    view[id] *= view[id];
});
ev.wait();
```

Capture the view by value in the lambda; do not capture the `DeviceArray` itself (it's move-only and not trivially copyable).

**Critical lifetime rule:** The `DeviceArray` must outlive all queued work that uses its view. In the example above, `arr` is still in scope when the kernel runs. If you exit the function before the queue drains, the array is destroyed and its memory returned to the pool while the kernel is still running—undefined behavior. Always wait for work to complete before the array goes out of scope.

For a deeper dive on ownership, RAII, and safe view patterns, see [concepts.md](concepts.md).

## Scratch Memory

`DeviceScratchMemory` is a bump allocator over a device USM slab. It's useful for functions called repeatedly that need temporary device memory without the cost of pool allocation/deallocation on every call. The caller allocates a `DeviceScratchMemory` once (or per call-site scope), bumps into it for temporaries, then rewinds.

Create a scratch buffer using `make_scratch`:

```cpp
auto scratch = slabkit::make_scratch(ctx, 0, 2048);  // 2048 bytes for device 0
```

The core API:
- `allocate(size, alignment)` — returns a raw device pointer (do not dereference from host)
- `top()` — saves the current bump position as a `Marker`
- `unwind(marker)` — rewinds the bump pointer to a saved marker (does NOT touch device memory—purely adjusts the host-side offset)
- `reset()` — equivalent to `unwind(0)`, full reset

The canonical pattern is scope-based unwinding: save the position before a call that needs temporary memory, then restore afterward:

```cpp
auto scratch = slabkit::make_scratch(ctx, 0, 8192);

// First call
auto mark1 = scratch.top();
do_some_work(scratch /*, other args */);
scratch.unwind(mark1);  // reclaim all memory allocated inside do_some_work

// Second call — reuses the freed space
auto mark2 = scratch.top();
do_other_work(scratch /*, other args */);
scratch.unwind(mark2);
```

Inside `do_some_work`, you might allocate a temporary buffer:

```cpp
void do_some_work(slabkit::DeviceScratchMemory& scratch /*, ... */) {
    // Bump-allocate a temporary device buffer
    float* tmp = static_cast<float*>(
        scratch.allocate(256 * sizeof(float), alignof(float))
    );

    // Use tmp in a kernel via q.parallel_for ...
    // (tmp is a device pointer — pass it to kernels, do not dereference from host)

    // No explicit deallocation needed — the caller unwinds when done
}
```

Key points:
- `DeviceScratchMemory` is **move-only** (not copyable)
- Device pointers returned by `allocate()` are opaque from the host—pass them into kernels, never dereference them
- `unwind()` does not touch device memory—it only moves the host-side bump pointer. Any existing device-side work using old pointers remains valid until that work completes
- Thread safety: none—external synchronisation required

## Host ↔ Device Transfers

SlabKit provides async copy functions in `<slabkit/memory.hpp>` (included via the umbrella header):

```cpp
// Copy from host to device
sycl::event ev_h2d = slabkit::copy_to_device(
    host_ptr,  // const T*
    arr,       // DeviceArray<T, N>&
    q          // sycl::queue&
);

// Copy from device back to host
sycl::event ev_d2h = slabkit::copy_to_host(
    arr,       // const DeviceArray<T, N>&
    host_ptr,  // T*
    q,         // sycl::queue&
    {ev_h2d}   // optional: vector of dependency events
);

ev_d2h.wait();  // wait for transfer to complete
```

Both functions are asynchronous and return `sycl::event`. You can chain them with dependencies (as shown) or wait explicitly.

Here's a complete round-trip example:

```cpp
// Prepare host data
std::vector<float> host_data(1024);
std::iota(host_data.begin(), host_data.end(), 0.f);

// Allocate device array
auto arr = slabkit::make_device_array<float, 1>(
    sycl::range<1>{1024}, ctx, 0);

// Copy to device
auto ev_h2d = slabkit::copy_to_device(host_data.data(), arr, q);

// Get a view before submitting — DeviceView is trivially copyable
auto view = arr.view();

// Run kernel (depends on copy completing)
auto ev_kernel = q.parallel_for(arr.get_range(), [=](sycl::id<1> id) {
    view[id] *= 2.f;
}, ev_h2d);  // dependency on copy

// Copy result back
std::vector<float> host_result(1024);
auto ev_d2h = slabkit::copy_to_host(arr, host_result.data(), q, {ev_kernel});

ev_d2h.wait();  // wait for all work to complete
```

## What's Next

- **[concepts.md](concepts.md)** — Deep dive into ownership, RAII semantics, and safe view patterns. Read this before working with complex allocation patterns or multi-device scenarios.

- **[api-reference.md](api-reference.md)** — Full type and function signatures for `Context`, `ContextBuilder`, `DeviceArray`, `DeviceView`, transfer functions, and more.

- **[patterns.md](patterns.md)** — Best practices: when to use `DeviceArray` vs. `MirroredArray`, how to batch kernels, event dependency strategies, and multi-device allocation.

- **[anti-patterns.md](anti-patterns.md)** — Common mistakes: capturing arrays in kernels, dangling views, forgetting to wait, mixing pool and non-pool memory. Learn what to avoid.
