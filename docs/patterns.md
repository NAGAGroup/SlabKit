# SlabKit Patterns

This guide presents best-practice usage patterns for SlabKit. Each pattern shows the idiomatic way to accomplish a common task—with a complete, compilable code example and an explanation of *why* the approach is correct. This is not a tutorial; it assumes you've read [Getting Started](getting-started.md) and [Concepts](concepts.md). For API details, see [API Reference](api-reference.md).

---

## Pattern 1: Build Context Once, Share via shared_ptr

Create a `Context` once at startup and pass its `shared_ptr` throughout your program—across functions, threads, and subsystems. Never stack-allocate a Context.

```cpp
#include "slabkit/slabkit.hpp"
#include <iostream>
#include <vector>
#include <thread>

std::shared_ptr<naga::slabkit::Context> build_context() {
    auto dev = sycl::device{sycl::gpu_selector_v};
    return naga::slabkit::ContextBuilder()
        .devices({dev})
        .node_size(1024)
        .slab_size_per_device(256 * 1024 * 1024)
        .build();
}

void worker_thread(std::shared_ptr<naga::slabkit::Context> ctx) {
    // Each thread receives a shared_ptr to the same Context
    auto dev = ctx->devices[0];
    sycl::queue q = ctx->make_in_order_queue(dev);
    auto arr = naga::slabkit::make_device_array<float, 1>(
        sycl::range<1>{1024}, ctx, 0);
    // Use arr in kernels...
}

int main() {
    auto ctx = build_context();  // Heap-allocated shared_ptr
    
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back(worker_thread, ctx);  // Pass shared_ptr
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // Context destroyed here; all arrays using it are also destroyed
    return 0;
}
```

**Why:** `Context` owns the pools and SYCL contexts. Creating it once and sharing the `shared_ptr` ensures all arrays allocate from the same pools, and the Context stays alive as long as any arrays reference it. Stack-allocating Context is undefined behavior—arrays hold `weak_ptr` to the Context, and if the Context goes out of scope, the pools disappear while arrays are still alive. Sharing via `shared_ptr` allows threads and functions to participate in ownership, keeping the Context alive until the last reference is dropped.

---

## Pattern 2: Use `.view()` to Pass Arrays into Kernels

`DeviceArray` is move-only and non-copyable. Always extract a `DeviceView` via `.view()` before capturing in a kernel lambda.

```cpp
#include "slabkit/slabkit.hpp"
#include <sycl/sycl.hpp>

int main() {
    auto ctx = naga::slabkit::ContextBuilder()
        .devices({sycl::device{sycl::cpu_selector_v}})
        .build();
    
    sycl::queue q = ctx->make_in_order_queue(ctx->devices[0]);
    
    // Allocate a 2D array
    auto arr = naga::slabkit::make_device_array<float, 2>(
        sycl::range<2>{512, 512}, ctx, 0);
    
    // Extract a view BEFORE the lambda
    auto view = arr.view();  // DeviceView<float, 2> — trivially copyable
    
    // Capture the view by value, NOT the array
    q.parallel_for(view.get_range(), [=](sycl::id<2> id) {
        view[id] = id[0] + id[1];  // Capture by value is safe
    }).wait();
    
    return 0;
}
```

**Why:** `DeviceArray` holds a `unique_ptr` with a custom deleter; it is not trivially copyable. Capturing it by value in a lambda would fail to compile. `DeviceView` is a struct containing only a raw pointer and a shape—trivially copyable and safe to capture by value. The view is non-owning; the array must outlive all queued work, but the view itself is just a temporary projection.

---

## Pattern 3: Accept DeviceView in Functions, Not DeviceArray

When writing a function that runs a kernel, take a `DeviceView<T, N>` parameter—not a `DeviceArray&`. This decouples kernel code from ownership concerns.

```cpp
#include "slabkit/slabkit.hpp"
#include <sycl/sycl.hpp>

// Generic kernel function: accepts a view, not the array
void scale_array(naga::slabkit::DeviceView<float, 1> view, float factor, sycl::queue& q) {
    q.parallel_for(view.get_range(), [=](sycl::id<1> id) {
        view[id] *= factor;
    }).wait();
}

int main() {
    auto ctx = naga::slabkit::ContextBuilder()
        .devices({sycl::device{sycl::cpu_selector_v}})
        .build();
    
    auto dev = ctx->devices[0];
    sycl::queue q = ctx->make_in_order_queue(dev);
    
    auto arr = naga::slabkit::make_device_array<float, 1>(
        sycl::range<1>{1000}, ctx, 0);
    
    // Call with explicit view
    scale_array(arr.view(), 2.0f, q);
    
    // Or use implicit conversion (operator DeviceView<float, 1>)
    scale_array(arr, 3.0f, q);
    
    return 0;
}
```

**Why:** Functions that accept `DeviceView` are purely computational—they don't care who owns the array or which device it's on. This allows the same function to work with arrays, subviews, or even arrays from different contexts (as long as they're the same rank and type). The implicit conversion from `DeviceArray` to `DeviceView` means you can call the function with either, but the function signature makes the ownership constraint explicit: the caller retains ownership.

---

## Pattern 4: Use DeviceViewLike for Generic Kernels

When writing generic algorithms that work on both `DeviceView` and `DeviceSubView` (e.g., strided data), use the `DeviceViewLike` concept.

```cpp
#include "slabkit/slabkit.hpp"
#include <sycl/sycl.hpp>
#include <concepts>

// Generic algorithm that works on any view-like type
template <naga::slabkit::DeviceViewLike<float, 2> V>
void apply_2d(V view, float value, sycl::queue& q) {
    q.parallel_for(view.get_range(), [=](sycl::id<2> id) {
        view[id] = value;
    }).wait();
}

int main() {
    auto ctx = naga::slabkit::ContextBuilder()
        .devices({sycl::device{sycl::cpu_selector_v}})
        .build();
    
    sycl::queue q = ctx->make_in_order_queue(ctx->devices[0]);
    
    // Allocate and fill a full array
    auto arr = naga::slabkit::make_device_array<float, 2>(
        sycl::range<2>{1024, 1024}, ctx, 0);
    apply_2d(arr.view(), 1.0f, q);
    
    // Also works on subviews (strided or not)
    auto sub = arr.subview({0, 0}, {512, 512});
    apply_2d(sub, 2.0f, q);
    
    return 0;
}
```

**Why:** `DeviceViewLike<T, N>` is a concept that both `DeviceView<T, N>` and `DeviceSubView<T, N>` satisfy. It requires only `get_range()`, `size()`, `byte_size()`, and `operator[]`—no `data()` or contiguity guarantees. This allows functions to work on strided data (from subviews) without knowing whether the underlying memory is contiguous. `DeviceSubView` does not satisfy stricter concepts that require contiguity, but it satisfies `DeviceViewLike`.

---

## Pattern 5: Scratch Memory—Save/Unwind for Loop-Iteration Temporaries

Allocate scratch memory once outside a loop, and use save/unwind per iteration to reuse the same slab for multiple temporary allocations.

```cpp
#include "slabkit/slabkit.hpp"
#include <sycl/sycl.hpp>
#include <vector>

int main() {
    auto ctx = naga::slabkit::ContextBuilder()
        .devices({sycl::device{sycl::cpu_selector_v}})
        .build();
    
    sycl::queue q = ctx->make_in_order_queue(ctx->devices[0]);
    
    // Allocate scratch once outside the loop
    auto scratch = naga::slabkit::make_scratch(ctx, 0, 64 * 1024);  // 64 KB
    
    std::vector<naga::slabkit::DeviceArray<float, 1>> results;
    
    for (int i = 0; i < 10; ++i) {
        // Save the current bump position
        auto mark = scratch.top();
        
        // Allocate temporary buffer from scratch
        float* tmp = static_cast<float*>(
            scratch.allocate(512 * sizeof(float), alignof(float)));
        
        // Use tmp in a kernel
        q.parallel_for(sycl::range<1>{512}, [=](sycl::id<1> id) {
            tmp[id.get(0)] = i + id.get(0);
        }).wait();
        
        // Allocate the iteration result
        auto result = naga::slabkit::make_device_array<float, 1>(
            sycl::range<1>{512}, ctx, 0);
        auto result_view = result.view();
        
        // Copy from temporary to result
        q.parallel_for(sycl::range<1>{512}, [=](sycl::id<1> id) {
            result_view[id] = tmp[id.get(0)] * 2.0f;
        }).wait();
        
        results.push_back(std::move(result));
        
        // Unwind to reclaim tmp for next iteration
        scratch.unwind(mark);
    }
    
    return 0;
}
```

**Why:** `unwind()` only adjusts the host-side bump pointer—it does not touch device memory or synchronize with the device. All queued work using the old temporary pointers remains valid until that work completes. By unwinding after each iteration, you reuse the same 64 KB slab 10 times, avoiding the cost of allocating and deallocating 10 separate temporary buffers. This is essential for hot-loop optimizations.

---

## Pattern 6: Use MirroredArray for Host-Initialized Data

When you have data computed on the host that needs to be transferred to the device for processing, `MirroredArray` provides a clean, synchronized pattern.

```cpp
#include "slabkit/slabkit.hpp"
#include <sycl/sycl.hpp>
#include <numeric>
#include <iostream>

int main() {
    auto ctx = naga::slabkit::ContextBuilder()
        .devices({sycl::device{sycl::cpu_selector_v}})
        .build();
    
    sycl::queue q = ctx->make_in_order_queue(ctx->devices[0]);
    
    // Allocate paired host and device arrays
    naga::slabkit::MirroredArray<float, 1> arr(
        sycl::range<1>{1024}, ctx, 0);
    
    // Initialize on host
    auto host_span = arr.host_data();
    std::iota(host_span.begin(), host_span.end(), 0.0f);
    
    // Transfer to device
    arr.sync_to_device(q).wait();
    
    // Process on device
    auto view = arr.view();
    q.parallel_for(view.get_range(), [=](sycl::id<1> id) {
        view[id] *= 2.0f;
    }).wait();
    
    // Transfer back to host
    arr.sync_to_host(q).wait();
    
    // Read results on host
    host_span = arr.host_data();
    std::cout << "First element: " << host_span[0] << "\n";  // 0 (0*2)
    std::cout << "Second element: " << host_span[1] << "\n";  // 2 (1*2)
    
    return 0;
}
```

**Why:** `MirroredArray` owns both a `std::vector<T>` (host) and a `DeviceArray<T, N>` (device). The host side is zero-initialized on construction; the device side is uninitialized. This pattern avoids the boilerplate of manually allocating and managing separate host/device buffers and calling `copy_to_device`/`copy_to_host` separately. The `sync_to_device()` and `sync_to_host()` methods return events for dependency tracking, and the pattern naturally expresses the intent: "keep a host and device copy in sync."

---

## Pattern 7: Chain Async Transfers with In-Order Queue

When overlapping compute and transfer, use an in-order queue to guarantee sequential execution, or use the event-based `deps` parameter to manually chain work.

```cpp
#include "slabkit/slabkit.hpp"
#include <sycl/sycl.hpp>
#include <vector>

int main() {
    auto ctx = naga::slabkit::ContextBuilder()
        .devices({sycl::device{sycl::cpu_selector_v}})
        .build();
    
    auto dev = ctx->devices[0];
    
    // Create an IN-ORDER queue (guarantees sequential execution)
    sycl::queue q = ctx->make_in_order_queue(dev);
    
    // Host-side data
    std::vector<float> host_input(1024);
    std::vector<float> host_output(1024);
    for (int i = 0; i < 1024; ++i) {
        host_input[i] = i;
    }
    
    // Device-side arrays
    auto input = naga::slabkit::make_device_array<float, 1>(
        sycl::range<1>{1024}, ctx, 0);
    auto output = naga::slabkit::make_device_array<float, 1>(
        sycl::range<1>{1024}, ctx, 0);
    
    // Chain operations in submission order
    // 1. Upload input
    auto ev_upload = naga::slabkit::copy_to_device(
        host_input.data(), input, q);
    
    // 2. Kernel (implicitly depends on upload because queue is in-order)
    auto input_view = input.view();
    auto output_view = output.view();
    auto ev_kernel = q.parallel_for(sycl::range<1>{1024}, [=](sycl::id<1> id) {
        output_view[id] = input_view[id] * 2.0f;
    });
    
    // 3. Download result (implicitly depends on kernel)
    auto ev_download = naga::slabkit::copy_to_host(
        output, host_output.data(), q);
    
    // Wait for all work to complete
    ev_download.wait();
    
    return 0;
}
```

**Why:** An in-order queue guarantees that all submitted operations execute in submission order without gaps or reordering. This is the simplest way to chain dependent work without manually managing events. When you submit to an in-order queue, each submission waits for all previous submissions to complete. No explicit dependency tracking needed. Alternatively, for out-of-order queues, you could pass the `deps` vector to `copy_to_device` and `copy_to_host` to link operations explicitly.

---

## Pattern 8: Subviews and Strided Transfers

Copy a rectangular sub-region of a large multidimensional array using subviews and Tier 2 (staging) transfers when the subview is non-contiguous.

```cpp
#include "slabkit/slabkit.hpp"
#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>

int main() {
    auto ctx = naga::slabkit::ContextBuilder()
        .devices({sycl::device{sycl::cpu_selector_v}})
        .build();
    
    sycl::queue q = ctx->make_in_order_queue(ctx->devices[0]);
    
    // Allocate a large 2D array
    auto arr = naga::slabkit::make_device_array<float, 2>(
        sycl::range<2>{1024, 1024}, ctx, 0);
    
    // Initialize: fill with indices
    auto view = arr.view();
    q.parallel_for(view.get_range(), [=](sycl::id<2> id) {
        view[id] = id[0] * 1024 + id[1];
    }).wait();
    
    // Extract a rectangular subview (top-left 512x512)
    auto sub = arr.subview({0, 0}, {512, 512});
    
    // Allocate scratch for the Tier 2 transfer
    auto scratch = naga::slabkit::make_scratch(ctx, 0, 512 * 512 * sizeof(float));
    
    // Copy subview to host using scratch as staging buffer
    std::vector<float> host_buf(512 * 512);
    auto ev = naga::slabkit::copy_to_host(sub, host_buf.data(), q, scratch);
    ev.wait();
    
    // Verify
    std::cout << "First element: " << host_buf[0] << "\n";  // 0
    std::cout << "Last element: " << host_buf[512*512-1] << "\n";  // 512*512-513
    
    return 0;
}
```

**Why:** `DeviceSubView` provides strided or rectangular access into a larger array. When transferring non-contiguous data (strided memory), you cannot use Tier 1 (direct) transfers. Instead, Tier 2 transfers use a staging buffer in device scratch memory to gather or scatter the data, then copy the contiguous result to/from the host. The scratch memory is temporary and can be reused or destroyed after the transfer completes.

---

## Pattern 9: Catching AllocationError

Wrap allocations in try/catch and use the `reason()` method to diagnose and respond to different failure modes.

```cpp
#include "slabkit/slabkit.hpp"
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

int main() {
    auto ctx = naga::slabkit::ContextBuilder()
        .devices({sycl::device{sycl::cpu_selector_v}})
        .slab_size_per_device(64 * 1024 * 1024)  // 64 MB limit
        .build();
    
    // Try to allocate progressively larger arrays until we hit the limit
    std::vector<naga::slabkit::DeviceArray<float, 1>> arrays;
    
    for (int iteration = 0; iteration < 100; ++iteration) {
        try {
            size_t size = 8 * 1024 * 1024 / sizeof(float);  // 8 MB per array
            auto arr = naga::slabkit::make_device_array<float, 1>(
                sycl::range<1>{size}, ctx, 0);
            arrays.push_back(std::move(arr));
            std::cout << "Allocated array " << iteration << "\n";
        } catch (const naga::slabkit::AllocationError& e) {
            std::cout << "Allocation failed at iteration " << iteration << "\n";
            
            switch (e.reason()) {
                case naga::slabkit::AllocationError::Reason::PoolExhausted:
                    std::cout << "  Reason: Pool slab is full\n";
                    std::cout << "  Action: Destroy some arrays or increase slab_size_per_device\n";
                    break;
                    
                case naga::slabkit::AllocationError::Reason::Fragmented:
                    std::cout << "  Reason: Fragmentation (no contiguous block available)\n";
                    std::cout << "  Action: Reduce allocation size or recycle allocations\n";
                    break;
                    
                case naga::slabkit::AllocationError::Reason::ScratchFull:
                    std::cout << "  Reason: Scratch memory is exhausted\n";
                    std::cout << "  Action: Call reset() on scratch or increase size\n";
                    break;
            }
            
            return 1;
        }
    }
    
    return 0;
}
```

**Why:** `AllocationError` has a `reason()` method that distinguishes between three failure modes. `PoolExhausted` means the slab is completely full—you need more total device memory or to free arrays. `Fragmented` means free space exists but is scattered in fragments too small for the allocation—this typically happens after many alloc/dealloc cycles; solutions include reducing allocation size, using scratch memory instead, or reshaping your allocation pattern. `ScratchFull` means a scratch bump allocator ran out of capacity—call `reset()` or `unwind()` to reclaim space. Catching and inspecting the reason allows you to implement graceful degradation or automatic fallbacks.

---

## Pattern 10: Use make_scratch with Device Overload

Prefer the `sycl::device` overload of `make_scratch()` when you already have the device object, to avoid managing device indices.

```cpp
#include "slabkit/slabkit.hpp"
#include <sycl/sycl.hpp>

int main() {
    // Select a GPU device
    auto dev = sycl::device{sycl::gpu_selector_v};
    
    // Build context with that device
    auto ctx = naga::slabkit::ContextBuilder()
        .devices({dev})
        .node_size(2048)
        .slab_size_per_device(256 * 1024 * 1024)
        .build();
    
    // Use the device overload directly (no index lookup)
    auto scratch = naga::slabkit::make_scratch(ctx, dev, 128 * 1024);
    
    // Also allocate arrays using the device overload
    auto arr = naga::slabkit::make_device_array<int, 1>(
        sycl::range<1>{10000}, ctx, dev);
    
    sycl::queue q = ctx->make_in_order_queue(dev);
    
    // Use scratch and array...
    auto tmp = scratch.allocate(1024);
    auto view = arr.view();
    q.parallel_for(view.get_range(), [=](sycl::id<1> id) {
        view[id] = id.get(0);
    }).wait();
    
    return 0;
}
```

**Why:** When you already have a `sycl::device` object, using the device overload of `make_scratch()` and `make_device_array()` is cleaner than manually resolving the device index via `ctx->device_index(dev)`. The overloads do the index lookup internally, reducing boilerplate. This is especially useful in multi-device contexts where you have device objects but no pre-computed indices.

---

## See Also

- **[Getting Started](getting-started.md)** — Step-by-step walkthrough for your first SlabKit program.
- **[Concepts](concepts.md)** — Deep dive into ownership, RAII, and design rationale.
- **[API Reference](api-reference.md)** — Complete type and function signatures.
- **[Anti-Patterns](anti-patterns.md)** — Common pitfalls and how to avoid them.
