# SlabKit

SYCL slab allocator and device array library. Pool-backed USM allocation, N-dimensional device arrays with view/subview semantics, and async transfer utilities.

## Contents

- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
- [Transfer API](#transfer-api)
- [Patterns: DO THIS](#patterns-do-this)
- [Anti-Patterns: NEVER DO THIS](#anti-patterns-never-do-this)

---

## Quick Start

```cpp
#include "slabkit/slabkit.hpp"  // umbrella header — includes everything
using namespace slabkit;
```

For minimal builds, include only what you need:
- `slabkit/memory.hpp` — registry, `malloc_device`, scratch, queue factories
- `slabkit/array.hpp` — `DeviceArray`, `MirroredArray`, views, transfers
- `slabkit/context.hpp` — `SlabContext`, `SlabContextBuilder`
- `slabkit/scratch.hpp` — `DeviceScratchBuffer`, `DeviceScratchView`
- `slabkit/error.hpp` — `AllocationError`

---

## Core Concepts

### Ownership model

| Type | Ownership | Copyable | Kernel-safe |
|------|-----------|----------|-------------|
| `DeviceArray<T,N>` | Owns USM memory (unique) | No — move-only | No (owns memory) |
| `DeviceView<T,N>` | Non-owning | Yes — trivially copyable | **Yes** |
| `DeviceSubView<T,N>` | Non-owning, strided | Yes — trivially copyable | **Yes** |
| `MirroredArray<T,N>` | Owns device + host | No — move-only | No |
| `DeviceScratchBuffer` | Owns scratch USM | No — move-only | No |
| `DeviceScratchView` | Non-owning | Yes — trivially copyable | **Yes** |

**Rule:** Only views (`DeviceView`, `DeviceSubView`, `DeviceScratchView`) are safe to capture by value in SYCL kernels. Never capture arrays or scratch buffers.

### Two construction paths

**Registry-backed** — uses the global `AllocatorRegistry`. Initialises lazily on first access. Suitable for application code that doesn't need to control context lifetime explicitly.

```cpp
auto dev = sycl::device{sycl::gpu_selector_v};
auto arr = make_device_array<float>(sycl::range<2>{1024, 1024}, dev);
auto q   = make_in_order_queue(dev);
```

**Registry-free** — takes an explicit `shared_ptr<SlabContext>`. Required for tests, libraries, and any code that needs deterministic context lifetime or multiple isolated contexts.

```cpp
auto ctx = SlabContextBuilder{}
    .contexts({sycl::context{dev}})
    .devices({dev})
    .node_size(1024)
    .build();  // returns shared_ptr<SlabContext>

auto arr = make_device_array<float>(sycl::range<2>{1024, 1024}, ctx, 0);
auto q   = ctx->make_in_order_queue(dev);
```

---

## API Reference

### SlabContextBuilder

```cpp
SlabContextBuilder{}
    .contexts(std::vector<sycl::context>)   // one context per platform, all platform devices
    .devices(std::vector<sycl::device>)     // all devices to manage
    .node_size(size_t)                      // pool freelist node size in bytes (default: 1024)
    .slab_size_per_device(size_t)           // bytes per device slab (default: 64MB)
    .build()                                // → shared_ptr<SlabContext>
```

Pools are lazy — not constructed until first access. `build()` throws `std::invalid_argument` if contexts or devices are empty.

### SlabContext

```cpp
ctx->pool_at(device_index)              // DeviceMemoryPool& — lazy init, thread-safe
ctx->pool_for(dev)                      // DeviceMemoryPool& — by sycl::device
ctx->context_for(device_index)          // const sycl::context&
ctx->context_for(dev)                   // const sycl::context&
ctx->device_index(dev)                  // size_t — throws invalid_argument if not found
ctx->make_in_order_queue(dev)           // sycl::queue
ctx->make_out_of_order_queue(dev)       // sycl::queue
ctx->make_queue_with_props(dev, props)  // sycl::queue

// Public data (read after build, do not modify)
ctx->devices                            // std::vector<sycl::device>
ctx->platform_contexts                  // std::vector<sycl::context>
ctx->device_to_platform                 // std::vector<size_t>
ctx->device_slab_sizes                  // std::vector<size_t>
```

`SlabContext` is move-only. Always managed via `shared_ptr` — construct through `SlabContextBuilder::build()`.

### AllocatorRegistry

Global singleton. Lazy-initialises on first `get()`. Uses all SYCL devices returned by `sycl::device::get_devices()`.

```cpp
AllocatorRegistry::context()               // shared_ptr<SlabContext>
AllocatorRegistry::get().pool(dev)         // DeviceMemoryPool&
AllocatorRegistry::get().device_index(dev) // size_t
AllocatorRegistry::get().device_by_index(i)// const sycl::device&
AllocatorRegistry::get().make_in_order_queue(dev)
AllocatorRegistry::get().make_out_of_order_queue(dev)
AllocatorRegistry::get().make_queue_with_props(dev, props)

// One-shot configuration — must be called before first context()/pool() access
AllocatorRegistry::configure(AllocatorRegistry::Config{
    .block_size           = 64 * 1024,  // arena block granularity (bytes)
    .node_size            = 1024,       // pool freelist node size (bytes)
    .device_slab_fraction = 0.60f,      // fraction of device global_mem_size
    .cpu_slab_fraction    = 0.30f,      // fraction of CPU global_mem_size
});

// Test builds only (requires -DSLABKIT_TESTING)
AllocatorRegistry::reset_for_testing();
```

Prefer the free function wrappers over calling `get()` directly:

```cpp
make_in_order_queue(dev)           // → sycl::queue
make_out_of_order_queue(dev)       // → sycl::queue
make_queue_with_props(dev, props)  // → sycl::queue
malloc_device<T>(count, dev)       // registry-backed
malloc_device<T>(count, ctx, idx)  // registry-free
make_scratch(dev, size)            // registry-backed DeviceScratchBuffer
make_scratch(ctx, device_index, size)  // registry-free DeviceScratchBuffer
```

### DeviceArray\<T, N\>

Move-only. Owns its USM allocation. Returned by pool deleter on destruction — safe if `SlabContext` outlives it, no-op if context is already gone.

```cpp
// Construction
auto arr = make_device_array<float>(sycl::range<2>{H, W}, ctx, device_idx);
auto arr = make_device_array<float>(sycl::range<2>{H, W}, dev);  // registry-backed

// Views — use these for all kernel access
DeviceView<float, 2>       v  = arr.view();         // mutable view
DeviceView<const float, 2> cv = arr.const_view();   // const view
// or implicit conversion:
DeviceView<float, 2>       v  = arr;
DeviceView<const float, 2> cv = arr;                // from const DeviceArray

// Subviews
DeviceSubView<float, 2>       sv  = arr.subview({row_off, col_off}, sub_range);
DeviceSubView<const float, 2> csv = arr.const_subview({row_off, col_off}, sub_range);

// Metadata
arr.get_range()    // sycl::range<N>
arr.size()         // element count
arr.byte_size()    // bytes
arr.device_index() // size_t
arr.data()         // T* — raw pointer, see anti-patterns before using
```

### DeviceView\<T, N\>

Trivially copyable. Safe to capture by value in SYCL kernels.

```cpp
v[sycl::id<N>{i, j}]  // element access by N-dimensional index
v.get_range()          // sycl::range<N>
v.size()               // element count
v.byte_size()          // bytes
v.data()               // T*
// N==1 only:
v[size_t i]            // linear element access
```

### DeviceSubView\<T, N\>

Trivially copyable, strided. Safe to capture by value in SYCL kernels. Strides in elements.

```cpp
sv[sycl::id<N>{i, j}]  // strided element access
sv.get_range()          // sycl::range<N> of the subview extent
sv.size()               // element count in subview
sv.byte_size()          // bytes
// N==1 only:
sv[size_t i]            // linear strided element access
```

### MirroredArray\<T, N\>

Move-only. Paired device (`DeviceArray`) and host (`std::vector<T>`) storage.

```cpp
// Construction
MirroredArray<float, 2> m{sycl::range<2>{H, W}, ctx, device_idx};
MirroredArray<float, 2> m{sycl::range<2>{H, W}, dev};  // registry-backed

// Deep copy (device side uninitialized — call sync_to_device after)
auto m2 = m.clone();

// Views (same as DeviceArray)
m.view()               // DeviceView<T,N>
m.view()               // DeviceView<const T,N> on const MirroredArray
m.const_view()         // DeviceView<const T,N>
m.subview(offsets, sub_range)
m.const_subview(offsets, sub_range)

// Host side
m.host_data()          // std::span<T> (mutable)
m.host_data()          // std::span<const T> (const)

// Transfers
sycl::event e = m.sync_to_device(q, deps);  // host → device async
sycl::event e = m.sync_to_host(q, deps);    // device → host async
```

### DeviceScratchBuffer / DeviceScratchView

`DeviceScratchBuffer` is an alias for `DeviceScratchMemory`. Move-only, owns a pool-backed scratch region. `DeviceScratchView` is a non-owning, trivially copyable pointer+size pair.

```cpp
auto scratch = make_scratch(dev, 64 * 1024);          // registry-backed, 64KB
auto scratch = make_scratch(ctx, device_idx, 64*1024); // registry-free

DeviceScratchView sv = DeviceScratchView::from_allocation(scratch.data(), scratch.capacity());
```

### Error handling

```cpp
#include "slabkit/error.hpp"

try {
    auto arr = make_device_array<float>(huge_range, dev);
} catch (const AllocationError& e) {
    switch (e.reason()) {
        case AllocationError::Reason::PoolExhausted:  // pool has no free nodes
        case AllocationError::Reason::Fragmented:     // nodes exist but not contiguous
        case AllocationError::Reason::ScratchFull:    // scratch capacity exceeded
    }
}
```

---

## Transfer API

All transfers are async and return `sycl::event`. Pass dependency events via the `deps` parameter.

### Tier 1 — Contiguous (memcpy, no kernel)

```cpp
// Device → Host
sycl::event e = copy_to_host(arr,  host_ptr, q, deps);
sycl::event e = copy_to_host(view, host_ptr, q, deps);

// Host → Device
sycl::event e = copy_to_device(host_ptr, arr,  q, deps);
sycl::event e = copy_to_device(host_ptr, view, q, deps);

// Device → Device (contiguous)
sycl::event e = copy(src_arr, dst_arr, q, deps);
```

### Tier 2 — Strided host transfers (staging buffer via scratch)

```cpp
DeviceScratchBuffer scratch = make_scratch(dev, required_bytes);

sycl::event e = copy_to_host(subview,   host_ptr, q, scratch, deps);
sycl::event e = copy_to_device(host_ptr, subview, q, scratch, deps);
```

### Tier 3 — Strided device-to-device (parallel_for, no staging)

```cpp
sycl::event e = copy(src_subview, dst_subview, q, deps);
sycl::event e = copy(src_subview, dst_arr,     q, deps);
sycl::event e = copy(src_arr,     dst_subview, q, deps);
```

### Tier 4 — Blocking sync wrappers

```cpp
sync_to_host(arr,     host_ptr, q);
sync_to_host(subview, host_ptr, q, scratch);
```

---

## Patterns: DO THIS

### Always pass views into kernels

```cpp
auto arr  = make_device_array<float>(sycl::range<2>{H, W}, ctx, 0);
auto view = arr.view();  // DeviceView<float, 2> — trivially copyable

auto e = q.parallel_for(arr.get_range(), [=](sycl::id<2> id) {
    view[id] = compute(id);  // index by sycl::id<2>
});
```

### Subview for tiled / windowed access

```cpp
auto tile = arr.subview({row_off, col_off}, sycl::range<2>{TILE_H, TILE_W});

q.parallel_for(tile.get_range(), [=](sycl::id<2> id) {
    tile[id] = ...;
});
```

### Use const views when you only read

```cpp
void process(DeviceView<const float, 2> in, DeviceView<float, 2> out, sycl::queue& q) {
    q.parallel_for(in.get_range(), [=](sycl::id<2> id) {
        out[id] = in[id] * 2.0f;
    });
}
```

### Registry-free for tests and libraries

```cpp
// Tests — never use AllocatorRegistry
auto ctx = SlabContextBuilder{}
    .contexts({sycl::context{dev}})
    .devices({dev})
    .build();

auto arr = make_device_array<float>(range, ctx, 0);
```

### MirroredArray sync pattern

```cpp
MirroredArray<float, 2> m{range, ctx, 0};

// Populate host side
std::fill(m.host_data().begin(), m.host_data().end(), 0.0f);

// Push to device, then compute
auto e0 = m.sync_to_device(q);
auto e1 = q.parallel_for(m.get_range(), e0, [=](sycl::id<2> id) {
    auto v = m.view();
    v[id] = v[id] + 1.0f;
});

// Pull back
m.sync_to_host(q, {e1}).wait();
```

---

## Anti-Patterns: NEVER DO THIS

---

### ❌ Using `data()` to index inside a kernel

```cpp
// WRONG — raw pointer arithmetic in a kernel defeats the entire view system
auto arr     = make_device_array<float>(sycl::range<2>{H, W}, ctx, 0);
auto arr_ptr = arr.data();  // <-- NO

q.parallel_for(arr.get_range(), [=](sycl::id<2> id) {
    arr_ptr[id[0] * W + id[1]] = ...;  // <-- NO NO NO
});
```

```cpp
// CORRECT — use a view, index by sycl::id
auto view = arr.view();

q.parallel_for(arr.get_range(), [=](sycl::id<2> id) {
    view[id] = ...;
});
```

`data()` exists for interop with SYCL memcpy and transfer functions. It is **not** for manual pointer arithmetic in kernels. If you find yourself computing `ptr[row * stride + col]` inside a kernel, use `view[id]` or `subview[id]` instead.

---

### ❌ Capturing `DeviceArray` by reference into a kernel

```cpp
// WRONG — DeviceArray is not trivially copyable, capturing by ref is a lifetime hazard
auto arr = make_device_array<float>(range, ctx, 0);

q.parallel_for(range, [&](sycl::id<1> id) {  // <-- [&] captures arr by ref
    arr.view()[id] = 0;                        // <-- calling .view() inside kernel???
});
```

```cpp
// CORRECT — get the view before submitting, capture by value
auto arr  = make_device_array<float>(range, ctx, 0);
auto view = arr.view();  // get view outside the kernel

q.parallel_for(range, [=](sycl::id<1> id) {  // [=] captures view by value
    view[id] = 0;
});
```

---

### ❌ Creating a view, ignoring it, and using `data()` anyway

```cpp
// WRONG — the view was created for a reason, use it
auto arr      = make_device_array<float>(range, ctx, 0);
auto arr_view = arr.view();    // created but never used
auto arr_ptr  = arr.data();    // went straight for the pointer

q.parallel_for(range, [=](sycl::id<2> id) {
    arr_ptr[id[0] * stride + id[1]] = ...;  // manual stride arithmetic = bug waiting to happen
});
```

```cpp
// CORRECT
auto arr  = make_device_array<float>(range, ctx, 0);
auto view = arr.view();

q.parallel_for(range, [=](sycl::id<2> id) {
    view[id] = ...;  // no stride arithmetic, bounds-safe, correct
});
```

---

### ❌ Using `AllocatorRegistry` directly in tests

```cpp
// WRONG — tests must not depend on the global registry
AllocatorRegistry::get().init(...);
auto arr = make_device_array<float>(range, AllocatorRegistry::get().device_by_index(0));
```

```cpp
// CORRECT — use SlabContextBuilder for isolated, reproducible tests
auto ctx = SlabContextBuilder{}
    .contexts({sycl::context{sycl::device{sycl::cpu_selector_v}}})
    .devices({sycl::device{sycl::cpu_selector_v}})
    .build();

auto arr = make_device_array<float>(range, ctx, 0);
```

---

### ❌ Copying a `DeviceArray`

```cpp
// WRONG — won't compile, but the intent is wrong too
auto arr2 = arr;  // error: copy constructor is deleted
```

`DeviceArray` is move-only by design — it owns a unique USM allocation. If you need a second independent array with the same data, use `MirroredArray::clone()` for host-mirrored data, or allocate a new array and use `copy()` to populate it.

---

### ❌ Holding a `DeviceView` after the owning `DeviceArray` is destroyed

```cpp
// WRONG — dangling view
DeviceView<float, 2> view;
{
    auto arr = make_device_array<float>(range, ctx, 0);
    view = arr.view();
}  // arr destroyed here — view is now dangling

q.parallel_for(range, [=](sycl::id<2> id) {
    view[id] = 0;  // UB
});
```

Views do not extend the lifetime of the array. Ensure the owning `DeviceArray` outlives all queued work that uses its views.

---

### ❌ Passing stride as a separate variable instead of using subviews

```cpp
// WRONG — manual stride threading is fragile and defeats the subview API
float* ptr    = big_arr.data();
size_t stride = big_arr.get_range()[1];
size_t offset = row_start * stride + col_start;

q.parallel_for(sub_range, [=](sycl::id<2> id) {
    ptr[offset + id[0] * stride + id[1]] = ...;  // error-prone
});
```

```cpp
// CORRECT — subview carries offset and strides, no manual arithmetic
auto tile = big_arr.subview({row_start, col_start}, sub_range);

q.parallel_for(tile.get_range(), [=](sycl::id<2> id) {
    tile[id] = ...;  // correct by construction
});
```

---

### ❌ Using `copy_to_host` / `copy_to_device` when you have persistent host-side data

```cpp
// WRONG — manually managing a host buffer alongside a DeviceArray
auto arr = make_device_array<float>(range, ctx, 0);
std::vector<float> host(range.size());

// ... populate host ...

// upload
copy_to_device(host.data(), arr, q).wait();

// ... compute ...

// download
copy_to_host(arr, host.data(), q).wait();
// now process host...
```

This pattern is not wrong in isolation, but when `arr` and `host` travel together through your code — passed as a pair, kept in sync, copied together — you have manually reimplemented `MirroredArray`. Use it instead:

```cpp
// CORRECT — MirroredArray owns both sides, sync API is explicit and hard to misuse
MirroredArray<float, 2> m{range, ctx, 0};

// populate host side
std::fill(m.host_data().begin(), m.host_data().end(), 0.0f);

// upload, compute, download
auto e0 = m.sync_to_device(q);
auto v  = m.view();
auto e1 = q.parallel_for(m.get_range(), e0, [=](sycl::id<2> id) {
    v[id] = v[id] + 1.0f;
});
m.sync_to_host(q, {e1}).wait();

// process m.host_data() — guaranteed fresh
```

`copy_to_host` / `copy_to_device` are the right tool when you are doing a one-shot transfer with no persistent paired host buffer, or when operating on subviews. If you find yourself keeping a `std::vector` next to a `DeviceArray` across multiple functions, that's `MirroredArray`.

---

### ❌ Using `sycl::malloc_device` / `sycl::free` for temporary device memory

```cpp
// WRONG — bypasses the pool entirely, causes repeated USM alloc/free overhead
// and fragments device memory
void my_kernel_launcher(sycl::queue& q, size_t tmp_bytes) {
    float* tmp = static_cast<float*>(sycl::malloc_device(tmp_bytes, q));

    q.parallel_for(range, [=](sycl::id<1> id) {
        // use tmp as scratch
    }).wait();

    sycl::free(tmp, q);
}
```

Every `sycl::malloc_device` is a synchronising, driver-level allocation. Doing this per-invocation destroys throughput on repeated calls. Use `DeviceScratchBuffer` instead.

But there is a second level to this mistake: **creating the scratch buffer inside the function itself** is also wrong if that function is called repeatedly:

```cpp
// STILL WRONG — scratch is allocated and freed on every call.
// Pool allocation is cheap but not free. Don't do this in a hot loop.
void my_kernel(sycl::queue& q, std::shared_ptr<SlabContext> ctx,
               size_t device_idx) {
    auto scratch = make_scratch(ctx, device_idx, 256 * 1024);  // <-- allocates every call
    // ...
}

for (auto& tile : tiles)
    my_kernel(q, ctx, 0);  // allocates + frees 256KB on every iteration
```

**The correct pattern: allocate once at the call site and pass the scratch in.**
Functions that need temporary device memory should accept `DeviceScratchBuffer&` or `DeviceScratchView` as a parameter — the caller owns the lifetime:

```cpp
// CORRECT — scratch allocated once by the caller, reused across all calls
void my_kernel(sycl::queue& q, DeviceScratchView scratch) {
    auto* tmp = static_cast<float*>(scratch.data());

    q.parallel_for(range, [=](sycl::id<1> id) {
        tmp[id[0]] = ...;
    }).wait();
    // scratch ownership stays with caller — no deallocation here
}

// Caller: allocate once, pass into every invocation
auto scratch = make_scratch(ctx, 0, 256 * 1024);
auto sv      = DeviceScratchView::from_allocation(scratch.data(), scratch.capacity());

for (auto& tile : tiles)
    my_kernel(q, sv);  // zero allocation overhead per call
```

The same rule applies to `DeviceScratchBuffer&` when the function needs to resize or reset the scratch region between calls. `DeviceScratchView` is the right parameter type when the function just needs to read/write into a pre-allocated region.

The same applies to `sycl::malloc_shared` and `sycl::malloc_host` used for temporaries — if the allocation is short-lived and repeated, it should come from the pool, allocated once by the caller.
