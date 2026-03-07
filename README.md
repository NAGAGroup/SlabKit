# SlabKit

**Type-safe SYCL USM memory management without the boilerplate.** SlabKit eliminates the friction of raw USM pointers and buffer/accessor patterns: trivially-copyable views, pool-backed allocation, automatic RAII cleanup, and strided device↔host transfers all without hidden costs or fragmentation.

## Quick Reference

| Type | Owns? | Copyable | Use When |
|------|-------|----------|----------|
| `DeviceView<T,N>` | No | Yes (trivial) | Capturing arrays in kernels |
| `DeviceSubView<T,N>` | No | Yes (trivial) | Working with tiles or strides |
| `DeviceArray<T,N>` | Yes (shared) | Yes | Core allocation — RAII + refcounted |
| `MirroredArray<T,N>` | Both (device shared, host owned) | Yes | Paired host/device sync |
| `DeviceScratchMemory` | Yes (owning) | No (move-only) | Temporary allocations, no fragmentation |

## Patterns

### P1 — Allocate, Kernel, Read Back

The fundamental loop: get registry, make queue, allocate, compute, transfer:

```cpp
#include "slabkit/array.hpp"

auto& reg = slabkit::AllocatorRegistry::get();
const sycl::device& dev = reg.device_by_index(0);
sycl::queue q = slabkit::make_queue(dev);

// Allocate from pool
auto arr = slabkit::make_device_array<float, 1>(sycl::range<1>{1024}, dev);

// Upload — no .wait() needed; in-order queue ensures compute waits for this
slabkit::copy_to_device(host_in.data(), arr, q);

// Kernel — previous upload is guaranteed complete
q.parallel_for(sycl::range<1>{1024},
    [v = arr.view()](sycl::id<1> id) {
        v[id] *= 2.0f;
    });

// Download — wait only here, at the point we need the data on the host
std::vector<float> out(1024);
slabkit::copy_to_host(arr, out.data(), q).wait();
// arr destroyed here — memory automatically returned to pool
```

**Why `.view()`?** It's a `DeviceView`: pointer + range, trivially copyable. Safe to capture in lambdas. No kernel accessor boilerplate. When `arr` goes out of scope, its destructor returns memory to the pool via a deleter chain.

### P2 — 2D Array

Extend to multiple dimensions with `sycl::range<N>` and `sycl::id<N>`:

```cpp
auto arr = slabkit::make_device_array<float, 2>(sycl::range<2>{512, 512}, dev);

std::vector<float> data(512 * 512, 0.5f);
slabkit::copy_to_device(data.data(), arr, q).wait();

q.parallel_for(sycl::range<2>{512, 512},
    [v = arr.view()](sycl::id<2> id) {
        v[id] += 1.0f;  // row-major access
    }).wait();
```

Access is row-major: `v[{r, c}]` = `v.data()[r * cols + c]`.

### P3 — Subview: Tile a Large Array

Extract a strided slice without copying:

```cpp
// 1024x1024 device array
auto arr = slabkit::make_device_array<float, 2>(sycl::range<2>{1024, 1024}, dev);

// Tile at (row=100, col=200), size 256x256
auto tile = arr.subview({100, 200}, sycl::range<2>{256, 256});

// Kernel operates only on the tile
q.parallel_for(sycl::range<2>{256, 256},
    [sv = tile](sycl::id<2> id) {
        sv[id] = 3.14f;  // strided write
    }).wait();

// Surrounding elements of arr untouched
```

`subview()` computes strides automatically from the parent range. `sv[id]` applies the strides internally—you index as if working on the extracted 256x256 region, but writes land in the correct strided positions in the parent array.

### P4 — Gather: Strided Device to Device

Copy a strided region to a contiguous result without staging:

```cpp
auto src = slabkit::make_device_array<float, 2>(sycl::range<2>{1024, 1024}, dev);
auto dst = slabkit::make_device_array<float, 2>(sycl::range<2>{256, 256}, dev);

auto tile = src.subview({100, 200}, sycl::range<2>{256, 256});

// Device-to-device strided copy (runs as a kernel, no host staging)
slabkit::copy(tile, dst, q).wait();

// dst now contains the contiguous packed tile data
```

This overload detects strided sources and uses a `parallel_for` kernel on the device. No scratch buffer needed, no host-side staging. Efficient for device-internal rearrangement.

### P5 — MirroredArray: Host Setup → Push → Compute → Pull

For workflows where host preparation and device results are equally important:

```cpp
slabkit::MirroredArray<float, 1> ma(sycl::range<1>{1024}, 0);  // device index

// Populate on host
auto h = ma.host_data();  // std::span<float>
std::iota(h.begin(), h.end(), 0.0f);

// Push to device — no .wait(); in-order queue ensures kernel sees it
ma.sync_to_device(q);

// Compute on device
q.parallel_for(sycl::range<1>{1024},
    [v = ma.view()](sycl::id<1> id) {
        v[id] *= v[id];  // square
    });

// Pull back — wait only here
ma.sync_to_host(q).wait();

// Read results from host_data()
std::cout << "result[0] = " << h[0] << "\n";
```

**Copy semantics**: Device side (shared_ptr) is **shared** across copies. Host side (std::vector) is **deep-copied**. Use this when you need independent host buffers viewing the same device allocation.

### P6 — Scratch Memory: Zero-Fragmentation Temporary Allocations

When you allocate and deallocate many small buffers in a loop, pool fragmentation becomes expensive. Scratch memory uses a single pre-allocated slab with bump allocation—no per-allocation overhead, no fragmentation:

```cpp
auto scratch = slabkit::make_scratch_memory(dev, 4 * 1024 * 1024);  // 4 MB slab

for (auto& batch : batches) {
    // Save the current position
    auto mark = scratch.top();

    // Allocate temporaries — cheap (just offset bump)
    float* tmp_a = static_cast<float*>(scratch.allocate(batch.size() * sizeof(float)));
    float* tmp_b = static_cast<float*>(scratch.allocate(batch.size() * sizeof(float)));

    // Use in kernels
    auto e = q.parallel_for(sycl::range<1>{batch.size()},
        [=](sycl::id<1> id) { tmp_a[id[0]] = ...; });
    e.wait();  // must complete before unwind — scratch pointers must stay live

    // Rewind — reclaim all allocations for this batch (no deallocation cost)
    scratch.unwind(mark);
}
```

**When to use**: Batched workloads, temporary weight matrices, per-iteration staging buffers. Also good for tiny allocations that don't justify main-pool overhead.

**Important**: Keep `scratch` alive until all kernels that reference its pointers complete. The pointer itself (e.g., `tmp_a`) is device-side; the SYCL queue enforces event ordering.

### P7 — Strided Host↔Device Transfer via Scratch

When you have a strided device region and need it contiguous on the host:

```cpp
auto scratch = slabkit::make_scratch_memory(dev, 10 * 1024 * 1024);

auto large = slabkit::make_device_array<float, 2>(sycl::range<2>{4096, 4096}, dev);
auto tile = large.subview({1000, 1000}, sycl::range<2>{512, 512});

// Gather kernel + memcpy (using scratch staging)
std::vector<float> host_result(512 * 512);
slabkit::copy_to_host(tile, host_result.data(), q, scratch).wait();

// host_result now contains packed tile data
```

Internally: allocate staging buffer in scratch → gather (parallel_for) → memcpy staging to host. Caller controls scratch lifetime—essential for batched transfers.

### P8 — Event Chaining (Out-of-Order)

For concurrent independent operations, opt into an out-of-order queue and manage deps explicitly:

```cpp
sycl::queue q = slabkit::make_out_of_order_queue(dev);

auto e1 = slabkit::copy_to_device(src1.data(), arr1, q);
auto e2 = slabkit::copy_to_device(src2.data(), arr2, q);  // concurrent with e1

// Wait for both uploads before compute
auto e3 = q.submit([&](sycl::handler& h) {
    h.depends_on({e1, e2});
    h.parallel_for(arr1.get_range(), [a = arr1.view(), b = arr2.view()](sycl::id<1> id) {
        a[id] += b[id];
    });
});

slabkit::copy_to_host(arr1, out.data(), q, std::vector{e3}).wait();
```

For the common sequential case, prefer an in-order queue — no `depends_on`, no event handles.

## Ownership Model

| Type | Owns | Copy | Freed |
|------|------|------|-------|
| `DeviceView<T,N>` | No | Trivial (ptr+range) | Never |
| `DeviceSubView<T,N>` | No | Trivial (ptr+range+strides) | Never |
| `DeviceArray<T,N>` | Yes (shared) | Shared (ref-counted) | Last copy destroyed |
| `MirroredArray<T,N>` | Device: shared; Host: owned | Device shared, host deep-copied | Device: last copy; Host: on destroy |
| `unique_device_ptr<T>` | Yes (unique) | Move-only | Destroyed |
| `DeviceScratchMemory` | Yes (owns slab) | Move-only | On destroy |

## Setup

Initialize once per application:

```cpp
// Optional: configure before first get()
auto& cfg = slabkit::AllocatorRegistry::config();
cfg.device_slab_fraction = 0.5f;  // 50% of device global memory
cfg.node_size = 1024;              // pool freelist granularity

// First call initializes all devices
auto& reg = slabkit::AllocatorRegistry::get();

// Always use make_queue — never sycl::queue{dev}
const sycl::device& dev = reg.device_by_index(0);
sycl::queue q = slabkit::make_queue(dev);
```

**Why `make_queue`?** Two reasons. First, the registry creates one context per platform — all USM pointers are valid in that context; `sycl::queue{dev}` creates a new context and breaks USM access. Second, `make_queue` returns an **in-order queue**: each operation waits for the previous one to complete. This is the right default for USM workloads — sequential reasoning, no accidental races, no boilerplate `depends_on` chains for the common case.

**Queue variants**:

```cpp
// In-order (default) — each op waits for the previous
sycl::queue q = slabkit::make_queue(dev);

// Out-of-order — ops may run concurrently; caller manages event deps
sycl::queue q = slabkit::make_out_of_order_queue(dev);

// Fully custom — caller owns all properties, nothing added
sycl::queue q = slabkit::make_queue_with_properties(dev, props);
```

**In-order queue patterns**: With an in-order queue, the upload → compute → download sequence just works:

```cpp
slabkit::copy_to_device(host.data(), arr, q);   // no .wait()
q.parallel_for(arr.get_range(), [v = arr.view()](sycl::id<1> id) { v[id] *= 2.0f; });
slabkit::copy_to_host(arr, out.data(), q).wait(); // wait only at the end
```

Each step implicitly depends on the previous. Drop `.wait()` calls from the middle of pipelines — they become unnecessary synchronization points that stall the device.

**When to reach for out-of-order**: Independent uploads to multiple arrays, concurrent kernel streams, or any time you want to express a true dependency DAG. Use `make_out_of_order_queue` explicitly and manage event deps yourself.

**Config fields**:
- `block_size` (default 64 KB) — arena block granularity
- `node_size` (default 1 KB) — pool freelist node size
- `device_slab_fraction` (default 0.60) — fraction of device memory reserved for allocations
- `cpu_slab_fraction` (default 0.30) — fraction for CPU devices

## API Quick Reference

**Allocation**

```cpp
// Array (RAII, shared ownership)
auto arr = slabkit::make_device_array<T, N>(range, dev);

// Scratch memory (RAII, owning, move-only)
auto scratch = slabkit::make_scratch_memory(dev, size_bytes);
void* ptr = scratch.allocate(size_bytes, alignment);
auto mark = scratch.top();
scratch.unwind(mark);

// Low-level pointers
auto shared = slabkit::malloc_shared_device_ptr<T>(dev, count);
auto unique = slabkit::malloc_unique_device_ptr<T>(dev, count);
```

**Views**

```cpp
auto v = arr.view();                     // DeviceView<T, N>
auto sv = arr.subview({offs}, {range}); // DeviceSubView<T, N>
```

**Transfers (async, return `sycl::event`)**

```cpp
// Tier 1: contiguous
copy_to_device(host_ptr, arr_or_view, q, deps);
copy_to_host(arr_or_view, host_ptr, q, deps);
copy(src_array, dst_array, q, deps);

// Tier 2: strided host transfers (needs scratch)
copy_to_device(host_ptr, subview, q, scratch, deps);
copy_to_host(subview, host_ptr, q, scratch, deps);

// Tier 3: strided device-to-device (kernel, no scratch)
copy(src_subview, dst_subview, q, deps);
copy(src_subview, dst_array, q, deps);
copy(src_array, dst_subview, q, deps);

// Tier 4: blocking sync (calls .wait() internally)
sync_to_host(arr_or_subview, host_ptr, q, scratch?);
```

**Queue**

```cpp
sycl::queue q = slabkit::make_queue(dev);                          // in-order (default)
sycl::queue q = slabkit::make_out_of_order_queue(dev);             // out-of-order
sycl::queue q = slabkit::make_queue_with_properties(dev, props);   // fully custom
```

**Registry**

```cpp
auto& reg = slabkit::AllocatorRegistry::get();
const sycl::device& dev = reg.device_by_index(0);
size_t idx = reg.device_index(dev);
auto pool = reg.pool(dev);
auto q = reg.make_queue(dev);
```

## Design Principles

- **Type-safe**: Dimensions are template parameters (`DeviceArray<T, 2>`), not runtime values. Stride calculations are automatic.
- **Zero-copy views**: `DeviceView` and `DeviceSubView` are trivially copyable and capture-safe. No hidden allocations.
- **RAII everywhere**: Memory is tracked via `shared_ptr` deleter chains. No manual `sycl::free()` calls.
- **Pool-backed**: All device allocations come from a single pre-allocated slab per device, avoiding fragmentation and driver instability.
- **Async-first**: Transfer functions return `sycl::event`. Use `.wait()` only when needed.
- **No boilerplate**: No accessors, no buffer lifetime management, no lambda capture restrictions.

## License

See LICENSE file.
