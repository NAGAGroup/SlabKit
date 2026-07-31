# SlabKit

SYCL device memory pool library with N-dimensional arrays, safe view semantics, and RAII allocation.

## Why SlabKit?

SYCL's Unified Shared Memory (USM) offers powerful abstraction but requires careful management: repeated malloc/free is slow and fragments device memory. SlabKit solves this with **pool-backed allocation**: allocate a slab once per device, reuse nodes for short-lived arrays, and let RAII clean up automatically. No `sycl::buffer` required—just safe, ergonomic device arrays with trivially-copyable views that are kernel-safe by design.

## Quick Start

```cpp
#include <naga/slabkit/slabkit.hpp>
using namespace naga::slabkit;

// Create a context (manages devices and memory pools)
auto ctx = ContextBuilder{}
    .devices({sycl::device{sycl::gpu_selector_v}})
    .build();  // shared_ptr<Context>

// Allocate a 2D device array (RAII-managed)
auto arr = make_device_array<float>(sycl::range<2>{1024, 1024}, ctx, 0);

// Get a kernel-safe view (trivially copyable)
auto view = arr.view();

// Use in a SYCL kernel
sycl::queue q = ctx->make_in_order_queue(arr.devices[0]);
q.parallel_for(arr.get_range(), [=](sycl::id<2> id) {
    view[id] = compute(id);  // index by sycl::id<N>
}).wait();

// Array destroyed → memory automatically returned to pool
```

## Using SlabKit

SlabKit is a [pixi](https://pixi.sh) **source package**. From a pixi
project: `slabkit = { git = "https://github.com/NAGAGroup/SlabKit.git", tag = "..." }`,
then `find_package(slabkit CONFIG REQUIRED)` → `naga::slabkit::slabkit`. From a
non-pixi project: `pixi exec pixi-pack --environment pack --create-executable`
produces a self-extracting archive (`./environment.sh` → `./env`, no
pixi/conda needed on the target); point `CMAKE_PREFIX_PATH` at the unpacked
env and use the same `find_package`. The pixi pipeline is the only officially
supported build/consumption path (hermetic conda builds are the point);
issues specific to FetchContent/CPM-style embedding will be closed.

## Documentation

| Page | Description |
|------|-------------|
| [Getting Started](docs/getting-started.md) | Step-by-step walkthrough for your first project |
| [Concepts](docs/concepts.md) | Ownership model, RAII semantics, view safety, builder pattern |
| [API Reference](docs/api-reference.md) | Complete type and function documentation |
| [Patterns](docs/patterns.md) | Best-practice usage patterns with examples |
| [Anti-Patterns](docs/anti-patterns.md) | Common mistakes, diagnosis, and fixes |

## Status & known limitations

SlabKit is used in production by [CospanMDAO](https://github.com/NAGAGroup/CospanMDAO)
and the research projects built on it; the ownership model and pool allocator
are stable and exercised daily. A 2026 code audit left a few known, deliberate
limitations on record (all benign for the primary usage pattern of one
program-lifetime `Context`):

- Slabs are not returned to the SYCL runtime when a `Context` is destroyed —
  irrelevant for program-lifetime contexts, a leak if you create and destroy
  many contexts in one process.
- `DeviceScratchMemory` holds a raw pool pointer; keep the `Context` alive
  while scratch is in use (the docs' stronger lifetime claim is aspirational).
- Pool-corruption assertions compile out in release builds; use the debug
  preset when hunting allocator issues.

## License

MIT — see [LICENSE](LICENSE) for details.
