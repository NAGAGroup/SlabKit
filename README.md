# SlabKit

SYCL device memory pool library with N-dimensional arrays, safe view semantics, and RAII allocation.

## Why SlabKit?

SYCL's Unified Shared Memory (USM) offers powerful abstraction but requires careful management: repeated malloc/free is slow and fragments device memory. SlabKit solves this with **pool-backed allocation**: allocate a slab once per device, reuse nodes for short-lived arrays, and let RAII clean up automatically. No `sycl::buffer` required—just safe, ergonomic device arrays with trivially-copyable views that are kernel-safe by design.

## Quick Start

```cpp
#include "slabkit/slabkit.hpp"
using namespace slabkit;

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

## Documentation

| Page | Description |
|------|-------------|
| [Getting Started](docs/getting-started.md) | Step-by-step walkthrough for your first project |
| [Concepts](docs/concepts.md) | Ownership model, RAII semantics, view safety, builder pattern |
| [API Reference](docs/api-reference.md) | Complete type and function documentation |
| [Patterns](docs/patterns.md) | Best-practice usage patterns with examples |
| [Anti-Patterns](docs/anti-patterns.md) | Common mistakes, diagnosis, and fixes |

## License

See [LICENSE](LICENSE) for details.
