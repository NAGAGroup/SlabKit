# slabkit

USM device memory with owning handles, for SYCL.

Two families of free functions — **`malloc_*`** for uninitialized storage and
**`make_*`** for constructed objects — over four kinds of **placement**: a
queue, a device with its context, an arena, a pool. They all yield one handle
type, and the handle knows how to release what it holds.

There are no array or view types. Shapes belong to the code that has them.

```cpp
#include <naga/slabkit/slabkit.hpp>
namespace sk = naga::slabkit;

// One context per device, created before the queue that uses it.
sycl::device  dev{sycl::gpu_selector_v};
sycl::context ctx{dev};
sycl::queue   q{ctx, dev};

// Storage: SYCL's signatures, returning a handle instead of a raw pointer.
auto data = sk::malloc_device<float>(1024, q);       // unique_device_ptr<float[]>

// Objects: std::make_unique's forms, constructed on the device.
auto weights = sk::make_unique_device<float[]>(q, 1024, 0.5f);

float* p = weights.get();
q.parallel_for(sycl::range<1>{1024}, [=](sycl::id<1> i) { p[i] *= 2.0f; }).wait();
// Memory is released when the handle goes away.
```

Call the `malloc_*` functions **qualified**. They deliberately mirror SYCL's
names, so an unqualified call with SYCL arguments is ambiguous between
`naga::slabkit::malloc_device` and `sycl::malloc_device` — a namespace alias
like `sk` above is the comfortable way to work.

## Placement: which form to use

**For the device your queue was built for, use the queue forms.** Same device,
same context, nothing to get wrong.

```cpp
auto a = sk::malloc_device<float>(n, q);
auto b = sk::make_unique_device<float[]>(q, n, 1.0f);
```

**For any other device, use the device+context forms**, naming that device's
own context.

```cpp
sycl::context other_ctx{other_dev};        // that device's one context
auto c = sk::malloc_device<float>(n, other_dev, other_ctx);
auto d = sk::make_unique_device<float[]>(q, other_dev, other_ctx, n, 1.0f);
```

Using the device+context forms for **everything** is legitimate and keeps
allocation uniform. The single obligation is that the context you pass for a
queue's own device is the context that queue owns — so construct the context
first and the queue from it, as the example above does.

This matters because `sycl::free` resolves its allocator from the context it is
given: it takes the context's first device backend and the first device in it.
A context that spans several devices can therefore release a pointer through
the wrong allocator. One context per device, held for as long as the device is
used, makes that impossible.

## Memory objects

`DeviceArena` and `DevicePool` are memory, not allocators. They have no
allocation methods; they are simply placements the free functions accept.

```cpp
sk::DeviceArena arena{dev, ctx, 64 << 20};   // bump region
sk::DevicePool  pool {dev, ctx, 64 << 20, 1024};  // fixed-node free list

auto scratch = sk::malloc_device<float>(4096, arena);
auto node    = sk::malloc_device<float>(256,  pool);
```

Both are light handles over a shared implementation: copies refer to the same
slab, and **an allocation pins the slab it came from**. An allocation may
safely outlive every handle that named its memory —

```cpp
sk::unique_device_ptr<float[]> keep;
{
    sk::DeviceArena tmp{dev, ctx, 1 << 20};
    keep = sk::malloc_device<float>(1024, tmp);
}   // the arena handle is gone; `keep` is still valid
```

— because the slab is released by the last of {handles to the object,
allocations from it}. There is no lifetime rule to remember.

**Arena.** A bump: allocation is a pointer advance, it is lock-free, and there
is no per-allocation free and no rewind. It honors any alignment by aligning
the address the bump lands on, so padding is charged to the arena and shows up
in `used()`. Size it with slack if it will serve large alignments.

**Pool.** A fixed-node free list over `node_size` slots — `node_size` must be a
power of two. One node is O(1); a contiguous run is O(N/64). Allocations are
node-aligned and rounded up to whole nodes, and asking for an alignment larger
than the node size is an error. It is mutex-protected, and a double release is
caught by an assertion in debug builds.

## What you get back, and when it is null

`malloc_*` returns **null**, exactly as `sycl::malloc_*` does, and additionally
when an arena or pool cannot satisfy the request — exhausted, or too fragmented
to place a contiguous run. Check the handle.

A **zero-element** request is implementation-defined: slabkit passes it through
and adds no rule of its own, because SYCL implementations do not agree on it.
An arena or pool, having no SYCL call to mirror, returns null.

`make_*` throws **`std::bad_alloc`** when storage cannot be obtained, exactly as
`std::make_unique` does. Usage errors — a pool node size that is not a power of
two, an alignment a pool cannot honor — are **`std::invalid_argument`**. There
is no slabkit exception type.

## Objects: the make family

`make_unique_device` and `make_shared_device` offer std's forms: a single
object, `for_overwrite`, a value-initialized array, a filled array, and a copy
from host data. The initialization queue comes first and the placement second.

```cpp
auto one    = sk::make_unique_device<Params>(q, 3, 4.5f);       // constructor args
auto raw    = sk::make_unique_device_for_overwrite<Params>(q);  // storage only
auto zeros  = sk::make_unique_device<float[]>(q, n);            // value-initialized
auto filled = sk::make_unique_device<float[]>(q, n, 1.0f);      // filled
auto copied = sk::make_unique_device<float[]>(q, std::span<const float>{host});
```

**They block.** Initialization is enqueued only because that is how device
memory is written, and it is waited on before you get the handle — the object
is ready the instant the call returns. There are no event-returning variants.

**How the object gets there** depends on its type. A trivially copyable type is
built on the host and copied in. Anything else gets storage and then a single
kernel that placement-news it *on the device*, so its constructor runs where
the object lives. That is the point: an object holding a pointer into itself,
built on the host and copied in, would carry a host address into device memory.

Constructor arguments are copied into that kernel and reach the constructor as
const lvalues, so they must be device-copyable, and move-only arguments cannot
be expressed.

**The type gate.** `T` must be trivially destructible — a handle has no queue
on which to run a destructor, and never runs one — and **must not be
polymorphic**. Constructing on the device is exactly what would make a vtable
device-correct, but AdaptiveCpp's generic (SSCP) flow cannot resolve device
vtables at all: the call faults on the host backend and can return silently
wrong results on CUDA. slabkit rejects such types at compile time rather than
letting you find that out at runtime. Array forms additionally need trivially
copyable elements, since they initialize by fill or copy.

## Building and consuming

slabkit is a compiled shared library, and a [pixi](https://pixi.sh) source
package. From a pixi project:

```toml
[dependencies]
slabkit = { git = "https://github.com/NAGAGroup/SlabKit.git", tag = "..." }
```

then in CMake:

```cmake
find_package(slabkit CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE naga::slabkit)
```

From a non-pixi project, `pixi exec pixi-pack --create-executable` produces a
self-extracting archive; point `CMAKE_PREFIX_PATH` at the unpacked environment
and use the same `find_package`. The pixi pipeline is the only officially
supported build and consumption path.

To work on slabkit itself, `pixi run -e dev-omp test` builds and runs the suite
on the CPU via AdaptiveCpp's OpenMP backend; `-e dev-cuda` runs it on an NVIDIA
device.

## License

MIT — see [LICENSE](LICENSE).
