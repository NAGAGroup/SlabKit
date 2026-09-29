# slabkit

USM memory for SYCL programs that run on several devices: owning handles, one
in-order stream of work, and copies that route themselves.

Built for AdaptiveCpp's generic compilation flow. It is a compiled shared
library, not a header-only one.

```cpp
#include <naga/slabkit/slabkit.hpp>
namespace sk = naga::slabkit;

sk::context ctx{sycl::device{sycl::gpu_selector_v}};
sk::queue   q{ctx};

auto data = sk::malloc_device<float>(1 << 20, q);   // owning handle
sk::fill(q, data.get(), 1.0f, 1 << 20);
q.parallel_for(sycl::range<1>{1 << 20},
               [p = data.get()](sycl::id<1> i) { p[i[0]] *= 2.0f; });
q.wait();
```

## The pieces

| | |
|---|---|
| `context` | the device set, one `sycl::context` per backend, and the answers to what the machine can do |
| `queue` | one stream over that context: kernel work and host work, placement, groups and chains |
| `malloc_*` / `aligned_alloc_*` | uninitialized storage, SYCL's names and signatures |
| `make_unique_*` / `make_shared_*` | constructed objects, `std::make_unique`'s forms |
| `DeviceArena` / `DevicePool` | a bump region and a fixed-node free list, each over one slab |
| `memcpy` / `copy` / `memset` / `fill` | SYCL's four explicit operations, routed across devices and backends |
| `allocator<T>` | one std-conforming allocator over every kind and route |
| `usm_vector<T>` | a value-semantic array whose every mutation is enqueued |

Three USM kinds — `device`, `shared`, `host` — over six placements.

## Contexts

A USM allocation belongs to a SYCL context, and any access must use that
context. A copy between two devices is legal only within one context. Each
platform has a default context spanning all its devices, and that is the one
slabkit holds — one per backend, retrieved, never constructed.

```cpp
sk::context by_device  {my_device};                       // one device
sk::context by_devices {{dev_a, dev_b}};                  // several
sk::context by_platform{my_platform};                     // every device of a platform
sk::context by_need    {{sk::aspect::gpu, sk::aspect::has_shared}};   // choose the support matrix
```

The last form is usually the one you want: state the properties your program
needs and let the machine answer. An unsatisfiable set throws
`std::invalid_argument` rather than handing back an empty context.

```cpp
ctx.devices();              // what was selected
ctx.backends();             // deduplicated, one per platform
ctx.context_of(dev);        // the backend context — pass this to any SYCL call that wants one
ctx.get_pointer_type(p);    // kind::device | kind::shared | kind::host
ctx.get_pointer_device(p);  // which device owns it
```

**Never construct a `sycl::context` yourself.** Use `context_of`. A context
spanning two platforms is illegal, AdaptiveCpp cannot diagnose it, and the
failure is a crash on free rather than an error.

## The queue

One logical in-order stream. Work submitted later observes work submitted
earlier, on whichever device it ran.

```cpp
sk::queue q{ctx};                     // optionally: async handler, host-thread count

q.single_task([]{ ... });                        // on the default device
q.single_task(other_device, []{ ... });          // placed
q.parallel_for(sycl::range<1>{n}, kernel);       // range, nd_range, offset and reduction forms
q.submit([](sycl::handler& h) { ... });          // full command group
q.submit(sycl::property_list{sycl::property::command_group::AdaptiveCpp_retarget{dev}}, cgf);

q.wait();  q.wait_and_throw();  q.throw_asynchronous();
q.get_wait_list();                    // the events to hand to depends_on
```

**A group is N submissions that are one step.** They run concurrently; the next
item waits for all of them.

```cpp
auto events = q.submit_group([&](sk::queue::group& g) {
    g.submit(dev_a, [&](sycl::handler& h) { h.parallel_for(range_a, kernel_a); });
    g.submit(dev_b, [&](sycl::handler& h) { h.parallel_for(range_b, kernel_b); });
});
```

**Host work is the other half of the stream.** Anything that runs on the host
— including every copy — is host work. It runs off your thread, after the
kernel work outstanding when you submitted it, and gives you an ordinary
event.

```cpp
sycl::event done = q.submit_host([p = data.get()] {
    use_on_the_host(p);              // runs after the kernel work above it
});
q.sync_host_work();                  // wait the host side alone
```

Kernel work is in order with kernel work; host work carries no dependency on
other host work; and the two interlock — all outstanding host work finishes
before the next kernel item starts, and all outstanding kernel work finishes
before the next host item starts. That is what makes the stream feel in order
without forcing everything into one line.

When host work *must* be ordered against other host work, chain it. A chain
is one host item as far as the stream is concerned:

```cpp
q.submit_chain([&](sk::queue::chain& c) {
    c.then([&] { stage_it(); });
    c.then([&] { ship_it(); });      // begins after the first completes
});
```

And when you want host work to run *alongside* kernel work rather than
interlocked with it, put both in a group:

```cpp
q.submit_group([&](sk::queue::group& g) {
    g.submit(dev, [&](sycl::handler& h) { h.parallel_for(r, kernel); });
    g.submit_host([&] { prepare_the_next_batch(); });
});
```

A host task's exception is asynchronous: it surfaces from `wait_and_throw`,
`throw_asynchronous` or `sync_host_work_and_throw`, never at the call.

One limit worth knowing: AdaptiveCpp's OpenMP backend offers a single
execution lane per device for the whole process, so host items run one at a
time even though nothing orders them. slabkit does not add a dependency
between them, so an implementation with more lanes runs them concurrently
with no change to your code.

Every member is safe to call from any thread.

## Allocating

Storage keeps SYCL's names, so the kind is in the function name:

```cpp
auto d = sk::malloc_device<float>(n, q);            // on q's default device
auto e = sk::malloc_device<float>(n, q, dev);       // placed
auto s = sk::malloc_shared<float>(n, q, dev);
auto h = sk::malloc_host<float>(n, q);              // host kind: no device argument
auto a = sk::aligned_alloc_device<float>(256, n, q);
```

Objects keep `std::make_unique`'s forms, block until ready, and throw
`std::bad_alloc`:

```cpp
auto one   = sk::make_unique_device<Params>(q, dev, arg1, arg2);
auto many  = sk::make_unique_device<float[]>(q, n, 1.0f);        // fill
auto copied= sk::make_unique_shared<float[]>(q, std::span{host_data});
auto raw   = sk::make_unique_device_for_overwrite<float[]>(q, n);
auto shared_obj = sk::make_shared_device<Params>(q, args...);
```

Element types must be trivially destructible and non-polymorphic — AdaptiveCpp's
generic flow cannot resolve virtual dispatch on a device, and the failure is a
wrong answer rather than an error, so the type gate rejects it at compile time.

### Arenas and pools

Both take their kind at construction and own one slab. Handles from them keep
the slab alive; the object itself may go out of scope first.

```cpp
sk::DeviceArena arena{q, dev, sk::kind::device, 64 << 20};
auto block = sk::malloc_device<float>(n, arena);    // the name must match the kind

sk::DevicePool pool{q, dev, sk::kind::shared, 64 << 20, /*node_size=*/4096};
auto node = sk::malloc_shared<std::byte>(4096, pool);
```

An arena is a budget and never falls back; a pool hands out fixed nodes and its
allocations are node-size aligned.

## Copying

The four mirrors take the stream and return an event. **Every copy is host
work** (see above): it runs off your thread, after the kernel work it was
ordered behind, and nothing here blocks the caller. `memcpy` and `copy` find
their operands and choose a route — a direct device-to-device copy when the
devices share a backend, a host-side copy when both allocations are
host-accessible, and otherwise a staged copy through a host intermediate. A
plain host pointer is a legal operand.

That ordering is not a nicety. A copy whose host operand is not USM in the
issuing queue''s context has that operand read when the copy is *issued*, so a
dependency cannot protect it; slabkit never issues a copy until what it
depends on has already finished.

```cpp
sk::memcpy(q, dst, src, bytes);
sk::copy(q, src, dst, count);                 // typed
sk::copy(q, device_ptr, host_vector.data(), count);
sk::memset(q, p, 0, bytes);
sk::fill(q, p, 3.5f, count);
// each also takes a std::vector<sycl::event> of dependencies
sk::context::copy_routes(kind_a, dev_a, kind_b, dev_b);   // what routes exist, best first
```

There is always a route; the staged one is always available and always last.

## Containers

`allocator<T>` is a standard allocator carrying where memory comes from:

```cpp
sk::allocator<float> on_device{q, sk::kind::device, dev};
sk::allocator<float> in_pool  {q, pool};
std::vector<float, sk::allocator<float>> v{sk::allocator<float>{q, sk::kind::shared, dev}};
```

`std::vector` works over the **shared** and **host** kinds. Over the device kind
it allocates correctly but element access is a host dereference of device
memory — use `usm_vector` instead.

`usm_vector<T>` has value semantics and enqueues everything:

```cpp
sk::usm_vector<float> a{n, 1.0f, on_device};    // enqueued, returns immediately
sk::usm_vector<float> b{a, on_other_device};    // routed copy across devices or kinds
b = a;                                           // routed copy
a.resize(2 * n);                                 // old storage released after the work using it
a.fill(2.0f);
a.data();                                        // the pointer kernels take

{ auto acc = a.get_access(); acc[0] = 5.0f; }    // syncs; mutable; shared/host kinds only
{ auto view = b.host_view(); use(view[0]); }     // syncs; read-only; every kind
```

Deallocation never happens on your thread: it is posted, and the memory is
returned after the work using it has finished. So `resize` and destruction
cannot pull memory out from under a running kernel.

Both containers refuse operations between two different streams; use `copy` on
`data()` with explicit events for that.

## Two traps

**Call `malloc_*` qualified.** `sycl::malloc_device` exists too, and ADL reaches
it from any call whose arguments include a `sycl::queue`. Write
`sk::malloc_device<float>(n, q)`, never `malloc_device<float>(n, q)`.

**Take turns.** Do not touch shared memory from the host while a kernel that
uses it is running. No device reports support for concurrent access through
AdaptiveCpp, and on Windows and WSL the NVIDIA driver genuinely does not have
it — the result is corruption or a fault, not a slowdown. The routed mirrors
enforce this for you; `get_access()` and `host_view()` synchronize before
handing you a view, and what you do while holding one is yours to get right.

## What each backend actually does

No query distinguishes these, so they are written down instead. The answers
slabkit gives for capabilities it cannot query are pinned to the AdaptiveCpp
version this library is built against.

| backend | notes |
|---|---|
| CUDA, Linux | shared memory is managed memory and migrates on page faults; peer access between devices is enabled automatically |
| CUDA, Windows and WSL | no full managed memory: prefetch is unavailable and concurrent host/device access is unsupported |
| HIP | shared memory needs XNACK, which is available on limited hardware and often off by default; without it, shared behaves as pinned host memory — correct and slow |
| Level Zero, discrete Intel | migration is per allocation, not per page: a host touch moves the whole thing, so keep host-touched data out of a large shared arena |
| Level Zero, integrated Intel | unified memory, page-granular; shared is a good default |
| Metal | USM is emulated; no `double`, no 64-bit atomics |
| Vulkan | device allocations only — no shared, no host |
| OpenMP host | every kind is host memory; `get_pointer_type` answers `host` for any address, including device allocations from other backends |

Cross-backend ordering is not free: the stream waits on the host at a backend
transition, because dependencies between backends are not reliably honoured by
the runtime.

## Building

```
pixi run -e dev-cuda build      # or dev-omp, dev-l0, dev-ocl, dev-amd
pixi run -e dev-cuda test
```

Each environment carries the backends it names and constrains the runtime to
them, so a test run exercises what its name says.

Consuming it:

```cmake
find_package(slabkit REQUIRED)
target_link_libraries(my_target PRIVATE naga::slabkit)
```

## Platform notes

### CUDA on WSL2: shared memory

CUDA under WSL2 (as on Windows) has no concurrent managed access. The host may
not read or write ANY shared-kind (managed) allocation while ANY kernel or copy
is executing on the GPU, even work that does not touch that allocation, and even
work on another stream. Doing so is a segmentation fault, typically intermittent.

`usm_vector::get_access()` waits for the container's own work only. On WSL2,
wait every stream that uses the device (`queue::wait()`) before touching shared
memory from the host, or use device-kind containers with `host_view()` and
`from_host()`, which are safe everywhere.
