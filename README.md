# slabkit

USM memory for SYCL programs that run on several devices: owning handles, one
in-order stream of work, and copies that route themselves.

Built for AdaptiveCpp's generic compilation flow. It is a compiled shared
library, not a header-only one.

Platforms: linux-64 (x86-64-v3), linux-aarch64, osx-arm64 (macOS 26), win-64
(x86-64-v3) and win-arm64. Status: experimental. The API is still changing.

```cpp
#include <naga/slabkit/slabkit.hpp>
namespace sk = naga::slabkit;

sk::context ctx{sycl::device{sycl::gpu_selector_v}};
sk::queue   q{ctx};

auto data = sk::malloc_device<float>(1 << 20, q);   // owning handle
q.fill(data.get(), 1.0f, 1 << 20);
q.parallel_for(sycl::range<1>{1 << 20},
               [p = data.get()](sycl::id<1> i) { p[i[0]] *= 2.0f; });
q.wait();
```

## The pieces

| | |
|---|---|
| `context` | the device set, one `sycl::context` per backend, and the answers to what the machine can do |
| `queue` | one stream over that context: kernel tasks, host and memory tasks, placement, groups |
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

One logical stream over the context. Three kinds of submission:

- **Kernel tasks** (`submit`, `single_task`, `parallel_for`) run in the order they were submitted, on whichever device they name.
- **Host and memory tasks** (`async_host`, `memcpy`, `copy`, `memset`, `fill`) run off your thread. They are ordered against kernel tasks and group tasks in both directions, but are asynchronous with respect to each other.
- **Group tasks** (`submit_group`): the subtasks, any mix of the two kinds above, are unordered among themselves and count as one task to everything outside the group.

```cpp
sk::queue q{ctx};                     // optionally: an async handler

q.single_task([]{ ... });                        // on the default device
q.single_task(other_device, []{ ... });          // placed
q.parallel_for(sycl::range<1>{n}, kernel);       // range, nd_range, offset and reduction forms
q.submit([](sycl::handler& h) { ... });          // full command group
q.submit(sycl::property_list{sycl::property::command_group::AdaptiveCpp_retarget{dev}}, cgf);

q.wait();  q.wait_and_throw();  q.throw_asynchronous();
q.get_wait_list();                    // the events to hand to depends_on
```

Every shortcut has four forms: `(args)`, `(dev, args)`, `(deps, args)` and `(dev, deps, args)`, where
`deps` is a `std::vector<sycl::event>`. Ordering is by merging wait lists when you submit, never by waiting.

**A group is N submissions that are one step.** They run concurrently; the next
item waits for all of them.

```cpp
auto events = q.submit_group([&](sk::queue::group& g) {
    g.submit(dev_a, [&](sycl::handler& h) { h.parallel_for(range_a, kernel_a); });
    g.submit(dev_b, [&](sycl::handler& h) { h.parallel_for(range_b, kernel_b); });
});
```

**Host work runs on the host, off your thread.** `async_host` takes any callable. It runs after the
kernel work outstanding when you submitted it, and gives you an ordinary event.

```cpp
sycl::event done = q.async_host([p = data.get()] {
    use_on_the_host(p);              // runs after the kernel work above it
});
q.async_host(std::vector<sycl::event>{done}, [&] { ship_it(); });   // ordered after `done`
```

Ordering between two host or memory tasks is yours: hand the first one's event to the second, or use the
`_in_order` variant (`async_host_in_order`, `memcpy_in_order`, `copy_in_order`, `memset_in_order`,
`fill_in_order`), which joins the kernel order instead. Raw pointers are yours too: a task orders itself
against the stream, not against the bytes it names.

A group can mix kernel work and host work, which is how you run them side by side:

```cpp
q.submit_group([&](sk::queue::group& g) {
    g.submit(dev, [&](sycl::handler& h) { h.parallel_for(r, kernel); });
    g.async_host([&] { prepare_the_next_batch(); });
});
```

A host task's exception is asynchronous: it surfaces from `wait_and_throw` or
`throw_asynchronous`, never at the call.

`wait()` or `wait_and_throw()` is the last thing a program does with its queue before it shuts down:
destruction does not drain, and work still running at program exit is illegal.

Every member is safe to call from any thread.

An `unordered_queue` is the fully out-of-order stream: kernel tasks and `async_host` only, ordered
by nothing but the events you pass. It has no memory operations or groups; containers and
allocators use `queue`.

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

The four mirrors are members of the queue and return an event. **Every copy is a
host and memory task** (see above): it runs off your thread, after the kernel work it was
ordered behind, and nothing here blocks the caller. `memcpy` and `copy` find
their operands and choose a route — a direct device-to-device copy when the
devices share a backend, a host-side copy when both allocations are
host or shared memory (a `memcpy` in a host task, in any context, never staged),
one backend copy between a device allocation and host or shared memory of the same
context (or plain memory), and otherwise a staged copy through a pageable host
intermediate. A SYCL command touches only the USM of its own context, so the staged
form is what crosses contexts for a device allocation. Fills and sets of shared
memory go through SYCL, because a host loop migrates every page to the host and
back. OpenMP device memory is device-side like any other backend's.

**Warning:** copies that touch shared memory from the host assume the device
supports concurrent access to shared allocations. CUDA on Windows and WSL2 is the
canonical case where that guarantee does not hold: there, such a copy can fault if
a kernel is running on that device. Know your platform, and either wait for the
device first or use device memory. A
plain host pointer is a legal operand.

That ordering is not a nicety. A copy whose host operand is not USM in the
issuing queue's context has that operand read when the copy is *issued*, so a
dependency cannot protect it; slabkit never issues a copy until what it
depends on has already finished.

```cpp
q.memcpy(dst, src, bytes);
q.copy(src, dst, count);                      // typed
q.copy(device_ptr, host_vector.data(), count);
q.memset(p, 0, bytes);
q.fill(p, 3.5f, count);
// each also takes a device and/or a std::vector<sycl::event> of dependencies in front, and has an _in_order variant
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
a.resize(2 * n);                                 // old storage freed once its own copy is done; wait your kernels first
a.fill(2.0f);
a.data();                                        // the pointer kernels take

{ auto acc = a.get_access(); acc[0] = 5.0f; }    // syncs; mutable; shared/host kinds only
{ auto view = b.host_view(); use(view[0]); }     // syncs; read-only; every kind
```

Deallocation frees on your thread. `usm_vector` first waits for its own last
task; work you launched over the memory is yours to wait for before the
container drops it.

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
pixi run test                  # builds the library and its tests, runs them
```

The default environment carries every runtime backend for its platform: CUDA and
Level Zero on linux-64 and win-64, CUDA on linux-aarch64. OpenMP, and Metal on
macOS, come built in. The `dev` environment holds the full build closure with the
toolchain, for working on the library with CMake presets (`default`, `release`,
`dev`, `dev-debug`) and `scripts/configure.nu`, `build.nu` and `test.nu`.

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

## License

Apache-2.0. Copyright 2026 Jack Myers. See `LICENSE` and `NOTICE`.
