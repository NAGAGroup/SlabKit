# Spec 02.A — slabkit surface

The slabkit surface. Binding contract; declarations only. AdaptiveCpp's
generic compilation flow is the only implementation targeted; contracts are
borrowed from SYCL and the C++ standard library, never invented.

## 1. What slabkit is

USM memory for SYCL programs that run on several devices, with owning
handles and one in-order stream of work. Seven parts, each specified below:

1. `context` — the device set, one `sycl::context` per backend, and the
   answers to what the machine can do.
2. `queue` — one in-order stream over the context, mirroring `sycl::queue`,
   with placement, concurrent groups, and a host lane.
3. The **storage** (`malloc_*`) and **object** (`make_*`) families over
   three USM kinds and six placements, yielding one handle type.
4. **Capabilities** — the `aspect` vocabulary, three-valued answers, copy
   routes.
5. The four **routed mirrors** — `memcpy`, `copy`, `memset`, `fill`.
6. One **allocator** type over all kinds and routes, with synchronous release.
7. `usm_vector` — a value-semantic array whose every mutation is enqueued.

**Contracts are borrowed.** `malloc_*` behaves as `sycl::malloc_*`: null
exactly when the underlying call returns null (zero elements is
implementation-defined and slabkit adds no rule), null when an arena or pool
cannot satisfy the request. `make_*` behaves as `std::make_*`: returns with
the object ready, throws `std::bad_alloc`. Usage errors are
`std::invalid_argument`. There is no library exception type.

**Contexts are the platform's.** Every USM allocation belongs to a SYCL
context and any access must use that context (SYCL 2020 §4.8.2); a copy
between two devices is legal only within one context. Each platform has a
default context spanning all its devices (§4.6.2), and `queue{device}`'s
context is it. slabkit never constructs a `sycl::context`; it retrieves the
platform default and holds one per backend. A context spanning two
platforms is illegal and AdaptiveCpp cannot diagnose it.

**Take turns.** Concurrent host and device access to shared memory is not
assumed anywhere: no device reports `usm_atomic_shared_allocations` through
AdaptiveCpp's public API, and Windows/WSL NVIDIA drivers do not support it.
Host access happens after the work that uses the memory has completed. The
routed mirrors enforce this themselves; the container's two host-access
calls sync first and leave the rest to the caller.

**Shape.** A compiled shared library. Everything element-type-independent
compiles once; only device-side construction is necessarily a template.
Memory objects own their slabs as slabless allocations from this library, so
`sycl::free` is reached from exactly one deleter and no class has a
destructor of its own.

## 2. `context`

The set of devices a program places work on, with each device's backend
default `sycl::context`, and the capability answers for that set computed at
construction. Immutable. A light handle: copies share state.

```cpp
namespace naga::slabkit {

using backend_id = sycl::platform;   // two devices share a backend iff their platforms compare equal

class context {
public:
    // Explicit devices.
    explicit context(std::vector<sycl::device> devices);
    explicit context(const sycl::device& dev);
    // Every root device of the listed platforms.
    explicit context(std::vector<sycl::platform> platforms);
    explicit context(const sycl::platform& platform);
    // Every available device (optionally within `platforms`) satisfying ALL of `required`.
    // A pair aspect in `required` means: with every other selected device of its backend.
    explicit context(std::initializer_list<aspect> required);
    context(std::vector<sycl::platform> platforms, std::initializer_list<aspect> required);
    context(const sycl::platform& platform,        std::initializer_list<aspect> required);
    // Throws std::invalid_argument: empty result, a device listed twice.

    std::span<const sycl::device> devices()  const;   // selection order: platform order, then device order
    std::span<const backend_id>   backends() const;   // deduplicated, first-seen order
    bool       contains  (const sycl::device& dev) const;
    backend_id backend_of(const sycl::device& dev) const;                 // throws if !contains
    const sycl::context& context_of(const sycl::device& dev) const;       // the backend default; throws if !contains
    const sycl::context& context_of(const backend_id& b)     const;

    // Devices of this context satisfying A, read from the answers cached at construction.
    //   device aspect -> std::vector<sycl::device>
    //   pair aspect   -> std::vector<std::pair<sycl::platform, std::vector<sycl::device>>>
    //                    one group per backend; every pair within a group satisfies A;
    //                    groups are the connected components under the pair relation.
    template <aspect A> typename selection<A>::type select_devices() const;

    // SYCL's pointer queries, tried against every backend context held here.
    // Throw errc::invalid for a pointer none of them recognises.
    kind         get_pointer_type  (const void* p) const;
    sycl::device get_pointer_device(const void* p) const;

    // Machine facts, independent of any instance (§5).
    template <aspect A> static answer satisfies(const sycl::device& dev);
    template <aspect A> static answer satisfies(const sycl::device& a, const sycl::device& b);
    static std::vector<route> copy_routes(kind src_kind, const sycl::device& src,
                                          kind dst_kind, const sycl::device& dst);
};

} // namespace naga::slabkit
```

Rules:

- Construction is the only place a backend context is obtained, and it is
  retrieved (the platform default), never created.
- A device outside the set is `std::invalid_argument` everywhere downstream.
- The same physical device visible through two backends is two devices;
  filtering that is the program's (`ACPP_VISIBILITY_MASK`).
- Selection by aspects is a filter over `sycl::device::get_devices()`;
  `unknown` answers are not selected; the first selected device is a stable
  default a program may adopt. Policy ("GPU if present, else CPU") is two
  attempts by the caller, not an aspect.
- The answers for the instance's own devices — including the same-backend
  pair measurements — are computed once at construction and read by
  `select_devices`, the routed mirrors and `usm_vector`. There is no
  process-wide state.

## 3. `queue`

One stream of work over a `context`, with `sycl::queue`'s surface. A light
handle: copies are the same stream. Every member is safe to call from any
thread.

**Two kinds of work.** *Kernel work* is what runs on a device. *Host work* is
what runs on the host — including every memory operation, since a copy is
host-issued. That is the axis the stream is built on, not memory versus
compute.

**How they order.** Kernel work is in order with kernel work. Host work
carries **no dependency** on other host work. The two interlock: all
outstanding host work completes before any kernel work begins, and all
outstanding kernel work completes before any host work begins. The stream
therefore alternates phases and behaves in order from outside.

Whether two host items *execute* concurrently is the backend's to decide, and
AdaptiveCpp's OpenMP backend offers one execution lane per device for the
whole process — so host items run one at a time there. That is a property of
the runtime, not of this contract: the absence of a dependency is what
slabkit promises, and an implementation with more lanes delivers the
concurrency without any change here.

Internally, per backend in the context: one in-order and one out-of-order
`sycl::queue`, both `AdaptiveCpp_retargetable`, on `context_of(backend)`,
created lazily. Plus one out-of-order **host lane** on the OpenMP backend,
which always exists.

```cpp
namespace naga::slabkit {

class queue {
public:
    explicit queue(context ctx, sycl::async_handler handler = {});

    const context&      get_context() const;
    const sycl::device& get_device()  const;   // the default device: ctx.devices().front()

    // ---- kernel work: sycl::queue's surface ---------------------------------
    // `props` may carry AdaptiveCpp_retarget; a device argument is that property
    // applied by slabkit. Every shortcut is implemented through submit, never through
    // sycl::queue's shortcuts, and every shortcut has the forms:
    //   (cgf...), (dev, cgf...), (deps, cgf...), (dev, deps, cgf...)
    template <class CGF> sycl::event submit(CGF&& cgf);
    template <class CGF> sycl::event submit(const sycl::property_list& props, CGF&& cgf);
    template <class CGF> sycl::event submit(const sycl::device& dev, CGF&& cgf);
    // single_task, parallel_for (range, nd_range, offset, reduction forms) as sycl::queue declares them

    // ---- host work ----------------------------------------------------------
    // Runs on the host lane, off the caller's thread, after the kernel work
    // outstanding when it was submitted. Parallel with other host work.
    template <class F> sycl::event submit_host(F&& f);

    // Host work items forced in order with each other: the opposite of a group.
    // Each item begins after the previous one completes; the chain is ONE host-work
    // item as far as the stream is concerned.
    class chain {
    public:
        template <class F> sycl::event then(F&& f);
    };
    template <class F> sycl::event submit_chain(F&& f);    // f(chain&); returns the chain's last event

    // ---- one step of concurrent work ---------------------------------------
    // Members run concurrently and are one step: each depends on the previous step,
    // none on each other, and the next step depends on all of them. A group may mix
    // kernel work and host work, which is how the interlock is opted out of.
    class group {
    public:
        template <class CGF> sycl::event submit(CGF&& cgf);
        template <class CGF> sycl::event submit(const sycl::property_list& props, CGF&& cgf);
        template <class CGF> sycl::event submit(const sycl::device& dev, CGF&& cgf);
        template <class F>   sycl::event submit_host(F&& f);
        template <class F>   sycl::event submit_chain(F&& f);   // contributes the chain's LAST event
    };
    template <class F> std::vector<sycl::event> submit_group(F&& f);   // f(group&)

    // ---- waiting ------------------------------------------------------------
    void wait();                      // everything, kernel and host
    void wait_and_throw();
    void sync_host_work();            // the host side alone
    void sync_host_work_and_throw();
    void throw_asynchronous();
    std::vector<sycl::event> get_wait_list() const;   // union over the internal queues (ACPP_EXT_QUEUE_WAIT_LIST)
};

} // namespace naga::slabkit
```

Rules:

- **The interlock.** A kernel item depends on the outstanding host events and
  the kernel tail, and absorbs that host set. A host item depends on the
  kernel tail and **not** on other host items. A group's members depend on
  the previous step and not on each other, whatever kind they are.
- **Host work is an OpenMP `single_task` whose body is an indirect call**
  through a function pointer into ordinary host code. A direct call to
  complex host code from a kernel fails the generic flow's JIT; an indirect
  call does not, and the two captured values — a function pointer and a
  `void*` — are trivially copyable, which kernel captures must be.
- **A host task must never submit work to the OpenMP backend and wait on
  it.** That deadlocks: the wait holds the very lane the submitted work
  needs. Work on *other* backends may be issued and waited freely — that does
  not consume the host lane's capacity. A copy whose operands are both on the
  OpenMP backend is therefore a plain host `memcpy`, never a submission.
- **Ordering across backends is by `depends_on`, and it holds**, because the
  consumer of every cross-backend dependency is the host lane, which honours
  foreign events. slabkit never submits a non-OpenMP operation carrying a
  dependency on another backend's event. That restriction is load-bearing: a
  copy whose host operand is not USM in the issuing queue's context has that
  operand read when the copy is *issued*, so a dependency cannot protect it —
  measured wrong 40 times in 40, indistinguishable from no dependency at all.
- **Placement** is `AdaptiveCpp_retarget{dev}` routed to that device's
  backend pair — the extension's proper use: same backend, same context. A
  device outside the context is `std::invalid_argument`.
- **Errors.** `throw_asynchronous` and `wait_and_throw` forward to every
  internal queue including the host lane, and the handler is installed on
  each. A host task's exception is asynchronous: the stream holds it and
  rethrows it from `throw_asynchronous`, `wait_and_throw` or
  `sync_host_work_and_throw`. It is never thrown at the caller of
  `submit_host`, and it is never a future.
- **Thread-safety** is one internal mutex over the lane map and step state,
  never held while waiting or submitting to a `sycl::queue`.
- Not provided: memory shortcuts on the queue (§6 owns the routed forms),
  barriers (the interlock is the barrier), priorities.

## 4. Kinds, handles, storage, objects

```cpp
namespace naga::slabkit {

enum class kind { device, shared, host };

// ---- handles: three deleters, one handle type ---------------------------------
struct OsFree    { sycl::context ctx; };                                     // sycl::free(p, ctx)
struct PoolFree  { std::shared_ptr<detail::PoolImpl>  impl; size_t nodes; }; // impl->release(p, nodes)
struct ArenaFree { std::shared_ptr<detail::ArenaImpl> impl; };               // nothing: the arena owns the bytes
struct UsmDeleter { std::variant<OsFree, PoolFree, ArenaFree> how; void operator()(void* p) const noexcept; };
template <class T> using unique_usm_ptr = std::unique_ptr<T, UsmDeleter>;
template <class T> using shared_usm_ptr = std::shared_ptr<T>;              // deleter erased

// ---- placements ---------------------------------------------------------------
//   (queue, device)                  -> that device, its backend's context          PRIMARY
//   (queue)                          -> the queue's default device
//   (sycl::queue)                    -> SYCL's own form, kept for interop
//   (sycl::device, sycl::context)    -> SYCL's own form, kept for interop
//   (DeviceArena&) / (DevicePool&)   -> the object's kind and device
// The host kind takes a queue (for the backend) and NO device: a device argument is a compile error.

// ---- STORAGE: SYCL's names, SYCL's signatures, a handle for `count` elements ---
template <class T> unique_usm_ptr<T[]> malloc_device(size_t count, const queue& q, const sycl::device& dev, const sycl::property_list& props = {});
template <class T> unique_usm_ptr<T[]> malloc_device(size_t count, const queue& q,                          const sycl::property_list& props = {});
template <class T> unique_usm_ptr<T[]> malloc_shared(size_t count, const queue& q, const sycl::device& dev, const sycl::property_list& props = {});
template <class T> unique_usm_ptr<T[]> malloc_shared(size_t count, const queue& q,                          const sycl::property_list& props = {});
template <class T> unique_usm_ptr<T[]> malloc_host  (size_t count, const queue& q,                          const sycl::property_list& props = {});
// + the (sycl::queue) and (device, context) forms of each; aligned_alloc_{device,shared,host} with a leading alignment;
// + malloc_<kind>(count, DeviceArena&) and (count, DevicePool&): the name must match the object's kind, else std::invalid_argument.
// <std::byte> is the void* form. Alignment: queue/device forms as SYCL (the runtime may ignore the request);
// an arena honors any alignment at the landing address and charges the padding to itself;
// a pool's allocations are node_size-aligned and a larger request is std::invalid_argument.

// ---- OBJECTS: std::make_* in every form; initialization queue first, placement second; BLOCKS ---
template <class T, class... Args> unique_usm_ptr<T> make_unique_device(const queue& q, const sycl::device& dev, Args&&... args);
template <class T, class... Args> unique_usm_ptr<T> make_unique_device(const queue& q,                          Args&&... args);
template <class T, class... Args> unique_usm_ptr<T> make_unique_shared(const queue& q, const sycl::device& dev, Args&&... args);
template <class T, class... Args> unique_usm_ptr<T> make_unique_shared(const queue& q,                          Args&&... args);
template <class T, class... Args> unique_usm_ptr<T> make_unique_host  (const queue& q,                          Args&&... args);
// For each kind and each placement P (queue+device · queue · sycl::queue · (device, context) · arena · pool):
//   make_unique_<kind>              (P, Args&&...)                      // single object
//   make_unique_<kind>_for_overwrite(P)                                 // storage only
//   make_unique_<kind><TArr>        (P, size_t n)                       // value-initialized array
//   make_unique_<kind>_for_overwrite<TArr>(P, size_t n)
//   make_unique_<kind><TArr>        (P, size_t n, const E& value)       // fill
//   make_unique_<kind><TArr>        (P, std::span<const E> src)         // copy
//   and make_shared_<kind> in the identical forms, returning shared_usm_ptr.

// ---- memory objects: placement only; kind at construction; no allocation methods ---
class DeviceArena {   // bump region over one slab; lock-free; no per-allocation free, no rewind
public:
    DeviceArena(const queue& q, const sycl::device& dev, kind k, size_t bytes, const sycl::property_list& props = {});
    // + (sycl::queue, kind, bytes) and (device, context, kind, bytes)
    kind get_kind() const;  size_t capacity() const;  size_t used() const;  size_t remaining() const;
    const sycl::device& device() const;  const sycl::context& context() const;
};
class DevicePool {    // fixed-node free list over one slab; node_size a power of two; mutex-protected
public:
    DevicePool(const queue& q, const sycl::device& dev, kind k, size_t bytes, size_t node_size = 1024, const sycl::property_list& props = {});
    // + (sycl::queue, kind, bytes, node_size) and (device, context, kind, bytes, node_size)
    kind get_kind() const;  size_t node_size() const;  size_t slot_count() const;  size_t free_count() const;  size_t outstanding() const;
    const sycl::device& device() const;  const sycl::context& context() const;
};

} // namespace naga::slabkit
```

Rules:

- **The `queue` forms are primary**: they resolve `(dev, context_of(dev))`
  themselves, so the context discipline cannot be violated through them. The
  SYCL-native forms exist for code that already holds a `sycl::queue` or a
  `(device, context)` pair; the context passed must be that device's backend
  default.
- **`make_*` initialization** runs on the `queue` placed on the target
  device (retargeted within its backend) and is waited on before return.
  Mechanism by type: trivially copyable — host-constructed and written with
  `fill`/`memcpy`, or a plain host write for shared and host kinds;
  otherwise — storage, then one `single_task` performing placement-new with
  the arguments copied into the kernel as const lvalues.
- **The type gate** (concept-checked, so a rejection is observable): `T`
  trivially destructible (the handle never runs a destructor); `T` not
  polymorphic (AdaptiveCpp's generic flow cannot resolve device vtables —
  the call fails on the host backend and returns silently wrong results on
  CUDA); every argument device-copyable; array forms trivially copyable
  elements. Constructor device-callability is the compiler's check.
- **Memory objects** are light handles over a shared implementation; copies
  refer to the same slab; allocations pin it; the slab dies with the last of
  handles and allocations. The slab is itself a slabless allocation from
  this library.
- No kind parameter on the families (the name is the kind, as SYCL spells
  it); no allocation from a bare `context`; `get_pointer_type` is how a
  handle's kind is learned.

## 5. Capabilities

```cpp
namespace naga::slabkit {

enum class aspect {
    // every sycl::aspect, same names, same meaning (forwarded)
    cpu, gpu, accelerator, fp16, fp64, atomic64, queue_profiling, /* ... */
    // the honest USM trio: what the backend can do, not what AdaptiveCpp hard-codes
    has_device, has_shared, has_host,
    // backend identity
    backend_omp, backend_cuda, backend_hip, backend_level_zero, backend_metal, backend_vulkan,
    // pair aspects; in a selector: with every other selected device of the same backend
    shares_context,   // same backend -> same platform default context
    d2d,              // a memcpy between device allocations on the two is a direct backend copy
    device_p2p,       // a kernel on one may dereference a device allocation of the other
    shared_p2p,       // a kernel on one may dereference a shared allocation of the other
};

enum class answer     { yes, no, unknown };
enum class provenance { queried, documented, measured };

struct route {
    enum class via { d2d, host_memcpy, staged } method;
    provenance source;
};

} // namespace naga::slabkit
```

Rules:

- **Three sources, fixed precedence:** measured > queried (a backend that
  answers the SYCL aspect honestly: OpenCL, OMP) > documented (slabkit's
  table for the backend, pinned to an AdaptiveCpp version and stated as
  such in the README). AdaptiveCpp hard-codes the USM trio on CUDA, HIP and
  Level Zero and does not map `usm_atomic_shared_allocations` at all, which
  is why the honest trio exists.
- **Pair aspects across backends are `no`**, never `unknown`.
- **A documented `unknown` for a same-backend pair is measured on the
  call** — a small allocation on one device dereferenced by a kernel on the
  other, result returned, nothing retained. `context` runs these once for
  its own pairs at construction and caches the answers in the instance.
- **`copy_routes` is derived, never empty:** `d2d` if `shares_context` is
  `yes` and both kinds are resident on their backend; `host_memcpy` if both
  allocations are host-accessible (shared or host kinds) — legal regardless
  of backend, since host code may touch host-accessible USM; `staged` —
  D2H into a host intermediate, then H2D — always available and always
  last. Best first.
- Performance is not an aspect: Intel dGPUs migrate whole allocations, AMD
  shared memory without XNACK behaves as pinned host memory, WSL has no
  prefetch — README text beside the backend, not answers.

Documented answers within one backend: OMP — one device, pair aspects not
applicable; CUDA — `d2d` yes, `device_p2p` and `shared_p2p` unknown (peer
access is enabled automatically, hardware-limited); HIP — as CUDA; Level
Zero — `d2d` yes where the driver allows multi-device contexts,
`device_p2p`/`shared_p2p` unknown; Metal, Vulkan — one device. USM trio:
queried on OpenCL and OMP; documented true/true/true on CUDA, HIP, Level
Zero and Metal (emulated); true/false/false on Vulkan.

## 6. The routed mirrors

```cpp
namespace naga::slabkit {

sycl::event memcpy(queue& q, void* dst, const void* src, size_t bytes);
sycl::event memcpy(queue& q, void* dst, const void* src, size_t bytes, const std::vector<sycl::event>& deps);
template <class T> sycl::event copy  (queue& q, const T* src, T* dst, size_t count);
template <class T> sycl::event copy  (queue& q, const T* src, T* dst, size_t count, const std::vector<sycl::event>& deps);
sycl::event memset(queue& q, void* p, int value, size_t bytes);
sycl::event memset(queue& q, void* p, int value, size_t bytes, const std::vector<sycl::event>& deps);
template <class T> sycl::event fill  (queue& q, void* p, const T& pattern, size_t count);
template <class T> sycl::event fill  (queue& q, void* p, const T& pattern, size_t count, const std::vector<sycl::event>& deps);

} // namespace naga::slabkit
```

Rules:

- **Every mirror is host work** (§3): one host item, returning its event, off
  the caller's thread, ordered after the kernel work outstanding when it was
  called and parallel with other host work. Nothing here blocks the caller.
- `memcpy` and `copy` locate their operands through the queue's context
  (`get_pointer_type`, `get_pointer_device`; a non-USM pointer is a host
  operand) and take `copy_routes(...).front()`. Inside the host task:
  **d2d** — one `sycl::queue::memcpy` on the destination's backend queue,
  waited there; **host_memcpy** — a plain `std::memcpy` between the two
  host-accessible allocations; **staged** — a chain, D2H into an
  operation-owned host intermediate and then H2D.
- **The copy is issued only after its dependencies are already satisfied**,
  because the host task carrying it is what waits. So no submission ever
  carries a dependency on another backend's event, and the issue-time read of
  a host operand cannot see unwritten memory. This is what makes a pageable
  intermediate safe; it is not made safe by the intermediate's kind, and an
  intermediate cannot be USM in two contexts at once anyway.
- **A copy whose operands are both on the OpenMP backend is a plain host
  `memcpy`**, never a submission — a host task that submits to the OpenMP
  backend and waits on it deadlocks.
- Take turns is enforced here: the host task runs after the kernel work it
  was ordered behind, so slabkit never touches shared memory concurrently
  with a kernel.
- `memset` and `fill` have one operand and never route: they run on the
  operand's owning device, or as a host write for a trivially-copyable
  pattern into shared or host memory.
- A non-USM pointer is a host operand, always. There is no unrecognised
  operand.
- Not provided: `prefetch`, `mem_advise` (backend hints; call SYCL's
  directly), staging-buffer reuse.

## 7. The allocator

```cpp
namespace naga::slabkit {

template <class T>
class allocator {
public:
    using value_type = T;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap            = std::true_type;
    using is_always_equal                        = std::false_type;

    allocator(queue q, kind k, const sycl::device& dev);   // OS route
    allocator(queue q, kind k);                            // OS route, default device; the host kind uses this form
    allocator(queue q, DeviceArena arena);                 // arena route: the object's kind and device
    allocator(queue q, DevicePool  pool);                  // pool route:  the object's kind and device
    template <class U> allocator(const allocator<U>&);     // rebind keeps state

    T*   allocate(std::size_t n);                          // throws std::bad_alloc
    void deallocate(T* p, std::size_t n) noexcept;         // waits, then frees; see rules
    void deallocate(T* p, std::size_t n, const std::vector<sycl::event>& deps) noexcept;  // waits deps first

    queue        get_queue()  const;
    kind         get_kind()   const;
    sycl::device get_device() const;                       // throws for the host kind
    template <class U> bool operator==(const allocator<U>&) const;   // state equality
};

} // namespace naga::slabkit
```

Rules:

- `allocate` is the storage family with `bad_alloc` instead of null. The
  pool route tries the pool, then the OS route on exhaustion; the arena
  route never falls back — an arena is a budget.
- `deallocate` frees on the calling thread, by the route the address belongs
  to (pool if inside the pool's slab, otherwise OS; arena route: nothing). The
  overload taking events waits for them first. Work still using the memory is
  the caller's to wait for.
- std containers over the shared and host kinds work unchanged. Over the
  device kind they are a usage error: element access is a host dereference
  of device memory. The device kind is for `usm_vector` and for handle
  users.
- No construct/destroy hooks (objects are trivially destructible), no
  allocation hints.

## 8. `usm_vector`

A contiguous array of trivially-copyable elements over `allocator<T>` with
value semantics. Every mutation is an item of the container's queue and
returns its event; nothing blocks except the two host-access calls.

```cpp
namespace naga::slabkit {

template <class T>   // trivially copyable, trivially destructible
class usm_vector {
public:
    using value_type = T;  using allocator_type = allocator<T>;  using size_type = std::size_t;

    explicit usm_vector(allocator_type alloc);
    usm_vector(size_type n, allocator_type alloc);                       // value-initialized, enqueued
    usm_vector(size_type n, const T& value, allocator_type alloc);       // fill, enqueued
    usm_vector(std::span<const T> src, allocator_type alloc);            // H2D from a copy of src, enqueued
    usm_vector(std::initializer_list<T> src, allocator_type alloc);

    usm_vector(const usm_vector& other);                                 // routed copy, enqueued
    usm_vector(const usm_vector& other, allocator_type alloc);           // routed copy into another placement or kind
    usm_vector& operator=(const usm_vector& other);                      // routed copy, enqueued; reallocates if needed
    usm_vector(usm_vector&&) noexcept;
    usm_vector& operator=(usm_vector&&) noexcept;                        // steals storage between equal allocators only

    sycl::event resize(size_type n);
    sycl::event resize(size_type n, const T& value);
    sycl::event fill(const T& value);
    sycl::event from_host(std::span<const T> src);
    void reserve(size_type n);  void shrink_to_fit();  void clear();
    size_type size() const;  size_type capacity() const;  bool empty() const;
    sycl::event last_event() const;                                      // the event of the most recent mutation

    T*       data();       const T* data() const;                        // valid until the next reallocation
    host_access<T> get_access();                                         // syncs; mutable view; shared/host kinds only
    host_view<T>   host_view();                                          // syncs; read-only view; all kinds

    allocator_type get_allocator() const;  queue get_queue() const;  kind get_kind() const;
};

template <class T> class host_access { public: T& operator[](size_type) ; T* data(); size_type size(); /* iterators */ };
template <class T> class host_view   { public: const T& operator[](size_type) const; const T* data() const; size_type size() const; /* iterators */ };

} // namespace naga::slabkit
```

Rules:

- **Every copy is a routed `copy` on the container's queue** — between
  devices, backends or kinds — ordered as one item; the caller never sees
  the route except in time.
- **Old storage is freed synchronously** once the container's own last task
  (the copy that read it) has completed. Kernels the caller launched over
  `data()` are the caller's to wait for before `resize`, `reserve`,
  `shrink_to_fit`, reassignment or destruction; `data()`'s value changes at
  reallocation.
- **`get_access()`** calls `q.wait()` and returns a mutable view; on a
  device-kind container it throws `std::invalid_argument`. **`host_view()`**
  calls `q.wait()` and returns a read-only view: borrowed for shared and
  host kinds, an owned D2H copy for the device kind. Take-turns while
  holding either view is the caller's responsibility.
- **Container-to-container operations require the same queue**;
  otherwise `std::invalid_argument`. The cross-stream tool is §6's `copy` on
  `data()` with explicit events. A move between unequal allocators is
  refused the same way.
- Not a `std::vector`: no `operator[]` on the container, no iterators, no
  `push_back`/`insert`/`erase`, no `at()`.
