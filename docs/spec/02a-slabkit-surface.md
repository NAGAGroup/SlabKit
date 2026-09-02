# Spec 02.A — slabkit surface

The slabkit surface, as CospanMDAO's memory substrate consumes it.
Binding contract; declarations only.

## 1. What slabkit is

USM device memory with owning handles. Two families of free functions —
**storage** (`malloc_*`: uninitialized bytes, SYCL's USM API mirrored
signature-for-signature, returning handles instead of raw pointers) and
**objects** (`make_*`: `std::make_unique`/`std::make_shared` in every
form, constructed on the device) — over four kinds of **placement** (a
queue, a device with its context, an arena, a pool), yielding one handle
type. Why: cospan ledger §2.1 user types on the device.

There are no array or view types: shapes belong to consumers.
`DeviceArena` and `DevicePool` are memory, not allocators: they expose no
allocation methods; the free functions are the only allocation API.

slabkit is a compiled shared library. Everything that does not depend on
the element type is compiled once — the memory objects, the deleters, and
a byte-level core for allocation and initialization — and the typed
functions are thin wrappers over it. Only device-side construction is
necessarily a template, because only it generates per-type device code.

**Contracts are borrowed, not invented.** `malloc_*` behaves exactly as
`sycl::malloc_*`: null exactly when the underlying `sycl::malloc_*`
returns null, and null when an arena or pool cannot satisfy the request.
A zero-element request is implementation-defined and slabkit adds no rule
of its own — AdaptiveCpp's host and CUDA backends do not answer it alike.
An arena or pool, having no SYCL call to mirror, returns null for it.
`make_*` behaves exactly as `std::make_*`: returns with the object ready,
throws `std::bad_alloc` when storage cannot be obtained. Usage errors — a
pool node size that is not a power of two, an alignment a pool cannot
honor — are `std::invalid_argument`. There is no library exception type.

**Contexts are held, never managed.** One `sycl::context` per device,
created once by whoever owns the device and kept for as long as the
device is used. slabkit stores the context it was given and frees with
it — correct on every conforming SYCL, not only where the runtime happens
to ignore it. One context per device is a condition, not a preference:
`sycl::free(p, ctx)` resolves its allocator from the context's first
device backend and the first device within it, so a context spanning
several devices can release a pointer through the wrong allocator.
Managing several contexts per device (emulating multiple devices on one,
isolating pointers) is a niche the library does not serve.

**One place frees memory.** Every slab an arena or pool owns is itself an
allocation from this library's slabless API, so `sycl::free` is called
from exactly one deleter and no class in the library has a destructor of
its own. Why: cospan ledger P9.

```cpp
namespace naga::slabkit {

// -----------------------------------------------------------------------------
// 2. Handles — three deleters, one handle type
// -----------------------------------------------------------------------------
struct OsFree    { sycl::context ctx; };                                     // sycl::free(p, ctx)
struct PoolFree  { std::shared_ptr<detail::PoolImpl>  impl; size_t nodes; }; // impl->release(p, nodes)
struct ArenaFree { std::shared_ptr<detail::ArenaImpl> impl; };               // nothing: the arena owns the bytes

/// The deleter holds exactly what release needs. OsFree is the only one
/// that frees memory; the other two account for it. For the pool and
/// arena routes the deleter holds a share of the memory object's
/// implementation: an outstanding allocation keeps its slab alive by
/// construction, and the slab is freed by the last of {handles to the
/// object, allocations from it}. There is no lifetime contract to
/// remember. Why: cospan ledger P9. The deleter never runs a destructor
/// (§5.2).
struct DeviceDeleter {
    std::variant<OsFree, PoolFree, ArenaFree> how;
    void operator()(void* p) const noexcept;
};
template <class T> using unique_device_ptr = std::unique_ptr<T, DeviceDeleter>;
template <class T> using shared_device_ptr = std::shared_ptr<T>;   // deleter erased

// -----------------------------------------------------------------------------
// 3. Placement — where memory comes from
// -----------------------------------------------------------------------------
/// Four kinds, each an overload set below:
///   sycl::queue                    -> the queue's device and context
///   sycl::device + sycl::context   -> that device, freed with that context
///   DeviceArena                    -> bump region
///   DevicePool                     -> fixed-node free list
///
/// WHICH FORM. Allocating for the device a queue was constructed for: the
/// queue forms — same device, same context, nothing to get wrong.
/// Allocating for any other device: the device+context forms with THAT
/// device's one context. Choosing the device+context forms for everything
/// is legitimate and keeps allocation uniform; the single obligation is
/// that the context passed for a queue's own device is the context the
/// queue owns — construct the context first and the queue from it.

// -----------------------------------------------------------------------------
// 4. STORAGE — the malloc family
// -----------------------------------------------------------------------------
/// SYCL's signatures with T* replaced by a handle for `count` elements —
/// an array handle, so it is the type an entry's storage already has;
/// <std::byte> is the void* form. Alignment: queue and device forms as
/// SYCL, which is to say the runtime may ignore the request entirely; an
/// arena honors any alignment, charging the padding to itself; a pool's
/// allocations are node_size-aligned and a larger request is
/// std::invalid_argument.
template <class T> unique_device_ptr<T[]> malloc_device(size_t count, const sycl::queue& q,
                                                      const sycl::property_list& props = {});
template <class T> unique_device_ptr<T[]> malloc_device(size_t count, const sycl::device& dev,
                                                      const sycl::context& ctx,
                                                      const sycl::property_list& props = {});
template <class T> unique_device_ptr<T[]> malloc_device(size_t count, DeviceArena& arena);
template <class T> unique_device_ptr<T[]> malloc_device(size_t count, DevicePool& pool);

template <class T> unique_device_ptr<T[]> aligned_alloc_device(size_t align, size_t count, const sycl::queue& q,
                                                             const sycl::property_list& props = {});
template <class T> unique_device_ptr<T[]> aligned_alloc_device(size_t align, size_t count, const sycl::device& dev,
                                                             const sycl::context& ctx,
                                                             const sycl::property_list& props = {});
template <class T> unique_device_ptr<T[]> aligned_alloc_device(size_t align, size_t count, DeviceArena& arena);
template <class T> unique_device_ptr<T[]> aligned_alloc_device(size_t align, size_t count, DevicePool& pool);

// -----------------------------------------------------------------------------
// 5. OBJECTS — the make family
// -----------------------------------------------------------------------------
/// The initialization queue comes first, the placement second; placement
/// omitted means the queue's own device and context. Returns when the
/// object is READY — initialization is enqueued only because that is how
/// device memory is written, and it is waited on before return. When the
/// placement's device is not the queue's, initialization runs on a queue
/// built from THAT device's own context. It cannot run on the given queue:
/// a runtime that classifies a copy's operands against the queue's context
/// will read another context's device pointer as host memory.
///
/// 5.1 MECHANISM, chosen by type at compile time:
///   * trivially copyable T: constructed on the host and copied in — a
///     fill is a replicated pattern, a copy is a memcpy. This path is not
///     per-type: it is the compiled core, and generates no device code for T.
///   * otherwise: storage, then ONE single_task performing placement-new —
///     the constructor runs where the object lives, so every invariant and
///     every interior pointer is device-correct. This is the door for types
///     that are not device-copyable: an object holding a pointer into
///     itself, host-constructed and copied in, would carry a host address
///     into device memory, and this path makes that mistake unwritable.
///     Arguments are copied into the kernel and reach the constructor as
///     const lvalues; a kernel's operator() is const, so nothing can be
///     moved out of the capture and move-only arguments are not expressible.
/// 5.2 THE TYPE GATE (concept-checked, so a rejection is observable):
///   * T is trivially destructible — the handle has no queue to run ~T on;
///   * T is NOT polymorphic — constructing on the device is what would make
///     a vtable device-correct, but AdaptiveCpp's generic/SSCP flow cannot
///     resolve device vtables at all: the call faults on the host backend
///     and can return silently wrong results on CUDA. Compile-time
///     rejection is the only safe answer available;
///   * every argument is device-copyable (arguments are kernel captures);
///   * array forms require trivially copyable elements (they go through
///     fill/copy).
///   The constructor's device-callability is enforced by the compiler at
///   kernel instantiation.
/// 5.3 The forms. `P` stands for each placement overload:
///   (sycl::queue& q) · (q, const sycl::device&, const sycl::context&) ·
///   (q, DeviceArena&) · (q, DevicePool&). `E` = std::remove_extent_t<TArr>.
template <class T,    class... Args> unique_device_ptr<T>    make_unique_device(P, Args&&... args);
template <class T>                   unique_device_ptr<T>    make_unique_device_for_overwrite(P);   // storage only
template <class TArr>                unique_device_ptr<TArr> make_unique_device(P, size_t n);       // value-initialized
template <class TArr>                unique_device_ptr<TArr> make_unique_device_for_overwrite(P, size_t n);
template <class TArr>                unique_device_ptr<TArr> make_unique_device(P, size_t n, const E& value);  // fill
template <class TArr>                unique_device_ptr<TArr> make_unique_device(P, std::span<const E> src);   // copy
// make_shared_device: the identical six, returning shared_device_ptr.

// -----------------------------------------------------------------------------
// 6. Memory objects — placement only, no allocation API
// -----------------------------------------------------------------------------
/// Each memory object owns its slab as a slabless allocation from this
/// library (unique_device_ptr<std::byte> obtained through the
/// device+context form), so it stores its context and frees exactly like
/// any other allocation, and has no destructor of its own.

/// Bump region over one USM slab. A light handle over a shared
/// implementation: copyable and movable, copies refer to the same slab,
/// allocations pin it. No per-allocation free and no rewind: the slab is
/// released when the last handle and the last allocation are gone. Bump
/// is lock-free. An alignment is honored by aligning the address the bump
/// lands on, so the padding skipped to reach it is charged to the arena
/// and appears in used(); size an arena with slack if it will serve large
/// alignments.
class DeviceArena {
public:
    DeviceArena(const sycl::queue& q, size_t bytes, const sycl::property_list& props = {});
    DeviceArena(const sycl::device& dev, const sycl::context& ctx, size_t bytes,
                const sycl::property_list& props = {});
    size_t capacity() const;  size_t used() const;  size_t remaining() const;
    const sycl::device&  device()  const;
    const sycl::context& context() const;
};

/// Fixed-node free list over one USM slab in node_size slots (a power of
/// two, else std::invalid_argument): two-level bitmap, O(1) for one node,
/// O(N/64) for a contiguous run; mutex-protected; double release
/// asserted. Same handle model as DeviceArena. outstanding() is
/// information.
class DevicePool {
public:
    DevicePool(const sycl::queue& q, size_t bytes, size_t node_size = 1024,
               const sycl::property_list& props = {});
    DevicePool(const sycl::device& dev, const sycl::context& ctx, size_t bytes,
               size_t node_size = 1024, const sycl::property_list& props = {});
    size_t node_size() const;  size_t slot_count() const;
    size_t free_count() const; size_t outstanding() const;
    const sycl::device&  device()  const;
    const sycl::context& context() const;
};

} // namespace naga::slabkit
```

## 7. What cospan builds from this

Per device it runs on, `Model` holds a `MemorySubstrate { sycl::context
ctx; DeviceArena arena; std::optional<DevicePool> pool; }`, keyed by
`sycl::device`. Construction order is the invariant: the default device's
context is created first and both queues are constructed from it; every
other device gets its own `context{dev}`; arenas and pools are built from
`(dev, ctx)`. Subsystems allocate through their substrate's `(device,
context)` for every device, the default included, and never learn which
device the queues were constructed for. The dynamic chain is three lines
in cospan: `malloc_device<T>(n, *pool)`, else `malloc_device<T>(n, dev,
ctx)`, else throw. An entry's storage is a `unique_device_ptr<std::byte[]>`
plus capacity, obtained through `make_unique_device_for_overwrite<std::byte[]>`
against the substrate, never inspecting the route.
