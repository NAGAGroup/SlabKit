# Spec 02.A — slabkit surface

Phase-2 chunk A of the CospanMDAO redesign spec (`cospan/docs/spec/PLAN.md`);
this document lives in slabkit because slabkit owns the surface. Binding
contract; declarations only.

## 1. What slabkit is

USM device memory with RAII handles. No global state, no Context, no device
registry: every memory object is ONE device, ONE strategy, owned by the
user, who composes as many as they need. Two families of free functions
over four placement targets, one handle type.

- **STORAGE — the `malloc` family**: uninitialized bytes, SYCL's USM API
  mirrored name-for-name, returning handles instead of raw pointers.
- **OBJECTS — the `make` family**: `std::make_unique`/`std::make_shared`
  in every form, with the object(s) constructed ON THE DEVICE. This is
  the door through which types that are not device-copyable enter
  kernels at all: the object is built where it lives, and the pointer to
  it — which IS device-copyable — is what kernels capture.

`DeviceArray`, `DeviceView`, `DeviceScratchView`, `make_scratch`, `Context`
and `ContextBuilder` are DELETED. `DeviceArena` and `DevicePool` are
memory, not allocators: they expose NO allocation methods; the free
functions are the only allocation API.

```cpp
namespace naga::slabkit {

// -----------------------------------------------------------------------------
// 2. Handles
// -----------------------------------------------------------------------------
/// One deleter for every route. It stores exactly what release needs:
///   os:    sycl::free(ptr, ctx) — ctx held BY VALUE (sycl::context is a
///          reference-counted handle; no lifetime coupling);
///   pool:  pool->release(ptr, node_count);
///   arena: no-op — the arena owns the bytes.
/// The deleter never runs a destructor (see §4.3).
struct DeviceDeleter {
    enum class Route : uint8_t { os, pool, arena };
    Route         route;
    sycl::context ctx;             // os route
    DevicePool*   pool;            // pool route
    size_t        node_count;      // pool route
    void operator()(void*) const noexcept;
};
template <class T> using unique_device_ptr = std::unique_ptr<T, DeviceDeleter>;
template <class T> using shared_device_ptr = std::shared_ptr<T>;   // deleter erased

/// AllocationError::Reason: { OutOfMemory, ArenaFull, PoolExhausted,
/// Fragmented }. No function in this library returns a null handle.

// -----------------------------------------------------------------------------
// 3. Placement targets
// -----------------------------------------------------------------------------
/// Every allocation names WHERE. Four kinds satisfy the concept:
///   sycl::queue     -> OS route in q.get_context() for q.get_device()
///   Placement       -> OS route, explicit device + context
///   DeviceArena     -> arena route
///   DevicePool      -> pool route
struct Placement { sycl::device dev; sycl::context ctx; };
template <class W> concept MemoryTarget =
    std::same_as<W, sycl::queue> || std::same_as<W, Placement> ||
    std::same_as<W, DeviceArena> || std::same_as<W, DevicePool>;

/// MULTI-DEVICE-QUEUE TOUCHPOINT: the queue target allocates in
/// q.get_context() for q.get_device(). For an acpp multi-device queue
/// the extension's own USM semantics apply — NON-NORMATIVE until
/// verified against the extension's documentation.

// -----------------------------------------------------------------------------
// 4. STORAGE: the malloc family — uninitialized, SYCL mirrored
// -----------------------------------------------------------------------------
template <class T, MemoryTarget W>
unique_device_ptr<T> malloc_device(size_t count, W& where);
template <class T, MemoryTarget W>
unique_device_ptr<T> aligned_alloc_device(size_t align, size_t count, W& where);
/// shared/host memory has no arena/pool form (slabs are device USM):
template <class T> unique_device_ptr<T> malloc_shared(size_t count, const sycl::queue&);
template <class T> unique_device_ptr<T> malloc_shared(size_t count, const Placement&);
template <class T> unique_device_ptr<T> malloc_host  (size_t count, const sycl::context&);
/// aligned_alloc_shared / aligned_alloc_host mirror SYCL likewise.

/// Non-throwing probe for consumers composing fallback chains (cospan's
/// dynamic chain lives in cospan, not here) — the ONLY allocation
/// function that can return empty:
template <class T>
std::optional<unique_device_ptr<T>> try_malloc_device(size_t count, DevicePool& pool);

// -----------------------------------------------------------------------------
// 5. OBJECTS: the make family — constructed on the device
// -----------------------------------------------------------------------------
/// The std::make_unique / std::make_shared forms, every one. The queue
/// that performs initialization comes first, the placement target second
/// (defaults to the queue itself). Initialization is ENQUEUED on q: the
/// handle is valid immediately, its contents once q reaches the op —
/// in-order queues give the natural ordering; out-of-order users take
/// the `_ev` twin (§5.4).
///
/// 5.1 INITIALIZATION MECHANISM — chosen by type, at compile time:
///   * trivially copyable T: construct on the host, then q.memcpy (or
///     q.fill for broadcast forms). No device code involved.
///   * otherwise: allocate storage, then ONE single_task performing
///     placement-new with the forwarded arguments — the constructor runs
///     where the object lives, so every invariant, pointer and vtable is
///     device-correct. (Host-constructing a polymorphic object and
///     copying its bytes plants a HOST vtable pointer on the device;
///     this path exists to make that mistake unwritable.)
///     Verified: acpp generic accepts device placement-new of a
///     polymorphic, non-trivially-copyable type and virtual dispatch on
///     the result.
///
/// 5.2 THE TYPE GATE (concept-checked; failure is a compile error):
///   * T's constructor is device-callable and every argument is
///     device-copyable (arguments are captured into the kernel);
///   * T is TRIVIALLY DESTRUCTIBLE. The deleter frees storage and has no
///     queue to run ~T on; supporting a non-trivial destructor would
///     couple the handle to a queue for life. A non-trivial destructor
///     is refused at the make_* call, with the reason in the message.
///   Array forms (value-init, fill, copy) go through q.fill/q.memcpy and
///   therefore require trivially copyable elements.

// 5.3 forms — unique
template <class T, MemoryTarget W = sycl::queue, class... Args>
unique_device_ptr<T> make_unique_device(sycl::queue& q, W& where, Args&&... args);
template <class T, MemoryTarget W = sycl::queue>
unique_device_ptr<T> make_unique_device_for_overwrite(sycl::queue& q, W& where);
                                              // storage only == malloc_device<T>(1)
template <class TArr, MemoryTarget W = sycl::queue>            // TArr = T[]
unique_device_ptr<TArr> make_unique_device(sycl::queue& q, W& where, size_t n);
                                              // value-initialized (T{}; memset 0 when trivial)
template <class TArr, MemoryTarget W = sycl::queue>
unique_device_ptr<TArr> make_unique_device_for_overwrite(sycl::queue& q, W& where, size_t n);
                                              // uninitialized
/// slabkit extensions beyond std (no std spelling exists):
template <class TArr, MemoryTarget W = sycl::queue>            // fill
unique_device_ptr<TArr> make_unique_device(sycl::queue& q, W& where, size_t n,
                                           const std::remove_extent_t<TArr>& value);
template <class TArr, MemoryTarget W = sycl::queue>            // copy from host
unique_device_ptr<TArr> make_unique_device(sycl::queue& q, W& where,
                                           std::span<const std::remove_extent_t<TArr>> src);

// 5.3 forms — shared: the identical set, returning shared_device_ptr
template <class T, MemoryTarget W = sycl::queue, class... Args>
shared_device_ptr<T> make_shared_device(sycl::queue& q, W& where, Args&&... args);
template <class T, MemoryTarget W = sycl::queue>
shared_device_ptr<T> make_shared_device_for_overwrite(sycl::queue& q, W& where);
template <class TArr, MemoryTarget W = sycl::queue>
shared_device_ptr<TArr> make_shared_device(sycl::queue& q, W& where, size_t n);
template <class TArr, MemoryTarget W = sycl::queue>
shared_device_ptr<TArr> make_shared_device_for_overwrite(sycl::queue& q, W& where, size_t n);
template <class TArr, MemoryTarget W = sycl::queue>
shared_device_ptr<TArr> make_shared_device(sycl::queue& q, W& where, size_t n,
                                           const std::remove_extent_t<TArr>& value);
template <class TArr, MemoryTarget W = sycl::queue>
shared_device_ptr<TArr> make_shared_device(sycl::queue& q, W& where,
                                           std::span<const std::remove_extent_t<TArr>> src);

// 5.4 event form — every make_* form has an `_ev` twin:
template <class H> struct Initialized { H handle; sycl::event ready; };

// -----------------------------------------------------------------------------
// 6. The memory objects — no allocation API on either
// -----------------------------------------------------------------------------
/// Bump region over one USM slab (allocated at construction). Allocations
/// live as long as the arena; there is no per-allocation free. Handles
/// from an arena carry the no-op route so callers hold ONE handle type
/// regardless of where memory came from. Markers/unwind retained for
/// consumers that stack.
class DeviceArena {
public:
    using Marker = size_t;
    DeviceArena(const sycl::queue& q, size_t bytes);
    DeviceArena(const Placement&, size_t bytes);
    DeviceArena(const DeviceArena&) = delete;      // movable
    ~DeviceArena();                                // frees the slab
    Marker top() const;  void unwind(Marker);  void reset();
    size_t capacity() const;  size_t used() const;  size_t remaining() const;
    const sycl::device& device() const;  const sycl::context& context() const;
private:
    void* bump(size_t bytes, size_t align);        // throws ArenaFull
    friend struct detail::Allocate;                // the free functions' one door
};

/// Fixed-node free-list over one USM slab in `node_size` slots (two-level
/// bitmap; O(1) node, O(N/64) contiguous array; mutex-protected;
/// double-free asserted). LIFETIME CONTRACT — the user's, by design: a
/// pool outlives every handle it issued. The destructor checks
/// outstanding() == 0 and reports a violation (assert in debug, logged
/// error in release) rather than freeing memory something still holds.
class DevicePool {
public:
    DevicePool(const sycl::queue& q, size_t bytes, size_t node_size = 1024);
    DevicePool(const Placement&, size_t bytes, size_t node_size = 1024);
    DevicePool(const DevicePool&) = delete;        // NON-movable (handles hold its address)
    ~DevicePool();
    size_t node_size() const;  size_t slot_count() const;
    size_t free_count() const; size_t outstanding() const;
    const sycl::device& device() const;  const sycl::context& context() const;
private:
    void* take(size_t node_count);                 // throws PoolExhausted | Fragmented
    void  release(void*, size_t node_count) noexcept;
    friend struct detail::Allocate;  friend struct DeviceDeleter;
};

} // namespace naga::slabkit
```

## 7. What cospan builds from this (forward-link: 02.B, 02.C)

`Model` holds, per device it runs on, one `DeviceArena` sized by the
finalize accumulation pass (durable ∧ static entries) and optionally one
`DevicePool` sized by the single user knob (default: none). The dynamic
chain is three lines in cospan: `pool ? try_malloc_device<T>(n, *pool) :
nullopt` → `malloc_device<T>(n, q)` → throw. `OutputEntry` owns
`unique_device_ptr<std::byte>` + capacity — the "smart pointer to data"
01.B left open — obtained via `make_unique_device_for_overwrite<std::byte[]>`
against the Model's chosen target, and never inspects the route.
Transient launch streams (01.C/01.E) are plain handles scoped to the
forward call.
