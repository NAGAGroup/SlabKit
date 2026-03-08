# SlabKit Design Specification

**Status**: Reformulated post-first-working-design. This spec supersedes ad-hoc decisions made
during initial implementation and is the reference for all subsequent work.

---

## What SlabKit Is

A header-only C++20/SYCL library providing RAII memory management and typed array abstractions
over device USM (Unified Shared Memory). Design goals:

- GPU memory should feel as safe and ergonomic as CPU memory
- Ownership is explicit and unambiguous at every level
- No hidden allocations anywhere in the public API
- The registry is a convenience, not a requirement — users may manage everything themselves
- Composable async transfers with no surprise overheads

---

## Header Structure

```
slabkit/
  memory.hpp      ← primary entry point: make_device_array, queue factories, raw alloc
  array.hpp       ← DeviceView, DeviceSubView, DeviceArray, MirroredArray, transfers
  scratch.hpp     ← DeviceScratchBuffer, DeviceScratchView (needed for Tier 2 transfers)
  registry.hpp    ← AllocatorRegistry (advanced use: custom config, introspection)
  detail/         ← DeviceMemoryPool, bitmap internals (do not include directly)
```

**Normal users include only `memory.hpp` and `array.hpp`**.  
Include `scratch.hpp` when doing strided transfers.  
Include `registry.hpp` only for custom configuration, multi-device introspection, or testing.  
`detail/` is never user-facing.

---

## Central Lifetime Object: `SlabContext`

The key architectural insight: every internal `shared_ptr` in the old design was a lifetime
dependency on a context or pool. Those dependencies are unified into a single object.

```cpp
struct SlabContext {
    std::vector<sycl::context>     platform_contexts;   // owned by value, one per platform
    std::vector<sycl::device>      devices;             // owned by value
    std::vector<size_t>            device_to_platform;  // devices[i] → platform_contexts[j]
    std::vector<size_t>            device_slab_sizes;   // computed at construction
    std::vector<DeviceMemoryPool>  pools;               // owned by value, pools[i] ↔ devices[i]

    // Lookup
    size_t                  device_index(const sycl::device&) const;
    DeviceMemoryPool&       pool_at(size_t device_index);
    const DeviceMemoryPool& pool_at(size_t device_index) const;
    DeviceMemoryPool&       pool_for(const sycl::device&);
    const sycl::context&    context_for(size_t device_index) const;
    const sycl::context&    context_for(const sycl::device&) const;

    // Queue factories (canonical form — registry delegates here)
    sycl::queue make_in_order_queue(const sycl::device&) const;
    sycl::queue make_out_of_order_queue(const sycl::device&) const;
    sycl::queue make_queue_with_props(const sycl::device&, const sycl::property_list&) const;
};
```

**Key properties:**
- All objects are stored by value — no internal `shared_ptr` chains
- `DeviceMemoryPool` objects are co-located with the context that owns their slab; the pool
  dtor calls `sycl::free(slab_base_, platform_contexts[device_to_platform[i]])` via a raw
  reference (pointer or index), guaranteed valid because they share lifetime with `SlabContext`
- The entire structure is wrapped in exactly **one** `shared_ptr<SlabContext>`, which is the
  single currency for lifetime extension throughout the library

### `SlabContextBuilder` — construction for both registry and user-managed workflows

`SlabContextBuilder` is the single construction path for `SlabContext`, used internally by
`AllocatorRegistry` and directly by users who want full control.

**Input**: a vector of `sycl::context` objects (one per platform) and a vector of
`sycl::device` objects. The builder validates and populates the `SlabContext` in two passes:

**Pass 1 — Populate `platform_contexts`** (O(N²) over contexts, N is typically 1–3):
- For each provided context, extract its platform via `ctx.get_platform()`
- Check for duplicate platforms — two contexts from the same platform is an error
  (`throw std::invalid_argument`)
- Populate `SlabContext::platform_contexts` in the order provided

**Pass 2 — Populate `devices` and `device_to_platform`** (O(D×P), D devices, P platforms):
- For each provided device, find its platform in `platform_contexts`
- If no context covers the device's platform: `throw std::invalid_argument`
- Populate `SlabContext::devices` and `SlabContext::device_to_platform`

Pools are **not** created at build time — they are created on first access (lazy, per-device).
The builder produces a fully validated, immutable-topology `SlabContext`; pool state is the
only mutable part after construction.

```cpp
// Registry-free: user provides their own contexts and devices
auto ctx = std::make_shared<SlabContext>(
    SlabContextBuilder{}
        .contexts({sycl_ctx_a, sycl_ctx_b})   // one per platform
        .devices({dev1, dev2, dev3})           // any devices whose platforms are covered
        .node_size(1024)                       // pool node granularity (bytes)
        .slab_size_per_device(128 * 1024 * 1024)  // or provide per-device sizes
        .build()
);

// Registry path (same builder, called internally by AllocatorRegistry::get())
// Registry auto-discovers contexts and devices via SYCL device enumeration
```

Users who want full control pass `shared_ptr<SlabContext>` directly to all factory
functions. The registry never needs to be touched.

---

## Ownership Model

### Three levels, each explicit

```
DeviceArray<T,N>                — move-only, unique ownership (unique_ptr semantics)
DeviceView<T,N>                 — non-owning borrow, trivially copyable
DeviceSubView<T,N>              — non-owning strided borrow, trivially copyable
```

Shared ownership is **opt-in only**:
```cpp
std::shared_ptr<DeviceArray<T,N>> shared = std::make_shared<DeviceArray<T,N>>(std::move(arr));
```

There is no `.share()` method — if you need shared ownership of an array, wrap it in a
`shared_ptr` explicitly. This makes shared ownership visible at every callsite.

### Deleters

One deleter variant. `from_usm` factories are deferred to a future design iteration.

```cpp
// Pool-backed allocation — the only allocation path
struct DeviceUSMPoolDeleter {
    std::weak_ptr<SlabContext> ctx;
    size_t device_index;
    size_t node_count;     // pre-computed at allocation time, never recomputed

    void operator()(void* ptr) const noexcept {
        if (auto c = ctx.lock())
            c->pool_at(device_index).deallocate_array(ptr, node_count);
        // expired: SlabContext destroyed, slab already freed via pool dtor — no-op is correct
    }
};
```

`node_count` is computed **once** using the ceiling formula `(count * sizeof(T) + node_size - 1) / node_size`
at allocation time and stored in the deleter. The formula never appears in `operator()`.

### `DeviceArray<T,N>`

```cpp
template <typename T, int N>
class DeviceArray {
    // Internal: unique_ptr<T, DeviceUSMPoolDeleter>
    // T must not be const — use DeviceView<const T,N> for read-only access

    // Factory — primary construction path
    static DeviceArray from_pool(sycl::range<N> range,
                                  std::shared_ptr<SlabContext> ctx,
                                  size_t device_index);

    // Accessors
    T*             data() noexcept;
    const T*       data() const noexcept;
    sycl::range<N> get_range() const noexcept;
    size_t         size() const noexcept;        // stored, not recomputed from range
    size_t         byte_size() const noexcept;
    size_t         device_index() const noexcept;

    // View production
    DeviceView<T, N>       view() noexcept;
    DeviceView<const T, N> view() const noexcept;      // const handle → const view
    DeviceView<const T, N> const_view() const noexcept;

    DeviceSubView<T, N>       subview(std::array<size_t,N> offsets, sycl::range<N> sub_range);
    DeviceSubView<const T, N> subview(std::array<size_t,N> offsets, sycl::range<N> sub_range) const;
    DeviceSubView<const T, N> const_subview(std::array<size_t,N> offsets, sycl::range<N> sub_range) const;

    // Implicit conversion to view — enables passing DeviceArray where DeviceView expected
    // Both provided to avoid double-conversion failure in overload resolution
    operator DeviceView<T, N>() noexcept;
    operator DeviceView<const T, N>() const noexcept;
};
```

`DeviceArray<const T, N>` is **not provided**. Constness of device data is expressed via
`DeviceView<const T,N>`. `const DeviceArray<T,N>` (const handle, mutable data) is valid and
produces const views via the `const` overloads above.

### `MirroredArray<T,N>`

Move-only. Explicit `clone()` for deep copy of both sides.

```cpp
template <typename T, int N>
class MirroredArray {
    DeviceArray<T, N>  device_;
    std::vector<T>     host_;

    // No copy constructor, no copy assignment — deleted
    // Move constructor, move assignment — defaulted

    MirroredArray clone() const;  // deep copies both device and host sides

    // Device access — views only, no DeviceArray exposure
    DeviceView<T, N>          device_view() noexcept;
    DeviceView<const T, N>    device_view() const noexcept;
    DeviceSubView<T, N>       device_subview(std::array<size_t,N>, sycl::range<N>);
    DeviceSubView<const T, N> device_subview(...) const;

    std::span<T>       host_data() noexcept;
    std::span<const T> host_data() const noexcept;

    sycl::event sync_to_device(const sycl::queue&, std::span<const sycl::event> deps = {});
    sycl::event sync_to_host(const sycl::queue&, std::span<const sycl::event> deps = {});
};
```

`device_array()` accessor is **not provided**. Sync semantics are owned by `MirroredArray`;
callers access device data only through views.

---

## Array View Types

### `DeviceView<T,N>` — contiguous, row-major

Trivially copyable. Non-owning. Kernel-safe to capture by value **provided the backing
`DeviceArray` outlives kernel execution** (see Kernel Safety section).

```cpp
template <typename T, int N>
struct DeviceView {
    T*             ptr_;
    sycl::range<N> range_;
    size_t         element_count_;  // stored for O(1) size()

    // Element access — integral types only, N arguments
    template <typename... Ts> requires (std::integral<Ts> && ...) && (sizeof...(Ts) == N)
    T& operator()(Ts... idx) const noexcept;

    T& operator[](sycl::id<N> id) const noexcept;
    T& operator[](size_t idx) const noexcept requires (N == 1);

    T*             data() const noexcept;
    sycl::range<N> get_range() const noexcept;
    size_t         size() const noexcept;
    size_t         byte_size() const noexcept;

    // Implicit const conversion
    operator DeviceView<const T, N>() const noexcept requires (!std::is_const_v<T>);
};
```

### `DeviceSubView<T,N>` — strided

Trivially copyable. Strides are in **elements**, not bytes.

```cpp
template <typename T, int N>
struct DeviceSubView {
    T*             ptr_;
    sycl::range<N> range_;
    sycl::range<N> strides_;       // element strides: strides_[d] = ∏(range_[d+1:N]) for row-major
    size_t         element_count_; // stored for O(1) size()

    // Same element access interface as DeviceView
    template <typename... Ts> requires (std::integral<Ts> && ...) && (sizeof...(Ts) == N)
    T& operator()(Ts... idx) const noexcept;

    T& operator[](sycl::id<N> id) const noexcept;
    T& operator[](size_t idx) const noexcept requires (N == 1);

    sycl::range<N> get_strides() const noexcept { return strides_; }  // element strides

    // Implicit const conversion
    operator DeviceSubView<const T, N>() const noexcept requires (!std::is_const_v<T>);
};
```

### `DeviceViewLike<V,T,N>` concept

Does **not** require `data()` — only element access semantics.

```cpp
template <typename V, typename T, int N>
concept DeviceViewLike = requires(V v, sycl::id<N> id) {
    { v.get_range() } -> std::same_as<sycl::range<N>>;
    { v.size() }      -> std::convertible_to<size_t>;
    { v.byte_size() } -> std::convertible_to<size_t>;
    { v[id] }         -> std::same_as<T&>;
};
```

### Bounds checking

- **Construction** (`subview()`, `from_usm()` with explicit range): throws `std::out_of_range`.
  This is host-side code; exceptions are legal.
- **Access** (`operator()`, `operator[]`): **unchecked by design** — these execute in device
  kernels where exceptions are not permitted by the SYCL spec. Same contract as `std::span`.
  Compile with `SLABKIT_DEVICE_BOUNDS_CHECK` to enable device-trap on supporting backends.

---

## Primary API (`memory.hpp` free functions)

Everything a normal user needs. `SlabContext` and `AllocatorRegistry` are never visible.

```cpp
// Arrays — delegates to default registry SlabContext
template <typename T, int N>
DeviceArray<T,N> make_device_array(sycl::range<N> range, const sycl::device& dev);

// Arrays — explicit SlabContext (registry-free)
template <typename T, int N>
DeviceArray<T,N> make_device_array(sycl::range<N> range,
                                    std::shared_ptr<SlabContext> ctx,
                                    size_t device_index);

// Raw device memory
template <typename T>
std::unique_ptr<T, DeviceUSMPoolDeleter>    malloc_device(size_t count, const sycl::device&);
template <typename T>
std::unique_ptr<T, DeviceUSMPoolDeleter>    malloc_device(size_t count,
                                                           std::shared_ptr<SlabContext>,
                                                           size_t device_index);

// Scratch
DeviceScratchBuffer make_scratch(const sycl::device&, size_t size = 64*1024);
DeviceScratchBuffer make_scratch(std::shared_ptr<SlabContext>, size_t device_index,
                                  size_t size = 64*1024);

// Queue factories — explicit naming, no ambiguity about in/out-of-order
sycl::queue make_in_order_queue(const sycl::device&);
sycl::queue make_out_of_order_queue(const sycl::device&);
sycl::queue make_queue_with_props(const sycl::device&, const sycl::property_list&);

// Registry-free queue factories — user passes their own SlabContext
sycl::queue make_in_order_queue(const sycl::device&, const SlabContext&);
sycl::queue make_out_of_order_queue(const sycl::device&, const SlabContext&);
sycl::queue make_queue_with_props(const sycl::device&, const sycl::property_list&,
                                   const SlabContext&);
```

**`make_device_array` is the standard entry point.** `malloc_device` is for when you need a
raw typed pointer without the array abstraction (rare).

---

## Scratch Memory (`scratch.hpp`)

Two types, explicit semantics:

```cpp
// Pool-owning: holds shared_ptr<SlabContext> + device_index + node_count
// dtor returns allocation to pool
class DeviceScratchBuffer {
    // Move-only
    void*  allocate(size_t size, size_t alignment = alignof(std::max_align_t));
    bool   can_allocate(size_t size, size_t alignment = ...) const noexcept;
    std::optional<void*> try_allocate(size_t size, size_t alignment = ...) noexcept;

    using Marker = size_t;
    Marker top() const noexcept;
    void   unwind(Marker) noexcept;
    void   reset() noexcept;

    size_t capacity() const noexcept;
    size_t used() const noexcept;
    size_t remaining() const noexcept;
};

// Non-owning: thin bump allocator view over caller-managed slab
// Trivially copyable, no dtor side effects
class DeviceScratchView {
    // Same interface as DeviceScratchBuffer minus ownership
    // Construction: DeviceScratchView(void* slab_base, size_t capacity)
};
```

---

## Transfer System (`array.hpp`)

Four tiers resolved at compile time by argument types. A decision flowchart is provided in
the `array.hpp` header comment.

All async functions: `const sycl::queue& q`, `std::span<const sycl::event> deps = {}`.

**Tier 2 returns `(sycl::event, DeviceScratchBuffer::Marker)`** — caller unwinds the marker
after `event.wait()`. This makes staging lifetime explicit and prevents accumulation.

```
Tier 1 — Contiguous memcpy (no kernel, no scratch)
  Source → Dest:
    DeviceView<const T,N> → T*              copy_to_host
    T*                    → DeviceView<T,N> copy_to_device
    DeviceView<const T,N> → DeviceView<T,N> copy  (same device only — asserted)

Tier 2 — Strided host transfer (gather/scatter kernel + staging buffer)
  DeviceSubView<const T,N> → T*                copy_to_host  → (event, Marker)
  T*                       → DeviceSubView<T,N> copy_to_device → (event, Marker)
  Takes DeviceScratchBuffer& or DeviceScratchView&

Tier 3 — Strided device-to-device (parallel_for, no staging, no host)
  DeviceSubView<const T,N> → DeviceSubView<T,N>  copy
  DeviceSubView<const T,N> → DeviceView<T,N>     copy
  DeviceView<const T,N>    → DeviceSubView<T,N>  copy

Tier 4 — Blocking sync wrappers (call Tier 1-3 + wait, return void)
  sync_to_host(src, dst, queue)
  sync_to_device(src, dst, queue)
```

**Public APIs take views, not arrays.** `DeviceArray` implicitly converts to `DeviceView`,
so passing an array works without overloads. Both mutable and const implicit conversions are
provided on `DeviceArray` to avoid double-conversion failures.

**`copy` (device-to-device) asserts same `device_index`** on both source and destination.
Cross-device copies are not supported silently.

---

## Error Handling

| Situation | Mechanism |
|---|---|
| Pool / scratch exhausted | `throw AllocationError` (custom, not `bad_alloc`) |
| Bounds violation in `subview()` | `throw std::out_of_range` (host-side only) |
| Device access out of bounds | Unchecked (device code); `SLABKIT_DEVICE_BOUNDS_CHECK` for trap |
| Double-free / slot corruption | `throw std::logic_error` |
| Cross-device copy attempted | `assert` in debug, `throw std::invalid_argument` in release |
| `configure()` called after init | `throw std::logic_error` |
| `SlabContext` destroyed before allocation freed | Deleter no-ops silently (correct — pool dtor already freed slab) |

```cpp
class AllocationError : public std::exception {
public:
    enum class Reason { PoolExhausted, Fragmented, ScratchFull };
    Reason reason() const noexcept;
    const char* what() const noexcept override;
};
```

---

## `AllocatorRegistry` (`registry.hpp`)

Thin wrapper around a lazily-initialized `shared_ptr<SlabContext>`. Not primary API — users
who want custom configuration or full control use `SlabContextBuilder` directly.

```cpp
class AllocatorRegistry {
public:
    struct Config {
        size_t block_size           = 64ull * 1024;
        size_t node_size            = 1024;
        // 60% of device global memory reserved for pool slab.
        // Chosen to leave headroom for SYCL runtime overhead, driver allocations,
        // and implicit buffers (e.g., desktop compositor on shared GPU).
        // CPU devices use a lower fraction — host memory is less scarce.
        float  device_slab_fraction = 0.60f;
        float  cpu_slab_fraction    = 0.30f;
    };

    // One-shot configuration — must be called before any make_device_array / make_*_queue.
    // Throws std::logic_error if called after the SlabContext has been initialized.
    static void configure(const Config&);

    // Access the shared SlabContext (lazily initializes on first call).
    // Internally uses SlabContextBuilder with auto-discovered SYCL devices and contexts.
    static std::shared_ptr<SlabContext> get();

    // Convenience accessors (delegate to SlabContext)
    static const sycl::context& get_context(const sycl::device&);
    static DeviceMemoryPool&    get_pool(const sycl::device&);

    // Test support — tears down singleton state for a clean slate.
    // Only available when SLABKIT_TESTING is defined.
    static void reset_for_testing();
};
```

`config()` returning a mutable ref is removed. `configure()` takes a value, is one-shot,
and throws if the registry is already live.

---

## Kernel Safety

Views are non-owning. The backing `DeviceArray` **must** outlive kernel execution.

```cpp
// SAFE — array outlives kernel by scope
auto arr = make_device_array<float,2>(range, dev);
q.parallel_for(range, [v = arr.view()](sycl::id<2> id) {
    v[id] = 1.0f;
}).wait();
// arr still alive here ✓

// UNSAFE — do not do this
DeviceView<float,2> view = get_view_from_somewhere();
arr = DeviceArray<float,2>{};         // backing freed
q.parallel_for(range, [v = view]...); // dangling pointer ✗
```

For cases where scope-based safety is insufficient (e.g., views stored as class members,
futures, callbacks), wrap the `DeviceArray` in a `shared_ptr` and capture that alongside
the view:

```cpp
auto shared_arr = std::make_shared<DeviceArray<float,2>>(std::move(arr));
auto view = shared_arr->view();
q.parallel_for(range, [shared_arr, v = view](sycl::id<2> id) {
    v[id] = 1.0f;
});
// shared_arr captured in lambda — backing kept alive for kernel duration ✓
```

---

## Testability

Tests should **never** touch `AllocatorRegistry` directly. The primary test pattern:

```cpp
// In test setup — construct SlabContext explicitly, no singleton involved
sycl::device dev{sycl::cpu_selector_v};
sycl::context sycl_ctx{dev};

auto ctx = std::make_shared<SlabContext>(
    SlabContextBuilder{}
        .contexts({sycl_ctx})
        .devices({dev})
        .node_size(1024)
        .slab_size_per_device(4 * 1024 * 1024)
        .build()
);

// All factories use explicit ctx — registry never touched
auto arr = make_device_array<float,1>(sycl::range<1>{64}, ctx, 0);
auto q   = make_in_order_queue(dev, *ctx);
```

`AllocatorRegistry::reset_for_testing()` is available for integration tests that must
exercise the registry path, but unit tests should not need it.

---

## Design Invariants

1. **One `shared_ptr<SlabContext>` per logical device group.** Never create two `SlabContext`
   objects for the same physical device — USM from one context cannot be used from a queue
   bound to another context.

2. **`node_count` is computed once, at allocation.** It is stored in the deleter. The
   ceiling formula never appears in a destructor.

3. **Views are non-owning by contract.** They carry no reference counts, no pool references,
   nothing. The performance contract of SYCL kernel capture is maintained.

4. **Transfer APIs take views.** `DeviceArray` implicitly converts; no array-specific
   overloads exist in the transfer layer.

5. **Strides are always in elements.** Byte strides are never used internally or exposed.
   This is documented at every point where strides appear.

6. **Queue creation always binds to a `SlabContext` context.** This guarantees USM
   compatibility. Users who create their own `sycl::queue` directly must use
   `AllocatorRegistry::get_context(dev)` or their own `SlabContext::context_for(dev)`.
