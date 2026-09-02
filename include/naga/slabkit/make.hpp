#pragma once

// =============================================================================
// naga/slabkit/make.hpp — OBJECTS: make_unique_device
// =============================================================================
//
// std::make_unique in every form, with the object constructed on the device.
// The initialization queue comes first and the placement second; omitting the
// placement means the queue's own device and context.
//
// These BLOCK. Initialization is enqueued only because that is how device
// memory is written; it is waited on before the handle is returned, so the
// object is ready the instant you have it. There are no event-returning
// variants.
//
// A placement on a device other than the queue's is initialized on a queue
// built from that device's own context -- see detail::init_queue for why that
// is not the retarget extension.
//
// MECHANISM, chosen by type at compile time:
//   * trivially copyable T: constructed on the host and copied in. No device
//     code is generated for it at all; the copy goes through the compiled core.
//   * otherwise: storage, then ONE single_task performing placement-new. The
//     constructor runs where the object lives, so every invariant and interior
//     pointer is device-correct -- host-constructing such an object and
//     copying its bytes would plant host addresses in device memory, and this
//     path makes that mistake unwritable.
//
// THE TYPE GATE:
//   * T is trivially destructible -- the handle has no queue to run ~T on;
//   * T is NOT polymorphic -- see device_object below for why;
//   * arguments are device-copyable, since they are kernel captures;
//   * array forms require trivially copyable elements.
// The constructor's device-callability is enforced by the compiler when the
// kernel is instantiated.

#include <cstddef>
#include <memory>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>

#include <new>

#include <sycl/sycl.hpp>

#include "naga/slabkit/arena.hpp"
#include "naga/slabkit/detail/core.hpp"
#include "naga/slabkit/handles.hpp"
#include "naga/slabkit/malloc.hpp"
#include "naga/slabkit/pool.hpp"

namespace naga::slabkit {
namespace detail {

// -----------------------------------------------------------------------------
// The type gate
// -----------------------------------------------------------------------------

/// An object slabkit will construct on a device.
///
///   * trivially destructible: a device handle has no queue on which to run a
///     destructor, and it never runs one;
///   * not polymorphic: constructing the object on the device is what makes
///     its vtable device-correct, but AdaptiveCpp's generic/SSCP flow cannot
///     resolve device vtables at all -- the call fails outright on the host
///     backend and, worse, can return silently wrong results on CUDA. Rejecting
///     these at compile time is the only safe answer available.
///
/// This is a constraint rather than a static_assert so that the rejection is
/// observable: `requires { make_unique_device<T>(q); }` answers false.
template <class T>
concept device_object = std::is_trivially_destructible_v<T> && !std::is_polymorphic_v<T>;

/// Array forms initialize through fill/copy, so elements must additionally be
/// trivially copyable.
template <class E>
concept device_element = device_object<E> && std::is_trivially_copyable_v<E>;

template <class... Args>
constexpr void gate_args() {
    static_assert((sycl::is_device_copyable_v<std::decay_t<Args>> && ...),
                  "slabkit: constructor arguments are captured by the construction kernel, "
                  "so each must be device-copyable");
}

// -----------------------------------------------------------------------------
// The one genuinely per-T piece of device code
// -----------------------------------------------------------------------------

/// ONE single_task performing placement-new. The arguments are copied into the
/// kernel and reach the constructor as const lvalues: a kernel's operator() is
/// const, so nothing can be moved out of the capture, and move-only arguments
/// are not expressible here (they are not device-copyable either).
template <class T, class... Args>
void construct_on_device(sycl::queue q, T* p, const Args&... args) {
    const std::tuple<Args...> captured{args...};
    q.single_task([=]() {
         std::apply([p](const Args&... a) { ::new (static_cast<void*>(p)) T(a...); }, captured);
     }).wait();
}

// -----------------------------------------------------------------------------
// Initialization, by mechanism
// -----------------------------------------------------------------------------

template <class T, class... Args>
void init_object(sycl::queue q, T* p, Args&&... args) {
    if constexpr (std::is_trivially_copyable_v<T>) {
        T tmp = T(std::forward<Args>(args)...);  // value-initialized when there are no args
        copy_from_host(p, &tmp, sizeof(T), std::move(q));
    } else {
        gate_args<Args...>();
        construct_on_device<T>(std::move(q), p, static_cast<const std::decay_t<Args>&>(args)...);
    }
}

template <class E>
void init_array_value(sycl::queue q, E* p, std::size_t n) {
    if (n == 0) return;
    const E value = E();
    fill_pattern(p, &value, sizeof(E), n, std::move(q));
}

template <class E>
void init_array_fill(sycl::queue q, E* p, std::size_t n, const E& value) {
    if (n == 0) return;
    fill_pattern(p, &value, sizeof(E), n, std::move(q));
}

template <class E>
void init_array_copy(sycl::queue q, E* p, std::span<const E> src) {
    if (src.empty()) return;
    copy_from_host(p, src.data(), src.size() * sizeof(E), std::move(q));
}

// -----------------------------------------------------------------------------
// Storage, per placement (throwing, as std::make_unique does)
// -----------------------------------------------------------------------------

inline unique_device_ptr<std::byte[]> make_storage(std::size_t bytes, std::size_t align,
                                                   const sycl::queue& q) {
    return require(aligned_alloc_bytes(align, bytes, q, {}), bytes);
}
inline unique_device_ptr<std::byte[]> make_storage(std::size_t bytes, std::size_t align,
                                                   const sycl::device&  dev,
                                                   const sycl::context& ctx) {
    return require(aligned_alloc_bytes(align, bytes, dev, ctx, {}), bytes);
}
inline unique_device_ptr<std::byte[]> make_storage(std::size_t bytes, std::size_t align,
                                                   DeviceArena& arena) {
    return require(arena_bytes(bytes, align, arena), bytes);
}
inline unique_device_ptr<std::byte[]> make_storage(std::size_t bytes, std::size_t align,
                                                   DevicePool& pool) {
    return require(pool_bytes(bytes, align, pool), bytes);
}

/// The queue an initialization runs on, per placement.
inline sycl::queue site(const sycl::queue& q) { return q; }
inline sycl::queue site(const sycl::queue& q, const sycl::device& dev,
                        const sycl::context& ctx) {
    return init_queue(q, dev, ctx);
}
inline sycl::queue site(const sycl::queue& q, const DeviceArena& a) {
    return init_queue(q, a.device(), a.context());
}
inline sycl::queue site(const sycl::queue& q, const DevicePool& p) {
    return init_queue(q, p.device(), p.context());
}

template <class T>
using elem_t = std::remove_extent_t<T>;

/// True when the first of a pack is a placement rather than a constructor
/// argument. The queue-only forms exclude it, so that
/// `make_unique_device<T>(q, arena, args...)` selects the arena overload
/// instead of trying to build a T out of the arena.
template <class... Args>
inline constexpr bool starts_with_placement = false;

template <class First, class... Rest>
inline constexpr bool starts_with_placement<First, Rest...> =
    std::is_same_v<std::decay_t<First>, sycl::device> ||
    std::is_same_v<std::decay_t<First>, DeviceArena> ||
    std::is_same_v<std::decay_t<First>, DevicePool>;

}  // namespace detail

// =============================================================================
// Single object
// =============================================================================

template <class T, class... Args>
    requires(!std::is_array_v<T> && detail::device_object<T> &&
             !detail::starts_with_placement<Args...>)
unique_device_ptr<T> make_unique_device(sycl::queue q, Args&&... args) {
    auto h = detail::retype<T>(detail::make_storage(sizeof(T), alignof(T), q));
    detail::init_object<T>(detail::site(q), h.get(), std::forward<Args>(args)...);
    return h;
}

template <class T, class... Args>
    requires(!std::is_array_v<T> && detail::device_object<T>)
unique_device_ptr<T> make_unique_device(sycl::queue q, const sycl::device& dev,
                                        const sycl::context& ctx, Args&&... args) {
    auto h = detail::retype<T>(detail::make_storage(sizeof(T), alignof(T), dev, ctx));
    detail::init_object<T>(detail::site(q, dev, ctx), h.get(), std::forward<Args>(args)...);
    return h;
}

template <class T, class... Args>
    requires(!std::is_array_v<T> && detail::device_object<T>)
unique_device_ptr<T> make_unique_device(sycl::queue q, DeviceArena& arena, Args&&... args) {
    auto h = detail::retype<T>(detail::make_storage(sizeof(T), alignof(T), arena));
    detail::init_object<T>(detail::site(q, arena), h.get(), std::forward<Args>(args)...);
    return h;
}

template <class T, class... Args>
    requires(!std::is_array_v<T> && detail::device_object<T>)
unique_device_ptr<T> make_unique_device(sycl::queue q, DevicePool& pool, Args&&... args) {
    auto h = detail::retype<T>(detail::make_storage(sizeof(T), alignof(T), pool));
    detail::init_object<T>(detail::site(q, pool), h.get(), std::forward<Args>(args)...);
    return h;
}

// =============================================================================
// Single object, storage only
// =============================================================================

template <class T>
    requires(!std::is_array_v<T> && detail::device_object<T>)
unique_device_ptr<T> make_unique_device_for_overwrite(sycl::queue q) {
    return detail::retype<T>(detail::make_storage(sizeof(T), alignof(T), q));
}

template <class T>
    requires(!std::is_array_v<T> && detail::device_object<T>)
unique_device_ptr<T> make_unique_device_for_overwrite(sycl::queue, const sycl::device& dev,
                                                      const sycl::context& ctx) {
    return detail::retype<T>(detail::make_storage(sizeof(T), alignof(T), dev, ctx));
}

template <class T>
    requires(!std::is_array_v<T> && detail::device_object<T>)
unique_device_ptr<T> make_unique_device_for_overwrite(sycl::queue, DeviceArena& arena) {
    return detail::retype<T>(detail::make_storage(sizeof(T), alignof(T), arena));
}

template <class T>
    requires(!std::is_array_v<T> && detail::device_object<T>)
unique_device_ptr<T> make_unique_device_for_overwrite(sycl::queue, DevicePool& pool) {
    return detail::retype<T>(detail::make_storage(sizeof(T), alignof(T), pool));
}

// =============================================================================
// Array, value-initialized
// =============================================================================

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
unique_device_ptr<TArr> make_unique_device(sycl::queue q, std::size_t n) {
    using E = detail::elem_t<TArr>;
    auto h = detail::retype<TArr>(detail::make_storage(n * sizeof(E), alignof(E), q));
    detail::init_array_value<E>(detail::site(q), h.get(), n);
    return h;
}

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
unique_device_ptr<TArr> make_unique_device(sycl::queue q, const sycl::device& dev,
                                           const sycl::context& ctx, std::size_t n) {
    using E = detail::elem_t<TArr>;
    auto h = detail::retype<TArr>(detail::make_storage(n * sizeof(E), alignof(E), dev, ctx));
    detail::init_array_value<E>(detail::site(q, dev, ctx), h.get(), n);
    return h;
}

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
unique_device_ptr<TArr> make_unique_device(sycl::queue q, DeviceArena& arena, std::size_t n) {
    using E = detail::elem_t<TArr>;
    auto h = detail::retype<TArr>(detail::make_storage(n * sizeof(E), alignof(E), arena));
    detail::init_array_value<E>(detail::site(q, arena), h.get(), n);
    return h;
}

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
unique_device_ptr<TArr> make_unique_device(sycl::queue q, DevicePool& pool, std::size_t n) {
    using E = detail::elem_t<TArr>;
    auto h = detail::retype<TArr>(detail::make_storage(n * sizeof(E), alignof(E), pool));
    detail::init_array_value<E>(detail::site(q, pool), h.get(), n);
    return h;
}

// =============================================================================
// Array, storage only
// =============================================================================

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
unique_device_ptr<TArr> make_unique_device_for_overwrite(sycl::queue q, std::size_t n) {
    using E = detail::elem_t<TArr>;
    return detail::retype<TArr>(detail::make_storage(n * sizeof(E), alignof(E), q));
}

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
unique_device_ptr<TArr> make_unique_device_for_overwrite(sycl::queue, const sycl::device& dev,
                                                         const sycl::context& ctx,
                                                         std::size_t          n) {
    using E = detail::elem_t<TArr>;
    return detail::retype<TArr>(detail::make_storage(n * sizeof(E), alignof(E), dev, ctx));
}

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
unique_device_ptr<TArr> make_unique_device_for_overwrite(sycl::queue, DeviceArena& arena,
                                                         std::size_t n) {
    using E = detail::elem_t<TArr>;
    return detail::retype<TArr>(detail::make_storage(n * sizeof(E), alignof(E), arena));
}

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
unique_device_ptr<TArr> make_unique_device_for_overwrite(sycl::queue, DevicePool& pool,
                                                         std::size_t n) {
    using E = detail::elem_t<TArr>;
    return detail::retype<TArr>(detail::make_storage(n * sizeof(E), alignof(E), pool));
}

// =============================================================================
// Array, filled with a value
// =============================================================================

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
unique_device_ptr<TArr> make_unique_device(sycl::queue q, std::size_t n,
                                           const detail::elem_t<TArr>& value) {
    using E = detail::elem_t<TArr>;
    auto h = detail::retype<TArr>(detail::make_storage(n * sizeof(E), alignof(E), q));
    detail::init_array_fill<E>(detail::site(q), h.get(), n, value);
    return h;
}

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
unique_device_ptr<TArr> make_unique_device(sycl::queue q, const sycl::device& dev,
                                           const sycl::context& ctx, std::size_t n,
                                           const detail::elem_t<TArr>& value) {
    using E = detail::elem_t<TArr>;
    auto h = detail::retype<TArr>(detail::make_storage(n * sizeof(E), alignof(E), dev, ctx));
    detail::init_array_fill<E>(detail::site(q, dev, ctx), h.get(), n, value);
    return h;
}

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
unique_device_ptr<TArr> make_unique_device(sycl::queue q, DeviceArena& arena, std::size_t n,
                                           const detail::elem_t<TArr>& value) {
    using E = detail::elem_t<TArr>;
    auto h = detail::retype<TArr>(detail::make_storage(n * sizeof(E), alignof(E), arena));
    detail::init_array_fill<E>(detail::site(q, arena), h.get(), n, value);
    return h;
}

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
unique_device_ptr<TArr> make_unique_device(sycl::queue q, DevicePool& pool, std::size_t n,
                                           const detail::elem_t<TArr>& value) {
    using E = detail::elem_t<TArr>;
    auto h = detail::retype<TArr>(detail::make_storage(n * sizeof(E), alignof(E), pool));
    detail::init_array_fill<E>(detail::site(q, pool), h.get(), n, value);
    return h;
}

// =============================================================================
// Array, copied from host data
// =============================================================================

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
unique_device_ptr<TArr> make_unique_device(sycl::queue                             q,
                                           std::span<const detail::elem_t<TArr>> src) {
    using E = detail::elem_t<TArr>;
    auto h = detail::retype<TArr>(detail::make_storage(src.size() * sizeof(E), alignof(E), q));
    detail::init_array_copy<E>(detail::site(q), h.get(), src);
    return h;
}

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
unique_device_ptr<TArr> make_unique_device(sycl::queue q, const sycl::device& dev,
                                           const sycl::context&                  ctx,
                                           std::span<const detail::elem_t<TArr>> src) {
    using E = detail::elem_t<TArr>;
    auto h =
        detail::retype<TArr>(detail::make_storage(src.size() * sizeof(E), alignof(E), dev, ctx));
    detail::init_array_copy<E>(detail::site(q, dev, ctx), h.get(), src);
    return h;
}

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
unique_device_ptr<TArr> make_unique_device(sycl::queue q, DeviceArena& arena,
                                           std::span<const detail::elem_t<TArr>> src) {
    using E = detail::elem_t<TArr>;
    auto h =
        detail::retype<TArr>(detail::make_storage(src.size() * sizeof(E), alignof(E), arena));
    detail::init_array_copy<E>(detail::site(q, arena), h.get(), src);
    return h;
}

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
unique_device_ptr<TArr> make_unique_device(sycl::queue q, DevicePool& pool,
                                           std::span<const detail::elem_t<TArr>> src) {
    using E = detail::elem_t<TArr>;
    auto h = detail::retype<TArr>(detail::make_storage(src.size() * sizeof(E), alignof(E), pool));
    detail::init_array_copy<E>(detail::site(q, pool), h.get(), src);
    return h;
}

}  // namespace naga::slabkit
