#pragma once

// =============================================================================
// naga/slabkit/make.hpp — OBJECTS: make_unique_{device,shared,host}
// =============================================================================
//
// std::make_unique in every form, over three kinds and six placements. The
// placement comes first; a queue in it is both the placement and the stream the
// initialization runs on.
//
// These BLOCK. Initialization is enqueued only because that is how USM is
// written; it is waited on before the handle is returned, so the object is
// ready the instant you have it. There are no event-returning variants.
//
// MECHANISM, chosen by kind and type at compile time:
//   * a host-accessible kind (shared, host): a plain host write. No device work
//     at all, and no queue involved.
//   * device kind, trivially copyable T: constructed on the host and copied in
//     through the stream, retargeted to the placement's device.
//   * otherwise: storage, then ONE single_task performing placement-new. The
//     constructor runs where the object lives, so every invariant and interior
//     pointer is device-correct -- host-constructing such an object and copying
//     its bytes would plant host addresses in device memory, and this path
//     makes that mistake unwritable.
//
// THE TYPE GATE:
//   * T is trivially destructible -- the handle has no queue to run ~T on;
//   * T is NOT polymorphic -- constructing the object on the device is what
//     would make its vtable device-correct, but AdaptiveCpp's generic/SSCP flow
//     cannot resolve device vtables at all: the call fails outright on the host
//     backend and returns silently wrong results on CUDA;
//   * arguments are device-copyable, since they are kernel captures;
//   * array forms require trivially copyable elements.
// The gate is a concept, so a rejection is OBSERVABLE:
// `requires { make_unique_device<T>(q); }` answers false.

#include <cstddef>
#include <memory>
#include <new>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>

#include <sycl/sycl.hpp>

#include "naga/slabkit/arena.hpp"
#include "naga/slabkit/detail/core.hpp"
#include "naga/slabkit/detail/placement.hpp"
#include "naga/slabkit/handles.hpp"
#include "naga/slabkit/kinds.hpp"
#include "naga/slabkit/pool.hpp"
#include "naga/slabkit/queue.hpp"

namespace naga::slabkit {
namespace detail {

// -----------------------------------------------------------------------------
// The type gate
// -----------------------------------------------------------------------------

template <class T>
concept usm_object = std::is_trivially_destructible_v<T> && !std::is_polymorphic_v<T>;

/// Array forms initialize through fill/copy, so elements must additionally be
/// trivially copyable.
template <class E>
concept usm_element = usm_object<E> && std::is_trivially_copyable_v<E>;

template <class T>
using elem_t = std::remove_extent_t<T>;

/// What make_* accepts at all: a single object, or an unbounded array of
/// elements it can fill and copy.
template <class T>
concept makeable = (!std::is_array_v<T> && usm_object<T>) ||
                   (std::is_unbounded_array_v<T> && usm_element<elem_t<T>>);

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
void construct_on_device(const Site& s, T* p, const Args&... args) {
    const std::tuple<Args...> captured{args...};
    auto kernel = [=]() {
        std::apply([p](const Args&... a) { ::new (static_cast<void*>(p)) T(a...); }, captured);
    };

    if (s.has_stream())
        s.stream().single_task(s.device(), kernel).wait();
    else
        s.interop_queue().single_task(kernel).wait();
}

// -----------------------------------------------------------------------------
// The forms
// -----------------------------------------------------------------------------

template <class T, class... Args>
unique_usm_ptr<T> make_object(const Site& s, Args&&... args) {
    auto h = retype<T>(require(s.storage(sizeof(T), alignof(T), alignof(T), {}), sizeof(T)));

    if constexpr (std::is_trivially_copyable_v<T>) {
        T tmp = T(std::forward<Args>(args)...);  // value-initialized when there are no args
        s.write(h.get(), &tmp, sizeof(T));
    } else {
        gate_args<Args...>();
        construct_on_device<T>(s, h.get(), static_cast<const std::decay_t<Args>&>(args)...);
    }
    return h;
}

template <class T>
unique_usm_ptr<T> storage_only(const Site& s) {
    return retype<T>(require(s.storage(sizeof(T), alignof(T), alignof(T), {}), sizeof(T)));
}

template <class TArr>
unique_usm_ptr<TArr> storage_only(const Site& s, std::size_t n) {
    using E = elem_t<TArr>;
    return retype<TArr>(require(s.storage(n * sizeof(E), alignof(E), alignof(E), {}), n * sizeof(E)));
}

/// Value-initialized array.
template <class TArr>
unique_usm_ptr<TArr> make_array(const Site& s, std::size_t n) {
    using E = elem_t<TArr>;
    auto h  = storage_only<TArr>(s, n);
    if (n > 0) {
        const E value = E();
        s.fill(h.get(), &value, sizeof(E), n);
    }
    return h;
}

/// Filled array.
template <class TArr>
unique_usm_ptr<TArr> make_array(const Site& s, std::size_t n, const elem_t<TArr>& value) {
    auto h = storage_only<TArr>(s, n);
    if (n > 0) s.fill(h.get(), &value, sizeof(elem_t<TArr>), n);
    return h;
}

/// Array copied from host data.
template <class TArr>
unique_usm_ptr<TArr> make_array(const Site& s, std::span<const elem_t<TArr>> src) {
    using E = elem_t<TArr>;
    auto h  = storage_only<TArr>(s, src.size());
    if (!src.empty()) s.write(h.get(), src.data(), src.size() * sizeof(E));
    return h;
}

/// The body every make_unique_<kind> shares.
template <class T, kind K, class... A>
unique_usm_ptr<T> make_family(A&&... a) {
    return with_placement<K>(
        [&](const Site& s, auto&&... rest) -> unique_usm_ptr<T> {
            if constexpr (std::is_unbounded_array_v<T>)
                return make_array<T>(s, std::forward<decltype(rest)>(rest)...);
            else
                return make_object<T>(s, std::forward<decltype(rest)>(rest)...);
        },
        std::forward<A>(a)...);
}

template <class T, kind K, class... A>
unique_usm_ptr<T> overwrite_family(A&&... a) {
    return with_placement<K>(
        [&](const Site& s, auto&&... rest) -> unique_usm_ptr<T> {
            return storage_only<T>(s, std::forward<decltype(rest)>(rest)...);
        },
        std::forward<A>(a)...);
}

}  // namespace detail

// =============================================================================
// The public families
// =============================================================================

template <class T, class... A>
    requires detail::makeable<T>
unique_usm_ptr<T> make_unique_device(A&&... a) {
    return detail::make_family<T, kind::device>(std::forward<A>(a)...);
}

template <class T, class... A>
    requires detail::makeable<T>
unique_usm_ptr<T> make_unique_shared(A&&... a) {
    return detail::make_family<T, kind::shared>(std::forward<A>(a)...);
}

/// The host kind takes a queue or a context for the backend and NO device.
template <class T, class... A>
    requires detail::makeable<T> && detail::placement_kind_ok<kind::host, A...>
unique_usm_ptr<T> make_unique_host(A&&... a) {
    return detail::make_family<T, kind::host>(std::forward<A>(a)...);
}

template <class T, class... A>
    requires detail::makeable<T>
unique_usm_ptr<T> make_unique_device_for_overwrite(A&&... a) {
    return detail::overwrite_family<T, kind::device>(std::forward<A>(a)...);
}

template <class T, class... A>
    requires detail::makeable<T>
unique_usm_ptr<T> make_unique_shared_for_overwrite(A&&... a) {
    return detail::overwrite_family<T, kind::shared>(std::forward<A>(a)...);
}

template <class T, class... A>
    requires detail::makeable<T> && detail::placement_kind_ok<kind::host, A...>
unique_usm_ptr<T> make_unique_host_for_overwrite(A&&... a) {
    return detail::overwrite_family<T, kind::host>(std::forward<A>(a)...);
}

}  // namespace naga::slabkit
