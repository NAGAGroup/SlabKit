#pragma once

// =============================================================================
// naga/slabkit/malloc.hpp — STORAGE: the malloc family
// =============================================================================
//
// SYCL's USM signatures with T* replaced by an owning handle, over three kinds
// and six placements. These return uninitialized storage for `count` elements,
// so the handle is an array handle: unique_usm_ptr<T[]>.
//
// CONTRACTS ARE BORROWED. A queue or device placement returns null exactly when
// the underlying sycl::malloc_* returns null; a zero-element request is passed
// through and whatever the backend does is what you get, because that is
// implementation-defined and slabkit adds no rule of its own. An arena or pool
// returns null when it cannot satisfy the request, and for a zero-element
// request (there is no underlying SYCL call whose behaviour could be mirrored).
//
// ALIGNMENT. The queue and device forms behave as SYCL does -- which on
// AdaptiveCpp's CUDA backend means the requested alignment is IGNORED and you
// get cudaMalloc's own guarantee. An arena honors any alignment. A pool's
// allocations are node_size-aligned, and asking for more is
// std::invalid_argument.
//
// NAME COLLISION. These names are also sycl::'s, and both live in namespaces
// reachable by ADL from a call whose arguments include a sycl::queue or
// sycl::device. Call them QUALIFIED -- naga::slabkit::malloc_device<T>(...) --
// or the wrong one may win.
//
// <std::byte> is the void* form.

#include <cstddef>

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

/// The one body every storage function shares.
template <class T, kind K, class... P>
unique_usm_ptr<T[]> storage_family(std::size_t count, std::size_t req_align, P&&... p) {
    return with_placement<K>(
        [&](const Site& s, const sycl::property_list& props = {}) {
            return retype<T[]>(s.storage(count * sizeof(T), req_align, alignof(T), props));
        },
        std::forward<P>(p)...);
}

}  // namespace detail

// -----------------------------------------------------------------------------
// malloc_* — nothing is asked of the runtime's alignment
// -----------------------------------------------------------------------------

template <class T, class... P>
unique_usm_ptr<T[]> malloc_device(std::size_t count, P&&... placement) {
    return detail::storage_family<T, kind::device>(count, 0, std::forward<P>(placement)...);
}

template <class T, class... P>
unique_usm_ptr<T[]> malloc_shared(std::size_t count, P&&... placement) {
    return detail::storage_family<T, kind::shared>(count, 0, std::forward<P>(placement)...);
}

/// The host kind takes a queue or a context for the backend and NO device:
/// naming one is a compile error, and an observable one.
template <class T, class... P>
    requires detail::placement_kind_ok<kind::host, P...>
unique_usm_ptr<T[]> malloc_host(std::size_t count, P&&... placement) {
    return detail::storage_family<T, kind::host>(count, 0, std::forward<P>(placement)...);
}

// -----------------------------------------------------------------------------
// aligned_alloc_* — the alignment leads, as SYCL spells it
// -----------------------------------------------------------------------------

template <class T, class... P>
unique_usm_ptr<T[]> aligned_alloc_device(std::size_t align, std::size_t count, P&&... placement) {
    return detail::storage_family<T, kind::device>(count, align, std::forward<P>(placement)...);
}

template <class T, class... P>
unique_usm_ptr<T[]> aligned_alloc_shared(std::size_t align, std::size_t count, P&&... placement) {
    return detail::storage_family<T, kind::shared>(count, align, std::forward<P>(placement)...);
}

template <class T, class... P>
    requires detail::placement_kind_ok<kind::host, P...>
unique_usm_ptr<T[]> aligned_alloc_host(std::size_t align, std::size_t count, P&&... placement) {
    return detail::storage_family<T, kind::host>(count, align, std::forward<P>(placement)...);
}

}  // namespace naga::slabkit
