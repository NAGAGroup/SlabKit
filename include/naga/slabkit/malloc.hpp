#pragma once

// =============================================================================
// naga/slabkit/malloc.hpp — STORAGE: the malloc family
// =============================================================================
//
// SYCL's USM signatures with T* replaced by an owning handle, over four
// placements: a queue, a device with its context, an arena, a pool.
//
// These return uninitialized storage for `count` elements, so the handle is an
// array handle: unique_device_ptr<T[]>.
//
// CONTRACTS ARE BORROWED. A queue or device placement returns null exactly
// when the underlying sycl::malloc_* returns null; a zero-element request is
// passed through and whatever the backend does is what you get, because that
// is implementation-defined and slabkit adds no rule of its own. An arena or
// pool returns null when it cannot satisfy the request, and for a zero-element
// request (there is no underlying SYCL call whose behaviour could be mirrored).
//
// ALIGNMENT. The queue and device forms behave as SYCL does -- which on
// AdaptiveCpp's CUDA backend means the requested alignment is ignored and you
// get cudaMalloc's own guarantee. An arena honors any alignment. A pool's
// allocations are node_size-aligned, and asking for more is std::invalid_argument.

#include <cstddef>

#include <sycl/sycl.hpp>

#include "naga/slabkit/arena.hpp"
#include "naga/slabkit/detail/core.hpp"
#include "naga/slabkit/handles.hpp"
#include "naga/slabkit/pool.hpp"

namespace naga::slabkit {

// -----------------------------------------------------------------------------
// malloc_device
// -----------------------------------------------------------------------------

template <class T>
unique_device_ptr<T[]> malloc_device(std::size_t count, const sycl::queue& q,
                                     const sycl::property_list& props = {}) {
    return detail::retype<T[]>(detail::malloc_bytes(count * sizeof(T), q, props));
}

template <class T>
unique_device_ptr<T[]> malloc_device(std::size_t count, const sycl::device& dev,
                                     const sycl::context&       ctx,
                                     const sycl::property_list& props = {}) {
    return detail::retype<T[]>(detail::malloc_bytes(count * sizeof(T), dev, ctx, props));
}

template <class T>
unique_device_ptr<T[]> malloc_device(std::size_t count, DeviceArena& arena) {
    return detail::retype<T[]>(detail::arena_bytes(count * sizeof(T), alignof(T), arena));
}

template <class T>
unique_device_ptr<T[]> malloc_device(std::size_t count, DevicePool& pool) {
    return detail::retype<T[]>(detail::pool_bytes(count * sizeof(T), alignof(T), pool));
}

// -----------------------------------------------------------------------------
// aligned_alloc_device
// -----------------------------------------------------------------------------

template <class T>
unique_device_ptr<T[]> aligned_alloc_device(std::size_t align, std::size_t count,
                                            const sycl::queue&         q,
                                            const sycl::property_list& props = {}) {
    return detail::retype<T[]>(detail::aligned_alloc_bytes(align, count * sizeof(T), q, props));
}

template <class T>
unique_device_ptr<T[]> aligned_alloc_device(std::size_t align, std::size_t count,
                                            const sycl::device& dev, const sycl::context& ctx,
                                            const sycl::property_list& props = {}) {
    return detail::retype<T[]>(
        detail::aligned_alloc_bytes(align, count * sizeof(T), dev, ctx, props));
}

template <class T>
unique_device_ptr<T[]> aligned_alloc_device(std::size_t align, std::size_t count,
                                            DeviceArena& arena) {
    return detail::retype<T[]>(detail::arena_bytes(count * sizeof(T), align, arena));
}

template <class T>
unique_device_ptr<T[]> aligned_alloc_device(std::size_t align, std::size_t count,
                                            DevicePool& pool) {
    return detail::retype<T[]>(detail::pool_bytes(count * sizeof(T), align, pool));
}

}  // namespace naga::slabkit
