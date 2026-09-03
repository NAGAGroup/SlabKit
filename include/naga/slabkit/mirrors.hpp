#pragma once

// =============================================================================
// naga/slabkit/mirrors.hpp — the four routed mirrors
// =============================================================================
//
// memcpy, copy, memset and fill, over any pair of allocations slabkit's context
// can locate. Each call is ONE item of the stream, so it is ordered against
// everything before it and everything after it without the caller naming an
// event.
//
// ROUTING. memcpy and copy locate their operands through the queue's context
// and take copy_routes(...).front():
//
//   d2d          one sycl::queue::memcpy on the destination's backend queue.
//   host_memcpy  the copy runs on the host pool, after the dependencies and the
//                stream's wait list -- the one place slabkit touches shared
//                memory from the host, and where take-turns is enforced.
//   staged       D2H into an operation-owned host intermediate, then H2D: two
//                ordinary items, so the second waits for the first by the
//                stream's own contract. The intermediate is released by a host
//                task that waits on the second item.
//
// A pointer no backend context recognises is treated as a HOST operand, which
// is what makes an ordinary std::vector a legal source or destination.
//
// memset and fill have one operand and never route: they run on that operand's
// owning device, as one item.
//
// WHERE THIS DIVERGES FROM THE SPEC, and why. The spec asks a host route to
// return "the event of a trailing no-op item ordered after the host work".
// AdaptiveCpp offers no way to build one: there is no host_task, custom
// operations run on the calling thread before their dependencies, and a kernel
// polling a host flag would be exactly the concurrent host/device access to
// shared memory that slabkit's take-turns rule refuses everywhere else. So the
// host route DOES return such an event -- but the call blocks until the host
// copy has finished, because that is the only honest way to order an item after
// it. The copy still runs off the caller's thread and after the work it
// depends on; the caller merely waits for it.

#include <cstddef>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/slabkit/capabilities.hpp"
#include "naga/slabkit/queue.hpp"

namespace naga::slabkit {

namespace detail {

/// The compiled core. `bytes == 0` is a no-op item, not an error.
sycl::event copy_bytes(queue& q, void* dst, const void* src, std::size_t bytes,
                       const std::vector<sycl::event>& deps);

/// The same copy, forced onto a named route. Exposed so the suite can exercise
/// every route on a machine whose topology would only ever choose one.
sycl::event copy_bytes_via(queue& q, void* dst, const void* src, std::size_t bytes,
                           route::via method, const std::vector<sycl::event>& deps);

/// The route this pair of operands would take.
route chosen_route(queue& q, const void* dst, const void* src);

sycl::event fill_bytes(queue& q, void* p, const void* pattern, std::size_t pattern_size,
                       std::size_t count, const std::vector<sycl::event>& deps);

}  // namespace detail

// -----------------------------------------------------------------------------
// memcpy / copy
// -----------------------------------------------------------------------------

inline sycl::event memcpy(queue& q, void* dst, const void* src, std::size_t bytes) {
    return detail::copy_bytes(q, dst, src, bytes, {});
}

inline sycl::event memcpy(queue& q, void* dst, const void* src, std::size_t bytes,
                          const std::vector<sycl::event>& deps) {
    return detail::copy_bytes(q, dst, src, bytes, deps);
}

template <class T>
sycl::event copy(queue& q, const T* src, T* dst, std::size_t count) {
    return detail::copy_bytes(q, dst, src, count * sizeof(T), {});
}

template <class T>
sycl::event copy(queue& q, const T* src, T* dst, std::size_t count,
                 const std::vector<sycl::event>& deps) {
    return detail::copy_bytes(q, dst, src, count * sizeof(T), deps);
}

// -----------------------------------------------------------------------------
// memset / fill — one operand, no route
// -----------------------------------------------------------------------------

sycl::event memset(queue& q, void* p, int value, std::size_t bytes);
sycl::event memset(queue& q, void* p, int value, std::size_t bytes,
                   const std::vector<sycl::event>& deps);

template <class T>
sycl::event fill(queue& q, void* p, const T& pattern, std::size_t count) {
    return detail::fill_bytes(q, p, &pattern, sizeof(T), count, {});
}

template <class T>
sycl::event fill(queue& q, void* p, const T& pattern, std::size_t count,
                 const std::vector<sycl::event>& deps) {
    return detail::fill_bytes(q, p, &pattern, sizeof(T), count, deps);
}

}  // namespace naga::slabkit
