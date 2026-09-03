#pragma once

// =============================================================================
// naga/slabkit/mirrors.hpp — the four routed mirrors
// =============================================================================
//
// memcpy, copy, memset and fill, over any pair of allocations slabkit's context
// can locate.
//
// EVERY MIRROR IS HOST WORK. Each call is one host item of the stream (queue.hpp
// §"two kinds of work"): it returns immediately with that item's event, runs off
// the caller's thread, begins after the kernel work outstanding when it was
// called, and runs in parallel with other host work. Nothing here blocks.
//
// ROUTING. memcpy and copy locate their operands through the queue's context
// and take copy_routes(...).front():
//
//   d2d          one sycl::queue::memcpy on the DESTINATION's backend queue,
//                issued and waited inside the host task.
//   host_memcpy  a plain std::memcpy between two host-accessible allocations.
//   staged       D2H into an operation-owned host intermediate, then H2D: a
//                CHAIN, which the stream counts as one host item. The
//                intermediate dies with the second leg's closure.
//
// WHY THE COPY IS ISSUED FROM INSIDE THE TASK. A copy whose host operand is not
// USM in the issuing queue's context has that operand read when the copy is
// ISSUED -- the driver stages pageable memory through a pinned buffer at issue
// time -- so a stream-level dependency cannot protect it, and was measured wrong
// 40 times in 40. The host task waits for its dependencies FIRST and issues
// after, which is the only ordering that survives that. A pinned intermediate is
// not an alternative: it would have to be USM in the source's and the
// destination's context at once, which is impossible.
//
// A LEG BETWEEN TWO HOST-SIDE OPERANDS IS A PLAIN memcpy, never a submission: a
// host task that submits to the OpenMP backend and waits on it deadlocks. A
// pointer no backend context recognises is a HOST operand -- always; there is no
// unrecognised operand, which is what makes an ordinary std::vector a legal
// source or destination.
//
// memset and fill have one operand and never route: a host store when the
// operand is host-side, otherwise one submission on that operand's own device,
// issued and waited inside the host task.

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

/// The same copy as a member of a GROUP rather than an item of the stream: the
/// group orders it against the step before it and against nothing else, and a
/// staged copy contributes its chain's LAST event to the step. §6's mirrors name
/// a queue, so this is the only spelling of a routed copy inside a group.
sycl::event copy_bytes_in_group(queue::group& g, queue& q, void* dst, const void* src,
                                std::size_t bytes, const std::vector<sycl::event>& deps);

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
