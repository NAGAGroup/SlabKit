#pragma once

// =============================================================================
// naga/slabkit/capabilities.hpp — the aspect vocabulary and its answers
// =============================================================================
//
// What the machine can do, asked in one vocabulary and answered honestly.
//
// SYCL's own aspects are forwarded unchanged -- same names, same meaning. On
// top of them slabkit adds three groups:
//
//   * the HONEST USM TRIO (has_device, has_shared, has_host). AdaptiveCpp
//     hard-codes usm_device_allocations, usm_host_allocations and
//     usm_atomic_host_allocations to true on CUDA, HIP and Level Zero, and
//     never maps usm_atomic_shared_allocations at all, so sycl::device::has
//     cannot drive a placement decision. These three answer from the honest
//     source instead: queried where the backend answers truthfully (OpenCL,
//     OMP), documented otherwise.
//   * BACKEND IDENTITY, so "a CUDA device" is expressible in the same
//     vocabulary as "a device with fp64".
//   * PAIR ASPECTS, which are properties of two devices rather than one. In a
//     selector, a pair aspect means "with every other selected device of the
//     same backend".
//
// An answer has three values because two would be a lie: `unknown` is what a
// documented table says when the honest answer depends on hardware the table
// cannot see (CUDA peer access is enabled automatically but is topology
// limited). A documented `unknown` for a same-backend pair is resolved by
// MEASURING it -- see context, which does so once for its own pairs.
//
// PRECEDENCE, fixed: measured > queried > documented.
//
// Performance is deliberately NOT an aspect. Intel dGPUs migrate whole
// allocations, AMD shared memory without XNACK behaves as pinned host memory,
// and WSL has no prefetch: those are README paragraphs beside a backend, not
// yes/no answers a program can branch on.

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/slabkit/kinds.hpp"

namespace naga::slabkit {

enum class aspect {
    // ---- every sycl::aspect, same names, same meaning (forwarded) -----------
    cpu,
    gpu,
    accelerator,
    custom,
    emulated,
    host_debuggable,
    fp16,
    fp64,
    atomic64,
    image,
    online_compiler,
    online_linker,
    queue_profiling,
    usm_device_allocations,
    usm_host_allocations,
    usm_atomic_host_allocations,
    usm_shared_allocations,
    usm_atomic_shared_allocations,
    usm_system_allocations,
    AdaptiveCpp_free_memory,

    // ---- the honest USM trio ------------------------------------------------
    has_device,
    has_shared,
    has_host,

    // ---- backend identity ---------------------------------------------------
    backend_omp,
    backend_cuda,
    backend_hip,
    backend_level_zero,
    backend_metal,
    backend_vulkan,

    // ---- pair aspects -------------------------------------------------------
    shares_context,  // same backend -> same platform default context
    d2d,             // a memcpy between device allocations on the two is a direct backend copy
    device_p2p,      // a kernel on one may dereference a device allocation of the other
    shared_p2p,      // a kernel on one may dereference a shared allocation of the other
};

/// The number of aspects, so a fixed-size answer cache can be indexed by one.
inline constexpr std::size_t aspect_count = static_cast<std::size_t>(aspect::shared_p2p) + 1;

/// A pair aspect is a property of TWO devices; the single-device queries reject
/// it and the two-device queries require it.
constexpr bool is_pair_aspect(aspect a) noexcept {
    return a == aspect::shares_context || a == aspect::d2d || a == aspect::device_p2p ||
           a == aspect::shared_p2p;
}

enum class answer { yes, no, unknown };
enum class provenance { queried, documented, measured };

struct route {
    enum class via { d2d, host_memcpy, staged } method;
    provenance                                  source;
};

/// Two devices share a backend iff their platforms compare equal, so the
/// platform IS the backend identity slabkit keys everything on.
using backend_id = sycl::platform;

/// The result shape of a pair-aspect selection: one entry per group, each a
/// backend and the devices of that group.
using device_groups = std::vector<std::pair<sycl::platform, std::vector<sycl::device>>>;

/// What `context::select_devices<A>()` returns, by the kind of aspect A is.
template <aspect A>
struct selection {
    using type = std::conditional_t<is_pair_aspect(A), device_groups, std::vector<sycl::device>>;
};

namespace detail {

// The compiled core. Nothing above the aspect vocabulary is a template, so a
// consumer instantiating `satisfies<aspect::fp64>` compiles a function call.

/// \throws std::invalid_argument if `a` is a pair aspect.
answer satisfies_device(aspect a, const sycl::device& dev);

/// The measured/documented answer for an ordered pair. Devices on different
/// backends answer `no`, never `unknown`; a device paired with itself answers
/// `yes`. A documented `unknown` is measured here, on the call.
/// \throws std::invalid_argument if `a` is not a pair aspect.
answer satisfies_pair(aspect a, const sycl::device& x, const sycl::device& y);

}  // namespace detail
}  // namespace naga::slabkit
