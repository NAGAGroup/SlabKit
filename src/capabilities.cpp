// =============================================================================
// capabilities.cpp — the answers, and where each one comes from
// =============================================================================
//
// Three sources, fixed precedence: measured > queried > documented.
//
//   QUERIED     the backend answers the SYCL aspect honestly. OpenCL and OMP
//               do; that is the whole list.
//   DOCUMENTED  slabkit's own table for the backend, pinned to an AdaptiveCpp
//               version. This exists because AdaptiveCpp hard-codes the USM
//               aspects true on CUDA, HIP and Level Zero and never maps
//               usm_atomic_shared_allocations at all, so device::has cannot be
//               believed there.
//   MEASURED    a documented `unknown` for a same-backend pair, resolved by
//               doing the thing: a small allocation on one device, a kernel on
//               the other dereferencing it, the result read back. Nothing is
//               retained. context runs these once for its own pairs.
//
// The documented table, pinned to AdaptiveCpp 2026.09.03:
//
//   OMP          one device; pair aspects not applicable. USM trio queried.
//   OpenCL       USM trio queried; pair aspects unknown -> measured.
//   CUDA         d2d yes; device_p2p, shared_p2p unknown (peer access is
//                enabled automatically but is topology limited) -> measured.
//                USM trio true/true/true.
//   HIP          as CUDA.
//   Level Zero   d2d yes where the driver allows multi-device contexts;
//                device_p2p, shared_p2p unknown -> measured.
//                USM trio true/true/true.
//   Metal        one device. USM trio true/true/true (shared is emulated).
//   Vulkan       one device. USM trio true/false/false.

#include "naga/slabkit/capabilities.hpp"

#include <stdexcept>

#include "naga/slabkit/context.hpp"

#include <sycl/sycl.hpp>

namespace naga::slabkit::detail {
namespace {

// -----------------------------------------------------------------------------
// The forwarded half of the vocabulary
// -----------------------------------------------------------------------------

bool forwarded(aspect a, sycl::aspect& out) {
    switch (a) {
        case aspect::cpu: out = sycl::aspect::cpu; return true;
        case aspect::gpu: out = sycl::aspect::gpu; return true;
        case aspect::accelerator: out = sycl::aspect::accelerator; return true;
        case aspect::custom: out = sycl::aspect::custom; return true;
        case aspect::emulated: out = sycl::aspect::emulated; return true;
        case aspect::host_debuggable: out = sycl::aspect::host_debuggable; return true;
        case aspect::fp16: out = sycl::aspect::fp16; return true;
        case aspect::fp64: out = sycl::aspect::fp64; return true;
        case aspect::atomic64: out = sycl::aspect::atomic64; return true;
        case aspect::image: out = sycl::aspect::image; return true;
        case aspect::online_compiler: out = sycl::aspect::online_compiler; return true;
        case aspect::online_linker: out = sycl::aspect::online_linker; return true;
        case aspect::queue_profiling: out = sycl::aspect::queue_profiling; return true;
        case aspect::usm_device_allocations: out = sycl::aspect::usm_device_allocations; return true;
        case aspect::usm_host_allocations: out = sycl::aspect::usm_host_allocations; return true;
        case aspect::usm_atomic_host_allocations:
            out = sycl::aspect::usm_atomic_host_allocations;
            return true;
        case aspect::usm_shared_allocations: out = sycl::aspect::usm_shared_allocations; return true;
        case aspect::usm_atomic_shared_allocations:
            out = sycl::aspect::usm_atomic_shared_allocations;
            return true;
        case aspect::usm_system_allocations: out = sycl::aspect::usm_system_allocations; return true;
        case aspect::AdaptiveCpp_free_memory: out = sycl::aspect::AdaptiveCpp_free_memory; return true;
        default: return false;
    }
}

bool backend_identity(aspect a, sycl::backend& out) {
    switch (a) {
        case aspect::backend_omp: out = sycl::backend::omp; return true;
        case aspect::backend_cuda: out = sycl::backend::cuda; return true;
        case aspect::backend_hip: out = sycl::backend::hip; return true;
        case aspect::backend_level_zero: out = sycl::backend::level_zero; return true;
        case aspect::backend_metal: out = sycl::backend::metal; return true;
        case aspect::backend_vulkan: out = sycl::backend::vk; return true;
        default: return false;
    }
}

/// The backends whose SYCL USM aspects are worth believing.
bool answers_usm_honestly(sycl::backend b) {
    return b == sycl::backend::ocl || b == sycl::backend::omp;
}

answer documented_usm(sycl::backend b, aspect a) {
    switch (b) {
        case sycl::backend::cuda:
        case sycl::backend::hip:
        case sycl::backend::level_zero:
        case sycl::backend::metal:  // shared is emulated, but it works
            return answer::yes;
        case sycl::backend::vk:
            return a == aspect::has_device ? answer::yes : answer::no;
        default:
            // Every backend AdaptiveCpp has is listed above or answers
            // honestly; a new one is honestly unknown rather than assumed.
            return answer::unknown;
    }
}

answer from_bool(bool b) { return b ? answer::yes : answer::no; }

// -----------------------------------------------------------------------------
// Measurement — the only place slabkit runs device work to answer a question
// -----------------------------------------------------------------------------

/// A kernel on `user` writes through an allocation owned by `owner`. Returns
/// `no` for any failure at all: a capability that cannot be demonstrated is not
/// one a program may rely on.
answer measure_p2p(const sycl::device& owner, const sycl::device& user, kind k) {
    try {
        sycl::context ctx = owner.get_platform().khr_get_default_context();
        sycl::queue   qo{ctx, owner, sycl::property_list{sycl::property::queue::in_order{}}};
        sycl::queue   qu{ctx, user, sycl::property_list{sycl::property::queue::in_order{}}};

        void* raw = (k == kind::shared) ? sycl::malloc_shared(sizeof(int), owner, ctx)
                                        : sycl::malloc_device(sizeof(int), owner, ctx);
        if (raw == nullptr) return answer::no;
        auto* p = static_cast<int*>(raw);

        const int seed = 41;
        qo.memcpy(p, &seed, sizeof(int)).wait();
        qu.single_task([=]() { p[0] = p[0] + 1; }).wait();

        int out = 0;
        qo.memcpy(&out, p, sizeof(int)).wait();
        sycl::free(p, ctx);
        return from_bool(out == seed + 1);
    } catch (...) {
        return answer::no;
    }
}

/// A memcpy between device allocations on the two, within the one context.
answer measure_d2d(const sycl::device& src, const sycl::device& dst) {
    try {
        sycl::context ctx = src.get_platform().khr_get_default_context();
        sycl::queue   qs{ctx, src, sycl::property_list{sycl::property::queue::in_order{}}};
        sycl::queue   qd{ctx, dst, sycl::property_list{sycl::property::queue::in_order{}}};

        void* a = sycl::malloc_device(sizeof(int), src, ctx);
        void* b = sycl::malloc_device(sizeof(int), dst, ctx);
        if (a == nullptr || b == nullptr) {
            if (a) sycl::free(a, ctx);
            if (b) sycl::free(b, ctx);
            return answer::no;
        }

        const int seed = 7;
        qs.memcpy(a, &seed, sizeof(int)).wait();
        qd.memcpy(b, a, sizeof(int)).wait();

        int out = 0;
        qd.memcpy(&out, b, sizeof(int)).wait();
        sycl::free(a, ctx);
        sycl::free(b, ctx);
        return from_bool(out == seed);
    } catch (...) {
        return answer::no;
    }
}

/// The documented answer for a same-backend pair, before measurement.
answer documented_pair(sycl::backend b, aspect a) {
    if (a == aspect::shares_context) return answer::yes;  // same platform, same default context

    switch (b) {
        case sycl::backend::cuda:
        case sycl::backend::hip:
            return a == aspect::d2d ? answer::yes : answer::unknown;
        case sycl::backend::level_zero:
            // d2d holds where the driver allows a multi-device context; the
            // pair only exists if it did, so the table says yes.
            return a == aspect::d2d ? answer::yes : answer::unknown;
        case sycl::backend::omp:
        case sycl::backend::metal:
        case sycl::backend::vk:
            // One device per backend, so a two-device pair on these is outside
            // what the table describes.
            return answer::unknown;
        default: return answer::unknown;
    }
}

}  // namespace

// -----------------------------------------------------------------------------
// The public core
// -----------------------------------------------------------------------------

answer satisfies_device(aspect a, const sycl::device& dev) {
    if (is_pair_aspect(a))
        throw std::invalid_argument{
            "naga::slabkit: a pair aspect cannot be answered for a single device"};

    if (sycl::backend b{}; backend_identity(a, b)) return from_bool(dev.get_backend() == b);

    if (a == aspect::has_device || a == aspect::has_shared || a == aspect::has_host) {
        const sycl::backend b = dev.get_backend();
        if (answers_usm_honestly(b)) {
            const sycl::aspect asp = (a == aspect::has_device) ? sycl::aspect::usm_device_allocations
                                     : (a == aspect::has_shared)
                                         ? sycl::aspect::usm_shared_allocations
                                         : sycl::aspect::usm_host_allocations;
            return from_bool(dev.has(asp));
        }
        return documented_usm(b, a);
    }

    if (sycl::aspect asp{}; forwarded(a, asp)) return from_bool(dev.has(asp));

    return answer::unknown;
}

answer satisfies_pair(aspect a, const sycl::device& x, const sycl::device& y) {
    if (!is_pair_aspect(a))
        throw std::invalid_argument{
            "naga::slabkit: a device aspect cannot be answered for a pair of devices"};

    // Cross-backend pairs are `no`, never `unknown`: two platforms cannot share
    // a context, and every pair aspect presupposes one.
    if (!(x.get_platform() == y.get_platform())) return answer::no;

    // A device with itself trivially satisfies every pair aspect, and measuring
    // it would be an expensive way to learn nothing.
    if (x == y) return answer::yes;

    const answer doc = documented_pair(x.get_backend(), a);
    if (doc != answer::unknown) return doc;

    switch (a) {
        case aspect::d2d: return measure_d2d(x, y);
        case aspect::device_p2p: return measure_p2p(x, y, kind::device);
        case aspect::shared_p2p: return measure_p2p(x, y, kind::shared);
        default: return doc;
    }
}

}  // namespace naga::slabkit::detail

// =============================================================================
// copy_routes — derived from the pair answers and the kinds, never empty
// =============================================================================

namespace naga::slabkit {

std::vector<route> context::copy_routes(kind src_kind, const sycl::device& src, kind dst_kind,
                                        const sycl::device& dst) {
    std::vector<route> routes;

    // d2d: one backend copy. Needs one context over both devices, and both
    // allocations resident on that backend.
    if (device_resident(src_kind) && device_resident(dst_kind) &&
        detail::satisfies_pair(aspect::shares_context, src, dst) == answer::yes &&
        detail::satisfies_pair(aspect::d2d, src, dst) == answer::yes) {
        const bool measured = (src != dst) && src.get_backend() != sycl::backend::cuda &&
                              src.get_backend() != sycl::backend::hip &&
                              src.get_backend() != sycl::backend::level_zero;
        routes.push_back({route::via::d2d, measured ? provenance::measured : provenance::documented});
    }

    // host_memcpy: both allocations are host-accessible, so host code may copy
    // between them directly. Legal regardless of backend.
    if (host_accessible(src_kind) && host_accessible(dst_kind))
        routes.push_back({route::via::host_memcpy, provenance::documented});

    // staged: D2H into a host intermediate, then H2D. Always available, always
    // last, which is what makes this list never empty.
    routes.push_back({route::via::staged, provenance::documented});

    return routes;
}

}  // namespace naga::slabkit
