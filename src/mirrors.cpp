// =============================================================================
// mirrors.cpp — locating operands, choosing a route, and taking it
// =============================================================================

#include "naga/slabkit/mirrors.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

#include <sycl/sycl.hpp>

#include "detail/os_alloc.hpp"
#include "naga/slabkit/context.hpp"

namespace naga::slabkit {
namespace detail {
namespace {

struct operand {
    kind         k;
    sycl::device dev;
    bool         recognised;
};

/// Where an operand lives. A pointer no backend context knows is a HOST
/// operand -- that is what makes an ordinary std::vector a legal source, and it
/// is the reading the routing rules need; the alternative (errc::invalid) would
/// make H2D and D2H copies unspellable.
operand locate(const context& c, const void* p) {
    try {
        return operand{c.get_pointer_type(p), c.get_pointer_device(p), true};
    } catch (const sycl::exception&) {
        return operand{kind::host, c.devices().front(), false};
    }
}

sycl::event no_op(queue& q, const std::vector<sycl::event>& deps) {
    return q.single_task(deps, [] {});
}

/// One backend copy on the destination's queue.
sycl::event via_d2d(queue& q, const sycl::device& dst_dev, void* dst, const void* src,
                    std::size_t bytes, const std::vector<sycl::event>& deps) {
    return q.submit(dst_dev, deps, [&](sycl::handler& h) { h.memcpy(dst, src, bytes); });
}

/// Host code copies between two host-accessible allocations, on the pool, after
/// everything already in flight. See the header for why this blocks.
sycl::event via_host(queue& q, void* dst, const void* src, std::size_t bytes,
                     const std::vector<sycl::event>& deps) {
    std::vector<sycl::event> wait_for = deps;
    for (auto& e : q.get_wait_list()) wait_for.push_back(e);

    auto done = q.submit_host([wait_for, dst, src, bytes]() mutable {
        for (auto& e : wait_for) e.wait();
        std::memcpy(dst, src, bytes);
    });
    done.get();  // propagates an exception from the copy, and orders what follows

    return no_op(q, {});
}

/// D2H into an operation-owned intermediate, then H2D. Two items, so the second
/// waits for the first by the stream's own contract.
sycl::event via_staged(queue& q, const sycl::device& src_dev, const sycl::device& dst_dev,
                       void* dst, const void* src, std::size_t bytes,
                       const std::vector<sycl::event>& deps) {
    auto stage = std::make_shared<std::vector<std::byte>>(bytes);

    q.submit(src_dev, deps,
             [&](sycl::handler& h) { h.memcpy(stage->data(), src, bytes); });
    sycl::event landed =
        q.submit(dst_dev, [&](sycl::handler& h) { h.memcpy(dst, stage->data(), bytes); });

    // The intermediate outlives both items and is dropped by a host task that
    // waits on the second -- no blocking, and nothing leaks if the caller never
    // waits at all.
    q.submit_host([landed, stage]() mutable { landed.wait(); });
    return landed;
}

}  // namespace

route chosen_route(queue& q, const void* dst, const void* src) {
    const context& c = q.get_context();
    const operand  s = locate(c, src);
    const operand  d = locate(c, dst);
    return context::copy_routes(s.k, s.dev, d.k, d.dev).front();
}

sycl::event copy_bytes_via(queue& q, void* dst, const void* src, std::size_t bytes,
                           route::via method, const std::vector<sycl::event>& deps) {
    if (bytes == 0) return no_op(q, deps);

    const context& c = q.get_context();
    const operand  s = locate(c, src);
    const operand  d = locate(c, dst);

    switch (method) {
        case route::via::d2d: return via_d2d(q, d.dev, dst, src, bytes, deps);
        case route::via::host_memcpy: return via_host(q, dst, src, bytes, deps);
        case route::via::staged: break;
    }
    return via_staged(q, s.dev, d.dev, dst, src, bytes, deps);
}

sycl::event copy_bytes(queue& q, void* dst, const void* src, std::size_t bytes,
                       const std::vector<sycl::event>& deps) {
    if (bytes == 0) return no_op(q, deps);
    return copy_bytes_via(q, dst, src, bytes, chosen_route(q, dst, src).method, deps);
}

sycl::event fill_bytes(queue& q, void* p, const void* pattern, std::size_t pattern_size,
                       std::size_t count, const std::vector<sycl::event>& deps) {
    if (p == nullptr || pattern == nullptr || pattern_size == 0 || count == 0)
        return no_op(q, deps);

    const operand      o     = locate(q.get_context(), p);
    const std::size_t  total = pattern_size * count;
    const std::byte*   pat   = static_cast<const std::byte*>(pattern);

    // A pattern of identical bytes -- what value-initializing anything zeroable
    // looks like -- is a memset, and needs no kernel and no staging.
    bool uniform = true;
    for (std::size_t i = 1; i < pattern_size; ++i) uniform = uniform && (pat[i] == pat[0]);

    if (uniform) {
        const int value = static_cast<int>(std::to_integer<unsigned char>(pat[0]));
        return q.submit(o.dev, deps, [&](sycl::handler& h) { h.memset(p, value, total); });
    }

    auto*             out   = static_cast<std::byte*>(p);
    const std::size_t psize = pattern_size;

    // A pattern that fits in a register is copied INTO the kernel, so there is
    // no staging allocation and no second item.
    if (psize <= sizeof(std::uint64_t)) {
        std::uint64_t packed = 0;
        std::memcpy(&packed, pat, psize);
        return q.parallel_for(o.dev, deps, sycl::range<1>{total}, [=](sycl::id<1> i) {
            const auto* bytes = reinterpret_cast<const std::byte*>(&packed);
            out[i[0]]         = bytes[i[0] % psize];
        });
    }

    // A wider one is staged in device memory for the kernel to replicate. The
    // staging allocation is an ordinary slabkit handle, released by a host task
    // that waits on the fill.
    const sycl::context& sctx  = q.get_context().context_of(o.dev);
    auto                 stage = std::make_shared<unique_usm_ptr<std::byte[]>>(
        os_allocate(kind::device, 0, psize, o.dev, sctx, {}));
    if (!*stage) throw std::bad_alloc{};

    const std::byte* src = stage->get();
    q.submit(o.dev, [&](sycl::handler& h) { h.memcpy(stage->get(), pat, psize); });
    sycl::event e = q.parallel_for(o.dev, deps, sycl::range<1>{total},
                                   [=](sycl::id<1> i) { out[i[0]] = src[i[0] % psize]; });
    q.submit_host([e, stage]() mutable { e.wait(); });
    return e;
}

}  // namespace detail

sycl::event memset(queue& q, void* p, int value, std::size_t bytes) {
    return memset(q, p, value, bytes, {});
}

sycl::event memset(queue& q, void* p, int value, std::size_t bytes,
                   const std::vector<sycl::event>& deps) {
    if (p == nullptr || bytes == 0) return q.single_task(deps, [] {});

    const sycl::device dev = [&] {
        try {
            return q.get_context().get_pointer_device(p);
        } catch (const sycl::exception&) {
            return q.get_device();
        }
    }();

    return q.submit(dev, deps, [&](sycl::handler& h) { h.memset(p, value, bytes); });
}

}  // namespace naga::slabkit
