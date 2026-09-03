// =============================================================================
// mirrors.cpp — locating operands, choosing a route, and taking it as host work
// =============================================================================
//
// Every mirror is ONE host item (§6). The caller is never blocked; the copy is
// issued from inside the host task, by which time everything it was ordered
// behind has already completed. That is what makes a pageable operand safe: a
// copy whose host operand is not USM in the issuing queue's context has that
// operand read when the copy is ISSUED, so no dependency could have protected
// it -- only issuing later can.
//
// A leg between two host-side operands is a plain std::memcpy and never a
// submission. Partly because it is faster, and entirely because a host task that
// submits to the OpenMP backend and waits on it deadlocks.

#include "naga/slabkit/mirrors.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <optional>
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
    /// True when the bytes can be reached with a plain host store AND naming
    /// them in a submission would mean submitting to the OpenMP backend: a
    /// non-USM pointer, or an allocation on the OpenMP backend.
    bool host_side;
};

/// Where an operand lives. A pointer no backend context knows is a HOST operand
/// -- always; there is no unrecognised operand. That is what makes an ordinary
/// std::vector a legal source or destination.
operand locate(const context& c, const void* p) {
    try {
        const sycl::device d = c.get_pointer_device(p);
        return operand{c.get_pointer_type(p), d, is_host_backend(d)};
    } catch (const sycl::exception&) {
        return operand{kind::host, c.devices().front(), true};
    }
}

/// One leg of a copy, resolved on the calling thread and carried into the host
/// task by value. An EMPTY target is a plain host memcpy -- and the emptiness is
/// an optional rather than a flag because default-constructing a sycl::queue or
/// a sycl::device would run the default selector and build one.
struct leg {
    struct target {
        sycl::queue  bq;
        sycl::device dev;
    };
    std::optional<target> on;
};

/// Where a copy between these two operands must be issued. The destination wins
/// when both are device-side, which is the d2d rule: one memcpy on the
/// DESTINATION's backend queue.
leg leg_for(queue& q, const operand& dst, const operand& src) {
    leg l;
    if (dst.host_side && src.host_side) return l;

    const sycl::device dev = dst.host_side ? src.dev : dst.dev;
    l.on.emplace(leg::target{backend_queue_for(state_of(q), dev), dev});
    return l;
}

/// Issue one copy leg and wait for it here, on the host task's thread. Nothing
/// downstream ever sees the event, so nothing downstream ever carries a
/// dependency across a backend.
void run_leg(const leg& l, void* dst, const void* src, std::size_t bytes) {
    if (!l.on) {
        std::memcpy(dst, src, bytes);
        return;
    }
    sycl::queue bq = l.on->bq;
    bq.submit(sycl::property_list{sycl::property::command_group::AdaptiveCpp_retarget{l.on->dev}},
              [&](sycl::handler& h) { h.memcpy(dst, src, bytes); })
        .wait();
}

/// Explicit dependencies are waited INSIDE the host task, beside the ordering
/// the stream gave it. A host wait is the whole mechanism here, so an extra one
/// costs nothing and needs no special case.
void wait_all(const std::vector<sycl::event>& deps) {
    for (auto& e : deps) sycl::event{e}.wait();
}

/// A mirror that has nothing to do is still a host item, so that a caller who
/// waits on the returned event waits for the same thing either way.
///
/// `Stream` is either the queue or one of its groups: both spell submit_host and
/// submit_chain the same way, and a copy does not care which of them is ordering
/// it. That is the whole of what makes a routed copy a legal group member.
template <class Stream>
sycl::event no_op_on(Stream& s, const std::vector<sycl::event>& deps) {
    return s.submit_host([deps] { wait_all(deps); });
}

sycl::event no_op(queue& q, const std::vector<sycl::event>& deps) { return no_op_on(q, deps); }

/// The copy itself, over either submitter.
template <class Stream>
sycl::event copy_bytes_on(Stream& s, queue& q, void* dst, const void* src, std::size_t bytes,
                          route::via method, const std::vector<sycl::event>& deps) {
    if (bytes == 0) return no_op_on(s, deps);

    const context& c = q.get_context();
    const operand  from = locate(c, src);
    const operand  to   = locate(c, dst);

    if (method != route::via::staged) {
        // d2d and host_memcpy are both one leg; which of the two it is was
        // already decided by where the operands live.
        const leg l = leg_for(q, to, from);
        return s.submit_host([l, dst, src, bytes, deps] {
            wait_all(deps);
            run_leg(l, dst, src, bytes);
        });
    }

    // Staged: D2H into an operation-owned host intermediate, then H2D. Two host
    // items forced in order -- a chain -- which the stream counts as one.
    const operand host_mid{kind::host, c.devices().front(), true};
    const leg     down = leg_for(q, host_mid, from);
    const leg     up   = leg_for(q, to, host_mid);

    return s.submit_chain([&](queue::chain& ch) {
        auto stage = std::make_shared<std::vector<std::byte>>(bytes);

        ch.then([down, stage, src, bytes, deps] {
            wait_all(deps);
            run_leg(down, stage->data(), src, bytes);
        });
        // The intermediate is dropped when this closure is destroyed, which is
        // after it has run: nothing leaks even if the caller never waits.
        ch.then([up, stage, dst, bytes] { run_leg(up, dst, stage->data(), bytes); });
    });
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
    return copy_bytes_on(q, q, dst, src, bytes, method, deps);
}

sycl::event copy_bytes(queue& q, void* dst, const void* src, std::size_t bytes,
                       const std::vector<sycl::event>& deps) {
    if (bytes == 0) return no_op(q, deps);
    return copy_bytes_via(q, dst, src, bytes, chosen_route(q, dst, src).method, deps);
}

sycl::event copy_bytes_in_group(queue::group& g, queue& q, void* dst, const void* src,
                                std::size_t bytes, const std::vector<sycl::event>& deps) {
    if (bytes == 0) return no_op_on(g, deps);
    return copy_bytes_on(g, q, dst, src, bytes, chosen_route(q, dst, src).method, deps);
}

// -----------------------------------------------------------------------------
// fill — one operand, no route
// -----------------------------------------------------------------------------

namespace {

/// The pattern is COPIED here, on the calling thread. The host task runs later
/// and the caller's `const T&` is long gone by then.
std::shared_ptr<std::vector<std::byte>> own_pattern(const void* pattern, std::size_t size) {
    auto out = std::make_shared<std::vector<std::byte>>(size);
    std::memcpy(out->data(), pattern, size);
    return out;
}

}  // namespace

sycl::event fill_bytes(queue& q, void* p, const void* pattern, std::size_t pattern_size,
                       std::size_t count, const std::vector<sycl::event>& deps) {
    if (p == nullptr || pattern == nullptr || pattern_size == 0 || count == 0)
        return no_op(q, deps);

    const operand     o     = locate(q.get_context(), p);
    const std::size_t total = pattern_size * count;
    const std::byte*  pat   = static_cast<const std::byte*>(pattern);

    // A pattern of identical bytes -- what value-initializing anything zeroable
    // looks like -- is a memset, and needs no kernel and no staging.
    bool uniform = true;
    for (std::size_t i = 1; i < pattern_size; ++i) uniform = uniform && (pat[i] == pat[0]);
    if (uniform) {
        const int value = static_cast<int>(std::to_integer<unsigned char>(pat[0]));
        return naga::slabkit::memset(q, p, value, total, deps);
    }

    auto*             out   = static_cast<std::byte*>(p);
    const std::size_t psize = pattern_size;
    auto              own   = own_pattern(pattern, psize);

    // The operand is host-reachable: replicate it here rather than submitting to
    // the OpenMP backend from a host task, which would deadlock.
    if (o.host_side) {
        return q.submit_host([out, own, psize, total, deps] {
            wait_all(deps);
            for (std::size_t i = 0; i < total; ++i) out[i] = (*own)[i % psize];
        });
    }

    const sycl::device dev = o.dev;
    sycl::queue        bq  = backend_queue_for(state_of(q), dev);
    const sycl::property_list place{sycl::property::command_group::AdaptiveCpp_retarget{dev}};

    // A pattern that fits in a register is copied INTO the kernel, so there is
    // no staging allocation and no second submission.
    if (psize <= sizeof(std::uint64_t)) {
        std::uint64_t packed = 0;
        std::memcpy(&packed, pat, psize);

        return q.submit_host([bq, place, out, packed, psize, total, deps]() mutable {
            wait_all(deps);
            bq.submit(place,
                      [&](sycl::handler& h) {
                          h.parallel_for(sycl::range<1>{total}, [=](sycl::id<1> i) {
                              const auto* bytes = reinterpret_cast<const std::byte*>(&packed);
                              out[i[0]]         = bytes[i[0] % psize];
                          });
                      })
                .wait();
        });
    }

    // A wider one is staged in device memory for the kernel to replicate. The
    // staging allocation is an ordinary slabkit handle, released when the host
    // task's closure is destroyed -- after the fill has completed.
    const sycl::context& sctx  = q.get_context().context_of(dev);
    auto                 stage = std::make_shared<unique_usm_ptr<std::byte[]>>(
        os_allocate(kind::device, 0, psize, dev, sctx, {}));
    if (!*stage) throw std::bad_alloc{};

    return q.submit_host([bq, place, stage, own, out, psize, total, deps]() mutable {
        wait_all(deps);
        const std::byte* src = stage->get();

        bq.submit(place, [&](sycl::handler& h) { h.memcpy(stage->get(), own->data(), psize); })
            .wait();
        bq.submit(place,
                  [&](sycl::handler& h) {
                      h.parallel_for(sycl::range<1>{total},
                                     [=](sycl::id<1> i) { out[i[0]] = src[i[0] % psize]; });
                  })
            .wait();
    });
}

}  // namespace detail

sycl::event memset(queue& q, void* p, int value, std::size_t bytes) {
    return memset(q, p, value, bytes, {});
}

sycl::event memset(queue& q, void* p, int value, std::size_t bytes,
                   const std::vector<sycl::event>& deps) {
    if (p == nullptr || bytes == 0) return detail::no_op(q, deps);

    const detail::operand o = detail::locate(q.get_context(), p);

    // On the OpenMP backend the bytes are ordinary memory and a submission from
    // a host task would deadlock, so this is a host store.
    if (o.host_side) {
        return q.submit_host([p, value, bytes, deps] {
            detail::wait_all(deps);
            std::memset(p, value, bytes);
        });
    }

    sycl::queue               bq = detail::backend_queue_for(detail::state_of(q), o.dev);
    const sycl::property_list place{sycl::property::command_group::AdaptiveCpp_retarget{o.dev}};

    return q.submit_host([bq, place, p, value, bytes, deps]() mutable {
        detail::wait_all(deps);
        bq.submit(place, [&](sycl::handler& h) { h.memset(p, value, bytes); }).wait();
    });
}

}  // namespace naga::slabkit
