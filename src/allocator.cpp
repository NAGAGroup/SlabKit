// =============================================================================
// allocator.cpp — the allocator's route, and its release
// =============================================================================

#include "naga/slabkit/allocator.hpp"

#include <new>
#include <stdexcept>
#include <utility>

#include "detail/access.hpp"
#include "detail/arena_impl.hpp"
#include "detail/os_alloc.hpp"
#include "detail/pool_impl.hpp"
#include "naga/slabkit/context.hpp"
#include "naga/slabkit/detail/core.hpp"

namespace naga::slabkit::detail {

AllocRoute::AllocRoute(queue q, kind k, const sycl::device& dev)
    : q_{std::move(q)}, kind_{k}, from_{from::os}, dev_{dev} {}

AllocRoute::AllocRoute(queue q, kind k) : q_{std::move(q)}, kind_{k}, from_{from::os} {
    dev_ = q_.get_device();
}

AllocRoute::AllocRoute(queue q, DeviceArena arena)
    : q_{std::move(q)}
    , kind_{arena.get_kind()}
    , from_{from::arena}
    , dev_{arena.device()}
    , arena_{std::move(arena)} {}

AllocRoute::AllocRoute(queue q, DevicePool pool)
    : q_{std::move(q)}
    , kind_{pool.get_kind()}
    , from_{from::pool}
    , dev_{pool.device()}
    , pool_{std::move(pool)} {}

const queue& AllocRoute::get_queue() const noexcept { return q_; }
kind         AllocRoute::get_kind() const noexcept { return kind_; }

const sycl::device& AllocRoute::get_device() const {
    if (kind_ == kind::host)
        throw std::invalid_argument{
            "naga::slabkit::allocator: host memory belongs to a context, not a device"};
    return *dev_;
}

void* AllocRoute::raw_allocate(std::size_t bytes, std::size_t align) {
    if (bytes == 0) return nullptr;

    unique_usm_ptr<std::byte[]> h;

    switch (from_) {
        case from::arena:
            // An arena is a budget: no fallback. Running out is bad_alloc.
            h = Site::at(kind_, q_, *arena_).storage(bytes, align, align, {});
            break;
        case from::pool:
            h = Site::at(kind_, q_, *pool_).storage(bytes, align, align, {});
            if (!h) {
                // Exhausted or too fragmented: the OS route still works, and
                // the release recognises the address as not the
                // pool's.
                h = Site::at(kind_, q_, *dev_).storage(bytes, align, align, {});
            }
            break;
        case from::os:
            // The host kind ignores the device except to reach its backend's
            // context, which is exactly what this placement resolves.
            h = Site::at(kind_, q_, *dev_).storage(bytes, align, align, {});
            break;
    }

    if (!h) throw std::bad_alloc{};

    // The handle's deleter is dropped here on purpose: an allocator's release
    // decides by address, so carrying a deleter would only
    // offer a second, immediate way to free -- exactly what deallocate exists
    // to prevent.
    return h.release();
}

void AllocRoute::raw_deallocate(void* p, std::size_t bytes) noexcept {
    raw_deallocate(p, bytes, {});
}

void AllocRoute::raw_deallocate(void* p, std::size_t bytes,
                                const std::vector<sycl::event>& deps) noexcept {
    if (p == nullptr) return;

    // An arena has no per-allocation free; the arena handle this route holds is
    // what keeps the slab alive, and it outlives every allocation from it.
    if (from_ == from::arena) return;

    try {
        // Waits deps (the container's own last task), then frees here, on the
        // calling thread. Kernels the caller launched over the memory are the
        // caller's to wait for.
        sycl::event::wait(deps);

        if (from_ == from::pool) {
            const auto pool_impl = PoolAccess::impl(*pool_);
            if (pool_impl->owns(p)) {
                const std::size_t nodes =
                    align_up(bytes, pool_impl->node_size()) / pool_impl->node_size();
                pool_impl->release(p, nodes);
                return;
            }
        }
        sycl::free(p, q_.get_context().context_of(*dev_));
    } catch (...) {
        // deallocate is noexcept. Losing the memory is worse than leaking it,
        // but there is nowhere to report from here.
    }
}

bool AllocRoute::equals(const AllocRoute& other) const noexcept {
    if (kind_ != other.kind_ || from_ != other.from_) return false;
    if (!(q_ == other.q_)) return false;

    switch (from_) {
        case from::arena:
            return ArenaAccess::impl(*arena_) == ArenaAccess::impl(*other.arena_);
        case from::pool: return PoolAccess::impl(*pool_) == PoolAccess::impl(*other.pool_);
        case from::os: break;
    }
    if (kind_ == kind::host) return true;  // host memory names no device
    return dev_.has_value() && other.dev_.has_value() && *dev_ == *other.dev_;
}

}  // namespace naga::slabkit::detail
