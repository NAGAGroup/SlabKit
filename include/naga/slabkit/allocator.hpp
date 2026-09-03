#pragma once

// =============================================================================
// naga/slabkit/allocator.hpp — one allocator over all kinds and routes
// =============================================================================
//
// A standard Allocator whose memory is USM. `allocate` is the storage family
// with std::bad_alloc instead of null; the pool route tries the pool and falls
// back to the OS on exhaustion, while the arena route never falls back --
// an arena is a budget, and a budget that silently grows is not one.
//
// DEFERRED RELEASE is the whole point. `deallocate` never frees on the calling
// thread: it posts a host task holding a copy of the queue handle, which waits
// the stream and only then releases. So a container may reallocate, shrink or
// die while a kernel still holds the old pointer, and the memory outlives the
// stream's use of it. The address range is the only record kept -- the release
// decides by it whether the bytes belong to the pool's slab or to the runtime.
//
// std containers over the SHARED and HOST kinds work unchanged. Over the DEVICE
// kind they are a usage error rather than a feature: element access would be a
// host dereference of device memory. The device kind is for usm_vector and for
// handle users.

#include <cstddef>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

#include <sycl/sycl.hpp>

#include "naga/slabkit/arena.hpp"
#include "naga/slabkit/kinds.hpp"
#include "naga/slabkit/pool.hpp"
#include "naga/slabkit/queue.hpp"

namespace naga::slabkit {

namespace detail {

/// Where an allocator's bytes come from and how they go back. Compiled, and
/// shared between rebinds: rebinding keeps state, which is what makes
/// allocator<T> and allocator<U> compare equal when they should.
class AllocRoute {
public:
    AllocRoute(queue q, kind k, const sycl::device& dev);
    AllocRoute(queue q, kind k);
    AllocRoute(queue q, DeviceArena arena);
    AllocRoute(queue q, DevicePool pool);

    /// \throws std::bad_alloc
    void* raw_allocate(std::size_t bytes, std::size_t align);

    /// Posted, never immediate. Never throws.
    void raw_deallocate(void* p, std::size_t bytes) noexcept;

    const queue&        get_queue() const noexcept;
    kind                get_kind() const noexcept;
    const sycl::device& get_device() const;  // throws for the host kind

    bool equals(const AllocRoute& other) const noexcept;

private:
    enum class from { os, arena, pool };

    queue                       q_;
    kind                        kind_;
    from                        from_;
    std::optional<sycl::device> dev_;
    std::optional<DeviceArena>  arena_;
    std::optional<DevicePool>   pool_;
};

}  // namespace detail

template <class T>
class allocator {
public:
    using value_type                             = T;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap            = std::true_type;
    using is_always_equal                        = std::false_type;

    /// OS route, on that device.
    allocator(queue q, kind k, const sycl::device& dev)
        : route_{std::make_shared<detail::AllocRoute>(std::move(q), k, dev)} {}

    /// OS route, the stream's default device. The host kind uses this form.
    allocator(queue q, kind k) : route_{std::make_shared<detail::AllocRoute>(std::move(q), k)} {}

    /// Arena route: the object's kind and device.
    allocator(queue q, DeviceArena arena)
        : route_{std::make_shared<detail::AllocRoute>(std::move(q), std::move(arena))} {}

    /// Pool route: the object's kind and device.
    allocator(queue q, DevicePool pool)
        : route_{std::make_shared<detail::AllocRoute>(std::move(q), std::move(pool))} {}

    /// Rebind keeps state.
    template <class U>
    allocator(const allocator<U>& other) : route_{other.route_} {}

    T* allocate(std::size_t n) {
        return static_cast<T*>(route_->raw_allocate(n * sizeof(T), alignof(T)));
    }

    void deallocate(T* p, std::size_t n) noexcept { route_->raw_deallocate(p, n * sizeof(T)); }

    queue        get_queue() const { return route_->get_queue(); }
    kind         get_kind() const { return route_->get_kind(); }
    sycl::device get_device() const { return route_->get_device(); }

    template <class U>
    bool operator==(const allocator<U>& other) const {
        return route_->equals(*other.route_);
    }

private:
    template <class U>
    friend class allocator;

    std::shared_ptr<detail::AllocRoute> route_;
};

}  // namespace naga::slabkit
