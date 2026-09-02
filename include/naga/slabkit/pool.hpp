#pragma once

// =============================================================================
// naga/slabkit/pool.hpp — DevicePool
// =============================================================================
//
// A fixed-node free list over one USM slab, cut into node_size slots: O(1) for
// a single node, O(N/64) for a contiguous run, mutex-protected, double release
// asserted. Same handle model as DeviceArena -- copyable, copies share the
// slab, allocations pin it -- and, like the arena, it exposes no allocation
// method of its own.

#include <cstddef>
#include <memory>

#include <sycl/sycl.hpp>

#include "naga/slabkit/handles.hpp"

namespace naga::slabkit {

class DevicePool;

namespace detail {
struct PoolAccess;
}  // namespace detail

class DevicePool {
public:
    /// \throws std::invalid_argument if node_size is not a power of two.
    DevicePool(const sycl::queue& q, std::size_t bytes, std::size_t node_size = 1024,
               const sycl::property_list& props = {});

    /// \throws std::invalid_argument if node_size is not a power of two.
    DevicePool(const sycl::device& dev, const sycl::context& ctx, std::size_t bytes,
               std::size_t node_size = 1024, const sycl::property_list& props = {});

    std::size_t node_size() const;
    std::size_t slot_count() const;
    std::size_t free_count() const;

    /// Slots currently handed out. Information, not a lifetime signal.
    std::size_t outstanding() const;

    const sycl::device&  device() const;
    const sycl::context& context() const;

private:
    friend struct detail::PoolAccess;
    std::shared_ptr<detail::PoolImpl> impl_;
};

}  // namespace naga::slabkit
