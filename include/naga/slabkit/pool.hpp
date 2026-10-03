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
//
// The KIND is fixed at construction, as it is for an arena, and the allocation
// name must match it.

#include <cstddef>
#include <memory>

#include <sycl/sycl.hpp>

#include "naga/slabkit/handles.hpp"
#include "naga/slabkit/kinds.hpp"

namespace naga::slabkit {

class queue;
class DevicePool;

namespace detail {
struct PoolAccess;
}  // namespace detail

class DevicePool {
public:
    /// PRIMARY. \throws std::invalid_argument if node_size is not a power of two.
    DevicePool(const queue& q, const sycl::device& dev, kind k, std::size_t bytes,
               std::size_t node_size = 1024, const sycl::property_list& props = {});

    /// The stream's default device.
    DevicePool(const queue& q, kind k, std::size_t bytes, std::size_t node_size = 1024,
               const sycl::property_list& props = {});

    /// Interop: SYCL's own queue.
    DevicePool(const sycl::queue& q, kind k, std::size_t bytes, std::size_t node_size = 1024,
               const sycl::property_list& props = {});

    /// Interop: that device, freed with that context.
    DevicePool(const sycl::device& dev, const sycl::context& ctx, kind k, std::size_t bytes,
               std::size_t node_size = 1024, const sycl::property_list& props = {});

    kind        get_kind() const;
    std::size_t node_size() const;
    std::size_t slot_count() const;
    std::size_t free_count() const;

    /// Slots currently handed out. Information, not a lifetime signal.
    std::size_t outstanding() const;

    /// Allocations made through this pool's route that the pool could not serve (it was full or too fragmented), which
    /// went to the OS instead. CUMULATIVE: the count of such allocations ever made.
    std::size_t fallback_allocations() const;
    /// The bytes of those OS allocations that are still LIVE (a release subtracts them).
    std::size_t fallback_bytes_in_use() const;

    const sycl::device&  device() const;
    const sycl::context& context() const;

private:
    friend struct detail::PoolAccess;
    std::shared_ptr<detail::PoolImpl> impl_;
};

}  // namespace naga::slabkit
