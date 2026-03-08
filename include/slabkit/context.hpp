#pragma once

// =============================================================================
// slabkit/context.hpp — SlabContext, SlabContextBuilder
// =============================================================================
//
// SlabContext: manages SYCL contexts, devices, and device-specific memory pools
//
// SlabContextBuilder: builder pattern for constructing SlabContext with
//   validation of platform coverage and lazy pool initialization
//

#include <algorithm>
#include <cassert>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "slabkit/error.hpp"
#include "slabkit/detail/device_pool.hpp"

namespace slabkit {

// =============================================================================
// SlabContext — SYCL platform contexts, devices, and memory pools
// =============================================================================

struct SlabContext : std::enable_shared_from_this<SlabContext> {
    // Public data members (owned and managed)
    std::vector<sycl::context>                     platform_contexts;
    std::vector<sycl::device>                      devices;
    std::vector<size_t>                            device_to_platform;
    std::vector<size_t>                            device_slab_sizes;
    std::vector<std::unique_ptr<DeviceMemoryPool>> pools;

    // Private config and synchronization
    size_t              node_size_ = 1024;
    mutable std::mutex  pool_mutex_;

    // =========================================================================
    // Device lookup
    // =========================================================================

    /// Finds the index of a device in the devices vector.
    /// Uses exact device identity (operator==) for matching.
    /// \throws std::invalid_argument if device not found.
    size_t device_index(const sycl::device& dev) const {
        for (size_t i = 0; i < devices.size(); ++i) {
            if (devices[i] == dev)
                return i;
        }
        throw std::invalid_argument("slabkit: device not found in SlabContext");
    }

    // =========================================================================
    // Pool access — lazy initialization
    // =========================================================================

    /// Returns a reference to the pool for the given device index.
    /// Lazily initializes the pool on first call.
    /// \throws std::out_of_range if index is out of bounds.
    DeviceMemoryPool& pool_at(size_t idx) {
        if (idx >= pools.size())
            throw std::out_of_range("slabkit: device index out of range");

        {
            std::lock_guard<std::mutex> lk(pool_mutex_);
            if (!pools[idx]) {
                size_t plat_idx = device_to_platform[idx];
                pools[idx] = std::make_unique<DeviceMemoryPool>(
                    devices[idx],
                    &platform_contexts[plat_idx],
                    std::weak_ptr<void>(shared_from_this()),
                    device_slab_sizes[idx],
                    node_size_);
            }
        }

        return *pools[idx];
    }

    /// Const version of pool_at — lazy init is logically non-mutating (idempotent).
    const DeviceMemoryPool& pool_at(size_t idx) const {
        return const_cast<SlabContext*>(this)->pool_at(idx);
    }

    /// Returns the pool for a given device.
    /// Lazily initializes on first call.
    DeviceMemoryPool& pool_for(const sycl::device& dev) {
        return pool_at(device_index(dev));
    }

    /// Const version of pool_for.
    const DeviceMemoryPool& pool_for(const sycl::device& dev) const {
        return pool_at(device_index(dev));
    }

    // =========================================================================
    // Context lookup
    // =========================================================================

    /// Returns the SYCL context for a given device index.
    /// \throws std::out_of_range if index is out of bounds.
    const sycl::context& context_for(size_t device_index_val) const {
        if (device_index_val >= devices.size()) {
            throw std::out_of_range("slabkit: device index out of range");
        }
        size_t plat_idx = device_to_platform[device_index_val];
        return platform_contexts[plat_idx];
    }

    /// Returns the SYCL context for a given device.
    /// \throws std::invalid_argument if device not found.
    const sycl::context& context_for(const sycl::device& dev) const {
        return context_for(device_index(dev));
    }

    // =========================================================================
    // Queue factories
    // =========================================================================

    /// Creates an in-order queue for the given device.
    sycl::queue make_in_order_queue(const sycl::device& dev) const {
        const auto& ctx = context_for(dev);
        return sycl::queue(ctx, dev, sycl::property::queue::in_order());
    }

    /// Creates an out-of-order queue for the given device.
    sycl::queue make_out_of_order_queue(const sycl::device& dev) const {
        const auto& ctx = context_for(dev);
        return sycl::queue(ctx, dev);
    }

    /// Creates a queue with a fully custom property list.
    sycl::queue make_queue_with_props(const sycl::device& dev,
                                       const sycl::property_list& props) const {
        const auto& ctx = context_for(dev);
        return sycl::queue(ctx, dev, props);
    }

    // =========================================================================
    // Move semantics (non-copyable)
    // =========================================================================

    SlabContext(const SlabContext&)            = delete;
    SlabContext& operator=(const SlabContext&) = delete;

    SlabContext(SlabContext&& other) noexcept
        : std::enable_shared_from_this<SlabContext>(std::move(other))
        , platform_contexts(std::move(other.platform_contexts))
        , devices(std::move(other.devices))
        , device_to_platform(std::move(other.device_to_platform))
        , device_slab_sizes(std::move(other.device_slab_sizes))
        , pools(std::move(other.pools))
        , node_size_(other.node_size_)
    {}

    SlabContext& operator=(SlabContext&&) = delete;

    SlabContext() = default;
};

// =============================================================================
// SlabContextBuilder — builder pattern with validation
// =============================================================================

class SlabContextBuilder {
public:
    /// Set the list of SYCL contexts (platform contexts).
    SlabContextBuilder& contexts(std::vector<sycl::context> ctxs) {
        contexts_ = std::move(ctxs);
        return *this;
    }

    /// Set the list of SYCL devices.
    SlabContextBuilder& devices(std::vector<sycl::device> devs) {
        devices_ = std::move(devs);
        return *this;
    }

    /// Set the node size (granularity of pool allocations).
    SlabContextBuilder& node_size(size_t ns) {
        node_size_ = ns;
        return *this;
    }

    /// Set uniform slab size for all devices.
    SlabContextBuilder& slab_size_per_device(size_t sz) {
        slab_size_ = sz;
        return *this;
    }

    /// Build and return a shared_ptr<SlabContext>.
    /// SlabContext must always be heap-allocated via shared_ptr so that
    /// enable_shared_from_this works for pool lazy init.
    std::shared_ptr<SlabContext> build() {
        if (devices_.empty()) {
            throw std::invalid_argument("slabkit: no devices provided");
        }

        if (contexts_.empty()) {
            throw std::invalid_argument("slabkit: no contexts provided");
        }

        // Pass 1: Check for duplicate platforms in contexts_
        std::vector<std::pair<sycl::backend, std::string>> seen_platforms;
        for (const auto& ctx : contexts_) {
            auto ctx_devices = ctx.get_devices();
            for (const auto& d : ctx_devices) {
                auto backend = d.get_platform().get_backend();
                auto name    = d.get_platform().get_info<sycl::info::platform::name>();

                for (const auto& [seen_backend, seen_name] : seen_platforms) {
                    if (backend == seen_backend && name == seen_name) {
                        throw std::invalid_argument(
                            "slabkit: duplicate platform in contexts");
                    }
                }
                seen_platforms.emplace_back(backend, name);
            }
        }

        // Pass 2: Check that each device is covered by some context
        SlabContext result;
        result.devices                = devices_;
        result.platform_contexts      = contexts_;
        result.device_to_platform.resize(devices_.size());
        result.device_slab_sizes.resize(devices_.size(),
                                         slab_size_ > 0 ? slab_size_ : (64 * 1024 * 1024));
        result.pools.resize(devices_.size());
        result.node_size_ = node_size_;

        for (size_t i = 0; i < devices_.size(); ++i) {
            auto dev_backend = devices_[i].get_platform().get_backend();
            auto dev_name =
                devices_[i].get_platform().get_info<sycl::info::platform::name>();

            size_t matching_ctx_idx = std::numeric_limits<size_t>::max();

            // Find which context covers this device
            for (size_t ctx_idx = 0; ctx_idx < contexts_.size(); ++ctx_idx) {
                auto ctx_devices = contexts_[ctx_idx].get_devices();
                for (const auto& ctx_dev : ctx_devices) {
                    auto ctx_backend = ctx_dev.get_platform().get_backend();
                    auto ctx_name =
                        ctx_dev.get_platform().get_info<sycl::info::platform::name>();

                    if (dev_backend == ctx_backend && dev_name == ctx_name) {
                        matching_ctx_idx = ctx_idx;
                        break;
                    }
                }
                if (matching_ctx_idx != std::numeric_limits<size_t>::max()) {
                    break;
                }
            }

            if (matching_ctx_idx == std::numeric_limits<size_t>::max()) {
                throw std::invalid_argument(
                    "slabkit: device not covered by any provided context");
            }

            result.device_to_platform[i] = matching_ctx_idx;
        }

        return std::shared_ptr<SlabContext>(new SlabContext(std::move(result)));
    }

private:
    std::vector<sycl::context> contexts_;
    std::vector<sycl::device>  devices_;
    size_t                     node_size_  = 1024;
    size_t                     slab_size_  = 0;  // 0 = use default (64MB)
};

}  // namespace slabkit
