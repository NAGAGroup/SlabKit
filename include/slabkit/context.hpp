#pragma once

// =============================================================================
// slabkit/context.hpp — Context, ContextBuilder
// =============================================================================
//
// Context: manages SYCL contexts, devices, and device-specific memory pools
//
// ContextBuilder: builder pattern for constructing Context.
//   Accepts a list of sycl::device objects and auto-creates one sycl::context
//   per platform group. No explicit sycl::context objects are required.
//

#include <algorithm>
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
// Context — SYCL platform contexts, devices, and memory pools
// =============================================================================

struct Context : std::enable_shared_from_this<Context> {
    // Public data members (owned and managed)
    std::vector<sycl::context>                     platform_contexts;
    std::vector<sycl::device>                      devices;
    std::vector<size_t>                            device_to_platform;
    std::vector<size_t>                            device_slab_sizes;
    mutable std::vector<std::unique_ptr<DeviceMemoryPool>> pools;

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
        throw std::invalid_argument("slabkit: device not found in Context");
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

        if (!pools[idx]) {
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
    /// Delegates to mutable overload via const_cast: safe because pools and pool_mutex_
    /// are mutable, so no const object is modified.
    const DeviceMemoryPool& pool_at(size_t idx) const {
        return const_cast<Context*>(this)->pool_at(idx);
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
    [[nodiscard]] sycl::queue make_in_order_queue(const sycl::device& dev) const {
        const auto& ctx = context_for(dev);
        return sycl::queue(ctx, dev, sycl::property::queue::in_order());
    }

    /// Creates an out-of-order queue for the given device.
    [[nodiscard]] sycl::queue make_out_of_order_queue(const sycl::device& dev) const {
        const auto& ctx = context_for(dev);
        return sycl::queue(ctx, dev);
    }

    /// Creates a queue with a fully custom property list.
    [[nodiscard]] sycl::queue make_queue_with_props(const sycl::device& dev,
                                       const sycl::property_list& props) const {
        const auto& ctx = context_for(dev);
        return sycl::queue(ctx, dev, props);
    }

    // =========================================================================
    // Move semantics (non-copyable, non-movable)
    // =========================================================================
    //
    // Context contains std::mutex which is not movable. Context is always
    // managed via shared_ptr and never moved in practice. Both move constructor
    // and move assignment are explicitly deleted to make this invariant clear.
    //

    Context(const Context&)               = delete;
    Context& operator=(const Context&)    = delete;
    Context(Context&&)                    = delete;
    Context& operator=(Context&&)         = delete;

    Context() = default;
};

// =============================================================================
// ContextBuilder — builder pattern with validation
// =============================================================================

class ContextBuilder {
public:
    /// Set the list of SYCL devices.
    ContextBuilder& devices(std::vector<sycl::device> devs) {
        devices_ = std::move(devs);
        return *this;
    }

    /// Set the node size (granularity of pool allocations).
    ContextBuilder& node_size(size_t ns) {
        node_size_ = ns;
        return *this;
    }

    /// Set uniform slab size for all devices.
    ContextBuilder& slab_size_per_device(size_t sz) {
        slab_size_ = sz;
        return *this;
    }

    /// Build and return a shared_ptr<Context>.
    /// Automatically creates one sycl::context per platform group from the
    /// provided devices. Context must always be heap-allocated via
    /// shared_ptr so that enable_shared_from_this works for pool lazy init.
    std::shared_ptr<Context> build() {
        if (devices_.empty()) {
            throw std::invalid_argument("slabkit: no devices provided");
        }

        // Group devices by (backend, platform_name) — one sycl::context per group.
        std::vector<std::vector<sycl::device>>             devices_by_platform;
        std::vector<std::pair<sycl::backend, std::string>> platform_keys;

        for (const auto& dev : devices_) {
            const sycl::platform plat    = dev.get_platform();
            const auto           backend = plat.get_backend();
            const auto           name    = plat.get_info<sycl::info::platform::name>();

            auto it = std::find_if(platform_keys.begin(), platform_keys.end(),
                                   [backend, &name](const auto& key) {
                                       return key.first == backend && key.second == name;
                                   });
            if (it == platform_keys.end()) {
                platform_keys.push_back({backend, name});
                devices_by_platform.push_back({dev});
            } else {
                devices_by_platform[std::distance(platform_keys.begin(), it)].push_back(dev);
            }
        }

        auto result = std::shared_ptr<Context>(new Context());
        result->devices = devices_;
        result->device_to_platform.resize(devices_.size());
        result->device_slab_sizes.resize(devices_.size(),
                                        slab_size_ > 0 ? slab_size_ : (64 * 1024 * 1024));
        result->pools.resize(devices_.size());
        result->node_size_ = node_size_;

        for (const auto& dev_group : devices_by_platform)
            result->platform_contexts.push_back(sycl::context(dev_group));

        for (size_t i = 0; i < devices_.size(); ++i) {
            const sycl::platform plat    = devices_[i].get_platform();
            const auto           backend = plat.get_backend();
            const auto           name    = plat.get_info<sycl::info::platform::name>();

            auto it = std::find_if(platform_keys.begin(), platform_keys.end(),
                                   [backend, &name](const auto& key) {
                                       return key.first == backend && key.second == name;
                                   });
            result->device_to_platform[i] = std::distance(platform_keys.begin(), it);
        }

        return result;
    }

private:
    std::vector<sycl::device>  devices_;
    size_t                     node_size_  = 1024;
    size_t                     slab_size_  = 0;  // 0 = use default (64MB)
};

}  // namespace slabkit
