#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <sycl/sycl.hpp>

#include "slabkit/detail/device_pool.hpp"
#include "slabkit/detail/device_stack.hpp"

namespace slabkit {

// =============================================================================
// DeviceUSMFromPoolDeleter — deleter for USM pointers from DeviceMemoryPool
// =============================================================================
//
// Holds a weak_ptr<DeviceMemoryPool> — does not extend pool lifetime.
// If pool is destroyed before deleter fires, memory was freed with it, skip silently.
//

template <typename T>
struct DeviceUSMFromPoolDeleter {
    size_t                           count;  // number of T elements allocated
    std::weak_ptr<DeviceMemoryPool> pool;

    void operator()(T* ptr) const noexcept {
        if (auto p = pool.lock()) {
            const size_t node_count = (count * sizeof(T) + p->node_size() - 1) / p->node_size();
            p->deallocate_array(ptr, node_count);
        }
        // if expired: pool destroyed, USM memory already freed, nothing to do
    }
};

template <typename T>
using unique_device_ptr = std::unique_ptr<T, DeviceUSMFromPoolDeleter<T>>;

// =============================================================================
// AllocatorRegistry — singleton, lazy-init per device
// =============================================================================
//
// First call to get() triggers full registry init via std::call_once.
// All vectors sized once at init, never resized.
//
// On first access for a given device, groups devices by platform,
// creates one shared_ptr<sycl::context> per platform, and lazily
// constructs the pool with an inline DeviceSlabAllocator.
//

class AllocatorRegistry {
public:
    static AllocatorRegistry& get();

    /// Returns the pool for a given SYCL device.
    /// \throws std::invalid_argument if the device was not discovered at init() time.
    std::shared_ptr<DeviceMemoryPool> pool(const sycl::device& dev);

    /// Returns the device index for a given SYCL device.
    /// \throws std::invalid_argument if the device was not discovered at init() time.
    size_t device_index(const sycl::device& dev);

    /// Returns the device for a given device index.
    /// \throws std::out_of_range if the device_index is invalid.
    const sycl::device& device_by_index(size_t device_index);

    /// Creates an in-order sycl::queue for the given device using the registry's context.
    /// In-order means each operation waits for the previous one to complete.
    /// This guarantees the queue shares the same context as the USM allocations.
    /// \throws std::invalid_argument if the device was not discovered at init() time.
    sycl::queue make_queue(const sycl::device& dev);

    /// Creates a sycl::queue with a fully caller-specified property list.
    /// The caller is responsible for all properties — no hidden additions.
    /// \throws std::invalid_argument if the device was not discovered at init() time.
    sycl::queue make_queue_with_properties(const sycl::device& dev, const sycl::property_list& props);

    /// Creates an out-of-order sycl::queue for the given device using the registry's context.
    /// Operations may execute concurrently; caller is responsible for event dependencies.
    /// \throws std::invalid_argument if the device was not discovered at init() time.
    sycl::queue make_out_of_order_queue(const sycl::device& dev);

    // Configuration struct for memory allocation
    struct Config {
        size_t block_size           = 64ull * 1024;   // arena block granularity (bytes)
        size_t node_size            = 1024;            // pool freelist node size (bytes)
        float  device_slab_fraction = 0.60f;           // fraction of global_mem_size for non-CPU devices
        float  cpu_slab_fraction    = 0.30f;           // fraction of global_mem_size for CPU devices
    };

    // Returns mutable config. Modifications have no effect after the first call to get().
    static Config& config();

private:
    AllocatorRegistry() = default;

    // Called once via registry_init_ on first get()
    // - Enumerates sycl::device::get_devices() under sycl_init_mutex_
    // - Groups devices by platform using backend + name as composite key
    // - Constructs one shared_ptr<sycl::context> per platform
    // - Populates device_list_, platform_contexts_, device_context_
    // - Computes device_slab_sizes_ based on Config and device memory
    // - Resizes pool_flags_, pools_ to device count
    void init();

    // Called once per device via pool_flags_[idx]
    // - Constructs pools_[idx] with DeviceMemoryPool and inline DeviceSlabAllocator
    void init_pool(size_t idx);

    std::once_flag registry_init_;
    std::mutex     sycl_init_mutex_;  // guards device enumeration + context construction

    // Populated at init()
    std::vector<sycl::device>                    device_list_;
    std::vector<std::shared_ptr<sycl::context>>  platform_contexts_;  // one per platform
    std::vector<std::shared_ptr<sycl::context>>  device_context_;     // indexed by device index
    std::vector<size_t>                          device_slab_sizes_;   // computed at init(), indexed by device index

    // Lazy per-device — null until first access
    std::vector<std::once_flag>                    pool_flags_;
    std::vector<std::shared_ptr<DeviceMemoryPool>> pools_;
};

// =============================================================================
// Free functions
// =============================================================================

/// Allocate `count` elements of T from the global pool for device_index.
/// Returns a shared_ptr<T> — pool reference kept alive via shared ownership.
/// Deleter holds weak_ptr<DeviceMemoryPool> — does not extend pool lifetime.
template <typename T>
std::shared_ptr<T> malloc_shared_device_ptr(size_t count, size_t device_index) {
    auto pool = AllocatorRegistry::get().pool(AllocatorRegistry::get().device_by_index(device_index));
    const size_t node_count = (count * sizeof(T) + pool->node_size() - 1) / pool->node_size();
    T* ptr    = static_cast<T*>(pool->allocate_array(node_count));
    // Deleter captures shared_ptr to pool — ensures pool outlives this allocation
    return std::shared_ptr<T>(
        ptr,
        [p = std::weak_ptr<DeviceMemoryPool>(pool), count](T* raw) noexcept {
            if (auto sp = p.lock()) {
                const size_t node_count = (count * sizeof(T) + sp->node_size() - 1) / sp->node_size();
                sp->deallocate_array(raw, node_count);
            }
        });
}

/// Allocate `count` elements of T from a user-provided pool.
/// Returns a shared_ptr<T> — pool reference kept alive via shared ownership.
/// Deleter holds weak_ptr<DeviceMemoryPool> — does not extend pool lifetime.
template <typename T>
std::shared_ptr<T> malloc_shared_device_ptr(std::shared_ptr<DeviceMemoryPool> pool, size_t count) {
    const size_t node_count = (count * sizeof(T) + pool->node_size() - 1) / pool->node_size();
    T* ptr    = static_cast<T*>(pool->allocate_array(node_count));
    // Deleter captures shared_ptr to pool — ensures pool outlives this allocation
    return std::shared_ptr<T>(
        ptr,
        [p = std::weak_ptr<DeviceMemoryPool>(pool), count](T* raw) noexcept {
            if (auto sp = p.lock()) {
                const size_t node_count = (count * sizeof(T) + sp->node_size() - 1) / sp->node_size();
                sp->deallocate_array(raw, node_count);
            }
        });
}

/// Convenience overload: looks up the pool for the given device via AllocatorRegistry.
template <typename T>
std::shared_ptr<T> malloc_shared_device_ptr(const sycl::device& dev, size_t count) {
    return malloc_shared_device_ptr<T>(AllocatorRegistry::get().pool(dev), count);
}

/// Allocate `count` elements of T from the global pool for device_index.
/// Returns a unique_device_ptr<T> with DeviceUSMFromPoolDeleter.
template <typename T>
unique_device_ptr<T> malloc_unique_device_ptr(size_t count, size_t device_index) {
    auto pool = AllocatorRegistry::get().pool(AllocatorRegistry::get().device_by_index(device_index));
    const size_t node_count = (count * sizeof(T) + pool->node_size() - 1) / pool->node_size();
    T* ptr    = static_cast<T*>(pool->allocate_array(node_count));
    return unique_device_ptr<T>{ptr,
                                DeviceUSMFromPoolDeleter<T>{count, std::weak_ptr<DeviceMemoryPool>(pool)}};
}

/// Allocate `count` elements of T from a user-provided pool.
/// Returns a unique_device_ptr<T> with DeviceUSMFromPoolDeleter.
template <typename T>
unique_device_ptr<T> malloc_unique_device_ptr(std::shared_ptr<DeviceMemoryPool> pool, size_t count) {
    const size_t node_count = (count * sizeof(T) + pool->node_size() - 1) / pool->node_size();
    T* ptr    = static_cast<T*>(pool->allocate_array(node_count));
    return unique_device_ptr<T>{ptr,
                                DeviceUSMFromPoolDeleter<T>{count, std::weak_ptr<DeviceMemoryPool>(pool)}};
}

/// Convenience overload: looks up the pool for the given device via AllocatorRegistry.
template <typename T>
unique_device_ptr<T> malloc_unique_device_ptr(const sycl::device& dev, size_t count) {
    return malloc_unique_device_ptr<T>(AllocatorRegistry::get().pool(dev), count);
}

/// Create a DeviceScratchMemory owning a region allocated from the global
/// pool for the given device. The memory is automatically returned to the
/// pool when the DeviceScratchMemory is destroyed.
inline DeviceScratchMemory make_scratch_memory(const sycl::device& dev,
                                               size_t scratch_size = 64 * 1024) {
    auto pool               = AllocatorRegistry::get().pool(dev);
    const size_t node_size  = pool->node_size();
    const size_t slot_count = (scratch_size + node_size - 1) / node_size;
    const size_t byte_count = slot_count * node_size;
    auto owner = malloc_shared_device_ptr<std::byte>(pool, byte_count);
    return DeviceScratchMemory{std::move(owner), byte_count};
}

/// Create a DeviceScratchMemory owning a region allocated from a
/// user-provided pool. The memory is automatically returned to the
/// pool when the DeviceScratchMemory is destroyed.
inline DeviceScratchMemory make_scratch_memory(std::shared_ptr<DeviceMemoryPool> pool,
                                               size_t scratch_size = 64 * 1024) {
    const size_t node_size  = pool->node_size();
    const size_t slot_count = (scratch_size + node_size - 1) / node_size;
    const size_t byte_count = slot_count * node_size;
    auto owner = malloc_shared_device_ptr<std::byte>(pool, byte_count);
    return DeviceScratchMemory{std::move(owner), byte_count};
}

// =============================================================================
// Queue factory — always use these to construct queues, never sycl::queue{}
// directly. Guarantees the queue uses the same context as the allocator registry,
// making USM pointers valid for the returned queue.
//
// make_queue                → in-order  (default — each op waits for previous)
// make_out_of_order_queue   → out-of-order (concurrent ops, caller manages event deps)
// make_queue_with_properties → fully caller-specified property list, no hidden additions
// =============================================================================

/// Creates an in-order context-compatible queue for the given device.
inline sycl::queue make_queue(const sycl::device& dev) {
    return AllocatorRegistry::get().make_queue(dev);
}

/// Creates an out-of-order context-compatible queue for the given device.
/// Operations may execute concurrently; caller is responsible for event dependencies.
inline sycl::queue make_out_of_order_queue(const sycl::device& dev) {
    return AllocatorRegistry::get().make_out_of_order_queue(dev);
}

/// Creates a context-compatible queue with a fully caller-specified property list.
/// The caller owns all properties — no in_order or other properties are added.
inline sycl::queue make_queue_with_properties(const sycl::device& dev,
                                               const sycl::property_list& props) {
    return AllocatorRegistry::get().make_queue_with_properties(dev, props);
}

// =============================================================================
// Implementation
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// AllocatorRegistry implementations
// ─────────────────────────────────────────────────────────────────────────────

inline AllocatorRegistry& AllocatorRegistry::get() {
    static AllocatorRegistry instance;
    std::call_once(instance.registry_init_, [&] { instance.init(); });
    return instance;
}

inline void AllocatorRegistry::init_pool(size_t idx) {
    std::lock_guard<std::mutex> lk{sycl_init_mutex_};
    const Config& cfg      = config();
    const size_t  slab_sz  = device_slab_sizes_[idx];
    const size_t  slot_cnt = slab_sz / cfg.node_size;

    // Allocate the device slab for this pool
    void* slab_ptr = sycl::malloc_device(slab_sz,
                                          device_list_[idx],
                                          *device_context_[idx]);
    if (!slab_ptr)
        throw std::bad_alloc{};

    // Wrap slab in a shared_ptr with a SYCL-free deleter so the pool
    // can be shared without the registry needing to track the raw pointer.
    auto ctx_copy = device_context_[idx];
    auto slab_owner = std::shared_ptr<void>(
        slab_ptr,
        [ctx_copy](void* p) noexcept { sycl::free(p, *ctx_copy); });

    // Construct pool — does NOT own slab_ptr, so we store slab_owner alongside.
    // Use a custom deleter on the DeviceMemoryPool shared_ptr that keeps
    // slab_owner alive until the pool itself is destroyed.
    pools_[idx] = std::shared_ptr<DeviceMemoryPool>(
        new DeviceMemoryPool(slab_ptr, cfg.node_size, slot_cnt),
        [slab_owner](DeviceMemoryPool* p) mutable noexcept {
            delete p;
            // slab_owner goes out of scope here → sycl::free called
        });
}

inline void AllocatorRegistry::init() {
    std::lock_guard<std::mutex> lk{sycl_init_mutex_};

    // Enumerate devices
    device_list_ = sycl::device::get_devices();

    // Group devices by platform
    // Use backend + name as composite key (do NOT use .get() on platform)
    std::vector<std::vector<sycl::device>> devices_by_platform;
    std::vector<std::pair<sycl::backend, std::string>> platform_keys;

    for (const auto& dev : device_list_) {
        const sycl::platform plat = dev.get_platform();
        const auto backend        = plat.get_backend();
        const auto plat_name      = plat.get_info<sycl::info::platform::name>();

        // Find if we've already seen this platform (by backend + name)
        auto it = std::find_if(platform_keys.begin(), platform_keys.end(),
                               [backend, &plat_name](const auto& key) {
                                   return key.first == backend && key.second == plat_name;
                               });

        if (it == platform_keys.end()) {
            // New platform
            platform_keys.push_back({backend, plat_name});
            devices_by_platform.push_back({dev});
        } else {
            // Existing platform
            const size_t idx = std::distance(platform_keys.begin(), it);
            devices_by_platform[idx].push_back(dev);
        }
    }

    // Create one context per platform
    for (const auto& dev_group : devices_by_platform) {
        platform_contexts_.push_back(std::make_shared<sycl::context>(dev_group));
    }

    // Build device_context_ mapping
    device_context_.resize(device_list_.size());
    device_slab_sizes_.resize(device_list_.size());
    
    const Config& cfg = config();
    
    for (size_t dev_idx = 0; dev_idx < device_list_.size(); ++dev_idx) {
        const sycl::device& dev       = device_list_[dev_idx];
        const sycl::platform dev_plat = dev.get_platform();
        const auto backend            = dev_plat.get_backend();
        const auto plat_name          = dev_plat.get_info<sycl::info::platform::name>();

        // Find the matching platform context (by backend + name)
        auto it = std::find_if(platform_keys.begin(), platform_keys.end(),
                               [backend, &plat_name](const auto& key) {
                                   return key.first == backend && key.second == plat_name;
                               });
        assert(it != platform_keys.end());
        const size_t plat_idx      = std::distance(platform_keys.begin(), it);
        device_context_[dev_idx] = platform_contexts_[plat_idx];

        // Compute slab size for this device
        const size_t total_mem = dev.get_info<sycl::info::device::global_mem_size>();
        const float  fraction  = dev.is_cpu() ? cfg.cpu_slab_fraction : cfg.device_slab_fraction;
        const size_t computed  = static_cast<size_t>(total_mem * fraction);
        const size_t raw = std::max(computed, 2 * cfg.block_size);
        // Round down to nearest block_size multiple (required by DeviceSlabAllocator assert)
        device_slab_sizes_[dev_idx] = (raw / cfg.block_size) * cfg.block_size;
    }

    // Resize once_flag and shared_ptr vectors to device count
    pool_flags_ = std::vector<std::once_flag>(device_list_.size());
    pools_.resize(device_list_.size());
}

inline std::shared_ptr<DeviceMemoryPool> AllocatorRegistry::pool(const sycl::device& dev) {
    const auto it = std::find(device_list_.begin(), device_list_.end(), dev);
    if (it == device_list_.end())
        throw std::invalid_argument(
            "slabkit: device not found in AllocatorRegistry — "
            "device must be present at init() time");
    const size_t idx = static_cast<size_t>(std::distance(device_list_.begin(), it));
    std::call_once(pool_flags_[idx], [this, idx] { init_pool(idx); });
    return pools_[idx];
}

inline size_t AllocatorRegistry::device_index(const sycl::device& dev) {
    const auto it = std::find(device_list_.begin(), device_list_.end(), dev);
    if (it == device_list_.end())
        throw std::invalid_argument(
            "slabkit: device not found in AllocatorRegistry — "
            "device must be present at init() time");
    return static_cast<size_t>(std::distance(device_list_.begin(), it));
}

inline const sycl::device& AllocatorRegistry::device_by_index(size_t device_index) {
    if (device_index >= device_list_.size())
        throw std::out_of_range(
            "slabkit: device_index out of range");
    return device_list_[device_index];
}

inline sycl::queue AllocatorRegistry::make_queue(const sycl::device& dev) {
    const size_t idx = device_index(dev);
    return sycl::queue{*device_context_[idx], dev, sycl::property::queue::in_order{}};
}

inline sycl::queue AllocatorRegistry::make_out_of_order_queue(const sycl::device& dev) {
    const size_t idx = device_index(dev);
    return sycl::queue{*device_context_[idx], dev};  // SYCL default is out-of-order
}

inline sycl::queue AllocatorRegistry::make_queue_with_properties(const sycl::device& dev,
                                                                   const sycl::property_list& props) {
    const size_t idx = device_index(dev);
    return sycl::queue{*device_context_[idx], dev, props};
}

inline AllocatorRegistry::Config& AllocatorRegistry::config() {
    static Config instance;
    return instance;
}

}  // namespace slabkit
