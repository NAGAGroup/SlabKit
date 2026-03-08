#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <sycl/sycl.hpp>

#include "slabkit/context.hpp"
#include "slabkit/detail/device_pool.hpp"
#include "slabkit/detail/device_stack.hpp"
#include "slabkit/error.hpp"

namespace slabkit {

// =============================================================================
// DeviceUSMPoolDeleter — non-templated deleter for USM pointers from
// SlabContext pools
// =============================================================================
//
// Holds a weak_ptr<SlabContext> and device index.
// Takes void* in operator() — compatible with unique_ptr<T,
// DeviceUSMPoolDeleter> because T* is implicitly convertible to void*. If
// SlabContext is destroyed before deleter fires, memory was freed with it, skip
// silently.
//

struct DeviceUSMPoolDeleter {
    std::weak_ptr<SlabContext> ctx;
    size_t device_index;
    size_t node_count; // pre-computed: ceil(count*sizeof(T) / node_size)

    void operator()(void *ptr) const noexcept {
        if (auto c = ctx.lock())
            c->pool_at(device_index).deallocate_array(ptr, node_count);
        // if expired: SlabContext destroyed, USM slab already freed — no-op is
        // correct
    }
};

template <typename T>
using unique_device_ptr = std::unique_ptr<T, DeviceUSMPoolDeleter>;

// =============================================================================
// AllocatorRegistry — singleton, lazy-init per device, owns
// shared_ptr<SlabContext>
// =============================================================================
//
// First call to get() triggers full registry init via std::call_once.
// All vectors sized once at init, never resized.
//
// The registry owns a shared_ptr<SlabContext> built during init(). This context
// groups devices by platform, creates one sycl::context per platform, and
// lazily constructs pools with DeviceMemoryPool on demand.
//
// The registry's SlabContext is available via context() static method.
//

class AllocatorRegistry {
public:
    static AllocatorRegistry &get();

    /// Returns the registry's shared_ptr<SlabContext>.
    /// All pools, devices, and contexts are managed by this SlabContext.
    static std::shared_ptr<SlabContext> context();

    /// Returns a reference to the pool for a given SYCL device (lazy init).
    /// \throws std::invalid_argument if the device was not discovered at init()
    /// time.
    DeviceMemoryPool &pool(const sycl::device &dev);

    /// Returns the device index for a given SYCL device.
    /// \throws std::invalid_argument if the device was not discovered at init()
    /// time.
    size_t device_index(const sycl::device &dev);

    /// Returns the device for a given device index.
    /// \throws std::out_of_range if the device_index is invalid.
    const sycl::device &device_by_index(size_t device_index);

    /// Creates an in-order sycl::queue for the given device using the registry's
    /// context. This guarantees the queue shares the same context as the USM
    /// allocations.
    /// \throws std::invalid_argument if the device was not discovered at init()
    /// time.
    sycl::queue make_in_order_queue(const sycl::device &dev);

    /// Creates an out-of-order sycl::queue for the given device using the
    /// registry's context.
    /// \throws std::invalid_argument if the device was not discovered at init()
    /// time.
    sycl::queue make_out_of_order_queue(const sycl::device &dev);

    /// Creates a sycl::queue with a fully caller-specified property list.
    /// \throws std::invalid_argument if the device was not discovered at init()
    /// time.
    sycl::queue make_queue_with_props(const sycl::device &dev,
                                      const sycl::property_list &props);

    // Configuration struct for memory allocation
    struct Config {
        size_t block_size = 64ull * 1024; // arena block granularity (bytes)
        size_t node_size = 1024;          // pool freelist node size (bytes)
        float device_slab_fraction =
            0.60f; // fraction of global_mem_size for non-CPU devices
        float cpu_slab_fraction =
            0.30f; // fraction of global_mem_size for CPU devices
    };

    // Returns mutable config. Modifications have no effect after the first call
    // to get().
    static Config &config();

    /// One-shot configuration. Must be called before the first context()/pool() access.
    /// \throws std::logic_error if the registry is already live.
    static void configure(Config cfg);

#ifdef SLABKIT_TESTING
    /// Reset registry state — only available in test builds.
    /// Allows re-initialization between test cases.
    static void reset_for_testing();
#endif

private:
    AllocatorRegistry() = default;

    void init();

    std::once_flag registry_init_;
    std::mutex sycl_init_mutex_;
    std::shared_ptr<SlabContext> slab_ctx_;
    bool configured_ = false;
};

// =============================================================================
// Free functions
// =============================================================================

/// Allocate count elements of T from a SlabContext pool.
/// Returns unique_device_ptr<T> with DeviceUSMPoolDeleter holding
/// weak_ptr<SlabContext>.
template <typename T>
unique_device_ptr<T> malloc_device(size_t count,
                                   std::shared_ptr<SlabContext> ctx,
                                   size_t device_index) {
    auto &pool = ctx->pool_at(device_index);
    const size_t node_count =
        (count * sizeof(T) + pool.node_size() - 1) / pool.node_size();
    T *ptr = static_cast<T *>(pool.allocate_array(node_count));
    return unique_device_ptr<T>{
        ptr, DeviceUSMPoolDeleter{std::weak_ptr<SlabContext>(ctx), device_index,
                                  node_count}};
}

/// Registry-backed: allocate count elements of T from the registry's global
/// context.
template <typename T>
unique_device_ptr<T> malloc_device(size_t count, const sycl::device &dev) {
    auto ctx = AllocatorRegistry::context();
    const size_t idx = ctx->device_index(dev);
    return malloc_device<T>(count, ctx, idx);
}

/// Create a DeviceScratchMemory owning a region allocated from a user-provided
/// pool. Registry-backed scratch factory.
inline DeviceScratchMemory make_scratch(const sycl::device &dev,
                                        std::size_t scratch_size = 64 * 1024) {
    auto &pool = AllocatorRegistry::context()->pool_at(
        AllocatorRegistry::context()->device_index(dev));
    return DeviceScratchMemory{&pool, scratch_size};
}

// Note: registry-free make_scratch(ctx, device_index, size) is defined in
// scratch.hpp.

// =============================================================================
// Queue factory free functions — use these (ensures correct context)
// =============================================================================

/// In-order queue for the given device (registry context).
inline sycl::queue make_in_order_queue(const sycl::device &dev) {
    return AllocatorRegistry::get().make_in_order_queue(dev);
}

/// Out-of-order queue for the given device (registry context).
inline sycl::queue make_out_of_order_queue(const sycl::device &dev) {
    return AllocatorRegistry::get().make_out_of_order_queue(dev);
}

/// Queue with fully caller-specified property list (registry context).
inline sycl::queue make_queue_with_props(const sycl::device &dev,
                                         const sycl::property_list &props) {
    return AllocatorRegistry::get().make_queue_with_props(dev, props);
}

// =============================================================================
// Implementation
// =============================================================================

inline AllocatorRegistry &AllocatorRegistry::get() {
    // Force AdaptiveCpp's allocation_tracker::amap to construct before instance,
    // so it destructs AFTER instance. Prevents SEGFAULT on exit via sycl::free.
    [[maybe_unused]] static const bool amap_anchor_ = [] {
        auto devs = sycl::device::get_devices();
        if (!devs.empty()) {
            sycl::context ctx{devs[0]};
            if (void *p = sycl::malloc_device(1, devs[0], ctx))
                sycl::free(p, ctx);
        }
        return true;
    }();
    static AllocatorRegistry instance;
    std::call_once(instance.registry_init_, [&] { instance.init(); });
    return instance;
}

inline std::shared_ptr<SlabContext> AllocatorRegistry::context() {
    return get().slab_ctx_;
}

inline void AllocatorRegistry::init() {
    std::lock_guard<std::mutex> lk{sycl_init_mutex_};

    auto device_list = sycl::device::get_devices();

    // Group devices by platform — one sycl::context per platform group.
    // NOTE: We construct SlabContext directly here rather than via SlabContextBuilder
    // because the builder's duplicate-platform check uses ctx.get_devices(), which
    // returns all platform devices (not just the subset passed to the context ctor).
    // That causes false duplicate-platform errors when init() creates multiple contexts
    // for a multi-platform system. SlabContextBuilder is validated on user-supplied
    // contexts; init() constructs internally and guarantees one context per platform.
    std::vector<std::vector<sycl::device>> devices_by_platform;
    std::vector<std::pair<sycl::backend, std::string>> platform_keys;

    for (const auto& dev : device_list) {
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

    slab_ctx_ = std::make_shared<SlabContext>();
    slab_ctx_->devices = device_list;

    for (const auto& dev_group : devices_by_platform)
        slab_ctx_->platform_contexts.push_back(sycl::context(dev_group));

    slab_ctx_->device_to_platform.resize(device_list.size());
    slab_ctx_->device_slab_sizes.resize(device_list.size());
    slab_ctx_->pools.resize(device_list.size());

    const Config& cfg = config();

    for (size_t dev_idx = 0; dev_idx < device_list.size(); ++dev_idx) {
        const sycl::device&  dev      = device_list[dev_idx];
        const sycl::platform plat     = dev.get_platform();
        const auto           backend  = plat.get_backend();
        const auto           name     = plat.get_info<sycl::info::platform::name>();

        auto it = std::find_if(platform_keys.begin(), platform_keys.end(),
                               [backend, &name](const auto& key) {
                                   return key.first == backend && key.second == name;
                               });
        assert(it != platform_keys.end());
        slab_ctx_->device_to_platform[dev_idx] = std::distance(platform_keys.begin(), it);

        const size_t total_mem = dev.get_info<sycl::info::device::global_mem_size>();
        const float  fraction  = dev.is_cpu() ? cfg.cpu_slab_fraction : cfg.device_slab_fraction;
        const size_t computed  = static_cast<size_t>(total_mem * fraction);
        const size_t raw       = std::max(computed, 2 * cfg.block_size);
        slab_ctx_->device_slab_sizes[dev_idx] = (raw / cfg.block_size) * cfg.block_size;
    }

    slab_ctx_->node_size_ = cfg.node_size;
}

inline DeviceMemoryPool &AllocatorRegistry::pool(const sycl::device &dev) {
    const size_t idx = slab_ctx_->device_index(dev);
    return slab_ctx_->pool_at(idx);
}

inline size_t AllocatorRegistry::device_index(const sycl::device &dev) {
    return slab_ctx_->device_index(dev);
}

inline const sycl::device &AllocatorRegistry::device_by_index(size_t idx) {
    if (idx >= slab_ctx_->devices.size())
        throw std::out_of_range("slabkit: device_index out of range");
    return slab_ctx_->devices[idx];
}

inline sycl::queue
AllocatorRegistry::make_in_order_queue(const sycl::device &dev) {
    return slab_ctx_->make_in_order_queue(dev);
}

inline sycl::queue
AllocatorRegistry::make_out_of_order_queue(const sycl::device &dev) {
    return slab_ctx_->make_out_of_order_queue(dev);
}

inline sycl::queue
AllocatorRegistry::make_queue_with_props(const sycl::device &dev,
                                         const sycl::property_list &props) {
    return slab_ctx_->make_queue_with_props(dev, props);
}

inline AllocatorRegistry::Config &AllocatorRegistry::config() {
    static Config instance;
    return instance;
}

inline void AllocatorRegistry::configure(Config cfg) {
    auto& reg = get();
    if (reg.slab_ctx_)
        throw std::logic_error("slabkit: AllocatorRegistry::configure() called after registry is live");
    reg.config() = cfg;
    reg.configured_ = true;
}

#ifdef SLABKIT_TESTING
inline void AllocatorRegistry::reset_for_testing() {
    auto& reg = get();
    reg.slab_ctx_.reset();
    reg.configured_ = false;
    reg.init();  // re-initialize immediately so the registry is usable after reset
}
#endif

} // namespace slabkit
