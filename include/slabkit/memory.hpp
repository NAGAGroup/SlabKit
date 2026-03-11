#pragma once

// =============================================================================
// slabkit/memory.hpp — USM device memory allocation
// =============================================================================
//
// DeviceUSMPoolDeleter: custom deleter for unique_ptr holding USM device memory
//   allocated from a SlabKitContext pool.
//
// unique_device_ptr<T>: alias for std::unique_ptr<T, DeviceUSMPoolDeleter>
//
// malloc_device<T>: free function templates for allocating USM device memory
//   from a SlabKitContext pool. Two overloads:
//   - (count, ctx, device_index) — allocate by pre-resolved device index
//   - (count, ctx, dev)          — resolve device index via ctx->device_index(dev)
//

#include <cstddef>
#include <memory>

#include <sycl/sycl.hpp>

#include "slabkit/context.hpp"
#include "slabkit/detail/device_pool.hpp"

namespace slabkit {

// =============================================================================
// DeviceUSMPoolDeleter — non-templated deleter for USM pointers allocated from
// SlabKitContext pools
// =============================================================================
//
// Holds a weak_ptr<SlabKitContext> and device index.
// Takes void* in operator() — compatible with unique_ptr<T, DeviceUSMPoolDeleter>
// because T* is implicitly convertible to void*. If SlabKitContext is destroyed
// before deleter fires, memory was freed with it — skip silently.
//

struct DeviceUSMPoolDeleter {
    std::weak_ptr<SlabKitContext> ctx;
    size_t device_index;
    size_t node_count; // pre-computed: ceil(count*sizeof(T) / node_size)

    void operator()(void *ptr) const noexcept {
        if (auto c = ctx.lock())
            c->pool_at(device_index).deallocate_array(ptr, node_count);
        // if expired: SlabKitContext destroyed, USM slab already freed — no-op is correct
    }
};

template <typename T>
using unique_device_ptr = std::unique_ptr<T, DeviceUSMPoolDeleter>;

// =============================================================================
// Free functions
// =============================================================================

/// Allocate count elements of T from a SlabKitContext pool by device index.
/// Returns unique_device_ptr<T> with DeviceUSMPoolDeleter holding
/// weak_ptr<SlabKitContext>.
template <typename T>
unique_device_ptr<T> malloc_device(size_t count,
                                   std::shared_ptr<SlabKitContext> ctx,
                                   size_t device_index) {
    auto &pool = ctx->pool_at(device_index);
    const size_t node_count =
        (count * sizeof(T) + pool.node_size() - 1) / pool.node_size();
    T *ptr = static_cast<T *>(pool.allocate_array(node_count));
    return unique_device_ptr<T>{
        ptr, DeviceUSMPoolDeleter{std::weak_ptr<SlabKitContext>(ctx), device_index,
                                  node_count}};
}

/// Allocate count elements of T from a SlabKitContext pool for a given device.
/// Resolves device index via ctx->device_index(dev).
template <typename T>
unique_device_ptr<T> malloc_device(size_t count,
                                   std::shared_ptr<SlabKitContext> ctx,
                                   const sycl::device& dev) {
    return malloc_device<T>(count, ctx, ctx->device_index(dev));
}

} // namespace slabkit
