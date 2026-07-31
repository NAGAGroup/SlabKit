#pragma once

// =============================================================================
// slabkit/scratch.hpp — public scratch buffer types and factories
// =============================================================================
//
// Exports:
//   - DeviceScratchMemory: bump allocator for temporary device memory
//   - DeviceScratchView: non-owning, trivially copyable view into scratch memory
//   - make_scratch(ctx, device_index, size): explicit-context factory
//   - make_scratch(ctx, dev, size): device-based explicit-context factory
//

#include <cstddef>

#include <sycl/sycl.hpp>

#include "naga/slabkit/detail/device_stack.hpp"
#include "naga/slabkit/context.hpp"

namespace naga::slabkit {

// =============================================================================
// DeviceScratchView — non-owning trivially copyable view
// =============================================================================
//
// Points into a DeviceScratchMemory allocation (or any device USM region).
// Safe to copy into SYCL kernels, like DeviceView.
// No ownership, no destructor side effects.
//

struct DeviceScratchView {
    std::byte* ptr_;      // device pointer (opaque from host)
    std::size_t size_;    // bytes

    /// Construct a view from a device pointer and size.
    static DeviceScratchView from_allocation(void* ptr, std::size_t size) noexcept {
        return DeviceScratchView{static_cast<std::byte*>(ptr), size};
    }

    void*       data() noexcept       { return ptr_; }
    const void* data() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return size_; }
    bool        empty() const noexcept { return size_ == 0; }
};

// =============================================================================
// Factory functions
// =============================================================================

/// Explicit-context scratch factory: allocates from a Context pool by device index.
/// \param ctx          Shared Context.
/// \param device_index Index of the device in ctx.
/// \param size         Bytes to allocate (default: 64 KB).
/// \throws std::out_of_range if device_index is invalid.
/// \throws AllocationError if the pool cannot satisfy the allocation.
[[nodiscard]] inline DeviceScratchMemory make_scratch(std::shared_ptr<Context> ctx,
                                         std::size_t device_index,
                                         std::size_t size = 64 * 1024) {
    return DeviceScratchMemory{&ctx->pool_at(device_index), size};
}

/// Device overload: resolves device_index via ctx->device_index(dev).
[[nodiscard]] inline DeviceScratchMemory make_scratch(std::shared_ptr<Context> ctx,
                                         const sycl::device& dev,
                                         std::size_t size = 64 * 1024) {
    return DeviceScratchMemory{&ctx->pool_for(dev), size};
}

}  // namespace naga::slabkit
