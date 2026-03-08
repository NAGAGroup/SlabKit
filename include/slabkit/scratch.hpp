#pragma once

// =============================================================================
// slabkit/scratch.hpp — public scratch buffer types and factories
// =============================================================================
//
// Exports:
//   - DeviceScratchBuffer: type alias for DeviceScratchMemory
//   - DeviceScratchView: non-owning, trivially copyable view into scratch memory
//   - make_scratch(dev, size): registry-backed factory
//   - make_scratch(ctx, device_index, size): registry-free factory
//
// Note: Factory implementations are in memory.hpp to avoid circular includes.
//

#include <cstddef>

#include "slabkit/detail/device_stack.hpp"
#include "slabkit/context.hpp"

namespace slabkit {

// =============================================================================
// Type aliases
// =============================================================================

/// DeviceScratchBuffer = DeviceScratchMemory (preferred public name)
using DeviceScratchBuffer = DeviceScratchMemory;

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
// Factory functions — declared here, implemented in memory.hpp
// =============================================================================

// =============================================================================
// Factory function — registry-free, defined here
// =============================================================================

/// Registry-free scratch factory: allocates from an explicit SlabContext pool.
/// Uses the aliasing shared_ptr trick to keep SlabContext alive via the pool pointer.
/// \param ctx          Shared SlabContext (lifetime extended via aliasing shared_ptr).
/// \param device_index Index of the device in ctx.devices.
/// \param size         Bytes to allocate (default: 64 KB).
/// \throws std::out_of_range if device_index is invalid.
/// \throws AllocationError if the pool cannot satisfy the allocation.
inline DeviceScratchBuffer make_scratch(std::shared_ptr<SlabContext> ctx,
                                         std::size_t device_index,
                                         std::size_t size = 64 * 1024) {
    return DeviceScratchBuffer{&ctx->pool_at(device_index), size};
}

// Note: registry-backed make_scratch(dev, size) is defined in memory.hpp
// (requires AllocatorRegistry which lives there).

}  // namespace slabkit
