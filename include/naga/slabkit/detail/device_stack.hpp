#pragma once

// =============================================================================
// slabkit/detail/device_stack.hpp — DeviceScratchMemory
// =============================================================================
//
// A pure bump allocator over a device USM slab, optionally owning a pool allocation.
// No free-list, no bitmap — just a pointer advance.
//
// Design constraints:
//   - slab_base is a device memory pointer — host CPU cannot write into it.
//   - All metadata lives entirely in host memory (this object).
//   - The device pointer is treated as an opaque integer offset base.
//   - Can be pool-owning (allocated from DeviceMemoryPool) or non-owning (pre-allocated).
//
// Allocation strategy:
//   allocate(size, alignment) aligns the current offset and advances by size.
//   unwind(marker)             rewinds the offset to a saved marker.
//   reset()                    equivalent to unwind(0).
//
// Thread safety: NONE. External synchronisation required.
//

#include <cassert>
#include <cstddef>
#include <memory>
#include <new>
#include <stdexcept>

#include "naga/slabkit/error.hpp"
#include "naga/slabkit/detail/device_pool.hpp"

namespace naga::slabkit {

class DeviceScratchMemory {
public:
    // -------------------------------------------------------------------------
    // Marker type
    // -------------------------------------------------------------------------

    /// Opaque offset into the slab. Used for unwind().
    using Marker = std::size_t;

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    /// Non-owning constructor — takes a pre-allocated slab.
    /// \param slab_base  Opaque device USM pointer (sycl::malloc_device result).
    ///                   Must remain valid for the lifetime of this object.
    ///                   The allocator does NOT own this pointer — caller manages it.
    /// \param capacity   Total slab size in bytes. Must be > 0.
    DeviceScratchMemory(void* slab_base, std::size_t capacity) noexcept
        : pool_{nullptr}
        , node_count_{0}
        , slab_base_{static_cast<std::byte*>(slab_base)}
        , capacity_{capacity}
        , offset_{0}
    {
        assert(slab_base != nullptr);
        assert(capacity > 0);
    }

    /// Pool-owning constructor — allocates \p scratch_size bytes from \p pool.
    /// The allocation is returned to the pool automatically when this object is destroyed.
    /// \param pool         Shared pool to allocate from. Kept alive by this object.
    /// \param scratch_size Number of bytes to allocate (rounded up to node granularity).
    DeviceScratchMemory(DeviceMemoryPool* pool, std::size_t scratch_size)
        : pool_{pool}
        , node_count_{(scratch_size + pool_->node_size() - 1) / pool_->node_size()}
        , slab_base_{static_cast<std::byte*>(pool_->allocate_array(node_count_))}
        , capacity_{node_count_ * pool_->node_size()}
        , offset_{0}
    {
        assert(pool_ != nullptr);
        assert(scratch_size > 0);
    }

    DeviceScratchMemory(const DeviceScratchMemory&)            = delete;
    DeviceScratchMemory& operator=(const DeviceScratchMemory&) = delete;

    DeviceScratchMemory(DeviceScratchMemory&& other) noexcept
        : pool_{other.pool_}
        , node_count_{other.node_count_}
        , slab_base_{other.slab_base_}
        , capacity_{other.capacity_}
        , offset_{other.offset_}
    {
        other.slab_base_ = nullptr;
        other.pool_      = nullptr;
    }

    DeviceScratchMemory& operator=(DeviceScratchMemory&& other) noexcept {
        if (this != &other) {
            if (pool_ && slab_base_)
                pool_->deallocate_array(slab_base_, node_count_);
            pool_       = other.pool_;
            node_count_ = other.node_count_;
            slab_base_  = other.slab_base_;
            capacity_   = other.capacity_;
            offset_     = other.offset_;
            other.slab_base_ = nullptr;
            other.pool_      = nullptr;
        }
        return *this;
    }

    ~DeviceScratchMemory() {
        if (pool_ && slab_base_)
            pool_->deallocate_array(slab_base_, node_count_);
    }

    // -------------------------------------------------------------------------
    // Allocation
    // -------------------------------------------------------------------------

    /// Allocates 'size' bytes aligned to 'alignment' (must be power of two).
    /// Returns device pointer (opaque — do not dereference from host).
    /// \throws AllocationError if not enough space.
    void* allocate(std::size_t size,
                   std::size_t alignment = alignof(std::max_align_t)) {
        assert(size > 0);
        assert((alignment & (alignment - 1)) == 0 && "alignment must be a power of two");

        // Align the current offset upward.
        const std::size_t aligned_offset = (offset_ + alignment - 1) & ~(alignment - 1);

        // Check for overflow and capacity.
        if (aligned_offset + size > capacity_)
            throw AllocationError{AllocationError::Reason::ScratchFull,
                                  "scratch allocator capacity exceeded"};

        void* result = static_cast<void*>(slab_base_ + aligned_offset);
        offset_      = aligned_offset + size;

        return result;
    }

    // -------------------------------------------------------------------------
    // Markers and unwinding
    // -------------------------------------------------------------------------

    /// Returns the current bump position as an opaque marker.
    Marker top() const noexcept {
        return offset_;
    }

    /// Rewinds the bump pointer to a previously-saved marker.
    /// \pre marker <= offset_ (asserted).
    /// Does NOT touch device memory — purely adjusts the host offset.
    void unwind(Marker marker) noexcept {
        assert(marker <= offset_);
        offset_ = marker;
    }

    /// Reset to empty — equivalent to unwind(0).
    void reset() noexcept {
        offset_ = 0;
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    /// Total slab capacity in bytes.
    std::size_t capacity() const noexcept {
        return capacity_;
    }

    /// Bytes currently allocated.
    std::size_t used() const noexcept {
        return offset_;
    }

    /// Bytes still available.
    std::size_t remaining() const noexcept {
        return capacity_ - offset_;
    }

    /// True iff remaining() == 0.
    bool full() const noexcept {
        return offset_ == capacity_;
    }

    /// True iff used() == 0.
    bool empty() const noexcept {
        return offset_ == 0;
    }

    /// Returns the device base pointer (opaque — do not dereference from host).
    const std::byte* slab_base() const noexcept {
        return slab_base_;
    }

private:
    // -------------------------------------------------------------------------
    // Data members
    // -------------------------------------------------------------------------

    DeviceMemoryPool* pool_;                   // non-null iff pool-owned
    std::size_t                       node_count_  = 0;        // slots allocated from pool_
    std::byte*                        slab_base_   = nullptr;  // device memory base — never dereferenced from host
    std::size_t                       capacity_    = 0;        // total slab size in bytes
    std::size_t                       offset_      = 0;        // current bump position (bytes from slab_base_)
};

} // namespace naga::slabkit
