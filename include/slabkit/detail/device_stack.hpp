#pragma once

// =============================================================================
// slabkit/detail/device_stack.hpp — DeviceScratchMemory
// =============================================================================
//
// A pure bump allocator over a pre-allocated device USM slab.
// No free-list, no bitmap — just a pointer advance.
//
// Design constraints:
//   - slab_base is sycl::malloc_device memory — host CPU cannot write into it.
//   - All metadata lives entirely in host memory (this object).
//   - The device pointer is treated as an opaque integer offset base.
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

namespace slabkit {

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

    /// \param slab_base  Opaque device USM pointer (sycl::malloc_device result).
    ///                   Must remain valid for the lifetime of this object.
    ///                   The allocator does NOT own this pointer — caller manages it.
    /// \param capacity   Total slab size in bytes. Must be > 0.
    DeviceScratchMemory(void* slab_base, std::size_t capacity) noexcept
        : owner_{}
        , slab_base_{static_cast<std::byte*>(slab_base)}
        , capacity_{capacity}
        , offset_{0}
    {
        assert(slab_base != nullptr);
        assert(capacity > 0);
    }

    /// Owning constructor — takes shared ownership of a pool allocation.
    /// The allocation is released automatically when this object is destroyed.
    /// \param owner    shared_ptr<std::byte> managing the device allocation.
    /// \param capacity Total size in bytes (must equal the allocation size).
    DeviceScratchMemory(std::shared_ptr<std::byte> owner, std::size_t capacity) noexcept
        : owner_{std::move(owner)}
        , slab_base_{owner_.get()}
        , capacity_{capacity}
        , offset_{0}
    {
        assert(owner_ != nullptr);
        assert(capacity > 0);
    }

    DeviceScratchMemory(const DeviceScratchMemory&)            = delete;
    DeviceScratchMemory& operator=(const DeviceScratchMemory&) = delete;
    DeviceScratchMemory(DeviceScratchMemory&&)                 = default;
    DeviceScratchMemory& operator=(DeviceScratchMemory&&)      = default;

    ~DeviceScratchMemory() = default;

    // -------------------------------------------------------------------------
    // Allocation
    // -------------------------------------------------------------------------

    /// Allocates 'size' bytes aligned to 'alignment' (must be power of two).
    /// Returns device pointer (opaque — do not dereference from host).
    /// \throws std::bad_alloc if not enough space.
    void* allocate(std::size_t size,
                   std::size_t alignment = alignof(std::max_align_t)) {
        assert(size > 0);
        assert((alignment & (alignment - 1)) == 0 && "alignment must be a power of two");

        // Align the current offset upward.
        const std::size_t aligned_offset = (offset_ + alignment - 1) & ~(alignment - 1);

        // Check for overflow and capacity.
        if (aligned_offset + size > capacity_)
            throw std::bad_alloc{};

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

    std::shared_ptr<std::byte> owner_;      // non-null iff this object owns its memory
    std::byte*                 slab_base_;  // device memory base — never dereferenced from host
    std::size_t                capacity_;   // total slab size in bytes
    std::size_t                offset_;     // current bump position (bytes from slab_base_)
};

} // namespace slabkit
