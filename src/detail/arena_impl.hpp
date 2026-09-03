#pragma once

// =============================================================================
// naga/slabkit/detail/arena_impl.hpp — the bump region behind DeviceArena
// =============================================================================
//
// One USM slab, one atomic offset, no free list and no rewind. Allocation is a
// compare-exchange loop over the offset, so it is lock-free.
//
// The slab is an ordinary slabkit allocation obtained through the library's own
// (device, context) form, which is why this class has no destructor: the byte
// handle releases the slab when the last owner drops it.
//
// ALIGNMENT. The arena honors any alignment by aligning the ABSOLUTE address
// the bump lands on, not by demanding an aligned base -- the underlying
// runtime will not reliably give one. The padding that costs is charged to the
// arena, so it shows up in used() and shrinks remaining(). Size an arena with
// slack if you intend to allocate at large alignments.

#include <atomic>
#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "detail/os_alloc.hpp"
#include "naga/slabkit/handles.hpp"
#include "naga/slabkit/kinds.hpp"

namespace naga::slabkit::detail {

class ArenaImpl {
public:
    ArenaImpl(const sycl::device& dev, const sycl::context& ctx, kind k, std::size_t bytes,
              const sycl::property_list& props)
        : dev_{dev}
        , ctx_{ctx}
        , kind_{k}
        , slab_{os_allocate(k, 0, bytes, dev, ctx, props)}
        , capacity_{slab_ ? bytes : 0}
        , offset_{0} {}

    ArenaImpl(const ArenaImpl&)            = delete;
    ArenaImpl& operator=(const ArenaImpl&) = delete;

    /// Bump `bytes` off the region at `align`. Returns nullptr when what
    /// remains cannot satisfy the request -- including when the slab itself
    /// failed to allocate.
    void* allocate(std::size_t bytes, std::size_t align) noexcept {
        std::byte* const base = slab_.get();
        if (base == nullptr || !is_power_of_two(align)) return nullptr;

        const auto base_addr = reinterpret_cast<std::uintptr_t>(base);

        std::size_t cur = offset_.load(std::memory_order_relaxed);
        for (;;) {
            const std::uintptr_t here    = base_addr + cur;
            const std::uintptr_t aligned = align_up(static_cast<std::size_t>(here), align);
            const std::size_t    pad     = static_cast<std::size_t>(aligned - here);

            // Overflow-safe capacity check: every term is a remainder.
            if (pad > capacity_ - cur) return nullptr;
            if (bytes > capacity_ - cur - pad) return nullptr;

            const std::size_t next = cur + pad + bytes;
            if (offset_.compare_exchange_weak(cur, next, std::memory_order_acq_rel,
                                              std::memory_order_relaxed))
                return reinterpret_cast<void*>(aligned);
            // cur was refreshed by the failed exchange; retry.
        }
    }

    kind        get_kind() const noexcept { return kind_; }
    std::size_t capacity() const noexcept { return capacity_; }
    std::size_t used() const noexcept { return offset_.load(std::memory_order_relaxed); }
    std::size_t remaining() const noexcept { return capacity_ - used(); }

    const sycl::device&  device() const noexcept { return dev_; }
    const sycl::context& context() const noexcept { return ctx_; }

private:
    sycl::device                dev_;
    sycl::context               ctx_;
    kind                        kind_;
    unique_usm_ptr<std::byte[]> slab_;
    std::size_t                 capacity_;
    std::atomic<std::size_t>    offset_;
};

}  // namespace naga::slabkit::detail
