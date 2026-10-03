#pragma once

// =============================================================================
// naga/slabkit/detail/pool_impl.hpp — the fixed-node free list behind DevicePool
// =============================================================================
//
// One USM slab cut into node_size slots, tracked by a two-level bitmap that
// lives entirely in host memory. The device pointer is never dereferenced
// here; it is an opaque offset base.
//
//   l0_[i]: bit j set  <=> slot (i*64 + j) is FREE
//   l1_[i]: bit j set  <=> l0_[i*64 + j] has at least one free bit
//   1 = free, throughout.
//
//   allocate_node()    O(1)      tzcnt(l1) -> tzcnt(l0[group]) -> mark used
//   allocate_array(n)  O(N/64)   bitmap scan for n contiguous free slots
//   release(p, n)      O(n)
//
// Exhaustion and fragmentation return nullptr rather than throwing: these are
// the malloc_* family's null cases, not errors.
//
// ALIGNMENT. Node addresses must be node_size-aligned, and the runtime will
// not promise that for the slab base (AdaptiveCpp's CUDA allocator ignores the
// requested alignment altogether). So the slab is over-allocated by one node
// and the grid starts at the first aligned address inside it.
//
// The slab is an ordinary slabkit allocation, so this class has no destructor.
//
// Thread safety: every mutating entry point takes the mutex.

#include <atomic>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <sycl/sycl.hpp>

#include "detail/os_alloc.hpp"
#include "naga/slabkit/handles.hpp"
#include "naga/slabkit/kinds.hpp"

namespace naga::slabkit::detail {

class PoolImpl {
public:
    static constexpr std::size_t npos = static_cast<std::size_t>(-1);

    /// \throws std::invalid_argument if node_size is not a power of two.
    PoolImpl(const sycl::device& dev, const sycl::context& ctx, kind k, std::size_t bytes,
             std::size_t node_size, const sycl::property_list& props)
        : dev_{dev}
        , ctx_{ctx}
        , kind_{k}
        , node_size_{node_size}
        , slot_count_{node_size == 0 ? 0 : bytes / node_size} {
        if (!is_power_of_two(node_size))
            throw std::invalid_argument{
                "naga::slabkit::DevicePool: node_size must be a power of two"};

        // A pool of less than one node (zero bytes included) is legal: it owns no slab, serves nothing, and every
        // allocation through its route falls back to the OS and is counted.
        if (slot_count_ == 0) return;

        // One extra node of slack so the grid can start at an aligned address.
        slab_ = os_allocate(k, node_size, bytes + node_size, dev, ctx, props);
        if (!slab_) {
            slot_count_ = 0;
            return;
        }
        const auto raw = reinterpret_cast<std::uintptr_t>(slab_.get());
        base_          = reinterpret_cast<std::byte*>(align_up(raw, node_size));

        free_count_ = slot_count_;
        l0_.assign((slot_count_ + 63) / 64, ~std::uint64_t{0});
        l1_.assign((l0_.size() + 63) / 64, ~std::uint64_t{0});

        // Mask the phantom trailing bits of the last word of each level.
        if (const std::size_t bits = slot_count_ % 64; bits != 0 && !l0_.empty())
            l0_.back() = (std::uint64_t{1} << bits) - 1;
        if (const std::size_t bits = l0_.size() % 64; bits != 0 && !l1_.empty())
            l1_.back() = (std::uint64_t{1} << bits) - 1;
    }

    PoolImpl(const PoolImpl&)            = delete;
    PoolImpl& operator=(const PoolImpl&) = delete;

    /// Allocates `nodes` contiguous slots. Returns nullptr when the pool is
    /// exhausted or too fragmented to place the run.
    void* allocate(std::size_t nodes) noexcept {
        if (nodes == 0 || base_ == nullptr) return nullptr;

        std::lock_guard<std::mutex> lock{mutex_};
        if (free_count_ < nodes) return nullptr;

        const std::size_t start = (nodes == 1) ? find_one() : find_contiguous(nodes);
        if (start == npos) return nullptr;

        for (std::size_t i = 0; i < nodes; ++i) mark_used(start + i);
        return slot_to_ptr(start);
    }

    /// Returns `nodes` slots starting at `p`.
    void release(void* p, std::size_t nodes) noexcept {
        if (p == nullptr || nodes == 0) return;

        std::lock_guard<std::mutex> lock{mutex_};
        const std::size_t           start = ptr_to_slot(p);
        assert(start + nodes <= slot_count_);
        for (std::size_t i = 0; i < nodes; ++i) {
            // Double-release detection. NOTE: this is an assert, so it is gone
            // in release builds -- hunt allocator corruption in a Debug build.
            assert(!is_free(start + i) && "double release detected in DevicePool");
            mark_free(start + i);
        }
    }

    kind get_kind() const noexcept { return kind_; }

    /// Is this address inside the pool's own grid? The allocator's release
    /// has only the address to go on.
    bool owns(const void* p) const noexcept {
        const auto* b = static_cast<const std::byte*>(p);
        return base_ != nullptr && b >= base_ && b < base_ + slot_count_ * node_size_;
    }

    std::size_t node_size() const noexcept { return node_size_; }
    std::size_t slot_count() const noexcept { return slot_count_; }

    std::size_t free_count() const noexcept {
        std::lock_guard<std::mutex> lock{mutex_};
        return free_count_;
    }
    std::size_t outstanding() const noexcept {
        std::lock_guard<std::mutex> lock{mutex_};
        return slot_count_ - free_count_;
    }

    /// The OS fallback of this pool's allocator route: the number of allocations ever sent to the OS because the pool
    /// could not serve them (cumulative), and the bytes of those still live.
    std::size_t fallback_allocations() const noexcept { return fallback_allocations_.load(std::memory_order_relaxed); }
    std::size_t fallback_bytes_in_use() const noexcept { return fallback_bytes_.load(std::memory_order_relaxed); }
    void        add_fallback(std::size_t bytes) noexcept {
        fallback_allocations_.fetch_add(1, std::memory_order_relaxed);
        fallback_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    }
    void sub_fallback(std::size_t bytes) noexcept { fallback_bytes_.fetch_sub(bytes, std::memory_order_relaxed); }

    const sycl::device&  device() const noexcept { return dev_; }
    const sycl::context& context() const noexcept { return ctx_; }

private:
    void*       slot_to_ptr(std::size_t slot) const noexcept { return base_ + slot * node_size_; }
    std::size_t ptr_to_slot(void* p) const noexcept {
        return static_cast<std::size_t>(static_cast<std::byte*>(p) - base_) / node_size_;
    }

    bool is_free(std::size_t slot) const noexcept {
        return (l0_[slot / 64] >> (slot % 64)) & 1u;
    }

    void mark_used(std::size_t slot) noexcept {
        const std::size_t word = slot / 64;
        l0_[word] &= ~(std::uint64_t{1} << (slot % 64));
        if (l0_[word] == 0) l1_[word / 64] &= ~(std::uint64_t{1} << (word % 64));
        --free_count_;
    }

    void mark_free(std::size_t slot) noexcept {
        const std::size_t word = slot / 64;
        l0_[word] |= (std::uint64_t{1} << (slot % 64));
        l1_[word / 64] |= (std::uint64_t{1} << (word % 64));
        ++free_count_;
    }

    /// O(1) single-slot search through the summary level.
    std::size_t find_one() const noexcept {
        for (std::size_t w = 0; w < l1_.size(); ++w) {
            if (l1_[w] == 0) continue;
            const std::size_t group =
                w * 64 + static_cast<std::size_t>(std::countr_zero(l1_[w]));
            if (group >= l0_.size() || l0_[group] == 0) continue;
            const std::size_t slot =
                group * 64 + static_cast<std::size_t>(std::countr_zero(l0_[group]));
            if (slot < slot_count_) return slot;
        }
        return npos;
    }

    std::size_t find_contiguous(std::size_t n) const noexcept {
        return (n <= 64) ? find_contiguous_short(n) : find_contiguous_long(n);
    }

    /// n <= 64: fold the word to find n consecutive free bits, then check runs
    /// that straddle a word boundary.
    std::size_t find_contiguous_short(std::size_t n) const noexcept {
        const std::size_t nwords = l0_.size();
        for (std::size_t w = 0; w < nwords; ++w) {
            std::uint64_t mask = l0_[w];
            for (std::size_t shift = 1; shift < n; ++shift) mask &= (l0_[w] >> shift);
            if (mask != 0) {
                const std::size_t slot =
                    w * 64 + static_cast<std::size_t>(std::countr_zero(mask));
                if (slot + n <= slot_count_) return slot;
            }

            if (w + 1 < nwords) {
                // countl_one = free slots at the HIGH end of word w;
                // countr_one = free slots at the LOW end of word w+1.
                const auto tail_free = static_cast<std::size_t>(std::countl_one(l0_[w]));
                if (tail_free == 0) continue;
                const auto head_free = static_cast<std::size_t>(std::countr_one(l0_[w + 1]));
                if (tail_free + head_free >= n) {
                    const std::size_t slot = w * 64 + (64 - tail_free);
                    if (slot + n <= slot_count_) return slot;
                }
            }
        }
        return npos;
    }

    /// n > 64: a partial head, a run of fully free words, then a partial tail.
    std::size_t find_contiguous_long(std::size_t n) const noexcept {
        const std::size_t nwords = l0_.size();
        for (std::size_t w = 0; w < nwords;) {
            if (l0_[w] == 0) {
                ++w;
                continue;
            }
            const auto head_bits = static_cast<std::size_t>(std::countl_one(l0_[w]));
            if (head_bits == 0 || head_bits >= n) {
                ++w;
                continue;
            }

            const std::size_t after       = n - head_bits;
            const std::size_t full_needed = after / 64;
            const std::size_t rem_needed  = after % 64;

            std::size_t ww     = w + 1;
            bool        run_ok = true;
            for (std::size_t f = 0; f < full_needed; ++f, ++ww) {
                if (ww >= nwords || l0_[ww] != ~std::uint64_t{0}) {
                    run_ok = false;
                    w      = ww;
                    break;
                }
            }
            if (!run_ok) continue;

            if (rem_needed > 0) {
                if (ww >= nwords) {
                    ++w;
                    continue;
                }
                const auto low_free = static_cast<std::size_t>(std::countr_one(l0_[ww]));
                if (low_free < rem_needed) {
                    w = ww;
                    continue;
                }
            }

            const std::size_t slot = w * 64 + (64 - head_bits);
            if (slot + n <= slot_count_) return slot;
            ++w;
        }
        return npos;
    }

    sycl::device                dev_;
    sycl::context               ctx_;
    kind                        kind_ = kind::device;
    unique_usm_ptr<std::byte[]> slab_;
    std::byte*                  base_       = nullptr;  // node_size-aligned grid origin
    std::size_t                 node_size_  = 0;
    std::size_t                 slot_count_ = 0;
    std::size_t                 free_count_ = 0;

    std::vector<std::uint64_t> l0_;  // one bit per slot,          1 = free
    std::vector<std::uint64_t> l1_;  // one bit per 64-slot group, 1 = has a free slot

    mutable std::mutex mutex_;

    std::atomic<std::size_t> fallback_allocations_{0};  // cumulative
    std::atomic<std::size_t> fallback_bytes_{0};        // live
};

}  // namespace naga::slabkit::detail
