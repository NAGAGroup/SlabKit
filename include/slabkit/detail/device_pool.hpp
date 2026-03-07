#pragma once

// =============================================================================
// slabkit/detail/device_pool.hpp — DeviceMemoryPool
// =============================================================================
//
// A fixed-node-size memory pool over a pre-allocated device USM slab.
//
// Design constraints:
//   - slab_base is sycl::malloc_device memory — host CPU cannot write into it.
//   - All metadata lives entirely in host memory (this object).
//   - The device pointer is treated as an opaque integer offset base.
//
// Freelist design: two-level bitmap
//   l0_[i]: bit j set ⟺ slot (i*64 + j) is FREE
//   l1_[i]: bit j set ⟺ l0_[i*64 + j] has at least one free bit
//   1 = free convention throughout.
//
// allocate_node()    — O(1): tzcnt(l1) → tzcnt(l0[group]) → mark used
// deallocate_node()  — O(1): set l0 bit, update l1
// allocate_array(n)  — O(N/64): bitmap scan for n contiguous free slots
// deallocate_array() — O(n): mark n slots free
//
// Double-free detection: deallocate asserts the slot's free bit is NOT already
// set. In release builds this becomes a no-op check (assert disabled).
//
// Thread safety: NONE. External synchronisation required.
//
// Prior art: RMM pool_memory_resource, TensorFlow BFC allocator, Intel GPGMM.
// All store allocation metadata exclusively on the host and treat device
// pointers as opaque integers — same approach used here.
//

#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <vector>

namespace slabkit {

class DeviceMemoryPool {
public:
    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    /// \param slab_base  Opaque device USM pointer (sycl::malloc_device result).
    ///                   Must remain valid for the lifetime of this object.
    ///                   The pool does NOT own this pointer — caller manages it.
    /// \param node_size  Granularity of each slot in bytes. Must be > 0.
    /// \param slot_count Number of slots. slab must be >= node_size * slot_count.
    DeviceMemoryPool(void* slab_base, std::size_t node_size, std::size_t slot_count)
        : slab_base_{static_cast<char*>(slab_base)}
        , node_size_{node_size}
        , slot_count_{slot_count}
        , free_count_{slot_count}
        , l0_((slot_count + 63) / 64, ~uint64_t{0})   // all bits set = all free
        , l1_((l0_.size() + 63) / 64, ~uint64_t{0})   // all groups have free slots
    {
        assert(slab_base != nullptr);
        assert(node_size > 0);
        assert(slot_count > 0);

        // Mask off trailing phantom bits in the last l0_ word so they don't
        // appear as free slots beyond slot_count.
        const std::size_t used_bits = slot_count % 64;
        if (used_bits != 0)
            l0_.back() = (uint64_t{1} << used_bits) - 1;

        // Similarly for l1_ trailing bits.
        const std::size_t used_l1_bits = l0_.size() % 64;
        if (used_l1_bits != 0)
            l1_.back() = (uint64_t{1} << used_l1_bits) - 1;
    }

    DeviceMemoryPool(const DeviceMemoryPool&)            = delete;
    DeviceMemoryPool& operator=(const DeviceMemoryPool&) = delete;
    DeviceMemoryPool(DeviceMemoryPool&&)                 = default;
    DeviceMemoryPool& operator=(DeviceMemoryPool&&)      = default;

    ~DeviceMemoryPool() = default;

    // -------------------------------------------------------------------------
    // Single-node allocation — O(1)
    // -------------------------------------------------------------------------

    /// Allocates one slot of node_size() bytes.
    /// \throws std::bad_alloc if the pool is full.
    void* allocate_node() {
        if (free_count_ == 0)
            throw std::bad_alloc{};

        // Find a group with a free slot via l1_.
        const std::size_t l1_word = find_set_bit_word(l1_);
        const std::size_t group   = l1_word * 64 + static_cast<std::size_t>(
                                        std::countr_zero(l1_[l1_word]));

        assert(group < l0_.size());
        assert(l0_[group] != 0);

        const std::size_t bit  = static_cast<std::size_t>(std::countr_zero(l0_[group]));
        const std::size_t slot = group * 64 + bit;

        assert(slot < slot_count_);
        mark_used(slot);
        return slot_to_ptr(slot);
    }

    /// Deallocates a single node previously returned by allocate_node().
    /// \pre ptr must be a value returned by allocate_node() on this pool.
    void deallocate_node(void* ptr) noexcept {
        const std::size_t slot = ptr_to_slot(ptr);
        assert(slot < slot_count_);
        assert(!is_free(slot) && "double-free detected in DeviceMemoryPool");
        mark_free(slot);
    }

    // -------------------------------------------------------------------------
    // Array allocation — O(N/64) scan
    // -------------------------------------------------------------------------

    /// Allocates n contiguous slots (n * node_size() bytes).
    /// \throws std::bad_alloc if no contiguous run of n free slots exists.
    void* allocate_array(std::size_t n) {
        if (n == 0)
            return nullptr;
        if (n == 1)
            return allocate_node();
        if (free_count_ < n)
            throw std::bad_alloc{};

        const std::size_t start = find_contiguous(n); // throws bad_alloc if not found

        for (std::size_t i = 0; i < n; ++i)
            mark_used(start + i);

        return slot_to_ptr(start);
    }

    /// Deallocates n contiguous slots previously returned by allocate_array(n).
    /// \pre ptr and n must match a prior allocate_array(n) call on this pool.
    void deallocate_array(void* ptr, std::size_t n) noexcept {
        if (n == 0 || ptr == nullptr)
            return;
        if (n == 1) {
            deallocate_node(ptr);
            return;
        }

        const std::size_t start = ptr_to_slot(ptr);
        assert(start + n <= slot_count_);

        for (std::size_t i = 0; i < n; ++i) {
            assert(!is_free(start + i) && "double-free detected in DeviceMemoryPool::deallocate_array");
            mark_free(start + i);
        }
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    std::size_t node_size()   const noexcept { return node_size_; }
    std::size_t slot_count()  const noexcept { return slot_count_; }
    std::size_t free_count()  const noexcept { return free_count_; }
    bool        full()        const noexcept { return free_count_ == 0; }
    bool        empty()       const noexcept { return free_count_ == slot_count_; }

    /// Returns the device base pointer (opaque — do not dereference from host).
    const void* slab_base()   const noexcept { return slab_base_; }

private:
    // -------------------------------------------------------------------------
    // Pointer ↔ slot index conversion
    // -------------------------------------------------------------------------

    void* slot_to_ptr(std::size_t slot) const noexcept {
        return slab_base_ + slot * node_size_;
    }

    std::size_t ptr_to_slot(void* ptr) const noexcept {
        return (static_cast<char*>(ptr) - slab_base_) / node_size_;
    }

    // -------------------------------------------------------------------------
    // Bitmap helpers
    // -------------------------------------------------------------------------

    bool is_free(std::size_t slot) const noexcept {
        return (l0_[slot / 64] >> (slot % 64)) & 1u;
    }

    void mark_used(std::size_t slot) noexcept {
        const std::size_t word = slot / 64;
        const std::size_t bit  = slot % 64;
        l0_[word] &= ~(uint64_t{1} << bit);
        if (l0_[word] == 0)
            l1_[word / 64] &= ~(uint64_t{1} << (word % 64));
        --free_count_;
    }

    void mark_free(std::size_t slot) noexcept {
        const std::size_t word = slot / 64;
        const std::size_t bit  = slot % 64;
        l0_[word] |= (uint64_t{1} << bit);
        l1_[word / 64] |= (uint64_t{1} << (word % 64));
        ++free_count_;
    }

    /// Returns index of first word in v that has at least one set bit.
    /// Assumes at least one such word exists (caller checks free_count_ > 0).
    static std::size_t find_set_bit_word(const std::vector<uint64_t>& v) noexcept {
        for (std::size_t i = 0; i < v.size(); ++i)
            if (v[i] != 0)
                return i;
        // Unreachable if caller ensures free_count_ > 0.
        assert(false && "find_set_bit_word: no set bits found");
        return 0;
    }

    // -------------------------------------------------------------------------
    // Contiguous slot search for allocate_array(n)
    // -------------------------------------------------------------------------
    //
    // Strategy:
    //   - n <= 64: search within individual l0_ words for a run of n consecutive
    //              free bits. Use the bit-folding trick: a run of n consecutive
    //              1-bits exists in word w iff (w & w>>1 & ... & w>>(n-1)) != 0.
    //              Also check runs that span two adjacent words.
    //   - n > 64:  scan for a sequence of fully-free words (l0_ word == ~0)
    //              with optional partial words at head and tail.
    //
    // Returns the starting slot index of the found run.
    // Throws std::bad_alloc if no run of length n exists.
    //

    std::size_t find_contiguous(std::size_t n) const {
        assert(n > 1);

        if (n <= 64)
            return find_contiguous_short(n);
        else
            return find_contiguous_long(n);
    }

    // Find n consecutive free bits where n <= 64.
    // Checks within each word and across adjacent word boundaries.
    std::size_t find_contiguous_short(std::size_t n) const {
        assert(n >= 2 && n <= 64);

        const std::size_t nwords = l0_.size();

        for (std::size_t w = 0; w < nwords; ++w) {
            // --- Within-word search ---
            uint64_t mask = l0_[w];
            for (std::size_t shift = 1; shift < n; ++shift)
                mask &= (l0_[w] >> shift);
            // mask has a 1 at bit b iff bits [b, b+n) are all free in l0_[w].
            if (mask != 0) {
                const std::size_t bit  = static_cast<std::size_t>(std::countr_zero(mask));
                const std::size_t slot = w * 64 + bit;
                if (slot + n <= slot_count_)
                    return slot;
            }

            // --- Cross-boundary search: high-end of word w + low-end of word w+1 ---
            //
            // Bit layout: bit0 = slot w*64 (low index), bit63 = slot w*64+63 (high index)
            //   countl_one(l0_[w])   = free slots at HIGH end of word w (bits 63,62,...)
            //   countr_one(l0_[w+1]) = free slots at LOW  end of word w+1 (bits 0,1,...)
            //
            if (w + 1 < nwords) {
                const std::size_t tail_free = static_cast<std::size_t>(
                    std::countl_one(l0_[w]));        // free at high end of word w

                if (tail_free == 0) {
                    ++w;
                    continue;
                }

                const std::size_t head_free = static_cast<std::size_t>(
                    std::countr_one(l0_[w + 1]));    // free at low end of word w+1

                if (tail_free + head_free >= n) {
                    // Run starts at: w*64 + (64 - tail_free)
                    const std::size_t slot = w * 64 + (64 - tail_free);
                    if (slot + n <= slot_count_)
                        return slot;
                }
            }
        }

        throw std::bad_alloc{};
    }

    // Find n consecutive free slots where n > 64.
    // Scans l0_ for a run of fully-free words with partial words at boundaries.
    std::size_t find_contiguous_long(std::size_t n) const {
        assert(n > 64);

        const std::size_t nwords     = l0_.size();
        const std::size_t full_words = n / 64;        // minimum fully-free words needed
        const std::size_t remainder  = n % 64;        // extra bits needed beyond full words

        for (std::size_t w = 0; w < nwords; ) {
            // Find the start of a free run — either a partially-free word we
            // can use as a head, or a fully-free word.

            if (l0_[w] == 0) {
                // This word is completely used — skip it.
                ++w;
                continue;
            }

            // Count the free "tail" of word w (high-indexed slots).
            const std::size_t head_free = static_cast<std::size_t>(
                std::countl_one(l0_[w]));  // free slots at high end of word w

            // If using word w as a partial head:
            // We need head_free slots from word w, then (full_words_needed) fully-free
            // words, then remainder_needed more slots from the next partial word.

            const std::size_t head_bits   = head_free;
            const std::size_t bits_after_head = n - head_bits;  // could underflow if head_free >= n
            if (head_free >= n) {
                // Entire run fits within the tail of this word — shouldn't happen
                // since n > 64 > head_free <= 64. But handle gracefully.
                // Actually head_free <= 64 and n > 64, so head_free < n always. Skip.
                ++w;
                continue;
            }

            const std::size_t full_needed = bits_after_head / 64;
            const std::size_t rem_needed  = bits_after_head % 64;

            // Check that full_needed fully-free words follow word w.
            std::size_t ww = w + 1;
            bool run_ok = true;
            for (std::size_t f = 0; f < full_needed; ++f, ++ww) {
                if (ww >= nwords || l0_[ww] != ~uint64_t{0}) {
                    run_ok = false;
                    // Skip to ww to resume search after the break.
                    w = ww;
                    break;
                }
            }
            if (!run_ok)
                continue;

            // Check rem_needed free bits at the low end of word ww.
            if (rem_needed > 0) {
                if (ww >= nwords) {
                    ++w;
                    continue;
                }
                const std::size_t low_free = static_cast<std::size_t>(
                    std::countr_one(l0_[ww]));  // free slots at low end of word ww
                if (low_free < rem_needed) {
                    // Not enough — next candidate starts at ww.
                    w = ww;
                    continue;
                }
            }

            // Found a valid run.
            const std::size_t slot = w * 64 + (64 - head_bits);
            if (slot + n <= slot_count_)
                return slot;

            ++w;
        }

        throw std::bad_alloc{};
    }

    // -------------------------------------------------------------------------
    // Data members
    // -------------------------------------------------------------------------

    char*       slab_base_;    // device memory base — never dereferenced from host
    std::size_t node_size_;
    std::size_t slot_count_;
    std::size_t free_count_;

    std::vector<uint64_t> l0_;  // one bit per slot,       1 = free
    std::vector<uint64_t> l1_;  // one bit per 64-slot group, 1 = group has free slot
};

} // namespace slabkit
