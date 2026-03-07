#pragma once

#include <cassert>
#include <cstddef>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

#include <sycl/sycl.hpp>

#include <foonathan/memory/memory_pool.hpp>
#include <foonathan/memory/memory_pool_type.hpp>
#include <foonathan/memory/memory_stack.hpp>
#include <foonathan/memory/namespace_alias.hpp>

// =============================================================================
// Forward declarations
// =============================================================================

class DevicePageAllocatorImpl;
class AllocatorRegistry;

// =============================================================================
// FreeBlock
// =============================================================================

struct FreeBlock {
  size_t offset;
  size_t size;
};

// =============================================================================
// SubAllocator
// =============================================================================

struct SubAllocator {
  void *base;
  size_t page_start;
  size_t page_count;
  size_t capacity;
  size_t live_count = 0;
  size_t max_free = 0;

  std::vector<FreeBlock> free_list;

  SubAllocator(void *base, size_t page_start, size_t page_count,
               size_t page_size)
      : base(base), page_start(page_start), page_count(page_count),
        capacity(page_count * page_size), max_free(page_count * page_size) {
    free_list.push_back({0, capacity});
  }

  void *allocate(size_t size, size_t align) {
    for (size_t i = 0; i < free_list.size(); ++i) {
      auto &blk = free_list[i];

      size_t aligned_offset = (blk.offset + align - 1) & ~(align - 1);
      size_t padding = aligned_offset - blk.offset;

      if (padding + size > blk.size)
        continue;

      void *ptr = static_cast<char *>(base) + aligned_offset;
      size_t remaining = blk.size - padding - size;

      if (remaining == 0) {
        blk = free_list.back();
        free_list.pop_back();
      } else {
        blk.offset = aligned_offset + size;
        blk.size = remaining;
      }

      ++live_count;
      recompute_max_free();
      return ptr;
    }
    return nullptr;
  }

  void deallocate(void *ptr, size_t size) {
    size_t offset = static_cast<char *>(ptr) - static_cast<char *>(base);
    free_list.push_back({offset, size});
    --live_count;
    if (size > max_free)
      max_free = size;
  }

  bool can_fit(size_t size) const noexcept { return size <= max_free; }
  bool is_empty() const noexcept { return live_count == 0; }

private:
  void recompute_max_free() {
    max_free = 0;
    for (const auto &blk : free_list)
      if (blk.size > max_free)
        max_free = blk.size;
  }
};

// =============================================================================
// DevicePageAllocatorImpl
// =============================================================================

class DevicePageAllocatorImpl {
public:
  DevicePageAllocatorImpl(sycl::queue &q, size_t total_size, size_t page_size)
      : slab_(sycl::malloc_device(total_size, q)), ctx_(q.get_context()),
        page_size_(page_size), page_count_(total_size / page_size),
        page_map_(page_count_, nullptr) {
    assert(total_size % page_size == 0);
    assert(page_size % alignof(std::max_align_t) == 0);
    recompute_max_contiguous_pages();
    update_max_node_size();
  }

  ~DevicePageAllocatorImpl() { sycl::free(slab_, ctx_); }

  DevicePageAllocatorImpl(const DevicePageAllocatorImpl &) = delete;
  DevicePageAllocatorImpl &operator=(const DevicePageAllocatorImpl &) = delete;
  DevicePageAllocatorImpl(DevicePageAllocatorImpl &&) = delete;
  DevicePageAllocatorImpl &operator=(DevicePageAllocatorImpl &&) = delete;

  void *allocate_node(size_t size, size_t align) {
    if (size == 0 || size > max_node_size_)
      throw std::bad_alloc{};

    if (size < page_size_) {
      for (auto &sub : sub_allocators_) {
        if (!sub.can_fit(size))
          continue;

        size_t old_max = sub.max_free;
        void *ptr = sub.allocate(size, align);
        if (!ptr)
          continue;

        if (old_max == max_sub_free_)
          recompute_max_sub_free();
        update_max_node_size();
        return ptr;
      }

      SubAllocator &sub = make_sub_allocator(find_free_pages(1), 1);
      void *ptr = sub.allocate(size, align);
      assert(ptr);
      recompute_max_sub_free();
      update_max_node_size();
      return ptr;

    } else {
      size_t pages_needed = (size + page_size_ - 1) / page_size_;
      SubAllocator &sub =
          make_sub_allocator(find_free_pages(pages_needed), pages_needed);
      void *ptr = sub.allocate(size, align);
      assert(ptr);
      recompute_max_sub_free();
      update_max_node_size();
      return ptr;
    }
  }

  void deallocate_node(void *ptr, size_t size, size_t /*align*/) noexcept {
    size_t page_index =
        (static_cast<char *>(ptr) - static_cast<char *>(slab_)) / page_size_;
    SubAllocator *sub = page_map_[page_index];
    assert(sub && "double free or pointer not from this allocator");

    sub->deallocate(ptr, size);

    if (sub->is_empty()) {
      release_sub_allocator(sub);
      return;
    }

    if (sub->max_free > max_sub_free_)
      max_sub_free_ = sub->max_free;
    update_max_node_size();
  }

  size_t max_node_size() const noexcept { return max_node_size_; }

private:
  void *slab_;
  sycl::context ctx_;
  size_t page_size_;
  size_t page_count_;

  std::vector<SubAllocator *> page_map_;
  std::list<SubAllocator> sub_allocators_;

  size_t max_contiguous_pages_ = 0;
  size_t max_sub_free_ = 0;
  size_t max_node_size_ = 0;

  void recompute_max_contiguous_pages() {
    size_t best = 0, run = 0;
    for (size_t i = 0; i < page_count_; ++i) {
      run = page_map_[i] ? 0 : run + 1;
      if (run > best)
        best = run;
    }
    max_contiguous_pages_ = best;
  }

  void recompute_max_sub_free() {
    max_sub_free_ = 0;
    for (const auto &sub : sub_allocators_)
      if (sub.max_free > max_sub_free_)
        max_sub_free_ = sub.max_free;
  }

  void update_max_node_size() {
    max_node_size_ =
        std::max(max_contiguous_pages_ * page_size_, max_sub_free_);
  }

  size_t find_free_pages(size_t count) {
    size_t run_start = 0, run_len = 0;
    for (size_t i = 0; i < page_count_; ++i) {
      if (page_map_[i] == nullptr) {
        if (run_len == 0)
          run_start = i;
        if (++run_len == count)
          return run_start;
      } else {
        run_len = 0;
      }
    }
    throw std::bad_alloc{};
  }

  SubAllocator &make_sub_allocator(size_t page_start, size_t pages) {
    void *base = static_cast<char *>(slab_) + page_start * page_size_;
    sub_allocators_.emplace_back(base, page_start, pages, page_size_);
    SubAllocator *sub = &sub_allocators_.back();
    for (size_t i = page_start; i < page_start + pages; ++i)
      page_map_[i] = sub;
    recompute_max_contiguous_pages();
    return *sub;
  }

  void release_sub_allocator(SubAllocator *sub) noexcept {
    for (size_t i = sub->page_start; i < sub->page_start + sub->page_count; ++i)
      page_map_[i] = nullptr;
    sub_allocators_.remove_if(
        [sub](const SubAllocator &s) { return &s == sub; });
    recompute_max_contiguous_pages();
    recompute_max_sub_free();
    update_max_node_size();
  }
};

// =============================================================================
// DevicePageAllocator — lightweight handle, satisfies foonathan RawAllocator
// =============================================================================

struct DevicePageAllocator {
  using is_stateful = std::true_type;

  size_t device_index;

  void *allocate_node(size_t size, size_t align) {
    return AllocatorRegistry::get()
        .page_impl(device_index)
        .allocate_node(size, align);
  }

  void deallocate_node(void *ptr, size_t size, size_t align) noexcept {
    AllocatorRegistry::get()
        .page_impl(device_index)
        .deallocate_node(ptr, size, align);
  }

  size_t max_node_size() const noexcept {
    return AllocatorRegistry::get().page_impl(device_index).max_node_size();
  }
};

// =============================================================================
// Type aliases for foonathan allocators backed by DevicePageAllocator
// =============================================================================

/// Pool for hot-path contiguous array allocations (singleton per device).
using DeviceMemoryPool =
    memory::memory_pool<memory::array_pool, DevicePageAllocator>;

/// Stack for scratch memory (user-created, marker-based bulk unwind).
using DeviceScratchMemory = memory::memory_stack<DevicePageAllocator>;

// =============================================================================
// AllocatorRegistry — lazy-init singleton, one slab + pool per device
//
// On first access for a given device_index, allocates the full slab via
// sycl::malloc_device on the default context. All subsequent allocations
// for that device are served from the pre-allocated slab — no further
// device memory allocations occur.
//
// Configuration (call before first allocation to override defaults):
//     AllocatorRegistry::get().set_page_budget(1024 * 1024 * 1024);  // 1 GiB
//     AllocatorRegistry::get().set_page_size(128 * 1024);            // 128 KiB
// =============================================================================

class AllocatorRegistry {
public:
  static AllocatorRegistry &get() {
    static AllocatorRegistry instance;
    return instance;
  }

  DevicePageAllocatorImpl &page_impl(size_t device_index) {
    ensure_initialized(device_index);
    return *page_impls_[device_index];
  }

  DeviceMemoryPool &pool(size_t device_index) {
    ensure_initialized(device_index);
    return *pools_[device_index];
  }

  // ── Configuration (call before first allocation) ─────────────────────

  void set_page_budget(size_t bytes) { page_budget_ = bytes; }
  void set_page_size(size_t bytes) { page_size_ = bytes; }
  void set_pool_node_size(size_t bytes) { pool_node_size_ = bytes; }
  void set_pool_initial_size(size_t bytes) { pool_initial_ = bytes; }

private:
  AllocatorRegistry() = default;

  size_t page_budget_ = 512 * 1024 * 1024; // 512 MiB per device
  size_t page_size_ = 64 * 1024;           // 64 KiB pages
  size_t pool_node_size_ = 4 * 1024;       // 4 KiB pool nodes
  size_t pool_initial_ = 1 * 1024 * 1024;  // 1 MiB initial pool block

  std::unordered_map<size_t, std::unique_ptr<DevicePageAllocatorImpl>>
      page_impls_;
  std::unordered_map<size_t, std::unique_ptr<DeviceMemoryPool>> pools_;

  void ensure_initialized(size_t device_index) {
    if (pools_.contains(device_index))
      return;

    auto devices = sycl::device::get_devices();
    assert(device_index < devices.size() && "device_index out of range");

    // Temporary queue on the default context — used only for slab allocation
    sycl::queue q{devices[device_index]};

    page_impls_[device_index] =
        std::make_unique<DevicePageAllocatorImpl>(q, page_budget_, page_size_);

    pools_[device_index] = std::make_unique<DeviceMemoryPool>(
        pool_node_size_, pool_initial_, DevicePageAllocator{device_index});
  }
};

// =============================================================================
// Queue factory — free function, default context
// =============================================================================

/// Create an in-order queue for the given device on the default context.
inline sycl::queue make_device_queue(size_t device_index) {
  auto devices = sycl::device::get_devices();
  assert(device_index < devices.size());
  return sycl::queue{devices[device_index], sycl::property::queue::in_order{}};
}

// =============================================================================
// Scratch memory factory
// =============================================================================

/// Create a scratch memory stack for temporary allocations.
/// The stack bump-allocates forward; use top() / unwind() for bulk reset.
///
/// Example:
///     auto scratch = make_scratch_memory(0, 4 * 1024 * 1024);
///     auto marker  = scratch.top();
///     auto* tmp    = scratch.allocate(1024 * sizeof(float), alignof(float));
///     // use tmp in kernels...
///     scratch.unwind(marker);  // O(1) bulk free
///
inline DeviceScratchMemory make_scratch_memory(size_t device_index,
                                               size_t budget) {
  return DeviceScratchMemory{budget, DevicePageAllocator{device_index}};
}
