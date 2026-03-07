#pragma once

#include <array>
#include <cassert>
#include <concepts>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include <sycl/sycl.hpp>

#include "device_allocator.hpp"

// =============================================================================
// Forward declarations
// =============================================================================

template <typename T, int N> struct DeviceView;
template <typename T, int N> struct DeviceSubView;
template <typename T, int N> class DeviceArray;
template <typename T, int N> class MirroredArray;

// =============================================================================
// DeviceViewLike concept
// =============================================================================

template <typename V, typename T, int N>
concept DeviceViewLike = requires(V v, size_t i, sycl::id<N> id) {
  { v.data() } -> std::same_as<T *>;
  { v.get_range() } -> std::same_as<sycl::range<N>>;
  { v.size() } -> std::convertible_to<size_t>;
  { v.byte_size() } -> std::convertible_to<size_t>;
  { v[id] } -> std::same_as<T &>;
};

// =============================================================================
// DeviceView — non-owning, contiguous, trivially copyable
// =============================================================================

template <typename T, int N>
  requires sycl::is_device_copyable_v<sycl::range<N>>
struct DeviceView {
  T *ptr_;
  sycl::range<N> range_;

  // ── Queries ──────────────────────────────────────────────────────────

  T *data() const { return ptr_; }
  sycl::range<N> get_range() const { return range_; }
  size_t size() const { return range_.size(); }
  size_t byte_size() const { return size() * sizeof(T); }

  // ── Element access — variadic indices ────────────────────────────────

  T &operator()(auto... idx) const {
    static_assert(sizeof...(idx) == N);
    if constexpr (N == 1) {
      auto [i] = std::tie(idx...);
      return ptr_[i];
    } else if constexpr (N == 2) {
      auto [i, j] = std::tie(idx...);
      return ptr_[i * range_[1] + j];
    } else if constexpr (N == 3) {
      auto [i, j, k] = std::tie(idx...);
      return ptr_[i * range_[1] * range_[2] + j * range_[2] + k];
    }
  }

  // ── Element access — sycl::id ────────────────────────────────────────

  T &operator[](sycl::id<N> id) const {
    if constexpr (N == 1) {
      return (*this)(id[0]);
    } else if constexpr (N == 2) {
      return (*this)(id[0], id[1]);
    } else if constexpr (N == 3) {
      return (*this)(id[0], id[1], id[2]);
    }
  }
};

// =============================================================================
// DeviceSubView — non-owning, strided, trivially copyable
// =============================================================================

template <typename T, int N>
  requires sycl::is_device_copyable_v<sycl::range<N>>
struct DeviceSubView {
  T *ptr_;
  sycl::range<N> range_;
  sycl::range<N> strides_;

  // ── Queries ──────────────────────────────────────────────────────────

  T *data() const { return ptr_; }
  sycl::range<N> get_range() const { return range_; }
  size_t size() const { return range_.size(); }
  size_t byte_size() const { return size() * sizeof(T); }

  // ── Element access — variadic indices ────────────────────────────────

  T &operator()(auto... idx) const {
    static_assert(sizeof...(idx) == N);
    if constexpr (N == 1) {
      auto [i] = std::tie(idx...);
      return ptr_[i * strides_[0]];
    } else if constexpr (N == 2) {
      auto [i, j] = std::tie(idx...);
      return ptr_[i * strides_[0] + j * strides_[1]];
    } else if constexpr (N == 3) {
      auto [i, j, k] = std::tie(idx...);
      return ptr_[i * strides_[0] + j * strides_[1] + k * strides_[2]];
    }
  }

  // ── Element access — sycl::id ────────────────────────────────────────

  T &operator[](sycl::id<N> id) const {
    if constexpr (N == 1) {
      return (*this)(id[0]);
    } else if constexpr (N == 2) {
      return (*this)(id[0], id[1]);
    } else if constexpr (N == 3) {
      return (*this)(id[0], id[1], id[2]);
    }
  }
};

// =============================================================================
// DeviceArray — owning, RAII, move-only
// =============================================================================

template <typename T, int N> class DeviceArray {
public:
  // ── Deleter ──────────────────────────────────────────────────────────

  struct DeviceDeleter {
    size_t byte_size;
    DeviceMemoryPool *pool;

    void operator()(T *ptr) const noexcept {
      pool->deallocate_node(ptr, byte_size, alignof(T));
    }
  };

private:
  std::unique_ptr<T, DeviceDeleter> ptr_;
  sycl::range<N> range_;
  size_t device_index_;

public:
  // ── Construction (via factory functions, see make_device_array) ───────

  DeviceArray(T *ptr, sycl::range<N> range, size_t device_index,
              DeviceDeleter deleter)
      : ptr_(ptr, deleter), range_(range), device_index_(device_index) {}

  // ── Queries ──────────────────────────────────────────────────────────

  T *data() const { return ptr_.get(); }
  sycl::range<N> get_range() const { return range_; }
  size_t size() const { return range_.size(); }
  size_t byte_size() const { return size() * sizeof(T); }
  size_t device_index() const { return device_index_; }

  // ── View creation ────────────────────────────────────────────────────

  DeviceView<T, N> view() const { return DeviceView<T, N>{ptr_.get(), range_}; }

  DeviceSubView<T, N> subview(std::array<size_t, N> offsets,
                              sycl::range<N> sub_range) const {
    // Compute parent's packed row-major strides
    sycl::range<N> strides;
    strides[N - 1] = 1;
    for (int d = N - 2; d >= 0; --d) {
      strides[d] = strides[d + 1] * range_[d + 1];
    }

    // Compute linear offset for base pointer
    size_t linear_offset = 0;
    for (int d = 0; d < N; ++d) {
      linear_offset += offsets[d] * strides[d];
    }

    // Bounds check
    for (int d = 0; d < N; ++d) {
      assert(offsets[d] + sub_range[d] <= range_[d] &&
             "subview exceeds array bounds");
    }

    return DeviceSubView<T, N>{ptr_.get() + linear_offset, sub_range, strides};
  }
};

// =============================================================================
// Factory function
// =============================================================================

template <typename T, int N>
DeviceArray<T, N> make_device_array(sycl::range<N> range, size_t device_index) {
  auto &pool = AllocatorRegistry::get().pool(device_index);
  size_t byte_size = range.size() * sizeof(T);
  T *ptr = static_cast<T *>(pool.allocate_node(byte_size, alignof(T)));

  return DeviceArray<T, N>{
      ptr, range, device_index,
      typename DeviceArray<T, N>::DeviceDeleter{byte_size, &pool}};
}

// =============================================================================
// Transfer utilities — async (return sycl::event)
// =============================================================================

// ── DeviceArray transfers ────────────────────────────────────────────────

template <typename T, int N>
sycl::event copy_to_host(const DeviceArray<T, N> &src, std::span<T> dst,
                         sycl::queue &q,
                         const std::vector<sycl::event> &deps = {}) {
  assert(dst.size() >= src.size() && "destination span too small");
  return q.memcpy(dst.data(), src.data(), src.byte_size(), deps);
}

template <typename T, int N>
sycl::event copy_to_device(std::span<const T> src, DeviceArray<T, N> &dst,
                           sycl::queue &q,
                           const std::vector<sycl::event> &deps = {}) {
  assert(src.size() >= dst.size() && "source span too small");
  return q.memcpy(dst.data(), src.data(), dst.byte_size(), deps);
}

template <typename T, int N>
sycl::event copy(const DeviceArray<T, N> &src, DeviceArray<T, N> &dst,
                 sycl::queue &q, const std::vector<sycl::event> &deps = {}) {
  assert(src.size() == dst.size() && "array size mismatch");
  return q.memcpy(dst.data(), src.data(), src.byte_size(), deps);
}

// ── DeviceView transfers (contiguous) ────────────────────────────────────

template <typename T, int N>
sycl::event copy_to_host(DeviceView<T, N> src, std::span<T> dst, sycl::queue &q,
                         const std::vector<sycl::event> &deps = {}) {
  assert(dst.size() >= src.size() && "destination span too small");
  return q.memcpy(dst.data(), src.data(), src.byte_size(), deps);
}

template <typename T, int N>
sycl::event copy_to_device(std::span<const T> src, DeviceView<T, N> dst,
                           sycl::queue &q,
                           const std::vector<sycl::event> &deps = {}) {
  assert(src.size() >= dst.size() && "source span too small");
  return q.memcpy(dst.data(), src.data(), dst.byte_size(), deps);
}

// ── DeviceSubView transfers (strided — requires gather/scatter kernel) ───

template <typename T, int N>
sycl::event copy_to_host(DeviceSubView<T, N> src, std::span<T> dst,
                         sycl::queue &q,
                         const std::vector<sycl::event> &deps = {}) {
  assert(dst.size() >= src.size() && "destination span too small");

  // TODO: allocate staging buffer from arena/pool
  // T* staging = arena.alloc<T>(src.size());

  // Pack strided data into contiguous staging buffer
  auto pack_event = q.submit([&](sycl::handler &cgh) {
    cgh.depends_on(deps);
    cgh.parallel_for(src.get_range(), [=](sycl::id<N> id) {
      // Compute linear index into contiguous staging buffer
      size_t linear = 0;
      size_t stride = 1;
      for (int d = N - 1; d >= 0; --d) {
        linear += id[d] * stride;
        stride *= src.get_range()[d];
      }
      // staging[linear] = src[id];
    });
  });

  // Then memcpy staging → host
  // return q.memcpy(dst.data(), staging, src.byte_size(), {pack_event});

  return pack_event; // placeholder
}

template <typename T, int N>
sycl::event copy_to_device(std::span<const T> src, DeviceSubView<T, N> dst,
                           sycl::queue &q,
                           const std::vector<sycl::event> &deps = {}) {
  assert(src.size() >= dst.size() && "source span too small");

  // TODO: memcpy host → staging, then scatter kernel into strided dst
  // Mirror of copy_to_host logic

  return sycl::event{}; // placeholder
}

// =============================================================================
// Blocking convenience
// =============================================================================

template <typename T, int N>
std::vector<T> sync_to_host(const DeviceArray<T, N> &src, sycl::queue &q) {
  std::vector<T> result(src.size());
  q.memcpy(result.data(), src.data(), src.byte_size()).wait();
  return result;
}

// =============================================================================
// MirroredArray — paired device + host storage
// =============================================================================

template <typename T, int N> class MirroredArray {
  DeviceArray<T, N> device_;
  std::vector<T> host_;

public:
  MirroredArray(sycl::range<N> range, size_t device_index)
      : device_(make_device_array<T, N>(range, device_index)),
        host_(range.size()) {}

  // ── Device-side access ───────────────────────────────────────────────

  DeviceView<T, N> view() const { return device_.view(); }

  DeviceSubView<T, N> subview(std::array<size_t, N> offsets,
                              sycl::range<N> sub_range) const {
    return device_.subview(offsets, sub_range);
  }

  const DeviceArray<T, N> &device_array() const { return device_; }
  DeviceArray<T, N> &device_array() { return device_; }

  // ── Host-side access ─────────────────────────────────────────────────

  std::span<T> host_data() { return host_; }
  std::span<const T> host_data() const { return host_; }

  // ── Queries ──────────────────────────────────────────────────────────

  sycl::range<N> get_range() const { return device_.get_range(); }
  size_t size() const { return device_.size(); }
  size_t byte_size() const { return device_.byte_size(); }
  size_t device_index() const { return device_.device_index(); }

  // ── Sync ─────────────────────────────────────────────────────────────

  sycl::event sync_to_host(sycl::queue &q,
                           const std::vector<sycl::event> &deps = {}) {
    return copy_to_host(device_, host_data(), q, deps);
  }

  sycl::event sync_to_device(sycl::queue &q,
                             const std::vector<sycl::event> &deps = {}) {
    return copy_to_device(host_data(), device_, q, deps);
  }
};
