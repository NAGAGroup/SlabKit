#pragma once

#include <array>
#include <cassert>
#include <concepts>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "slabkit/memory.hpp"

namespace slabkit {

// =============================================================================
// Forward declarations
// =============================================================================

template <typename T, int N>
struct DeviceView;

template <typename T, int N>
struct DeviceSubView;

template <typename T, int N>
class DeviceArray;

template <typename T, int N>
class MirroredArray;

// =============================================================================
// detail namespace — row-major and strided offset helpers
// =============================================================================

namespace detail {

// Compute row-major linear offset from a sycl::id<N> and a sycl::range<N> of extents.
// offset = id[0]*range[1]*range[2]*...*range[N-1] + id[1]*range[2]*...*range[N-1] + ... + id[N-1]
// Implemented via index_sequence fold.
template <int N, std::size_t... Is>
std::size_t row_major_offset_impl(sycl::id<N> id, sycl::range<N> range,
                                  std::index_sequence<Is...>) noexcept {
    std::size_t result = 0;
    std::size_t stride = 1;
    // Process dimensions right-to-left
    // Use a fold with comma operator via initializer_list trick
    // Dimensions: Is = 0..N-1, process in reverse
    ((void)(result += id[N - 1 - Is] * stride, stride *= range[N - 1 - Is]), ...);
    return result;
}

template <int N>
std::size_t row_major_offset(sycl::id<N> id, sycl::range<N> range) noexcept {
    return row_major_offset_impl(id, range, std::make_index_sequence<N>{});
}

// Compute strided linear offset from a sycl::id<N> and a sycl::range<N> of strides.
// offset = id[0]*strides[0] + id[1]*strides[1] + ... + id[N-1]*strides[N-1]
template <int N, std::size_t... Is>
std::size_t strided_offset_impl(sycl::id<N> id, sycl::range<N> strides,
                                std::index_sequence<Is...>) noexcept {
    return ((id[Is] * strides[Is]) + ...);
}

template <int N>
std::size_t strided_offset(sycl::id<N> id, sycl::range<N> strides) noexcept {
    return strided_offset_impl(id, strides, std::make_index_sequence<N>{});
}

}  // namespace detail

// =============================================================================
// DeviceViewLike concept
// =============================================================================

template <typename V, typename T, int N>
concept DeviceViewLike = requires(V v, sycl::id<N> id) {
    { v.data() } -> std::same_as<T*>;
    { v.get_range() } -> std::same_as<sycl::range<N>>;
    { v.size() } -> std::convertible_to<size_t>;
    { v.byte_size() } -> std::convertible_to<size_t>;
    { v[id] } -> std::same_as<T&>;
};

// =============================================================================
// DeviceView<T, N> — non-owning, contiguous, trivially copyable
// =============================================================================

template <typename T, int N>
struct DeviceView {
    T*             ptr_;
    sycl::range<N> range_;

    // Explicit construction from raw pointer + range
    DeviceView(T* ptr, sycl::range<N> range) noexcept : ptr_{ptr}, range_{range} {}

    // Implicit conversion to const view — DeviceView<T,N> → DeviceView<const T,N>
    template <typename U>
        requires std::same_as<T, const U>
    DeviceView(DeviceView<U, N> other) noexcept
        : ptr_{other.ptr_}, range_{other.range_} {}

    T*             data() const noexcept { return ptr_; }
    sycl::range<N> get_range() const noexcept { return range_; }
    size_t         size() const noexcept { return range_.size(); }
    size_t         byte_size() const noexcept { return size() * sizeof(T); }

    // Variadic indices — exactly N arguments, row-major
    T& operator()(auto... idx) const noexcept {
        static_assert(sizeof...(idx) == N);
        sycl::id<N> id{static_cast<size_t>(idx)...};
        return ptr_[detail::row_major_offset(id, range_)];
    }

    // sycl::id<N> access
    T& operator[](sycl::id<N> id) const noexcept {
        return ptr_[detail::row_major_offset(id, range_)];
    }
};

// =============================================================================
// DeviceSubView<T, N> — non-owning, strided, trivially copyable
// =============================================================================

template <typename T, int N>
struct DeviceSubView {
    T*             ptr_;
    sycl::range<N> range_;
    sycl::range<N> strides_;  // element strides, not byte strides

    // Explicit construction from raw pointer + range + strides
    DeviceSubView(T* ptr, sycl::range<N> range, sycl::range<N> strides) noexcept
        : ptr_{ptr}, range_{range}, strides_{strides} {}

    // Implicit conversion to const view — DeviceSubView<T,N> → DeviceSubView<const T,N>
    template <typename U>
        requires std::same_as<T, const U>
    DeviceSubView(DeviceSubView<U, N> other) noexcept
        : ptr_{other.ptr_}, range_{other.range_}, strides_{other.strides_} {}

    T*             data() const noexcept { return ptr_; }
    sycl::range<N> get_range() const noexcept { return range_; }
    size_t         size() const noexcept { return range_.size(); }
    size_t         byte_size() const noexcept { return size() * sizeof(T); }

    // Variadic indices — exactly N arguments, strided
    T& operator()(auto... idx) const noexcept {
        static_assert(sizeof...(idx) == N);
        sycl::id<N> id{static_cast<size_t>(idx)...};
        return ptr_[detail::strided_offset(id, strides_)];
    }

    // sycl::id<N> access
    T& operator[](sycl::id<N> id) const noexcept {
        return ptr_[detail::strided_offset(id, strides_)];
    }
};

// =============================================================================
// DeviceArray<T, N> — owning, RAII, copyable via shared_ptr
// =============================================================================

template <typename T, int N>
class DeviceArray {
public:
    // Construct from raw ptr + deleter (used by make_device_array factory)
    DeviceArray(std::shared_ptr<T> ptr, sycl::range<N> range, size_t device_index)
        : ptr_(std::move(ptr)), range_(range), device_index_(device_index) {}

    // Copy/move — defaulted, shared_ptr handles ref counting
    DeviceArray(const DeviceArray&)            = default;
    DeviceArray& operator=(const DeviceArray&) = default;
    DeviceArray(DeviceArray&&)                 = default;
    DeviceArray& operator=(DeviceArray&&)      = default;

    T*             data() const noexcept { return ptr_.get(); }
    sycl::range<N> get_range() const noexcept { return range_; }
    size_t         size() const noexcept { return range_.size(); }
    size_t         byte_size() const noexcept { return size() * sizeof(T); }
    size_t         device_index() const noexcept { return device_index_; }

    DeviceView<T, N> view() const noexcept {
        return DeviceView<T, N>{ptr_.get(), range_};
    }

    DeviceView<const T, N> const_view() const noexcept {
        return view();  // implicit conversion via DeviceView<const T,N>(DeviceView<T,N>)
    }

    DeviceSubView<T, N> subview(std::array<size_t, N> offsets,
                                sycl::range<N> sub_range) const {
        // Compute parent's packed row-major strides
        sycl::range<N> strides;
        strides[N - 1] = 1;
        for (int d = N - 2; d >= 0; --d)
            strides[d] = strides[d + 1] * range_[d + 1];

        // Compute linear offset for base pointer
        size_t linear_offset = 0;
        for (int d = 0; d < N; ++d)
            linear_offset += offsets[d] * strides[d];

        // Bounds check
        for (int d = 0; d < N; ++d)
            assert(offsets[d] + sub_range[d] <= range_[d] &&
                   "subview exceeds array bounds");

        return DeviceSubView<T, N>{ptr_.get() + linear_offset, sub_range, strides};
    }

    DeviceSubView<const T, N> const_subview(std::array<size_t, N> offsets,
                                             sycl::range<N> sub_range) const {
        return subview(offsets, sub_range);  // implicit conversion via DeviceSubView<const T,N>(DeviceSubView<T,N>)
    }

private:
    std::shared_ptr<T> ptr_;   // shared ownership — copyable
    sycl::range<N>     range_;
    size_t             device_index_;
};

// =============================================================================
// Factory function
// =============================================================================

template <typename T, int N>
DeviceArray<T, N> make_device_array(sycl::range<N> range, const sycl::device& dev) {
    // malloc_shared_device_ptr handles pool lookup and deleter setup
    auto ptr = malloc_shared_device_ptr<T>(dev, range.size());
    const size_t idx = AllocatorRegistry::get().device_index(dev);
    return DeviceArray<T, N>{std::move(ptr), range, idx};
}

// =============================================================================
// Transfer utilities — async, return sycl::event
// =============================================================================
//
// Tier 1 — contiguous (memcpy, no kernel):
//   copy_to_host  (DeviceArray,  T* dst, q, deps)
//   copy_to_host  (DeviceView,   T* dst, q, deps)
//   copy_to_host  (DeviceView<const T,N>, T* dst, q, deps)  [const view]
//   copy_to_device(T* src, DeviceArray,  q, deps)
//   copy_to_device(T* src, DeviceView,   q, deps)
//   copy          (DeviceArray src, DeviceArray dst, q, deps)
//
// Tier 2 — strided host transfers (staging buffer via DeviceScratchMemory):
//   copy_to_host  (DeviceSubView, T* dst, q, scratch, deps)
//   copy_to_host  (DeviceSubView<const T,N>, T* dst, q, scratch, deps)  [const view]
//   copy_to_device(T* src, DeviceSubView, q, scratch, deps)
//
// Tier 3 — strided device-to-device (parallel_for kernel, no staging):
//   copy(DeviceSubView src, DeviceSubView dst, q, deps)
//   copy(DeviceSubView src, DeviceArray   dst, q, deps)
//   copy(DeviceArray   src, DeviceSubView dst, q, deps)
//
// Tier 4 — blocking sync wrappers (.wait() internally, named sync_ not copy_):
//   sync_to_host(DeviceArray,   T* dst, q)
//   sync_to_host(DeviceView<const T,N>, T* dst, q)  [const view]
//   sync_to_host(DeviceSubView, T* dst, q, scratch)
//   sync_to_host(DeviceSubView<const T,N>, T* dst, q, scratch)  [const view]
//
// Host pointers (T*) must point to memory accessible from the host side.
// No hidden allocations — caller provides DeviceScratchMemory for strided transfers.
// Thread safety: same as sycl::queue — caller's responsibility.
//

// ─────────────────────────────────────────────────────────────────────────────
// Tier 1 — contiguous
// ─────────────────────────────────────────────────────────────────────────────

template <typename T, int N>
sycl::event copy_to_host(const DeviceArray<T, N>& src, T* dst,
                         sycl::queue& q, const std::vector<sycl::event>& deps = {}) {
    assert(dst != nullptr);
    return q.memcpy(dst, src.data(), src.byte_size(), deps);
}

template <typename T, int N>
sycl::event copy_to_host(DeviceView<T, N> src, T* dst,
                         sycl::queue& q, const std::vector<sycl::event>& deps = {}) {
    assert(dst != nullptr);
    return q.memcpy(dst, src.data(), src.byte_size(), deps);
}

template <typename T, int N>
sycl::event copy_to_host(DeviceView<const T, N> src, T* dst,
                         sycl::queue& q, const std::vector<sycl::event>& deps = {}) {
    assert(dst != nullptr);
    return q.memcpy(dst, src.data(), src.byte_size(), deps);
}

template <typename T, int N>
sycl::event copy_to_device(const T* src, DeviceArray<T, N>& dst,
                            sycl::queue& q, const std::vector<sycl::event>& deps = {}) {
    assert(src != nullptr);
    return q.memcpy(dst.data(), src, dst.byte_size(), deps);
}

template <typename T, int N>
sycl::event copy_to_device(const T* src, DeviceView<T, N> dst,
                            sycl::queue& q, const std::vector<sycl::event>& deps = {}) {
    assert(src != nullptr);
    return q.memcpy(dst.data(), src, dst.byte_size(), deps);
}

template <typename T, int N>
sycl::event copy(const DeviceArray<T, N>& src, DeviceArray<T, N>& dst,
                 sycl::queue& q, const std::vector<sycl::event>& deps = {}) {
    assert(src.size() == dst.size() && "array size mismatch");
    return q.memcpy(dst.data(), src.data(), src.byte_size(), deps);
}

// ─────────────────────────────────────────────────────────────────────────────
// Tier 2 — strided host transfers (staging buffer)
// ─────────────────────────────────────────────────────────────────────────────
//
// Gather (device strided → contiguous staging → host):
//   1. Allocate contiguous staging buffer from scratch memory (device USM)
//   2. parallel_for: staging[linear_i] = src[id]  (gather)
//   3. memcpy staging → dst (host)
//   Returns event for step 3 (chains deps through all steps).
//
// Scatter (host → contiguous staging → device strided):
//   1. Allocate contiguous staging buffer from scratch memory (device USM)
//   2. memcpy src (host) → staging
//   3. parallel_for: dst[id] = staging[linear_i]  (scatter)
//   Returns event for step 3.
//
// Staging memory is allocated from the caller-provided DeviceScratchMemory.
// Lifetime: caller must keep scratch alive until the returned event completes.
//

template <typename T, int N>
sycl::event copy_to_host(DeviceSubView<T, N> src, T* dst,
                         sycl::queue& q, DeviceScratchMemory& scratch,
                         const std::vector<sycl::event>& deps = {}) {
    assert(dst != nullptr);
    const size_t n = src.size();

    // Allocate contiguous staging buffer on device from scratch
    T* staging = static_cast<T*>(scratch.allocate(n * sizeof(T), alignof(T)));

    // Step 1: gather — parallel_for copies strided device → contiguous staging
    sycl::event gather = q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        // Capture by value — DeviceSubView and staging pointer are trivially copyable
        auto sv      = src;
        T*   stg_ptr = staging;
        h.parallel_for(src.get_range(), [=](sycl::id<N> id) {
            const size_t linear_i = detail::row_major_offset(id, sv.get_range());
            stg_ptr[linear_i]     = sv[id];
        });
    });

    // Step 2: memcpy staging → host
    return q.memcpy(dst, staging, n * sizeof(T), {gather});
}

template <typename T, int N>
sycl::event copy_to_host(DeviceSubView<const T, N> src, T* dst,
                         sycl::queue& q, DeviceScratchMemory& scratch,
                         const std::vector<sycl::event>& deps = {}) {
    assert(dst != nullptr);
    const size_t n = src.size();
    const T* staging = static_cast<const T*>(scratch.allocate(n * sizeof(T), alignof(T)));
    // gather: strided const device → contiguous staging
    sycl::event gather = q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        auto sv         = src;
        const T* stg_ptr = staging;
        h.parallel_for(src.get_range(), [=](sycl::id<N> id) {
            const size_t linear_i = detail::row_major_offset(id, sv.get_range());
            // staging is device memory — write via non-const cast (host can't deref)
            const_cast<T*>(stg_ptr)[linear_i] = sv[id];
        });
    });
    return q.memcpy(dst, staging, n * sizeof(T), {gather});
}

template <typename T, int N>
sycl::event copy_to_device(const T* src, DeviceSubView<T, N> dst,
                            sycl::queue& q, DeviceScratchMemory& scratch,
                            const std::vector<sycl::event>& deps = {}) {
    assert(src != nullptr);
    const size_t n = dst.size();

    // Allocate contiguous staging buffer on device from scratch
    T* staging = static_cast<T*>(scratch.allocate(n * sizeof(T), alignof(T)));

    // Step 1: memcpy host → staging
    sycl::event h2d = q.memcpy(staging, src, n * sizeof(T), deps);

    // Step 2: scatter — parallel_for copies contiguous staging → strided device
    return q.submit([&](sycl::handler& h) {
        h.depends_on(h2d);
        auto dv      = dst;
        const T* stg_ptr = staging;
        h.parallel_for(dst.get_range(), [=](sycl::id<N> id) {
            const size_t linear_i = detail::row_major_offset(id, dv.get_range());
            dv[id]                = stg_ptr[linear_i];
        });
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Tier 3 — strided device-to-device (parallel_for, no staging)
// ─────────────────────────────────────────────────────────────────────────────

// DeviceSubView → DeviceSubView
template <typename T, int N>
sycl::event copy(DeviceSubView<T, N> src, DeviceSubView<T, N> dst,
                 sycl::queue& q, const std::vector<sycl::event>& deps = {}) {
    assert(src.size() == dst.size() && "subview size mismatch");
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        auto sv = src;
        auto dv = dst;
        h.parallel_for(src.get_range(), [=](sycl::id<N> id) {
            dv[id] = sv[id];
        });
    });
}

// DeviceSubView → DeviceArray (strided src, contiguous dst)
template <typename T, int N>
sycl::event copy(DeviceSubView<T, N> src, DeviceArray<T, N>& dst,
                 sycl::queue& q, const std::vector<sycl::event>& deps = {}) {
    assert(src.size() == dst.size() && "size mismatch");
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        auto sv      = src;
        T*   dst_ptr = dst.data();
        h.parallel_for(src.get_range(), [=](sycl::id<N> id) {
            const size_t linear_i = detail::row_major_offset(id, sv.get_range());
            dst_ptr[linear_i]     = sv[id];
        });
    });
}

// DeviceArray → DeviceSubView (contiguous src, strided dst)
template <typename T, int N>
sycl::event copy(const DeviceArray<T, N>& src, DeviceSubView<T, N> dst,
                 sycl::queue& q, const std::vector<sycl::event>& deps = {}) {
    assert(src.size() == dst.size() && "size mismatch");
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const T* src_ptr = src.data();
        auto     dv      = dst;
        h.parallel_for(dst.get_range(), [=](sycl::id<N> id) {
            const size_t linear_i = detail::row_major_offset(id, dv.get_range());
            dv[id]                = src_ptr[linear_i];
        });
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Tier 4 — blocking sync wrappers
// ─────────────────────────────────────────────────────────────────────────────

template <typename T, int N>
void sync_to_host(const DeviceArray<T, N>& src, T* dst, sycl::queue& q) {
    copy_to_host(src, dst, q).wait();
}

template <typename T, int N>
void sync_to_host(DeviceView<const T, N> src, T* dst, sycl::queue& q) {
    copy_to_host(src, dst, q).wait();
}

template <typename T, int N>
void sync_to_host(DeviceSubView<T, N> src, T* dst,
                  sycl::queue& q, DeviceScratchMemory& scratch) {
    copy_to_host(src, dst, q, scratch).wait();
}

template <typename T, int N>
void sync_to_host(DeviceSubView<const T, N> src, T* dst,
                  sycl::queue& q, DeviceScratchMemory& scratch) {
    copy_to_host(src, dst, q, scratch).wait();
}

// =============================================================================
// MirroredArray<T, N> — paired device + host storage
// =============================================================================
//
// Device side is a DeviceArray (now copyable via shared_ptr).
// Host side is std::vector<T> (deep copy on copy — intentional, two independent
// host buffers sharing device memory would be a footgun).
//

template <typename T, int N>
class MirroredArray {
    DeviceArray<T, N> device_;
    std::vector<T>    host_;

public:
    MirroredArray(sycl::range<N> range, size_t device_index)
        : device_(make_device_array<T, N>(
              range, AllocatorRegistry::get().device_by_index(device_index))),
          host_(range.size()) {}

    // Copy: device side is shared (shared_ptr), host side is deep copied
    // Two independent host buffers sharing the same device allocation
    MirroredArray(const MirroredArray&)            = default;
    MirroredArray& operator=(const MirroredArray&) = default;
    MirroredArray(MirroredArray&&)                 = default;
    MirroredArray& operator=(MirroredArray&&)      = default;

    DeviceView<T, N> view() const noexcept { return device_.view(); }

    DeviceView<const T, N> const_view() const noexcept {
        return device_.const_view();
    }

    DeviceSubView<T, N> subview(std::array<size_t, N> offsets,
                                sycl::range<N> sub_range) const {
        return device_.subview(offsets, sub_range);
    }

    DeviceSubView<const T, N> const_subview(std::array<size_t, N> offsets,
                                            sycl::range<N> sub_range) const {
        return device_.const_subview(offsets, sub_range);
    }

    const DeviceArray<T, N>& device_array() const noexcept { return device_; }
    DeviceArray<T, N>&       device_array() noexcept { return device_; }

    std::span<T>       host_data() noexcept { return host_; }
    std::span<const T> host_data() const noexcept { return host_; }

    sycl::range<N> get_range() const noexcept { return device_.get_range(); }
    size_t         size() const noexcept { return device_.size(); }
    size_t         byte_size() const noexcept { return device_.byte_size(); }
    size_t         device_index() const noexcept { return device_.device_index(); }

    sycl::event sync_to_host(sycl::queue& q,
                             const std::vector<sycl::event>& deps = {}) {
        return copy_to_host(device_, host_.data(), q, deps);
    }

    sycl::event sync_to_device(sycl::queue& q,
                               const std::vector<sycl::event>& deps = {}) {
        return copy_to_device(host_.data(), device_, q, deps);
    }
};

}  // namespace slabkit
