#pragma once

// =============================================================================
// naga/slabkit/detail/os_alloc.hpp — the library's only call into sycl::malloc_*
// =============================================================================
//
// Everything that needs bytes from the SYCL runtime comes through here: the
// slabless malloc_* forms, and the slabs that DeviceArena and DevicePool own.
// That is what makes an arena's slab an ordinary slabkit allocation and keeps
// sycl::free confined to OsFree.
//
// Zero-element requests are passed straight through. slabkit adds no rule of
// its own: whether sycl::malloc_device(0, ...) yields null is
// implementation-defined, and it genuinely differs between AdaptiveCpp's
// backends (the OpenMP host allocator rounds the request up to a 32-byte
// alignment and returns a live pointer).

#include <cstddef>

#include <sycl/sycl.hpp>

#include "naga/slabkit/handles.hpp"

namespace naga::slabkit::detail {

/// Raw allocation. `align == 0` selects sycl::malloc_device; any other value
/// selects sycl::aligned_alloc_device.
///
/// Be aware that a backend may ignore `align` entirely -- AdaptiveCpp's CUDA
/// allocator forwards to cudaMalloc and drops it, so nothing above CUDA's own
/// 256-byte guarantee is assured. Code that MUST have a stronger alignment
/// (the pool's node grid) aligns the base itself rather than trusting this.
inline std::byte* os_allocate_raw(std::size_t align, std::size_t bytes,
                                  const sycl::device& dev, const sycl::context& ctx,
                                  const sycl::property_list& props) {
    void* p = (align == 0) ? sycl::malloc_device(bytes, dev, ctx, props)
                           : sycl::aligned_alloc_device(align, bytes, dev, ctx, props);
    return static_cast<std::byte*>(p);
}

/// Owning form: the byte handle every slab is held by.
inline unique_device_ptr<std::byte[]> os_allocate(std::size_t align, std::size_t bytes,
                                                  const sycl::device&        dev,
                                                  const sycl::context&       ctx,
                                                  const sycl::property_list& props = {}) {
    return unique_device_ptr<std::byte[]>{os_allocate_raw(align, bytes, dev, ctx, props),
                                          DeviceDeleter{OsFree{ctx}}};
}

/// Round `value` up to a multiple of `align` (a power of two).
inline constexpr std::size_t align_up(std::size_t value, std::size_t align) noexcept {
    return (value + align - 1) & ~(align - 1);
}

inline constexpr bool is_power_of_two(std::size_t v) noexcept {
    return v != 0 && (v & (v - 1)) == 0;
}

}  // namespace naga::slabkit::detail
