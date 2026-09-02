#pragma once

// =============================================================================
// naga/slabkit/detail/core.hpp — the compiled core, declared
// =============================================================================
//
// slabkit is a compiled library. Everything that does not depend on the
// element type lives in a translation unit and is declared here: byte-level
// allocation for each placement, the initialization primitives, and the rule
// for choosing the queue an initialization runs on.
//
// The typed malloc_* and make_* functions in the public headers are thin
// wrappers over these -- they compute a byte count and an alignment, call in,
// and re-type the handle. Nothing here is a template, so consumers compile
// almost none of slabkit.

#include <cstddef>

#include <sycl/sycl.hpp>

#include "naga/slabkit/handles.hpp"

namespace naga::slabkit {

class DeviceArena;
class DevicePool;

namespace detail {

// -----------------------------------------------------------------------------
// Storage — the malloc_* family's byte core. Null propagates; nothing throws.
// -----------------------------------------------------------------------------

/// Mirrors sycl::malloc_device: no alignment is requested of the runtime, and
/// a zero-byte request is passed straight through to it.
unique_device_ptr<std::byte[]> malloc_bytes(std::size_t bytes, const sycl::queue& q,
                                            const sycl::property_list& props);
unique_device_ptr<std::byte[]> malloc_bytes(std::size_t bytes, const sycl::device& dev,
                                            const sycl::context&       ctx,
                                            const sycl::property_list& props);

/// Mirrors sycl::aligned_alloc_device. Note that a backend may ignore the
/// alignment (AdaptiveCpp's CUDA allocator does).
unique_device_ptr<std::byte[]> aligned_alloc_bytes(std::size_t align, std::size_t bytes,
                                                   const sycl::queue&         q,
                                                   const sycl::property_list& props);
unique_device_ptr<std::byte[]> aligned_alloc_bytes(std::size_t align, std::size_t bytes,
                                                   const sycl::device&        dev,
                                                   const sycl::context&       ctx,
                                                   const sycl::property_list& props);

/// Arena placement: honors any alignment, charging the padding to the arena.
/// Returns null when the remaining space cannot satisfy the request, and for a
/// zero-byte request.
unique_device_ptr<std::byte[]> arena_bytes(std::size_t bytes, std::size_t align,
                                           DeviceArena& arena);

/// Pool placement: allocations are node_size-aligned and rounded up to whole
/// nodes. Returns null when the pool is exhausted or too fragmented, and for a
/// zero-byte request.
/// \throws std::invalid_argument if align exceeds the pool's node size.
unique_device_ptr<std::byte[]> pool_bytes(std::size_t bytes, std::size_t align,
                                          DevicePool& pool);

// -----------------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------------

/// The queue an initialization for `dev` must run on. When `dev` is the
/// queue's own device this is `q` itself; otherwise it is a fresh queue on
/// (ctx, dev).
///
/// Why not retarget `q`: AdaptiveCpp's handler::memcpy classifies its pointers
/// against the QUEUE's context, so a pointer belonging to another device's
/// context is misread as host memory. A queue built on the allocation's own
/// context cannot make that mistake, and AdaptiveCpp queues are interface
/// objects that create no backend resources.
sycl::queue init_queue(const sycl::queue& q, const sycl::device& dev,
                       const sycl::context& ctx);

/// Copy `bytes` from host memory into device memory. Blocks.
void copy_from_host(void* dst, const void* src, std::size_t bytes, sycl::queue q);

/// Write `count` copies of a `pattern_size`-byte pattern into `dst`. Blocks.
void fill_pattern(void* dst, const void* pattern, std::size_t pattern_size,
                  std::size_t count, sycl::queue q);

// -----------------------------------------------------------------------------
// Handle plumbing
// -----------------------------------------------------------------------------

/// Re-type a byte handle, carrying its deleter across. The pointer is
/// untouched; only the static type changes.
template <class T>
unique_device_ptr<T> retype(unique_device_ptr<std::byte[]>&& h) noexcept {
    DeviceDeleter d = h.get_deleter();
    return unique_device_ptr<T>{reinterpret_cast<std::remove_extent_t<T>*>(h.release()),
                                std::move(d)};
}

/// make_* obtains storage or throws, mirroring std::make_unique. A zero-byte
/// request is allowed to come back empty; anything larger must not.
/// \throws std::bad_alloc
unique_device_ptr<std::byte[]> require(unique_device_ptr<std::byte[]>&& h, std::size_t bytes);

}  // namespace detail
}  // namespace naga::slabkit
