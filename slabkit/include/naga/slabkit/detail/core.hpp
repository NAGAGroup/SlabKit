#pragma once

// =============================================================================
// naga/slabkit/detail/core.hpp — the compiled core, declared
// =============================================================================
//
// slabkit is a compiled library. Everything that does not depend on the element
// type lives in a translation unit and is declared here.
//
// The centre of it is `Site`: one runtime value that answers "where do these
// bytes come from, and where does their initialization run". Six placements
// collapse into it, so the public families need one function per (kind, form)
// rather than one per (kind, form, placement) -- 18 functions instead of 108.
//
//   (queue, device)               the primary form; the queue resolves the
//                                 device's backend context itself, so the
//                                 context discipline cannot be violated
//   (queue)                       the stream's default device
//   (sycl::queue)                 interop
//   (device, context)             interop; the context must be that device's
//                                 backend default
//   (context)                     interop, host kind only
//   (queue, arena) / (queue, pool)  the object's kind and device; the queue is
//                                 there to initialize on, not to place with
//   (arena) / (pool)              storage only, which needs no stream
//
// A Site knows its kind at construction, so `malloc_shared(n, device_arena)` is
// std::invalid_argument rather than a silent reinterpretation.

#include <cstddef>
#include <optional>
#include <type_traits>

#include <sycl/sycl.hpp>

#include "naga/slabkit/arena.hpp"
#include "naga/slabkit/handles.hpp"
#include "naga/slabkit/kinds.hpp"
#include "naga/slabkit/pool.hpp"
#include "naga/slabkit/queue.hpp"

namespace naga::slabkit::detail {

class Site {
public:
    // ---- the six placements, one factory each ------------------------------
    static Site at(kind k, const queue& q, const sycl::device& dev);
    static Site at(kind k, const queue& q);
    static Site at(kind k, const sycl::queue& sq);
    static Site at(kind k, const sycl::device& dev, const sycl::context& ctx);
    static Site at(kind k, const sycl::context& ctx);
    static Site at(kind k, const queue& q, DeviceArena& a);
    static Site at(kind k, const queue& q, DevicePool& p);
    static Site at(kind k, DeviceArena& a);
    static Site at(kind k, DevicePool& p);

    kind get_kind() const noexcept { return kind_; }

    /// Bytes for this placement.
    ///
    /// `req_align` is what the caller ASKED for: 0 means "as sycl::malloc_*",
    /// which requests nothing of the runtime. `nat_align` is the type's own
    /// alignment, which an arena and a pool honour even when nothing was asked,
    /// because they place addresses themselves and can.
    ///
    /// Null propagates; nothing here throws except a pool asked for an
    /// alignment beyond its node size.
    unique_usm_ptr<std::byte[]> storage(std::size_t bytes, std::size_t req_align,
                                        std::size_t                nat_align,
                                        const sycl::property_list& props) const;

    // ---- initialization, blocking ------------------------------------------
    /// Host-accessible kinds are written by the host; device memory goes
    /// through the stream.
    void write(void* dst, const void* src, std::size_t bytes) const;
    void fill(void* dst, const void* pattern, std::size_t pattern_size,
              std::size_t count) const;

    // ---- the one genuinely per-type path needs a stream ---------------------
    bool                has_stream() const noexcept { return stream_.has_value(); }
    queue&              stream() const;         // a handle; the copy is the stream
    const sycl::device& device() const;
    sycl::queue         interop_queue() const;  // only for the interop placements

private:
    enum class from { os, arena, pool };

    kind                        kind_ = kind::device;
    from                        from_ = from::os;
    mutable std::optional<queue> stream_;
    std::optional<sycl::device> dev_;
    std::optional<sycl::context> ctx_;
    DeviceArena*                arena_ = nullptr;
    DevicePool*                 pool_  = nullptr;
};

// -----------------------------------------------------------------------------
// Handle plumbing
// -----------------------------------------------------------------------------

/// Re-type a byte handle, carrying its deleter across. The pointer is
/// untouched; only the static type changes.
template <class T>
unique_usm_ptr<T> retype(unique_usm_ptr<std::byte[]>&& h) noexcept {
    UsmDeleter d = h.get_deleter();
    return unique_usm_ptr<T>{reinterpret_cast<std::remove_extent_t<T>*>(h.release()), std::move(d)};
}

/// make_* obtains storage or throws, mirroring std::make_unique. A zero-byte
/// request is allowed to come back empty; anything larger must not.
/// \throws std::bad_alloc
unique_usm_ptr<std::byte[]> require(unique_usm_ptr<std::byte[]>&& h, std::size_t bytes);

}  // namespace naga::slabkit::detail
