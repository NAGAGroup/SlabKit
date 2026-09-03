// =============================================================================
// site.cpp — where bytes come from, and where their initialization runs
// =============================================================================
//
// The whole element-type-independent half of the storage and object families
// compiles here: a copy is a copy and a fill is a repeated pattern, neither of
// which needs to know what T is. Only the placement-new path for a
// non-trivially-copyable type is genuinely per-type device code, and that stays
// a template in make.hpp.

#include "naga/slabkit/detail/core.hpp"

#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#include <sycl/sycl.hpp>

#include "detail/access.hpp"
#include "detail/arena_impl.hpp"
#include "detail/os_alloc.hpp"
#include "detail/pool_impl.hpp"

namespace naga::slabkit::detail {
namespace {

/// Arena placement: honors any alignment, charging the padding to the arena.
/// Returns null when what remains cannot satisfy the request, and for a
/// zero-byte request.
unique_usm_ptr<std::byte[]> arena_bytes(std::size_t bytes, std::size_t align,
                                        DeviceArena& arena) {
    const std::shared_ptr<ArenaImpl>& impl = ArenaAccess::impl(arena);

    // A null pointer still carries the arena route, so the deleter is always
    // well formed and the handle never has to invent a context.
    void* p = (bytes == 0) ? nullptr : impl->allocate(bytes, align);
    return unique_usm_ptr<std::byte[]>{static_cast<std::byte*>(p), UsmDeleter{ArenaFree{impl}}};
}

/// Pool placement: allocations are node_size-aligned and rounded up to whole
/// nodes.
/// \throws std::invalid_argument if align exceeds the pool's node size.
unique_usm_ptr<std::byte[]> pool_bytes(std::size_t bytes, std::size_t align, DevicePool& pool) {
    const std::shared_ptr<PoolImpl>& impl = PoolAccess::impl(pool);

    if (align > impl->node_size())
        throw std::invalid_argument{
            "naga::slabkit: a pool cannot honor an alignment larger than its node size"};

    if (bytes == 0 || impl->node_size() == 0)
        return unique_usm_ptr<std::byte[]>{nullptr, UsmDeleter{PoolFree{impl, 0}}};

    const std::size_t nodes = align_up(bytes, impl->node_size()) / impl->node_size();
    void*             p     = impl->allocate(nodes);
    return unique_usm_ptr<std::byte[]>{static_cast<std::byte*>(p),
                                       UsmDeleter{PoolFree{impl, p ? nodes : 0}}};
}

void kind_must_match(kind requested, kind actual, const char* what) {
    if (requested != actual)
        throw std::invalid_argument{
            std::string{"naga::slabkit: the allocation name does not match the "} + what +
            "'s kind"};
}

/// A pattern of identical bytes -- which is what value-initialization of
/// anything zeroable looks like -- is a memset, and needs no kernel.
bool uniform_pattern(const void* pattern, std::size_t size) {
    const auto* p = static_cast<const std::byte*>(pattern);
    for (std::size_t i = 1; i < size; ++i)
        if (p[i] != p[0]) return false;
    return true;
}

}  // namespace

// -----------------------------------------------------------------------------
// The six placements
// -----------------------------------------------------------------------------

Site Site::at(kind k, const queue& q, const sycl::device& dev) {
    Site s;
    s.kind_   = k;
    s.from_   = from::os;
    s.stream_ = q;
    s.dev_    = dev;
    // The queue resolves the device's backend context itself, and throws for a
    // device it does not hold: the context discipline cannot be violated
    // through this form.
    s.ctx_ = q.get_context().context_of(dev);
    return s;
}

Site Site::at(kind k, const queue& q) { return at(k, q, q.get_device()); }

Site Site::at(kind k, const sycl::queue& sq) {
    Site s;
    s.kind_ = k;
    s.from_ = from::os;
    s.dev_  = sq.get_device();
    s.ctx_  = sq.get_context();
    return s;
}

Site Site::at(kind k, const sycl::device& dev, const sycl::context& ctx) {
    Site s;
    s.kind_ = k;
    s.from_ = from::os;
    s.dev_  = dev;
    s.ctx_  = ctx;
    return s;
}

Site Site::at(kind k, const sycl::context& ctx) {
    Site s;
    s.kind_ = k;
    s.from_ = from::os;
    s.ctx_  = ctx;

    // Host memory belongs to a context, not a device -- but a construction
    // kernel still has to run somewhere, so remember one of the context's own.
    const auto devices = ctx.get_devices();
    if (!devices.empty()) s.dev_ = devices.front();
    return s;
}

Site Site::at(kind k, const queue& q, DeviceArena& a) {
    kind_must_match(k, a.get_kind(), "arena");
    Site s;
    s.kind_   = k;
    s.from_   = from::arena;
    s.arena_  = &a;
    s.stream_ = q;
    s.dev_    = a.device();
    s.ctx_    = a.context();
    return s;
}

Site Site::at(kind k, const queue& q, DevicePool& p) {
    kind_must_match(k, p.get_kind(), "pool");
    Site s;
    s.kind_   = k;
    s.from_   = from::pool;
    s.pool_   = &p;
    s.stream_ = q;
    s.dev_    = p.device();
    s.ctx_    = p.context();
    return s;
}

Site Site::at(kind k, DeviceArena& a) {
    kind_must_match(k, a.get_kind(), "arena");
    Site s;
    s.kind_  = k;
    s.from_  = from::arena;
    s.arena_ = &a;
    s.dev_   = a.device();
    s.ctx_   = a.context();
    return s;
}

Site Site::at(kind k, DevicePool& p) {
    kind_must_match(k, p.get_kind(), "pool");
    Site s;
    s.kind_ = k;
    s.from_ = from::pool;
    s.pool_ = &p;
    s.dev_  = p.device();
    s.ctx_  = p.context();
    return s;
}

// -----------------------------------------------------------------------------
// Storage
// -----------------------------------------------------------------------------

unique_usm_ptr<std::byte[]> Site::storage(std::size_t bytes, std::size_t req_align,
                                          std::size_t                nat_align,
                                          const sycl::property_list& props) const {
    switch (from_) {
        case from::arena:
            // An arena places addresses itself, so it can honour the type's own
            // alignment even when the caller asked for nothing.
            return arena_bytes(bytes, req_align ? req_align : nat_align, *arena_);
        case from::pool: return pool_bytes(bytes, req_align ? req_align : nat_align, *pool_);
        case from::os: break;
    }
    return os_allocate(kind_, req_align, bytes, device(), *ctx_, props);
}

// -----------------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------------

queue& Site::stream() const {
    if (!stream_)
        throw std::invalid_argument{
            "naga::slabkit: this placement carries no stream to initialize on"};
    return *stream_;
}

const sycl::device& Site::device() const {
    if (!dev_)
        throw std::invalid_argument{"naga::slabkit: this placement names no device"};
    return *dev_;
}

sycl::queue Site::interop_queue() const { return sycl::queue{*ctx_, device()}; }

void Site::write(void* dst, const void* src, std::size_t bytes) const {
    if (dst == nullptr || src == nullptr || bytes == 0) return;

    // Host-accessible memory is written by the host. No queue, no kernel, and
    // no take-turns problem: nothing has been enqueued against this allocation
    // yet -- it did not exist a moment ago.
    if (host_accessible(kind_)) {
        std::memcpy(dst, src, bytes);
        return;
    }

    if (stream_)
        stream_->submit(device(), [&](sycl::handler& h) { h.memcpy(dst, src, bytes); }).wait();
    else
        interop_queue().memcpy(dst, src, bytes).wait();
}

void Site::fill(void* dst, const void* pattern, std::size_t pattern_size,
                std::size_t count) const {
    if (dst == nullptr || pattern == nullptr || pattern_size == 0 || count == 0) return;

    if (host_accessible(kind_)) {
        auto* out = static_cast<std::byte*>(dst);
        for (std::size_t i = 0; i < count; ++i)
            std::memcpy(out + i * pattern_size, pattern, pattern_size);
        return;
    }

    const auto* pat = static_cast<const std::byte*>(pattern);

    if (uniform_pattern(pattern, pattern_size)) {
        const int   value = static_cast<int>(std::to_integer<unsigned char>(pat[0]));
        const auto  bytes = pattern_size * count;
        if (stream_)
            stream_->submit(device(), [&](sycl::handler& h) { h.memset(dst, value, bytes); }).wait();
        else
            interop_queue().memset(dst, value, bytes).wait();
        return;
    }

    // Otherwise stage the pattern in device memory and replicate it with one
    // kernel. The staging allocation is an ordinary slabkit handle, so it is
    // released even if the kernel throws.
    auto stage = os_allocate(kind::device, 0, pattern_size, device(), *ctx_, {});
    if (!stage) throw std::bad_alloc{};

    const std::byte*  src   = stage.get();
    auto*             out   = static_cast<std::byte*>(dst);
    const std::size_t psize = pattern_size;
    const std::size_t total = pattern_size * count;

    if (stream_) {
        queue& q = *stream_;
        q.submit(device(), [&](sycl::handler& h) { h.memcpy(stage.get(), pattern, psize); }).wait();
        q.parallel_for(device(), sycl::range<1>{total},
                       [=](sycl::id<1> idx) { out[idx[0]] = src[idx[0] % psize]; })
            .wait();
    } else {
        sycl::queue q = interop_queue();
        q.memcpy(stage.get(), pattern, psize).wait();
        q.parallel_for(sycl::range<1>{total},
                       [=](sycl::id<1> idx) { out[idx[0]] = src[idx[0] % psize]; })
            .wait();
    }
}

// -----------------------------------------------------------------------------
// make_* obtains storage or throws
// -----------------------------------------------------------------------------

unique_usm_ptr<std::byte[]> require(unique_usm_ptr<std::byte[]>&& h, std::size_t bytes) {
    if (!h && bytes > 0) throw std::bad_alloc{};
    return std::move(h);
}

}  // namespace naga::slabkit::detail
