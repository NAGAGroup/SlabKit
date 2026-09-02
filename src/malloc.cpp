// =============================================================================
// malloc.cpp — the byte-level core of the malloc family
// =============================================================================
//
// The typed malloc_device<T>/aligned_alloc_device<T> overloads are one-line
// wrappers over these: they turn a count into a byte size and re-type the
// handle. All of the actual behaviour lives here.

#include <cstddef>
#include <stdexcept>
#include <utility>

#include "detail/access.hpp"
#include "detail/arena_impl.hpp"
#include "detail/os_alloc.hpp"
#include "detail/pool_impl.hpp"
#include "naga/slabkit/arena.hpp"
#include "naga/slabkit/detail/core.hpp"
#include "naga/slabkit/pool.hpp"

namespace naga::slabkit::detail {

unique_device_ptr<std::byte[]> malloc_bytes(std::size_t bytes, const sycl::queue& q,
                                            const sycl::property_list& props) {
    return os_allocate(0, bytes, q.get_device(), q.get_context(), props);
}

unique_device_ptr<std::byte[]> malloc_bytes(std::size_t bytes, const sycl::device& dev,
                                            const sycl::context&       ctx,
                                            const sycl::property_list& props) {
    return os_allocate(0, bytes, dev, ctx, props);
}

unique_device_ptr<std::byte[]> aligned_alloc_bytes(std::size_t align, std::size_t bytes,
                                                   const sycl::queue&         q,
                                                   const sycl::property_list& props) {
    return os_allocate(align, bytes, q.get_device(), q.get_context(), props);
}

unique_device_ptr<std::byte[]> aligned_alloc_bytes(std::size_t align, std::size_t bytes,
                                                   const sycl::device&        dev,
                                                   const sycl::context&       ctx,
                                                   const sycl::property_list& props) {
    return os_allocate(align, bytes, dev, ctx, props);
}

unique_device_ptr<std::byte[]> arena_bytes(std::size_t bytes, std::size_t align,
                                           DeviceArena& arena) {
    const std::shared_ptr<ArenaImpl>& impl = ArenaAccess::impl(arena);

    // A null pointer still carries the arena route, so the deleter is always
    // well formed and the handle never has to invent a context.
    void* p = (bytes == 0) ? nullptr : impl->allocate(bytes, align);
    return unique_device_ptr<std::byte[]>{static_cast<std::byte*>(p),
                                          DeviceDeleter{ArenaFree{impl}}};
}

unique_device_ptr<std::byte[]> pool_bytes(std::size_t bytes, std::size_t align,
                                          DevicePool& pool) {
    const std::shared_ptr<PoolImpl>& impl = PoolAccess::impl(pool);

    if (align > impl->node_size())
        throw std::invalid_argument{
            "naga::slabkit: a pool cannot honor an alignment larger than its node size"};

    if (bytes == 0 || impl->node_size() == 0)
        return unique_device_ptr<std::byte[]>{nullptr, DeviceDeleter{PoolFree{impl, 0}}};

    const std::size_t nodes = align_up(bytes, impl->node_size()) / impl->node_size();
    void*             p     = impl->allocate(nodes);
    return unique_device_ptr<std::byte[]>{static_cast<std::byte*>(p),
                                          DeviceDeleter{PoolFree{impl, p ? nodes : 0}}};
}

unique_device_ptr<std::byte[]> require(unique_device_ptr<std::byte[]>&& h, std::size_t bytes) {
    if (!h && bytes > 0) throw std::bad_alloc{};
    return std::move(h);
}

}  // namespace naga::slabkit::detail
