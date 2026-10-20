// =============================================================================
// arena.cpp — DeviceArena, the handle over detail::ArenaImpl
// =============================================================================

#include "naga/slabkit/arena.hpp"

#include <memory>

#include "detail/access.hpp"
#include "detail/arena_impl.hpp"
#include "naga/slabkit/queue.hpp"

namespace naga::slabkit {

DeviceArena::DeviceArena(const queue& q, const sycl::device& dev, kind k, std::size_t bytes,
                         const sycl::property_list& props)
    : impl_{std::make_shared<detail::ArenaImpl>(dev, q.get_context().context_of(dev), k, bytes,
                                                props)} {}

DeviceArena::DeviceArena(const queue& q, kind k, std::size_t bytes,
                         const sycl::property_list& props)
    : DeviceArena{q, q.get_device(), k, bytes, props} {}

DeviceArena::DeviceArena(const sycl::queue& q, kind k, std::size_t bytes,
                         const sycl::property_list& props)
    : impl_{std::make_shared<detail::ArenaImpl>(q.get_device(), q.get_context(), k, bytes,
                                                props)} {}

DeviceArena::DeviceArena(const sycl::device& dev, const sycl::context& ctx, kind k,
                         std::size_t bytes, const sycl::property_list& props)
    : impl_{std::make_shared<detail::ArenaImpl>(dev, ctx, k, bytes, props)} {}

kind        DeviceArena::get_kind() const { return impl_->get_kind(); }
std::size_t DeviceArena::capacity() const { return impl_->capacity(); }
std::size_t DeviceArena::used() const { return impl_->used(); }
std::size_t DeviceArena::remaining() const { return impl_->remaining(); }

const sycl::device&  DeviceArena::device() const { return impl_->device(); }
const sycl::context& DeviceArena::context() const { return impl_->context(); }

}  // namespace naga::slabkit
