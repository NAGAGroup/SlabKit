// =============================================================================
// pool.cpp — DevicePool, the handle over detail::PoolImpl
// =============================================================================

#include "naga/slabkit/pool.hpp"

#include <memory>

#include "detail/access.hpp"
#include "detail/pool_impl.hpp"
#include "naga/slabkit/queue.hpp"

namespace naga::slabkit {

DevicePool::DevicePool(const queue& q, const sycl::device& dev, kind k, std::size_t bytes,
                       std::size_t node_size, const sycl::property_list& props)
    : impl_{std::make_shared<detail::PoolImpl>(dev, q.get_context().context_of(dev), k, bytes,
                                               node_size, props)} {}

DevicePool::DevicePool(const queue& q, kind k, std::size_t bytes, std::size_t node_size,
                       const sycl::property_list& props)
    : DevicePool{q, q.get_device(), k, bytes, node_size, props} {}

DevicePool::DevicePool(const sycl::queue& q, kind k, std::size_t bytes, std::size_t node_size,
                       const sycl::property_list& props)
    : impl_{std::make_shared<detail::PoolImpl>(q.get_device(), q.get_context(), k, bytes,
                                               node_size, props)} {}

DevicePool::DevicePool(const sycl::device& dev, const sycl::context& ctx, kind k,
                       std::size_t bytes, std::size_t node_size,
                       const sycl::property_list& props)
    : impl_{std::make_shared<detail::PoolImpl>(dev, ctx, k, bytes, node_size, props)} {}

kind        DevicePool::get_kind() const { return impl_->get_kind(); }
std::size_t DevicePool::node_size() const { return impl_->node_size(); }
std::size_t DevicePool::slot_count() const { return impl_->slot_count(); }
std::size_t DevicePool::free_count() const { return impl_->free_count(); }
std::size_t DevicePool::outstanding() const { return impl_->outstanding(); }

const sycl::device&  DevicePool::device() const { return impl_->device(); }
const sycl::context& DevicePool::context() const { return impl_->context(); }

}  // namespace naga::slabkit
