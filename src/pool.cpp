// =============================================================================
// pool.cpp — DevicePool, the handle over detail::PoolImpl
// =============================================================================

#include "naga/slabkit/pool.hpp"

#include <memory>

#include "detail/access.hpp"
#include "detail/pool_impl.hpp"

namespace naga::slabkit {

DevicePool::DevicePool(const sycl::queue& q, std::size_t bytes, std::size_t node_size,
                       const sycl::property_list& props)
    : impl_{std::make_shared<detail::PoolImpl>(q.get_device(), q.get_context(), bytes, node_size,
                                               props)} {}

DevicePool::DevicePool(const sycl::device& dev, const sycl::context& ctx, std::size_t bytes,
                       std::size_t node_size, const sycl::property_list& props)
    : impl_{std::make_shared<detail::PoolImpl>(dev, ctx, bytes, node_size, props)} {}

std::size_t DevicePool::node_size() const { return impl_->node_size(); }
std::size_t DevicePool::slot_count() const { return impl_->slot_count(); }
std::size_t DevicePool::free_count() const { return impl_->free_count(); }
std::size_t DevicePool::outstanding() const { return impl_->outstanding(); }

const sycl::device&  DevicePool::device() const { return impl_->device(); }
const sycl::context& DevicePool::context() const { return impl_->context(); }

}  // namespace naga::slabkit
