#pragma once

// Internal: reaching the shared implementation behind a memory-object handle.
// Not installed; only the library's own translation units see this.

#include <memory>

#include "detail/arena_impl.hpp"
#include "detail/pool_impl.hpp"
#include "naga/slabkit/arena.hpp"
#include "naga/slabkit/pool.hpp"

namespace naga::slabkit::detail {

struct ArenaAccess {
    static const std::shared_ptr<ArenaImpl>& impl(const DeviceArena& a) noexcept {
        return a.impl_;
    }
    static std::shared_ptr<ArenaImpl>& impl(DeviceArena& a) noexcept { return a.impl_; }
};

struct PoolAccess {
    static const std::shared_ptr<PoolImpl>& impl(const DevicePool& p) noexcept {
        return p.impl_;
    }
    static std::shared_ptr<PoolImpl>& impl(DevicePool& p) noexcept { return p.impl_; }
};

}  // namespace naga::slabkit::detail
