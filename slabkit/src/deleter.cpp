// =============================================================================
// deleter.cpp — the one place slabkit frees memory
// =============================================================================

#include <type_traits>
#include <variant>

#include "detail/arena_impl.hpp"
#include "detail/pool_impl.hpp"
#include "naga/slabkit/handles.hpp"

namespace naga::slabkit {

void UsmDeleter::operator()(void* p) const noexcept {
    if (p == nullptr) return;
    std::visit(
        [p](const auto& route) {
            using R = std::decay_t<decltype(route)>;
            if constexpr (std::is_same_v<R, OsFree>) {
                sycl::free(p, route.ctx);
            } else if constexpr (std::is_same_v<R, PoolFree>) {
                route.impl->release(p, route.nodes);
            } else {
                // ArenaFree: an arena has no per-allocation free. Holding the
                // implementation alive until now WAS the release action, and
                // the shared_ptr in this deleter is about to drop -- which may
                // be what finally frees the slab.
                static_assert(std::is_same_v<R, ArenaFree>);
            }
        },
        how);
}

}  // namespace naga::slabkit
