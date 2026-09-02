#pragma once

// =============================================================================
// naga/slabkit/handles.hpp — the one handle type and its three deleters
// =============================================================================
//
// Every allocation slabkit hands out is owned by the same handle type. What
// differs is how the memory is released, and that is the deleter's whole job:
//
//   OsFree     the memory came from the SYCL runtime  -> sycl::free(p, ctx)
//   PoolFree   the memory came from a DevicePool      -> impl->release(p, nodes)
//   ArenaFree  the memory came from a DeviceArena     -> nothing to do
//
// OsFree is the only one that frees memory; the other two account for it. The
// pool and arena deleters each hold a share of the memory object's
// implementation, so an outstanding allocation keeps its slab alive by
// construction: the slab dies with the last of {handles to the object,
// allocations from it}. There is no lifetime contract for a caller to
// remember.
//
// sycl::free is therefore reached from exactly one place in the library, and
// no class in slabkit has a destructor of its own.
//
// The deleter NEVER runs a destructor. Objects made by the make_* family are
// required to be trivially destructible, so there is nothing to run.
//
// NOTE ON LAYERING: DeviceDeleter::operator() is declared here and defined in
// detail/deleter_impl.hpp, which is included once the pool and arena
// implementations are complete types. Every public slabkit header pulls that
// definition in, so the two-step is invisible to callers.

#include <cstddef>
#include <memory>
#include <variant>

#include <sycl/sycl.hpp>

namespace naga::slabkit {

namespace detail {
class ArenaImpl;
class PoolImpl;
}  // namespace detail

/// Memory obtained straight from the SYCL runtime. Holds the context it must
/// be freed with -- never a queue's context for some other device, because
/// sycl::free resolves its allocator from the context it is handed.
struct OsFree {
    sycl::context ctx;
};

/// Memory carved out of a DevicePool. `nodes` is the slot count to return.
struct PoolFree {
    std::shared_ptr<detail::PoolImpl> impl;
    std::size_t                       nodes;
};

/// Memory carved out of a DeviceArena. An arena has no per-allocation free;
/// holding the implementation alive is the entire release action.
struct ArenaFree {
    std::shared_ptr<detail::ArenaImpl> impl;
};

/// The deleter holds exactly what release needs, and nothing else.
struct DeviceDeleter {
    std::variant<OsFree, PoolFree, ArenaFree> how;

    void operator()(void* p) const noexcept;  // defined in detail/deleter_impl.hpp
};

template <class T>
using unique_device_ptr = std::unique_ptr<T, DeviceDeleter>;

/// shared_ptr erases the deleter, so no slabkit-specific type appears here.
template <class T>
using shared_device_ptr = std::shared_ptr<T>;

}  // namespace naga::slabkit
