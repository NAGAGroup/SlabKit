#pragma once

// =============================================================================
// naga/slabkit/arena.hpp — DeviceArena
// =============================================================================
//
// A bump region over one USM slab. DeviceArena is a light handle: copying it
// is cheap and copies refer to the same slab. It exposes no allocation method
// -- the malloc_* and make_* free functions are the only way to allocate, and
// an arena is simply one of the placements they accept.
//
// There is no per-allocation free and no rewind. The slab is released when the
// last of {handles to this arena, allocations made from it} goes away, so an
// allocation can safely outlive every DeviceArena that named its slab.

#include <cstddef>
#include <memory>

#include <sycl/sycl.hpp>

#include "naga/slabkit/handles.hpp"

namespace naga::slabkit {

class DeviceArena;

namespace detail {
struct ArenaAccess;
}  // namespace detail

class DeviceArena {
public:
    /// The queue's own device and context.
    DeviceArena(const sycl::queue& q, std::size_t bytes, const sycl::property_list& props = {});

    /// That device, freed with that context.
    DeviceArena(const sycl::device& dev, const sycl::context& ctx, std::size_t bytes,
                const sycl::property_list& props = {});

    std::size_t capacity() const;

    /// Bytes handed out, INCLUDING alignment padding the bump had to skip.
    std::size_t used() const;
    std::size_t remaining() const;

    const sycl::device&  device() const;
    const sycl::context& context() const;

private:
    friend struct detail::ArenaAccess;
    std::shared_ptr<detail::ArenaImpl> impl_;
};

}  // namespace naga::slabkit
