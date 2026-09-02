#pragma once

// =============================================================================
// naga/slabkit/make_shared.hpp — OBJECTS: make_shared_device
// =============================================================================
//
// The identical six forms over the identical four placements, returning
// shared_device_ptr. Every one of them is make_unique_device followed by the
// std::shared_ptr conversion, which adopts the deleter and erases its type --
// so the mechanism, the type gate and the blocking behaviour are exactly those
// documented in make.hpp, with no second implementation to keep in step.

#include <cstddef>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>

#include <sycl/sycl.hpp>

#include "naga/slabkit/arena.hpp"
#include "naga/slabkit/handles.hpp"
#include "naga/slabkit/make.hpp"
#include "naga/slabkit/pool.hpp"

namespace naga::slabkit {

// -----------------------------------------------------------------------------
// Single object
// -----------------------------------------------------------------------------

template <class T, class... Args>
    requires(!std::is_array_v<T> && detail::device_object<T> &&
             !detail::starts_with_placement<Args...>)
shared_device_ptr<T> make_shared_device(sycl::queue q, Args&&... args) {
    return shared_device_ptr<T>{make_unique_device<T>(std::move(q), std::forward<Args>(args)...)};
}

template <class T, class... Args>
    requires(!std::is_array_v<T> && detail::device_object<T>)
shared_device_ptr<T> make_shared_device(sycl::queue q, const sycl::device& dev,
                                        const sycl::context& ctx, Args&&... args) {
    return shared_device_ptr<T>{
        make_unique_device<T>(std::move(q), dev, ctx, std::forward<Args>(args)...)};
}

template <class T, class... Args>
    requires(!std::is_array_v<T> && detail::device_object<T>)
shared_device_ptr<T> make_shared_device(sycl::queue q, DeviceArena& arena, Args&&... args) {
    return shared_device_ptr<T>{
        make_unique_device<T>(std::move(q), arena, std::forward<Args>(args)...)};
}

template <class T, class... Args>
    requires(!std::is_array_v<T> && detail::device_object<T>)
shared_device_ptr<T> make_shared_device(sycl::queue q, DevicePool& pool, Args&&... args) {
    return shared_device_ptr<T>{
        make_unique_device<T>(std::move(q), pool, std::forward<Args>(args)...)};
}

// -----------------------------------------------------------------------------
// Single object, storage only
// -----------------------------------------------------------------------------

template <class T>
    requires(!std::is_array_v<T> && detail::device_object<T>)
shared_device_ptr<T> make_shared_device_for_overwrite(sycl::queue q) {
    return shared_device_ptr<T>{make_unique_device_for_overwrite<T>(std::move(q))};
}

template <class T>
    requires(!std::is_array_v<T> && detail::device_object<T>)
shared_device_ptr<T> make_shared_device_for_overwrite(sycl::queue q, const sycl::device& dev,
                                                      const sycl::context& ctx) {
    return shared_device_ptr<T>{make_unique_device_for_overwrite<T>(std::move(q), dev, ctx)};
}

template <class T>
    requires(!std::is_array_v<T> && detail::device_object<T>)
shared_device_ptr<T> make_shared_device_for_overwrite(sycl::queue q, DeviceArena& arena) {
    return shared_device_ptr<T>{make_unique_device_for_overwrite<T>(std::move(q), arena)};
}

template <class T>
    requires(!std::is_array_v<T> && detail::device_object<T>)
shared_device_ptr<T> make_shared_device_for_overwrite(sycl::queue q, DevicePool& pool) {
    return shared_device_ptr<T>{make_unique_device_for_overwrite<T>(std::move(q), pool)};
}

// -----------------------------------------------------------------------------
// Array, value-initialized
// -----------------------------------------------------------------------------

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
shared_device_ptr<TArr> make_shared_device(sycl::queue q, std::size_t n) {
    return shared_device_ptr<TArr>{make_unique_device<TArr>(std::move(q), n)};
}

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
shared_device_ptr<TArr> make_shared_device(sycl::queue q, const sycl::device& dev,
                                           const sycl::context& ctx, std::size_t n) {
    return shared_device_ptr<TArr>{make_unique_device<TArr>(std::move(q), dev, ctx, n)};
}

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
shared_device_ptr<TArr> make_shared_device(sycl::queue q, DeviceArena& arena, std::size_t n) {
    return shared_device_ptr<TArr>{make_unique_device<TArr>(std::move(q), arena, n)};
}

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
shared_device_ptr<TArr> make_shared_device(sycl::queue q, DevicePool& pool, std::size_t n) {
    return shared_device_ptr<TArr>{make_unique_device<TArr>(std::move(q), pool, n)};
}

// -----------------------------------------------------------------------------
// Array, storage only
// -----------------------------------------------------------------------------

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
shared_device_ptr<TArr> make_shared_device_for_overwrite(sycl::queue q, std::size_t n) {
    return shared_device_ptr<TArr>{make_unique_device_for_overwrite<TArr>(std::move(q), n)};
}

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
shared_device_ptr<TArr> make_shared_device_for_overwrite(sycl::queue q, const sycl::device& dev,
                                                         const sycl::context& ctx,
                                                         std::size_t          n) {
    return shared_device_ptr<TArr>{
        make_unique_device_for_overwrite<TArr>(std::move(q), dev, ctx, n)};
}

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
shared_device_ptr<TArr> make_shared_device_for_overwrite(sycl::queue q, DeviceArena& arena,
                                                         std::size_t n) {
    return shared_device_ptr<TArr>{make_unique_device_for_overwrite<TArr>(std::move(q), arena, n)};
}

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
shared_device_ptr<TArr> make_shared_device_for_overwrite(sycl::queue q, DevicePool& pool,
                                                         std::size_t n) {
    return shared_device_ptr<TArr>{make_unique_device_for_overwrite<TArr>(std::move(q), pool, n)};
}

// -----------------------------------------------------------------------------
// Array, filled with a value
// -----------------------------------------------------------------------------

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
shared_device_ptr<TArr> make_shared_device(sycl::queue q, std::size_t n,
                                           const detail::elem_t<TArr>& value) {
    return shared_device_ptr<TArr>{make_unique_device<TArr>(std::move(q), n, value)};
}

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
shared_device_ptr<TArr> make_shared_device(sycl::queue q, const sycl::device& dev,
                                           const sycl::context& ctx, std::size_t n,
                                           const detail::elem_t<TArr>& value) {
    return shared_device_ptr<TArr>{make_unique_device<TArr>(std::move(q), dev, ctx, n, value)};
}

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
shared_device_ptr<TArr> make_shared_device(sycl::queue q, DeviceArena& arena, std::size_t n,
                                           const detail::elem_t<TArr>& value) {
    return shared_device_ptr<TArr>{make_unique_device<TArr>(std::move(q), arena, n, value)};
}

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
shared_device_ptr<TArr> make_shared_device(sycl::queue q, DevicePool& pool, std::size_t n,
                                           const detail::elem_t<TArr>& value) {
    return shared_device_ptr<TArr>{make_unique_device<TArr>(std::move(q), pool, n, value)};
}

// -----------------------------------------------------------------------------
// Array, copied from host data
// -----------------------------------------------------------------------------

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
shared_device_ptr<TArr> make_shared_device(sycl::queue                           q,
                                           std::span<const detail::elem_t<TArr>> src) {
    return shared_device_ptr<TArr>{make_unique_device<TArr>(std::move(q), src)};
}

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
shared_device_ptr<TArr> make_shared_device(sycl::queue q, const sycl::device& dev,
                                           const sycl::context&                  ctx,
                                           std::span<const detail::elem_t<TArr>> src) {
    return shared_device_ptr<TArr>{make_unique_device<TArr>(std::move(q), dev, ctx, src)};
}

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
shared_device_ptr<TArr> make_shared_device(sycl::queue q, DeviceArena& arena,
                                           std::span<const detail::elem_t<TArr>> src) {
    return shared_device_ptr<TArr>{make_unique_device<TArr>(std::move(q), arena, src)};
}

template <class TArr>
    requires(std::is_unbounded_array_v<TArr> && detail::device_element<detail::elem_t<TArr>>)
shared_device_ptr<TArr> make_shared_device(sycl::queue q, DevicePool& pool,
                                           std::span<const detail::elem_t<TArr>> src) {
    return shared_device_ptr<TArr>{make_unique_device<TArr>(std::move(q), pool, src)};
}

}  // namespace naga::slabkit
