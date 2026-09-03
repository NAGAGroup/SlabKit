#pragma once

// =============================================================================
// naga/slabkit/make_shared.hpp — OBJECTS: make_shared_{device,shared,host}
// =============================================================================
//
// The identical forms over the identical placements, returning shared_usm_ptr.
// Every one of them is the make_unique_* of the same kind followed by the
// std::shared_ptr conversion, which adopts the deleter and erases its type --
// so the mechanism, the type gate and the blocking behaviour are exactly those
// documented in make.hpp, with no second implementation to keep in step.
//
// Note the two senses of "shared" that meet here: make_SHARED_shared is a
// shared_ptr to shared-KIND memory. Both spellings are borrowed (std's and
// SYCL's) and neither is slabkit's to rename.

#include <memory>
#include <utility>

#include "naga/slabkit/handles.hpp"
#include "naga/slabkit/make.hpp"

namespace naga::slabkit {

template <class T, class... A>
    requires detail::makeable<T>
shared_usm_ptr<T> make_shared_device(A&&... a) {
    return shared_usm_ptr<T>{make_unique_device<T>(std::forward<A>(a)...)};
}

template <class T, class... A>
    requires detail::makeable<T>
shared_usm_ptr<T> make_shared_shared(A&&... a) {
    return shared_usm_ptr<T>{make_unique_shared<T>(std::forward<A>(a)...)};
}

template <class T, class... A>
    requires detail::makeable<T> && detail::placement_kind_ok<kind::host, A...>
shared_usm_ptr<T> make_shared_host(A&&... a) {
    return shared_usm_ptr<T>{make_unique_host<T>(std::forward<A>(a)...)};
}

template <class T, class... A>
    requires detail::makeable<T>
shared_usm_ptr<T> make_shared_device_for_overwrite(A&&... a) {
    return shared_usm_ptr<T>{make_unique_device_for_overwrite<T>(std::forward<A>(a)...)};
}

template <class T, class... A>
    requires detail::makeable<T>
shared_usm_ptr<T> make_shared_shared_for_overwrite(A&&... a) {
    return shared_usm_ptr<T>{make_unique_shared_for_overwrite<T>(std::forward<A>(a)...)};
}

template <class T, class... A>
    requires detail::makeable<T> && detail::placement_kind_ok<kind::host, A...>
shared_usm_ptr<T> make_shared_host_for_overwrite(A&&... a) {
    return shared_usm_ptr<T>{make_unique_host_for_overwrite<T>(std::forward<A>(a)...)};
}

}  // namespace naga::slabkit
