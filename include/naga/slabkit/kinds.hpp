#pragma once

// =============================================================================
// naga/slabkit/kinds.hpp — the three USM kinds
// =============================================================================
//
// SYCL's three USM allocation kinds, spelled as slabkit spells them. The name
// of an allocation function IS its kind (malloc_device, malloc_shared,
// malloc_host), so this enum never appears as a parameter of the storage or
// object families. It is what a memory object is constructed with, what
// context::get_pointer_type answers, and what a route is computed from.

#include <sycl/sycl.hpp>

namespace naga::slabkit {

enum class kind { device, shared, host };

/// True when host code may dereference an allocation of this kind. Device
/// memory may not be touched from the host; shared and host memory may, after
/// the work using it has completed (slabkit's take-turns rule).
constexpr bool host_accessible(kind k) noexcept { return k != kind::device; }

/// True when an allocation of this kind lives on its device's backend, which
/// is what makes a direct device-to-device copy possible.
constexpr bool device_resident(kind k) noexcept { return k != kind::host; }

/// The SYCL spelling of a kind, for the pointer queries.
constexpr sycl::usm::alloc to_usm_alloc(kind k) noexcept {
    switch (k) {
        case kind::device: return sycl::usm::alloc::device;
        case kind::shared: return sycl::usm::alloc::shared;
        default: return sycl::usm::alloc::host;
    }
}

}  // namespace naga::slabkit
