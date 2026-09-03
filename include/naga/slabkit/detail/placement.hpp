#pragma once

// =============================================================================
// naga/slabkit/detail/placement.hpp — reading a placement off a call
// =============================================================================
//
// The public families are spelled the way SYCL and the standard library spell
// theirs: the placement comes first (after a count, where there is one) and
// whatever the form needs comes after. Six placements times six object forms
// times three kinds is 108 functions if each is written out; it is 18 if the
// placement is read off the front of the argument list instead.
//
// That is all this header does. `placement_arity` says how many leading
// arguments name the placement -- two for (queue, device), (queue, arena),
// (queue, pool) and (device, context), one for everything else -- and
// `with_placement` splits the pack there, builds the Site, and hands the rest
// to the form's own code.
//
// The KIND is a template parameter, not a value, so the host kind's rule that a
// device argument is a COMPILE error falls out of overload resolution:
// site_for<kind::host>(queue, device) has no viable candidate.

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <sycl/sycl.hpp>

#include "naga/slabkit/detail/core.hpp"

namespace naga::slabkit::detail {

// -----------------------------------------------------------------------------
// Which placement is being named
// -----------------------------------------------------------------------------

template <class A0, class... Rest>
constexpr std::size_t placement_arity() {
    using First = std::decay_t<A0>;

    if constexpr (std::is_same_v<First, sycl::device>) {
        return 2;  // (device, context)
    } else if constexpr (std::is_same_v<First, queue>) {
        if constexpr (sizeof...(Rest) == 0) {
            return 1;  // (queue)
        } else {
            using Second = std::decay_t<std::tuple_element_t<0, std::tuple<Rest...>>>;
            if constexpr (std::is_same_v<Second, sycl::device> ||
                          std::is_same_v<Second, DeviceArena> ||
                          std::is_same_v<Second, DevicePool>)
                return 2;
            else
                return 1;
        }
    } else {
        return 1;  // (sycl::queue), (context), (arena), (pool)
    }
}

/// No host placement names a device: host memory belongs to a context and to
/// no device in it. Stating that as a CONSTRAINT rather than leaving it to a
/// substitution failure deep inside the body is what makes the rejection
/// observable -- `requires { malloc_host<int>(n, q, dev); }` answers false
/// rather than hard-erroring on instantiation.
template <class... P>
inline constexpr bool names_no_device =
    !((std::is_same_v<std::decay_t<P>, sycl::device>) || ...);

/// The kinds that may name one.
template <kind K, class... P>
concept placement_kind_ok = (K != kind::host) || names_no_device<P...>;

// -----------------------------------------------------------------------------
// Building the Site, with the kind known at compile time
// -----------------------------------------------------------------------------

template <kind K>
Site site_for(const queue& q, const sycl::device& dev)
    requires(K != kind::host)
{
    return Site::at(K, q, dev);
}

template <kind K>
Site site_for(const queue& q) {
    return Site::at(K, q);
}

template <kind K>
Site site_for(const sycl::queue& sq) {
    return Site::at(K, sq);
}

template <kind K>
Site site_for(const sycl::device& dev, const sycl::context& ctx)
    requires(K != kind::host)
{
    return Site::at(K, dev, ctx);
}

/// The host kind's interop placement: a context and no device, because host
/// memory has none.
template <kind K>
Site site_for(const sycl::context& ctx)
    requires(K == kind::host)
{
    return Site::at(K, ctx);
}

template <kind K>
Site site_for(const queue& q, DeviceArena& a) {
    return Site::at(K, q, a);
}

template <kind K>
Site site_for(const queue& q, DevicePool& p) {
    return Site::at(K, q, p);
}

template <kind K>
Site site_for(DeviceArena& a) {
    return Site::at(K, a);
}

template <kind K>
Site site_for(DevicePool& p) {
    return Site::at(K, p);
}

// -----------------------------------------------------------------------------
// Splitting the call
// -----------------------------------------------------------------------------

template <kind K, std::size_t NP, class Form, class Tuple, std::size_t... P, std::size_t... R>
decltype(auto) split_placement(Form&& form, Tuple& args, std::index_sequence<P...>,
                               std::index_sequence<R...>) {
    Site s = site_for<K>(std::get<P>(args)...);
    return form(s, std::get<NP + R>(args)...);
}

/// `form(const Site&, rest...)` is the form's own code: allocate, and
/// initialize however this form initializes.
template <kind K, class Form, class... A>
decltype(auto) with_placement(Form&& form, A&&... a) {
    constexpr std::size_t np = placement_arity<A...>();
    static_assert(np <= sizeof...(A), "slabkit: this placement needs more arguments than given");

    auto args = std::forward_as_tuple(std::forward<A>(a)...);
    return split_placement<K, np>(std::forward<Form>(form), args, std::make_index_sequence<np>{},
                                  std::make_index_sequence<sizeof...(A) - np>{});
}

}  // namespace naga::slabkit::detail
