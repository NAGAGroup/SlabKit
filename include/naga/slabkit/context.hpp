#pragma once

// =============================================================================
// naga/slabkit/context.hpp — the device set and its backend contexts
// =============================================================================
//
// The set of devices a program places work on, each device's backend default
// sycl::context, and the capability answers for that set computed once at
// construction. Immutable. A light handle: copies share state.
//
// CONTEXTS ARE THE PLATFORM'S. Every USM allocation belongs to a SYCL context
// and any access must use that context (SYCL 2020 4.8.2); a copy between two
// devices is legal only within one context. Each platform has a default
// context spanning all its devices (4.6.2), and queue{device}'s context IS it.
// slabkit never constructs a sycl::context: it retrieves the platform default
// and holds one per backend. A context spanning two platforms is illegal and
// AdaptiveCpp cannot diagnose it -- the process terminates on the eventual
// cross-context free -- which is why obtaining a context is confined to this
// class's constructors.
//
// The answers for this instance's own devices -- INCLUDING the same-backend
// pair measurements, which cost real device work -- are computed once here and
// read from the cache by select_devices, the queue's memory tasks and
// usm_vector.
// There is no process-wide state and no probe function.

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <span>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/slabkit/capabilities.hpp"
#include "naga/slabkit/kinds.hpp"

namespace naga::slabkit {

namespace detail {
class ContextState;
struct ContextAccess;
}  // namespace detail

class context {
public:
    // ---- explicit devices ---------------------------------------------------
    explicit context(std::vector<sycl::device> devices);
    explicit context(const sycl::device& dev);

    // ---- every root device of the listed platforms ---------------------------
    explicit context(std::vector<sycl::platform> platforms);
    explicit context(const sycl::platform& platform);

    // ---- every available device (optionally within `platforms`) satisfying ALL
    //      of `required`. A pair aspect in `required` means: with every other
    //      selected device of its backend.
    explicit context(std::initializer_list<aspect> required);
    context(std::vector<sycl::platform> platforms, std::initializer_list<aspect> required);
    context(const sycl::platform& platform, std::initializer_list<aspect> required);

    /// Selection order: platform order, then device order.
    std::span<const sycl::device> devices() const;

    /// Deduplicated, first-seen order.
    std::span<const backend_id> backends() const;

    bool       contains(const sycl::device& dev) const;
    backend_id backend_of(const sycl::device& dev) const;  // throws if !contains

    /// The backend default context. \throws std::invalid_argument if !contains.
    const sycl::context& context_of(const sycl::device& dev) const;
    const sycl::context& context_of(const backend_id& b) const;

    /// Devices of this context satisfying A, read from the answers cached at
    /// construction.
    ///   device aspect -> std::vector<sycl::device>
    ///   pair aspect   -> device_groups: one entry per connected component of
    ///                    the pair relation, within one backend.
    template <aspect A>
    typename selection<A>::type select_devices() const {
        if constexpr (is_pair_aspect(A))
            return select_groups(A);
        else
            return select_list(A);
    }

    // ---- SYCL's pointer queries, tried against every backend context held here
    /// \throws sycl::exception errc::invalid for a pointer none of them knows.
    kind         get_pointer_type(const void* p) const;
    sycl::device get_pointer_device(const void* p) const;

    // ---- machine facts, independent of any instance --------------------------
    template <aspect A>
    static answer satisfies(const sycl::device& dev) {
        static_assert(!is_pair_aspect(A),
                      "slabkit: a pair aspect needs two devices; call the two-device overload");
        return detail::satisfies_device(A, dev);
    }

    template <aspect A>
    static answer satisfies(const sycl::device& a, const sycl::device& b) {
        static_assert(is_pair_aspect(A),
                      "slabkit: a device aspect takes one device; call the one-device overload");
        return detail::satisfies_pair(A, a, b);
    }

    /// Best route first, never empty.
    static std::vector<route> copy_routes(kind src_kind, const sycl::device& src, kind dst_kind,
                                          const sycl::device& dst);

private:
    std::vector<sycl::device> select_list(aspect a) const;
    device_groups             select_groups(aspect a) const;

    /// The index of the backend whose answer for `p` is strongest, or npos.
    std::size_t strongest_answer(const void* p, kind& out) const;

    friend struct detail::ContextAccess;
    std::shared_ptr<const detail::ContextState> state_;
};

}  // namespace naga::slabkit
