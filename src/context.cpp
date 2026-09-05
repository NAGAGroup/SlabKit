// =============================================================================
// context.cpp — building the device set and answering for it
// =============================================================================
//
// Construction does three things and then never changes again:
//
//   1. resolves the device list (given, or filtered by aspects),
//   2. retrieves ONE backend default context per platform present,
//   3. computes every capability answer for those devices, including the
//      same-backend pair measurements, and caches them.
//
// Step 2 is the whole reason this class exists. A sycl::context is never
// constructed here: platform::khr_get_default_context() returns the platform
// default, which is bit-for-bit the context sycl::queue{device} builds itself
// (queue's own get_default_context is `context{default_context_tag_t{},
// dev.get_platform()}`, and khr_get_default_context is the same expression).
// The test suite asserts that equality rather than trusting this comment.
//
// Step 3 costs device work when a pair aspect is documented `unknown` --
// paying it once, here, is why nothing downstream ever measures.

#include "naga/slabkit/context.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "detail/context_state.hpp"

namespace naga::slabkit {
namespace {

using detail::ContextState;

std::vector<sycl::device> root_devices_of(const std::vector<sycl::platform>& platforms) {
    std::vector<sycl::device> out;
    for (const auto& p : platforms)
        for (const auto& d : p.get_devices()) out.push_back(d);
    return out;
}

std::vector<sycl::device> all_root_devices() {
    return root_devices_of(sycl::platform::get_platforms());
}

/// Filter by the single-device aspects of `required`. `unknown` is not
/// selected: a capability the machine will not confirm is not one to place
/// work on.
std::vector<sycl::device> filter_by_device_aspects(std::vector<sycl::device>        candidates,
                                                   const std::vector<aspect>& required) {
    std::vector<sycl::device> out;
    for (const auto& d : candidates) {
        bool ok = true;
        for (aspect a : required) {
            if (is_pair_aspect(a)) continue;
            if (detail::satisfies_device(a, d) != answer::yes) {
                ok = false;
                break;
            }
        }
        if (ok) out.push_back(d);
    }
    return out;
}

/// Reduce the selection until every same-backend pair within it satisfies every
/// pair aspect of `required`.
///
/// The general problem is a maximum clique, which is not what a device selector
/// should be spending its time on. The rule instead is deterministic and
/// order-respecting: scan in selection order and drop the LATER device of the
/// first offending pair, repeating until stable. Earlier devices win, so the
/// first selected device -- the stable default a program may adopt -- is never
/// the one removed.
std::vector<sycl::device> filter_by_pair_aspects(std::vector<sycl::device>  selected,
                                                 const std::vector<aspect>& required) {
    bool has_pair = false;
    for (aspect a : required) has_pair = has_pair || is_pair_aspect(a);
    if (!has_pair) return selected;

    for (bool changed = true; changed;) {
        changed = false;
        for (std::size_t i = 0; i < selected.size() && !changed; ++i) {
            for (std::size_t j = i + 1; j < selected.size() && !changed; ++j) {
                if (!(selected[i].get_platform() == selected[j].get_platform())) continue;
                for (aspect a : required) {
                    if (!is_pair_aspect(a)) continue;
                    if (detail::satisfies_pair(a, selected[i], selected[j]) != answer::yes) {
                        selected.erase(selected.begin() + static_cast<std::ptrdiff_t>(j));
                        changed = true;
                        break;
                    }
                }
            }
        }
    }
    return selected;
}

std::vector<sycl::device> select_devices_by(std::vector<sycl::device>     candidates,
                                            std::initializer_list<aspect> required) {
    const std::vector<aspect> req{required};
    return filter_by_pair_aspects(filter_by_device_aspects(std::move(candidates), req), req);
}

/// The one place a ContextState is built, so every constructor gets the same
/// validation, the same context retrieval and the same cache.
std::shared_ptr<const ContextState> build(std::vector<sycl::device> devices) {
    if (devices.empty())
        throw std::invalid_argument{"naga::slabkit::context: no device satisfies the selection"};

    for (std::size_t i = 0; i < devices.size(); ++i)
        for (std::size_t j = i + 1; j < devices.size(); ++j)
            if (devices[i] == devices[j])
                throw std::invalid_argument{
                    "naga::slabkit::context: the same device was listed twice"};

    auto st     = std::make_shared<ContextState>();
    st->devices = std::move(devices);

    st->backend_of.resize(st->devices.size());
    for (std::size_t i = 0; i < st->devices.size(); ++i) {
        const sycl::platform p = st->devices[i].get_platform();

        std::size_t b = ContextState::npos;
        for (std::size_t k = 0; k < st->backends.size(); ++k)
            if (st->backends[k] == p) b = k;

        if (b == ContextState::npos) {
            b = st->backends.size();
            st->backends.push_back(p);
            // Retrieved, never created: this is queue{device}'s own context.
            st->backend_ctx.push_back(p.khr_get_default_context());
        }
        st->backend_of[i] = b;
    }

    const std::size_t n = st->devices.size();
    st->device_answers.assign(n * aspect_count, answer::unknown);
    st->pair_answers.assign(n * n * aspect_count, answer::unknown);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t a = 0; a < aspect_count; ++a) {
            const auto asp = static_cast<aspect>(a);
            if (is_pair_aspect(asp)) continue;
            st->device_answers[i * aspect_count + a] =
                detail::satisfies_device(asp, st->devices[i]);
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t a = 0; a < aspect_count; ++a) {
                const auto asp = static_cast<aspect>(a);
                if (!is_pair_aspect(asp)) continue;
                st->pair_answers[(i * n + j) * aspect_count + a] =
                    detail::satisfies_pair(asp, st->devices[i], st->devices[j]);
            }
        }
    }

    return st;
}

}  // namespace

// -----------------------------------------------------------------------------
// Constructors
// -----------------------------------------------------------------------------

context::context(std::vector<sycl::device> devices) : state_{build(std::move(devices))} {}

context::context(const sycl::device& dev) : state_{build(std::vector<sycl::device>{dev})} {}

context::context(std::vector<sycl::platform> platforms)
    : state_{build(root_devices_of(platforms))} {}

context::context(const sycl::platform& platform)
    : state_{build(root_devices_of(std::vector<sycl::platform>{platform}))} {}

context::context(std::initializer_list<aspect> required)
    : state_{build(select_devices_by(all_root_devices(), required))} {}

context::context(std::vector<sycl::platform> platforms, std::initializer_list<aspect> required)
    : state_{build(select_devices_by(root_devices_of(platforms), required))} {}

context::context(const sycl::platform& platform, std::initializer_list<aspect> required)
    : state_{build(
          select_devices_by(root_devices_of(std::vector<sycl::platform>{platform}), required))} {}

// -----------------------------------------------------------------------------
// The set
// -----------------------------------------------------------------------------

std::span<const sycl::device> context::devices() const { return state_->devices; }
std::span<const backend_id>   context::backends() const { return state_->backends; }

bool context::contains(const sycl::device& dev) const {
    return state_->index_of(dev) != detail::ContextState::npos;
}

backend_id context::backend_of(const sycl::device& dev) const {
    const std::size_t i = state_->index_of(dev);
    if (i == detail::ContextState::npos)
        throw std::invalid_argument{"naga::slabkit::context: device is not in this context"};
    return state_->backends[state_->backend_of[i]];
}

const sycl::context& context::context_of(const sycl::device& dev) const {
    const std::size_t i = state_->index_of(dev);
    if (i == detail::ContextState::npos)
        throw std::invalid_argument{"naga::slabkit::context: device is not in this context"};
    return state_->backend_ctx[state_->backend_of[i]];
}

const sycl::context& context::context_of(const backend_id& b) const {
    for (std::size_t k = 0; k < state_->backends.size(); ++k)
        if (state_->backends[k] == b) return state_->backend_ctx[k];
    throw std::invalid_argument{"naga::slabkit::context: backend is not in this context"};
}

// -----------------------------------------------------------------------------
// Selection, read from the cache
// -----------------------------------------------------------------------------

std::vector<sycl::device> context::select_list(aspect a) const {
    std::vector<sycl::device> out;
    for (std::size_t i = 0; i < state_->devices.size(); ++i)
        if (state_->device_answer(i, a) == answer::yes) out.push_back(state_->devices[i]);
    return out;
}

device_groups context::select_groups(aspect a) const {
    const std::size_t n = state_->devices.size();

    // Connected components under the pair relation, computed within a backend
    // (the relation is `no` across backends by construction, so a component can
    // never span two).
    std::vector<std::size_t> component(n, ContextState::npos);
    std::size_t              next = 0;

    for (std::size_t i = 0; i < n; ++i) {
        if (component[i] != ContextState::npos) continue;
        component[i] = next++;

        // Breadth-first over the relation; the answers are already cached, so
        // this walks memory rather than the machine.
        for (bool grew = true; grew;) {
            grew = false;
            for (std::size_t x = 0; x < n; ++x) {
                if (component[x] != component[i]) continue;
                for (std::size_t y = 0; y < n; ++y) {
                    if (component[y] != ContextState::npos) continue;
                    if (state_->pair_answer(x, y, a) == answer::yes) {
                        component[y] = component[i];
                        grew         = true;
                    }
                }
            }
        }
    }

    device_groups groups;
    for (std::size_t c = 0; c < next; ++c) {
        std::vector<sycl::device> members;
        sycl::platform            plat = state_->devices.front().get_platform();
        for (std::size_t i = 0; i < n; ++i) {
            if (component[i] != c) continue;
            if (members.empty()) plat = state_->devices[i].get_platform();
            members.push_back(state_->devices[i]);
        }
        if (!members.empty()) groups.emplace_back(plat, std::move(members));
    }
    return groups;
}

// -----------------------------------------------------------------------------
// Pointer queries, tried against every backend context held here
// -----------------------------------------------------------------------------

// A `host` answer is the WEAKEST one a backend can give, and one backend gives
// it for everything: AdaptiveCpp's OMP allocator answers `host` for any address
// it is handed, including another backend's device memory (measured -- see
// test_context). Taking the first backend's answer would therefore have made a
// CUDA device pointer look like host memory whenever an OMP backend happened to
// be enumerated first, and a routed copy would have chosen a host memcpy
// over it. So every backend is asked, a device or shared answer wins outright,
// and `host` is only believed when nothing stronger came back.
std::size_t context::strongest_answer(const void* p, kind& out) const {
    std::size_t host_index = detail::ContextState::npos;

    for (std::size_t i = 0; i < state_->backend_ctx.size(); ++i) {
        switch (sycl::get_pointer_type(p, state_->backend_ctx[i])) {
            case sycl::usm::alloc::device: out = kind::device; return i;
            case sycl::usm::alloc::shared: out = kind::shared; return i;
            case sycl::usm::alloc::host:
                if (host_index == detail::ContextState::npos) host_index = i;
                break;
            case sycl::usm::alloc::unknown: break;
        }
    }

    if (host_index != detail::ContextState::npos) {
        out = kind::host;
        return host_index;
    }
    return detail::ContextState::npos;
}

kind context::get_pointer_type(const void* p) const {
    kind k{};
    if (strongest_answer(p, k) == detail::ContextState::npos)
        throw sycl::exception{
            sycl::make_error_code(sycl::errc::invalid),
            "naga::slabkit::context: pointer belongs to no backend context here"};
    return k;
}

sycl::device context::get_pointer_device(const void* p) const {
    kind              k{};
    const std::size_t i = strongest_answer(p, k);
    if (i == detail::ContextState::npos)
        throw sycl::exception{
            sycl::make_error_code(sycl::errc::invalid),
            "naga::slabkit::context: pointer belongs to no backend context here"};
    return sycl::get_pointer_device(p, state_->backend_ctx[i]);
}

}  // namespace naga::slabkit
