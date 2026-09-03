#pragma once

// =============================================================================
// naga/slabkit/detail/context_state.hpp — what a slabkit context holds
// =============================================================================
//
// Internal; not installed. The state is immutable after construction, which is
// what lets a context be a light handle whose copies share it and lets every
// downstream object hold one without synchronization.
//
// The answer caches are flat arrays indexed by (device, aspect) and
// (device, device, aspect). An answer is one byte and a context holds a
// handful of devices, so the whole cache is smaller than one of the sycl
// objects beside it, and every lookup is an index rather than a hash.

#include <cstddef>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/slabkit/capabilities.hpp"
#include "naga/slabkit/context.hpp"

namespace naga::slabkit::detail {

class ContextState {
public:
    std::vector<sycl::device>   devices;
    std::vector<sycl::platform> backends;     // deduplicated, first-seen order
    std::vector<sycl::context>  backend_ctx;  // parallel to `backends`
    std::vector<std::size_t>    backend_of;   // per device: index into `backends`

    std::vector<answer> device_answers;  // devices.size() * aspect_count
    std::vector<answer> pair_answers;    // devices.size()^2 * aspect_count

    std::size_t index_of(const sycl::device& dev) const noexcept {
        for (std::size_t i = 0; i < devices.size(); ++i)
            if (devices[i] == dev) return i;
        return npos;
    }

    answer device_answer(std::size_t i, aspect a) const noexcept {
        return device_answers[i * aspect_count + static_cast<std::size_t>(a)];
    }
    answer pair_answer(std::size_t i, std::size_t j, aspect a) const noexcept {
        return pair_answers[(i * devices.size() + j) * aspect_count + static_cast<std::size_t>(a)];
    }

    static constexpr std::size_t npos = static_cast<std::size_t>(-1);
};

struct ContextAccess {
    static const std::shared_ptr<const ContextState>& state(const context& c) noexcept {
        return c.state_;
    }
};

}  // namespace naga::slabkit::detail
