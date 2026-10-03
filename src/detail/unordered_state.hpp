#pragma once

// =============================================================================
// naga/slabkit/detail/unordered_state.hpp — the stream behind slabkit::unordered_queue
// =============================================================================
//
// Internal; not installed.
//
// ONE sycl::queue, AdaptiveCpp_retargetable, constructed with no device and no
// in-order property. A submission reaches a device through AdaptiveCpp_retarget,
// naming the device and the context this stream's slabkit context holds for that
// device's backend.
//
// There is no ordering state at all: a task depends on the events its caller
// passed and on nothing else.
//
// HOST ERRORS are the same type slabkit::queue uses and are a shared_ptr for the
// same reason: a host callable outlives the stream that submitted it, because
// nothing drains at destruction.

#include <memory>

#include <sycl/sycl.hpp>

#include "detail/queue_state.hpp"
#include "naga/slabkit/context.hpp"
#include "naga/slabkit/unordered_queue.hpp"

namespace naga::slabkit::detail {

class UnorderedState {
public:
    UnorderedState(context ctx, const sycl::device* dev, sycl::async_handler handler);

    UnorderedState(const UnorderedState&)            = delete;
    UnorderedState& operator=(const UnorderedState&) = delete;

    /// The placement for `dev`, or for the default device when null.
    /// \throws std::invalid_argument if `dev` is not in this stream's context.
    sycl::property_list place(const sycl::device* dev) const;

    context      ctx;
    sycl::device default_device;

    sycl::queue stream;

    std::shared_ptr<HostErrors> errors;
};

}  // namespace naga::slabkit::detail
