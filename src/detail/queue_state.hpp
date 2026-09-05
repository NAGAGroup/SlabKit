#pragma once

// =============================================================================
// naga/slabkit/detail/queue_state.hpp — the stream behind slabkit::queue
// =============================================================================
//
// Internal; not installed.
//
// THREE sycl::queues, all AdaptiveCpp_retargetable and none constructed with a
// device: the PRIMARY carries the in-order property and takes kernel tasks and
// the _in_order memory tasks, the GROUP queue takes a group's subtasks, and the
// HOST/MEMORY queue takes the asynchronous memory tasks. A submission reaches a
// device through AdaptiveCpp_retarget, naming the device and the context this
// stream's slabkit context holds for that device's backend.
//
// There is no ordering state to keep: a queue's own wait list IS the record of
// what it has outstanding, and merging those lists at submission time is the
// whole of the ordering (queue.hpp).
//
// HOST ERRORS are a shared_ptr rather than a member, because a host callable
// outlives the stream that submitted it: nothing drains at destruction.

#include <exception>
#include <memory>
#include <mutex>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/slabkit/context.hpp"
#include "naga/slabkit/queue.hpp"

namespace naga::slabkit::detail {

/// Exceptions escaping host callables, held for throw_asynchronous.
class HostErrors {
public:
    void               push(std::exception_ptr e);
    std::exception_ptr pop();

private:
    std::mutex                      mutex_;
    std::vector<std::exception_ptr> errors_;
};

class QueueState {
public:
    QueueState(context ctx, const sycl::device* dev, sycl::async_handler handler);

    QueueState(const QueueState&)            = delete;
    QueueState& operator=(const QueueState&) = delete;

    /// The placement for `dev`, or for the default device when null.
    /// \throws std::invalid_argument if `dev` is not in this stream's context.
    sycl::property_list place(const sycl::device* dev) const;

    /// The device a submission placed on `dev` runs on.
    const sycl::device& device_or_default(const sycl::device* dev) const;

    context      ctx;
    sycl::device default_device;

    sycl::queue primary;
    sycl::queue group_queue;
    sycl::queue host_queue;

    /// Where a staged copy issues its two legs, from inside the host task that
    /// owns them. Nothing merges against it: the task waits each leg before it
    /// returns, so the task's own event is the whole of that copy's ordering.
    sycl::queue legs;

    std::shared_ptr<HostErrors> errors;
};

/// A group's captured dependency set and the events of its subtasks. The group
/// lives inside one submit_group call, and the stream outlives it.
class GroupState {
public:
    GroupState(QueueState& st, std::vector<sycl::event> deps);

    QueueState&                     state;
    const std::vector<sycl::event>  captured;

    void                     record(const sycl::event& e);
    std::vector<sycl::event> events() const;

private:
    mutable std::mutex       mutex_;
    std::vector<sycl::event> events_;
};

}  // namespace naga::slabkit::detail
