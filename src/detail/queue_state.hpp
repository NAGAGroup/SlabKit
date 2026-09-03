#pragma once

// =============================================================================
// naga/slabkit/detail/queue_state.hpp — the stream behind slabkit::queue
// =============================================================================
//
// Internal; not installed.
//
// Per backend of the context, created on first use: an in-order sycl::queue for
// ordinary kernel items and an out-of-order one for group members. Both carry
// AdaptiveCpp_retargetable, which is what makes the retarget command-group
// property legal on them, and both are built on that backend's default context
// so a retarget never crosses one.
//
// THE HOST LANE is a third queue, out-of-order, on the OpenMP backend's own
// device and default context. It always exists, whether or not the stream's
// context contains an OpenMP device, because host work has to have somewhere to
// run. Its items are OpenMP single_tasks whose body is an indirect call through
// a function pointer -- see `HostThunk` below.
//
// THE ORDERING STATE is two fields. `kernel_step` is the kernel work the next
// kernel item must follow; `host_pending` is the host work outstanding right
// now. A kernel item waits for both and empties `host_pending` (it has absorbed
// it). A host item waits for `kernel_step` alone and appends itself to
// `host_pending` -- host work is parallel with host work.
//
// The mutex guards those two, `pairs` and `host_errors`, and nothing else. It is
// taken to resolve a submission and taken again to record it, and is never held
// across a submit or a wait.

#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/slabkit/context.hpp"
#include "naga/slabkit/queue.hpp"

namespace naga::slabkit::detail {

/// What a host item's kernel actually captures: two trivially copyable values,
/// which is what kernel captures must be. `fn` is the trampoline's address --
/// ordinary host code in this library -- and calling it INDIRECTLY is what the
/// generic flow's JIT accepts where a direct call to complex host code does not.
struct HostThunk {
    void (*fn)(void*);
    void* arg;
};

class QueueState {
public:
    QueueState(context ctx, sycl::async_handler handler);
    ~QueueState();

    QueueState(const QueueState&)            = delete;
    QueueState& operator=(const QueueState&) = delete;

    struct BackendPair {
        sycl::queue inorder;
        sycl::queue concurrent;
    };

    static constexpr std::size_t npos = static_cast<std::size_t>(-1);

    /// The pair for backend index `b`, created on first use. Caller holds the
    /// mutex; construction touches the runtime but never submits.
    BackendPair& pair_for(std::size_t b);

    /// \throws std::invalid_argument if `dev` is not in the context.
    std::size_t backend_index_of(const sycl::device& dev) const;

    /// Every pair created so far, for wait/throw fan-out. Caller holds the mutex.
    std::vector<BackendPair*> live_pairs();

    /// Submit one host item on the lane, waiting for `deps` first. Takes no lock.
    sycl::event launch_host(const std::vector<sycl::event>& deps, HostFn fn);

    /// Drop the host events that have already completed, so a long run of host
    /// work does not accumulate a dependency list. Caller holds the mutex.
    void prune_host_pending();

    void record_host_error(std::exception_ptr e);

    context             ctx;
    sycl::async_handler handler;
    bool                has_handler;
    sycl::device        default_device;
    std::size_t         default_backend;

    // The host lane. `host_backend` is this lane's index within ctx.backends(),
    // or npos when the stream's context has no OpenMP device -- in which case a
    // kernel item can never take a host event through depends_on and always pays
    // the host wait.
    sycl::device  host_device;
    sycl::context host_context;
    sycl::queue   host_lane;
    std::size_t   host_backend;

    std::mutex                                mutex;
    std::vector<std::unique_ptr<BackendPair>> pairs;
    std::vector<annotated_event>              kernel_step;
    std::vector<sycl::event>                  host_pending;
    std::vector<std::exception_ptr>           host_errors;
};

/// The QueueState whose host lane the calling thread is currently running a task
/// for, or nullptr. This is how a wait issued FROM a host task avoids waiting on
/// the lane that task occupies -- the self-wait that deadlocks.
QueueState* running_host_task_of();

}  // namespace naga::slabkit::detail
