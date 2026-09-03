#pragma once

// =============================================================================
// naga/slabkit/detail/queue_state.hpp — the stream behind slabkit::queue
// =============================================================================
//
// Internal; not installed.
//
// Per backend of the context, created on first use: an in-order sycl::queue for
// ordinary items and an out-of-order one for group members. Both carry
// AdaptiveCpp_retargetable, which is what makes the retarget command-group
// property legal on them, and both are built on that backend's default context
// so a retarget never crosses one.
//
// `last_step` is the whole of the stream's ordering state: the events the next
// item must depend on. One item leaves one event there; a group leaves all of
// its members' events.
//
// The mutex guards `pairs` and `last_step` and nothing else. It is taken to
// resolve a submission and taken again to record it, and is never held across a
// submit or a wait.

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/slabkit/context.hpp"
#include "naga/slabkit/queue.hpp"

namespace naga::slabkit::detail {

/// The host-work pool. Fixed size, FIFO, exceptions land in the caller's future
/// because every task posted here is a packaged_task.
///
/// THE POOL OUTLIVES ITS OWNER, ON PURPOSE. A host task holds a copy of the
/// queue handle -- the deferred release in the allocator does exactly that --
/// so the last reference to the QueueState is routinely dropped BY A POOL
/// THREAD, from inside the task it just ran. If the pool were a member destroyed
/// at that moment, the worker would return from the task into a destroyed
/// mutex. So each worker holds a share of the pool itself and the threads are
/// detached: the pool dies with its last worker, after that worker has stopped
/// touching it. Measured the hard way -- this was a segfault at the end of a
/// cross-device test, not a theory.
class HostPool : public std::enable_shared_from_this<HostPool> {
public:
    static std::shared_ptr<HostPool> create(std::size_t threads);
    ~HostPool();

    HostPool(const HostPool&)            = delete;
    HostPool& operator=(const HostPool&) = delete;

    void post(std::function<void()> task);

    /// Asks the workers to drain the queue and exit. Idempotent.
    void stop();

    /// Blocks until every worker has exited. Must not be called FROM a worker.
    void wait_workers();

    bool on_pool_thread() const;

private:
    HostPool() = default;
    void worker();

    std::deque<std::function<void()>> tasks_;
    std::vector<std::thread::id>      worker_ids_;
    mutable std::mutex                mutex_;
    std::condition_variable           cv_;
    std::condition_variable           done_cv_;
    std::size_t                       live_workers_ = 0;
    bool                              stop_         = false;
};

class QueueState {
public:
    QueueState(context ctx, sycl::async_handler handler, std::size_t host_threads);
    ~QueueState();

    struct BackendPair {
        sycl::queue inorder;
        sycl::queue concurrent;
    };

    /// The pair for backend index `b`, created on first use. Caller holds the
    /// mutex; construction touches the runtime but never submits.
    BackendPair& pair_for(std::size_t b);

    /// \throws std::invalid_argument if `dev` is not in the context.
    std::size_t backend_index_of(const sycl::device& dev) const;

    /// Every pair created so far, for wait/throw fan-out. Caller holds the mutex.
    std::vector<BackendPair*> live_pairs();

    context             ctx;
    sycl::async_handler handler;
    bool                has_handler;
    sycl::device        default_device;
    std::size_t         default_backend;

    std::mutex                                mutex;
    std::vector<std::unique_ptr<BackendPair>> pairs;
    std::vector<annotated_event>              last_step;

    std::shared_ptr<HostPool> pool;
};

}  // namespace naga::slabkit::detail
