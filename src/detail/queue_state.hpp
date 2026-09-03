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
class HostPool {
public:
    explicit HostPool(std::size_t threads);
    ~HostPool();

    HostPool(const HostPool&)            = delete;
    HostPool& operator=(const HostPool&) = delete;

    void post(std::function<void()> task);

private:
    void worker();

    std::vector<std::thread>          threads_;
    std::deque<std::function<void()>> tasks_;
    std::mutex                        mutex_;
    std::condition_variable           cv_;
    bool                              stop_ = false;
};

class QueueState {
public:
    QueueState(context ctx, sycl::async_handler handler, std::size_t host_threads);

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
    std::vector<sycl::event>                  last_step;

    HostPool pool;
};

}  // namespace naga::slabkit::detail
