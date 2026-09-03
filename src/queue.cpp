// =============================================================================
// queue.cpp — the compiled core of the stream
// =============================================================================
//
// Nothing here is a template. The public queue's submit and its shortcuts are
// thin templates that call begin_item, submit to the sycl::queue they are
// handed, and call end_item -- so a consumer compiles the kernel and almost
// nothing else of slabkit.

#include "naga/slabkit/queue.hpp"

#include <stdexcept>
#include <utility>

#include "detail/queue_state.hpp"

namespace naga::slabkit::detail {

// -----------------------------------------------------------------------------
// The host pool
// -----------------------------------------------------------------------------

std::shared_ptr<HostPool> HostPool::create(std::size_t threads) {
    if (threads == 0) threads = 1;

    auto pool = std::shared_ptr<HostPool>{new HostPool()};
    for (std::size_t i = 0; i < threads; ++i) {
        // The worker holds a share of the pool, so the pool cannot be destroyed
        // out from under a running task. The thread is detached because the
        // owner may be released BY this thread.
        std::thread{[pool] { pool->worker(); }}.detach();
    }
    return pool;
}

HostPool::~HostPool() = default;

void HostPool::stop() {
    {
        std::lock_guard<std::mutex> lock{mutex_};
        stop_ = true;
    }
    cv_.notify_all();
}

void HostPool::wait_workers() {
    std::unique_lock<std::mutex> lock{mutex_};
    done_cv_.wait(lock, [this] { return live_workers_ == 0; });
}

bool HostPool::on_pool_thread() const {
    const auto                  me = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock{mutex_};
    for (const auto& id : worker_ids_)
        if (id == me) return true;
    return false;
}

void HostPool::post(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lock{mutex_};
        if (stop_) return;
        tasks_.push_back(std::move(task));
    }
    cv_.notify_one();
}

void HostPool::worker() {
    {
        std::lock_guard<std::mutex> lock{mutex_};
        worker_ids_.push_back(std::this_thread::get_id());
        ++live_workers_;
    }

    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock{mutex_};
            cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            if (tasks_.empty()) {
                // Stopping, and drained. Everything posted before stop() has
                // run -- which is what makes a deferred release reliable.
                --live_workers_;
                lock.unlock();
                done_cv_.notify_all();
                return;
            }
            task = std::move(tasks_.front());
            tasks_.pop_front();
        }
        // Outside the lock: a host task may wait on device work, and may post
        // to the core queues, for as long as it likes.
        task();

        // Releasing the task here, still outside the lock, is what drops the
        // queue handle it captured -- possibly the last one. The pool survives
        // that because this worker holds a share of it.
        task = nullptr;
    }
}

// -----------------------------------------------------------------------------
// The stream state
// -----------------------------------------------------------------------------

QueueState::QueueState(context c, sycl::async_handler h, std::size_t host_threads)
    : ctx{std::move(c)}
    , handler{std::move(h)}
    , has_handler{static_cast<bool>(handler)}
    , default_device{ctx.devices().front()}
    , default_backend{0}
    , pool{HostPool::create(host_threads)} {
    pairs.resize(ctx.backends().size());

    const backend_id b = ctx.backend_of(default_device);
    for (std::size_t i = 0; i < ctx.backends().size(); ++i)
        if (ctx.backends()[i] == b) default_backend = i;
}

/// Destroying the last handle to a stream drains its host pool: every task
/// already posted runs, and only then does the stream go away. That is what
/// makes the allocator's deferred release reliable rather than best-effort.
///
/// Unless we ARE a pool thread -- a host task that held the last handle -- in
/// which case waiting would be waiting on ourselves. The pool then dies with
/// its last worker, which is why the workers own a share of it.
QueueState::~QueueState() {
    pool->stop();
    if (!pool->on_pool_thread()) pool->wait_workers();
}

std::size_t QueueState::backend_index_of(const sycl::device& dev) const {
    if (!ctx.contains(dev))
        throw std::invalid_argument{
            "naga::slabkit::queue: device is not in this queue's context"};

    const backend_id b = ctx.backend_of(dev);
    for (std::size_t i = 0; i < ctx.backends().size(); ++i)
        if (ctx.backends()[i] == b) return i;

    throw std::invalid_argument{"naga::slabkit::queue: device has no backend in this context"};
}

QueueState::BackendPair& QueueState::pair_for(std::size_t b) {
    if (pairs[b]) return *pairs[b];

    // The pair is bound to the FIRST device of this context on that backend, so
    // an unplaced submission runs on the stream's default device and a placed
    // one retargets within the same context.
    sycl::device bound = ctx.devices().front();
    for (const auto& d : ctx.devices())
        if (ctx.backend_of(d) == ctx.backends()[b]) {
            bound = d;
            break;
        }

    const sycl::context& sctx = ctx.context_of(ctx.backends()[b]);

    const sycl::property_list ordered{sycl::property::queue::in_order{},
                                      sycl::property::queue::AdaptiveCpp_retargetable{}};
    const sycl::property_list unordered{sycl::property::queue::AdaptiveCpp_retargetable{}};

    if (has_handler)
        pairs[b] = std::unique_ptr<BackendPair>{
            new BackendPair{sycl::queue{sctx, bound, handler, ordered},
                            sycl::queue{sctx, bound, handler, unordered}}};
    else
        pairs[b] = std::unique_ptr<BackendPair>{new BackendPair{
            sycl::queue{sctx, bound, ordered}, sycl::queue{sctx, bound, unordered}}};

    return *pairs[b];
}

std::vector<QueueState::BackendPair*> QueueState::live_pairs() {
    std::vector<BackendPair*> out;
    for (auto& p : pairs)
        if (p) out.push_back(p.get());
    return out;
}

// -----------------------------------------------------------------------------
// The core the public templates call
// -----------------------------------------------------------------------------

std::shared_ptr<QueueState> make_queue_state(context ctx, sycl::async_handler handler,
                                             std::size_t host_threads) {
    return std::make_shared<QueueState>(std::move(ctx), std::move(handler), host_threads);
}

namespace {

/// Split dependencies by whether the target backend will honour them. Same
/// backend: depends_on. Different backend: the host has to wait, because the
/// dependency is silently ignored otherwise (measured; see `submission`).
void split_by_backend(const std::vector<annotated_event>& in, std::size_t target,
                      submission& out) {
    for (const auto& [e, b] : in) {
        if (b == target)
            out.deps.push_back(e);
        else
            out.host_waits.push_back(e);
    }
}

}  // namespace

submission begin_item(QueueState& st, const sycl::device* dev, bool concurrent) {
    // Resolving needs the map and the step state; submitting does not. The lock
    // ends with this scope, before the caller touches a sycl::queue -- or waits
    // on anything.
    std::lock_guard<std::mutex> lock{st.mutex};

    const std::size_t b = dev ? st.backend_index_of(*dev) : st.default_backend;
    auto&             p = st.pair_for(b);

    submission s;
    s.target  = concurrent ? &p.concurrent : &p.inorder;
    s.backend = b;
    split_by_backend(st.last_step, b, s);
    return s;
}

submission begin_group_item(QueueState& st, const sycl::device* dev,
                            const std::vector<annotated_event>& group_deps) {
    std::lock_guard<std::mutex> lock{st.mutex};

    const std::size_t b = dev ? st.backend_index_of(*dev) : st.default_backend;
    auto&             p = st.pair_for(b);

    submission s;
    s.target  = &p.concurrent;
    s.backend = b;
    split_by_backend(group_deps, b, s);
    return s;
}

void end_item(QueueState& st, const sycl::event& e, std::size_t b) {
    std::lock_guard<std::mutex> lock{st.mutex};
    st.last_step.assign(1, annotated_event{e, b});
}

std::vector<annotated_event> begin_group(QueueState& st) {
    std::lock_guard<std::mutex> lock{st.mutex};
    return st.last_step;
}

void end_group(QueueState& st, std::vector<annotated_event> events) {
    std::lock_guard<std::mutex> lock{st.mutex};
    // An empty group leaves the step where it was: it added no work, so it
    // cannot have removed the previous item's ordering.
    if (!events.empty()) st.last_step = std::move(events);
}

void post_host(QueueState& st, std::function<void()> task) { st.pool->post(std::move(task)); }

std::weak_ptr<QueueState> weak_state(const queue& q) { return q.state_; }

void wait_if_alive(const std::weak_ptr<QueueState>& ws) {
    if (auto st = ws.lock()) state_wait(*st);
}

const context&      state_context(const QueueState& st) { return st.ctx; }
const sycl::device& state_device(const QueueState& st) { return st.default_device; }

void state_wait(QueueState& st) {
    std::vector<QueueState::BackendPair*> live;
    {
        std::lock_guard<std::mutex> lock{st.mutex};
        live = st.live_pairs();
    }
    // Waiting outside the lock: a wait can take arbitrarily long, and a host
    // task may want to submit while it does.
    for (auto* p : live) {
        p->inorder.wait();
        p->concurrent.wait();
    }
}

void state_wait_and_throw(QueueState& st) {
    state_wait(st);
    state_throw_asynchronous(st);
}

void state_throw_asynchronous(QueueState& st) {
    std::vector<QueueState::BackendPair*> live;
    {
        std::lock_guard<std::mutex> lock{st.mutex};
        live = st.live_pairs();
    }
    for (auto* p : live) {
        p->inorder.throw_asynchronous();
        p->concurrent.throw_asynchronous();
    }
}

std::vector<sycl::event> state_wait_list(QueueState& st) {
    std::vector<QueueState::BackendPair*> live;
    {
        std::lock_guard<std::mutex> lock{st.mutex};
        live = st.live_pairs();
    }

    std::vector<sycl::event> out;
    for (auto* p : live) {
        for (auto& e : p->inorder.get_wait_list()) out.push_back(e);
        for (auto& e : p->concurrent.get_wait_list()) out.push_back(e);
    }
    return out;
}

bool retarget_device(const sycl::property_list& props, sycl::device& out) {
    if (!props.has_property<sycl::property::command_group::AdaptiveCpp_retarget>()) return false;
    out = props.get_property<sycl::property::command_group::AdaptiveCpp_retarget>().dev;
    return true;
}

}  // namespace naga::slabkit::detail

namespace naga::slabkit {

queue::queue(context ctx, sycl::async_handler handler, std::size_t host_threads)
    : state_{detail::make_queue_state(std::move(ctx), std::move(handler), host_threads)} {}

}  // namespace naga::slabkit
