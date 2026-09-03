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

HostPool::HostPool(std::size_t threads) {
    if (threads == 0) threads = 1;
    threads_.reserve(threads);
    for (std::size_t i = 0; i < threads; ++i) threads_.emplace_back([this] { worker(); });
}

HostPool::~HostPool() {
    {
        std::lock_guard<std::mutex> lock{mutex_};
        stop_ = true;
    }
    cv_.notify_all();

    for (auto& t : threads_) {
        // A host task may hold the last reference to the queue it was posted
        // on -- the deferred release path does exactly that -- so this
        // destructor can run ON a pool thread. Joining yourself is a deadlock;
        // detaching is the honest answer, and the task that got us here is the
        // last one there will be.
        if (t.get_id() == std::this_thread::get_id())
            t.detach();
        else if (t.joinable())
            t.join();
    }
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
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock{mutex_};
            cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            if (tasks_.empty()) return;  // stopping, and drained
            task = std::move(tasks_.front());
            tasks_.pop_front();
        }
        // Outside the lock: a host task may wait on device work, and may post
        // to the core queues, for as long as it likes.
        task();
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
    , pool{host_threads} {
    pairs.resize(ctx.backends().size());

    const backend_id b = ctx.backend_of(default_device);
    for (std::size_t i = 0; i < ctx.backends().size(); ++i)
        if (ctx.backends()[i] == b) default_backend = i;
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

submission begin_item(QueueState& st, const sycl::device* dev, bool concurrent) {
    // Resolving needs the map and the step state; submitting does not. The lock
    // ends with this scope, before the caller touches a sycl::queue.
    std::lock_guard<std::mutex> lock{st.mutex};

    const std::size_t b = dev ? st.backend_index_of(*dev) : st.default_backend;
    auto&             p = st.pair_for(b);

    return submission{concurrent ? &p.concurrent : &p.inorder, st.last_step};
}

void end_item(QueueState& st, const sycl::event& e) {
    std::lock_guard<std::mutex> lock{st.mutex};
    st.last_step.assign(1, e);
}

std::vector<sycl::event> begin_group(QueueState& st) {
    std::lock_guard<std::mutex> lock{st.mutex};
    return st.last_step;
}

void end_group(QueueState& st, std::vector<sycl::event> events) {
    std::lock_guard<std::mutex> lock{st.mutex};
    // An empty group leaves the step where it was: it added no work, so it
    // cannot have removed the previous item's ordering.
    if (!events.empty()) st.last_step = std::move(events);
}

void post_host(QueueState& st, std::function<void()> task) { st.pool.post(std::move(task)); }

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
