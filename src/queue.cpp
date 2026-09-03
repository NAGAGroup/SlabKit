// =============================================================================
// queue.cpp — the compiled core of the stream
// =============================================================================
//
// Nothing here is a template. The public queue's submit and its shortcuts are
// thin templates that call begin_item, submit to the sycl::queue they are
// handed, and call end_item -- so a consumer compiles the kernel and almost
// nothing else of slabkit.
//
// Host work is the same shape: the public submit_host type-erases the callable
// into a HostFn and everything after that happens here, including the one
// OpenMP kernel every host item in the process runs.

#include "naga/slabkit/queue.hpp"

#include <stdexcept>
#include <utility>

#include "detail/queue_state.hpp"

namespace naga::slabkit::detail {

namespace {

/// A host item, owned by the trampoline that runs it.
struct HostTask {
    HostFn      fn;
    QueueState* st;
};

thread_local QueueState* g_running_host_task_of = nullptr;

/// THE TRAMPOLINE. Reached by an indirect call from inside an OpenMP kernel, and
/// therefore the boundary between kernel code and ordinary host code: everything
/// a host task does happens below this line, where the JIT never looks.
///
/// It cannot let an exception escape -- unwinding back into JIT-compiled kernel
/// code is not something the runtime is prepared for -- so a throwing task is
/// recorded on the stream and surfaces from wait_and_throw or
/// sync_host_work_and_throw, which is where SYCL puts asynchronous errors.
void host_trampoline(void* p) noexcept {
    HostTask*   task = static_cast<HostTask*>(p);
    QueueState* prev = g_running_host_task_of;
    g_running_host_task_of = task->st;

    try {
        task->fn.run(task->fn.payload);
    } catch (...) {
        task->st->record_host_error(std::current_exception());
    }

    try {
        task->fn.destroy(task->fn.payload);
    } catch (...) {  // NOLINT: a destructor that throws is not worth a crash
    }

    g_running_host_task_of = prev;
    delete task;
}

/// The OpenMP backend's device, and the platform default context beside it.
/// AdaptiveCpp always enumerates this backend; a build that somehow did not
/// would have nowhere to run host work, and saying so here is better than
/// failing on the first copy.
sycl::device find_host_device() {
    for (const auto& p : sycl::platform::get_platforms())
        for (const auto& d : p.get_devices())
            if (d.get_backend() == sycl::backend::omp) return d;

    throw std::runtime_error{
        "naga::slabkit::queue: no OpenMP device is visible, so there is no host lane "
        "(check ACPP_VISIBILITY_MASK)"};
}

/// Split dependencies by whether the target backend will honour them. Same
/// backend: depends_on. Different backend: the host waits, because slabkit never
/// hands a foreign event to a non-OpenMP consumer.
void split_by_backend(const std::vector<annotated_event>& in, std::size_t target,
                      submission& out) {
    for (const auto& [e, b] : in) {
        if (b == target)
            out.deps.push_back(e);
        else
            out.host_waits.push_back(e);
    }
}

/// The host side of the interlock, from a kernel item's point of view: every
/// outstanding host event, routed to depends_on or to a host wait by the same
/// rule.
void split_host_pending(const std::vector<sycl::event>& in, std::size_t host_backend,
                        std::size_t target, submission& out) {
    for (const auto& e : in) {
        if (host_backend == target)
            out.deps.push_back(e);
        else
            out.host_waits.push_back(e);
    }
}

}  // namespace

QueueState* running_host_task_of() { return g_running_host_task_of; }

// -----------------------------------------------------------------------------
// The stream state
// -----------------------------------------------------------------------------

QueueState::QueueState(context c, sycl::async_handler h)
    : ctx{std::move(c)}
    , handler{std::move(h)}
    , has_handler{static_cast<bool>(handler)}
    , default_device{ctx.devices().front()}
    , default_backend{0}
    , host_device{find_host_device()}
    , host_context{host_device.get_platform().khr_get_default_context()}
    , host_lane{has_handler
                    ? sycl::queue{host_context, host_device, handler,
                                  sycl::property_list{}}
                    : sycl::queue{host_context, host_device, sycl::property_list{}}}
    , host_backend{npos} {
    pairs.resize(ctx.backends().size());

    const backend_id b = ctx.backend_of(default_device);
    for (std::size_t i = 0; i < ctx.backends().size(); ++i) {
        if (ctx.backends()[i] == b) default_backend = i;
        if (ctx.backends()[i] == host_device.get_platform()) host_backend = i;
    }
}

/// Destroying the last handle to a stream drains its host lane: every item
/// already submitted runs, and only then does the stream go away. That is what
/// makes the allocator's deferred release reliable rather than best-effort.
///
/// Unless we ARE the lane -- a host task that held the last handle -- in which
/// case waiting would be waiting on ourselves. Nothing in slabkit puts a strong
/// stream handle inside a host task, exactly so this stays a guard rather than a
/// mechanism.
QueueState::~QueueState() {
    if (running_host_task_of() == this) return;
    try {
        host_lane.wait();
    } catch (...) {  // NOLINT: a destructor reports nothing
    }
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

void QueueState::prune_host_pending() {
    std::vector<sycl::event> keep;
    keep.reserve(host_pending.size());
    for (auto& e : host_pending) {
        const auto status = e.get_info<sycl::info::event::command_execution_status>();
        if (status != sycl::info::event_command_status::complete) keep.push_back(e);
    }
    host_pending.swap(keep);
}

void QueueState::record_host_error(std::exception_ptr e) {
    std::lock_guard<std::mutex> lock{mutex};
    host_errors.push_back(std::move(e));
}

sycl::event QueueState::launch_host(const std::vector<sycl::event>& deps, HostFn fn) {
    auto*           task = new HostTask{fn, this};
    const HostThunk thunk{&host_trampoline, task};

    try {
        // The ONE kernel every host item in the process runs. Its body is an
        // indirect call and nothing else, so the JIT compiles it once and every
        // task differs only in the two values captured here.
        return host_lane.submit([&](sycl::handler& h) {
            h.depends_on(deps);
            h.single_task([thunk] { thunk.fn(thunk.arg); });
        });
    } catch (...) {
        fn.destroy(fn.payload);
        delete task;
        throw;
    }
}

// -----------------------------------------------------------------------------
// The core the public templates call
// -----------------------------------------------------------------------------

std::shared_ptr<QueueState> make_queue_state(context ctx, sycl::async_handler handler) {
    return std::make_shared<QueueState>(std::move(ctx), std::move(handler));
}

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

    // The interlock: this item waits for the outstanding host work as well as
    // for the kernel tail, and having waited for it, absorbs it.
    st.prune_host_pending();
    split_host_pending(st.host_pending, st.host_backend, b, s);
    st.host_pending.clear();

    split_by_backend(st.kernel_step, b, s);
    return s;
}

submission begin_group_item(QueueState& st, const sycl::device* dev, const group_deps& deps) {
    std::lock_guard<std::mutex> lock{st.mutex};

    const std::size_t b = dev ? st.backend_index_of(*dev) : st.default_backend;
    auto&             p = st.pair_for(b);

    submission s;
    s.target  = &p.concurrent;
    s.backend = b;
    split_host_pending(deps.host, st.host_backend, b, s);
    split_by_backend(deps.kernel, b, s);
    return s;
}

void end_item(QueueState& st, const sycl::event& e, std::size_t b) {
    std::lock_guard<std::mutex> lock{st.mutex};
    st.kernel_step.assign(1, annotated_event{e, b});
}

group_deps begin_group(QueueState& st) {
    std::lock_guard<std::mutex> lock{st.mutex};
    st.prune_host_pending();
    return group_deps{st.kernel_step, st.host_pending};
}

void end_group(QueueState& st, std::vector<annotated_event> kernel_events,
               std::vector<sycl::event> host_events) {
    std::lock_guard<std::mutex> lock{st.mutex};

    // An empty group leaves the stream's order alone: it added no work, so it
    // cannot have removed the previous item's ordering.
    if (!kernel_events.empty()) {
        st.kernel_step = std::move(kernel_events);
        // Every kernel member waited for the host work outstanding before the
        // group, so it is absorbed -- exactly as it would be for a lone item.
        st.host_pending.clear();
    }
    for (auto& e : host_events) st.host_pending.push_back(e);
}

// -----------------------------------------------------------------------------
// Host work
// -----------------------------------------------------------------------------

namespace {

/// The kernel events a host item waits for. Every one of them, whatever backend
/// it ran on: the lane is OpenMP, and an OpenMP consumer honours a foreign event
/// (measured, 0 wrong in 20 depending on a CUDA event).
std::vector<sycl::event> kernel_deps_of(const std::vector<annotated_event>& step) {
    std::vector<sycl::event> out;
    out.reserve(step.size());
    for (const auto& [e, b] : step) out.push_back(e);
    return out;
}

}  // namespace

sycl::event submit_host_item(QueueState& st, HostFn fn) {
    std::vector<sycl::event> deps;
    {
        std::lock_guard<std::mutex> lock{st.mutex};
        st.prune_host_pending();
        deps = kernel_deps_of(st.kernel_step);
    }

    sycl::event e = st.launch_host(deps, fn);

    std::lock_guard<std::mutex> lock{st.mutex};
    st.host_pending.push_back(e);
    return e;
}

sycl::event submit_group_host_item(QueueState& st, const std::vector<annotated_event>& deps,
                                   HostFn fn) {
    return st.launch_host(kernel_deps_of(deps), fn);
}

void register_host_item(QueueState& st, const sycl::event& e) {
    std::lock_guard<std::mutex> lock{st.mutex};
    st.host_pending.push_back(e);
}

ChainState begin_chain(QueueState& st) {
    ChainState cs;
    cs.st = &st;
    std::lock_guard<std::mutex> lock{st.mutex};
    cs.deps = kernel_deps_of(st.kernel_step);
    return cs;
}

ChainState begin_chain_with(QueueState& st, std::vector<annotated_event> kernel_deps) {
    ChainState cs;
    cs.st   = &st;
    cs.deps = kernel_deps_of(kernel_deps);
    return cs;
}

sycl::event chain_then(ChainState& cs, HostFn fn) {
    const std::vector<sycl::event> deps =
        cs.started ? std::vector<sycl::event>{cs.last} : cs.deps;

    cs.last    = cs.st->launch_host(deps, fn);
    cs.started = true;
    return cs.last;
}

sycl::event finish_chain(ChainState& cs) {
    // An empty chain is still one host item: it has to carry the stream's
    // ordering, and an item that does nothing is the cheapest way to say so.
    if (!cs.started) chain_then(cs, make_host_fn([] {}));
    return cs.last;
}

// -----------------------------------------------------------------------------
// Waiting, errors, introspection
// -----------------------------------------------------------------------------

const context&      state_context(const QueueState& st) { return st.ctx; }
const sycl::device& state_device(const QueueState& st) { return st.default_device; }

namespace {

std::vector<QueueState::BackendPair*> snapshot(QueueState& st) {
    std::lock_guard<std::mutex> lock{st.mutex};
    return st.live_pairs();
}

/// Waiting on the lane FROM the lane is the self-wait that deadlocks, so a host
/// task's own wait covers the device queues only.
void wait_host_lane(QueueState& st) {
    if (running_host_task_of() == &st) return;
    st.host_lane.wait();
    // Pruned, not cleared: another thread may have submitted a host item while
    // this wait was running, and that one is still outstanding.
    std::lock_guard<std::mutex> lock{st.mutex};
    st.prune_host_pending();
}

}  // namespace

void state_wait(QueueState& st) {
    // Host first: a host item may issue device work and wait on it, so draining
    // the lane is also draining what the lane started. Then the device queues,
    // for kernel work submitted after the last host item. Then the lane once
    // more, for host items a kernel item's completion released.
    wait_host_lane(st);

    // Waiting outside the lock: a wait can take arbitrarily long, and a host
    // task may want to submit while it does.
    for (auto* p : snapshot(st)) {
        p->inorder.wait();
        p->concurrent.wait();
    }

    wait_host_lane(st);
}

void state_wait_and_throw(QueueState& st) {
    state_wait(st);
    state_throw_asynchronous(st);
}

void state_sync_host(QueueState& st) { wait_host_lane(st); }

void state_sync_host_and_throw(QueueState& st) {
    state_sync_host(st);
    state_throw_asynchronous(st);
}

void state_throw_asynchronous(QueueState& st) {
    for (auto* p : snapshot(st)) {
        p->inorder.throw_asynchronous();
        p->concurrent.throw_asynchronous();
    }
    st.host_lane.throw_asynchronous();

    // A host task's exception is asynchronous in exactly SYCL's sense, but the
    // runtime has no way to be told about it, so the stream keeps its own list.
    std::exception_ptr first;
    {
        std::lock_guard<std::mutex> lock{st.mutex};
        if (!st.host_errors.empty()) {
            first = st.host_errors.front();
            st.host_errors.erase(st.host_errors.begin());
        }
    }
    if (first) std::rethrow_exception(first);
}

std::vector<sycl::event> state_wait_list(QueueState& st) {
    std::vector<sycl::event> out;
    for (auto* p : snapshot(st)) {
        for (auto& e : p->inorder.get_wait_list()) out.push_back(e);
        for (auto& e : p->concurrent.get_wait_list()) out.push_back(e);
    }
    for (auto& e : st.host_lane.get_wait_list()) out.push_back(e);
    return out;
}

QueueState& state_of(const queue& q) { return *q.state_; }

sycl::queue backend_queue_for(QueueState& st, const sycl::device& dev) {
    std::lock_guard<std::mutex> lock{st.mutex};
    return st.pair_for(st.backend_index_of(dev)).concurrent;
}

bool is_host_backend(const sycl::device& dev) {
    return dev.get_backend() == sycl::backend::omp;
}

bool retarget_device(const sycl::property_list& props, sycl::device& out) {
    if (!props.has_property<sycl::property::command_group::AdaptiveCpp_retarget>()) return false;
    out = props.get_property<sycl::property::command_group::AdaptiveCpp_retarget>().dev;
    return true;
}

}  // namespace naga::slabkit::detail

namespace naga::slabkit {

queue::queue(context ctx, sycl::async_handler handler)
    : state_{detail::make_queue_state(std::move(ctx), std::move(handler))} {}

}  // namespace naga::slabkit
