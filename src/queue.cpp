// =============================================================================
// queue.cpp — the compiled core of the stream
// =============================================================================
//
// Nothing here is a template. The public queue's submit and its shortcuts are
// thin templates that call begin_item, submit to the sycl::queue they are
// handed, and return the event -- so a consumer compiles the kernel and almost
// nothing else of slabkit.
//
// The memory tasks are compiled whole: locating the operands, choosing the
// route and issuing its legs are the same for every element type, and a pattern
// is bytes by the time it arrives here.

#include "naga/slabkit/queue.hpp"

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "detail/os_alloc.hpp"
#include "detail/queue_state.hpp"
#include "naga/slabkit/context.hpp"

namespace naga::slabkit::detail {

namespace {

const sycl::property_list kPrimaryProps{sycl::property::queue::in_order{},
                                        sycl::property::queue::AdaptiveCpp_retargetable{}};
const sycl::property_list kConcurrentProps{sycl::property::queue::AdaptiveCpp_retargetable{}};

sycl::queue make_lane(const sycl::async_handler& handler, const sycl::property_list& props) {
    return handler ? sycl::queue{handler, props} : sycl::queue{props};
}

void append(std::vector<sycl::event>& into, std::vector<sycl::event> from) {
    into.insert(into.end(), from.begin(), from.end());
}

void append(std::vector<sycl::event>& into, const std::vector<sycl::event>* from) {
    if (from) into.insert(into.end(), from->begin(), from->end());
}

}  // namespace

// -----------------------------------------------------------------------------
// The stream state
// -----------------------------------------------------------------------------

void HostErrors::push(std::exception_ptr e) {
    std::lock_guard<std::mutex> lock{mutex_};
    errors_.push_back(std::move(e));
}

std::exception_ptr HostErrors::pop() {
    std::lock_guard<std::mutex> lock{mutex_};
    if (errors_.empty()) return {};
    std::exception_ptr first = errors_.front();
    errors_.erase(errors_.begin());
    return first;
}

QueueState::QueueState(context c, const sycl::device* dev, sycl::async_handler handler)
    : ctx{std::move(c)}
    // SYCL's default device (the one a default-constructed sycl::queue picks), never the context's first device.
    , default_device{dev ? *dev : sycl::device{sycl::default_selector_v}}
    , primary{make_lane(handler, kPrimaryProps)}
    , group_queue{make_lane(handler, kConcurrentProps)}
    , host_queue{make_lane(handler, kConcurrentProps)}
    , legs{make_lane(handler, kConcurrentProps)}
    , errors{std::make_shared<HostErrors>()} {
    if (!ctx.contains(default_device))
        throw std::invalid_argument{
            dev ? "naga::slabkit::queue: the default device is not in this queue's context"
                : "naga::slabkit::queue: SYCL's default device is not in this queue's context; "
                  "name the default device with queue(ctx, dev)"};
}

const sycl::device& QueueState::device_or_default(const sycl::device* dev) const {
    return dev ? *dev : default_device;
}

sycl::property_list QueueState::place(const sycl::device* dev) const {
    const sycl::device& d = device_or_default(dev);
    return sycl::property_list{
        sycl::property::command_group::AdaptiveCpp_retarget{ctx.context_of(d), d}};
}

GroupState::GroupState(QueueState& st, std::vector<sycl::event> deps)
    : state{st}, captured{std::move(deps)} {}

void GroupState::record(const sycl::event& e) {
    std::lock_guard<std::mutex> lock{mutex_};
    events_.push_back(e);
}

std::vector<sycl::event> GroupState::events() const {
    std::lock_guard<std::mutex> lock{mutex_};
    return events_;
}

// -----------------------------------------------------------------------------
// The core the public templates call
// -----------------------------------------------------------------------------

std::shared_ptr<QueueState> make_queue_state(context ctx, const sycl::device* dev,
                                             sycl::async_handler handler) {
    return std::make_shared<QueueState>(std::move(ctx), dev, std::move(handler));
}

submission begin_item(QueueState& st, lane l, const sycl::device* dev,
                      const std::vector<sycl::event>* extra) {
    submission          s;
    const sycl::device& target = st.device_or_default(dev);
    s.props                    = st.place(dev);

    if (l == lane::primary) {
        // The primary is in order with itself, so it merges the other two only.
        s.target = &st.primary;
        append(s.deps, st.group_queue.get_wait_list());
        append(s.deps, st.host_queue.get_wait_list());
    } else {
        // Host and memory tasks are asynchronous with each other: this queue's
        // own outstanding work is deliberately not merged.
        s.target = &st.host_queue;
        append(s.deps, st.primary.get_wait_list());
        append(s.deps, st.group_queue.get_wait_list());
    }

    append(s.deps, extra);
    return s;
}

std::shared_ptr<GroupState> open_group(QueueState& st) {
    std::vector<sycl::event> deps;
    append(deps, st.primary.get_wait_list());
    append(deps, st.group_queue.get_wait_list());
    append(deps, st.host_queue.get_wait_list());
    return std::make_shared<GroupState>(st, std::move(deps));
}

submission begin_group_item(GroupState& gs, const sycl::device* dev,
                            const std::vector<sycl::event>* extra) {
    submission s;
    s.target = &gs.state.group_queue;
    s.props  = gs.state.place(dev);
    s.deps   = gs.captured;
    append(s.deps, extra);
    return s;
}

void record_group_event(GroupState& gs, const sycl::event& e) { gs.record(e); }

std::vector<sycl::event> group_events(const GroupState& gs) { return gs.events(); }

std::shared_ptr<HostErrors> errors_of(QueueState& st) { return st.errors; }

void record_error(HostErrors& errors, std::exception_ptr e) { errors.push(std::move(e)); }

// -----------------------------------------------------------------------------
// Memory work
// -----------------------------------------------------------------------------

namespace {

/// Where a memory task is submitted: a lane of the stream, or a group of it.
/// Both resolve a submission the same way; only a group records its subtasks.
struct Sink {
    QueueState* st = nullptr;
    GroupState* gs = nullptr;
    lane        l  = lane::host_mem;

    QueueState& state() const { return gs ? gs->state : *st; }

    submission begin(const sycl::device* dev, const std::vector<sycl::event>* extra) const {
        return gs ? begin_group_item(*gs, dev, extra) : begin_item(*st, l, dev, extra);
    }

    void record(const sycl::event& e) const {
        if (gs) gs->record(e);
    }
};

struct operand {
    kind         k;
    sycl::device dev;
    /// True when the bytes can be reached with a plain host store: a pointer no
    /// context knows, an allocation of the HOST kind, or one on the OpenMP
    /// backend.
    bool host_side;
};

/// Where an operand lives. A pointer no backend context knows is a HOST operand
/// -- always; there is no unrecognised operand, which is what makes an ordinary
/// std::vector a legal source or destination.
///
/// A HOST-kind allocation is host-side whatever context it belongs to. It has
/// to be: a copy between two of them, submitted to a device queue, resolves
/// both operands to the host device and becomes a node on the OpenMP backend --
/// which the next kernel on that device would then have to wait for, and a
/// non-OpenMP kernel that waits for an OpenMP node never starts.
///
/// `fallback` is the stream's default device (SYCL's default device, never the
/// context's first); it stands for the unrecognised host pointer.
operand locate(const context& c, const sycl::device& fallback, const void* p) {
    try {
        const sycl::device d = c.get_pointer_device(p);
        const kind         k = c.get_pointer_type(p);
        return operand{k, d, k == kind::host || is_host_backend(d)};
    } catch (const sycl::exception&) {
        return operand{kind::host, fallback, true};
    }
}

/// One host callable on the sink's queue, with the sink's ordering.
template <class F>
sycl::event host_item(const Sink& s, const sycl::device* dev,
                      const std::vector<sycl::event>* extra, F f) {
    submission sub = s.begin(dev, extra);
    return sub.target->submit(sub.props, [&](sycl::handler& h) {
        h.depends_on(sub.deps);
        h.async_host(f);
    });
}

/// A task with nothing to do is still a task, so a caller who waits on the
/// returned event waits for the same thing either way.
sycl::event no_op(const Sink& s, const sycl::device* dev,
                  const std::vector<sycl::event>* extra) {
    sycl::event e = host_item(s, dev, extra, [] {});
    s.record(e);
    return e;
}

/// True when the two device-side operands can be copied between with one
/// backend copy: one context over both devices, and a direct route between them.
bool direct_between(const operand& from, const operand& to) {
    return context::copy_routes(from.k, from.dev, to.k, to.dev).front().method ==
           route::via::d2d;
}

sycl::event copy_on(const Sink& s, const sycl::device* dev, void* dst, const void* src,
                    std::size_t bytes, const std::vector<sycl::event>* extra) {
    if (dst == nullptr || src == nullptr || bytes == 0) return no_op(s, dev, extra);

    const context&      c    = s.state().ctx;
    const sycl::device& dflt = s.state().default_device;
    const operand       from = locate(c, dflt, src);
    const operand       to   = locate(c, dflt, dst);

    // Both ends are ordinary memory: a host copy, and no submission.
    if (from.host_side && to.host_side) {
        sycl::event e = host_item(s, dev, extra,
                                  [dst, src, bytes] { std::memcpy(dst, src, bytes); });
        s.record(e);
        return e;
    }

    // One backend copy. A named device is where it is ISSUED -- placing it
    // validly is the caller's -- and the device operand decides otherwise. A
    // host operand is read when the copy RUNS, which is SYCL's own contract
    // for queue::memcpy: the caller owns that memory until the event
    // completes.
    if (from.host_side || to.host_side || direct_between(from, to)) {
        const sycl::device& on  = dev ? *dev : (to.host_side ? from.dev : to.dev);
        submission          sub = s.begin(&on, extra);
        sycl::event         e   = sub.target->submit(sub.props, [&](sycl::handler& h) {
            h.depends_on(sub.deps);
            h.memcpy(dst, src, bytes);
        });
        s.record(e);
        return e;
    }

    // Two device operands with no context over both: down into a host
    // intermediate, then up. Both legs are issued from INSIDE one host task,
    // which waits each of them, so the second never carries a dependency across
    // a backend and the caller is never blocked by either. The intermediate
    // dies with the task's closure, after the copy that read it.
    sycl::queue          legs = s.state().legs;
    const sycl::context& sctx = s.state().ctx.context_of(from.dev);
    const sycl::context& dctx = s.state().ctx.context_of(to.dev);
    const sycl::device   sdev = from.dev;
    const sycl::device   ddev = to.dev;

    sycl::event e = host_item(s, nullptr, extra, [legs, sctx, dctx, sdev, ddev, dst, src,
                                                  bytes]() mutable {
        auto stage = std::vector<std::byte>(bytes);
        legs.submit(sycl::property_list{
                        sycl::property::command_group::AdaptiveCpp_retarget{sctx, sdev}},
                    [&](sycl::handler& h) { h.memcpy(stage.data(), src, bytes); })
            .wait();
        legs.submit(sycl::property_list{
                        sycl::property::command_group::AdaptiveCpp_retarget{dctx, ddev}},
                    [&](sycl::handler& h) { h.memcpy(dst, stage.data(), bytes); })
            .wait();
    });
    s.record(e);
    return e;
}

sycl::event set_on(const Sink& s, const sycl::device* dev, void* p, int value,
                   std::size_t bytes, const std::vector<sycl::event>* extra) {
    if (p == nullptr || bytes == 0) return no_op(s, dev, extra);

    const operand o = locate(s.state().ctx, s.state().default_device, p);
    if (o.host_side) {
        sycl::event e =
            host_item(s, dev, extra, [p, value, bytes] { std::memset(p, value, bytes); });
        s.record(e);
        return e;
    }

    const sycl::device& on  = dev ? *dev : o.dev;
    submission          sub = s.begin(&on, extra);
    sycl::event         e   = sub.target->submit(sub.props, [&](sycl::handler& h) {
        h.depends_on(sub.deps);
        h.memset(p, value, bytes);
    });
    s.record(e);
    return e;
}

/// The pattern is COPIED here, on the calling thread: the task runs later and
/// the caller's `const T&` is gone by then.
std::shared_ptr<std::vector<std::byte>> own_pattern(const void* pattern, std::size_t size) {
    auto out = std::make_shared<std::vector<std::byte>>(size);
    std::memcpy(out->data(), pattern, size);
    return out;
}

sycl::event fill_on(const Sink& s, const sycl::device* dev, void* p, const void* pattern,
                    std::size_t pattern_size, std::size_t count,
                    const std::vector<sycl::event>* extra) {
    if (p == nullptr || pattern == nullptr || pattern_size == 0 || count == 0)
        return no_op(s, dev, extra);

    const std::size_t total = pattern_size * count;
    const std::byte*  pat   = static_cast<const std::byte*>(pattern);

    // A pattern of identical bytes -- what value-initializing anything zeroable
    // looks like -- is a memset, and needs no kernel and no staging.
    bool uniform = true;
    for (std::size_t i = 1; i < pattern_size; ++i) uniform = uniform && (pat[i] == pat[0]);
    if (uniform)
        return set_on(s, dev, p, static_cast<int>(std::to_integer<unsigned char>(pat[0])), total,
                      extra);

    const operand     o     = locate(s.state().ctx, s.state().default_device, p);
    auto*             out   = static_cast<std::byte*>(p);
    const std::size_t psize = pattern_size;
    auto              own   = own_pattern(pattern, psize);

    if (o.host_side) {
        sycl::event e = host_item(s, dev, extra, [out, own, psize, total] {
            for (std::size_t i = 0; i < total; ++i) out[i] = (*own)[i % psize];
        });
        s.record(e);
        return e;
    }

    // A pattern that fits in a register is copied INTO the kernel, so there is
    // no staging allocation and no second submission.
    if (psize <= sizeof(std::uint64_t)) {
        std::uint64_t packed = 0;
        std::memcpy(&packed, pat, psize);

        const sycl::device& on  = dev ? *dev : o.dev;
        submission          sub = s.begin(&on, extra);
        sycl::event e   = sub.target->submit(sub.props, [&](sycl::handler& h) {
            h.depends_on(sub.deps);
            h.parallel_for(sycl::range<1>{total}, [=](sycl::id<1> i) {
                const auto* bytes = reinterpret_cast<const std::byte*>(&packed);
                out[i[0]]         = bytes[i[0] % psize];
            });
        });
        s.record(e);
        return e;
    }

    // A wider one is staged in device memory for the kernel to replicate.
    const sycl::context& sctx = s.state().ctx.context_of(o.dev);
    auto                 stage = std::make_shared<unique_usm_ptr<std::byte[]>>(
        os_allocate(kind::device, 0, psize, o.dev, sctx, {}));
    if (!*stage) throw std::bad_alloc{};

    std::byte* staged = stage->get();

    submission  up   = s.begin(&o.dev, extra);
    sycl::event sent = up.target->submit(up.props, [&](sycl::handler& h) {
        h.depends_on(up.deps);
        h.memcpy(staged, own->data(), psize);
    });

    const std::vector<sycl::event> after_send{sent};
    submission                     run = s.begin(&o.dev, &after_send);
    sycl::event                    e   = run.target->submit(run.props, [&](sycl::handler& h) {
        h.depends_on(run.deps);
        h.parallel_for(sycl::range<1>{total},
                       [=](sycl::id<1> i) { out[i[0]] = staged[i[0] % psize]; });
    });

    // The pattern and its staging allocation are released after the fill.
    const std::vector<sycl::event> after_fill{e};
    sycl::event last = host_item(s, nullptr, &after_fill, [stage, own] {});
    s.record(last);
    return last;
}

Sink lane_sink(QueueState& st, lane l) { return Sink{&st, nullptr, l}; }
Sink group_sink(GroupState& gs) { return Sink{nullptr, &gs, lane::host_mem}; }

}  // namespace

sycl::event copy_bytes(QueueState& st, lane l, const sycl::device* dev, void* dst,
                       const void* src, std::size_t bytes,
                       const std::vector<sycl::event>* extra) {
    return copy_on(lane_sink(st, l), dev, dst, src, bytes, extra);
}

sycl::event copy_bytes(GroupState& gs, const sycl::device* dev, void* dst, const void* src,
                       std::size_t bytes, const std::vector<sycl::event>* extra) {
    return copy_on(group_sink(gs), dev, dst, src, bytes, extra);
}

sycl::event set_bytes(QueueState& st, lane l, const sycl::device* dev, void* p, int value,
                      std::size_t bytes, const std::vector<sycl::event>* extra) {
    return set_on(lane_sink(st, l), dev, p, value, bytes, extra);
}

sycl::event set_bytes(GroupState& gs, const sycl::device* dev, void* p, int value,
                      std::size_t bytes, const std::vector<sycl::event>* extra) {
    return set_on(group_sink(gs), dev, p, value, bytes, extra);
}

sycl::event fill_bytes(QueueState& st, lane l, const sycl::device* dev, void* p,
                       const void* pattern, std::size_t pattern_size, std::size_t count,
                       const std::vector<sycl::event>* extra) {
    return fill_on(lane_sink(st, l), dev, p, pattern, pattern_size, count, extra);
}

sycl::event fill_bytes(GroupState& gs, const sycl::device* dev, void* p, const void* pattern,
                       std::size_t pattern_size, std::size_t count,
                       const std::vector<sycl::event>* extra) {
    return fill_on(group_sink(gs), dev, p, pattern, pattern_size, count, extra);
}

// -----------------------------------------------------------------------------
// Waiting, errors, introspection
// -----------------------------------------------------------------------------

const context&      state_context(const QueueState& st) { return st.ctx; }
const sycl::device& state_device(const QueueState& st) { return st.default_device; }
QueueState&         state_of_group(GroupState& gs) { return gs.state; }

void state_wait(QueueState& st) {
    st.primary.wait();
    st.group_queue.wait();
    st.host_queue.wait();
}

void state_wait_and_throw(QueueState& st) {
    state_wait(st);
    state_throw_asynchronous(st);
}

void state_throw_asynchronous(QueueState& st) {
    st.primary.throw_asynchronous();
    st.group_queue.throw_asynchronous();
    st.host_queue.throw_asynchronous();

    // A host callable's exception is asynchronous in exactly SYCL's sense, but
    // the runtime has no way to be told about it, so the stream keeps its own.
    if (std::exception_ptr first = st.errors->pop()) std::rethrow_exception(first);
}

std::vector<sycl::event> state_wait_list(QueueState& st) {
    std::vector<sycl::event> out;
    append(out, st.primary.get_wait_list());
    append(out, st.group_queue.get_wait_list());
    append(out, st.host_queue.get_wait_list());
    return out;
}

QueueState& state_of(const queue& q) { return *q.state_; }

bool is_host_backend(const sycl::device& dev) { return dev.get_backend() == sycl::backend::omp; }

bool retarget_device(const sycl::property_list& props, sycl::device& out) {
    if (!props.has_property<sycl::property::command_group::AdaptiveCpp_retarget>()) return false;
    out = props.get_property<sycl::property::command_group::AdaptiveCpp_retarget>().dev;
    return true;
}

}  // namespace naga::slabkit::detail
