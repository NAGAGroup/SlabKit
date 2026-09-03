#pragma once

// =============================================================================
// naga/slabkit/queue.hpp — one stream of work over a context
// =============================================================================
//
// sycl::queue's surface, with one stream semantics that hold ACROSS backends.
//
// TWO KINDS OF WORK. Kernel work is what runs on a device. Host work is what
// runs on the host -- including every memory operation, since a copy is
// host-issued. Kernel work is in order with kernel work; host work is PARALLEL
// with other host work; the two interlock, so all outstanding host work
// completes before any kernel work begins and all outstanding kernel work
// completes before any host work begins. The stream alternates phases, each
// internally parallel, and behaves in order from outside.
//
// A GROUP opts out of the interlock: its members depend on the step before it
// and on none of each other, whatever kind they are. A CHAIN is the opposite --
// host items forced in order with each other, counting as ONE host item to the
// stream. A staged copy is exactly that.
//
// INTERNALLY, per backend of the context and created lazily: one in-order and
// one out-of-order sycl::queue, both AdaptiveCpp_retargetable, on that backend's
// default context. Plus one out-of-order HOST LANE on the OpenMP backend, which
// always exists.
//
// HOST WORK IS AN OPENMP single_task whose body is an INDIRECT call through a
// function pointer into ordinary host code -- the host task AdaptiveCpp never
// gave us. A direct call to complex host code from a kernel fails the generic
// flow's JIT; an indirect call does not, and the two captured values (a function
// pointer and a void*) are trivially copyable, which kernel captures must be.
//
// ORDERING ACROSS BACKENDS is by depends_on, and it holds, because the consumer
// of every cross-backend dependency is the host lane, which honours foreign
// events. slabkit never submits a NON-OpenMP operation carrying a dependency on
// another backend's event: those edges are paid with a host wait before the
// submission. That restriction is load-bearing. A copy whose host operand is not
// USM in the issuing queue's context has that operand read when the copy is
// ISSUED -- the driver stages pageable memory through a pinned buffer at issue
// time -- so a stream-level dependency cannot protect it, and was measured wrong
// 40 times in 40, indistinguishable from no dependency at all.
//
// WHAT A HOST TASK MUST NOT DO. It must never submit work to the OpenMP backend
// and wait on it: the wait holds the very lane the submitted work needs, and
// nothing completes. Work on other backends may be issued and waited freely. For
// the same reason a host task must not hold the LAST handle to its own stream:
// the stream's destructor drains the host lane, and draining it from a task
// running on it is that same self-wait.
//
// THREAD SAFETY. Every member is safe to call from any thread: one internal
// mutex guards the backend-pair map and the stream's step state, and it is never
// held while waiting on or submitting to a sycl::queue.

#include <cstddef>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/slabkit/context.hpp"

namespace naga::slabkit {

class queue;

namespace detail {

class QueueState;

/// An event, and the index of the backend it ran on. The stream has to remember
/// the backend because a NON-OpenMP consumer cannot be handed a foreign event.
using annotated_event = std::pair<sycl::event, std::size_t>;

/// Everything a kernel submission needs, resolved under the state mutex and
/// handed back before anything is submitted.
///
/// `deps` are same-backend and go to depends_on. `host_waits` are the ones from
/// ANOTHER backend, which are waited on the host BEFORE submitting -- the price
/// of the restriction above, paid only at a transition.
struct submission {
    sycl::queue*             target;
    std::size_t              backend;
    std::vector<sycl::event> deps;
    std::vector<sycl::event> host_waits;
};

/// A type-erased host callable: run it, then destroy it. Neither pointer is ever
/// seen by a kernel; only the trampoline's own address is.
struct HostFn {
    void (*run)(void*);
    void (*destroy)(void*);
    void* payload;
};

template <class F>
HostFn make_host_fn(F&& f) {
    using Fn = std::decay_t<F>;
    return HostFn{+[](void* p) { (*static_cast<Fn*>(p))(); },
                  +[](void* p) { delete static_cast<Fn*>(p); }, new Fn(std::forward<F>(f))};
}

/// The deps a group's members share, split by kind: kernel members wait for
/// both, host members wait for the kernel side only, because a host item must
/// never depend on another host item.
struct group_deps {
    std::vector<annotated_event> kernel;
    std::vector<sycl::event>     host;
};

/// A chain's cursor. Each `then` waits only for the item before it.
struct ChainState {
    QueueState*              st = nullptr;
    std::vector<sycl::event> deps;
    sycl::event              last;
    bool                     started = false;
};

std::shared_ptr<QueueState> make_queue_state(context ctx, sycl::async_handler handler);

/// \param dev nullptr for the stream's default device.
/// \param concurrent true selects the backend's out-of-order queue (groups).
submission begin_item(QueueState& st, const sycl::device* dev, bool concurrent);

/// The same, for a kernel member of a group: the deps are the group's, not the
/// stream's, and they include the host side, because a group opts its MEMBERS
/// out of the interlock, not the group itself.
submission begin_group_item(QueueState& st, const sycl::device* dev, const group_deps& deps);

/// Record `e`, run on backend `b`, as the whole of the current kernel step.
void end_item(QueueState& st, const sycl::event& e, std::size_t b);

group_deps begin_group(QueueState& st);
void       end_group(QueueState& st, std::vector<annotated_event> kernel_events,
                     std::vector<sycl::event> host_events);

/// One host item of the stream: ordered after the outstanding kernel work,
/// parallel with every other host item.
sycl::event submit_host_item(QueueState& st, HostFn fn);

/// One host item of a group: ordered after the group's kernel deps only.
sycl::event submit_group_host_item(QueueState& st, const std::vector<annotated_event>& deps,
                                   HostFn fn);

/// Record an already-submitted host event as one outstanding host item of the
/// stream. A chain's last event is the only thing that reaches this.
void register_host_item(QueueState& st, const sycl::event& e);

ChainState  begin_chain(QueueState& st);
ChainState  begin_chain_with(QueueState& st, std::vector<annotated_event> kernel_deps);
sycl::event chain_then(ChainState& cs, HostFn fn);
/// Guarantees the chain has at least one item and returns its last event.
sycl::event finish_chain(ChainState& cs);

const context&      state_context(const QueueState& st);
const sycl::device& state_device(const QueueState& st);

void                     state_wait(QueueState& st);
void                     state_wait_and_throw(QueueState& st);
void                     state_sync_host(QueueState& st);
void                     state_sync_host_and_throw(QueueState& st);
void                     state_throw_asynchronous(QueueState& st);
std::vector<sycl::event> state_wait_list(QueueState& st);

/// The stream behind a handle. Internal: the routed mirrors need the backend
/// queues, which the public surface deliberately does not expose.
QueueState& state_of(const queue& q);

/// The backend queue for `dev`, as a copy a host task may hold: the one way
/// host work reaches a device without going through the stream's ordering. A
/// host task must NOT hold the slabkit queue instead -- dropping the last handle
/// from inside a task would make the stream's destructor drain the very lane it
/// is running on.
sycl::queue backend_queue_for(QueueState& st, const sycl::device& dev);

/// True when `dev` belongs to the OpenMP backend -- the one backend a host task
/// may not submit to and wait on.
bool is_host_backend(const sycl::device& dev);

/// The device of an AdaptiveCpp_retarget property, if the list carries one.
bool retarget_device(const sycl::property_list& props, sycl::device& out);

}  // namespace detail

class queue {
public:
    explicit queue(context ctx, sycl::async_handler handler = {});

    const context&      get_context() const { return detail::state_context(*state_); }
    const sycl::device& get_device() const { return detail::state_device(*state_); }

    /// Two handles compare equal when they are the same stream. Containers and
    /// allocators lean on this: a cross-stream operation is a usage error, and
    /// this is how it is recognised.
    friend bool operator==(const queue& a, const queue& b) { return a.state_ == b.state_; }
    friend bool operator!=(const queue& a, const queue& b) { return !(a == b); }

    // -------------------------------------------------------------------------
    // submit
    // -------------------------------------------------------------------------

    template <class CGF>
    sycl::event submit(CGF&& cgf) {
        return enqueue(nullptr, nullptr, std::forward<CGF>(cgf));
    }

    template <class CGF>
    sycl::event submit(const sycl::property_list& props, CGF&& cgf) {
        sycl::device dev;
        if (detail::retarget_device(props, dev))
            return enqueue(&dev, nullptr, std::forward<CGF>(cgf));
        return enqueue(nullptr, nullptr, std::forward<CGF>(cgf));
    }

    template <class CGF>
    sycl::event submit(const sycl::device& dev, CGF&& cgf) {
        return enqueue(&dev, nullptr, std::forward<CGF>(cgf));
    }

    template <class CGF>
    sycl::event submit(const std::vector<sycl::event>& deps, CGF&& cgf) {
        return enqueue(nullptr, &deps, std::forward<CGF>(cgf));
    }

    template <class CGF>
    sycl::event submit(const sycl::device& dev, const std::vector<sycl::event>& deps, CGF&& cgf) {
        return enqueue(&dev, &deps, std::forward<CGF>(cgf));
    }

    // -------------------------------------------------------------------------
    // single_task
    // -------------------------------------------------------------------------

    template <class K>
    sycl::event single_task(const K& k) {
        return body(nullptr, nullptr, [&](sycl::handler& h) { h.single_task(k); });
    }
    template <class K>
    sycl::event single_task(const sycl::device& dev, const K& k) {
        return body(&dev, nullptr, [&](sycl::handler& h) { h.single_task(k); });
    }
    template <class K>
    sycl::event single_task(const std::vector<sycl::event>& deps, const K& k) {
        return body(nullptr, &deps, [&](sycl::handler& h) { h.single_task(k); });
    }
    template <class K>
    sycl::event single_task(const sycl::device& dev, const std::vector<sycl::event>& deps,
                            const K& k) {
        return body(&dev, &deps, [&](sycl::handler& h) { h.single_task(k); });
    }

    // -------------------------------------------------------------------------
    // parallel_for — range, range+offset and nd_range, each with the four forms.
    // The trailing pack is sycl's ReductionsAndKernel: reductions then kernel.
    // -------------------------------------------------------------------------

    // ---- range --------------------------------------------------------------
    template <class... RK, int Dim>
    sycl::event parallel_for(sycl::range<Dim> r, const RK&... rk) {
        return body(nullptr, nullptr, [&](sycl::handler& h) { h.parallel_for(r, rk...); });
    }
    template <class... RK, int Dim>
    sycl::event parallel_for(const sycl::device& dev, sycl::range<Dim> r, const RK&... rk) {
        return body(&dev, nullptr, [&](sycl::handler& h) { h.parallel_for(r, rk...); });
    }
    template <class... RK, int Dim>
    sycl::event parallel_for(const std::vector<sycl::event>& deps, sycl::range<Dim> r,
                             const RK&... rk) {
        return body(nullptr, &deps, [&](sycl::handler& h) { h.parallel_for(r, rk...); });
    }
    template <class... RK, int Dim>
    sycl::event parallel_for(const sycl::device& dev, const std::vector<sycl::event>& deps,
                             sycl::range<Dim> r, const RK&... rk) {
        return body(&dev, &deps, [&](sycl::handler& h) { h.parallel_for(r, rk...); });
    }

    // ---- range + offset ------------------------------------------------------
    template <class... RK, int Dim>
    sycl::event parallel_for(sycl::range<Dim> r, sycl::id<Dim> off, const RK&... rk) {
        return body(nullptr, nullptr, [&](sycl::handler& h) { h.parallel_for(r, off, rk...); });
    }
    template <class... RK, int Dim>
    sycl::event parallel_for(const sycl::device& dev, sycl::range<Dim> r, sycl::id<Dim> off,
                             const RK&... rk) {
        return body(&dev, nullptr, [&](sycl::handler& h) { h.parallel_for(r, off, rk...); });
    }
    template <class... RK, int Dim>
    sycl::event parallel_for(const std::vector<sycl::event>& deps, sycl::range<Dim> r,
                             sycl::id<Dim> off, const RK&... rk) {
        return body(nullptr, &deps, [&](sycl::handler& h) { h.parallel_for(r, off, rk...); });
    }
    template <class... RK, int Dim>
    sycl::event parallel_for(const sycl::device& dev, const std::vector<sycl::event>& deps,
                             sycl::range<Dim> r, sycl::id<Dim> off, const RK&... rk) {
        return body(&dev, &deps, [&](sycl::handler& h) { h.parallel_for(r, off, rk...); });
    }

    // ---- nd_range ------------------------------------------------------------
    template <class... RK, int Dim>
    sycl::event parallel_for(sycl::nd_range<Dim> nd, const RK&... rk) {
        return body(nullptr, nullptr, [&](sycl::handler& h) { h.parallel_for(nd, rk...); });
    }
    template <class... RK, int Dim>
    sycl::event parallel_for(const sycl::device& dev, sycl::nd_range<Dim> nd, const RK&... rk) {
        return body(&dev, nullptr, [&](sycl::handler& h) { h.parallel_for(nd, rk...); });
    }
    template <class... RK, int Dim>
    sycl::event parallel_for(const std::vector<sycl::event>& deps, sycl::nd_range<Dim> nd,
                             const RK&... rk) {
        return body(nullptr, &deps, [&](sycl::handler& h) { h.parallel_for(nd, rk...); });
    }
    template <class... RK, int Dim>
    sycl::event parallel_for(const sycl::device& dev, const std::vector<sycl::event>& deps,
                             sycl::nd_range<Dim> nd, const RK&... rk) {
        return body(&dev, &deps, [&](sycl::handler& h) { h.parallel_for(nd, rk...); });
    }

    // -------------------------------------------------------------------------
    // Host work — on the host lane, off the caller's thread
    // -------------------------------------------------------------------------

    /// One host item: it begins after the kernel work outstanding when it was
    /// submitted, and runs in parallel with every other host item. The returned
    /// event completes when the callable returns. An exception escaping the
    /// callable is asynchronous: it surfaces from wait_and_throw,
    /// sync_host_work_and_throw or throw_asynchronous, never from here.
    template <class F>
    sycl::event submit_host(F&& f) {
        return detail::submit_host_item(*state_, detail::make_host_fn(std::forward<F>(f)));
    }

    /// Host items forced in order with each other: the opposite of a group. The
    /// chain is ONE host item as far as the stream is concerned.
    class chain {
    public:
        template <class F>
        sycl::event then(F&& f) {
            return detail::chain_then(*cs_, detail::make_host_fn(std::forward<F>(f)));
        }

    private:
        friend class queue;
        explicit chain(detail::ChainState& cs) : cs_{&cs} {}
        detail::ChainState* cs_;
    };

    /// f(chain&). Returns the chain's LAST event, which is the item the stream
    /// records.
    template <class F>
    sycl::event submit_chain(F&& f) {
        detail::ChainState cs = detail::begin_chain(*state_);
        chain              c{cs};
        f(c);
        const sycl::event e = detail::finish_chain(cs);
        detail::register_host_item(*state_, e);
        return e;
    }

    // -------------------------------------------------------------------------
    // Synchronization
    // -------------------------------------------------------------------------

    void wait() { detail::state_wait(*state_); }
    void wait_and_throw() { detail::state_wait_and_throw(*state_); }
    void sync_host_work() { detail::state_sync_host(*state_); }
    void sync_host_work_and_throw() { detail::state_sync_host_and_throw(*state_); }
    void throw_asynchronous() { detail::state_throw_asynchronous(*state_); }
    std::vector<sycl::event> get_wait_list() const { return detail::state_wait_list(*state_); }

    // -------------------------------------------------------------------------
    // group — one step of concurrent work, of either kind
    // -------------------------------------------------------------------------

    class group {
    public:
        template <class CGF>
        sycl::event submit(CGF&& cgf) {
            return enqueue(nullptr, std::forward<CGF>(cgf));
        }

        template <class CGF>
        sycl::event submit(const sycl::property_list& props, CGF&& cgf) {
            sycl::device dev;
            if (detail::retarget_device(props, dev)) return enqueue(&dev, std::forward<CGF>(cgf));
            return enqueue(nullptr, std::forward<CGF>(cgf));
        }

        template <class CGF>
        sycl::event submit(const sycl::device& dev, CGF&& cgf) {
            return enqueue(&dev, std::forward<CGF>(cgf));
        }

        /// A host member. It waits for the step before the group and for no
        /// member of it, this one's own kind included.
        template <class F>
        sycl::event submit_host(F&& f) {
            sycl::event e = detail::submit_group_host_item(
                *state_, deps_.kernel, detail::make_host_fn(std::forward<F>(f)));
            std::lock_guard<std::mutex> lock{mutex_};
            host_events_.push_back(e);
            return e;
        }

        /// A chain as a member: it contributes its LAST event to the step.
        template <class F>
        sycl::event submit_chain(F&& f) {
            detail::ChainState cs = detail::begin_chain_with(*state_, deps_.kernel);
            chain              c{cs};
            f(c);
            sycl::event e = detail::finish_chain(cs);

            std::lock_guard<std::mutex> lock{mutex_};
            host_events_.push_back(e);
            return e;
        }

        std::vector<sycl::event> events() const {
            std::lock_guard<std::mutex> lock{mutex_};
            std::vector<sycl::event>    out;
            out.reserve(kernel_events_.size() + host_events_.size());
            for (const auto& [e, b] : kernel_events_) out.push_back(e);
            for (const auto& e : host_events_) out.push_back(e);
            return out;
        }

    private:
        friend class queue;
        group(detail::QueueState& st, detail::group_deps deps)
            : state_{&st}, deps_{std::move(deps)} {}

        template <class CGF>
        sycl::event enqueue(const sycl::device* dev, CGF&& cgf) {
            detail::submission s = detail::begin_group_item(*state_, dev, deps_);

            // A dependency the target backend will not honour is waited here.
            for (auto& e : s.host_waits) e.wait();

            sycl::event e = s.target->submit(
                dev ? sycl::property_list{sycl::property::command_group::AdaptiveCpp_retarget{*dev}}
                    : sycl::property_list{},
                [&](sycl::handler& h) {
                    h.depends_on(s.deps);
                    cgf(h);
                });

            std::lock_guard<std::mutex> lock{mutex_};
            kernel_events_.emplace_back(e, s.backend);
            return e;
        }

        detail::QueueState*                  state_;
        detail::group_deps                   deps_;
        mutable std::mutex                   mutex_;
        std::vector<detail::annotated_event> kernel_events_;
        std::vector<sycl::event>             host_events_;
    };

    /// f(group&). Returns the step's events, which the next item will wait on.
    template <class F>
    std::vector<sycl::event> submit_group(F&& f) {
        group g{*state_, detail::begin_group(*state_)};
        f(g);

        std::vector<detail::annotated_event> kernel;
        std::vector<sycl::event>             host;
        {
            std::lock_guard<std::mutex> lock{g.mutex_};
            kernel = g.kernel_events_;
            host   = g.host_events_;
        }
        detail::end_group(*state_, kernel, host);

        std::vector<sycl::event> events;
        events.reserve(kernel.size() + host.size());
        for (auto& [e, b] : kernel) events.push_back(e);
        for (auto& e : host) events.push_back(e);
        return events;
    }

private:
    friend detail::QueueState& detail::state_of(const queue& q);

    template <class CGF>
    sycl::event enqueue(const sycl::device* dev, const std::vector<sycl::event>* extra, CGF&& cgf) {
        return body(dev, extra, [&](sycl::handler& h) { cgf(h); });
    }

    template <class Body>
    sycl::event body(const sycl::device* dev, const std::vector<sycl::event>* extra, Body&& fill) {
        detail::submission s = detail::begin_item(*state_, dev, /*concurrent=*/false);

        // The interlock's cross-backend edges, and the stream's own: the backend
        // being submitted to will not honour them (see the header).
        for (auto& e : s.host_waits) e.wait();

        if (extra) s.deps.insert(s.deps.end(), extra->begin(), extra->end());

        sycl::event e = s.target->submit(
            dev ? sycl::property_list{sycl::property::command_group::AdaptiveCpp_retarget{*dev}}
                : sycl::property_list{},
            [&](sycl::handler& h) {
                h.depends_on(s.deps);
                fill(h);
            });

        detail::end_item(*state_, e, s.backend);
        return e;
    }

    std::shared_ptr<detail::QueueState> state_;
};

}  // namespace naga::slabkit
