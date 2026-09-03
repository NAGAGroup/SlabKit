#pragma once

// =============================================================================
// naga/slabkit/queue.hpp — one in-order stream of work over a context
// =============================================================================
//
// sycl::queue's surface, with one stream semantics that hold ACROSS backends.
// Internally, per backend of the context and created lazily: one in-order and
// one out-of-order sycl::queue, both AdaptiveCpp_retargetable, on that
// backend's default context; plus one thread pool. A light handle: copies are
// the same stream.
//
// PLACEMENT is the AdaptiveCpp_retarget command-group property, routed to that
// device's backend pair -- the extension's proper use: same backend, same
// context. Every shortcut here is implemented through submit and that property;
// none forwards to sycl::queue's own shortcuts, because those would bind the
// work to the internal queue's own device.
//
// IN ORDER ACROSS BACKENDS. A submission depends on the events of the previous
// step, whichever backends those ran on. A group's submissions all depend on
// the step before the group and on none of each other; the next item depends on
// all of them. Events cross queues and backends in AdaptiveCpp, so cross-backend
// order costs one dependency edge and no synchronization.
//
// WHY A THREAD POOL AND NOT CUSTOM OPERATIONS. Measured on this machine, CUDA
// and OMP, in-order and out-of-order alike: AdaptiveCpp evaluates a custom
// operation's callable INSIDE submit() on the calling thread on backends with
// native streams, before its dependencies have completed -- depends_on orders
// only the backend work the callable enqueues, not the callable itself -- and a
// queue submission from inside it deadlocks. AdaptiveCpp also has no host_task
// (not in 25.10, not on develop; the maintainer's position in issue #1589 is
// that it is "not a good solution for most use cases"). A pool is therefore the
// only place host work can run later, off the caller, after work it chose to
// wait on.
//
// THREAD SAFETY. Every member is safe to call from any thread: one internal
// mutex guards the backend-pair map and the stream's step state, and it is
// never held while waiting on or submitting to a sycl::queue. Two submissions
// racing from two threads are ordered by whichever reaches the mutex first,
// exactly as two threads sharing a sycl::queue are -- "in order" is a statement
// about the sequence of calls, not a promise to serialize callers.

#include <cstddef>
#include <functional>
#include <future>
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

/// Everything a submission needs, resolved under the state mutex and handed
/// back before anything is submitted.
struct submission {
    sycl::queue*             target;
    std::vector<sycl::event> deps;
};

std::shared_ptr<QueueState> make_queue_state(context ctx, sycl::async_handler handler,
                                             std::size_t host_threads);

/// \param dev nullptr for the stream's default device.
/// \param concurrent true selects the backend's out-of-order queue (groups).
submission begin_item(QueueState& st, const sycl::device* dev, bool concurrent);

/// Record `e` as the whole of the current step.
void end_item(QueueState& st, const sycl::event& e);

/// The deps a group's members share: the step before the group.
std::vector<sycl::event> begin_group(QueueState& st);
void                     end_group(QueueState& st, std::vector<sycl::event> events);

void post_host(QueueState& st, std::function<void()> task);

const context&      state_context(const QueueState& st);
const sycl::device& state_device(const QueueState& st);

void                     state_wait(QueueState& st);
void                     state_wait_and_throw(QueueState& st);
void                     state_throw_asynchronous(QueueState& st);
std::vector<sycl::event> state_wait_list(QueueState& st);

/// The device of an AdaptiveCpp_retarget property, if the list carries one.
bool retarget_device(const sycl::property_list& props, sycl::device& out);

}  // namespace detail

class queue {
public:
    explicit queue(context ctx, sycl::async_handler handler = {}, std::size_t host_threads = 2);

    const context&      get_context() const { return detail::state_context(*state_); }
    const sycl::device& get_device() const { return detail::state_device(*state_); }

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
        if (detail::retarget_device(props, dev)) return enqueue(&dev, nullptr, std::forward<CGF>(cgf));
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
    // Synchronization
    // -------------------------------------------------------------------------

    void                     wait() { detail::state_wait(*state_); }
    void                     wait_and_throw() { detail::state_wait_and_throw(*state_); }
    void                     throw_asynchronous() { detail::state_throw_asynchronous(*state_); }
    std::vector<sycl::event> get_wait_list() const { return detail::state_wait_list(*state_); }

    // -------------------------------------------------------------------------
    // group — one logical step of concurrent submissions
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

        std::vector<sycl::event> events() const {
            std::lock_guard<std::mutex> lock{mutex_};
            return events_;
        }

    private:
        friend class queue;
        group(detail::QueueState& st, std::vector<sycl::event> deps)
            : state_{&st}, deps_{std::move(deps)} {}

        template <class CGF>
        sycl::event enqueue(const sycl::device* dev, CGF&& cgf) {
            detail::submission s = detail::begin_item(*state_, dev, /*concurrent=*/true);
            s.deps               = deps_;

            sycl::event e = s.target->submit(
                dev ? sycl::property_list{sycl::property::command_group::AdaptiveCpp_retarget{*dev}}
                    : sycl::property_list{},
                [&](sycl::handler& h) {
                    h.depends_on(s.deps);
                    cgf(h);
                });

            std::lock_guard<std::mutex> lock{mutex_};
            events_.push_back(e);
            return e;
        }

        detail::QueueState*      state_;
        std::vector<sycl::event> deps_;
        mutable std::mutex       mutex_;
        std::vector<sycl::event> events_;
    };

    /// f(group&). Returns the step's events, which the next item will wait on.
    template <class F>
    std::vector<sycl::event> submit_group(F&& f) {
        group g{*state_, detail::begin_group(*state_)};
        f(g);
        std::vector<sycl::event> events = g.events();
        detail::end_group(*state_, events);
        return events;
    }

    // -------------------------------------------------------------------------
    // Host work — on the pool, never on the caller's thread
    // -------------------------------------------------------------------------

    template <class F>
    std::future<std::invoke_result_t<F>> submit_host(F&& f) {
        using R   = std::invoke_result_t<F>;
        auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
        auto fut  = task->get_future();
        detail::post_host(*state_, [task]() { (*task)(); });
        return fut;
    }

private:
    template <class CGF>
    sycl::event enqueue(const sycl::device* dev, const std::vector<sycl::event>* extra, CGF&& cgf) {
        return body(dev, extra, [&](sycl::handler& h) { cgf(h); });
    }

    template <class Body>
    sycl::event body(const sycl::device* dev, const std::vector<sycl::event>* extra, Body&& fill) {
        detail::submission s = detail::begin_item(*state_, dev, /*concurrent=*/false);
        if (extra) s.deps.insert(s.deps.end(), extra->begin(), extra->end());

        sycl::event e = s.target->submit(
            dev ? sycl::property_list{sycl::property::command_group::AdaptiveCpp_retarget{*dev}}
                : sycl::property_list{},
            [&](sycl::handler& h) {
                h.depends_on(s.deps);
                fill(h);
            });

        detail::end_item(*state_, e);
        return e;
    }

    std::shared_ptr<detail::QueueState> state_;
};

}  // namespace naga::slabkit
