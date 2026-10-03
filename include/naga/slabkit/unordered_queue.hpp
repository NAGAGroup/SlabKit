#pragma once

// =============================================================================
// naga/slabkit/unordered_queue.hpp — one out-of-order stream of work over a context
// =============================================================================
//
// sycl::queue's surface for KERNEL tasks and HOST tasks, device-agnostic and
// completely out of order. Every task is ordered ONLY by the events passed to
// it: nothing is merged, including the stream's own earlier work. A task that is
// given no events may start the moment it is submitted, whatever came before it.
//
//   KERNEL TASKS  submit, single_task, parallel_for
//   HOST TASKS    async_host
//
// INTERNALLY one retargetable sycl::queue, not bound to a device by construction.
// A task is placed by its device argument, or by the stream's default device,
// exactly as with slabkit::queue: every submission names its device and that
// device's context through AdaptiveCpp_retarget.
//
// wait() and wait_and_throw() wait for everything submitted to this stream, and
// one of them is the LAST thing a program does with it before shutdown, as with
// any sycl::queue and as with slabkit::queue: work still running at program exit
// is illegal and dies ungracefully. Destruction does not drain.
//
// Memory operations and groups are deliberately absent. Containers and
// allocators use slabkit::queue; a later change merges both into one queue type.
//
// THREAD SAFETY. Every member is safe to call from any thread.

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/slabkit/context.hpp"
#include "naga/slabkit/queue.hpp"

namespace naga::slabkit {

class unordered_queue;

namespace detail {

class UnorderedState;

std::shared_ptr<UnorderedState> make_unordered_state(context ctx, const sycl::device* dev,
                                                     sycl::async_handler handler);

/// The queue to submit to, the placement, and exactly the events the caller
/// passed (none when `extra` is null).
submission begin_unordered_item(UnorderedState& st, const sycl::device* dev,
                                const std::vector<sycl::event>* extra);

std::shared_ptr<HostErrors> errors_of(UnorderedState& st);

const context&      unordered_context(const UnorderedState& st);
const sycl::device& unordered_device(const UnorderedState& st);

void unordered_wait(UnorderedState& st);
void unordered_wait_and_throw(UnorderedState& st);
void unordered_throw_asynchronous(UnorderedState& st);

}  // namespace detail

class unordered_queue {
public:
    explicit unordered_queue(context ctx, sycl::async_handler handler = {})
        : state_{detail::make_unordered_state(std::move(ctx), nullptr, std::move(handler))} {}

    /// The same stream with `dev` as its default device, which every unplaced
    /// task runs on. \throws std::invalid_argument if `dev` is not in `ctx`.
    unordered_queue(context ctx, const sycl::device& dev, sycl::async_handler handler = {})
        : state_{detail::make_unordered_state(std::move(ctx), &dev, std::move(handler))} {}

    const context&      get_context() const { return detail::unordered_context(*state_); }
    const sycl::device& get_device() const { return detail::unordered_device(*state_); }

    /// Two handles compare equal when they are the same stream.
    friend bool operator==(const unordered_queue& a, const unordered_queue& b) {
        return a.state_ == b.state_;
    }
    friend bool operator!=(const unordered_queue& a, const unordered_queue& b) {
        return !(a == b);
    }

    // -------------------------------------------------------------------------
    // submit
    // -------------------------------------------------------------------------

    template <class CGF>
    sycl::event submit(CGF&& cgf) {
        return kernel(nullptr, nullptr, std::forward<CGF>(cgf));
    }

    template <class CGF>
    sycl::event submit(const sycl::property_list& props, CGF&& cgf) {
        sycl::device dev;
        if (detail::retarget_device(props, dev))
            return kernel(&dev, nullptr, std::forward<CGF>(cgf));
        return kernel(nullptr, nullptr, std::forward<CGF>(cgf));
    }

    template <class CGF>
    sycl::event submit(const sycl::device& dev, CGF&& cgf) {
        return kernel(&dev, nullptr, std::forward<CGF>(cgf));
    }

    template <class CGF>
    sycl::event submit(const std::vector<sycl::event>& deps, CGF&& cgf) {
        return kernel(nullptr, &deps, std::forward<CGF>(cgf));
    }

    template <class CGF>
    sycl::event submit(const sycl::device& dev, const std::vector<sycl::event>& deps, CGF&& cgf) {
        return kernel(&dev, &deps, std::forward<CGF>(cgf));
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
    // Host work — a callable on the runtime's own threads. It is copyable and
    // non-throwing as the runtime requires; an escaping exception is
    // asynchronous and surfaces from wait_and_throw or throw_asynchronous.
    // -------------------------------------------------------------------------

    template <class F>
    sycl::event async_host(F&& f) {
        return host_call(nullptr, nullptr, std::forward<F>(f));
    }
    template <class F>
    sycl::event async_host(const sycl::device& dev, F&& f) {
        return host_call(&dev, nullptr, std::forward<F>(f));
    }
    template <class F>
    sycl::event async_host(const std::vector<sycl::event>& deps, F&& f) {
        return host_call(nullptr, &deps, std::forward<F>(f));
    }
    template <class F>
    sycl::event async_host(const sycl::device& dev, const std::vector<sycl::event>& deps, F&& f) {
        return host_call(&dev, &deps, std::forward<F>(f));
    }

    // -------------------------------------------------------------------------
    // Synchronization
    // -------------------------------------------------------------------------

    void wait() { detail::unordered_wait(*state_); }
    void wait_and_throw() { detail::unordered_wait_and_throw(*state_); }
    void throw_asynchronous() { detail::unordered_throw_asynchronous(*state_); }

private:
    template <class CGF>
    sycl::event kernel(const sycl::device* dev, const std::vector<sycl::event>* extra,
                       CGF&& cgf) {
        return body(dev, extra, [&](sycl::handler& h) { cgf(h); });
    }

    /// Every submission: the compiled core resolves the target, the placement
    /// and the caller's events; nothing else is added.
    template <class Body>
    sycl::event body(const sycl::device* dev, const std::vector<sycl::event>* extra, Body&& fill) {
        detail::submission s = detail::begin_unordered_item(*state_, dev, extra);
        return s.target->submit(s.props, [&](sycl::handler& h) {
            h.depends_on(s.deps);
            fill(h);
        });
    }

    template <class F>
    sycl::event host_call(const sycl::device* dev, const std::vector<sycl::event>* extra, F&& f) {
        auto guarded = detail::guard_host(detail::errors_of(*state_), std::forward<F>(f));
        return body(dev, extra, [&](sycl::handler& h) { h.async_host(guarded); });
    }

    std::shared_ptr<detail::UnorderedState> state_;
};

}  // namespace naga::slabkit
