#pragma once

// =============================================================================
// naga/slabkit/queue.hpp — one in-order stream of work over a context
// =============================================================================
//
// sycl::queue's surface, device-agnostic and in order across backends. Three
// kinds of submission:
//
//   KERNEL TASKS       submit, single_task, parallel_for
//   HOST/MEMORY TASKS  async_host, memcpy, copy, memset, fill
//   GROUP TASKS        submit_group
//
// Kernel tasks run in the order they were submitted. Host/memory tasks are
// asynchronous with respect to each other and ordered against kernel tasks and
// group tasks in both directions. A group's subtasks -- any mix of the two
// kinds -- are unordered among themselves and are ONE task to everything
// outside the group. The _in_order variant of a host/memory task joins the
// kernel order instead.
//
// INTERNALLY three retargetable sycl::queues, none bound to a device by
// construction: an in-order PRIMARY, an out-of-order GROUP queue and an
// out-of-order HOST/MEMORY queue. Every submission names its device and that
// device's context through AdaptiveCpp_retarget, which is what lets one stream
// span backends.
//
// ORDERING IS BY MERGING WAIT LISTS at submission time, never by waiting. The
// primary merges the group and host/memory queues; a group merges all three,
// itself included, when it OPENS, and every subtask carries that set; a
// host/memory task merges the primary and the group queue and never the
// host/memory queue itself.
//
// ORDERING BETWEEN TWO HOST/MEMORY TASKS IS THE CALLER'S: pass the first one's
// event to the second, or use the _in_order variants. Raw pointers are the
// caller's too -- a task orders itself against the stream, not against the
// bytes it names.
//
// A MEMORY TASK TAKES THE ROUTE ITS OPERANDS ALLOW: a direct backend copy when
// they meet on one device, a host memcpy when both are host-reachable, and a
// staged pair of copies through a host intermediate when they are in different
// contexts. The route decides HOW; a named device decides WHERE the operation
// is issued, and placing it validly is the caller's. Without one, the device
// operand decides.
//
// wait() and wait_and_throw() are the global syncs, and one of them is the
// LAST thing a program does with its queue before shutdown, as with any
// sycl::queue: work still running at program exit is illegal and dies
// ungracefully. Destruction does not drain -- the queues carry their
// submitted work out, as sycl queues do.
//
// THREAD SAFETY. Every member is safe to call from any thread.

#include <cstddef>
#include <exception>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#ifndef ACPP_EXT_ASYNC_HOST
#error "slabkit needs an AdaptiveCpp with the async_host extension (ACPP_EXT_ASYNC_HOST)"
#endif

#include "naga/slabkit/context.hpp"

namespace naga::slabkit {

class queue;

namespace detail {

class QueueState;
class GroupState;
class HostErrors;

/// Which of the stream's three queues a task goes to.
enum class lane : unsigned char { primary, host_mem };

/// Everything a submission needs, resolved before anything is submitted: the
/// queue to submit to, the merged dependencies, and the placement.
struct submission {
    sycl::queue*             target = nullptr;
    std::vector<sycl::event> deps;
    sycl::property_list      props{};
};

std::shared_ptr<QueueState> make_queue_state(context ctx, const sycl::device* dev,
                                             sycl::async_handler handler);

submission begin_item(QueueState& st, lane l, const sycl::device* dev,
                      const std::vector<sycl::event>* extra);
submission begin_group_item(GroupState& gs, const sycl::device* dev,
                            const std::vector<sycl::event>* extra);
void       record_group_event(GroupState& gs, const sycl::event& e);

std::shared_ptr<GroupState> open_group(QueueState& st);
std::vector<sycl::event>    group_events(const GroupState& gs);

// ---- memory work, compiled --------------------------------------------------

sycl::event copy_bytes(QueueState& st, lane l, const sycl::device* dev, void* dst,
                       const void* src, std::size_t bytes,
                       const std::vector<sycl::event>* extra);
sycl::event copy_bytes(GroupState& gs, const sycl::device* dev, void* dst, const void* src,
                       std::size_t bytes, const std::vector<sycl::event>* extra);

sycl::event set_bytes(QueueState& st, lane l, const sycl::device* dev, void* p, int value,
                      std::size_t bytes, const std::vector<sycl::event>* extra);
sycl::event set_bytes(GroupState& gs, const sycl::device* dev, void* p, int value,
                      std::size_t bytes, const std::vector<sycl::event>* extra);

sycl::event fill_bytes(QueueState& st, lane l, const sycl::device* dev, void* p,
                       const void* pattern, std::size_t pattern_size, std::size_t count,
                       const std::vector<sycl::event>* extra);
sycl::event fill_bytes(GroupState& gs, const sycl::device* dev, void* p, const void* pattern,
                       std::size_t pattern_size, std::size_t count,
                       const std::vector<sycl::event>* extra);

// ---- host callables ---------------------------------------------------------

/// The stream's asynchronous-error list, which outlives the stream: a host
/// callable holds one and reports into it from whatever thread ran it.
std::shared_ptr<HostErrors> errors_of(QueueState& st);
void                        record_error(HostErrors& errors, std::exception_ptr e);

/// A user callable made copyable and non-throwing, which is what the runtime's
/// host operations require of one. An escaping exception is asynchronous: it
/// surfaces from wait_and_throw or throw_asynchronous.
template <class F>
class guarded_host_fn {
public:
    guarded_host_fn(std::shared_ptr<HostErrors> errors, F&& f)
        : errors_{std::move(errors)}, fn_{std::make_shared<F>(std::move(f))} {}

    void operator()() const {
        try {
            (*fn_)();
        } catch (...) {
            record_error(*errors_, std::current_exception());
        }
    }

private:
    std::shared_ptr<HostErrors> errors_;
    std::shared_ptr<F>          fn_;
};

template <class F>
auto guard_host(QueueState& st, F&& f) {
    return guarded_host_fn<std::decay_t<F>>{errors_of(st), std::decay_t<F>{std::forward<F>(f)}};
}

/// The same, for a stream that keeps its own error list (unordered_queue).
template <class F>
auto guard_host(std::shared_ptr<HostErrors> errors, F&& f) {
    return guarded_host_fn<std::decay_t<F>>{std::move(errors), std::decay_t<F>{std::forward<F>(f)}};
}

const context&      state_context(const QueueState& st);
const sycl::device& state_device(const QueueState& st);
QueueState&         state_of_group(GroupState& gs);

void                     state_wait(QueueState& st);
void                     state_wait_and_throw(QueueState& st);
void                     state_throw_asynchronous(QueueState& st);
std::vector<sycl::event> state_wait_list(QueueState& st);

/// The stream behind a handle. Internal: the compiled core takes the state,
/// which the public surface deliberately does not expose.
QueueState& state_of(const queue& q);

/// True when `dev` belongs to the OpenMP backend.
bool is_host_backend(const sycl::device& dev);

/// The device of an AdaptiveCpp_retarget property, if the list carries one.
bool retarget_device(const sycl::property_list& props, sycl::device& out);

}  // namespace detail

class queue {
public:
    /// A stream over `ctx` whose default device, which every unplaced task runs on,
    /// is SYCL's default device (`sycl::device{sycl::default_selector_v}`, the one a
    /// default-constructed sycl::queue picks), never the context's first device.
    /// \throws std::invalid_argument if that device is not in `ctx`; name another
    /// default device with the form below.
    explicit queue(context ctx, sycl::async_handler handler = {})
        : state_{detail::make_queue_state(std::move(ctx), nullptr, std::move(handler))} {}

    /// The same stream with `dev` as its default device, which every unplaced
    /// task runs on. \throws std::invalid_argument if `dev` is not in `ctx`.
    queue(context ctx, const sycl::device& dev, sycl::async_handler handler = {})
        : state_{detail::make_queue_state(std::move(ctx), &dev, std::move(handler))} {}

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
        return body(detail::lane::primary, nullptr, nullptr,
                    [&](sycl::handler& h) { h.single_task(k); });
    }
    template <class K>
    sycl::event single_task(const sycl::device& dev, const K& k) {
        return body(detail::lane::primary, &dev, nullptr,
                    [&](sycl::handler& h) { h.single_task(k); });
    }
    template <class K>
    sycl::event single_task(const std::vector<sycl::event>& deps, const K& k) {
        return body(detail::lane::primary, nullptr, &deps,
                    [&](sycl::handler& h) { h.single_task(k); });
    }
    template <class K>
    sycl::event single_task(const sycl::device& dev, const std::vector<sycl::event>& deps,
                            const K& k) {
        return body(detail::lane::primary, &dev, &deps,
                    [&](sycl::handler& h) { h.single_task(k); });
    }

    // -------------------------------------------------------------------------
    // parallel_for — range, range+offset and nd_range, each with the four forms.
    // The trailing pack is sycl's ReductionsAndKernel: reductions then kernel.
    // -------------------------------------------------------------------------

    // ---- range --------------------------------------------------------------
    template <class... RK, int Dim>
    sycl::event parallel_for(sycl::range<Dim> r, const RK&... rk) {
        return body(detail::lane::primary, nullptr, nullptr,
                    [&](sycl::handler& h) { h.parallel_for(r, rk...); });
    }
    template <class... RK, int Dim>
    sycl::event parallel_for(const sycl::device& dev, sycl::range<Dim> r, const RK&... rk) {
        return body(detail::lane::primary, &dev, nullptr,
                    [&](sycl::handler& h) { h.parallel_for(r, rk...); });
    }
    template <class... RK, int Dim>
    sycl::event parallel_for(const std::vector<sycl::event>& deps, sycl::range<Dim> r,
                             const RK&... rk) {
        return body(detail::lane::primary, nullptr, &deps,
                    [&](sycl::handler& h) { h.parallel_for(r, rk...); });
    }
    template <class... RK, int Dim>
    sycl::event parallel_for(const sycl::device& dev, const std::vector<sycl::event>& deps,
                             sycl::range<Dim> r, const RK&... rk) {
        return body(detail::lane::primary, &dev, &deps,
                    [&](sycl::handler& h) { h.parallel_for(r, rk...); });
    }

    // ---- range + offset ------------------------------------------------------
    template <class... RK, int Dim>
    sycl::event parallel_for(sycl::range<Dim> r, sycl::id<Dim> off, const RK&... rk) {
        return body(detail::lane::primary, nullptr, nullptr,
                    [&](sycl::handler& h) { h.parallel_for(r, off, rk...); });
    }
    template <class... RK, int Dim>
    sycl::event parallel_for(const sycl::device& dev, sycl::range<Dim> r, sycl::id<Dim> off,
                             const RK&... rk) {
        return body(detail::lane::primary, &dev, nullptr,
                    [&](sycl::handler& h) { h.parallel_for(r, off, rk...); });
    }
    template <class... RK, int Dim>
    sycl::event parallel_for(const std::vector<sycl::event>& deps, sycl::range<Dim> r,
                             sycl::id<Dim> off, const RK&... rk) {
        return body(detail::lane::primary, nullptr, &deps,
                    [&](sycl::handler& h) { h.parallel_for(r, off, rk...); });
    }
    template <class... RK, int Dim>
    sycl::event parallel_for(const sycl::device& dev, const std::vector<sycl::event>& deps,
                             sycl::range<Dim> r, sycl::id<Dim> off, const RK&... rk) {
        return body(detail::lane::primary, &dev, &deps,
                    [&](sycl::handler& h) { h.parallel_for(r, off, rk...); });
    }

    // ---- nd_range ------------------------------------------------------------
    template <class... RK, int Dim>
    sycl::event parallel_for(sycl::nd_range<Dim> nd, const RK&... rk) {
        return body(detail::lane::primary, nullptr, nullptr,
                    [&](sycl::handler& h) { h.parallel_for(nd, rk...); });
    }
    template <class... RK, int Dim>
    sycl::event parallel_for(const sycl::device& dev, sycl::nd_range<Dim> nd, const RK&... rk) {
        return body(detail::lane::primary, &dev, nullptr,
                    [&](sycl::handler& h) { h.parallel_for(nd, rk...); });
    }
    template <class... RK, int Dim>
    sycl::event parallel_for(const std::vector<sycl::event>& deps, sycl::nd_range<Dim> nd,
                             const RK&... rk) {
        return body(detail::lane::primary, nullptr, &deps,
                    [&](sycl::handler& h) { h.parallel_for(nd, rk...); });
    }
    template <class... RK, int Dim>
    sycl::event parallel_for(const sycl::device& dev, const std::vector<sycl::event>& deps,
                             sycl::nd_range<Dim> nd, const RK&... rk) {
        return body(detail::lane::primary, &dev, &deps,
                    [&](sycl::handler& h) { h.parallel_for(nd, rk...); });
    }

    // -------------------------------------------------------------------------
    // Host work — a callable on the runtime's own threads
    // -------------------------------------------------------------------------

    template <class F>
    sycl::event async_host(F&& f) {
        return host_call(detail::lane::host_mem, nullptr, nullptr, std::forward<F>(f));
    }
    template <class F>
    sycl::event async_host(const sycl::device& dev, F&& f) {
        return host_call(detail::lane::host_mem, &dev, nullptr, std::forward<F>(f));
    }
    template <class F>
    sycl::event async_host(const std::vector<sycl::event>& deps, F&& f) {
        return host_call(detail::lane::host_mem, nullptr, &deps, std::forward<F>(f));
    }
    template <class F>
    sycl::event async_host(const sycl::device& dev, const std::vector<sycl::event>& deps, F&& f) {
        return host_call(detail::lane::host_mem, &dev, &deps, std::forward<F>(f));
    }

    template <class F>
    sycl::event async_host_in_order(F&& f) {
        return host_call(detail::lane::primary, nullptr, nullptr, std::forward<F>(f));
    }
    template <class F>
    sycl::event async_host_in_order(const sycl::device& dev, F&& f) {
        return host_call(detail::lane::primary, &dev, nullptr, std::forward<F>(f));
    }
    template <class F>
    sycl::event async_host_in_order(const std::vector<sycl::event>& deps, F&& f) {
        return host_call(detail::lane::primary, nullptr, &deps, std::forward<F>(f));
    }
    template <class F>
    sycl::event async_host_in_order(const sycl::device& dev, const std::vector<sycl::event>& deps,
                                    F&& f) {
        return host_call(detail::lane::primary, &dev, &deps, std::forward<F>(f));
    }

    // -------------------------------------------------------------------------
    // Memory work — SYCL's spellings, over any pair of operands the context can
    // locate. The _in_order variants join the kernel order.
    // -------------------------------------------------------------------------

    sycl::event memcpy(void* dst, const void* src, std::size_t bytes) {
        return detail::copy_bytes(*state_, detail::lane::host_mem, nullptr, dst, src, bytes,
                                  nullptr);
    }
    sycl::event memcpy(const sycl::device& dev, void* dst, const void* src, std::size_t bytes) {
        return detail::copy_bytes(*state_, detail::lane::host_mem, &dev, dst, src, bytes,
                                  nullptr);
    }
    sycl::event memcpy(const std::vector<sycl::event>& deps, void* dst, const void* src,
                       std::size_t bytes) {
        return detail::copy_bytes(*state_, detail::lane::host_mem, nullptr, dst, src, bytes,
                                  &deps);
    }
    sycl::event memcpy(const sycl::device& dev, const std::vector<sycl::event>& deps, void* dst,
                       const void* src, std::size_t bytes) {
        return detail::copy_bytes(*state_, detail::lane::host_mem, &dev, dst, src, bytes, &deps);
    }

    sycl::event memcpy_in_order(void* dst, const void* src, std::size_t bytes) {
        return detail::copy_bytes(*state_, detail::lane::primary, nullptr, dst, src, bytes,
                                  nullptr);
    }
    sycl::event memcpy_in_order(const sycl::device& dev, void* dst, const void* src,
                                std::size_t bytes) {
        return detail::copy_bytes(*state_, detail::lane::primary, &dev, dst, src, bytes, nullptr);
    }
    sycl::event memcpy_in_order(const std::vector<sycl::event>& deps, void* dst, const void* src,
                                std::size_t bytes) {
        return detail::copy_bytes(*state_, detail::lane::primary, nullptr, dst, src, bytes,
                                  &deps);
    }
    sycl::event memcpy_in_order(const sycl::device& dev, const std::vector<sycl::event>& deps,
                                void* dst, const void* src, std::size_t bytes) {
        return detail::copy_bytes(*state_, detail::lane::primary, &dev, dst, src, bytes, &deps);
    }

    template <class T>
    sycl::event copy(const T* src, T* dst, std::size_t count) {
        return memcpy(dst, src, count * sizeof(T));
    }
    template <class T>
    sycl::event copy(const sycl::device& dev, const T* src, T* dst, std::size_t count) {
        return memcpy(dev, dst, src, count * sizeof(T));
    }
    template <class T>
    sycl::event copy(const std::vector<sycl::event>& deps, const T* src, T* dst,
                     std::size_t count) {
        return memcpy(deps, dst, src, count * sizeof(T));
    }
    template <class T>
    sycl::event copy(const sycl::device& dev, const std::vector<sycl::event>& deps, const T* src,
                     T* dst, std::size_t count) {
        return memcpy(dev, deps, dst, src, count * sizeof(T));
    }

    template <class T>
    sycl::event copy_in_order(const T* src, T* dst, std::size_t count) {
        return memcpy_in_order(dst, src, count * sizeof(T));
    }
    template <class T>
    sycl::event copy_in_order(const sycl::device& dev, const T* src, T* dst, std::size_t count) {
        return memcpy_in_order(dev, dst, src, count * sizeof(T));
    }
    template <class T>
    sycl::event copy_in_order(const std::vector<sycl::event>& deps, const T* src, T* dst,
                              std::size_t count) {
        return memcpy_in_order(deps, dst, src, count * sizeof(T));
    }
    template <class T>
    sycl::event copy_in_order(const sycl::device& dev, const std::vector<sycl::event>& deps,
                              const T* src, T* dst, std::size_t count) {
        return memcpy_in_order(dev, deps, dst, src, count * sizeof(T));
    }

    sycl::event memset(void* p, int value, std::size_t bytes) {
        return detail::set_bytes(*state_, detail::lane::host_mem, nullptr, p, value, bytes,
                                 nullptr);
    }
    sycl::event memset(const sycl::device& dev, void* p, int value, std::size_t bytes) {
        return detail::set_bytes(*state_, detail::lane::host_mem, &dev, p, value, bytes, nullptr);
    }
    sycl::event memset(const std::vector<sycl::event>& deps, void* p, int value,
                       std::size_t bytes) {
        return detail::set_bytes(*state_, detail::lane::host_mem, nullptr, p, value, bytes,
                                 &deps);
    }
    sycl::event memset(const sycl::device& dev, const std::vector<sycl::event>& deps, void* p,
                       int value, std::size_t bytes) {
        return detail::set_bytes(*state_, detail::lane::host_mem, &dev, p, value, bytes, &deps);
    }

    sycl::event memset_in_order(void* p, int value, std::size_t bytes) {
        return detail::set_bytes(*state_, detail::lane::primary, nullptr, p, value, bytes,
                                 nullptr);
    }
    sycl::event memset_in_order(const sycl::device& dev, void* p, int value, std::size_t bytes) {
        return detail::set_bytes(*state_, detail::lane::primary, &dev, p, value, bytes, nullptr);
    }
    sycl::event memset_in_order(const std::vector<sycl::event>& deps, void* p, int value,
                                std::size_t bytes) {
        return detail::set_bytes(*state_, detail::lane::primary, nullptr, p, value, bytes, &deps);
    }
    sycl::event memset_in_order(const sycl::device& dev, const std::vector<sycl::event>& deps,
                                void* p, int value, std::size_t bytes) {
        return detail::set_bytes(*state_, detail::lane::primary, &dev, p, value, bytes, &deps);
    }

    template <class T>
    sycl::event fill(void* p, const T& pattern, std::size_t count) {
        return detail::fill_bytes(*state_, detail::lane::host_mem, nullptr, p, &pattern,
                                  sizeof(T), count, nullptr);
    }
    template <class T>
    sycl::event fill(const sycl::device& dev, void* p, const T& pattern, std::size_t count) {
        return detail::fill_bytes(*state_, detail::lane::host_mem, &dev, p, &pattern, sizeof(T),
                                  count, nullptr);
    }
    template <class T>
    sycl::event fill(const std::vector<sycl::event>& deps, void* p, const T& pattern,
                     std::size_t count) {
        return detail::fill_bytes(*state_, detail::lane::host_mem, nullptr, p, &pattern,
                                  sizeof(T), count, &deps);
    }
    template <class T>
    sycl::event fill(const sycl::device& dev, const std::vector<sycl::event>& deps, void* p,
                     const T& pattern, std::size_t count) {
        return detail::fill_bytes(*state_, detail::lane::host_mem, &dev, p, &pattern, sizeof(T),
                                  count, &deps);
    }

    template <class T>
    sycl::event fill_in_order(void* p, const T& pattern, std::size_t count) {
        return detail::fill_bytes(*state_, detail::lane::primary, nullptr, p, &pattern, sizeof(T),
                                  count, nullptr);
    }
    template <class T>
    sycl::event fill_in_order(const sycl::device& dev, void* p, const T& pattern,
                              std::size_t count) {
        return detail::fill_bytes(*state_, detail::lane::primary, &dev, p, &pattern, sizeof(T),
                                  count, nullptr);
    }
    template <class T>
    sycl::event fill_in_order(const std::vector<sycl::event>& deps, void* p, const T& pattern,
                              std::size_t count) {
        return detail::fill_bytes(*state_, detail::lane::primary, nullptr, p, &pattern, sizeof(T),
                                  count, &deps);
    }
    template <class T>
    sycl::event fill_in_order(const sycl::device& dev, const std::vector<sycl::event>& deps,
                              void* p, const T& pattern, std::size_t count) {
        return detail::fill_bytes(*state_, detail::lane::primary, &dev, p, &pattern, sizeof(T),
                                  count, &deps);
    }

    // -------------------------------------------------------------------------
    // Synchronization
    // -------------------------------------------------------------------------

    void wait() { detail::state_wait(*state_); }
    void wait_and_throw() { detail::state_wait_and_throw(*state_); }
    void throw_asynchronous() { detail::state_throw_asynchronous(*state_); }
    std::vector<sycl::event> get_wait_list() const { return detail::state_wait_list(*state_); }

    // -------------------------------------------------------------------------
    // group — one task made of concurrent subtasks, of either kind
    // -------------------------------------------------------------------------

    class group {
    public:
        template <class CGF>
        sycl::event submit(CGF&& cgf) {
            return body(nullptr, nullptr, [&](sycl::handler& h) { cgf(h); });
        }

        template <class CGF>
        sycl::event submit(const sycl::property_list& props, CGF&& cgf) {
            sycl::device dev;
            if (detail::retarget_device(props, dev))
                return body(&dev, nullptr, [&](sycl::handler& h) { cgf(h); });
            return body(nullptr, nullptr, [&](sycl::handler& h) { cgf(h); });
        }

        template <class CGF>
        sycl::event submit(const sycl::device& dev, CGF&& cgf) {
            return body(&dev, nullptr, [&](sycl::handler& h) { cgf(h); });
        }

        template <class CGF>
        sycl::event submit(const std::vector<sycl::event>& deps, CGF&& cgf) {
            return body(nullptr, &deps, [&](sycl::handler& h) { cgf(h); });
        }

        template <class CGF>
        sycl::event submit(const sycl::device& dev, const std::vector<sycl::event>& deps,
                           CGF&& cgf) {
            return body(&dev, &deps, [&](sycl::handler& h) { cgf(h); });
        }

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
        sycl::event async_host(const sycl::device& dev, const std::vector<sycl::event>& deps,
                               F&& f) {
            return host_call(&dev, &deps, std::forward<F>(f));
        }

        sycl::event memcpy(void* dst, const void* src, std::size_t bytes) {
            return detail::copy_bytes(*state_, nullptr, dst, src, bytes, nullptr);
        }
        sycl::event memcpy(const sycl::device& dev, void* dst, const void* src,
                           std::size_t bytes) {
            return detail::copy_bytes(*state_, &dev, dst, src, bytes, nullptr);
        }
        sycl::event memcpy(const std::vector<sycl::event>& deps, void* dst, const void* src,
                           std::size_t bytes) {
            return detail::copy_bytes(*state_, nullptr, dst, src, bytes, &deps);
        }
        sycl::event memcpy(const sycl::device& dev, const std::vector<sycl::event>& deps,
                           void* dst, const void* src, std::size_t bytes) {
            return detail::copy_bytes(*state_, &dev, dst, src, bytes, &deps);
        }

        template <class T>
        sycl::event copy(const T* src, T* dst, std::size_t count) {
            return memcpy(dst, src, count * sizeof(T));
        }
        template <class T>
        sycl::event copy(const sycl::device& dev, const T* src, T* dst, std::size_t count) {
            return memcpy(dev, dst, src, count * sizeof(T));
        }
        template <class T>
        sycl::event copy(const std::vector<sycl::event>& deps, const T* src, T* dst,
                         std::size_t count) {
            return memcpy(deps, dst, src, count * sizeof(T));
        }
        template <class T>
        sycl::event copy(const sycl::device& dev, const std::vector<sycl::event>& deps,
                         const T* src, T* dst, std::size_t count) {
            return memcpy(dev, deps, dst, src, count * sizeof(T));
        }

        sycl::event memset(void* p, int value, std::size_t bytes) {
            return detail::set_bytes(*state_, nullptr, p, value, bytes, nullptr);
        }
        sycl::event memset(const sycl::device& dev, void* p, int value, std::size_t bytes) {
            return detail::set_bytes(*state_, &dev, p, value, bytes, nullptr);
        }
        sycl::event memset(const std::vector<sycl::event>& deps, void* p, int value,
                           std::size_t bytes) {
            return detail::set_bytes(*state_, nullptr, p, value, bytes, &deps);
        }
        sycl::event memset(const sycl::device& dev, const std::vector<sycl::event>& deps, void* p,
                           int value, std::size_t bytes) {
            return detail::set_bytes(*state_, &dev, p, value, bytes, &deps);
        }

        template <class T>
        sycl::event fill(void* p, const T& pattern, std::size_t count) {
            return detail::fill_bytes(*state_, nullptr, p, &pattern, sizeof(T), count, nullptr);
        }
        template <class T>
        sycl::event fill(const sycl::device& dev, void* p, const T& pattern, std::size_t count) {
            return detail::fill_bytes(*state_, &dev, p, &pattern, sizeof(T), count, nullptr);
        }
        template <class T>
        sycl::event fill(const std::vector<sycl::event>& deps, void* p, const T& pattern,
                         std::size_t count) {
            return detail::fill_bytes(*state_, nullptr, p, &pattern, sizeof(T), count, &deps);
        }
        template <class T>
        sycl::event fill(const sycl::device& dev, const std::vector<sycl::event>& deps, void* p,
                         const T& pattern, std::size_t count) {
            return detail::fill_bytes(*state_, &dev, p, &pattern, sizeof(T), count, &deps);
        }

        std::vector<sycl::event> events() const { return detail::group_events(*state_); }

    private:
        friend class queue;
        explicit group(std::shared_ptr<detail::GroupState> state) : state_{std::move(state)} {}

        template <class Body>
        sycl::event body(const sycl::device* dev, const std::vector<sycl::event>* extra,
                         Body&& fill) {
            detail::submission s = detail::begin_group_item(*state_, dev, extra);
            sycl::event        e = s.target->submit(s.props, [&](sycl::handler& h) {
                h.depends_on(s.deps);
                fill(h);
            });
            detail::record_group_event(*state_, e);
            return e;
        }

        template <class F>
        sycl::event host_call(const sycl::device* dev, const std::vector<sycl::event>* extra,
                              F&& f) {
            auto guarded = detail::guard_host(detail::state_of_group(*state_), std::forward<F>(f));
            return body(dev, extra, [&](sycl::handler& h) { h.async_host(guarded); });
        }

        std::shared_ptr<detail::GroupState> state_;
    };

    /// f(group&). Returns the group's subtask events; the group is one task to
    /// everything submitted after it.
    template <class F>
    std::vector<sycl::event> submit_group(F&& f) {
        group g{detail::open_group(*state_)};
        f(g);
        return g.events();
    }

private:
    friend detail::QueueState& detail::state_of(const queue& q);

    template <class CGF>
    sycl::event kernel(const sycl::device* dev, const std::vector<sycl::event>* extra,
                       CGF&& cgf) {
        return body(detail::lane::primary, dev, extra, [&](sycl::handler& h) { cgf(h); });
    }

    template <class Body>
    sycl::event body(detail::lane l, const sycl::device* dev,
                     const std::vector<sycl::event>* extra, Body&& fill) {
        detail::submission s = detail::begin_item(*state_, l, dev, extra);
        return s.target->submit(s.props, [&](sycl::handler& h) {
            h.depends_on(s.deps);
            fill(h);
        });
    }

    template <class F>
    sycl::event host_call(detail::lane l, const sycl::device* dev,
                          const std::vector<sycl::event>* extra, F&& f) {
        auto guarded = detail::guard_host(*state_, std::forward<F>(f));
        return body(l, dev, extra, [&](sycl::handler& h) { h.async_host(guarded); });
    }

    std::shared_ptr<detail::QueueState> state_;
};

}  // namespace naga::slabkit
