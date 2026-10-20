// =============================================================================
// unordered_queue.cpp — the compiled core of the out-of-order stream
// =============================================================================
//
// Nothing here is a template. The public unordered_queue's submit and its
// shortcuts are thin templates that call begin_unordered_item, submit to the
// sycl::queue they are handed, and return the event -- so a consumer compiles
// the kernel and almost nothing else of slabkit.
//
// The whole of the ordering is the caller's: the dependencies of a submission
// are exactly the events it was given.

#include "naga/slabkit/unordered_queue.hpp"

#include <exception>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "detail/unordered_state.hpp"
#include "naga/slabkit/context.hpp"

namespace naga::slabkit::detail {

namespace {

const sycl::property_list kUnorderedProps{sycl::property::queue::AdaptiveCpp_retargetable{}};

sycl::queue make_lane(const sycl::async_handler& handler, const sycl::property_list& props) {
    return handler ? sycl::queue{handler, props} : sycl::queue{props};
}

}  // namespace

// -----------------------------------------------------------------------------
// The stream state
// -----------------------------------------------------------------------------

UnorderedState::UnorderedState(context c, const sycl::device* dev, sycl::async_handler handler)
    : ctx{std::move(c)}
    // SYCL's default device (the one a default-constructed sycl::queue picks), never the context's first device.
    , default_device{dev ? *dev : sycl::device{sycl::default_selector_v}}
    , stream{make_lane(handler, kUnorderedProps)}
    , errors{std::make_shared<HostErrors>()} {
    if (!ctx.contains(default_device))
        throw std::invalid_argument{
            dev ? "naga::slabkit::unordered_queue: the default device is not in this queue's context"
                : "naga::slabkit::unordered_queue: SYCL's default device is not in this queue's context; "
                  "name the default device with unordered_queue(ctx, dev)"};
}

sycl::property_list UnorderedState::place(const sycl::device* dev) const {
    const sycl::device& d = dev ? *dev : default_device;
    return sycl::property_list{
        sycl::property::command_group::AdaptiveCpp_retarget{ctx.context_of(d), d}};
}

// -----------------------------------------------------------------------------
// The core the public templates call
// -----------------------------------------------------------------------------

std::shared_ptr<UnorderedState> make_unordered_state(context ctx, const sycl::device* dev,
                                                     sycl::async_handler handler) {
    return std::make_shared<UnorderedState>(std::move(ctx), dev, std::move(handler));
}

submission begin_unordered_item(UnorderedState& st, const sycl::device* dev,
                                const std::vector<sycl::event>* extra) {
    submission s;
    s.target = &st.stream;
    s.props  = st.place(dev);
    if (extra) s.deps = *extra;
    return s;
}

std::shared_ptr<HostErrors> errors_of(UnorderedState& st) { return st.errors; }

// -----------------------------------------------------------------------------
// Waiting, errors, introspection
// -----------------------------------------------------------------------------

const context&      unordered_context(const UnorderedState& st) { return st.ctx; }
const sycl::device& unordered_device(const UnorderedState& st) { return st.default_device; }

void unordered_wait(UnorderedState& st) { st.stream.wait(); }

void unordered_wait_and_throw(UnorderedState& st) {
    unordered_wait(st);
    unordered_throw_asynchronous(st);
}

void unordered_throw_asynchronous(UnorderedState& st) {
    st.stream.throw_asynchronous();

    // A host callable's exception is asynchronous in exactly SYCL's sense, but
    // the runtime has no way to be told about it, so the stream keeps its own.
    if (std::exception_ptr first = st.errors->pop()) std::rethrow_exception(first);
}

}  // namespace naga::slabkit::detail
