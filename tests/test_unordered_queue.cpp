// =============================================================================
// test_unordered_queue.cpp — one out-of-order stream: ordered only by the events given
// =============================================================================
//
// The properties worth proving are the ones a caller would otherwise have to
// take on trust:
//
//   1. a task given an event does not start before that event completes,
//   2. a task given NO events is not held back by anything else on the stream,
//   3. wait() covers everything submitted, and a host task's exception surfaces
//      from wait_and_throw,
//   4. usm_vector::events() is what a task on this stream passes to come after a
//      container's work and its registered events.
//
// A held-back dependency is a Gate: a kernel that spins on a flag in host USM
// until the test opens it. Every wait in a test that could hang if the property
// failed is bounded (wait_done), and a Gate opens itself when it goes out of
// scope, so a failing REQUIRE cannot leave a spinning kernel behind.

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <sycl/sycl.hpp>

#include "naga/slabkit/slabkit.hpp"
#include "support.hpp"

using namespace naga::slabkit;
using slabkit_test::Fixture;

namespace {

/// The fixture's context, the unordered stream over it, and the host USM the
/// tests' kernels write to. The stream is waited before that memory is freed.
struct UFixture {
    Fixture           f;
    unordered_queue   uq;
    std::vector<int*> hosts;

    UFixture() : uq{f.sctx} {}

    int* host_int(int init = 0) {
        int* p = sycl::malloc_host<int>(1, f.ctx);
        REQUIRE(p != nullptr);
        *p = init;
        hosts.push_back(p);
        return p;
    }

    ~UFixture() {
        uq.wait();
        for (int* p : hosts) sycl::free(p, f.ctx);
    }
};

/// A kernel that spins until the host flips a flag in host USM. On the
/// UNORDERED stream under test (`on_stream`), or on the fixture's raw sycl queue,
/// outside the stream. Declare a Gate AFTER the UFixture: it is destroyed first
/// and opens itself.
class Gate {
public:
    sycl::event event;

    Gate(UFixture& u, bool on_stream) : flag_{u.host_int(0)} {
        int* flag = flag_;
        auto spin = [=] {
            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::system> a{
                *flag};
            while (a.load() == 0) {
            }
        };
        event = on_stream ? u.uq.single_task(spin) : u.f.sq.single_task(spin);
    }

    Gate(const Gate&)            = delete;
    Gate& operator=(const Gate&) = delete;

    void open() { std::atomic_ref<int>{*flag_}.store(1); }

    ~Gate() {
        open();
        event.wait();
    }

private:
    int* flag_;
};

bool is_done(const sycl::event& e) {
    return e.get_info<sycl::info::event::command_execution_status>() ==
           sycl::info::event_command_status::complete;
}

/// Long enough that a task which does not wait for a closed gate would have
/// finished by now.
void settle() { std::this_thread::sleep_for(std::chrono::milliseconds{150}); }

/// Polls instead of blocking, so a property that fails is reported, not hung.
bool wait_done(const sycl::event& e, std::chrono::milliseconds limit = std::chrono::seconds{10}) {
    const auto until = std::chrono::steady_clock::now() + limit;
    while (!is_done(e)) {
        if (std::chrono::steady_clock::now() > until) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    return true;
}

allocator<int> device_alloc(const Fixture& f) { return allocator<int>{f.q, kind::device, f.dev}; }

}  // namespace

TEST_CASE("a kernel with a dependency does not run until the event completes", "[unordered_queue]") {
    UFixture u;
    int*     out = u.host_int(0);
    Gate     g{u, /*on_stream=*/false};

    sycl::event e = u.uq.single_task(std::vector<sycl::event>{g.event}, [=] { *out = 1; });

    settle();
    CHECK(*out == 0);
    CHECK_FALSE(is_done(e));

    g.open();
    REQUIRE(wait_done(e));
    CHECK(*out == 1);
}

TEST_CASE("a kernel without dependencies is not ordered after earlier work on the stream",
          "[unordered_queue]") {
    UFixture u;
    int*     out = u.host_int(0);
    Gate     g{u, /*on_stream=*/true};  // submitted FIRST, spinning on the stream itself

    // Nothing is merged: this task is not held back by the gate before it.
    sycl::event e = u.uq.single_task([=] { *out = 1; });

    const bool finished = wait_done(e);
    const bool gate_closed = !is_done(g.event);
    g.open();
    REQUIRE(finished);
    CHECK(gate_closed);
    CHECK(*out == 1);
}

TEST_CASE("an async_host task runs after its dependency, and its exception surfaces",
          "[unordered_queue]") {
    UFixture u;

    SECTION("ordered after the event it is given") {
        auto ran = std::make_shared<std::atomic<int>>(0);
        Gate g{u, /*on_stream=*/false};

        sycl::event e =
            u.uq.async_host(std::vector<sycl::event>{g.event}, [ran] { ran->store(1); });

        settle();
        CHECK(ran->load() == 0);
        CHECK_FALSE(is_done(e));

        g.open();
        REQUIRE(wait_done(e));
        CHECK(ran->load() == 1);
    }

    SECTION("an exception escaping the callable surfaces from wait_and_throw") {
        u.uq.async_host([] { throw std::runtime_error{"boom"}; });
        REQUIRE_THROWS_AS(u.uq.wait_and_throw(), std::runtime_error);
        REQUIRE_NOTHROW(u.uq.wait_and_throw());  // reported once
    }
}

TEST_CASE("wait returns only after all submitted work has finished", "[unordered_queue]") {
    UFixture u;
    int*     out = u.host_int(0);
    Gate     g{u, /*on_stream=*/true};

    sycl::event after = u.uq.single_task(std::vector<sycl::event>{g.event}, [=] { *out = 7; });

    // wait() blocks the calling thread, so it runs on its own; the gate opens
    // only after we have seen it still waiting.
    std::atomic<bool> returned{false};
    std::thread       t{[&] {
        u.uq.wait();
        returned = true;
    }};
    settle();
    const bool returned_early = returned.load();
    g.open();
    t.join();

    CHECK_FALSE(returned_early);
    CHECK(returned.load());
    CHECK(is_done(g.event));
    CHECK(is_done(after));
    CHECK(*out == 7);
}

TEST_CASE("the property_list form honors an AdaptiveCpp_retarget device", "[unordered_queue]") {
    context ctx{slabkit_test::all_devices()};

    SECTION("the device the property names is the one extracted") {
        const sycl::device dev = ctx.devices().front();
        const sycl::property_list props{
            sycl::property::command_group::AdaptiveCpp_retarget{ctx.context_of(dev), dev}};
        sycl::device got;
        REQUIRE(detail::retarget_device(props, got));
        REQUIRE(got == dev);
    }

    SECTION("a task is placed on each device the context holds") {
        unordered_queue uq{ctx};
        for (const sycl::device& dev : ctx.devices()) {
            const sycl::property_list props{
                sycl::property::command_group::AdaptiveCpp_retarget{ctx.context_of(dev), dev}};
            sycl::event e = uq.submit(props, [&](sycl::handler& h) { h.single_task([] {}); });
            REQUIRE(wait_done(e));
        }
        uq.wait();
    }
}

TEST_CASE("a device outside the context cannot be the default device", "[unordered_queue]") {
    Fixture      f;
    sycl::device other;
    if (!slabkit_test::second_device(f.dev, other)) return;

    // The fixture's context holds only its own device.
    REQUIRE_THROWS_AS((unordered_queue{f.sctx, other}), std::invalid_argument);
}

TEST_CASE("a queue reports its context and its default device, and compares by stream",
          "[unordered_queue]") {
    Fixture         f;
    unordered_queue a{f.sctx};
    unordered_queue b{f.sctx};

    REQUIRE(a.get_device() == f.dev);
    REQUIRE(a.get_context().devices().size() == f.sctx.devices().size());

    unordered_queue copy = a;
    REQUIRE(copy == a);
    REQUIRE(a != b);
}

TEST_CASE("usm_vector::events includes registered events, and a task depending on it waits",
          "[unordered_queue][usm_vector]") {
    UFixture        u;
    int*            out = u.host_int(0);
    usm_vector<int> v{8, 1, device_alloc(u.f)};  // before the gate: the gate is destroyed first
    v.sync();
    Gate g{u, /*on_stream=*/false};

    REQUIRE(v.events().size() == 1);  // the container's own last event
    v.register_event(g.event);
    const std::vector<sycl::event> evs = v.events();
    REQUIRE(evs.size() == 2);  // last event + the registered one

    sycl::event e = u.uq.single_task(evs, [=] { *out = 1; });

    settle();
    CHECK(*out == 0);
    CHECK_FALSE(is_done(e));

    g.open();
    REQUIRE(wait_done(e));
    CHECK(*out == 1);
}
