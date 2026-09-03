// =============================================================================
// test_queue.cpp — one in-order stream, groups, and the host pool
// =============================================================================
//
// The three properties worth proving here are the ones a caller would otherwise
// have to take on trust:
//
//   1. placing an item on a device does not cost the in-order contract,
//   2. a group's members do not depend on each other, and the next item depends
//      on all of them,
//   3. host work runs off the caller's thread, after work it chose to wait on,
//      and may itself submit to the stream.

#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <sycl/sycl.hpp>

#include "naga/slabkit/context.hpp"
#include "naga/slabkit/queue.hpp"
#include "support.hpp"

using namespace naga::slabkit;

namespace {

/// A queue over the default device, and that device's storage.
struct Stream {
    context      ctx;
    queue        q;
    sycl::device dev;

    Stream() : ctx{sycl::device{sycl::default_selector_v}}, q{ctx}, dev{ctx.devices().front()} {}

    int* alloc(std::size_t n) {
        void* p = sycl::malloc_device(n * sizeof(int), dev, ctx.context_of(dev));
        REQUIRE(p != nullptr);
        return static_cast<int*>(p);
    }
    void release(int* p) { sycl::free(p, ctx.context_of(dev)); }

    std::vector<int> read(const int* p, std::size_t n) {
        std::vector<int> host(n, -1);
        q.submit([&](sycl::handler& h) { h.memcpy(host.data(), p, n * sizeof(int)); });
        q.wait();
        return host;
    }
};

}  // namespace

TEST_CASE("a queue reports its context and its default device", "[queue]") {
    Stream s;
    REQUIRE(s.q.get_device() == s.ctx.devices().front());
    REQUIRE(s.q.get_context().devices().size() == s.ctx.devices().size());

    // Copies are the same stream, not a second one.
    queue copy = s.q;
    REQUIRE(copy.get_device() == s.q.get_device());
}

TEST_CASE("a device outside the context cannot be placed on", "[queue]") {
    Stream       s;
    sycl::device other;
    if (!slabkit_test::second_device(s.dev, other)) return;

    REQUIRE_THROWS_AS(s.q.single_task(other, [] {}), std::invalid_argument);
}

TEST_CASE("in-order semantics survive a retarget", "[queue]") {
    Stream            s;
    const std::size_t steps = 16;

    int* acc = s.alloc(1);
    int* out = s.alloc(steps);
    s.q.submit([&](sycl::handler& h) { h.memset(acc, 0, sizeof(int)); });

    // Every one of these carries AdaptiveCpp_retarget, so each is submitted
    // through the placement path rather than the plain one. If the property
    // broke the chain, `out` would not come back as 0..15.
    for (std::size_t k = 0; k < steps; ++k)
        s.q.single_task(s.dev, [=] {
            out[k] = acc[0];
            acc[0] = acc[0] + 1;
        });

    const auto values = s.read(out, steps);
    for (std::size_t k = 0; k < steps; ++k) REQUIRE(values[k] == static_cast<int>(k));

    s.release(acc);
    s.release(out);
}

TEST_CASE("an unplaced item and a placed item share one order", "[queue]") {
    Stream s;

    int* acc = s.alloc(1);
    int* out = s.alloc(4);
    s.q.submit([&](sycl::handler& h) { h.memset(acc, 0, sizeof(int)); });

    s.q.single_task([=] { out[0] = acc[0]; acc[0] = acc[0] + 1; });          // unplaced
    s.q.single_task(s.dev, [=] { out[1] = acc[0]; acc[0] = acc[0] + 1; });   // placed
    s.q.parallel_for(sycl::range<1>{1}, [=](sycl::id<1>) { out[2] = acc[0]; acc[0] = acc[0] + 1; });
    s.q.submit(s.dev, [=](sycl::handler& h) {
        h.single_task([=] { out[3] = acc[0]; acc[0] = acc[0] + 1; });
    });

    const auto values = s.read(out, 4);
    REQUIRE(values == std::vector<int>{0, 1, 2, 3});

    s.release(acc);
    s.release(out);
}

TEST_CASE("explicit dependencies compose with the stream's own", "[queue]") {
    Stream s;

    int* p = s.alloc(2);
    s.q.submit([&](sycl::handler& h) { h.memset(p, 0, 2 * sizeof(int)); });

    sycl::event first = s.q.single_task([=] { p[0] = 7; });
    s.q.single_task(std::vector<sycl::event>{first}, [=] { p[1] = p[0] + 1; });

    const auto values = s.read(p, 2);
    REQUIRE(values == std::vector<int>{7, 8});

    s.release(p);
}

TEST_CASE("a group's members do not depend on each other, and the next item waits for all",
          "[queue]") {
    Stream            s;
    const std::size_t members = 4;

    int* out = s.alloc(members + 1);
    s.q.submit([&](sycl::handler& h) { h.memset(out, 0, (members + 1) * sizeof(int)); });

    const auto events = s.q.submit_group([&](queue::group& g) {
        for (std::size_t i = 0; i < members; ++i)
            g.submit([&, i](sycl::handler& h) {
                h.single_task([=] { out[i] = static_cast<int>(i) + 1; });
            });
    });
    REQUIRE(events.size() == members);

    // The item after the group sees every member's write. This is the group's
    // load-bearing property and it is deterministic.
    s.q.single_task([=] {
        int sum = 0;
        for (std::size_t i = 0; i < members; ++i) sum += out[i];
        out[members] = sum;
    });

    const auto values = s.read(out, members + 1);
    REQUIRE(values[members] == 1 + 2 + 3 + 4);

    s.release(out);
}

TEST_CASE("an empty group leaves the stream's order alone", "[queue]") {
    Stream s;

    int* p = s.alloc(2);
    s.q.submit([&](sycl::handler& h) { h.memset(p, 0, 2 * sizeof(int)); });
    s.q.single_task([=] { p[0] = 5; });

    const auto events = s.q.submit_group([](queue::group&) {});
    REQUIRE(events.empty());

    s.q.single_task([=] { p[1] = p[0] + 1; });

    const auto values = s.read(p, 2);
    REQUIRE(values == std::vector<int>{5, 6});

    s.release(p);
}

TEST_CASE("group members are given the chance to run concurrently", "[queue]") {
    // A bounded spin, so a backend that serializes its out-of-order queue is
    // reported rather than hanging the suite. Each member sets its own flag and
    // then waits, briefly, to see the other's.
    Stream s;

    int* flags = s.alloc(2);
    int* seen  = s.alloc(2);
    s.q.submit([&](sycl::handler& h) { h.memset(flags, 0, 2 * sizeof(int)); });
    s.q.submit([&](sycl::handler& h) { h.memset(seen, 0, 2 * sizeof(int)); });
    s.q.wait();

    s.q.submit_group([&](queue::group& g) {
        for (int me = 0; me < 2; ++me)
            g.submit([&, me](sycl::handler& h) {
                h.single_task([=] {
                    flags[me] = 1;
                    for (int spin = 0; spin < 1000000; ++spin) {
                        if (flags[1 - me] != 0) {
                            seen[me] = 1;
                            break;
                        }
                    }
                });
            });
    });
    s.q.wait();

    const auto observed = s.read(seen, 2);
    if (observed[0] == 0 && observed[1] == 0)
        WARN("neither group member observed the other: this backend ran the out-of-order "
             "queue's members one after another");

    s.release(flags);
    s.release(seen);
}

TEST_CASE("the stream's order holds ACROSS backends", "[queue]") {
    // The regression test for the finding that shaped this class: on acpp
    // 2026.09.03 a depends_on that CROSSES a backend is unreliable -- in this
    // exact shape (OMP producer, OMP memcpy, CUDA memcpy) it was wrong 20 of 20
    // times in raw SYCL, silently. The stream buys the edge back with a
    // host-side wait at the transition; this is what proves it stays bought.
    context ctx{slabkit_test::all_devices()};
    queue   q{ctx};
    if (ctx.backends().size() < 2) return;

    const sycl::device first = ctx.devices().front();
    sycl::device       other = first;
    for (const auto& d : ctx.devices())
        if (!(d.get_platform() == first.get_platform())) other = d;

    void* src = sycl::malloc_device(64, first, ctx.context_of(first));
    void* dst = sycl::malloc_device(64, other, ctx.context_of(other));
    REQUIRE(src != nullptr);
    REQUIRE(dst != nullptr);

    for (int round = 0; round < 8; ++round) {
        const int mark = 0xA0 + round;
        std::vector<std::byte> stage(64, std::byte{0});

        // A slow producer on the first backend, so the item after it is
        // genuinely still pending when the next backend is submitted to.
        auto* p = static_cast<int*>(src);
        q.parallel_for(first, sycl::range<1>{16}, [=](sycl::id<1> i) {
            int acc = mark;
            for (int spin = 0; spin < 200000; ++spin) acc = acc + 0;
            p[i[0]] = acc;
        });

        q.submit(first, [&](sycl::handler& h) { h.memcpy(stage.data(), src, 64); });
        q.submit(other, [&](sycl::handler& h) { h.memcpy(dst, stage.data(), 64); });
        q.wait();

        std::vector<int> back(16, 0);
        q.submit(other, [&](sycl::handler& h) { h.memcpy(back.data(), dst, 64); });
        q.wait();

        for (int v : back) REQUIRE(v == mark);
    }

    sycl::free(src, ctx.context_of(first));
    sycl::free(dst, ctx.context_of(other));
}

TEST_CASE("host work runs off the caller's thread", "[queue]") {
    Stream s;

    const std::thread::id caller = std::this_thread::get_id();
    auto                  fut    = s.q.submit_host([caller] { return std::this_thread::get_id(); });
    REQUIRE(fut.get() != caller);
}

TEST_CASE("a host task waits on the events it captured", "[queue]") {
    Stream s;

    int* p = s.alloc(1);
    s.q.submit([&](sycl::handler& h) { h.memset(p, 0, sizeof(int)); });
    s.q.wait();

    sycl::event slow = s.q.single_task([=] {
        // Long enough that a task which did not wait would read the old value.
        for (int spin = 0; spin < 20000000; ++spin) p[0] = p[0] + 1;
    });

    queue q = s.q;
    auto  fut = s.q.submit_host([slow, p, q]() mutable {
        slow.wait();
        int host = 0;
        q.submit([&](sycl::handler& h) { h.memcpy(&host, p, sizeof(int)); });
        q.wait();
        return host;
    });

    REQUIRE(fut.get() == 20000000);
    s.release(p);
}

TEST_CASE("a host task may submit to the core queues and wait on what it submitted", "[queue]") {
    Stream s;

    int* p = s.alloc(1);
    s.q.submit([&](sycl::handler& h) { h.memset(p, 0, sizeof(int)); });
    s.q.wait();

    queue q   = s.q;
    auto  fut = s.q.submit_host([p, q]() mutable {
        sycl::event e = q.single_task([=] { p[0] = 99; });
        e.wait();

        int host = 0;
        q.submit([&](sycl::handler& h) { h.memcpy(&host, p, sizeof(int)); });
        q.wait();
        return host;
    });

    REQUIRE(fut.get() == 99);
    s.release(p);
}

TEST_CASE("a host task's exception lives in its future", "[queue]") {
    Stream s;
    auto   fut = s.q.submit_host([]() -> int { throw std::runtime_error{"boom"}; });
    REQUIRE_THROWS_AS(fut.get(), std::runtime_error);
}

TEST_CASE("wait does not drain the host pool", "[queue]") {
    Stream s;

    auto fut = s.q.submit_host([] {
        std::this_thread::sleep_for(std::chrono::milliseconds{150});
        return 1;
    });

    const auto before = std::chrono::steady_clock::now();
    s.q.wait();
    const auto elapsed = std::chrono::steady_clock::now() - before;

    REQUIRE(elapsed < std::chrono::milliseconds{100});
    REQUIRE(fut.get() == 1);
}

TEST_CASE("get_wait_list reports the stream's outstanding work", "[queue]") {
    Stream s;

    int* p = s.alloc(1);
    s.q.single_task([=] { p[0] = 1; });

    // The union over the internal queues; on an in-order queue AdaptiveCpp
    // returns at most one event per queue.
    const auto list = s.q.get_wait_list();
    REQUIRE(list.size() <= 2);

    s.q.wait();
    s.release(p);
}
