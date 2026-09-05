// =============================================================================
// test_queue.cpp — kernel tasks, placement and groups
// =============================================================================
//
// The three properties worth proving here are the ones a caller would otherwise
// have to take on trust:
//
//   1. placing a task on a device does not cost the in-order contract,
//   2. a group's subtasks do not depend on each other, and the task after the
//      group depends on all of them,
//   3. the order holds across backends, which is where it used to break.

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

/// A recurrence the optimizer cannot fold away and the host can reproduce
/// exactly. The instrument matters: a producer spelled `acc = acc + 0` in a loop
/// is deleted at -O2, and both of the spikes that first measured this defect
/// were measuring nothing.
inline unsigned churn(unsigned seed, int rounds) {
    unsigned acc = seed;
    for (int i = 0; i < rounds; ++i) acc = acc * 1103515245u + 12345u;
    return acc;
}

TEST_CASE("the stream's order holds ACROSS backends", "[queue]") {
    // Forty rounds of the shape that used to be wrong forty times in forty: a
    // producer on one backend, a copy out through pageable host memory, and a
    // copy back in on another backend, ordered by nothing but the stream.
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

    for (int round = 0; round < 40; ++round) {
        const unsigned mark     = 0xA0u + static_cast<unsigned>(round);
        const int      spins    = 200000;
        const int      expected = static_cast<int>(churn(mark, spins));
        std::vector<std::byte> stage(64, std::byte{0});

        // A slow producer on the first backend, so the item after it is
        // genuinely still pending when the next backend is submitted to.
        auto* p = static_cast<int*>(src);
        q.parallel_for(first, sycl::range<1>{16}, [=](sycl::id<1> i) {
            unsigned acc = mark;
            for (int spin = 0; spin < spins; ++spin) acc = acc * 1103515245u + 12345u;
            p[i[0]] = static_cast<int>(acc);
        });

        q.submit(first, [&](sycl::handler& h) { h.memcpy(stage.data(), src, 64); });
        q.submit(other, [&](sycl::handler& h) { h.memcpy(dst, stage.data(), 64); });
        q.wait();

        std::vector<int> back(16, 0);
        q.submit(other, [&](sycl::handler& h) { h.memcpy(back.data(), dst, 64); });
        q.wait();

        for (int v : back) REQUIRE(v == expected);
    }

    sycl::free(src, ctx.context_of(first));
    sycl::free(dst, ctx.context_of(other));
}

TEST_CASE("get_wait_list reports the stream's outstanding work", "[queue]") {
    Stream s;

    int* p = s.alloc(1);
    s.q.single_task([=] { p[0] = 1; });

    // The union over the three internal queues. On an in-order queue
    // AdaptiveCpp returns at most one event, and nothing here has posted
    // host or memory work.
    const auto list = s.q.get_wait_list();
    REQUIRE(list.size() <= 3);

    s.q.wait();
    s.release(p);
}
