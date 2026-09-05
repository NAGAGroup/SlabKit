// =============================================================================
// test_host_work.cpp — the three submission types and how they order
// =============================================================================
//
// Every assertion here is ORDER-SENSITIVE and built from operations that do not
// commute: +1, then ×10, then +1 is 11 only in that order. A test whose steps
// commute passes on a broken ordering, which is the failure mode this file
// exists to rule out.
//
//   1. a host task runs off the caller's thread and yields a usable event,
//   2. host/memory tasks are concurrent with each other,
//   3. the _in_order variants are serialized,
//   4. kernel-then-memory and memory-then-kernel are ordered by the merges,
//   5. a group is one task: unordered inside, ordered in both directions
//      outside,
//   6. an in-order host task may be placed on a device.

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <sycl/sycl.hpp>

#include "naga/slabkit/slabkit.hpp"
#include "support.hpp"

using namespace naga::slabkit;
namespace sk = naga::slabkit;

namespace {

struct Stream {
    context      ctx;
    queue        q;
    sycl::device dev;

    Stream() : ctx{sycl::device{sycl::default_selector_v}}, q{ctx}, dev{ctx.devices().front()} {}

    /// Host USM: reachable from a kernel AND from a host task, which is what
    /// every ordering assertion here needs.
    int* alloc_host(std::size_t n) {
        void* p = sycl::malloc_host(n * sizeof(int), ctx.context_of(dev));
        REQUIRE(p != nullptr);
        return static_cast<int*>(p);
    }
    void release(void* p) { sycl::free(p, ctx.context_of(dev)); }
};

/// A recurrence the optimizer cannot fold away and the host can reproduce
/// exactly. A producer spelled `acc = acc + 0` in a loop is deleted at -O2, and
/// a test built on one measures nothing at all.
unsigned churn(unsigned seed, int rounds) {
    unsigned acc = seed;
    for (int i = 0; i < rounds; ++i) acc = acc * 1103515245u + 12345u;
    return acc;
}

constexpr int kSpins = 4000000;

}  // namespace

// -----------------------------------------------------------------------------
// 1. off the caller's thread, with a usable event
// -----------------------------------------------------------------------------

TEST_CASE("a host task runs off the caller's thread and yields a usable event", "[host]") {
    Stream s;

    auto ran    = std::make_shared<std::atomic<bool>>(false);
    auto thread = std::make_shared<std::atomic<std::thread::id>>();

    sycl::event e = s.q.async_host([ran, thread] {
        thread->store(std::this_thread::get_id());
        ran->store(true);
    });

    e.wait();
    REQUIRE(ran->load());
    REQUIRE(thread->load() != std::this_thread::get_id());
}

TEST_CASE("a host task's exception is asynchronous, not thrown at the caller", "[host]") {
    Stream s;

    sycl::event e = s.q.async_host([] { throw std::runtime_error{"boom"}; });
    e.wait();  // the task completes; the throw did not escape the runtime

    REQUIRE_THROWS_AS(s.q.wait_and_throw(), std::runtime_error);
}

// -----------------------------------------------------------------------------
// 2. host/memory tasks are concurrent with each other
// -----------------------------------------------------------------------------

TEST_CASE("host tasks run concurrently with each other and never block the caller", "[host]") {
    Stream s;

    const int  tasks = 4;
    const auto ms    = std::chrono::milliseconds{200};

    auto ran = std::make_shared<std::atomic<int>>(0);

    const auto submitted_at = std::chrono::steady_clock::now();
    for (int i = 0; i < tasks; ++i)
        s.q.async_host([ran, ms] {
            std::this_thread::sleep_for(ms);
            ran->fetch_add(1);
        });
    const auto submit_cost = std::chrono::steady_clock::now() - submitted_at;

    // Submitting host work never runs it: the caller is back before the first
    // task can possibly have finished.
    REQUIRE(submit_cost < ms);

    const auto before = std::chrono::steady_clock::now();
    s.q.wait();
    const auto elapsed = std::chrono::steady_clock::now() - before;

    REQUIRE(ran->load() == tasks);
    // Serialized, these would take four sleeps; concurrent, they take one.
    REQUIRE(elapsed < (tasks - 1) * ms);
}

TEST_CASE("two host tasks over the same bytes are ordered by the caller's event", "[host]") {
    Stream s;

    int* p = s.alloc_host(1);
    p[0]   = 1;

    // ×10 then +1 is 11; the other order is 20. Only the event says which.
    sycl::event scale = s.q.async_host([p] {
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
        p[0] = p[0] * 10;
    });
    s.q.async_host(std::vector<sycl::event>{scale}, [p] { p[0] = p[0] + 1; });
    s.q.wait();

    REQUIRE(p[0] == 11);
    s.release(p);
}

// -----------------------------------------------------------------------------
// 3. the _in_order variants are serialized
// -----------------------------------------------------------------------------

TEST_CASE("in-order host tasks run one after another, in submission order", "[host]") {
    Stream s;

    int* p = s.alloc_host(1);
    p[0]   = 1;

    // A descending sleep, and three operations no two of which commute: run in
    // submission order this is 1 -> 2 -> 20 -> 27, and run in sleep order --
    // which is what a concurrent lane produces -- it is 81.
    s.q.async_host_in_order([p] {
        std::this_thread::sleep_for(std::chrono::milliseconds{120});
        p[0] = p[0] + 1;
    });
    s.q.async_host_in_order([p] {
        std::this_thread::sleep_for(std::chrono::milliseconds{60});
        p[0] = p[0] * 10;
    });
    s.q.async_host_in_order([p] { p[0] = p[0] + 7; });
    s.q.wait();

    REQUIRE(p[0] == 27);
    s.release(p);
}

TEST_CASE("in-order memory tasks are serialized against each other", "[host]") {
    Stream s;

    // Two memory tasks over the SAME bytes and nothing else ordering them: the
    // asynchronous variants would overlap, and the second's bytes would land
    // among the first's. Large enough that they genuinely overlap.
    const std::size_t n = 1u << 22;

    int* p = s.alloc_host(n);
    std::vector<int> src(n, 7);

    const int five = 5;
    s.q.fill_in_order(p, five, n);
    s.q.memcpy_in_order(p, src.data(), n * sizeof(int));
    s.q.wait();

    std::size_t wrong = 0;
    for (std::size_t i = 0; i < n; ++i)
        if (p[i] != 7) ++wrong;
    REQUIRE(wrong == 0);

    s.release(p);
}

TEST_CASE("in-order memory tasks join the kernel order", "[host]") {
    Stream s;

    int* p = s.alloc_host(2);
    p[0]   = 0;
    p[1]   = 0;

    const int one = 1;
    const int ten = 10;

    // fill 1, then ×10 in a kernel, then +1 by a host task: 11 in this order,
    // and 1, 2 or 10 in any other.
    s.q.fill_in_order(p, one, 1);
    s.q.single_task([=] { p[0] = p[0] * ten; });
    s.q.async_host_in_order([p] { p[0] = p[0] + 1; });
    s.q.memcpy_in_order(p + 1, p, sizeof(int));
    s.q.wait();

    REQUIRE(p[0] == 11);
    REQUIRE(p[1] == 11);
    s.release(p);
}

// -----------------------------------------------------------------------------
// 4. kernel-then-memory and memory-then-kernel
// -----------------------------------------------------------------------------

TEST_CASE("a memory task begins after the kernel work outstanding when it was submitted",
          "[host]") {
    Stream s;

    int* p   = s.alloc_host(1);
    int* out = s.alloc_host(1);
    p[0]     = 0;
    out[0]   = 0;

    const unsigned expected = churn(7u, kSpins);
    s.q.single_task([=] {
        unsigned acc = 7u;
        for (int spin = 0; spin < kSpins; ++spin) acc = acc * 1103515245u + 12345u;
        p[0] = static_cast<int>(acc);
    });

    // No wait between the two: the merge is the only thing ordering them.
    s.q.memcpy(out, p, sizeof(int)).wait();

    REQUIRE(out[0] == static_cast<int>(expected));
    s.release(p);
    s.release(out);
}

TEST_CASE("a kernel begins after the memory work outstanding when it was submitted", "[host]") {
    Stream s;

    int* p = s.alloc_host(1);
    p[0]   = 0;

    const int five = 5;
    s.q.fill(p, five, 1);
    // No wait: the merge is the only thing ordering these. ×10 then +1 is 51;
    // a kernel that ran first would leave 5.
    s.q.single_task([=] { p[0] = p[0] * 10 + 1; });
    s.q.wait();

    REQUIRE(p[0] == 51);
    s.release(p);
}

// -----------------------------------------------------------------------------
// 5. a group is one task
// -----------------------------------------------------------------------------

TEST_CASE("a group mixes kernel and memory subtasks, and is one task to both sides", "[host]") {
    Stream s;

    int* p = s.alloc_host(4);
    for (int i = 0; i < 4; ++i) p[i] = 0;

    // Before the group: the value every subtask starts from.
    s.q.single_task([=] { p[0] = 2; });

    const int seven = 7;
    const auto events = s.q.submit_group([&](queue::group& g) {
        g.submit([&](sycl::handler& h) { h.single_task([=] { p[1] = p[0] * 10; }); });
        g.memcpy(p + 2, p, sizeof(int));
        g.fill(p + 3, seven, 1);
    });
    REQUIRE(events.size() == 3);

    // After the group: every subtask's write is visible, whichever order they
    // ran in. 20 + 2 + 7 = 29, and no other value is reachable.
    s.q.single_task([=] { p[0] = p[1] + p[2] + p[3]; });
    s.q.wait();

    REQUIRE(p[0] == 29);
    s.release(p);
}

TEST_CASE("a group's subtasks do not wait for each other", "[host]") {
    Stream s;

    const auto ms = std::chrono::milliseconds{200};
    auto       ran = std::make_shared<std::atomic<int>>(0);

    const auto before = std::chrono::steady_clock::now();
    s.q.submit_group([&](queue::group& g) {
        for (int i = 0; i < 3; ++i)
            g.async_host([ran, ms] {
                std::this_thread::sleep_for(ms);
                ran->fetch_add(1);
            });
    });
    s.q.wait();
    const auto elapsed = std::chrono::steady_clock::now() - before;

    REQUIRE(ran->load() == 3);
    REQUIRE(elapsed < 2 * ms);
}

TEST_CASE("an empty group leaves the stream's order alone", "[host]") {
    Stream s;

    int* p = s.alloc_host(1);
    p[0]   = 1;

    s.q.single_task([=] { p[0] = p[0] * 10; });
    const auto events = s.q.submit_group([](queue::group&) {});
    REQUIRE(events.empty());
    s.q.single_task([=] { p[0] = p[0] + 1; });
    s.q.wait();

    REQUIRE(p[0] == 11);
    s.release(p);
}

// -----------------------------------------------------------------------------
// 6. a placed in-order host task
// -----------------------------------------------------------------------------

TEST_CASE("an in-order host task placed on a device keeps the stream's order", "[host]") {
    Stream s;

    int* p = s.alloc_host(1);
    p[0]   = 1;

    s.q.single_task(s.dev, [=] { p[0] = p[0] + 1; });
    s.q.async_host_in_order(s.dev, [p] { p[0] = p[0] * 10; });
    s.q.single_task(s.dev, [=] { p[0] = p[0] + 1; });
    s.q.wait();

    REQUIRE(p[0] == 21);
    s.release(p);
}
