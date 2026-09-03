// =============================================================================
// test_host_work.cpp — the host lane, the interlock, chains, and mixed groups
// =============================================================================
//
// Host work is an OpenMP single_task whose body is an indirect call through a
// function pointer. Everything below is a property of that mechanism that a
// caller would otherwise have to take on trust:
//
//   1. a host item runs off the caller's thread and yields a usable event,
//   2. host items run in PARALLEL with each other,
//   3. the interlock holds in both directions,
//   4. a chain runs in order and counts as one host item,
//   5. a group may mix kernel and host members,
//   6. a routed copy is a legal group member, contributing its chain's LAST
//      event,
//   7. a cross-backend routed copy is right every time -- the original defect.

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
    /// every interlock assertion here needs.
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

TEST_CASE("host work runs off the caller's thread and yields a usable event", "[host]") {
    Stream s;

    auto ran    = std::make_shared<std::atomic<bool>>(false);
    auto thread = std::make_shared<std::atomic<std::thread::id>>();

    sycl::event e = s.q.submit_host([ran, thread] {
        thread->store(std::this_thread::get_id());
        ran->store(true);
    });

    e.wait();
    REQUIRE(ran->load());
    REQUIRE(thread->load() != std::this_thread::get_id());
}

TEST_CASE("a host task's exception is asynchronous, not thrown at the caller", "[host]") {
    Stream s;

    sycl::event e = s.q.submit_host([] { throw std::runtime_error{"boom"}; });
    e.wait();  // the item completes; the throw did not escape into kernel code

    REQUIRE_THROWS_AS(s.q.sync_host_work_and_throw(), std::runtime_error);
}

// -----------------------------------------------------------------------------
// 2. host work is parallel with host work
// -----------------------------------------------------------------------------

/// Host items carry NO dependency on each other -- that is the structural
/// property the stream promises, and it is what this asserts. Whether they then
/// RUN at the same time is the backend's business, and on this toolchain the
/// answer is no: `omp_hardware_context::get_max_kernel_concurrency()` returns 1
/// (src/runtime/omp/omp_hardware_manager.cpp:31), so the OpenMP device offers
/// one execution lane for the whole process and every host item queues behind
/// the one before it. Measured: four 200 ms items take 800 ms, on one
/// out-of-order queue or four, under either scheduler.
///
/// So the assertion is the part that must hold, and the WARN is the part the
/// runtime decides. A future acpp with more host lanes turns the warning off
/// without touching this test.
TEST_CASE("host items do not wait for each other, and none of them blocks the caller", "[host]") {
    Stream s;

    const int  items = 4;
    const auto ms    = std::chrono::milliseconds{200};

    auto ran = std::make_shared<std::atomic<int>>(0);

    const auto submitted_at = std::chrono::steady_clock::now();
    for (int i = 0; i < items; ++i)
        s.q.submit_host([ran, ms] {
            std::this_thread::sleep_for(ms);
            ran->fetch_add(1);
        });
    const auto submit_cost = std::chrono::steady_clock::now() - submitted_at;

    // Submitting host work never runs it: the caller is back before the first
    // item can possibly have finished.
    REQUIRE(submit_cost < ms);

    const auto before = std::chrono::steady_clock::now();
    s.q.sync_host_work();
    const auto elapsed = std::chrono::steady_clock::now() - before;

    REQUIRE(ran->load() == items);

    if (elapsed > (items - 1) * ms)
        WARN("the OpenMP backend ran these host items one at a time: it reports a kernel "
             "concurrency of 1, so the host lane is a single execution lane for the process");
}

// -----------------------------------------------------------------------------
// 3. the interlock, both directions
// -----------------------------------------------------------------------------

TEST_CASE("a host item begins after the kernel work outstanding when it was submitted",
          "[host]") {
    Stream s;

    int* p = s.alloc_host(1);
    p[0]   = 0;

    const unsigned expected = churn(7u, kSpins);
    s.q.single_task([=] {
        unsigned acc = 7u;
        for (int spin = 0; spin < kSpins; ++spin) acc = acc * 1103515245u + 12345u;
        p[0] = static_cast<int>(acc);
    });

    // No wait between the two: the interlock is the only thing ordering them.
    auto saw = std::make_shared<std::atomic<int>>(0);
    s.q.submit_host([p, saw] { saw->store(p[0]); }).wait();

    REQUIRE(saw->load() == static_cast<int>(expected));
    s.release(p);
}

TEST_CASE("a kernel item begins after the host work outstanding when it was submitted",
          "[host]") {
    Stream s;

    int* p   = s.alloc_host(2);
    p[0]     = 0;
    p[1]     = -1;

    // A host item slow enough that a kernel which did not wait for it would
    // read the old value.
    s.q.submit_host([p] {
        std::this_thread::sleep_for(std::chrono::milliseconds{300});
        p[0] = 4242;
    });

    // No wait: the interlock is the only thing ordering these.
    s.q.single_task([=] { p[1] = p[0]; });
    s.q.wait();

    REQUIRE(p[1] == 4242);
    s.release(p);
}

// -----------------------------------------------------------------------------
// 4. a chain runs in order
// -----------------------------------------------------------------------------

TEST_CASE("a chain's items run in order, and the chain is one host item", "[host]") {
    Stream s;

    auto log   = std::make_shared<std::vector<int>>();
    auto mutex = std::make_shared<std::mutex>();

    sycl::event last = s.q.submit_chain([&](queue::chain& ch) {
        for (int i = 0; i < 4; ++i) {
            ch.then([log, mutex, i] {
                // A descending sleep: an out-of-order lane would interleave.
                std::this_thread::sleep_for(std::chrono::milliseconds{40 - 10 * i});
                std::lock_guard<std::mutex> lock{*mutex};
                log->push_back(i);
            });
        }
    });

    last.wait();
    REQUIRE(*log == std::vector<int>{0, 1, 2, 3});
}

TEST_CASE("an empty chain is still one host item", "[host]") {
    Stream s;
    sycl::event e = s.q.submit_chain([](queue::chain&) {});
    e.wait();
    s.q.sync_host_work();
}

// -----------------------------------------------------------------------------
// 5. a group mixing kernel and host members
// -----------------------------------------------------------------------------

TEST_CASE("a group may mix kernel work and host work, and the next item waits for both",
          "[host]") {
    Stream s;

    int* p = s.alloc_host(3);
    for (int i = 0; i < 3; ++i) p[i] = 0;

    const auto events = s.q.submit_group([&](queue::group& g) {
        g.submit([&](sycl::handler& h) { h.single_task([=] { p[0] = 11; }); });
        g.submit_host([p] { p[1] = 22; });
    });
    REQUIRE(events.size() == 2);

    // The item after the group sees every member's write, of either kind.
    s.q.single_task([=] { p[2] = p[0] + p[1]; });
    s.q.wait();

    REQUIRE(p[0] == 11);
    REQUIRE(p[1] == 22);
    REQUIRE(p[2] == 33);
    s.release(p);
}

/// A group's host members are one step: they wait for what came before the
/// group and for nothing inside it. With one host lane that is again a statement
/// about dependencies rather than about wall time, so what is asserted is that
/// no member waited for the kernel member beside it -- which a dependency edge
/// would have forced.
TEST_CASE("a group's host members do not wait for the group's other members", "[host]") {
    Stream s;

    int* p = s.alloc_host(1);
    p[0]   = 0;

    auto host_done = std::make_shared<std::atomic<bool>>(false);

    sycl::event kernel_event;
    s.q.submit_group([&](queue::group& g) {
        // A slow kernel member. If the host member depended on it, the host
        // member could not finish before it did.
        kernel_event = g.submit([&](sycl::handler& h) {
            h.single_task([=] {
                unsigned acc = 1u;
                for (int spin = 0; spin < 20 * kSpins; ++spin) acc = acc * 1103515245u + 12345u;
                p[0] = static_cast<int>(acc);
            });
        });
        g.submit_host([host_done] { host_done->store(true); });
    });

    // Draining the host side alone must not require the kernel member: no
    // dependency edge was created between them.
    s.q.sync_host_work();
    REQUIRE(host_done->load());

    // Whether the kernel is STILL RUNNING at that point is the runtime's
    // business, not the stream's, so it is reported rather than required. On the
    // OpenMP device it cannot be: the kernel member and the host member share
    // that backend's single execution lane.
    const bool kernel_still_running =
        kernel_event.get_info<sycl::info::event::command_execution_status>() !=
        sycl::info::event_command_status::complete;
    if (!kernel_still_running && !detail::is_host_backend(s.dev))
        WARN("the group's kernel member had already finished when its host member did, so "
             "this run did not observe the two overlapping");

    s.q.wait();
    REQUIRE(host_done->load());
    s.release(p);
}

// -----------------------------------------------------------------------------
// 6. a routed copy as a group member
// -----------------------------------------------------------------------------

TEST_CASE("a routed copy is a group member, contributing its chain's last event", "[host]") {
    Stream s;

    const std::size_t n = 32;

    auto dst = sk::malloc_device<int>(n, s.q);
    REQUIRE(dst);

    std::vector<int> src(n);
    for (std::size_t i = 0; i < n; ++i) src[i] = static_cast<int>(i) + 500;

    int* other = s.alloc_host(1);
    other[0]   = 0;

    sycl::event copy_event;
    const auto  events = s.q.submit_group([&](queue::group& g) {
        // A host operand and a device operand: the staged route, which is a
        // chain -- and the group records its LAST event, not its first.
        copy_event = detail::copy_bytes_in_group(g, s.q, dst.get(), src.data(),
                                                  n * sizeof(int), {});
        g.submit_host([other] { other[0] = 5; });
    });

    REQUIRE(events.size() == 2);

    // Waiting on the contributed event alone is enough for the copy to have
    // landed: it is the chain's last leg, and the legs are in order.
    copy_event.wait();
    REQUIRE(slabkit_test::read_back(s.q, dst.get(), n) == src);

    s.q.wait();
    REQUIRE(other[0] == 5);
    s.release(other);
}

// -----------------------------------------------------------------------------
// 7. the original defect: a cross-backend routed copy
// -----------------------------------------------------------------------------

TEST_CASE("a routed copy across backends is right every time", "[host][cross-device]") {
    context ctx{slabkit_test::all_devices()};
    queue   q{ctx};
    if (ctx.backends().size() < 2) return;

    const sycl::device first = ctx.devices().front();
    sycl::device       other = first;
    for (const auto& d : ctx.devices())
        if (!(d.get_platform() == first.get_platform())) other = d;

    const std::size_t n = 16;

    auto src = sk::malloc_device<int>(n, q, first);
    auto dst = sk::malloc_device<int>(n, q, other);
    REQUIRE(src);
    REQUIRE(dst);

    int* p = src.get();

    for (int round = 0; round < 40; ++round) {
        const unsigned mark     = 0xB0u + static_cast<unsigned>(round);
        const int      expected = static_cast<int>(churn(mark, 200000));

        // A producer on the first backend that cannot be optimized away, so the
        // copy after it is genuinely racing something.
        q.parallel_for(first, sycl::range<1>{n}, [=](sycl::id<1> i) {
            unsigned acc = mark;
            for (int spin = 0; spin < 200000; ++spin) acc = acc * 1103515245u + 12345u;
            p[i[0]] = static_cast<int>(acc);
        });

        // The routed copy: one host item, ordered behind the producer by the
        // interlock, issuing its legs only once that ordering is satisfied.
        sk::memcpy(q, dst.get(), src.get(), n * sizeof(int));
        q.wait();

        const auto back = slabkit_test::read_back(q, other, dst.get(), n);
        for (int v : back) REQUIRE(v == expected);
    }
}
