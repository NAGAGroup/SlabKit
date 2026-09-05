// =============================================================================
// test_memory.cpp — the queue's memory tasks and the routes they take
// =============================================================================
//
// memcpy, copy, memset and fill over any pair of operands the queue's context
// can locate: two allocations on one device, an allocation and ordinary host
// memory, and two allocations no single context spans -- which is the staged
// route, and the one the original cross-backend defect lived on.

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "naga/slabkit/slabkit.hpp"
#include "support.hpp"

using namespace naga::slabkit;
namespace sk = naga::slabkit;
using slabkit_test::Fixture;

namespace {

/// A recurrence the optimizer cannot fold away and the host can reproduce.
unsigned churn(unsigned seed, int rounds) {
    unsigned acc = seed;
    for (int i = 0; i < rounds; ++i) acc = acc * 1103515245u + 12345u;
    return acc;
}

}  // namespace

TEST_CASE("a copy between two allocations on one device moves the bytes", "[memory]") {
    Fixture f;

    auto src = make_unique_device<int[]>(f.q, 8, 4);
    auto dst = sk::malloc_device<int>(8, f.q);

    f.q.copy(src.get(), dst.get(), 8);
    f.q.wait();
    for (int v : slabkit_test::read_back(f.q, dst.get(), 8)) REQUIRE(v == 4);
}

TEST_CASE("ordinary host memory is a legal operand in both directions", "[memory]") {
    Fixture f;

    std::vector<int> host{5, 6, 7, 8};
    auto             dst = sk::malloc_device<int>(4, f.q);

    // The caller owns the source until the task completes, as for sycl::queue.
    f.q.copy(host.data(), dst.get(), host.size()).wait();
    REQUIRE(slabkit_test::read_back(f.q, dst.get(), 4) == host);

    std::vector<int> back(4, 0);
    f.q.copy(dst.get(), back.data(), back.size()).wait();
    REQUIRE(back == host);
}

TEST_CASE("a copy sees the kernel that produced its source", "[memory]") {
    Fixture f;

    auto src = make_unique_device<int[]>(f.q, 4, 0);
    auto dst = sk::malloc_device<int>(4, f.q);

    int* p = src.get();
    f.q.parallel_for(sycl::range<1>{4}, [=](sycl::id<1> i) {
        for (int spin = 0; spin < 2000000; ++spin) p[i[0]] = p[i[0]] + 1;
    });

    // No wait: the merge orders the copy behind the kernel.
    f.q.copy(src.get(), dst.get(), 4);
    f.q.wait();
    for (int v : slabkit_test::read_back(f.q, dst.get(), 4)) REQUIRE(v == 2000000);
}

TEST_CASE("memset and fill run on the operand's own device", "[memory]") {
    Fixture f;

    auto p = sk::malloc_device<int>(8, f.q);

    f.q.memset(p.get(), 0, 8 * sizeof(int));
    f.q.wait();
    for (int v : slabkit_test::read_back(f.q, p.get(), 8)) REQUIRE(v == 0);

    f.q.fill(p.get(), 12345, 8);
    f.q.wait();
    for (int v : slabkit_test::read_back(f.q, p.get(), 8)) REQUIRE(v == 12345);
}

TEST_CASE("a fill pattern wider than a register is staged", "[memory]") {
    struct Wide {
        int a, b, c, d, e;
    };
    STATIC_REQUIRE(sizeof(Wide) > sizeof(std::uint64_t));

    Fixture f;
    auto    p = sk::malloc_device<Wide>(6, f.q);

    const Wide pattern{1, 2, 3, 4, 5};
    f.q.fill(p.get(), pattern, 6);
    f.q.wait();

    for (const Wide& w : slabkit_test::read_back(f.q, p.get(), 6)) {
        REQUIRE(w.a == 1);
        REQUIRE(w.e == 5);
    }
}

TEST_CASE("a memory task composes with explicit dependencies", "[memory]") {
    Fixture f;

    auto        a     = make_unique_device<int[]>(f.q, 4, 2);
    auto        b     = sk::malloc_device<int>(4, f.q);
    sycl::event first = f.q.single_task([p = a.get()] { p[0] = 99; });

    f.q.copy(std::vector<sycl::event>{first}, a.get(), b.get(), 4);
    f.q.wait();

    const auto host = slabkit_test::read_back(f.q, b.get(), 4);
    REQUIRE(host[0] == 99);
    REQUIRE(host[1] == 2);
}

TEST_CASE("a zero-byte memory task is a no-op task, not an error", "[memory]") {
    Fixture f;
    auto    p = sk::malloc_device<int>(4, f.q);

    REQUIRE_NOTHROW(f.q.memcpy(p.get(), p.get(), 0));
    REQUIRE_NOTHROW(f.q.memset(p.get(), 0, 0));
    REQUIRE_NOTHROW(f.q.fill(p.get(), 1, 0));
    f.q.wait();
}

TEST_CASE("memory tasks are subtasks of a group", "[memory]") {
    Fixture f;

    auto a = make_unique_device<int[]>(f.q, 4, 3);
    auto b = sk::malloc_device<int>(4, f.q);
    auto c = sk::malloc_device<int>(4, f.q);

    const auto events = f.q.submit_group([&](queue::group& g) {
        g.copy(a.get(), b.get(), 4);
        g.fill(c.get(), 9, 4);
    });
    REQUIRE(events.size() == 2);
    f.q.wait();

    for (int v : slabkit_test::read_back(f.q, b.get(), 4)) REQUIRE(v == 3);
    for (int v : slabkit_test::read_back(f.q, c.get(), 4)) REQUIRE(v == 9);
}

TEST_CASE("a copy across backends is right every time", "[memory][cross-device]") {
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
        // staged copy after it is genuinely racing something.
        q.parallel_for(first, sycl::range<1>{n}, [=](sycl::id<1> i) {
            unsigned acc = mark;
            for (int spin = 0; spin < 200000; ++spin) acc = acc * 1103515245u + 12345u;
            p[i[0]] = static_cast<int>(acc);
        });

        q.memcpy(dst.get(), src.get(), n * sizeof(int));
        q.wait();

        const auto back = slabkit_test::read_back(q, other, dst.get(), n);
        for (int v : back) REQUIRE(v == expected);
    }
}

TEST_CASE("a copy across devices moves the bytes on every machine", "[memory][cross-device]") {
    context ctx{slabkit_test::all_devices()};
    queue   q{ctx};
    if (ctx.devices().size() < 2) return;

    const sycl::device a = ctx.devices()[0];
    const sycl::device b = ctx.devices()[1];

    auto src = make_unique_device<int[]>(q, a, 8, 3);
    auto dst = sk::malloc_device<int>(8, q, b);

    q.copy(src.get(), dst.get(), 8);
    q.wait();
    for (int v : slabkit_test::read_back(q, b, dst.get(), 8)) REQUIRE(v == 3);
}
