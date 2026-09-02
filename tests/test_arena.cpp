// DeviceArena: the bump, the alignment promise, exhaustion, and the fact that
// copies of the handle are the same arena.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

#include "naga/slabkit/slabkit.hpp"
#include "support.hpp"

using namespace naga::slabkit;
namespace sk = naga::slabkit;
using slabkit_test::Fixture;

TEST_CASE("a fresh arena is empty and reports its capacity") {
    Fixture     f;
    DeviceArena arena{f.q, 4096};

    REQUIRE(arena.capacity() == 4096);
    REQUIRE(arena.used() == 0);
    REQUIRE(arena.remaining() == 4096);
    REQUIRE(arena.device() == f.dev);
    REQUIRE(arena.context() == f.ctx);
}

TEST_CASE("the device+context constructor names the same memory") {
    Fixture     f;
    DeviceArena arena{f.dev, f.ctx, 2048};

    REQUIRE(arena.capacity() == 2048);
    REQUIRE(arena.device() == f.dev);
    REQUIRE(arena.context() == f.ctx);
}

TEST_CASE("allocation advances the bump") {
    Fixture     f;
    DeviceArena arena{f.q, 4096};

    auto a = sk::malloc_device<std::byte>(100, arena);
    REQUIRE(a);
    REQUIRE(arena.used() >= 100);
    const std::size_t after_first = arena.used();

    auto b = sk::malloc_device<std::byte>(200, arena);
    REQUIRE(b);
    REQUIRE(arena.used() >= after_first + 200);
    REQUIRE(arena.remaining() == arena.capacity() - arena.used());
}

TEST_CASE("an arena honors any alignment, and charges the padding") {
    Fixture     f;
    DeviceArena arena{f.q, 65536};

    // Push the bump off any convenient boundary first.
    auto skew = sk::malloc_device<std::byte>(1, arena);
    REQUIRE(skew);

    for (std::size_t align : {std::size_t{64}, std::size_t{256}, std::size_t{4096}}) {
        const std::size_t before = arena.used();
        auto              h      = sk::aligned_alloc_device<std::byte>(align, 32, arena);
        REQUIRE(h);
        const auto addr = reinterpret_cast<std::uintptr_t>(h.get());
        REQUIRE(addr % align == 0);
        // Padding skipped to reach the boundary is charged to the arena.
        REQUIRE(arena.used() >= before + 32);
    }
}

TEST_CASE("an arena that cannot satisfy a request returns null") {
    Fixture     f;
    DeviceArena arena{f.q, 1024};

    auto ok = sk::malloc_device<std::byte>(512, arena);
    REQUIRE(ok);

    auto too_big = sk::malloc_device<std::byte>(4096, arena);
    REQUIRE_FALSE(too_big);

    // The failed request must not have consumed anything.
    const std::size_t used = arena.used();
    auto              fits = sk::malloc_device<std::byte>(256, arena);
    REQUIRE(fits);
    REQUIRE(arena.used() >= used + 256);
}

TEST_CASE("an exhausted arena keeps returning null") {
    Fixture     f;
    DeviceArena arena{f.q, 256};

    auto all = sk::malloc_device<std::byte>(256, arena);
    REQUIRE(all);
    REQUIRE(arena.remaining() == 0);
    REQUIRE_FALSE(sk::malloc_device<std::byte>(1, arena));
}

TEST_CASE("a zero-element request from an arena is null") {
    Fixture     f;
    DeviceArena arena{f.q, 1024};

    auto h = sk::malloc_device<int>(0, arena);
    REQUIRE_FALSE(h);
    REQUIRE(arena.used() == 0);
}

TEST_CASE("copies of an arena handle are the same arena") {
    Fixture     f;
    DeviceArena arena{f.q, 4096};
    DeviceArena copy = arena;

    auto h = sk::malloc_device<std::byte>(128, copy);
    REQUIRE(h);
    REQUIRE(arena.used() == copy.used());
    REQUIRE(arena.used() >= 128);
}

TEST_CASE("allocations from an arena are usable from a kernel") {
    Fixture     f;
    DeviceArena arena{f.q, 4096};

    auto h = sk::malloc_device<int>(64, arena);
    REQUIRE(h);
    int* p = h.get();
    f.q.parallel_for(sycl::range<1>{64}, [=](sycl::id<1> i) {
         p[i[0]] = static_cast<int>(i[0]) * 2;
     }).wait();

    const auto host = slabkit_test::read_back(f.q, h.get(), 64);
    for (std::size_t i = 0; i < 64; ++i) REQUIRE(host[i] == static_cast<int>(i) * 2);
}

TEST_CASE("the bump survives concurrent allocation") {
    Fixture     f;
    DeviceArena arena{f.q, 1 << 20};

    constexpr int kThreads = 8;
    constexpr int kEach    = 64;

    std::vector<std::thread>              threads;
    std::vector<std::vector<std::byte*>>  got(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < kEach; ++i) {
                auto h = sk::malloc_device<std::byte>(64, arena);
                if (h) got[static_cast<std::size_t>(t)].push_back(h.release());
            }
        });
    }
    for (auto& th : threads) th.join();

    // Every allocation is distinct: no two threads got the same bytes.
    std::vector<std::byte*> all;
    for (auto& v : got) all.insert(all.end(), v.begin(), v.end());
    REQUIRE(all.size() == kThreads * kEach);
    std::sort(all.begin(), all.end());
    REQUIRE(std::adjacent_find(all.begin(), all.end()) == all.end());
}
