// DevicePool: node geometry, the alignment guarantee, exhaustion,
// fragmentation, and thread safety.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

#include "naga/slabkit/slabkit.hpp"
#include "support.hpp"

using namespace naga::slabkit;
namespace sk = naga::slabkit;
using slabkit_test::Fixture;

TEST_CASE("a fresh pool reports its geometry") {
    Fixture    f;
    DevicePool pool{f.q, kind::device, 4096, 256};

    REQUIRE(pool.node_size() == 256);
    REQUIRE(pool.slot_count() == 16);
    REQUIRE(pool.free_count() == 16);
    REQUIRE(pool.outstanding() == 0);
    REQUIRE(pool.device() == f.dev);
    REQUIRE(pool.context() == f.ctx);
}

TEST_CASE("a node size that is not a power of two is a usage error") {
    Fixture f;
    REQUIRE_THROWS_AS((DevicePool{f.q, kind::device, 4096, 300}), std::invalid_argument);
    REQUIRE_THROWS_AS((DevicePool{f.dev, f.ctx, kind::device, 4096, 0}), std::invalid_argument);
    REQUIRE_NOTHROW((DevicePool{f.q, kind::device, 4096, 512}));
}

TEST_CASE("pool allocations are node-aligned") {
    Fixture f;
    for (std::size_t node : {std::size_t{256}, std::size_t{1024}, std::size_t{4096}}) {
        DevicePool pool{f.q, kind::device, node * 8, node};
        auto       h = sk::malloc_device<std::byte>(node / 2, pool);
        REQUIRE(h);
        REQUIRE(reinterpret_cast<std::uintptr_t>(h.get()) % node == 0);
    }
}

TEST_CASE("an allocation is rounded up to whole nodes") {
    Fixture    f;
    DevicePool pool{f.q, kind::device, 4096, 256};

    auto one = sk::malloc_device<std::byte>(1, pool);
    REQUIRE(one);
    REQUIRE(pool.outstanding() == 1);
    one.reset();

    auto spans_three = sk::malloc_device<std::byte>(513, pool);
    REQUIRE(spans_three);
    REQUIRE(pool.outstanding() == 3);
}

TEST_CASE("an alignment larger than the node size is a usage error") {
    Fixture    f;
    DevicePool pool{f.q, kind::device, 4096, 256};
    REQUIRE_THROWS_AS(sk::aligned_alloc_device<std::byte>(512, 16, pool), std::invalid_argument);
    REQUIRE_NOTHROW(sk::aligned_alloc_device<std::byte>(256, 16, pool));
}

TEST_CASE("an exhausted pool returns null") {
    Fixture    f;
    DevicePool pool{f.q, kind::device, 1024, 256};  // four slots

    std::vector<unique_usm_ptr<std::byte[]>> held;
    for (int i = 0; i < 4; ++i) {
        auto h = sk::malloc_device<std::byte>(256, pool);
        REQUIRE(h);
        held.push_back(std::move(h));
    }
    REQUIRE(pool.free_count() == 0);
    REQUIRE_FALSE(sk::malloc_device<std::byte>(256, pool));

    held.pop_back();
    REQUIRE(pool.free_count() == 1);
    REQUIRE(sk::malloc_device<std::byte>(256, pool));
}

TEST_CASE("a fragmented pool refuses a run it cannot place") {
    Fixture    f;
    DevicePool pool{f.q, kind::device, 1024, 256};  // four slots

    auto a = sk::malloc_device<std::byte>(256, pool);
    auto b = sk::malloc_device<std::byte>(256, pool);
    auto c = sk::malloc_device<std::byte>(256, pool);
    auto d = sk::malloc_device<std::byte>(256, pool);
    REQUIRE(a);
    REQUIRE(b);
    REQUIRE(c);
    REQUIRE(d);

    // Free slots 0 and 2: two nodes are free, but never adjacent.
    a.reset();
    c.reset();
    REQUIRE(pool.free_count() == 2);
    REQUIRE_FALSE(sk::malloc_device<std::byte>(512, pool));  // needs two in a row
    REQUIRE(sk::malloc_device<std::byte>(256, pool));        // one still fits
}

TEST_CASE("a contiguous run really is contiguous") {
    Fixture    f;
    DevicePool pool{f.q, kind::device, 8192, 256};

    auto h = sk::malloc_device<std::byte>(1024, pool);  // four nodes
    REQUIRE(h);
    REQUIRE(pool.outstanding() == 4);

    std::byte* p = h.get();
    f.q.parallel_for(sycl::range<1>{1024}, [=](sycl::id<1> i) {
         p[i[0]] = static_cast<std::byte>(i[0] % 251);
     }).wait();

    const auto host = slabkit_test::read_back(f.q, h.get(), 1024);
    for (std::size_t i = 0; i < 1024; ++i)
        REQUIRE(host[i] == static_cast<std::byte>(i % 251));
}

TEST_CASE("a run longer than one bitmap word is placed") {
    Fixture    f;
    DevicePool pool{f.q, kind::device, 256 * 200, 256};  // 200 slots, spans several words

    auto h = sk::malloc_device<std::byte>(256 * 100, pool);  // 100 nodes
    REQUIRE(h);
    REQUIRE(pool.outstanding() == 100);
    h.reset();
    REQUIRE(pool.outstanding() == 0);
}

TEST_CASE("a zero-element request from a pool is null") {
    Fixture    f;
    DevicePool pool{f.q, kind::device, 4096, 256};

    REQUIRE_FALSE(sk::malloc_device<int>(0, pool));
    REQUIRE(pool.outstanding() == 0);
}

TEST_CASE("copies of a pool handle are the same pool") {
    Fixture    f;
    DevicePool pool{f.q, kind::device, 4096, 256};
    DevicePool copy = pool;

    auto h = sk::malloc_device<std::byte>(256, copy);
    REQUIRE(h);
    REQUIRE(pool.outstanding() == 1);
    REQUIRE(copy.outstanding() == 1);
}

TEST_CASE("the free list survives concurrent traffic") {
    Fixture    f;
    DevicePool pool{f.q, kind::device, 256 * 512, 256};

    constexpr int kThreads = 8;
    constexpr int kRounds  = 200;

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < kRounds; ++i) {
                auto h = sk::malloc_device<std::byte>(256, pool);
                if (h) {
                    // Touch nothing; the point is the bitmap, not the memory.
                    h.reset();
                }
            }
        });
    }
    for (auto& th : threads) th.join();

    REQUIRE(pool.outstanding() == 0);
    REQUIRE(pool.free_count() == pool.slot_count());
}
