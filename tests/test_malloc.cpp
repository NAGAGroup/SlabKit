// The malloc family: every form against every placement, plus the null cases.

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "naga/slabkit/slabkit.hpp"
#include "support.hpp"

using namespace naga::slabkit;
namespace sk = naga::slabkit;
using slabkit_test::Fixture;

TEST_CASE("storage handles are array handles") {
    Fixture f;
    STATIC_REQUIRE(
        std::is_same_v<decltype(sk::malloc_device<float>(1, f.q)), unique_usm_ptr<float[]>>);
    STATIC_REQUIRE(std::is_same_v<decltype(sk::aligned_alloc_device<float>(64, 1, f.q)),
                                  unique_usm_ptr<float[]>>);
}

TEST_CASE("malloc_device serves every placement") {
    Fixture     f;
    DeviceArena arena{f.q, kind::device, 65536};
    DevicePool  pool{f.q, kind::device, 65536, 256};

    SECTION("queue") {
        auto h = sk::malloc_device<int>(32, f.q);
        REQUIRE(h);
    }
    SECTION("queue with properties") {
        auto h = sk::malloc_device<int>(32, f.q, sycl::property_list{});
        REQUIRE(h);
    }
    SECTION("device and context") {
        auto h = sk::malloc_device<int>(32, f.dev, f.ctx);
        REQUIRE(h);
    }
    SECTION("arena") {
        auto h = sk::malloc_device<int>(32, arena);
        REQUIRE(h);
    }
    SECTION("pool") {
        auto h = sk::malloc_device<int>(32, pool);
        REQUIRE(h);
    }
}

TEST_CASE("aligned_alloc_device serves every placement") {
    Fixture     f;
    DeviceArena arena{f.q, kind::device, 65536};
    DevicePool  pool{f.q, kind::device, 65536, 256};

    // NOTE: no alignment is asserted for the queue and device forms. SYCL makes
    // no promise beyond the natural alignment, and AdaptiveCpp's CUDA allocator
    // discards the request outright -- asserting here would be testing luck.
    SECTION("queue") { REQUIRE(sk::aligned_alloc_device<int>(128, 32, f.q)); }
    SECTION("device and context") {
        REQUIRE(sk::aligned_alloc_device<int>(128, 32, f.dev, f.ctx));
    }
    SECTION("arena honors it") {
        auto h = sk::aligned_alloc_device<int>(128, 32, arena);
        REQUIRE(h);
        REQUIRE(reinterpret_cast<std::uintptr_t>(h.get()) % 128 == 0);
    }
    SECTION("pool honors it up to the node size") {
        auto h = sk::aligned_alloc_device<int>(256, 32, pool);
        REQUIRE(h);
        REQUIRE(reinterpret_cast<std::uintptr_t>(h.get()) % 256 == 0);
    }
}

TEST_CASE("storage from every placement is writable and readable from a kernel") {
    Fixture     f;
    DeviceArena arena{f.q, kind::device, 65536};
    DevicePool  pool{f.q, kind::device, 65536, 256};

    auto check = [&](unique_usm_ptr<int[]> h) {
        REQUIRE(h);
        int* p = h.get();
        f.q.parallel_for(sycl::range<1>{32},
                         [=](sycl::id<1> i) { p[i[0]] = static_cast<int>(i[0]) + 7; })
            .wait();
        const auto host = slabkit_test::read_back(f.q, h.get(), 32);
        for (std::size_t i = 0; i < 32; ++i) REQUIRE(host[i] == static_cast<int>(i) + 7);
    };

    check(sk::malloc_device<int>(32, f.q));
    check(sk::malloc_device<int>(32, f.dev, f.ctx));
    check(sk::malloc_device<int>(32, arena));
    check(sk::malloc_device<int>(32, pool));
}

TEST_CASE("a zero-element request mirrors SYCL and never leaks") {
    Fixture f;

    // slabkit adds no rule of its own here: whatever the backend returns is
    // what you get. Either answer is correct; both must be safe.
    auto from_queue = sk::malloc_device<int>(0, f.q);
    auto from_dev   = sk::malloc_device<int>(0, f.dev, f.ctx);

    SUCCEED("zero-element storage requests completed");
    // Destruction at scope exit is the actual assertion: a live allocation
    // must free cleanly, and a null handle must not be freed at all.
}

TEST_CASE("handles are movable and release ownership on reset") {
    Fixture    f;
    DevicePool pool{f.q, kind::device, 4096, 256};

    auto h = sk::malloc_device<std::byte>(256, pool);
    REQUIRE(pool.outstanding() == 1);

    auto moved = std::move(h);
    REQUIRE_FALSE(h);
    REQUIRE(moved);
    REQUIRE(pool.outstanding() == 1);

    moved.reset();
    REQUIRE(pool.outstanding() == 0);
}
