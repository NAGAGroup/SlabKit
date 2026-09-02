// Foreign placement: allocating and initializing for a device other than the
// one the initialization queue was built for.
//
// This is the case that killed the retarget mechanism. AdaptiveCpp resolves a
// memcpy's operands by querying each pointer against the QUEUE's context, so a
// pointer belonging to another device's context is misread as host memory.
// slabkit initializes on a queue built from the allocation's own context
// instead, which cannot make that mistake.
//
// The tests skip themselves when the environment enumerates only one device.

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <span>
#include <vector>

#include "naga/slabkit/slabkit.hpp"
#include "support.hpp"

using namespace naga::slabkit;
namespace sk = naga::slabkit;
using slabkit_test::Fixture;

TEST_CASE("a foreign device's storage is allocated and freed with its own context") {
    Fixture      f;
    sycl::device other;
    if (!slabkit_test::second_device(f.dev, other)) {
        SUCCEED("only one device is visible; nothing to test");
        return;
    }

    // The contract's construction order: this device gets its OWN context.
    sycl::context other_ctx{other};

    auto h = sk::malloc_device<int>(64, other, other_ctx);
    REQUIRE(h);
    REQUIRE(std::get<OsFree>(h.get_deleter().how).ctx == other_ctx);
}

TEST_CASE("make initializes correctly on a device the queue was not built for") {
    Fixture      f;
    sycl::device other;
    if (!slabkit_test::second_device(f.dev, other)) {
        SUCCEED("only one device is visible; nothing to test");
        return;
    }

    sycl::context other_ctx{other};
    sycl::queue   other_q{other_ctx, other};

    SECTION("fill form") {
        // f.q belongs to the DEFAULT device; the placement is the other one.
        auto h = make_unique_device<int[]>(f.q, other, other_ctx, 32, 77);
        REQUIRE(h);
        const auto host = slabkit_test::read_back(other_q, h.get(), 32);
        for (std::size_t i = 0; i < 32; ++i) REQUIRE(host[i] == 77);
    }

    SECTION("copy form -- the one retarget could not do") {
        std::vector<int> src(32);
        for (std::size_t i = 0; i < src.size(); ++i) src[i] = static_cast<int>(i) + 100;

        auto h = make_unique_device<int[]>(f.q, other, other_ctx, std::span<const int>{src});
        REQUIRE(h);
        const auto host = slabkit_test::read_back(other_q, h.get(), src.size());
        REQUIRE(host == src);
    }

    SECTION("value-initialized form") {
        auto h = make_unique_device<int[]>(f.q, other, other_ctx, 16);
        REQUIRE(h);
        for (int v : slabkit_test::read_back(other_q, h.get(), 16)) REQUIRE(v == 0);
    }
}

TEST_CASE("an arena on a foreign device initializes through that device") {
    Fixture      f;
    sycl::device other;
    if (!slabkit_test::second_device(f.dev, other)) {
        SUCCEED("only one device is visible; nothing to test");
        return;
    }

    sycl::context other_ctx{other};
    sycl::queue   other_q{other_ctx, other};

    DeviceArena arena{other, other_ctx, 65536};
    REQUIRE(arena.device() == other);

    auto h = make_unique_device<int[]>(f.q, arena, 24, 13);
    REQUIRE(h);
    const auto host = slabkit_test::read_back(other_q, h.get(), 24);
    for (std::size_t i = 0; i < 24; ++i) REQUIRE(host[i] == 13);
}

TEST_CASE("a pool on a foreign device initializes through that device") {
    Fixture      f;
    sycl::device other;
    if (!slabkit_test::second_device(f.dev, other)) {
        SUCCEED("only one device is visible; nothing to test");
        return;
    }

    sycl::context other_ctx{other};
    sycl::queue   other_q{other_ctx, other};

    DevicePool pool{other, other_ctx, 65536, 256};
    auto       h = make_unique_device<int[]>(f.q, pool, 24, 21);
    REQUIRE(h);
    const auto host = slabkit_test::read_back(other_q, h.get(), 24);
    for (std::size_t i = 0; i < 24; ++i) REQUIRE(host[i] == 21);
}
