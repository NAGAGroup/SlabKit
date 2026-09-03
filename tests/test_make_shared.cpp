// make_shared_device: the same six forms, returning a shared handle whose
// deleter has been erased but still routes correctly.

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <span>
#include <type_traits>
#include <vector>

#include "naga/slabkit/slabkit.hpp"
#include "support.hpp"

using namespace naga::slabkit;
namespace sk = naga::slabkit;
using slabkit_test::Fixture;

namespace {
struct Plain {
    int a;
    int b;
};
}  // namespace

TEST_CASE("a shared handle is a plain shared_ptr") {
    STATIC_REQUIRE(std::is_same_v<shared_usm_ptr<int>, std::shared_ptr<int>>);
    STATIC_REQUIRE(std::is_same_v<shared_usm_ptr<int[]>, std::shared_ptr<int[]>>);
}

TEST_CASE("shared single-object forms work on every placement") {
    Fixture     f;
    DeviceArena arena{f.q, kind::device, 65536};
    DevicePool  pool{f.q, kind::device, 65536, 256};

    auto check = [&](shared_usm_ptr<Plain> h) {
        REQUIRE(h);
        const Plain got = slabkit_test::read_one(f.q, h.get());
        REQUIRE(got.a == 1);
        REQUIRE(got.b == 2);
    };

    check(make_shared_device<Plain>(f.q, Plain{1, 2}));
    check(make_shared_device<Plain>(f.q, f.dev, Plain{1, 2}));
    check(make_shared_device<Plain>(f.q, arena, Plain{1, 2}));
    check(make_shared_device<Plain>(f.q, pool, Plain{1, 2}));

    REQUIRE(make_shared_device_for_overwrite<Plain>(f.q));
    REQUIRE(make_shared_device_for_overwrite<Plain>(f.q, f.dev));
    REQUIRE(make_shared_device_for_overwrite<Plain>(f.q, arena));
    REQUIRE(make_shared_device_for_overwrite<Plain>(f.q, pool));
}

TEST_CASE("shared array forms work on every placement") {
    Fixture     f;
    DeviceArena arena{f.q, kind::device, 65536};
    DevicePool  pool{f.q, kind::device, 65536, 256};

    auto check = [&](shared_usm_ptr<int[]> h) {
        REQUIRE(h);
        const auto host = slabkit_test::read_back(f.q, h.get(), 16);
        for (std::size_t i = 0; i < 16; ++i) REQUIRE(host[i] == 4);
    };

    check(make_shared_device<int[]>(f.q, 16, 4));
    check(make_shared_device<int[]>(f.q, f.dev, 16, 4));
    check(make_shared_device<int[]>(f.q, arena, 16, 4));
    check(make_shared_device<int[]>(f.q, pool, 16, 4));

    std::array<int, 16> src{};
    src.fill(4);
    check(make_shared_device<int[]>(f.q, std::span<const int>{src}));

    auto zeroed = make_shared_device<int[]>(f.q, 8);
    REQUIRE(zeroed);
    for (int v : slabkit_test::read_back(f.q, zeroed.get(), 8)) REQUIRE(v == 0);

    REQUIRE(make_shared_device_for_overwrite<int[]>(f.q, 16));
    REQUIRE(make_shared_device_for_overwrite<int[]>(f.q, arena, 16));
    REQUIRE(make_shared_device_for_overwrite<int[]>(f.q, pool, 16));
}

TEST_CASE("a shared handle returns its nodes when the last owner drops") {
    Fixture    f;
    DevicePool pool{f.q, kind::device, 4096, 256};

    {
        auto a = make_shared_device<std::byte[]>(f.q, pool, 256);
        REQUIRE(pool.outstanding() == 1);
        {
            auto b = a;
            REQUIRE(pool.outstanding() == 1);
        }
        REQUIRE(pool.outstanding() == 1);
    }
    REQUIRE(pool.outstanding() == 0);
}

TEST_CASE("a shared allocation outlives its arena") {
    Fixture                  f;
    shared_usm_ptr<int[]> h;
    {
        DeviceArena arena{f.q, kind::device, 4096};
        h = make_shared_device<int[]>(f.q, arena, 8, 5);
    }
    const auto host = slabkit_test::read_back(f.q, h.get(), 8);
    for (std::size_t i = 0; i < 8; ++i) REQUIRE(host[i] == 5);
}
