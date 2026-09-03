// Handles: which deleter an allocation carries, and the pinning contract that
// the pool and arena deleters exist to enforce.

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <memory>
#include <type_traits>
#include <variant>

#include "naga/slabkit/slabkit.hpp"
#include "support.hpp"

using namespace naga::slabkit;
namespace sk = naga::slabkit;
using slabkit_test::Fixture;

namespace {
// Index of each alternative in UsmDeleter::how.
constexpr std::size_t kOs    = 0;
constexpr std::size_t kPool  = 1;
constexpr std::size_t kArena = 2;
}  // namespace

TEST_CASE("handle aliases are the documented types") {
    STATIC_REQUIRE(std::is_same_v<unique_usm_ptr<int>, std::unique_ptr<int, UsmDeleter>>);
    STATIC_REQUIRE(std::is_same_v<shared_usm_ptr<int>, std::shared_ptr<int>>);
}

TEST_CASE("each placement routes to its own deleter") {
    Fixture f;

    SECTION("queue placement frees through the runtime") {
        auto h = sk::malloc_device<int>(16, f.q);
        REQUIRE(h);
        REQUIRE(h.get_deleter().how.index() == kOs);
    }

    SECTION("device+context placement frees through the runtime") {
        auto h = sk::malloc_device<int>(16, f.dev, f.ctx);
        REQUIRE(h);
        REQUIRE(h.get_deleter().how.index() == kOs);
    }

    SECTION("arena placement accounts, it does not free") {
        DeviceArena arena{f.q, kind::device, 4096};
        auto        h = sk::malloc_device<int>(16, arena);
        REQUIRE(h);
        REQUIRE(h.get_deleter().how.index() == kArena);
    }

    SECTION("pool placement returns nodes") {
        DevicePool pool{f.q, kind::device, 4096, 256};
        auto       h = sk::malloc_device<int>(16, pool);
        REQUIRE(h);
        REQUIRE(h.get_deleter().how.index() == kPool);
        REQUIRE(std::get<PoolFree>(h.get_deleter().how).nodes == 1);
    }
}

TEST_CASE("the context an OsFree holds is the one that was passed") {
    Fixture f;
    auto    h = sk::malloc_device<int>(4, f.dev, f.ctx);
    REQUIRE(std::get<OsFree>(h.get_deleter().how).ctx == f.ctx);
}

TEST_CASE("an allocation outliving its arena keeps the slab alive") {
    Fixture              f;
    unique_usm_ptr<int[]> h;

    {
        DeviceArena arena{f.q, kind::device, 4096};
        h = sk::malloc_device<int>(8, arena);
        REQUIRE(h);
        // Every DeviceArena handle goes away here; only the allocation is left
        // holding the implementation.
    }

    int* p = h.get();
    f.q.single_task([=]() {
         for (int i = 0; i < 8; ++i) p[i] = i * 3;
     }).wait();

    const auto host = slabkit_test::read_back(f.q, h.get(), 8);
    for (int i = 0; i < 8; ++i) REQUIRE(host[static_cast<std::size_t>(i)] == i * 3);
}

TEST_CASE("an allocation outliving its pool keeps the slab alive") {
    Fixture                  f;
    unique_usm_ptr<int[]> h;

    {
        DevicePool pool{f.q, kind::device, 4096, 256};
        h = sk::malloc_device<int>(8, pool);
        REQUIRE(h);
    }

    int* p = h.get();
    f.q.single_task([=]() {
         for (int i = 0; i < 8; ++i) p[i] = i + 1;
     }).wait();

    const auto host = slabkit_test::read_back(f.q, h.get(), 8);
    for (int i = 0; i < 8; ++i) REQUIRE(host[static_cast<std::size_t>(i)] == i + 1);
}

TEST_CASE("releasing an allocation returns its nodes to a live pool") {
    Fixture    f;
    DevicePool pool{f.q, kind::device, 4096, 256};

    const std::size_t before = pool.free_count();
    {
        auto h = sk::malloc_device<std::byte>(512, pool);
        REQUIRE(h);
        REQUIRE(pool.outstanding() == 2);
        REQUIRE(pool.free_count() == before - 2);
    }
    REQUIRE(pool.free_count() == before);
    REQUIRE(pool.outstanding() == 0);
}

TEST_CASE("a unique handle converts to a shared handle, deleter and all") {
    Fixture    f;
    DevicePool pool{f.q, kind::device, 4096, 256};

    {
        shared_usm_ptr<std::byte[]> s{sk::malloc_device<std::byte>(256, pool)};
        REQUIRE(s);
        REQUIRE(pool.outstanding() == 1);

        auto copy = s;  // shared ownership
        s.reset();
        REQUIRE(pool.outstanding() == 1);  // the copy still holds it
    }
    REQUIRE(pool.outstanding() == 0);
}
