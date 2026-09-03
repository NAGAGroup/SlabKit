// =============================================================================
// test_kinds.cpp — the three USM kinds across the placements
// =============================================================================
//
// v1 had one kind and four placements. What is new here is that the NAME of the
// call is the kind, that a memory object carries its kind and refuses a call
// that does not match it, and that the host kind's placement has no device in
// it -- checked as a constraint, so the rejection is observable rather than a
// wall of instantiation errors.

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include "naga/slabkit/slabkit.hpp"
#include "support.hpp"

using namespace naga::slabkit;
namespace sk = naga::slabkit;
using slabkit_test::Fixture;

namespace {

/// Is this call spelled at all? A concept, so a rejection is a false rather
/// than a compile error.
template <class... A>
concept can_malloc_host = requires(std::size_t n, A... a) { sk::malloc_host<int>(n, a...); };

template <class... A>
concept can_make_host = requires(A... a) { make_unique_host<int>(a...); };

}  // namespace

TEST_CASE("the host kind takes no device, and saying so does not compile", "[kinds]") {
    STATIC_REQUIRE(can_malloc_host<queue&>);
    STATIC_REQUIRE(can_malloc_host<sycl::queue&>);
    STATIC_REQUIRE(can_malloc_host<sycl::context&>);
    STATIC_REQUIRE_FALSE(can_malloc_host<queue&, sycl::device&>);
    STATIC_REQUIRE_FALSE(can_malloc_host<sycl::device&, sycl::context&>);

    STATIC_REQUIRE(can_make_host<queue&>);
    STATIC_REQUIRE_FALSE(can_make_host<queue&, sycl::device&>);
}

TEST_CASE("the device kind reaches every placement", "[kinds]") {
    Fixture     f;
    DeviceArena arena{f.q, kind::device, 65536};
    DevicePool  pool{f.q, kind::device, 65536, 256};

    REQUIRE(sk::malloc_device<int>(8, f.q, f.dev));
    REQUIRE(sk::malloc_device<int>(8, f.q));
    REQUIRE(sk::malloc_device<int>(8, f.sq));
    REQUIRE(sk::malloc_device<int>(8, f.dev, f.ctx));
    REQUIRE(sk::malloc_device<int>(8, arena));
    REQUIRE(sk::malloc_device<int>(8, pool));
}

TEST_CASE("the shared kind reaches every placement and is host-readable", "[kinds]") {
    Fixture     f;
    DeviceArena arena{f.q, kind::shared, 65536};
    DevicePool  pool{f.q, kind::shared, 65536, 256};

    auto check = [](unique_usm_ptr<int[]> h) {
        REQUIRE(h);
        h[0] = 42;  // shared memory is host-writable; nothing is in flight over it
        REQUIRE(h[0] == 42);
    };

    check(sk::malloc_shared<int>(8, f.q, f.dev));
    check(sk::malloc_shared<int>(8, f.q));
    check(sk::malloc_shared<int>(8, f.sq));
    check(sk::malloc_shared<int>(8, f.dev, f.ctx));
    check(sk::malloc_shared<int>(8, arena));
    check(sk::malloc_shared<int>(8, pool));
}

TEST_CASE("the host kind reaches its placements and is host-readable", "[kinds]") {
    Fixture     f;
    DeviceArena arena{f.q, kind::host, 65536};
    DevicePool  pool{f.q, kind::host, 65536, 256};

    auto check = [](unique_usm_ptr<int[]> h) {
        REQUIRE(h);
        h[0] = 7;
        REQUIRE(h[0] == 7);
    };

    check(sk::malloc_host<int>(8, f.q));
    check(sk::malloc_host<int>(8, f.sq));
    check(sk::malloc_host<int>(8, f.ctx));
    check(sk::malloc_host<int>(8, arena));
    check(sk::malloc_host<int>(8, pool));
}

TEST_CASE("a memory object carries its kind, and the call must match it", "[kinds]") {
    Fixture     f;
    DeviceArena device_arena{f.q, kind::device, 4096};
    DeviceArena shared_arena{f.q, kind::shared, 4096};
    DevicePool  host_pool{f.q, kind::host, 4096, 256};

    REQUIRE(device_arena.get_kind() == kind::device);
    REQUIRE(shared_arena.get_kind() == kind::shared);
    REQUIRE(host_pool.get_kind() == kind::host);

    REQUIRE_THROWS_AS(sk::malloc_shared<int>(4, device_arena), std::invalid_argument);
    REQUIRE_THROWS_AS(sk::malloc_device<int>(4, shared_arena), std::invalid_argument);
    REQUIRE_THROWS_AS(sk::malloc_device<int>(4, host_pool), std::invalid_argument);
    REQUIRE_THROWS_AS(make_unique_device<int>(f.q, host_pool, 1), std::invalid_argument);

    // Matching names are fine.
    REQUIRE(sk::malloc_device<int>(4, device_arena));
    REQUIRE(sk::malloc_shared<int>(4, shared_arena));
    REQUIRE(sk::malloc_host<int>(4, host_pool));
}

TEST_CASE("make_* over the shared and host kinds writes on the host", "[kinds]") {
    Fixture f;

    SECTION("single object") {
        auto s = make_unique_shared<int>(f.q, 11);
        REQUIRE(*s == 11);

        auto h = make_unique_host<int>(f.q, 12);
        REQUIRE(*h == 12);
    }

    SECTION("value-initialized array") {
        auto s = make_unique_shared<int[]>(f.q, 6);
        for (std::size_t i = 0; i < 6; ++i) REQUIRE(s[i] == 0);
    }

    SECTION("filled array") {
        auto h = make_unique_host<int[]>(f.q, 6, 9);
        for (std::size_t i = 0; i < 6; ++i) REQUIRE(h[i] == 9);
    }

    SECTION("array copied from host data") {
        std::vector<int> src{1, 2, 3, 4};
        auto             s = make_unique_shared<int[]>(f.q, std::span<const int>{src});
        for (std::size_t i = 0; i < src.size(); ++i) REQUIRE(s[i] == src[i]);
    }

    SECTION("for_overwrite gives storage and nothing else") {
        REQUIRE(make_unique_shared_for_overwrite<int[]>(f.q, 4));
        REQUIRE(make_unique_host_for_overwrite<int>(f.q));
    }
}

TEST_CASE("shared memory made on the host is visible to a kernel", "[kinds]") {
    Fixture f;

    auto h = make_unique_shared<int[]>(f.q, 16, 3);
    int*  p = h.get();
    f.q.parallel_for(sycl::range<1>{16}, [=](sycl::id<1> i) { p[i[0]] = p[i[0]] * 2; });
    f.q.wait();

    for (std::size_t i = 0; i < 16; ++i) REQUIRE(h[i] == 6);
}

TEST_CASE("host memory made on the host is visible to a kernel", "[kinds]") {
    Fixture f;

    auto h = make_unique_host<int[]>(f.q, 16, 5);
    int*  p = h.get();
    f.q.parallel_for(sycl::range<1>{16}, [=](sycl::id<1> i) { p[i[0]] = p[i[0]] + 1; });
    f.q.wait();

    for (std::size_t i = 0; i < 16; ++i) REQUIRE(h[i] == 6);
}

TEST_CASE("the shared handle families cover all three kinds", "[kinds]") {
    Fixture f;

    shared_usm_ptr<int> d = make_shared_device<int>(f.q, 1);
    REQUIRE(d);
    REQUIRE(slabkit_test::read_one(f.q, d.get()) == 1);

    shared_usm_ptr<int> s = make_shared_shared<int>(f.q, 2);
    REQUIRE(*s == 2);

    shared_usm_ptr<int> h = make_shared_host<int>(f.q, 3);
    REQUIRE(*h == 3);
}

TEST_CASE("a pool of any kind still honours its node alignment", "[kinds]") {
    Fixture f;

    for (kind k : {kind::device, kind::shared, kind::host}) {
        DevicePool pool{f.q, k, 65536, 512};
        REQUIRE(pool.node_size() == 512);

        auto h = (k == kind::device)   ? sk::malloc_device<std::byte>(100, pool)
                 : (k == kind::shared) ? sk::malloc_shared<std::byte>(100, pool)
                                       : sk::malloc_host<std::byte>(100, pool);
        REQUIRE(h);
        REQUIRE(reinterpret_cast<std::uintptr_t>(h.get()) % 512 == 0);
    }
}
