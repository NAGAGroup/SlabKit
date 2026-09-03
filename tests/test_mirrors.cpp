// =============================================================================
// test_mirrors.cpp — the four routed mirrors
// =============================================================================
//
// Every route is exercised on every machine: the route a pair of operands would
// CHOOSE depends on the topology, so the suite also forces each one by name
// through detail::copy_bytes_via. A one-backend box would otherwise never see
// the staged path, which is exactly the path a two-backend box depends on.

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <vector>

#include "naga/slabkit/slabkit.hpp"
#include "support.hpp"

using namespace naga::slabkit;
namespace sk = naga::slabkit;
using slabkit_test::Fixture;

TEST_CASE("a copy between two device allocations on one device takes the direct route",
          "[mirrors]") {
    Fixture f;

    auto src = make_unique_device<int[]>(f.q, 8, 4);
    auto dst = sk::malloc_device<int>(8, f.q);

    // The route follows the KIND the runtime reports, not the name the
    // allocation was made with -- and on the OMP backend AdaptiveCpp reports
    // `host` for everything, device allocations included (see test_context).
    // There a host memcpy IS the direct route: the memory is host memory.
    const route::via expected =
        f.dev.get_backend() == sycl::backend::omp ? route::via::host_memcpy : route::via::d2d;
    REQUIRE(detail::chosen_route(f.q, dst.get(), src.get()).method == expected);

    sk::copy(f.q, src.get(), dst.get(), 8);
    f.q.wait();
    for (int v : slabkit_test::read_back(f.q, dst.get(), 8)) REQUIRE(v == 4);
}

TEST_CASE("every route moves the right bytes", "[mirrors]") {
    Fixture f;

    // Shared on both ends, so all three routes are legal for the same pair and
    // the results are directly comparable.
    auto src = make_unique_shared<int[]>(f.q, 16);
    auto dst = make_unique_shared<int[]>(f.q, 16);

    for (std::size_t i = 0; i < 16; ++i) src[i] = static_cast<int>(i) + 1;

    for (auto method : {route::via::d2d, route::via::host_memcpy, route::via::staged}) {
        for (std::size_t i = 0; i < 16; ++i) dst[i] = 0;
        f.q.wait();

        detail::copy_bytes_via(f.q, dst.get(), src.get(), 16 * sizeof(int), method, {});
        f.q.wait();

        for (std::size_t i = 0; i < 16; ++i) REQUIRE(dst[i] == static_cast<int>(i) + 1);
    }
}

TEST_CASE("the host route does not overlap a running kernel", "[mirrors]") {
    Fixture f;

    auto src = make_unique_shared<int[]>(f.q, 4, 0);
    auto dst = make_unique_shared<int[]>(f.q, 4, 0);
    f.q.wait();

    int* p = src.get();
    f.q.parallel_for(sycl::range<1>{4}, [=](sycl::id<1> i) {
        for (int spin = 0; spin < 2000000; ++spin) p[i[0]] = p[i[0]] + 1;
    });

    // The host copy must see the finished kernel, not a value part-way through.
    detail::copy_bytes_via(f.q, dst.get(), src.get(), 4 * sizeof(int),
                           route::via::host_memcpy, {});
    f.q.wait();
    for (std::size_t i = 0; i < 4; ++i) REQUIRE(dst[i] == 2000000);
}

TEST_CASE("a copy from ordinary host memory is a legal operand", "[mirrors]") {
    Fixture f;

    std::vector<int> host{5, 6, 7, 8};
    auto             dst = sk::malloc_device<int>(4, f.q);

    sk::copy(f.q, host.data(), dst.get(), host.size());
    f.q.wait();
    REQUIRE(slabkit_test::read_back(f.q, dst.get(), 4) == host);
}

TEST_CASE("a copy across backends has only the staged route", "[mirrors]") {
    context ctx{slabkit_test::all_devices()};
    queue   q{ctx};

    if (ctx.backends().size() < 2) return;  // one backend here; nothing to cross

    const sycl::device a = ctx.devices().front();
    sycl::device       b = a;
    for (const auto& d : ctx.devices())
        if (!(d.get_platform() == a.get_platform())) b = d;

    auto src = make_unique_device<int[]>(q, a, 8, 3);
    auto dst = sk::malloc_device<int>(8, q, b);

    REQUIRE(detail::chosen_route(q, dst.get(), src.get()).method == route::via::staged);

    sk::copy(q, src.get(), dst.get(), 8);
    q.wait();
    for (int v : slabkit_test::read_back(q, b, dst.get(), 8)) REQUIRE(v == 3);
}

TEST_CASE("memset and fill run on the operand's own device", "[mirrors]") {
    Fixture f;

    auto p = sk::malloc_device<int>(8, f.q);

    sk::memset(f.q, p.get(), 0, 8 * sizeof(int));
    f.q.wait();
    for (int v : slabkit_test::read_back(f.q, p.get(), 8)) REQUIRE(v == 0);

    sk::fill(f.q, p.get(), 12345, 8);
    f.q.wait();
    for (int v : slabkit_test::read_back(f.q, p.get(), 8)) REQUIRE(v == 12345);
}

TEST_CASE("a fill pattern wider than a register is staged", "[mirrors]") {
    struct Wide {
        int a, b, c, d, e;
    };
    STATIC_REQUIRE(sizeof(Wide) > sizeof(std::uint64_t));

    Fixture f;
    auto    p = sk::malloc_device<Wide>(6, f.q);

    const Wide pattern{1, 2, 3, 4, 5};
    sk::fill(f.q, p.get(), pattern, 6);
    f.q.wait();

    for (const Wide& w : slabkit_test::read_back(f.q, p.get(), 6)) {
        REQUIRE(w.a == 1);
        REQUIRE(w.e == 5);
    }
}

TEST_CASE("a mirror composes with explicit dependencies", "[mirrors]") {
    Fixture f;

    auto        a     = make_unique_device<int[]>(f.q, 4, 2);
    auto        b     = sk::malloc_device<int>(4, f.q);
    sycl::event first = f.q.single_task([p = a.get()] { p[0] = 99; });

    sk::copy(f.q, a.get(), b.get(), 4, std::vector<sycl::event>{first});
    f.q.wait();

    const auto host = slabkit_test::read_back(f.q, b.get(), 4);
    REQUIRE(host[0] == 99);
    REQUIRE(host[1] == 2);
}

TEST_CASE("a zero-byte mirror is a no-op item, not an error", "[mirrors]") {
    Fixture f;
    auto    p = sk::malloc_device<int>(4, f.q);

    REQUIRE_NOTHROW(sk::memcpy(f.q, p.get(), p.get(), 0));
    REQUIRE_NOTHROW(sk::memset(f.q, p.get(), 0, 0));
    REQUIRE_NOTHROW(sk::fill(f.q, p.get(), 1, 0));
    f.q.wait();
}
