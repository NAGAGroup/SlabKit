// The make family: every form, the two mechanisms, the type gate, and the
// promise that the object is ready when the call returns.

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

/// The type the device-side construction path exists for: not trivially
/// copyable, and self-referential, so bytes constructed on the host would
/// carry a host address into device memory. Constructing it on the device is
/// the only way to get it right.
struct SelfRef {
    int  values[4];
    int* cursor;

    explicit SelfRef(int base) {
        for (int i = 0; i < 4; ++i) values[i] = base + i;
        cursor = &values[0];
    }
    SelfRef(const SelfRef& other) {
        for (int i = 0; i < 4; ++i) values[i] = other.values[i];
        cursor = &values[0];
    }
    SelfRef& operator=(const SelfRef&) = delete;

    int sum() const { return cursor[0] + cursor[1] + cursor[2] + cursor[3]; }
};

struct Plain {
    int a;
    int b;
};

struct Polymorphic {
    int x = 0;
    virtual int f() const { return x; }
};

struct NonTriviallyDestructible {
    int  x = 0;
    ~NonTriviallyDestructible() {}
};

template <class T, class... Args>
concept can_make_object =
    requires(sycl::queue q, Args... args) { make_unique_device<T>(q, args...); };

template <class TArr>
concept can_make_array =
    requires(sycl::queue q, std::size_t n) { make_unique_device<TArr>(q, n); };

}  // namespace

// -----------------------------------------------------------------------------
// The type gate
// -----------------------------------------------------------------------------

TEST_CASE("the type gate accepts what it should") {
    STATIC_REQUIRE(can_make_object<int>);
    STATIC_REQUIRE(can_make_object<Plain>);
    STATIC_REQUIRE(can_make_object<SelfRef, int>);
    STATIC_REQUIRE(can_make_array<float[]>);
    STATIC_REQUIRE(can_make_array<std::byte[]>);
}

TEST_CASE("a polymorphic type is refused at compile time") {
    // AdaptiveCpp's generic/SSCP flow cannot resolve device vtables: the call
    // faults on the host backend and can return silently wrong results on
    // CUDA. The gate turns that into a compile-time rejection.
    STATIC_REQUIRE_FALSE(can_make_object<Polymorphic>);
    STATIC_REQUIRE_FALSE(can_make_array<Polymorphic[]>);
}

TEST_CASE("a type the handle could not destroy is refused") {
    STATIC_REQUIRE_FALSE(can_make_object<NonTriviallyDestructible>);
}

TEST_CASE("array forms refuse elements they cannot fill or copy") {
    // SelfRef is fine as a single object -- it is constructed on the device --
    // but array initialization goes through fill/copy, which needs trivially
    // copyable elements.
    STATIC_REQUIRE_FALSE(can_make_array<SelfRef[]>);
}

// -----------------------------------------------------------------------------
// Single objects
// -----------------------------------------------------------------------------

TEST_CASE("a trivially copyable object is constructed and ready on return") {
    Fixture f;

    auto h = make_unique_device<Plain>(f.q, Plain{3, 4});
    REQUIRE(h);
    // No wait() here on purpose: make_* blocks, so the data is already there.
    const Plain got = slabkit_test::read_one(f.q, h.get());
    REQUIRE(got.a == 3);
    REQUIRE(got.b == 4);
}

TEST_CASE("an object with no arguments is value-initialized") {
    Fixture f;

    auto h = make_unique_device<Plain>(f.q);
    REQUIRE(h);
    const Plain got = slabkit_test::read_one(f.q, h.get());
    REQUIRE(got.a == 0);
    REQUIRE(got.b == 0);
}

TEST_CASE("THE HEADLINE: a self-referential object is constructed ON the device") {
    Fixture f;

    auto h = make_unique_device<SelfRef>(f.q, 10);
    REQUIRE(h);

    // The constructor ran on the device, so `cursor` points into device memory
    // and dereferencing it in a kernel is valid. A host-constructed copy would
    // have planted a host address here.
    SelfRef* p   = h.get();
    auto     out = make_unique_device_for_overwrite<int[]>(f.q, 1);
    int*     o   = out.get();
    f.q.single_task([=]() { o[0] = p->sum(); }).wait();

    REQUIRE(slabkit_test::read_one(f.q, out.get()) == 10 + 11 + 12 + 13);
}

TEST_CASE("for_overwrite gives storage and nothing else") {
    Fixture f;
    auto    h = make_unique_device_for_overwrite<Plain>(f.q);
    REQUIRE(h);
}

TEST_CASE("single-object forms accept every placement") {
    Fixture     f;
    DeviceArena arena{f.q, kind::device, 65536};
    DevicePool  pool{f.q, kind::device, 65536, 256};

    auto check = [&](unique_usm_ptr<Plain> h) {
        REQUIRE(h);
        const Plain got = slabkit_test::read_one(f.q, h.get());
        REQUIRE(got.a == 5);
        REQUIRE(got.b == 6);
    };

    check(make_unique_device<Plain>(f.q, Plain{5, 6}));
    check(make_unique_device<Plain>(f.q, f.dev, Plain{5, 6}));
    check(make_unique_device<Plain>(f.q, arena, Plain{5, 6}));
    check(make_unique_device<Plain>(f.q, pool, Plain{5, 6}));

    REQUIRE(make_unique_device_for_overwrite<Plain>(f.q, f.dev));
    REQUIRE(make_unique_device_for_overwrite<Plain>(f.q, arena));
    REQUIRE(make_unique_device_for_overwrite<Plain>(f.q, pool));
}

// -----------------------------------------------------------------------------
// Arrays
// -----------------------------------------------------------------------------

TEST_CASE("an array form value-initializes") {
    Fixture f;

    auto h = make_unique_device<int[]>(f.q, 64);
    REQUIRE(h);
    const auto host = slabkit_test::read_back(f.q, h.get(), 64);
    for (std::size_t i = 0; i < 64; ++i) REQUIRE(host[i] == 0);
}

TEST_CASE("an array form fills with a value") {
    Fixture f;

    auto h = make_unique_device<int[]>(f.q, 64, 42);
    REQUIRE(h);
    const auto host = slabkit_test::read_back(f.q, h.get(), 64);
    for (std::size_t i = 0; i < 64; ++i) REQUIRE(host[i] == 42);
}

TEST_CASE("a fill pattern wider than one byte is replicated correctly") {
    Fixture f;

    // Not a uniform byte pattern, so this exercises the replication kernel
    // rather than the memset fast path.
    const Plain pattern{0x01020304, 0x05060708};
    auto        h = make_unique_device<Plain[]>(f.q, 16, pattern);
    REQUIRE(h);

    const auto host = slabkit_test::read_back(f.q, h.get(), 16);
    for (std::size_t i = 0; i < 16; ++i) {
        REQUIRE(host[i].a == pattern.a);
        REQUIRE(host[i].b == pattern.b);
    }
}

TEST_CASE("an array form copies from host data") {
    Fixture f;

    std::vector<int> src(32);
    for (std::size_t i = 0; i < src.size(); ++i) src[i] = static_cast<int>(i) * 3;

    auto h = make_unique_device<int[]>(f.q, std::span<const int>{src});
    REQUIRE(h);
    const auto host = slabkit_test::read_back(f.q, h.get(), src.size());
    REQUIRE(host == src);
}

TEST_CASE("array for_overwrite gives storage and nothing else") {
    Fixture f;
    auto    h = make_unique_device_for_overwrite<std::byte[]>(f.q, 1024);
    REQUIRE(h);
}

TEST_CASE("array forms accept every placement") {
    Fixture     f;
    DeviceArena arena{f.q, kind::device, 65536};
    DevicePool  pool{f.q, kind::device, 65536, 256};

    auto check = [&](unique_usm_ptr<int[]> h) {
        REQUIRE(h);
        const auto host = slabkit_test::read_back(f.q, h.get(), 16);
        for (std::size_t i = 0; i < 16; ++i) REQUIRE(host[i] == 9);
    };

    check(make_unique_device<int[]>(f.q, 16, 9));
    check(make_unique_device<int[]>(f.q, f.dev, 16, 9));
    check(make_unique_device<int[]>(f.q, arena, 16, 9));
    check(make_unique_device<int[]>(f.q, pool, 16, 9));

    REQUIRE(make_unique_device<int[]>(f.q, arena, 16));
    REQUIRE(make_unique_device<int[]>(f.q, pool, 16));
    REQUIRE(make_unique_device_for_overwrite<int[]>(f.q, arena, 16));
    REQUIRE(make_unique_device_for_overwrite<int[]>(f.q, pool, 16));

    std::array<int, 16> src{};
    src.fill(9);
    check(make_unique_device<int[]>(f.q, std::span<const int>{src}));
    check(make_unique_device<int[]>(f.q, arena, std::span<const int>{src}));
    check(make_unique_device<int[]>(f.q, pool, std::span<const int>{src}));
}

TEST_CASE("cospan's entry-storage shape works") {
    Fixture     f;
    DeviceArena arena{f.q, kind::device, 65536};

    // The exact expression the consumer builds an entry from.
    unique_usm_ptr<std::byte[]> storage =
        make_unique_device_for_overwrite<std::byte[]>(f.q, arena, 4096);
    REQUIRE(storage);

    // And the dynamic chain's fallback shape: same handle type, different route.
    unique_usm_ptr<std::byte[]> fallback = sk::malloc_device<std::byte>(4096, f.dev, f.ctx);
    REQUIRE(fallback);

    storage = std::move(fallback);  // interchangeable, as the contract requires
    REQUIRE(storage);
}

TEST_CASE("make throws when storage cannot be obtained") {
    Fixture     f;
    DeviceArena tiny{f.q, kind::device, 64};

    REQUIRE_THROWS_AS(make_unique_device<int[]>(f.q, tiny, 1024), std::bad_alloc);
}
