// =============================================================================
// test_cross_device.cpp — placing work and memory on a device that is not the
// stream's default one
// =============================================================================
//
// v1 could not do this through the retarget extension and used a temporary
// sycl::queue built on the allocation's own context instead: AdaptiveCpp
// resolves a memcpy's operands by querying each pointer against the QUEUE's
// context, so a pointer belonging to another context was misread as host
// memory.
//
// That mechanism is gone. The stream now holds one retargetable queue pair PER
// BACKEND, each on that backend's own default context, and a placement selects
// the pair before it retargets within it -- so the pointer is always queried
// against the context that owns it, whether the placement crosses a device or a
// whole backend.
//
// These tests run over EVERY device the machine enumerates, so on a one-device
// box they still assert the primary path and on this one they cross a backend.

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <span>
#include <vector>

#include "naga/slabkit/slabkit.hpp"
#include "support.hpp"

using namespace naga::slabkit;
namespace sk = naga::slabkit;

namespace {

/// A stream over every device the machine has.
struct AllDevices {
    context ctx;
    queue   q;
    AllDevices() : ctx{slabkit_test::all_devices()}, q{ctx} {}
};

}  // namespace

TEST_CASE("storage is allocated and freed with the placed device's own context",
          "[cross-device]") {
    AllDevices all;

    for (const auto& d : all.ctx.devices()) {
        auto h = sk::malloc_device<int>(64, all.q, d);
        REQUIRE(h);
        REQUIRE(std::get<OsFree>(h.get_deleter().how).ctx == all.ctx.context_of(d));
    }
}

TEST_CASE("make initializes correctly on every device of the context", "[cross-device]") {
    AllDevices all;

    for (const auto& d : all.ctx.devices()) {
        SECTION("fill form") {
            auto h = make_unique_device<int[]>(all.q, d, 32, 77);
            REQUIRE(h);
            for (int v : slabkit_test::read_back(all.q, d, h.get(), 32)) REQUIRE(v == 77);
        }

        SECTION("copy form -- the one the old retarget could not do") {
            std::vector<int> src(32);
            for (std::size_t i = 0; i < src.size(); ++i) src[i] = static_cast<int>(i) + 100;

            auto h = make_unique_device<int[]>(all.q, d, std::span<const int>{src});
            REQUIRE(h);
            REQUIRE(slabkit_test::read_back(all.q, d, h.get(), src.size()) == src);
        }

        SECTION("value-initialized form") {
            auto h = make_unique_device<int[]>(all.q, d, 16);
            REQUIRE(h);
            for (int v : slabkit_test::read_back(all.q, d, h.get(), 16)) REQUIRE(v == 0);
        }
    }
}

TEST_CASE("the interop device+context placement names the same memory", "[cross-device]") {
    AllDevices all;

    for (const auto& d : all.ctx.devices()) {
        // The context must be that device's backend default -- which is exactly
        // what the slabkit context hands out. This placement names no stream,
        // so its initialization runs on a sycl::queue built from the pair: the
        // one part of v1's mechanism that has to survive, because an interop
        // placement has no slabkit queue to retarget.
        auto h = make_unique_device<int[]>(d, all.ctx.context_of(d), 8, 5);
        REQUIRE(h);
        for (int v : slabkit_test::read_back(all.q, d, h.get(), 8)) REQUIRE(v == 5);
    }
}

TEST_CASE("an arena on any device initializes through that device", "[cross-device]") {
    AllDevices all;

    for (const auto& d : all.ctx.devices()) {
        DeviceArena arena{all.q, d, kind::device, 65536};
        REQUIRE(arena.device() == d);
        REQUIRE(arena.get_kind() == kind::device);

        auto h = make_unique_device<int[]>(all.q, arena, 24, 13);
        REQUIRE(h);
        for (int v : slabkit_test::read_back(all.q, d, h.get(), 24)) REQUIRE(v == 13);
    }
}

TEST_CASE("a pool on any device initializes through that device", "[cross-device]") {
    AllDevices all;

    for (const auto& d : all.ctx.devices()) {
        DevicePool pool{all.q, d, kind::device, 65536, 256};
        auto       h = make_unique_device<int[]>(all.q, pool, 24, 21);
        REQUIRE(h);
        for (int v : slabkit_test::read_back(all.q, d, h.get(), 24)) REQUIRE(v == 21);
    }
}

TEST_CASE("work placed on each device runs there, in one order", "[cross-device]") {
    AllDevices all;

    // One allocation per device, written by a kernel placed on that device.
    std::vector<unique_usm_ptr<int[]>> handles;
    for (const auto& d : all.ctx.devices()) {
        auto h = sk::malloc_device<int>(4, all.q, d);
        REQUIRE(h);
        int* p = h.get();
        all.q.parallel_for(d, sycl::range<1>{4},
                           [=](sycl::id<1> i) { p[i[0]] = static_cast<int>(i[0]) + 1; });
        handles.push_back(std::move(h));
    }
    all.q.wait();

    for (std::size_t i = 0; i < handles.size(); ++i) {
        const auto host =
            slabkit_test::read_back(all.q, all.ctx.devices()[i], handles[i].get(), 4);
        REQUIRE(host == std::vector<int>{1, 2, 3, 4});
    }
}
