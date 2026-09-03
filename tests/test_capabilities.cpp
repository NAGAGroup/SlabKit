// =============================================================================
// test_capabilities.cpp — aspects, answers and routes
// =============================================================================

#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <sycl/sycl.hpp>

#include "naga/slabkit/capabilities.hpp"
#include "naga/slabkit/context.hpp"
#include "support.hpp"

using namespace naga::slabkit;

TEST_CASE("forwarded aspects mean what sycl means by them", "[capabilities]") {
    for (const auto& d : slabkit_test::all_devices()) {
        REQUIRE((context::satisfies<aspect::cpu>(d) == answer::yes) == d.has(sycl::aspect::cpu));
        REQUIRE((context::satisfies<aspect::gpu>(d) == answer::yes) == d.has(sycl::aspect::gpu));
        REQUIRE((context::satisfies<aspect::fp64>(d) == answer::yes) == d.has(sycl::aspect::fp64));
    }
}

TEST_CASE("the honest trio answers for every device", "[capabilities]") {
    for (const auto& d : slabkit_test::all_devices()) {
        // Every backend AdaptiveCpp ships supports explicit device USM, and the
        // trio must commit to an answer rather than shrug.
        REQUIRE(context::satisfies<aspect::has_device>(d) == answer::yes);
        REQUIRE(context::satisfies<aspect::has_shared>(d) != answer::unknown);
        REQUIRE(context::satisfies<aspect::has_host>(d) != answer::unknown);
    }
}

TEST_CASE("a device paired with itself satisfies every pair aspect", "[capabilities]") {
    const sycl::device dev{sycl::default_selector_v};
    REQUIRE(context::satisfies<aspect::shares_context>(dev, dev) == answer::yes);
    REQUIRE(context::satisfies<aspect::d2d>(dev, dev) == answer::yes);
    REQUIRE(context::satisfies<aspect::device_p2p>(dev, dev) == answer::yes);
    REQUIRE(context::satisfies<aspect::shared_p2p>(dev, dev) == answer::yes);
}

TEST_CASE("pair aspects across backends are no, never unknown", "[capabilities]") {
    const auto devices = slabkit_test::all_devices();

    bool found_cross_backend_pair = false;
    for (std::size_t i = 0; i < devices.size(); ++i) {
        for (std::size_t j = 0; j < devices.size(); ++j) {
            if (devices[i].get_platform() == devices[j].get_platform()) continue;
            found_cross_backend_pair = true;

            REQUIRE(context::satisfies<aspect::shares_context>(devices[i], devices[j]) ==
                    answer::no);
            REQUIRE(context::satisfies<aspect::d2d>(devices[i], devices[j]) == answer::no);
            REQUIRE(context::satisfies<aspect::device_p2p>(devices[i], devices[j]) == answer::no);
            REQUIRE(context::satisfies<aspect::shared_p2p>(devices[i], devices[j]) == answer::no);
        }
    }

    // On a machine with one backend there is nothing to assert; say so rather
    // than reporting a vacuous pass as a measurement.
    if (!found_cross_backend_pair)
        WARN("only one backend is enumerated here, so no cross-backend pair was exercised");
}

TEST_CASE("copy_routes is never empty, best first, staged last", "[capabilities]") {
    const sycl::device dev{sycl::default_selector_v};

    for (kind src : {kind::device, kind::shared, kind::host}) {
        for (kind dst : {kind::device, kind::shared, kind::host}) {
            const auto routes = context::copy_routes(src, dev, dst, dev);
            REQUIRE_FALSE(routes.empty());
            REQUIRE(routes.back().method == route::via::staged);

            // A host operand cannot be reached by a direct backend copy.
            if (src == kind::host || dst == kind::host)
                for (const auto& r : routes) REQUIRE(r.method != route::via::d2d);

            // Two host-accessible operands can always be copied by host code.
            if (host_accessible(src) && host_accessible(dst)) {
                bool has_host_route = false;
                for (const auto& r : routes)
                    has_host_route = has_host_route || r.method == route::via::host_memcpy;
                REQUIRE(has_host_route);
            }
        }
    }

    // Same device, both operands device-resident: the direct route is first.
    const auto d2d_routes = context::copy_routes(kind::device, dev, kind::device, dev);
    REQUIRE(d2d_routes.front().method == route::via::d2d);
}

TEST_CASE("copy_routes across backends falls back to a host route", "[capabilities]") {
    const sycl::device dev{sycl::default_selector_v};
    sycl::device       other;
    if (!slabkit_test::second_device(dev, other)) return;
    if (dev.get_platform() == other.get_platform()) return;

    const auto routes = context::copy_routes(kind::device, dev, kind::device, other);
    REQUIRE_FALSE(routes.empty());
    REQUIRE(routes.size() == 1);
    REQUIRE(routes.front().method == route::via::staged);
}

TEST_CASE("asking a device aspect of a pair, or the reverse, is a usage error",
          "[capabilities]") {
    const sycl::device dev{sycl::default_selector_v};

    // The template overloads are static_assert-gated, so the runtime core is
    // what carries the check for a dynamically chosen aspect.
    REQUIRE_THROWS_AS(naga::slabkit::detail::satisfies_device(aspect::d2d, dev),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(naga::slabkit::detail::satisfies_pair(aspect::fp64, dev, dev),
                      std::invalid_argument);
}
