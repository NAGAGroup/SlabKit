// =============================================================================
// test_context.cpp — the device set, its contexts, and selection
// =============================================================================
//
// The load-bearing claim of this file is the first test: the context slabkit
// hands out for a device is the SAME context sycl::queue{device} would build.
// If that ever stops being true, every allocation slabkit makes becomes a
// cross-context free waiting to terminate the process, and no other test in
// the suite would notice.

#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <sycl/sycl.hpp>

#include "naga/slabkit/context.hpp"
#include "support.hpp"

using namespace naga::slabkit;

TEST_CASE("context_of is the platform default, identical to queue{device}'s", "[context]") {
    const sycl::device dev{sycl::default_selector_v};
    const context      ctx{dev};

    REQUIRE(ctx.context_of(dev) == sycl::queue{dev}.get_context());
    REQUIRE(ctx.context_of(ctx.backend_of(dev)) == sycl::queue{dev}.get_context());
}

TEST_CASE("every device of the context reports its own backend's context", "[context]") {
    const context ctx{slabkit_test::all_devices()};

    for (const auto& d : ctx.devices()) {
        REQUIRE(ctx.contains(d));
        REQUIRE(ctx.context_of(d) == sycl::queue{d}.get_context());
        REQUIRE(ctx.backend_of(d) == d.get_platform());
    }

    // One context per backend, deduplicated.
    REQUIRE(ctx.backends().size() <= ctx.devices().size());
    for (std::size_t i = 0; i < ctx.backends().size(); ++i)
        for (std::size_t j = i + 1; j < ctx.backends().size(); ++j)
            REQUIRE_FALSE(ctx.backends()[i] == ctx.backends()[j]);
}

TEST_CASE("the default context covers every root device SYCL reports", "[context]") {
    std::vector<sycl::device> expected;
    for (const auto& p : sycl::platform::get_platforms())
        for (const auto& d : p.get_devices()) expected.push_back(d);

    const context ctx{};

    REQUIRE(ctx.devices().size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) REQUIRE(ctx.devices()[i] == expected[i]);  // platform order, then device order

    for (const auto& d : expected) {
        REQUIRE(ctx.contains(d));
        REQUIRE(ctx.context_of(d) == sycl::queue{d}.get_context());
    }
}

TEST_CASE("a device outside the set is rejected everywhere", "[context]") {
    const sycl::device dev{sycl::default_selector_v};
    sycl::device       other;
    if (!slabkit_test::second_device(dev, other)) return;  // single-device machine

    const context ctx{dev};
    REQUIRE_FALSE(ctx.contains(other));
    REQUIRE_THROWS_AS(ctx.context_of(other), std::invalid_argument);
    REQUIRE_THROWS_AS(ctx.backend_of(other), std::invalid_argument);
}

TEST_CASE("a platform selects its root devices", "[context]") {
    const sycl::platform plat = sycl::device{sycl::default_selector_v}.get_platform();
    const context        ctx{plat};

    REQUIRE(ctx.devices().size() == plat.get_devices().size());
    REQUIRE(ctx.backends().size() == 1);
    REQUIRE(ctx.backends()[0] == plat);
}

TEST_CASE("the same device listed twice is a usage error", "[context]") {
    const sycl::device dev{sycl::default_selector_v};
    REQUIRE_THROWS_AS((context{std::vector<sycl::device>{dev, dev}}), std::invalid_argument);
}

TEST_CASE("an empty device list is a usage error", "[context]") {
    REQUIRE_THROWS_AS((context{std::vector<sycl::device>{}}), std::invalid_argument);
}

TEST_CASE("aspect selection filters, and an empty result throws", "[context]") {
    // Every machine AdaptiveCpp runs on has an OMP host device, so this
    // selection is never empty.
    const context cpus{aspect::cpu};
    REQUIRE(cpus.devices().size() >= 1);
    for (const auto& d : cpus.devices()) REQUIRE(d.is_cpu());

    // No device is both a CPU and a GPU.
    REQUIRE_THROWS_AS((context{aspect::cpu, aspect::gpu}), std::invalid_argument);
}

TEST_CASE("backend identity is an aspect like any other", "[context]") {
    const context omp{aspect::backend_omp};
    for (const auto& d : omp.devices()) REQUIRE(d.get_backend() == sycl::backend::omp);

    REQUIRE(context::satisfies<aspect::backend_omp>(omp.devices().front()) == answer::yes);
    REQUIRE(context::satisfies<aspect::backend_cuda>(omp.devices().front()) == answer::no);
}

TEST_CASE("selection within a platform stays within it", "[context]") {
    const sycl::platform plat = sycl::device{sycl::default_selector_v}.get_platform();
    const context        ctx{plat, {aspect::has_device}};

    for (const auto& d : ctx.devices()) REQUIRE(d.get_platform() == plat);
}

TEST_CASE("select_devices reads the cached device answers", "[context]") {
    const context ctx{slabkit_test::all_devices()};

    const auto cpus = ctx.select_devices<aspect::cpu>();
    for (const auto& d : cpus) REQUIRE(d.is_cpu());

    std::size_t expected = 0;
    for (const auto& d : ctx.devices())
        if (context::satisfies<aspect::cpu>(d) == answer::yes) ++expected;
    REQUIRE(cpus.size() == expected);
}

TEST_CASE("select_devices on a pair aspect groups by the relation", "[context]") {
    const context ctx{slabkit_test::all_devices()};
    const auto    groups = ctx.select_devices<aspect::shares_context>();

    // Every device appears in exactly one group, and no group spans backends.
    std::size_t total = 0;
    for (const auto& [plat, members] : groups) {
        total += members.size();
        REQUIRE_FALSE(members.empty());
        for (const auto& d : members) REQUIRE(d.get_platform() == plat);
    }
    REQUIRE(total == ctx.devices().size());

    // Devices of one backend share a context, so they are one group.
    REQUIRE(groups.size() == ctx.backends().size());
}

TEST_CASE("the pointer queries answer for allocations of this context", "[context]") {
    const sycl::device dev{sycl::default_selector_v};
    const context      ctx{dev};

    void* p = sycl::malloc_device(64, dev, ctx.context_of(dev));
    REQUIRE(p != nullptr);

    // MEASURED, and not what the spec assumes: on the OMP backend AdaptiveCpp
    // answers `host` for EVERY pointer, including a device allocation, because
    // get_pointer_type returns host whenever the allocation came from the host
    // backend. That is not wrong -- OMP device memory IS host memory -- but it
    // means the query cannot recover the kind an allocation was requested with
    // there. Only backends with their own allocator distinguish the three.
    if (dev.get_backend() == sycl::backend::omp)
        REQUIRE(ctx.get_pointer_type(p) == kind::host);
    else
        REQUIRE(ctx.get_pointer_type(p) == kind::device);

    REQUIRE(ctx.get_pointer_device(p) == dev);
    sycl::free(p, ctx.context_of(dev));
}

TEST_CASE("a pointer no backend context knows is errc::invalid", "[context]") {
    const sycl::device dev{sycl::default_selector_v};
    const context      ctx{dev};

    // MEASURED: the OMP backend's allocator recognises any host address, so on
    // an OMP-only context there is no such thing as an unrecognised pointer and
    // nothing to throw about. The contract holds on every backend that owns its
    // own memory.
    if (dev.get_backend() == sycl::backend::omp) return;

    int stack_value = 0;
    REQUIRE_THROWS_AS(ctx.get_pointer_type(&stack_value), sycl::exception);
    REQUIRE_THROWS_AS(ctx.get_pointer_device(&stack_value), sycl::exception);
}
