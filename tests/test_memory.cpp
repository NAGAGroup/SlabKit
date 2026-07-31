#include <catch2/catch_test_macros.hpp>

#include <sycl/sycl.hpp>

#include "naga/slabkit/context.hpp"
#include "naga/slabkit/memory.hpp"
#include "naga/slabkit/scratch.hpp"

// =============================================================================
// DeviceScratchMemory
// =============================================================================

TEST_CASE("make_scratch construction with context", "[memory][scratch]") {
    sycl::device dev{sycl::cpu_selector_v};
    auto ctx = naga::slabkit::ContextBuilder()
                   .devices({dev})
                   .slab_size_per_device(1024 * 1024)
                   .build();
    REQUIRE_NOTHROW(naga::slabkit::make_scratch(ctx, 0));
}

TEST_CASE("DeviceScratchMemory allocate and unwind", "[memory][scratch]") {
    sycl::device dev{sycl::cpu_selector_v};
    auto ctx = naga::slabkit::ContextBuilder()
                   .devices({dev})
                   .slab_size_per_device(1024 * 1024)
                   .build();
    auto scratch = naga::slabkit::make_scratch(ctx, 0);

    SECTION("allocate_node returns non-null pointer") {
        void* ptr = scratch.allocate(256, alignof(std::max_align_t));
        REQUIRE(ptr != nullptr);
    }

    SECTION("top()/unwind() — allocate, save top, allocate more, unwind, first pointer still valid") {
        void* ptr1 = scratch.allocate(256, alignof(std::max_align_t));
        REQUIRE(ptr1 != nullptr);
        auto marker = scratch.top();
        void* ptr2 = scratch.allocate(256, alignof(std::max_align_t));
        REQUIRE(ptr2 != nullptr);
        scratch.unwind(marker);
        void* ptr3 = scratch.allocate(256, alignof(std::max_align_t));
        REQUIRE(ptr3 != nullptr);
        REQUIRE(ptr3 == ptr2);
    }
}
