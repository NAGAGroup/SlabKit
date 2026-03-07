#include <catch2/catch_test_macros.hpp>

#include "slabkit/memory.hpp"

// =============================================================================
// AllocatorRegistry
// =============================================================================

TEST_CASE("AllocatorRegistry is a singleton", "[memory][registry]") {
    SECTION("get() returns the same instance twice") {
        auto& instance1 = slabkit::AllocatorRegistry::get();
        auto& instance2 = slabkit::AllocatorRegistry::get();
        REQUIRE(&instance1 == &instance2);
    }
}

TEST_CASE("AllocatorRegistry pool access", "[memory][registry]") {
    SECTION("pool(device) returns non-null shared_ptr") {
        auto& registry = slabkit::AllocatorRegistry::get();
        auto pool = registry.pool(registry.device_by_index(0));
        REQUIRE(pool != nullptr);
    }

    SECTION("pool(device) called twice returns the same shared_ptr (idempotent lazy init)") {
        auto& registry = slabkit::AllocatorRegistry::get();
        auto pool1 = registry.pool(registry.device_by_index(0));
        auto pool2 = registry.pool(registry.device_by_index(0));
        REQUIRE(pool1 == pool2);
    }
}

TEST_CASE("AllocatorRegistry::Config defaults", "[memory][registry]") {
    SECTION("block_size default is 64KB") {
        auto& config = slabkit::AllocatorRegistry::config();
        REQUIRE(config.block_size == 64ull * 1024);
    }

    SECTION("node_size default is 1024") {
        auto& config = slabkit::AllocatorRegistry::config();
        REQUIRE(config.node_size == 1024);
    }

    SECTION("device_slab_fraction default is 0.60f") {
        auto& config = slabkit::AllocatorRegistry::config();
        REQUIRE(config.device_slab_fraction == 0.60f);
    }

    SECTION("cpu_slab_fraction default is 0.30f") {
        auto& config = slabkit::AllocatorRegistry::config();
        REQUIRE(config.cpu_slab_fraction == 0.30f);
    }
}

// =============================================================================
// DeviceScratchMemory
// =============================================================================

TEST_CASE("make_scratch_memory construction", "[memory][scratch]") {
    SECTION("make_scratch_memory(device) constructs without throwing") {
        REQUIRE_NOTHROW([&]() {
            auto scratch = slabkit::make_scratch_memory(slabkit::AllocatorRegistry::get().device_by_index(0));
        }());
    }

    SECTION("make_scratch_memory(pool) constructs without throwing") {
        auto& registry = slabkit::AllocatorRegistry::get();
        auto pool = registry.pool(registry.device_by_index(0));
        REQUIRE_NOTHROW([&]() {
            auto scratch = slabkit::make_scratch_memory(pool);
        }());
    }
}

TEST_CASE("DeviceScratchMemory allocate and unwind", "[memory][scratch]") {
    auto scratch = slabkit::make_scratch_memory(slabkit::AllocatorRegistry::get().device_by_index(0));

    SECTION("allocate_node returns non-null pointer") {
        void* ptr = scratch.allocate(256, alignof(std::max_align_t));
        REQUIRE(ptr != nullptr);
    }

    SECTION("top()/unwind() — allocate, save top, allocate more, unwind, first pointer still valid") {
        // Allocate first block
        void* ptr1 = scratch.allocate(256, alignof(std::max_align_t));
        REQUIRE(ptr1 != nullptr);

        // Save marker
        auto marker = scratch.top();

        // Allocate second block
        void* ptr2 = scratch.allocate(256, alignof(std::max_align_t));
        REQUIRE(ptr2 != nullptr);

        // Unwind to marker (frees ptr2)
        scratch.unwind(marker);

        // Re-allocate after unwind — proves stack reclaimed the space
        void* ptr3 = scratch.allocate(256, alignof(std::max_align_t));
        REQUIRE(ptr3 != nullptr);
        // ptr3 should be at the same position as ptr2 was (stack rewound)
        REQUIRE(ptr3 == ptr2);
    }
}
