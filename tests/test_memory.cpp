#include <catch2/catch_test_macros.hpp>

// TODO: #include "slabkit/memory.hpp" once headers are compilable

// =============================================================================
// AllocatorRegistry
// =============================================================================

TEST_CASE("AllocatorRegistry is a singleton", "[memory][registry]") {
    SECTION("get() returns the same instance") {
        SUCCEED("not yet implemented");
    }
}

TEST_CASE("AllocatorRegistry lazy-initializes on first allocation", "[memory][registry]") {
    SECTION("first access for a device index initializes pool") {
        SUCCEED("not yet implemented");
    }
}

TEST_CASE("AllocatorRegistry configuration", "[memory][registry]") {
    SECTION("set_page_budget overrides default") {
        SUCCEED("not yet implemented");
    }
    SECTION("set_page_size overrides default") {
        SUCCEED("not yet implemented");
    }
}

// =============================================================================
// DevicePageAllocatorImpl
// =============================================================================

TEST_CASE("DevicePageAllocatorImpl allocates and deallocates nodes", "[memory][allocator]") {
    SECTION("allocate_node returns non-null pointer") {
        SUCCEED("not yet implemented");
    }
    SECTION("deallocate_node does not double-free") {
        SUCCEED("not yet implemented");
    }
    SECTION("allocate_node throws bad_alloc when exhausted") {
        SUCCEED("not yet implemented");
    }
}

// =============================================================================
// DeviceScratchMemory
// =============================================================================

TEST_CASE("DeviceScratchMemory bulk unwind", "[memory][scratch]") {
    SECTION("unwind to marker frees all allocations since marker") {
        SUCCEED("not yet implemented");
    }
}
