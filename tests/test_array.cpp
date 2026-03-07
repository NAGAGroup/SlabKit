#include <catch2/catch_test_macros.hpp>

// TODO: #include "slabkit/array.hpp" once headers are compilable

// =============================================================================
// DeviceView
// =============================================================================

TEST_CASE("DeviceView satisfies DeviceViewLike concept", "[array][view]") {
    SECTION("data() returns correct pointer") {
        SUCCEED("not yet implemented");
    }
    SECTION("get_range() returns correct range") {
        SUCCEED("not yet implemented");
    }
    SECTION("size() matches range product") {
        SUCCEED("not yet implemented");
    }
}

TEST_CASE("DeviceView element access", "[array][view]") {
    SECTION("operator() with variadic indices (1D)") {
        SUCCEED("not yet implemented");
    }
    SECTION("operator() with variadic indices (2D)") {
        SUCCEED("not yet implemented");
    }
    SECTION("operator[] with sycl::id") {
        SUCCEED("not yet implemented");
    }
}

// =============================================================================
// DeviceSubView
// =============================================================================

TEST_CASE("DeviceSubView element access uses explicit strides", "[array][subview]") {
    SECTION("operator() computes correct strided offset (1D)") {
        SUCCEED("not yet implemented");
    }
    SECTION("operator() computes correct strided offset (2D)") {
        SUCCEED("not yet implemented");
    }
}

// =============================================================================
// DeviceArray
// =============================================================================

TEST_CASE("DeviceArray is move-only", "[array][device_array]") {
    SECTION("move construction transfers ownership") {
        SUCCEED("not yet implemented");
    }
    SECTION("copy construction is deleted") {
        SUCCEED("not yet implemented");
    }
}

TEST_CASE("DeviceArray view and subview", "[array][device_array]") {
    SECTION("view() returns DeviceView over full range") {
        SUCCEED("not yet implemented");
    }
    SECTION("subview() returns DeviceSubView with correct strides") {
        SUCCEED("not yet implemented");
    }
    SECTION("subview() asserts on out-of-bounds offsets") {
        SUCCEED("not yet implemented");
    }
}

TEST_CASE("make_device_array factory", "[array][device_array]") {
    SECTION("allocates from AllocatorRegistry pool for given device") {
        SUCCEED("not yet implemented");
    }
}

// =============================================================================
// MirroredArray
// =============================================================================

TEST_CASE("MirroredArray sync", "[array][mirrored]") {
    SECTION("sync_to_host copies device data to host span") {
        SUCCEED("not yet implemented");
    }
    SECTION("sync_to_device copies host span to device") {
        SUCCEED("not yet implemented");
    }
}

// =============================================================================
// Transfer utilities
// =============================================================================

TEST_CASE("copy_to_host returns sycl::event", "[array][transfer]") {
    SECTION("DeviceArray overload") {
        SUCCEED("not yet implemented");
    }
    SECTION("DeviceView overload") {
        SUCCEED("not yet implemented");
    }
}

TEST_CASE("copy_to_device returns sycl::event", "[array][transfer]") {
    SECTION("DeviceArray overload") {
        SUCCEED("not yet implemented");
    }
    SECTION("DeviceView overload") {
        SUCCEED("not yet implemented");
    }
}
