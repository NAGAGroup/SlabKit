#include <catch2/catch_test_macros.hpp>

#include "slabkit/array.hpp"
#include "slabkit/context.hpp"
#include "slabkit/error.hpp"
#include "slabkit/memory.hpp"
#include "slabkit/scratch.hpp"

// =============================================================================
// Phase 3 API Tests
// =============================================================================

// =============================================================================
// Queue factories with new names
// =============================================================================

TEST_CASE("make_in_order_queue registry-backed", "[api][queue]") {
    auto q = slabkit::make_in_order_queue(slabkit::AllocatorRegistry::get().device_by_index(0));
    bool is_valid = q.get_device().is_cpu() || q.get_device().is_gpu() || q.get_device().is_accelerator();
    REQUIRE(is_valid);
}

TEST_CASE("make_out_of_order_queue registry-backed", "[api][queue]") {
    auto q = slabkit::make_out_of_order_queue(slabkit::AllocatorRegistry::get().device_by_index(0));
    bool is_valid = q.get_device().is_cpu() || q.get_device().is_gpu() || q.get_device().is_accelerator();
    REQUIRE(is_valid);
}

TEST_CASE("make_queue_with_props registry-backed", "[api][queue]") {
    sycl::property_list props;
    auto q = slabkit::make_queue_with_props(slabkit::AllocatorRegistry::get().device_by_index(0), props);
    bool is_valid = q.get_device().is_cpu() || q.get_device().is_gpu() || q.get_device().is_accelerator();
    REQUIRE(is_valid);
}

// =============================================================================
// Registry-backed array allocation
// =============================================================================

TEST_CASE("make_device_array registry-backed", "[api][make_device_array]") {
    SECTION("allocate a small device array from device") {
        auto dev = slabkit::AllocatorRegistry::get().device_by_index(0);
        auto arr = slabkit::make_device_array<int, 1>(sycl::range<1>{100}, dev);
        REQUIRE(arr.size() == 100);
        REQUIRE(arr.data() != nullptr);
        REQUIRE(arr.device_index() == 0);
    }
}

// =============================================================================
// Registry-backed malloc_device
// =============================================================================

TEST_CASE("malloc_device registry-backed", "[api][malloc_device]") {
    SECTION("allocate single element") {
        auto dev = slabkit::AllocatorRegistry::get().device_by_index(0);
        auto ptr = slabkit::malloc_device<int>(1, dev);
        REQUIRE(ptr != nullptr);
    }
}

TEST_CASE("malloc_device registry-free", "[api][malloc_device]") {
    SECTION("allocate with explicit context") {
        auto ctx = slabkit::AllocatorRegistry::context();
        auto ptr = slabkit::malloc_device<int>(1, ctx, 0);
        REQUIRE(ptr != nullptr);
    }
}

// =============================================================================
// AllocationError exception
// =============================================================================

TEST_CASE("AllocationError thrown on pool exhaustion", "[api][error]") {
    SECTION("allocate from tiny pool to trigger exhaustion") {
        // Create a tiny context with very limited space
        auto dev = slabkit::AllocatorRegistry::get().device_by_index(0);
        
        slabkit::SlabContextBuilder builder;
        builder
            .contexts({sycl::context(dev)})
            .devices({dev})
            .node_size(1024)
            .slab_size_per_device(4 * 1024);  // 4 slots of 1KB each
        
        auto ctx = builder.build();
        
        // First 4 allocations should succeed (4 slots)
        auto arr1 = slabkit::make_device_array<int, 1>(sycl::range<1>{256}, ctx, 0);
        auto arr2 = slabkit::make_device_array<int, 1>(sycl::range<1>{256}, ctx, 0);
        auto arr3 = slabkit::make_device_array<int, 1>(sycl::range<1>{256}, ctx, 0);
        auto arr4 = slabkit::make_device_array<int, 1>(sycl::range<1>{256}, ctx, 0);
        
        // 5th allocation should throw AllocationError
        using AE = slabkit::AllocationError;
        REQUIRE_THROWS_AS(
            (slabkit::make_device_array<int, 1>(sycl::range<1>{256}, ctx, 0)),
            AE
        );
    }
}

// =============================================================================
// AllocatorRegistry::context()
// =============================================================================

TEST_CASE("AllocatorRegistry::context() returns valid SlabContext", "[api][registry]") {
    SECTION("context is non-null") {
        auto ctx = slabkit::AllocatorRegistry::context();
        REQUIRE(ctx != nullptr);
    }
    
    SECTION("context can be used for allocation") {
        auto ctx = slabkit::AllocatorRegistry::context();
        auto arr = slabkit::make_device_array<int, 1>(sycl::range<1>{100}, ctx, 0);
        REQUIRE(arr.size() == 100);
    }
}

// =============================================================================
// Scratch memory (registry-backed and registry-free)
// =============================================================================

TEST_CASE("make_scratch registry-backed", "[api][scratch]") {
    SECTION("allocate scratch from device") {
        auto dev = slabkit::AllocatorRegistry::get().device_by_index(0);
        auto scratch = slabkit::make_scratch(dev);
        REQUIRE(scratch.capacity() > 0);
        REQUIRE(scratch.used() == 0);
    }
    
    SECTION("allocate with explicit size") {
        auto dev = slabkit::AllocatorRegistry::get().device_by_index(0);
        auto scratch = slabkit::make_scratch(dev, 1024 * 1024);
        REQUIRE(scratch.capacity() >= 1024 * 1024);
    }
}

TEST_CASE("make_scratch registry-free", "[api][scratch]") {
    SECTION("allocate scratch from explicit context") {
        auto ctx = slabkit::AllocatorRegistry::context();
        auto scratch = slabkit::make_scratch(ctx, 0, 64 * 1024);
        REQUIRE(scratch.capacity() >= 64 * 1024);
    }
}

// =============================================================================
// DeviceScratchBuffer is DeviceScratchMemory
// =============================================================================

TEST_CASE("DeviceScratchBuffer is DeviceScratchMemory", "[api][scratch]") {
    SECTION("type alias is correctly set") {
        auto dev = slabkit::AllocatorRegistry::get().device_by_index(0);
        slabkit::DeviceScratchBuffer buf = slabkit::make_scratch(dev);
        REQUIRE(buf.capacity() > 0);
    }
}

// =============================================================================
// DeviceScratchView
// =============================================================================

TEST_CASE("DeviceScratchView construction and access", "[api][scratch]") {
    SECTION("construct from allocation pointer and size") {
        auto dev = slabkit::AllocatorRegistry::get().device_by_index(0);
        auto scratch = slabkit::make_scratch(dev, 1024);
        
        void* ptr = scratch.allocate(256);
        auto view = slabkit::DeviceScratchView::from_allocation(ptr, 256);
        
        REQUIRE(view.data() == ptr);
        REQUIRE(view.size() == 256);
        REQUIRE(!view.empty());
    }
}

// =============================================================================
// MirroredArray with registry-backed constructor
// =============================================================================

TEST_CASE("MirroredArray registry-backed constructor", "[api][mirrored_array]") {
    SECTION("construct from device") {
        auto dev = slabkit::AllocatorRegistry::get().device_by_index(0);
        auto mirrored = slabkit::MirroredArray<int, 1>(sycl::range<1>{100}, dev);
        REQUIRE(mirrored.size() == 100);
        REQUIRE(mirrored.host_data().size() == 100);
    }
}


