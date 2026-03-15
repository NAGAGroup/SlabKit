#include <catch2/catch_test_macros.hpp>

#include "slabkit/array.hpp"
#include "slabkit/context.hpp"
#include "slabkit/error.hpp"
#include "slabkit/memory.hpp"
#include "slabkit/scratch.hpp"

// =============================================================================
// Context-backed API integration tests
// =============================================================================
//
// These tests exercise the explicit-context API path.
// The legacy global registry has been removed; all allocation now requires
// an explicit slabkit::Context built with ContextBuilder.
//

// =============================================================================
// Queue factories (context-backed)
// =============================================================================

TEST_CASE("Context make_in_order_queue returns usable queue", "[api][queue]") {
    sycl::device dev{sycl::cpu_selector_v};
    auto ctx = slabkit::ContextBuilder()
                   .devices({dev})
                   .slab_size_per_device(4 * 1024 * 1024)
                   .build();
    auto q = ctx->make_in_order_queue(dev);
    bool is_valid = q.get_device().is_cpu() || q.get_device().is_gpu() || q.get_device().is_accelerator();
    REQUIRE(is_valid);
}

TEST_CASE("Context make_out_of_order_queue returns usable queue", "[api][queue]") {
    sycl::device dev{sycl::cpu_selector_v};
    auto ctx = slabkit::ContextBuilder()
                   .devices({dev})
                   .slab_size_per_device(4 * 1024 * 1024)
                   .build();
    auto q = ctx->make_out_of_order_queue(dev);
    bool is_valid = q.get_device().is_cpu() || q.get_device().is_gpu() || q.get_device().is_accelerator();
    REQUIRE(is_valid);
}

TEST_CASE("Context make_queue_with_props returns usable queue", "[api][queue]") {
    sycl::device dev{sycl::cpu_selector_v};
    auto ctx = slabkit::ContextBuilder()
                   .devices({dev})
                   .slab_size_per_device(4 * 1024 * 1024)
                   .build();
    sycl::property_list props;
    auto q = ctx->make_queue_with_props(dev, props);
    bool is_valid = q.get_device().is_cpu() || q.get_device().is_gpu() || q.get_device().is_accelerator();
    REQUIRE(is_valid);
}

// =============================================================================
// make_device_array with sycl::device overload
// =============================================================================

TEST_CASE("make_device_array with sycl::device overload", "[api][make_device_array]") {
    SECTION("allocate a small device array from device") {
        sycl::device dev{sycl::cpu_selector_v};
        auto ctx = slabkit::ContextBuilder()
                       .devices({dev})
                       .slab_size_per_device(4 * 1024 * 1024)
                       .build();
        auto arr = slabkit::make_device_array<int, 1>(sycl::range<1>{100}, ctx, dev);
        REQUIRE(arr.size() == 100);
        REQUIRE(arr.data() != nullptr);
        REQUIRE(arr.device_index() == 0);
    }
}

// =============================================================================
// malloc_device with explicit context
// =============================================================================

TEST_CASE("malloc_device with explicit context", "[api][malloc_device]") {
    SECTION("allocate single element by device_index") {
        sycl::device dev{sycl::cpu_selector_v};
        auto ctx = slabkit::ContextBuilder()
                       .devices({dev})
                       .slab_size_per_device(4 * 1024 * 1024)
                       .build();
        auto ptr = slabkit::malloc_device<int>(1, ctx, 0);
        REQUIRE(ptr != nullptr);
    }

    SECTION("allocate single element by sycl::device") {
        sycl::device dev{sycl::cpu_selector_v};
        auto ctx = slabkit::ContextBuilder()
                       .devices({dev})
                       .slab_size_per_device(4 * 1024 * 1024)
                       .build();
        auto ptr = slabkit::malloc_device<int>(1, ctx, dev);
        REQUIRE(ptr != nullptr);
    }
}

// =============================================================================
// AllocationError exception
// =============================================================================

TEST_CASE("AllocationError thrown on pool exhaustion", "[api][error]") {
    SECTION("allocate from tiny pool to trigger exhaustion") {
        sycl::device dev{sycl::cpu_selector_v};
        slabkit::ContextBuilder builder;
        builder
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
// Scratch memory (context-backed)
// =============================================================================

TEST_CASE("make_scratch with device_index overload", "[api][scratch]") {
    SECTION("allocate scratch from context by device_index") {
        sycl::device dev{sycl::cpu_selector_v};
        auto ctx = slabkit::ContextBuilder()
                       .devices({dev})
                       .slab_size_per_device(4 * 1024 * 1024)
                       .build();
        auto scratch = slabkit::make_scratch(ctx, 0, 64 * 1024);
        REQUIRE(scratch.capacity() >= 64 * 1024);
        REQUIRE(scratch.used() == 0);
    }
}

TEST_CASE("make_scratch with sycl::device overload", "[api][scratch]") {
    SECTION("allocate scratch from context by sycl::device") {
        sycl::device dev{sycl::cpu_selector_v};
        auto ctx = slabkit::ContextBuilder()
                       .devices({dev})
                       .slab_size_per_device(4 * 1024 * 1024)
                       .build();
        auto scratch = slabkit::make_scratch(ctx, dev, 64 * 1024);
        REQUIRE(scratch.capacity() >= 64 * 1024);
    }
}

// =============================================================================
// DeviceScratchMemory
// =============================================================================

TEST_CASE("DeviceScratchMemory is DeviceScratchMemory", "[api][scratch]") {
    SECTION("type is correctly used") {
        sycl::device dev{sycl::cpu_selector_v};
        auto ctx = slabkit::ContextBuilder()
                       .devices({dev})
                       .slab_size_per_device(4 * 1024 * 1024)
                       .build();
        slabkit::DeviceScratchMemory buf = slabkit::make_scratch(ctx, 0);
        REQUIRE(buf.capacity() > 0);
    }
}

// =============================================================================
// DeviceScratchView
// =============================================================================

TEST_CASE("DeviceScratchView construction and access", "[api][scratch]") {
    SECTION("construct from allocation pointer and size") {
        sycl::device dev{sycl::cpu_selector_v};
        auto ctx = slabkit::ContextBuilder()
                       .devices({dev})
                       .slab_size_per_device(4 * 1024 * 1024)
                       .build();
        auto scratch = slabkit::make_scratch(ctx, 0, 1024);

        void* ptr = scratch.allocate(256);
        auto view = slabkit::DeviceScratchView::from_allocation(ptr, 256);

        REQUIRE(view.data() == ptr);
        REQUIRE(view.size() == 256);
        REQUIRE(!view.empty());
    }
}

// =============================================================================
// MirroredArray with explicit context constructor
// =============================================================================

TEST_CASE("MirroredArray explicit context constructor", "[api][mirrored_array]") {
    SECTION("construct from context and device_index") {
        sycl::device dev{sycl::cpu_selector_v};
        auto ctx = slabkit::ContextBuilder()
                       .devices({dev})
                       .slab_size_per_device(4 * 1024 * 1024)
                       .build();
        auto mirrored = slabkit::MirroredArray<int, 1>(sycl::range<1>{100}, ctx, 0);
        REQUIRE(mirrored.size() == 100);
        REQUIRE(mirrored.host_data().size() == 100);
    }
}
