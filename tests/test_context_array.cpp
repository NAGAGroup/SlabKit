#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <array>

#include "slabkit/array.hpp"
#include "slabkit/context.hpp"

// Helper: create a SlabContext for testing
inline std::shared_ptr<slabkit::SlabContext> make_test_context() {
    sycl::device dev{sycl::cpu_selector_v};
    sycl::context sycl_ctx{dev};
    return slabkit::SlabContextBuilder()
            .contexts({sycl_ctx})
            .devices({dev})
            .node_size(1024)
            .slab_size_per_device(4 * 1024 * 1024)
            .build();
}

// =============================================================================
// DeviceArray Phase 2 tests
// =============================================================================

TEST_CASE("DeviceArray from_pool move-only construction", "[array][device_array]") {
    auto ctx = make_test_context();
    auto arr = slabkit::DeviceArray<float, 1>::from_pool(sycl::range<1>{8}, ctx, 0);
    
    REQUIRE(arr.data() != nullptr);
    REQUIRE(arr.size() == 8);
}

TEST_CASE("DeviceArray from_pool round-trip via sync", "[array][device_array]") {
    auto ctx = make_test_context();
    auto arr = slabkit::make_device_array<float, 1>(sycl::range<1>{4}, ctx, 0);
    sycl::queue q = ctx->make_in_order_queue(ctx->devices[0]);

    std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f};
    slabkit::copy_to_device(src.data(), arr.view(), q).wait();
    
    std::vector<float> dst(4);
    slabkit::copy_to_host(arr, dst.data(), q).wait();
    
    REQUIRE(dst == src);
}

TEST_CASE("DeviceArray view() returns correct range", "[array][device_array]") {
    auto ctx = make_test_context();
    auto arr = slabkit::make_device_array<int, 2>(sycl::range<2>{3, 4}, ctx, 0);
    
    auto v = arr.view();
    REQUIRE(v.get_range() == sycl::range<2>{3, 4});
    REQUIRE(v.size() == 12);
}

TEST_CASE("DeviceArray const view has correct size", "[array][device_array]") {
    auto ctx = make_test_context();
    auto arr = slabkit::make_device_array<int, 1>(sycl::range<1>{16}, ctx, 0);
    
    auto cv = arr.view();
    REQUIRE(cv.size() == 16);
    REQUIRE(cv.byte_size() == 16 * sizeof(int));
}

TEST_CASE("DeviceArray subview has correct strides", "[array][device_array]") {
    auto ctx = make_test_context();
    auto arr = slabkit::make_device_array<float, 2>(sycl::range<2>{4, 5}, ctx, 0);
    
    auto sv = arr.subview({1, 1}, sycl::range<2>{2, 3});
    REQUIRE(sv.get_range() == sycl::range<2>{2, 3});
    REQUIRE(sv.size() == 6);
    REQUIRE(sv.strides_ == sycl::range<2>{5, 1});
}

TEST_CASE("DeviceArray implicit conversion to DeviceView", "[array][device_array]") {
    auto ctx = make_test_context();
    auto arr = slabkit::make_device_array<int, 1>(sycl::range<1>{8}, ctx, 0);
    
    // This should work via implicit conversion operator
    slabkit::DeviceView<int, 1> view = arr;
    REQUIRE(view.data() == arr.data());
    REQUIRE(view.get_range() == arr.get_range());
}

// =============================================================================
// MirroredArray Phase 2 tests
// =============================================================================

TEST_CASE("MirroredArray move-only semantics", "[array][mirrored_array]") {
    auto ctx = make_test_context();
    auto ma = slabkit::MirroredArray<float, 1>{sycl::range<1>{4}, ctx, 0};
    
    // Move constructor
    auto ma2 = std::move(ma);
    REQUIRE(ma2.size() == 4);
    // REQUIRE(!ma.data()); // Can't check moved-from state reliably
}

TEST_CASE("MirroredArray sync_to_device and sync_to_host round-trip", "[array][mirrored_array]") {
    auto ctx = make_test_context();
    auto ma = slabkit::MirroredArray<float, 1>{sycl::range<1>{4}, ctx, 0};
    sycl::queue q = ctx->make_in_order_queue(ctx->devices[0]);

    // Write to host, sync to device, read back
    auto h = ma.host_data();
    h[0] = 10.0f; h[1] = 20.0f; h[2] = 30.0f; h[3] = 40.0f;
    
    ma.sync_to_device(q).wait();
    
    // Clear host and sync back
    std::fill(h.begin(), h.end(), 0.0f);
    ma.sync_to_host(q).wait();
    
    REQUIRE(h[0] == 10.0f);
    REQUIRE(h[3] == 40.0f);
    
    q.wait();
}

TEST_CASE("MirroredArray clone creates independent copy", "[array][mirrored_array]") {
    auto ctx = make_test_context();
    auto ma = slabkit::MirroredArray<int, 1>{sycl::range<1>{4}, ctx, 0};
    
    // Initialize host data
    auto h = ma.host_data();
    h[0] = 100; h[1] = 200; h[2] = 300; h[3] = 400;
    
    // Clone
    auto ma2 = ma.clone();
    
    // Verify clone has same host data
    auto h2 = ma2.host_data();
    REQUIRE(h2[0] == 100);
    REQUIRE(h2[3] == 400);
    
    // Verify they're independent
    h[0] = 999;
    REQUIRE(h2[0] == 100);  // clone's copy unaffected
}

// =============================================================================
// malloc_device tests
// =============================================================================

TEST_CASE("malloc_device registry-free allocates and frees", "[array][malloc_device]") {
    auto ctx = make_test_context();
    
    {
        auto ptr = slabkit::malloc_device<float>(16, ctx, 0);
        REQUIRE(ptr.get() != nullptr);
    }  // ptr goes out of scope, deleter called
    
    // If we get here without segfault, deallocation worked
    REQUIRE(true);
}

// =============================================================================
// Concept tests
// =============================================================================

TEST_CASE("DeviceViewLike concept satisfied by DeviceView", "[array][concept]") {
    static_assert(slabkit::DeviceViewLike<slabkit::DeviceView<float, 2>, float, 2>);
    REQUIRE(true);
}

TEST_CASE("DeviceViewLike concept satisfied by DeviceSubView", "[array][concept]") {
    static_assert(slabkit::DeviceViewLike<slabkit::DeviceSubView<float, 2>, float, 2>);
    REQUIRE(true);
}
