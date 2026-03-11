#include <catch2/catch_test_macros.hpp>
#include <numeric>
#include <vector>

#include <sycl/sycl.hpp>

#include "slabkit/array.hpp"
#include "slabkit/context.hpp"
#include "slabkit/scratch.hpp"

// ============================================================================
// Fixture: Create Context for testing
// ============================================================================

inline std::shared_ptr<slabkit::Context> make_test_context() {
    sycl::device dev{sycl::cpu_selector_v};
    return slabkit::ContextBuilder()
            .devices({dev})
            .node_size(256)
            .slab_size_per_device(8 * 1024 * 1024)
            .build();
}

// ============================================================================
// Tier 1 Tests — contiguous memcpy
// ============================================================================

TEST_CASE("copy_to_host from DeviceArray round-trip", "[transfer][tier1]") {
    auto ctx = make_test_context();
    auto q = ctx->make_in_order_queue(ctx->devices[0]);

    // Prepare host source data
    std::vector<float> host_src(64);
    std::iota(host_src.begin(), host_src.end(), 0.f);

    // Allocate device array and copy to device
    auto arr = slabkit::make_device_array<float, 1>(sycl::range<1>{64}, ctx, 0);
    slabkit::copy_to_device(host_src.data(), arr, q).wait();

    // Read back from device
    std::vector<float> host_dst(64, 0.f);
    slabkit::copy_to_host(arr, host_dst.data(), q).wait();

    REQUIRE(host_dst == host_src);
}

TEST_CASE("copy_to_device then copy_to_host round-trip", "[transfer][tier1]") {
    auto ctx = make_test_context();
    auto q = ctx->make_in_order_queue(ctx->devices[0]);

    // Prepare host source data
    std::vector<int> host_src(32);
    std::iota(host_src.begin(), host_src.end(), 100);

    // Allocate two device arrays
    auto arr1 = slabkit::make_device_array<int, 1>(sycl::range<1>{32}, ctx, 0);
    auto arr2 = slabkit::make_device_array<int, 1>(sycl::range<1>{32}, ctx, 0);

    // Copy to device 1, then device-to-device copy to device 2
    slabkit::copy_to_device(host_src.data(), arr1, q).wait();
    slabkit::copy(arr1, arr2, q).wait();

    // Copy back from device 2
    std::vector<int> host_dst(32, 0);
    slabkit::copy_to_host(arr2, host_dst.data(), q).wait();

    REQUIRE(host_dst == host_src);
}

TEST_CASE("copy device-to-device same range", "[transfer][tier1]") {
    auto ctx = make_test_context();
    auto q = ctx->make_in_order_queue(ctx->devices[0]);

    const size_t size = 16;
    std::vector<double> host_src(size);
    std::iota(host_src.begin(), host_src.end(), 1.5);

    auto arr_src = slabkit::make_device_array<double, 1>(sycl::range<1>{size}, ctx, 0);
    auto arr_dst = slabkit::make_device_array<double, 1>(sycl::range<1>{size}, ctx, 0);

    slabkit::copy_to_device(host_src.data(), arr_src, q).wait();
    slabkit::copy(arr_src, arr_dst, q).wait();

    std::vector<double> host_dst(size, 0.0);
    slabkit::copy_to_host(arr_dst, host_dst.data(), q).wait();

    REQUIRE(host_dst == host_src);
}

TEST_CASE("copy_to_host via DeviceView", "[transfer][tier1]") {
    auto ctx = make_test_context();
    auto q = ctx->make_in_order_queue(ctx->devices[0]);

    std::vector<float> host_src(24);
    std::iota(host_src.begin(), host_src.end(), 10.f);

    auto arr = slabkit::make_device_array<float, 1>(sycl::range<1>{24}, ctx, 0);
    slabkit::copy_to_device(host_src.data(), arr, q).wait();

    // Get a view from the array
    auto view = arr.view();
    std::vector<float> host_dst(24, 0.f);
    slabkit::copy_to_host(view, host_dst.data(), q).wait();

    REQUIRE(host_dst == host_src);
}

// ============================================================================
// Tier 2 Tests — strided host transfers (via scratch)
// ============================================================================

TEST_CASE("copy_to_host from DeviceSubView (strided gather)", "[transfer][tier2]") {
    auto ctx = make_test_context();
    auto q = ctx->make_in_order_queue(ctx->devices[0]);
    auto scratch = slabkit::make_scratch(ctx, 0, 64 * 1024);

    // Create a 1D device array of 8 elements
    std::vector<int> host_src(8);
    std::iota(host_src.begin(), host_src.end(), 0);

    auto arr = slabkit::make_device_array<int, 1>(sycl::range<1>{8}, ctx, 0);
    slabkit::copy_to_device(host_src.data(), arr, q).wait();

    // Extract a subview: offset={2}, range={3} → elements [2, 3, 4]
    auto sv = arr.subview({2}, sycl::range<1>{3});

    // Gather strided subview to contiguous host buffer
    std::vector<int> host_dst(3, -1);
    slabkit::copy_to_host(sv, host_dst.data(), q, scratch).wait();

    std::vector<int> expected{2, 3, 4};
    REQUIRE(host_dst == expected);
}

TEST_CASE("copy_to_device to DeviceSubView (strided scatter)", "[transfer][tier2]") {
    auto ctx = make_test_context();
    auto q = ctx->make_in_order_queue(ctx->devices[0]);
    auto scratch = slabkit::make_scratch(ctx, 0, 64 * 1024);

    // Create a 1D device array of 8 elements, initialize to zeros
    auto arr = slabkit::make_device_array<int, 1>(sycl::range<1>{8}, ctx, 0);
    std::vector<int> host_zeros(8, 0);
    slabkit::copy_to_device(host_zeros.data(), arr, q).wait();

    // Scatter: copy [10, 20, 30] to elements [1, 2, 3] via subview
    std::vector<int> host_src{10, 20, 30};
    auto sv = arr.subview({1}, sycl::range<1>{3});
    slabkit::copy_to_device(host_src.data(), sv, q, scratch).wait();

    // Read back entire array
    std::vector<int> host_dst(8, -1);
    slabkit::copy_to_host(arr, host_dst.data(), q).wait();

    std::vector<int> expected{0, 10, 20, 30, 0, 0, 0, 0};
    REQUIRE(host_dst == expected);
}

TEST_CASE("copy_to_host from const DeviceSubView", "[transfer][tier2]") {
    auto ctx = make_test_context();
    auto q = ctx->make_in_order_queue(ctx->devices[0]);
    auto scratch = slabkit::make_scratch(ctx, 0, 64 * 1024);

    // Create and fill a device array
    std::vector<float> host_src(16);
    std::iota(host_src.begin(), host_src.end(), 0.f);

    auto arr = slabkit::make_device_array<float, 1>(sycl::range<1>{16}, ctx, 0);
    slabkit::copy_to_device(host_src.data(), arr, q).wait();

    // Extract const subview
    auto const_sv = arr.const_subview({5}, sycl::range<1>{6});

    // Copy from const subview
    std::vector<float> host_dst(6, -1.f);
    slabkit::copy_to_host(const_sv, host_dst.data(), q, scratch).wait();

    std::vector<float> expected{5.f, 6.f, 7.f, 8.f, 9.f, 10.f};
    REQUIRE(host_dst == expected);
}

// ============================================================================
// Tier 3 Tests — strided device-to-device
// ============================================================================

TEST_CASE("copy DeviceSubView to DeviceSubView", "[transfer][tier3]") {
    auto ctx = make_test_context();
    auto q = ctx->make_in_order_queue(ctx->devices[0]);

    // Create source array and fill with values 0..15
    std::vector<int> host_src(16);
    std::iota(host_src.begin(), host_src.end(), 0);

    auto src_arr = slabkit::make_device_array<int, 1>(sycl::range<1>{16}, ctx, 0);
    slabkit::copy_to_device(host_src.data(), src_arr, q).wait();

    // Create destination array (uninitialized)
    auto dst_arr = slabkit::make_device_array<int, 1>(sycl::range<1>{16}, ctx, 0);

    // Extract subviews: src[4..9], dst[0..5]
    auto src_sv = src_arr.subview({4}, sycl::range<1>{6});
    auto dst_sv = dst_arr.subview({0}, sycl::range<1>{6});

    // Copy between subviews
    slabkit::copy(src_sv, dst_sv, q).wait();

    // Read back destination
    std::vector<int> host_dst(16);
    slabkit::copy_to_host(dst_arr, host_dst.data(), q).wait();

    // Only the first 6 elements (subview positions [0..5]) should be updated
    for (int i = 0; i < 6; ++i)
        REQUIRE(host_dst[i] == i + 4);  // src[4..9] → dst[0..5]
}

TEST_CASE("copy DeviceSubView to DeviceArray", "[transfer][tier3]") {
    auto ctx = make_test_context();
    auto q = ctx->make_in_order_queue(ctx->devices[0]);

    // Create source array with 20 elements
    std::vector<float> host_src(20);
    std::iota(host_src.begin(), host_src.end(), 100.f);

    auto src_arr = slabkit::make_device_array<float, 1>(sycl::range<1>{20}, ctx, 0);
    slabkit::copy_to_device(host_src.data(), src_arr, q).wait();

    // Extract subview: elements [5..14]
    auto src_sv = src_arr.subview({5}, sycl::range<1>{10});

    // Create destination array of exact subview size
    auto dst_arr = slabkit::make_device_array<float, 1>(sycl::range<1>{10}, ctx, 0);

    // Copy strided subview to contiguous array
    slabkit::copy(src_sv, dst_arr, q).wait();

    // Verify
    std::vector<float> host_dst(10, -1.f);
    slabkit::copy_to_host(dst_arr, host_dst.data(), q).wait();

    std::vector<float> expected{105.f, 106.f, 107.f, 108.f, 109.f, 110.f, 111.f, 112.f, 113.f, 114.f};
    REQUIRE(host_dst == expected);
}

TEST_CASE("copy DeviceArray to DeviceSubView", "[transfer][tier3]") {
    auto ctx = make_test_context();
    auto q = ctx->make_in_order_queue(ctx->devices[0]);

    // Create source array (contiguous, small)
    std::vector<double> host_src(5);
    std::iota(host_src.begin(), host_src.end(), 1.1);

    auto src_arr = slabkit::make_device_array<double, 1>(sycl::range<1>{5}, ctx, 0);
    slabkit::copy_to_device(host_src.data(), src_arr, q).wait();

    // Create destination array (larger, initialized to 0)
    auto dst_arr = slabkit::make_device_array<double, 1>(sycl::range<1>{15}, ctx, 0);
    std::vector<double> host_zeros(15, 0.0);
    slabkit::copy_to_device(host_zeros.data(), dst_arr, q).wait();

    // Extract subview at offset 5, size 5
    auto dst_sv = dst_arr.subview({5}, sycl::range<1>{5});

    // Copy contiguous array to strided subview
    slabkit::copy(src_arr, dst_sv, q).wait();

    // Verify
    std::vector<double> host_dst(15, -1.0);
    slabkit::copy_to_host(dst_arr, host_dst.data(), q).wait();

    std::vector<double> expected{0.0, 0.0, 0.0, 0.0, 0.0, 1.1, 2.1, 3.1, 4.1, 5.1, 0.0, 0.0, 0.0, 0.0, 0.0};
    REQUIRE(host_dst == expected);
}

// ============================================================================
// Tier 4 Tests — blocking sync wrappers
// ============================================================================

TEST_CASE("sync_to_host from DeviceArray (blocking)", "[transfer][tier4]") {
    auto ctx = make_test_context();
    auto q = ctx->make_in_order_queue(ctx->devices[0]);

    std::vector<int> host_src(32);
    std::iota(host_src.begin(), host_src.end(), 50);

    auto arr = slabkit::make_device_array<int, 1>(sycl::range<1>{32}, ctx, 0);
    slabkit::copy_to_device(host_src.data(), arr, q).wait();

    // Use blocking sync_to_host (should wait internally)
    std::vector<int> host_dst(32, 0);
    slabkit::sync_to_host(arr, host_dst.data(), q);

    // No additional .wait() needed — sync_to_host blocks
    REQUIRE(host_dst == host_src);
}

TEST_CASE("sync_to_host from DeviceSubView (blocking)", "[transfer][tier4]") {
    auto ctx = make_test_context();
    auto q = ctx->make_in_order_queue(ctx->devices[0]);
    auto scratch = slabkit::make_scratch(ctx, 0, 64 * 1024);

    // Create source array
    std::vector<float> host_src(12);
    std::iota(host_src.begin(), host_src.end(), 2.5f);

    auto arr = slabkit::make_device_array<float, 1>(sycl::range<1>{12}, ctx, 0);
    slabkit::copy_to_device(host_src.data(), arr, q).wait();

    // Extract subview: offset={3}, range={6}
    auto sv = arr.subview({3}, sycl::range<1>{6});

    // Use blocking sync_to_host with scratch
    std::vector<float> host_dst(6, -1.f);
    slabkit::sync_to_host(sv, host_dst.data(), q, scratch);

    std::vector<float> expected{5.5f, 6.5f, 7.5f, 8.5f, 9.5f, 10.5f};
    REQUIRE(host_dst == expected);
}
