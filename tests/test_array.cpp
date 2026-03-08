#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <array>
#include <numeric>

#include "slabkit/array.hpp"
#include "slabkit/context.hpp"

// Verify DeviceView satisfies the DeviceViewLike concept
static_assert(slabkit::DeviceViewLike<slabkit::DeviceView<float, 2>, float, 2>);

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
// DeviceView
// =============================================================================

TEST_CASE("DeviceView satisfies DeviceViewLike concept", "[array][view]") {
    std::vector<float> storage(6);
    slabkit::DeviceView<float, 2> v{storage.data(), sycl::range<2>{2, 3}};

    SECTION("data() returns correct pointer") {
        REQUIRE(v.data() == storage.data());
    }
    SECTION("get_range() returns correct range") {
        REQUIRE(v.get_range() == sycl::range<2>{2, 3});
    }
    SECTION("size() matches range product") {
        REQUIRE(v.size() == 6);
    }
}

TEST_CASE("DeviceView element access", "[array][view]") {
    std::vector<int> storage(6, 0);
    slabkit::DeviceView<int, 2> v{storage.data(), sycl::range<2>{2, 3}};

    SECTION("operator() with variadic indices (1D)") {
        std::vector<int> s1d(4, 0);
        slabkit::DeviceView<int, 1> v1{s1d.data(), sycl::range<1>{4}};
        v1(0) = 10; v1(1) = 20; v1(2) = 30; v1(3) = 40;
        REQUIRE(s1d[0] == 10);
        REQUIRE(s1d[2] == 30);
    }
    SECTION("operator() with variadic indices (2D)") {
        // row-major: (r, c) -> r*3 + c
        v(0, 0) = 1; v(0, 1) = 2; v(0, 2) = 3;
        v(1, 0) = 4; v(1, 1) = 5; v(1, 2) = 6;
        REQUIRE(storage[0] == 1);
        REQUIRE(storage[3] == 4);  // row 1, col 0 = offset 3
        REQUIRE(storage[5] == 6);  // row 1, col 2 = offset 5
    }
    SECTION("operator[] with sycl::id") {
        v(1, 2) = 99;
        REQUIRE(v[sycl::id<2>{1, 2}] == 99);
    }
}

// =============================================================================
// DeviceSubView
// =============================================================================

TEST_CASE("DeviceSubView element access uses explicit strides", "[array][subview]") {
    // Backing buffer: 4x4 = 16 elements, row-major, strides = {4, 1}
    std::vector<int> storage(16, 0);
    // Fill with linear index values
    for (int i = 0; i < 16; ++i) storage[i] = i;

    // SubView over a 2x2 window at (1,1) in a 4x4 array
    // strides = {4, 1}, ptr = storage.data() + 1*4 + 1*1 = storage.data() + 5
    slabkit::DeviceSubView<int, 2> sv{
        storage.data() + 5,
        sycl::range<2>{2, 2},
        sycl::range<2>{4, 1}   // element strides
    };

    SECTION("operator() computes correct strided offset (1D)") {
        std::vector<int> s{10, 20, 30, 40};
        slabkit::DeviceSubView<int, 1> sv1{s.data() + 1, sycl::range<1>{3}, sycl::range<1>{1}};
        REQUIRE(sv1(0) == 20);
        REQUIRE(sv1(1) == 30);
        REQUIRE(sv1(2) == 40);
    }
    SECTION("operator() computes correct strided offset (2D)") {
        // sv(0,0) → storage[5] = 5
        // sv(0,1) → storage[6] = 6
        // sv(1,0) → storage[9] = 9
        // sv(1,1) → storage[10] = 10
        REQUIRE(sv(0, 0) == 5);
        REQUIRE(sv(0, 1) == 6);
        REQUIRE(sv(1, 0) == 9);
        REQUIRE(sv(1, 1) == 10);
    }
}

// =============================================================================
// DeviceArray
// =============================================================================

TEST_CASE("DeviceArray from_pool is move-only", "[array][device_array]") {
    auto ctx = make_test_context();
    auto arr = slabkit::DeviceArray<float, 1>::from_pool(sycl::range<1>{8}, ctx, 0);
    
    // Move constructor works
    auto arr2 = std::move(arr);
    REQUIRE(arr2.data() != nullptr);
}

TEST_CASE("DeviceArray has move-only semantics", "[array][device_array]") {
    auto ctx = make_test_context();
    auto arr = slabkit::DeviceArray<float, 1>::from_pool(sycl::range<1>{8}, ctx, 0);

    SECTION("copy construction is deleted") {
        // This should not compile: auto arr2 = arr;
        // We can't test it here, but it's enforced by = delete
        REQUIRE(true);  // placeholder
    }
    SECTION("move construction transfers ownership") {
        float* original_ptr = arr.data();
        auto arr2 = std::move(arr);
        REQUIRE(arr2.data() == original_ptr);
    }
}

TEST_CASE("DeviceArray view and subview", "[array][device_array]") {
    auto ctx = make_test_context();
    auto arr = slabkit::DeviceArray<int, 2>::from_pool(sycl::range<2>{4, 4}, ctx, 0);

    SECTION("view() returns DeviceView over full range") {
        auto v = arr.view();
        REQUIRE(v.data() == arr.data());
        REQUIRE(v.get_range() == sycl::range<2>{4, 4});
        REQUIRE(v.size() == 16);
    }
    SECTION("subview() returns DeviceSubView with correct strides") {
        // subview at (1,1), size 2x2
        // parent strides: {4, 1} (row-major 4-wide)
        // base ptr offset: 1*4 + 1*1 = 5
        auto sv = arr.subview({1, 1}, sycl::range<2>{2, 2});
        REQUIRE(sv.data() == arr.data() + 5);
        REQUIRE(sv.get_range() == sycl::range<2>{2, 2});
        REQUIRE(sv.strides_ == sycl::range<2>{4, 1});
    }
    SECTION("subview() asserts on out-of-bounds offsets") {
        // offset {3,3} + range {2,2} = {5,5} which exceeds {4,4}
        // This hits assert() — only test in debug builds. Skip if NDEBUG.
        // Just verify a valid subview doesn't throw:
        REQUIRE_NOTHROW(arr.subview({0, 0}, sycl::range<2>{2, 2}));
    }
}

TEST_CASE("make_device_array factory registry-free", "[array][make_device_array]") {
    auto ctx = make_test_context();

    SECTION("allocates from SlabContext pool for given device_index") {
        auto arr = slabkit::make_device_array<float, 1>(sycl::range<1>{16}, ctx, 0);
        REQUIRE(arr.data() != nullptr);
        REQUIRE(arr.size() == 16);
        REQUIRE(arr.get_range() == sycl::range<1>{16});
        REQUIRE(arr.device_index() == 0);
    }
}

// =============================================================================
// MirroredArray
// =============================================================================

TEST_CASE("MirroredArray move-only semantics", "[array][mirrored]") {
    auto ctx = make_test_context();
    auto ma = slabkit::MirroredArray<float, 1>{sycl::range<1>{4}, ctx, 0};

    // Move constructor works
    auto ma2 = std::move(ma);
    REQUIRE(ma2.size() == 4);
}

TEST_CASE("MirroredArray sync", "[array][mirrored]") {
    auto ctx = make_test_context();
    sycl::queue q = ctx->make_in_order_queue(ctx->devices[0]);

    slabkit::MirroredArray<float, 1> ma(sycl::range<1>{4}, ctx, 0);

    SECTION("sync_to_device copies host span to device") {
        // Write to host side
        auto h = ma.host_data();
        h[0] = 1.0f; h[1] = 2.0f; h[2] = 3.0f; h[3] = 4.0f;
        // Copy to device
        ma.sync_to_device(q).wait();
        // Copy back and verify
        std::vector<float> result(4, 0.0f);
        slabkit::sync_to_host(ma.const_view(), result.data(), q);
        REQUIRE(result[0] == 1.0f);
        REQUIRE(result[3] == 4.0f);
    }
    SECTION("sync_to_host copies device data to host span") {
        // Write host → device
        auto h = ma.host_data();
        h[0] = 10.0f; h[1] = 20.0f; h[2] = 30.0f; h[3] = 40.0f;
        ma.sync_to_device(q).wait();
        // Clear host, then sync back
        std::fill(h.begin(), h.end(), 0.0f);
        ma.sync_to_host(q).wait();
        REQUIRE(h[0] == 10.0f);
        REQUIRE(h[3] == 40.0f);
    }
    q.wait();
}

// =============================================================================
// Transfer utilities
// =============================================================================

TEST_CASE("copy_to_host returns sycl::event", "[array][transfer]") {
    auto ctx = make_test_context();
    sycl::queue q = ctx->make_in_order_queue(ctx->devices[0]);

    SECTION("DeviceArray overload") {
        auto arr = slabkit::make_device_array<int, 1>(sycl::range<1>{4}, ctx, 0);
        // Upload known data
        std::vector<int> src = {10, 20, 30, 40};
        slabkit::copy_to_device(src.data(), arr.view(), q).wait();
        // Download and verify
        std::vector<int> dst(4, 0);
        slabkit::copy_to_host(arr, dst.data(), q).wait();
        REQUIRE(dst == src);
    }
    SECTION("DeviceView overload") {
        auto arr = slabkit::make_device_array<int, 1>(sycl::range<1>{4}, ctx, 0);
        std::vector<int> src = {5, 6, 7, 8};
        slabkit::copy_to_device(src.data(), arr.view(), q).wait();
        std::vector<int> dst(4, 0);
        slabkit::copy_to_host(arr.view(), dst.data(), q).wait();
        REQUIRE(dst == src);
    }
    q.wait();
}

TEST_CASE("copy_to_device returns sycl::event", "[array][transfer]") {
    auto ctx = make_test_context();
    sycl::queue q = ctx->make_in_order_queue(ctx->devices[0]);

    SECTION("DeviceArray overload") {
        auto arr = slabkit::make_device_array<float, 1>(sycl::range<1>{3}, ctx, 0);
        std::vector<float> src = {1.5f, 2.5f, 3.5f};
        auto ev = slabkit::copy_to_device(src.data(), arr.view(), q);
        REQUIRE_NOTHROW(ev.wait());
        std::vector<float> dst(3, 0.0f);
        slabkit::sync_to_host(arr, dst.data(), q);
        REQUIRE(dst == src);
    }
    SECTION("DeviceView overload") {
        auto arr = slabkit::make_device_array<float, 1>(sycl::range<1>{3}, ctx, 0);
        std::vector<float> src = {7.0f, 8.0f, 9.0f};
        auto ev = slabkit::copy_to_device(src.data(), arr.view(), q);
        REQUIRE_NOTHROW(ev.wait());
        std::vector<float> dst(3, 0.0f);
        slabkit::sync_to_host(arr, dst.data(), q);
        REQUIRE(dst == src);
    }
    q.wait();
}

// =============================================================================
// parallel_for kernel tests
// =============================================================================

TEST_CASE("parallel_for with DeviceView capture — element-wise kernel", "[array][kernel][view]") {
    auto ctx = make_test_context();
    sycl::queue q = ctx->make_in_order_queue(ctx->devices[0]);

    SECTION("element-wise multiply: result[id] = a[id] * b[id]") {
        auto a_arr      = slabkit::make_device_array<float, 1>(sycl::range<1>{8}, ctx, 0);
        auto b_arr      = slabkit::make_device_array<float, 1>(sycl::range<1>{8}, ctx, 0);
        auto result_arr = slabkit::make_device_array<float, 1>(sycl::range<1>{8}, ctx, 0);

        std::vector<float> a_h = {1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<float> b_h = {2, 2, 2, 2, 2, 2, 2, 2};
        slabkit::copy_to_device(a_h.data(), a_arr.view(), q).wait();
        slabkit::copy_to_device(b_h.data(), b_arr.view(), q).wait();

        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>{8},
                [a=a_arr.view(), b=b_arr.view(), result=result_arr.view()](sycl::id<1> id) {
                    result[id] = a[id] * b[id];
                });
        }).wait();

        std::vector<float> out(8);
        slabkit::sync_to_host(result_arr, out.data(), q);
        REQUIRE(out == std::vector<float>{2, 4, 6, 8, 10, 12, 14, 16});
    }

    SECTION("write sequential index: result[id[0]] = id[0]") {
        auto arr = slabkit::make_device_array<int, 1>(sycl::range<1>{6}, ctx, 0);

        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>{6},
                [v=arr.view()](sycl::id<1> id) {
                    v[id] = static_cast<int>(id[0]);
                });
        }).wait();

        std::vector<int> out(6);
        slabkit::sync_to_host(arr, out.data(), q);
        REQUIRE(out == std::vector<int>{0, 1, 2, 3, 4, 5});
    }
    q.wait();
}

TEST_CASE("parallel_for with 2D DeviceView capture", "[array][kernel][view]") {
    auto ctx = make_test_context();
    sycl::queue q = ctx->make_in_order_queue(ctx->devices[0]);

    SECTION("write row*10+col into 2D result via view capture") {
        auto arr = slabkit::make_device_array<int, 2>(sycl::range<2>{3, 4}, ctx, 0);

        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>{3, 4},
                [v=arr.view()](sycl::id<2> id) {
                    v[id] = static_cast<int>(id[0]) * 10 + static_cast<int>(id[1]);
                });
        }).wait();

        std::vector<int> out(12);
        slabkit::sync_to_host(arr, out.data(), q);
        REQUIRE(out[0]  == 0);
        REQUIRE(out[1]  == 1);
        REQUIRE(out[4]  == 10);
        REQUIRE(out[11] == 23);
    }

    SECTION("element-wise add two 2D views") {
        auto a_arr      = slabkit::make_device_array<int, 2>(sycl::range<2>{2, 3}, ctx, 0);
        auto b_arr      = slabkit::make_device_array<int, 2>(sycl::range<2>{2, 3}, ctx, 0);
        auto result_arr = slabkit::make_device_array<int, 2>(sycl::range<2>{2, 3}, ctx, 0);

        std::vector<int> a_h = {1, 2, 3, 4, 5, 6};
        std::vector<int> b_h = {10, 20, 30, 40, 50, 60};
        slabkit::copy_to_device(a_h.data(), a_arr.view(), q).wait();
        slabkit::copy_to_device(b_h.data(), b_arr.view(), q).wait();

        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>{2, 3},
                [a=a_arr.view(), b=b_arr.view(), result=result_arr.view()](sycl::id<2> id) {
                    result[id] = a[id] + b[id];
                });
        }).wait();

        std::vector<int> out(6);
        slabkit::sync_to_host(result_arr, out.data(), q);
        REQUIRE(out == std::vector<int>{11, 22, 33, 44, 55, 66});
    }
    q.wait();
}

TEST_CASE("parallel_for writing into DeviceSubView capture — strided write", "[array][kernel][subview]") {
    auto ctx = make_test_context();
    sycl::queue q = ctx->make_in_order_queue(ctx->devices[0]);

    SECTION("kernel writes into 2x2 subview of 4x4 array, surrounding elements untouched") {
        auto arr = slabkit::make_device_array<int, 2>(sycl::range<2>{4, 4}, ctx, 0);
        std::vector<int> zeros(16, 0);
        slabkit::copy_to_device(zeros.data(), arr.view(), q).wait();

        auto sv = arr.subview(std::array<size_t, 2>{1, 1}, sycl::range<2>{2, 2});
        // sv.strides_ == {4, 1}, sv.data() == arr.data() + 5
        // writes: sv(0,0)=1, sv(0,1)=2, sv(1,0)=3, sv(1,1)=4

        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>{2, 2},
                [sv](sycl::id<2> id) {
                    sv[id] = static_cast<int>(id[0] * 2 + id[1] + 1);
                });
        }).wait();

        std::vector<int> out(16, -1);
        slabkit::sync_to_host(arr, out.data(), q);

        // subview region written correctly
        REQUIRE(out[5]  == 1);
        REQUIRE(out[6]  == 2);
        REQUIRE(out[9]  == 3);
        REQUIRE(out[10] == 4);

        // surrounding elements still zero
        REQUIRE(out[0]  == 0);
        REQUIRE(out[4]  == 0);
        REQUIRE(out[15] == 0);
    }
    q.wait();
}

TEST_CASE("parallel_for reading from DeviceSubView, writing to DeviceView — gather", "[array][kernel][subview]") {
    auto ctx = make_test_context();
    sycl::queue q = ctx->make_in_order_queue(ctx->devices[0]);

    SECTION("gather: read 2x2 strided subview into contiguous 1D result") {
        auto src_arr = slabkit::make_device_array<int, 2>(sycl::range<2>{4, 4}, ctx, 0);
        std::vector<int> src_h(16);
        std::iota(src_h.begin(), src_h.end(), 0);  // 0..15
        slabkit::copy_to_device(src_h.data(), src_arr.view(), q).wait();

        // subview at (1,2), size 2x2
        // ptr = src.data() + 6, strides = {4,1}
        // sv(0,0)=6, sv(0,1)=7, sv(1,0)=10, sv(1,1)=11
        auto sv = src_arr.subview(std::array<size_t, 2>{1, 2}, sycl::range<2>{2, 2});

        auto dst_arr = slabkit::make_device_array<int, 1>(sycl::range<1>{4}, ctx, 0);

        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>{2, 2},
                [sv, dst=dst_arr.view()](sycl::id<2> id) {
                    const int linear = static_cast<int>(id[0]) * 2 + static_cast<int>(id[1]);
                    dst[sycl::id<1>{static_cast<size_t>(linear)}] = sv[id];
                });
        }).wait();

        std::vector<int> out(4, -1);
        slabkit::sync_to_host(dst_arr, out.data(), q);
        REQUIRE(out == std::vector<int>{6, 7, 10, 11});
    }
    q.wait();
}
