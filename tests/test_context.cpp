#include <catch2/catch_test_macros.hpp>
#include <sycl/sycl.hpp>

#include "naga/slabkit/context.hpp"

// =============================================================================
// AllocationError
// =============================================================================

TEST_CASE("AllocationError carries reason", "[context][error]") {
    SECTION("PoolExhausted reason is retrievable") {
        const naga::slabkit::AllocationError err{
            naga::slabkit::AllocationError::Reason::PoolExhausted, "pool is full"};
        REQUIRE(err.reason() == naga::slabkit::AllocationError::Reason::PoolExhausted);
    }

    SECTION("Fragmented reason is retrievable") {
        const naga::slabkit::AllocationError err{
            naga::slabkit::AllocationError::Reason::Fragmented, "fragmented"};
        REQUIRE(err.reason() == naga::slabkit::AllocationError::Reason::Fragmented);
    }

    SECTION("ScratchFull reason is retrievable") {
        const naga::slabkit::AllocationError err{naga::slabkit::AllocationError::Reason::ScratchFull,
                                           "scratch full"};
        REQUIRE(err.reason() == naga::slabkit::AllocationError::Reason::ScratchFull);
    }
}

TEST_CASE("AllocationError what() returns non-empty string", "[context][error]") {
    SECTION("with non-empty message") {
        const naga::slabkit::AllocationError err{
            naga::slabkit::AllocationError::Reason::PoolExhausted, "custom error message"};
        const char* msg = err.what();
        REQUIRE(msg != nullptr);
        REQUIRE(std::string_view(msg).length() > 0);
    }

    SECTION("with empty message") {
        const naga::slabkit::AllocationError err{naga::slabkit::AllocationError::Reason::PoolExhausted, ""};
        const char* msg = err.what();
        REQUIRE(msg != nullptr);
        // Should still be valid C string, though may be empty
    }
}

// =============================================================================
// ContextBuilder and Context integration
// =============================================================================

TEST_CASE("ContextBuilder builds valid context for CPU device", "[context][builder]") {
    SECTION("single CPU device") {
        sycl::device dev{sycl::cpu_selector_v};
        auto ctx = naga::slabkit::ContextBuilder()
                       .devices({dev})
                       .node_size(256)
                       .slab_size_per_device(1024 * 1024)
                       .build();
        REQUIRE(ctx->devices.size() == 1);
        REQUIRE(ctx->platform_contexts.size() == 1);
        REQUIRE(ctx->device_to_platform.size() == 1);
        REQUIRE(ctx->device_to_platform[0] == 0);
        REQUIRE(ctx->device_slab_sizes.size() == 1);
        REQUIRE(ctx->device_slab_sizes[0] == 1024 * 1024);
    }
}

TEST_CASE("ContextBuilder throws if device list is empty", "[context][builder]") {
    SECTION("empty device list throws std::invalid_argument") {
        REQUIRE_THROWS_AS(naga::slabkit::ContextBuilder()
                              .devices({})
                              .slab_size_per_device(1024 * 1024)
                              .build(),
                          std::invalid_argument);
    }
}

// =============================================================================
// Context device_index lookup
// =============================================================================

TEST_CASE("Context device_index lookup", "[context][slab_context]") {
    SECTION("finds registered device by exact match") {
        sycl::device dev{sycl::cpu_selector_v};

        auto ctx = naga::slabkit::ContextBuilder()
                       .devices({dev})
                       .slab_size_per_device(1024 * 1024)
                       .build();

        size_t idx = ctx->device_index(dev);
        REQUIRE(idx == 0);
    }
}

TEST_CASE("Context device_index throws for unknown device", "[context][slab_context]") {
    SECTION("pool_at throws out_of_range for invalid index") {
        sycl::device dev{sycl::cpu_selector_v};

        auto ctx = naga::slabkit::ContextBuilder()
                       .devices({dev})
                       .slab_size_per_device(1024 * 1024)
                       .build();

        // Out-of-range index must throw
        REQUIRE_THROWS_AS(ctx->pool_at(99), std::out_of_range);
        REQUIRE_THROWS_AS(ctx->context_for(size_t(99)), std::out_of_range);
    }
}

// =============================================================================
// Context pool_at and pool_for
// =============================================================================

TEST_CASE("Context pool_at same index returns same pool", "[context][slab_context]") {
    SECTION("accessing same pool twice returns same reference") {
        sycl::device dev{sycl::cpu_selector_v};

        auto ctx = naga::slabkit::ContextBuilder()
                       .devices({dev})
                       .node_size(256)
                       .slab_size_per_device(1024 * 1024)
                       .build();

        auto& pool1 = ctx->pool_at(0);
        auto& pool2 = ctx->pool_at(0);
        REQUIRE(&pool1 == &pool2);
    }
}

TEST_CASE("Context pool_for lazily initialises pool", "[context][slab_context]") {
    SECTION("first call to pool_for initialises; second returns same") {
        sycl::device dev{sycl::cpu_selector_v};

        auto ctx = naga::slabkit::ContextBuilder()
                       .devices({dev})
                       .node_size(256)
                       .slab_size_per_device(1024 * 1024)
                       .build();

        auto& pool1 = ctx->pool_for(dev);
        auto& pool2 = ctx->pool_for(dev);
        REQUIRE(&pool1 == &pool2);
    }
}

// =============================================================================
// Context context_for
// =============================================================================

TEST_CASE("Context context_for returns correct platform context", "[context][slab_context]") {
    SECTION("by device index") {
        sycl::device dev{sycl::cpu_selector_v};

        auto ctx = naga::slabkit::ContextBuilder()
                       .devices({dev})
                       .slab_size_per_device(1024 * 1024)
                       .build();

        const auto& retrieved_ctx = ctx->context_for(size_t(0));
        // Verify the context has at least one device (is a valid SYCL context)
        REQUIRE(!retrieved_ctx.get_devices().empty());
    }

    SECTION("by device") {
        sycl::device dev{sycl::cpu_selector_v};

        auto ctx = naga::slabkit::ContextBuilder()
                       .devices({dev})
                       .slab_size_per_device(1024 * 1024)
                       .build();

        const auto& retrieved_ctx = ctx->context_for(dev);
        REQUIRE(!retrieved_ctx.get_devices().empty());
    }
}

// =============================================================================
// Context queue factories
// =============================================================================

TEST_CASE("Context make_in_order_queue returns usable queue", "[context][slab_context]") {
    sycl::device dev{sycl::cpu_selector_v};

    auto ctx = naga::slabkit::ContextBuilder()
                   .devices({dev})
                   .slab_size_per_device(1024 * 1024)
                   .build();

    auto q = ctx->make_in_order_queue(dev);
    REQUIRE(q.get_device() == dev);
}

TEST_CASE("Context make_out_of_order_queue returns usable queue", "[context][slab_context]") {
    sycl::device dev{sycl::cpu_selector_v};

    auto ctx = naga::slabkit::ContextBuilder()
                   .devices({dev})
                   .slab_size_per_device(1024 * 1024)
                   .build();

    auto q = ctx->make_out_of_order_queue(dev);
    REQUIRE(q.get_device() == dev);
}

TEST_CASE("Context make_queue_with_props returns usable queue", "[context][slab_context]") {
    sycl::device dev{sycl::cpu_selector_v};

    auto ctx = naga::slabkit::ContextBuilder()
                   .devices({dev})
                   .slab_size_per_device(1024 * 1024)
                   .build();

    sycl::property_list props{sycl::property::queue::in_order()};
    auto q = ctx->make_queue_with_props(dev, props);
    REQUIRE(q.get_device() == dev);
}
