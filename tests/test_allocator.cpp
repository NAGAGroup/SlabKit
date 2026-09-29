// =============================================================================
// test_allocator.cpp — one allocator over all kinds and routes
// =============================================================================

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "naga/slabkit/slabkit.hpp"
#include "support.hpp"

using namespace naga::slabkit;
namespace sk = naga::slabkit;
using slabkit_test::Fixture;

TEST_CASE("an allocator reports its own state", "[allocator]") {
    Fixture f;

    allocator<int> os{f.q, kind::device, f.dev};
    REQUIRE(os.get_kind() == kind::device);
    REQUIRE(os.get_device() == f.dev);
    REQUIRE(os.get_queue() == f.q);

    allocator<int> host{f.q, kind::host};
    REQUIRE(host.get_kind() == kind::host);
    REQUIRE_THROWS_AS(host.get_device(), std::invalid_argument);
}

TEST_CASE("equality is state equality, and rebinding keeps state", "[allocator]") {
    Fixture f;

    allocator<int>   a{f.q, kind::device, f.dev};
    allocator<int>   b{f.q, kind::device, f.dev};
    allocator<int>   shared{f.q, kind::shared, f.dev};
    allocator<float> rebound{a};

    REQUIRE(a == b);
    REQUIRE(a == rebound);
    REQUIRE_FALSE(a == shared);

    DeviceArena    arena{f.q, kind::device, 4096};
    allocator<int> from_arena{f.q, arena};
    allocator<int> same_arena{f.q, arena};
    REQUIRE(from_arena == same_arena);
    REQUIRE_FALSE(from_arena == a);
}

TEST_CASE("allocate throws where the storage family returns null", "[allocator]") {
    Fixture f;

    DeviceArena    tiny{f.q, kind::device, 256};
    allocator<int> budget{f.q, tiny};

    REQUIRE(budget.allocate(16));                                   // 64 bytes: fits
    REQUIRE_THROWS_AS(budget.allocate(1024), std::bad_alloc);       // an arena never falls back
}

TEST_CASE("the pool route falls back to the runtime when the pool is exhausted",
          "[allocator]") {
    Fixture f;

    DevicePool     pool{f.q, kind::device, 1024, 256};  // four slots
    allocator<std::byte> alloc{f.q, pool};

    std::vector<std::byte*> taken;
    for (int i = 0; i < 4; ++i) taken.push_back(alloc.allocate(256));
    REQUIRE(pool.outstanding() == 4);

    // Exhausted: the next one comes from the runtime instead of throwing.
    std::byte* overflow = alloc.allocate(256);
    REQUIRE(overflow != nullptr);
    REQUIRE(pool.outstanding() == 4);

    for (auto* p : taken) alloc.deallocate(p, 256);
    alloc.deallocate(overflow, 256);
    f.q.wait();
}

TEST_CASE("deallocate returns immediately while a long kernel runs, and frees afterwards",
          "[allocator]") {
    Fixture f;

    DevicePool           pool{f.q, kind::device, 4096, 256};
    allocator<std::byte> alloc{f.q, pool};

    std::byte* p = alloc.allocate(256);
    REQUIRE(pool.outstanding() == 1);

    // Something long enough that an immediate free would be a use-after-free.
    // Atomic, so the compiler cannot fold the loop into one add.
    int* busy = sk::malloc_device<int>(1, f.q).release();
    f.q.single_task([=] {
        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> a{busy[0]};
        for (int spin = 0; spin < 150000000; ++spin) a.fetch_add(1);
    });

    const auto before = std::chrono::steady_clock::now();
    alloc.deallocate(p, 256);
    const auto elapsed = std::chrono::steady_clock::now() - before;

    // The call itself did not wait for the kernel.
    REQUIRE(elapsed < std::chrono::milliseconds{50});
    REQUIRE(pool.outstanding() == 1);  // still held: the release is posted, not done

    // Once the stream drains and the posted task has run, the slot comes back.
    f.q.wait();
    for (int i = 0; i < 200 && pool.outstanding() != 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds{10});
    REQUIRE(pool.outstanding() == 0);
}

TEST_CASE("std containers work over the host-accessible kinds", "[allocator]") {
    Fixture f;

    std::vector<int, allocator<int>> shared_vec{allocator<int>{f.q, kind::shared, f.dev}};
    shared_vec.push_back(1);
    shared_vec.push_back(2);
    shared_vec.push_back(3);
    REQUIRE(shared_vec.size() == 3);
    REQUIRE(shared_vec[2] == 3);

    std::vector<int, allocator<int>> host_vec{allocator<int>{f.q, kind::host}};
    host_vec.assign(4, 9);
    REQUIRE(host_vec.size() == 4);
    for (int v : host_vec) REQUIRE(v == 9);

    // The deferred releases these leave behind must all run.
    f.q.wait();
}
