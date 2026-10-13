// =============================================================================
// test_allocator.cpp — one allocator over all kinds and routes
// =============================================================================

#include <catch2/catch_test_macros.hpp>

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

TEST_CASE("a zero-byte pool serves nothing: every allocation falls back to the runtime and is counted", "[allocator]") {
    Fixture f;

    DevicePool pool{f.q, kind::device, 0, 256};
    REQUIRE(pool.slot_count() == 0);
    REQUIRE(pool.free_count() == 0);
    allocator<std::byte> alloc{f.q, pool};

    std::byte* a = alloc.allocate(256);
    std::byte* b = alloc.allocate(100);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(pool.outstanding() == 0);
    REQUIRE(pool.fallback_allocations() == 2);
    REQUIRE(pool.fallback_bytes_in_use() == 356);

    alloc.deallocate(a, 256);
    alloc.deallocate(b, 100);
    REQUIRE(pool.fallback_bytes_in_use() == 0);
    REQUIRE(pool.fallback_allocations() == 2);  // cumulative
    f.q.wait();
}

TEST_CASE("deallocate with dependencies waits for them, then frees before returning",
          "[allocator]") {
    Fixture f;

    DevicePool           pool{f.q, kind::device, 4096, 256};
    allocator<std::byte> alloc{f.q, pool};

    std::byte* p = alloc.allocate(256);
    REQUIRE(pool.outstanding() == 1);

    // Something long enough that a free which did not wait would be a use-after-free.
    // Atomic, so the compiler cannot fold the loop into one add.
    int* busy = sk::malloc_device<int>(1, f.q).release();
    sycl::event e = f.q.single_task([=] {
        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> a{busy[0]};
        for (int spin = 0; spin < 150000000; ++spin) a.fetch_add(1);
    });

    alloc.deallocate(p, 256, std::vector<sycl::event>{e});

    // The call waited for the kernel, then freed on this thread.
    REQUIRE(e.get_info<sycl::info::event::command_execution_status>() ==
            sycl::info::event_command_status::complete);
    REQUIRE(pool.outstanding() == 0);  // no wait before this

    f.q.wait();
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

    // Frees are synchronous; nothing is left to drain.
    f.q.wait();
}

TEST_CASE("an OS-route free on the calling thread, with other kernels in flight, neither hangs nor corrupts",
          "[allocator]") {
    Fixture f;

    constexpr std::size_t big   = std::size_t{1} << 20;
    constexpr std::size_t small = 1024;
    constexpr int         reps  = 200;

    // The long-lived vector: every iteration's second kernel adds 1 to each element.
    usm_vector<float> steady{small, 0.0f, allocator<float>{f.q, kind::device, f.dev}};
    f.q.wait();
    float* keep = steady.data();

    for (int i = 0; i < reps; ++i) {
        {
            usm_vector<float> scratch{big, allocator<float>{f.q, kind::device, f.dev}};
            float*            tmp = scratch.data();
            sycl::event       k   = f.q.parallel_for(
                scratch.read_deps(), sycl::range<1>{big},
                [=](sycl::id<1> j) { tmp[j[0]] = static_cast<float>(j[0]); });
            k.wait();  // the caller waits for what they did
        }  // the OS-route free happens here, on this thread
        // Not waited: device work is in flight during the next iteration's free.
        f.q.parallel_for(sycl::range<1>{small}, [=](sycl::id<1> j) { keep[j[0]] += 1.0f; });
    }
    f.q.wait();

    const auto values = slabkit_test::read_back(f.q, keep, small);
    for (float v : values) REQUIRE(v == static_cast<float>(reps));
}

TEST_CASE("destroying a usm_vector returns its storage before the destructor returns",
          "[allocator]") {
    Fixture f;

    DevicePool pool{f.q, kind::device, 1024, 256};  // four slots of 256 bytes
    {
        usm_vector<float> a{64, 1.0f, allocator<float>{f.q, pool}};
        usm_vector<float> b{64, 2.0f, allocator<float>{f.q, pool}};
        usm_vector<float> c{64, 3.0f, allocator<float>{f.q, pool}};
        REQUIRE(pool.outstanding() == 3);
    }  // three synchronous frees

    REQUIRE(pool.outstanding() == 0);  // no f.q.wait() before this
    f.q.wait();
}

TEST_CASE("reserve frees the old storage once the copy that read it is done", "[allocator]") {
    Fixture f;

    DevicePool pool{f.q, kind::device, 4096, 256};

    usm_vector<float> v{32, 1.0f, allocator<float>{f.q, pool}};
    REQUIRE(pool.outstanding() == 1);

    v.reserve(64);
    REQUIRE(pool.outstanding() == 1);  // the old slot came back

    auto hv = v.host_view();
    REQUIRE(hv.size() == 32);
    for (float x : hv) REQUIRE(x == 1.0f);

    f.q.wait();
}


TEST_CASE("host allocators are equal only within one SYCL context", "[allocator][cross-device]") {
    context ctx{slabkit_test::all_devices()};
    queue   q{ctx};

    // devices of one context: equal, however the device is named (host memory belongs to the context)
    const sycl::device first = ctx.devices().front();
    allocator<int>     a{q, kind::host, first};
    allocator<int>     b{q, kind::host, first};
    REQUIRE(a == b);
    for (const auto& d : ctx.devices())
        if (d != first && ctx.context_of(d) == ctx.context_of(first)) REQUIRE(a == allocator<int>{q, kind::host, d});

    // two backends: different contexts are unequal, and a move between them is refused
    sycl::device other = first;
    bool         have  = false;
    for (const auto& d : ctx.devices())
        if (!(d.get_platform() == first.get_platform())) {
            other = d;
            have  = true;
        }
    if (!have) SKIP("needs two backends (two contexts)");

    allocator<int> c{q, kind::host, other};
    REQUIRE_FALSE(a == c);

    usm_vector<int> mine{4, 1, a};
    usm_vector<int> theirs{c};
    REQUIRE_THROWS_AS(theirs = std::move(mine), std::invalid_argument);
}
