// =============================================================================
// test_usm_vector.cpp — a value-semantic array whose every mutation is enqueued
// =============================================================================

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

#include "naga/slabkit/slabkit.hpp"
#include "support.hpp"

using namespace naga::slabkit;
namespace sk = naga::slabkit;
using slabkit_test::Fixture;

namespace {

allocator<int> alloc_of(const Fixture& f, kind k) {
    return k == kind::host ? allocator<int>{f.q, k} : allocator<int>{f.q, k, f.dev};
}

// ---- a controllable outside event --------------------------------------------
//
// A kernel on the fixture's RAW sycl queue -- outside the slabkit stream, whose
// memory tasks would otherwise be ordered after any kernel on it -- that spins
// until the host flips a flag in host USM, then writes 100 over `n` ints at
// `target`. Its event is the "outside work". Declare a Gate AFTER the containers
// it is registered on: it is destroyed first and opens itself, so a failing
// REQUIRE cannot leave a container's destructor waiting on a gate nobody opens.
class Gate {
public:
    sycl::event event;

    explicit Gate(Fixture& f, int* target = nullptr, std::size_t n = 0)
        : f_{f}, flag_{sycl::malloc_host<int>(1, f.ctx)} {
        REQUIRE(flag_ != nullptr);
        *flag_ = 0;
        int* flag = flag_;
        event     = f_.sq.single_task([=] {
            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::system> a{
                *flag};
            while (a.load() == 0) {
            }
            for (std::size_t i = 0; i < n; ++i) target[i] = 100;
        });
    }

    Gate(const Gate&)            = delete;
    Gate& operator=(const Gate&) = delete;

    void open() { std::atomic_ref<int>{*flag_}.store(1); }

    ~Gate() {
        if (flag_ == nullptr) return;
        open();
        event.wait();
        sycl::free(flag_, f_.ctx);
    }

    std::span<const sycl::event> as_deps() const { return std::span<const sycl::event>{&event, 1}; }

private:
    Fixture& f_;
    int*     flag_;
};

bool is_done(const sycl::event& e) {
    return e.get_info<sycl::info::event::command_execution_status>() ==
           sycl::info::event_command_status::complete;
}

/// Long enough that an operation which does not wait for the gate would have
/// finished by now.
void settle() { std::this_thread::sleep_for(std::chrono::milliseconds{150}); }

}  // namespace

TEST_CASE("construction fills, copies and value-initializes", "[usm_vector]") {
    Fixture f;

    for (kind k : {kind::device, kind::shared, kind::host}) {
        usm_vector<int> empty{alloc_of(f, k)};
        REQUIRE(empty.empty());
        REQUIRE(empty.size() == 0);
        REQUIRE(empty.get_kind() == k);

        usm_vector<int> zeros{6, alloc_of(f, k)};
        REQUIRE(zeros.size() == 6);
        for (int v : zeros.host_view()) REQUIRE(v == 0);

        usm_vector<int> filled{6, 7, alloc_of(f, k)};
        for (int v : filled.host_view()) REQUIRE(v == 7);

        std::vector<int> src{1, 2, 3, 4};
        usm_vector<int>  copied{std::span<const int>{src}, alloc_of(f, k)};
        REQUIRE(copied.size() == 4);
        const auto view = copied.host_view();
        for (std::size_t i = 0; i < src.size(); ++i) REQUIRE(view[i] == src[i]);

        usm_vector<int> listed{{9, 8, 7}, alloc_of(f, k)};
        REQUIRE(listed.size() == 3);
        REQUIRE(listed.host_view()[0] == 9);
    }
}

TEST_CASE("get_access refuses a device-kind container and serves the others", "[usm_vector]") {
    Fixture f;

    usm_vector<int> device{4, 1, alloc_of(f, kind::device)};
    REQUIRE_THROWS_AS(device.get_access(), std::invalid_argument);

    usm_vector<int> shared{4, 1, alloc_of(f, kind::shared)};
    // Host access to shared memory: on CUDA under WSL2 (no concurrent managed
    // access) the host may not touch shared memory while anything on the
    // device is still in flight, and get_access() waits for this container's
    // own work only. Wait the stream first (README, "Platform notes").
    f.q.wait();
    auto            access = shared.get_access();
    access[0]              = 42;
    REQUIRE(shared.host_view()[0] == 42);

    usm_vector<int> host{4, 1, alloc_of(f, kind::host)};
    host.get_access()[3] = 11;
    REQUIRE(host.host_view()[3] == 11);
}

TEST_CASE("host_view of a device container is an owned copy", "[usm_vector]") {
    Fixture         f;
    usm_vector<int> v{5, 3, alloc_of(f, kind::device)};

    const auto view = v.host_view();
    REQUIRE(view.size() == 5);
    REQUIRE(view.data() != v.data());  // a copy, not the device pointer
    for (int x : view) REQUIRE(x == 3);
}

TEST_CASE("assignment copies across kinds on one stream", "[usm_vector]") {
    Fixture f;

    usm_vector<int> device{8, 5, alloc_of(f, kind::device)};
    usm_vector<int> shared{alloc_of(f, kind::shared)};

    shared = device;  // a routed copy, device kind -> shared kind
    REQUIRE(shared.size() == 8);
    for (int v : shared.host_view()) REQUIRE(v == 5);

    usm_vector<int> host{2, 0, alloc_of(f, kind::host)};
    host = shared;  // shared -> host, and a reallocation on the way
    REQUIRE(host.size() == 8);
    for (int v : host.host_view()) REQUIRE(v == 5);
}

TEST_CASE("copy construction into another placement is a routed copy", "[usm_vector]") {
    Fixture f;

    usm_vector<int> device{4, 6, alloc_of(f, kind::device)};
    usm_vector<int> host{device, alloc_of(f, kind::host)};

    REQUIRE(host.get_kind() == kind::host);
    for (int v : host.host_view()) REQUIRE(v == 6);
}

TEST_CASE("assignment across devices moves the right bytes", "[usm_vector]") {
    context ctx{slabkit_test::all_devices()};
    queue   q{ctx};
    if (ctx.devices().size() < 2) return;

    const sycl::device a = ctx.devices()[0];
    const sycl::device b = ctx.devices()[1];

    usm_vector<int> on_a{8, 4, allocator<int>{q, kind::device, a}};
    usm_vector<int> on_b{allocator<int>{q, kind::device, b}};

    INFO("a = " << a.get_info<sycl::info::device::name>());
    INFO("b = " << b.get_info<sycl::info::device::name>());
    for (int v : on_a.host_view()) REQUIRE(v == 4);  // the source is right first

    on_b = on_a;
    REQUIRE(on_b.size() == 8);
    for (int v : on_b.host_view()) REQUIRE(v == 4);
}

TEST_CASE("a container refuses to work with another stream", "[usm_vector]") {
    Fixture f;
    context other_ctx{f.dev};
    queue   other_q{other_ctx};

    usm_vector<int> mine{4, 1, alloc_of(f, kind::shared)};
    usm_vector<int> theirs{allocator<int>{other_q, kind::shared, f.dev}};

    REQUIRE_THROWS_AS(theirs = mine, std::invalid_argument);
    REQUIRE_THROWS_AS((usm_vector<int>{mine, allocator<int>{other_q, kind::shared, f.dev}}),
                      std::invalid_argument);
}

TEST_CASE("a move between unequal allocators is refused", "[usm_vector]") {
    Fixture f;

    usm_vector<int> device{4, 1, alloc_of(f, kind::device)};
    usm_vector<int> shared{alloc_of(f, kind::shared)};

    REQUIRE_THROWS_AS(shared = std::move(device), std::invalid_argument);

    usm_vector<int> same{alloc_of(f, kind::shared)};
    usm_vector<int> source{4, 2, alloc_of(f, kind::shared)};
    same = std::move(source);
    REQUIRE(same.size() == 4);
    REQUIRE(source.size() == 0);
}

TEST_CASE("resize while a kernel holds the old pointer is safe", "[usm_vector]") {
    Fixture f;

    usm_vector<int> v{16, 1, alloc_of(f, kind::device)};
    int*            old = v.data();

    // A long kernel over the OLD storage, still running when resize reallocates.
    f.q.parallel_for(sycl::range<1>{16}, [=](sycl::id<1> i) {
        for (int spin = 0; spin < 500000; ++spin) old[i[0]] = old[i[0]] + 1;
    });

    v.resize(64, 9);
    REQUIRE(v.size() == 64);
    REQUIRE(v.capacity() >= 64);

    const auto view = v.host_view();
    // The first sixteen carry whatever the kernel left; the tail is the fill.
    for (std::size_t i = 16; i < 64; ++i) REQUIRE(view[i] == 9);
    for (std::size_t i = 0; i < 16; ++i) REQUIRE(view[i] == 500001);

    f.q.wait();
}

TEST_CASE("resize down keeps the head, and fill rewrites everything", "[usm_vector]") {
    Fixture         f;
    usm_vector<int> v{std::span<const int>{std::vector<int>{1, 2, 3, 4, 5}},
                      alloc_of(f, kind::shared)};

    v.resize(3);
    REQUIRE(v.size() == 3);
    REQUIRE(v.host_view()[2] == 3);

    v.fill(8);
    for (int x : v.host_view()) REQUIRE(x == 8);

    v.from_host(std::span<const int>{std::vector<int>{7, 7}});
    REQUIRE(v.size() == 2);
    REQUIRE(v.host_view()[1] == 7);
}

TEST_CASE("reserve and shrink_to_fit keep the elements", "[usm_vector]") {
    Fixture         f;
    usm_vector<int> v{4, 3, alloc_of(f, kind::shared)};

    v.reserve(32);
    REQUIRE(v.capacity() >= 32);
    REQUIRE(v.size() == 4);
    for (int x : v.host_view()) REQUIRE(x == 3);

    v.shrink_to_fit();
    REQUIRE(v.capacity() == 4);
    for (int x : v.host_view()) REQUIRE(x == 3);

    v.clear();
    REQUIRE(v.empty());
    f.q.wait();
}

TEST_CASE("two mutations of one container are ordered by its own last event",
          "[usm_vector]") {
    Fixture f;

    // Memory tasks are asynchronous with each other, so only the container's
    // self-edge orders these two. The first is large enough to still be running
    // when the second is submitted; without the edge some elements keep the 5.
    const std::size_t n = 1u << 20;
    usm_vector<int>   v{n, alloc_of(f, kind::shared)};

    v.fill(5);
    v.from_host(std::span<const int>{std::vector<int>(n, 7)});

    const auto view = v.host_view();
    REQUIRE(view.size() == n);
    for (std::size_t i = 0; i < n; ++i) REQUIRE(view[i] == 7);
}

TEST_CASE("a container syncs what it did, and the caller syncs what they did",
          "[usm_vector]") {
    Fixture         f;
    usm_vector<int> v{16, 1, alloc_of(f, kind::device)};

    // The caller's kernel is ordered after the container's fill by last_event.
    // The container never learns about that kernel, so sync() alone would not
    // wait for it: the caller's own wait is what makes the value readable.
    int* p = v.data();
    f.q.parallel_for(std::vector<sycl::event>{v.last_event()}, sycl::range<1>{16},
                     [=](sycl::id<1> i) { p[i[0]] = p[i[0]] * 10; });
    f.q.wait_and_throw();

    for (int x : v.host_view()) REQUIRE(x == 10);
}

TEST_CASE("a container's elements are visible to a kernel on its own stream",
          "[usm_vector]") {
    Fixture         f;
    usm_vector<int> v{16, 2, alloc_of(f, kind::device)};

    int* p = v.data();
    f.q.parallel_for(std::vector<sycl::event>{v.last_event()}, sycl::range<1>{16},
                     [=](sycl::id<1> i) { p[i[0]] = p[i[0]] * 3; });
    f.q.wait();

    for (int x : v.host_view()) REQUIRE(x == 6);
}

// -----------------------------------------------------------------------------
// Registered events: outside work the container orders itself after
// -----------------------------------------------------------------------------

TEST_CASE("a registered event delays a later fill, resize or copy-assign",
          "[usm_vector][events]") {
    Fixture         f;
    usm_vector<int> v{8, 1, alloc_of(f, kind::device)};
    v.reserve(64);  // room to grow in place: a reallocation frees, and the free waits
    v.sync();

    // In each case the gate's kernel overwrites the target with 100 once it is
    // opened, so a task that did not come after it leaves 100 behind.

    SECTION("fill") {
        Gate g{f, v.data(), 8};
        v.register_event(g.event);

        sycl::event e = v.fill(5);
        settle();
        REQUIRE_FALSE(is_done(e));
        REQUIRE_FALSE(is_done(g.event));

        g.open();
        e.wait();
        for (int x : v.host_view()) REQUIRE(x == 5);
    }

    SECTION("resize") {
        Gate g{f, v.data() + 8, 8};
        v.register_event(g.event);

        sycl::event e = v.resize(16, 9);
        settle();
        REQUIRE_FALSE(is_done(e));

        g.open();
        e.wait();
        const auto view = v.host_view();
        REQUIRE(view.size() == 16);
        for (std::size_t i = 0; i < 8; ++i) REQUIRE(view[i] == 1);
        for (std::size_t i = 8; i < 16; ++i) REQUIRE(view[i] == 9);
    }

    SECTION("copy-assign") {
        usm_vector<int> src{8, 3, alloc_of(f, kind::device)};
        src.sync();
        Gate g{f, v.data(), 8};
        v.register_event(g.event);

        v                   = src;
        sycl::event e = v.last_event();
        settle();
        REQUIRE_FALSE(is_done(e));

        g.open();
        e.wait();
        for (int x : v.host_view()) REQUIRE(x == 3);
    }
}

TEST_CASE("the free waits on registered events", "[usm_vector][events]") {
    Fixture f;
    auto    v = std::make_unique<usm_vector<int>>(8, 1, alloc_of(f, kind::device));
    v->sync();
    Gate g{f};
    v->register_event(g.event);

    // The destructor blocks the calling thread, so it runs on its own; the
    // gate opens only after we have seen it still waiting.
    std::atomic<bool> freed{false};
    std::thread       t{[&] {
        v.reset();
        freed = true;
    }};
    settle();
    const bool freed_early = freed.load();
    g.open();
    t.join();

    REQUIRE_FALSE(freed_early);
    REQUIRE(freed.load());
    REQUIRE(is_done(g.event));
}

TEST_CASE("copy-assign from a source with registered events waits on them",
          "[usm_vector][events]") {
    Fixture         f;
    usm_vector<int> src{8, 3, alloc_of(f, kind::device)};
    usm_vector<int> dst{8, 0, alloc_of(f, kind::device)};
    src.sync();
    dst.sync();

    // The gate rewrites the SOURCE to 100 after it opens. The copy reads the
    // source, so it must come after that: the destination ends up 100, not 3.
    Gate g{f, src.data(), 8};
    src.register_event(g.event);
    REQUIRE(src.registered_event_count() == 1);

    dst                 = src;
    sycl::event e = dst.last_event();
    settle();
    REQUIRE_FALSE(is_done(e));
    REQUIRE(src.registered_event_count() == 0);  // the copy is now the source's last event

    g.open();
    e.wait();
    for (int x : dst.host_view()) REQUIRE(x == 100);
}

TEST_CASE("sync covers registered events", "[usm_vector][events]") {
    Fixture         f;
    usm_vector<int> v{8, 1, alloc_of(f, kind::device)};
    v.sync();
    Gate g{f};
    v.register_event(g.event);

    std::atomic<bool> synced{false};
    std::thread       t{[&] {
        v.sync();
        synced = true;
    }};
    settle();
    const bool synced_early = synced.load();
    g.open();
    t.join();

    REQUIRE_FALSE(synced_early);
    REQUIRE(synced.load());
    REQUIRE(is_done(g.event));
}

TEST_CASE("the extra-dependency overloads order after the given events",
          "[usm_vector][events]") {
    Fixture         f;
    usm_vector<int> v{8, 1, alloc_of(f, kind::device)};
    v.reserve(64);
    v.sync();

    // Nothing is registered here: the events are passed to the one call, which
    // is ordered after them and does not keep them.

    SECTION("fill(value, deps)") {
        Gate g{f, v.data(), 8};
        sycl::event e = v.fill(5, g.as_deps());
        settle();
        REQUIRE_FALSE(is_done(e));
        REQUIRE(v.registered_event_count() == 0);

        g.open();
        e.wait();
        for (int x : v.host_view()) REQUIRE(x == 5);
    }

    SECTION("resize(n, deps)") {
        Gate g{f, v.data() + 8, 8};
        sycl::event e = v.resize(16, g.as_deps());
        settle();
        REQUIRE_FALSE(is_done(e));

        g.open();
        e.wait();
        const auto view = v.host_view();
        REQUIRE(view.size() == 16);
        for (std::size_t i = 8; i < 16; ++i) REQUIRE(view[i] == 0);
    }

    SECTION("resize(n, value, deps)") {
        Gate g{f, v.data() + 8, 8};
        sycl::event e = v.resize(16, 9, g.as_deps());
        settle();
        REQUIRE_FALSE(is_done(e));

        g.open();
        e.wait();
        const auto view = v.host_view();
        REQUIRE(view.size() == 16);
        for (std::size_t i = 8; i < 16; ++i) REQUIRE(view[i] == 9);
    }

    SECTION("from_host(src, deps)") {
        Gate                   g{f, v.data(), 8};
        const std::vector<int> src(8, 4);
        sycl::event e = v.from_host(std::span<const int>{src}, g.as_deps());
        settle();
        REQUIRE_FALSE(is_done(e));

        g.open();
        e.wait();
        for (int x : v.host_view()) REQUIRE(x == 4);
    }

    SECTION("assign(other, deps)") {
        usm_vector<int> src{8, 3, alloc_of(f, kind::device)};
        src.sync();
        Gate g{f, v.data(), 8};
        sycl::event e = v.assign(src, g.as_deps());
        settle();
        REQUIRE_FALSE(is_done(e));
        REQUIRE(e == v.last_event());

        g.open();
        e.wait();
        for (int x : v.host_view()) REQUIRE(x == 3);
    }
}

TEST_CASE("completed events are pruned on registration", "[usm_vector][events]") {
    Fixture         f;
    usm_vector<int> v{8, 1, alloc_of(f, kind::device)};
    usm_vector<int> moved{alloc_of(f, kind::device)};  // before the gates, so they outlive it
    v.sync();

    // Completed events, made BEFORE any gate: a kernel submitted to the raw
    // queue while a gate spins on it would queue behind the gate.
    std::vector<sycl::event> finished;
    for (int i = 0; i < 5; ++i) {
        finished.push_back(f.sq.single_task([] {}));
        finished.back().wait();
    }

    REQUIRE(v.registered_event_count() == 0);
    v.register_event(finished[0]);
    REQUIRE(v.registered_event_count() == 1);

    // The completed one is dropped first, then the new one is appended.
    v.register_event(finished[1]);
    REQUIRE(v.registered_event_count() == 1);

    // A pending event survives later registrations; completed ones do not.
    Gate g{f};
    v.register_event(g.event);
    REQUIRE(v.registered_event_count() == 1);  // the completed one was dropped
    v.register_event(finished[2]);
    REQUIRE(v.registered_event_count() == 2);  // gate + the new one
    v.register_events(std::span<const sycl::event>{finished.data() + 3, 2});
    REQUIRE(v.registered_event_count() == 3);  // gate + the two new ones; the old one dropped

    // A mutation depends on all of them, so it forgets them.
    sycl::event e = v.fill(2);
    REQUIRE(v.registered_event_count() == 0);
    settle();
    REQUIRE_FALSE(is_done(e));

    // Registered events travel with a move.
    Gate g2{f};
    v.register_event(g2.event);
    moved = std::move(v);
    REQUIRE(moved.registered_event_count() == 1);
    REQUIRE(v.registered_event_count() == 0);

    g.open();
    g2.open();
    moved.sync();
}
