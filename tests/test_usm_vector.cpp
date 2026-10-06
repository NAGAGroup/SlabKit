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

/// The outside task the tests order: *p = v, on the slabkit stream. A named
/// type, so every test launches the SAME kernel and warm_up can load it.
struct Poke {
    int* p;
    int  v;
    void operator()() const { *p = v; }
};

bool is_done(const sycl::event& e) {
    return e.get_info<sycl::info::event::command_execution_status>() ==
           sycl::info::event_command_status::complete;
}

/// Long enough that an operation which does not wait for the gate would have
/// finished by now.
void settle() { std::this_thread::sleep_for(std::chrono::milliseconds{150}); }

/// Polls instead of blocking, so a property that fails is reported, not hung.
bool wait_done(const sycl::event& e, std::chrono::milliseconds limit = std::chrono::seconds{10}) {
    const auto until = std::chrono::steady_clock::now() + limit;
    while (!is_done(e)) {
        if (std::chrono::steady_clock::now() > until) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    return true;
}

/// The reason the tests that need two kernels running at once give on OpenMP.
constexpr const char* kNoOverlap =
    "the OpenMP backend never runs two kernels at once, so unordered-ness can't be observed here";

bool is_openmp(const Fixture& f) { return f.dev.get_backend() == sycl::backend::omp; }

/// Launches every kernel a gating test uses once, with nothing held back, and
/// waits: the Gate, the Poke, and the empty task that joins an event list.
///
/// A kernel's FIRST launch JIT-compiles and loads its module, and on a device
/// backend that can block the submitting call behind work already running on
/// the device. A test that submits one while a Gate spins would then hang in
/// submit() instead of exercising ordering.
void warm_up(Fixture& f) {
    usm_vector<int> scratch{4, 0, alloc_of(f, kind::device)};
    {
        Gate g{f, scratch.data(), 4};
        g.open();
        g.event.wait();
        scratch.register_write(g.event);  // the write list now holds two events
    }
    scratch.sync();  // collapsing them launches the join task
    f.q.single_task(scratch.read_deps(), Poke{scratch.data(), 0}).wait();
    f.q.wait();
    f.sq.wait();
}

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
    // own lists only. Wait the stream first (README, "Platform notes").
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

TEST_CASE("two mutations of one container are ordered by its write list",
          "[usm_vector]") {
    Fixture f;

    // Memory tasks are asynchronous with each other, so only the container's
    // own writes waiting on its write list orders these two. The first is large
    // enough to still be running when the second is submitted; without the
    // edge some elements keep the 5.
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

    // The caller's kernel is ordered after the container's fill by read_deps
    // and handed back as a write. The container does not know about the kernel
    // until register_write, after which host_view() covers it.
    int*        p = v.data();
    sycl::event k = f.q.parallel_for(v.read_deps(), sycl::range<1>{16},
                                     [=](sycl::id<1> i) { p[i[0]] = p[i[0]] * 10; });
    v.register_write(k);
    f.q.wait_and_throw();

    for (int x : v.host_view()) REQUIRE(x == 10);
}

TEST_CASE("a container's elements are visible to a kernel on its own stream",
          "[usm_vector]") {
    Fixture         f;
    usm_vector<int> v{16, 2, alloc_of(f, kind::device)};

    int*        p = v.data();
    sycl::event k = f.q.parallel_for(v.read_deps(), sycl::range<1>{16},
                                     [=](sycl::id<1> i) { p[i[0]] = p[i[0]] * 3; });
    v.register_write(k);
    f.q.wait();

    for (int x : v.host_view()) REQUIRE(x == 6);
}

// -----------------------------------------------------------------------------
// Event lists: what a container waits on, and what an outside task waits on
// -----------------------------------------------------------------------------

TEST_CASE("outside readers do not wait on each other", "[usm_vector][events]") {
    Fixture f;
    if (is_openmp(f)) SKIP(kNoOverlap);
    warm_up(f);

    usm_vector<int> src{8, 3, alloc_of(f, kind::device)};
    usm_vector<int> dst{8, 0, alloc_of(f, kind::device)};
    src.sync();
    dst.sync();

    // Two readers of src still running: they are on its READ list.
    Gate g1{f};
    Gate g2{f};
    src.register_read(g1.event);
    src.register_read(g2.event);

    // What a reader waits on is the write list, which has only the fill.
    const std::vector<sycl::event> rd = src.read_deps();
    REQUIRE(rd.size() == 1);
    CHECK(is_done(rd.front()));

    // A third reader (the copy out of src) is not held back by the first two.
    sycl::event e = dst.assign(src, std::span<const sycl::event>{});
    const bool  finished     = wait_done(e);
    const bool  readers_open = !is_done(g1.event) && !is_done(g2.event);
    g1.open();
    g2.open();

    REQUIRE(finished);
    CHECK(readers_open);
    for (int x : dst.host_view()) REQUIRE(x == 3);
}

TEST_CASE("a writer waits for an in-flight reader", "[usm_vector][events]") {
    Fixture f;
    warm_up(f);

    usm_vector<int> v{8, 1, alloc_of(f, kind::device)};
    v.reserve(64);  // room to grow in place; the reallocation's free has happened by here
    v.sync();
    Gate g{f};
    v.register_read(g.event);

    SECTION("the container's own fill") {
        sycl::event e = v.fill(5);
        settle();
        REQUIRE_FALSE(is_done(e));

        g.open();
        REQUIRE(wait_done(e));
        for (int x : v.host_view()) REQUIRE(x == 5);
    }

    SECTION("the container's own resize") {
        sycl::event e = v.resize(16, 9);
        settle();
        REQUIRE_FALSE(is_done(e));

        g.open();
        REQUIRE(wait_done(e));
        const auto view = v.host_view();
        REQUIRE(view.size() == 16);
        for (std::size_t i = 0; i < 8; ++i) REQUIRE(view[i] == 1);
        for (std::size_t i = 8; i < 16; ++i) REQUIRE(view[i] == 9);
    }

    SECTION("the container's own from_host") {
        const std::vector<int> src(8, 4);
        sycl::event            e = v.from_host(std::span<const int>{src});
        settle();
        REQUIRE_FALSE(is_done(e));

        g.open();
        REQUIRE(wait_done(e));
        for (int x : v.host_view()) REQUIRE(x == 4);
    }

    SECTION("an outside writer given write_deps") {
        const std::vector<sycl::event> deps = v.write_deps();
        REQUIRE(deps.size() == 1);
        sycl::event e = f.q.single_task(deps, Poke{v.data(), 9});
        settle();
        REQUIRE_FALSE(is_done(e));

        g.open();
        REQUIRE(wait_done(e));
        v.register_write(e);
        const auto view = v.host_view();
        REQUIRE(view[0] == 9);
        for (std::size_t i = 1; i < 8; ++i) REQUIRE(view[i] == 1);
    }
}

TEST_CASE("outside writes do not wait on each other", "[usm_vector][events]") {
    Fixture f;
    if (is_openmp(f)) SKIP(kNoOverlap);
    warm_up(f);

    usm_vector<int> v{8, 1, alloc_of(f, kind::device)};
    v.sync();
    Gate g1{f};
    Gate g2{f};
    v.register_write(g1.event);
    v.register_write(g2.event);

    // A writer waits on the READ list (nothing has read) and on the container's own writes (the construction fill,
    // already finished after the sync): never on the outside writes registered above.
    const std::vector<sycl::event> deps = v.write_deps();
    REQUIRE(deps.size() == 1);
    REQUIRE(is_done(deps.front()));

    sycl::event e            = f.q.single_task(deps, Poke{v.data(), 9});
    const bool  finished     = wait_done(e);
    const bool  writers_open = !is_done(g1.event) && !is_done(g2.event);
    v.register_write(e);
    g1.open();
    g2.open();

    REQUIRE(finished);
    CHECK(writers_open);
    const auto view = v.host_view();
    REQUIRE(view[0] == 9);
    for (std::size_t i = 1; i < 8; ++i) REQUIRE(view[i] == 1);
}

TEST_CASE("an outside writer waits on the container's own writes", "[usm_vector][events]") {
    Fixture f;
    warm_up(f);

    usm_vector<int> v{8, 1, alloc_of(f, kind::device)};
    v.reserve(64);  // room to grow in place: a reallocation frees, and the free waits
    v.sync();
    // The gate holds the container's own write back (a write waits on the write list), so the write is still in flight
    // when the outside writer is ordered. Written 9 by the outside writer AFTER the own write: a writer that did not wait
    // for it would be overwritten by it.
    Gate g{f};
    v.register_write(g.event);

    SECTION("a fill") {
        sycl::event own = v.fill(5);
        const std::vector<sycl::event> deps = v.write_deps();
        REQUIRE_FALSE(deps.empty());
        sycl::event e = f.q.single_task(deps, Poke{v.data(), 9});
        settle();
        REQUIRE_FALSE(is_done(own));
        REQUIRE_FALSE(is_done(e));

        g.open();
        REQUIRE(wait_done(e));
        v.register_write(e);
        const auto view = v.host_view();
        REQUIRE(view[0] == 9);
        for (std::size_t i = 1; i < 8; ++i) REQUIRE(view[i] == 5);
    }

    SECTION("a resize's value-initializing fill of new elements") {
        sycl::event own = v.resize(16, 7);
        const std::vector<sycl::event> deps = v.write_deps();
        REQUIRE_FALSE(deps.empty());
        sycl::event e = f.q.single_task(deps, Poke{v.data() + 8, 9});
        settle();
        REQUIRE_FALSE(is_done(own));
        REQUIRE_FALSE(is_done(e));

        g.open();
        REQUIRE(wait_done(e));
        v.register_write(e);
        const auto view = v.host_view();
        REQUIRE(view.size() == 16);
        REQUIRE(view[8] == 9);
        for (std::size_t i = 9; i < 16; ++i) REQUIRE(view[i] == 7);
    }

    SECTION("a from_host") {
        const std::vector<int> src(8, 4);
        sycl::event            own = v.from_host(std::span<const int>{src});
        const std::vector<sycl::event> deps = v.write_deps();
        REQUIRE_FALSE(deps.empty());
        sycl::event e = f.q.single_task(deps, Poke{v.data(), 9});
        settle();
        REQUIRE_FALSE(is_done(own));
        REQUIRE_FALSE(is_done(e));

        g.open();
        REQUIRE(wait_done(e));
        v.register_write(e);
        const auto view = v.host_view();
        REQUIRE(view[0] == 9);
        for (std::size_t i = 1; i < 8; ++i) REQUIRE(view[i] == 4);
    }
}

TEST_CASE("a registered write delays a later fill, resize or copy-assign",
          "[usm_vector][events]") {
    Fixture f;
    warm_up(f);

    usm_vector<int> v{8, 1, alloc_of(f, kind::device)};
    v.reserve(64);  // room to grow in place: a reallocation frees, and the free waits
    v.sync();

    // In each case the gate's kernel overwrites the target with 100 once it is
    // opened, so a task that did not come after it leaves 100 behind.

    SECTION("fill") {
        Gate g{f, v.data(), 8};
        v.register_write(g.event);

        sycl::event e = v.fill(5);
        settle();
        REQUIRE_FALSE(is_done(e));
        REQUIRE_FALSE(is_done(g.event));

        g.open();
        REQUIRE(wait_done(e));
        for (int x : v.host_view()) REQUIRE(x == 5);
    }

    SECTION("resize") {
        Gate g{f, v.data() + 8, 8};
        v.register_write(g.event);

        sycl::event e = v.resize(16, 9);
        settle();
        REQUIRE_FALSE(is_done(e));

        g.open();
        REQUIRE(wait_done(e));
        const auto view = v.host_view();
        REQUIRE(view.size() == 16);
        for (std::size_t i = 0; i < 8; ++i) REQUIRE(view[i] == 1);
        for (std::size_t i = 8; i < 16; ++i) REQUIRE(view[i] == 9);
    }

    SECTION("copy-assign") {
        usm_vector<int> src{8, 3, alloc_of(f, kind::device)};
        src.sync();
        Gate g{f, v.data(), 8};
        v.register_write(g.event);

        sycl::event e = v.assign(src, std::span<const sycl::event>{});
        settle();
        REQUIRE_FALSE(is_done(e));

        g.open();
        REQUIRE(wait_done(e));
        for (int x : v.host_view()) REQUIRE(x == 3);
    }
}

TEST_CASE("assign orders against both containers", "[usm_vector][events]") {
    Fixture f;
    warm_up(f);

    usm_vector<int> src{8, 3, alloc_of(f, kind::device)};
    usm_vector<int> dst{8, 0, alloc_of(f, kind::device)};
    src.sync();
    dst.sync();

    SECTION("it waits on the source's writer, and is then a read of the source") {
        // The gate rewrites the SOURCE to 100 after it opens. The copy reads
        // the source, so it must come after that: the destination ends 100.
        Gate g{f, src.data(), 8};
        src.register_write(g.event);

        sycl::event e = dst.assign(src, std::span<const sycl::event>{});
        settle();
        REQUIRE_FALSE(is_done(e));

        // The copy is the source's only read, so a writer of src waits on it...
        // (and on src's own construction fill, listed after the read)
        const std::vector<sycl::event> wd = src.write_deps();
        REQUIRE(wd.size() == 2);
        CHECK(wd.front() == e);
        // ...and it is a write of dst, so a reader of dst waits on it too.
        const std::vector<sycl::event> rd = dst.read_deps();
        REQUIRE(rd.size() == 1);
        CHECK_FALSE(is_done(rd.front()));

        g.open();
        REQUIRE(wait_done(e));
        for (int x : dst.host_view()) REQUIRE(x == 100);
    }

    SECTION("it waits on the destination's reader") {
        Gate g{f};
        dst.register_read(g.event);

        sycl::event e = dst.assign(src, std::span<const sycl::event>{});
        settle();
        REQUIRE_FALSE(is_done(e));

        g.open();
        REQUIRE(wait_done(e));
        for (int x : dst.host_view()) REQUIRE(x == 3);
    }

    SECTION("it waits on the destination's writer") {
        // The gate rewrites the DESTINATION to 100 after it opens; the copy
        // comes after, so the destination ends 3.
        Gate g{f, dst.data(), 8};
        dst.register_write(g.event);

        sycl::event e = dst.assign(src, std::span<const sycl::event>{});
        settle();
        REQUIRE_FALSE(is_done(e));

        g.open();
        REQUIRE(wait_done(e));
        for (int x : dst.host_view()) REQUIRE(x == 3);
    }
}

TEST_CASE("a collapsed event covers every event it replaced", "[usm_vector][events]") {
    Fixture f;
    warm_up(f);

    usm_vector<int> v{8, 1, alloc_of(f, kind::device)};
    v.sync();
    Gate g1{f};
    Gate g2{f};
    Gate g3{f};

    SECTION("the write list, read through read_deps") {
        v.register_write(g1.event);
        v.register_write(g2.event);
        v.register_write(g3.event);

        const std::vector<sycl::event> deps = v.read_deps();
        REQUIRE(deps.size() == 1);
        const sycl::event joined = deps.front();
        // Read again, the list is already the one event: nothing new is joined.
        REQUIRE(v.read_deps().front() == joined);

        g1.open();
        g2.open();
        settle();
        REQUIRE_FALSE(is_done(joined));  // the third gate still holds it

        g3.open();
        REQUIRE(wait_done(joined));
        CHECK(is_done(g1.event));
        CHECK(is_done(g2.event));
        CHECK(is_done(g3.event));
    }

    SECTION("the read list, read through write_deps") {
        v.register_read(g1.event);
        v.register_read(g2.event);
        v.register_read(g3.event);

        // The joined read list, then the container's own construction fill.
        const std::vector<sycl::event> deps = v.write_deps();
        REQUIRE(deps.size() == 2);
        const sycl::event joined = deps.front();
        REQUIRE(v.write_deps().front() == joined);

        g1.open();
        g3.open();
        settle();
        REQUIRE_FALSE(is_done(joined));  // the second gate still holds it

        g2.open();
        REQUIRE(wait_done(joined));
    }

    SECTION("a task depending on the collapsed event waits for all of them") {
        v.register_write(g1.event);
        v.register_write(g2.event);
        v.register_write(g3.event);

        sycl::event e = f.q.single_task(v.read_deps(), Poke{v.data(), 9});
        g1.open();
        g2.open();
        settle();
        REQUIRE_FALSE(is_done(e));

        g3.open();
        REQUIRE(wait_done(e));
    }
}

TEST_CASE("a free waits on both lists", "[usm_vector][events]") {
    Fixture f;
    warm_up(f);

    auto v = std::make_unique<usm_vector<int>>(8, 1, alloc_of(f, kind::device));
    v->sync();
    Gate gw{f};
    Gate gr{f};
    v->register_write(gw.event);
    v->register_read(gr.event);

    // The destructor blocks the calling thread, so it runs on its own; the
    // gates open only after we have seen it still waiting.
    std::atomic<bool> freed{false};
    std::thread       t{[&] {
        v.reset();
        freed = true;
    }};
    settle();
    const bool freed_early = freed.load();
    gw.open();
    settle();
    const bool freed_on_write_only = freed.load();  // the reader still holds it
    gr.open();
    t.join();

    REQUIRE_FALSE(freed_early);
    REQUIRE_FALSE(freed_on_write_only);
    REQUIRE(freed.load());
    REQUIRE(is_done(gw.event));
    REQUIRE(is_done(gr.event));
}

TEST_CASE("sync waits on both lists", "[usm_vector][events]") {
    Fixture         f;
    usm_vector<int> v{8, 1, alloc_of(f, kind::device)};
    warm_up(f);
    v.sync();
    Gate gw{f};
    Gate gr{f};
    v.register_write(gw.event);
    v.register_read(gr.event);

    std::atomic<bool> synced{false};
    std::thread       t{[&] {
        v.sync();
        synced = true;
    }};
    settle();
    const bool synced_early = synced.load();
    gw.open();
    settle();
    const bool synced_on_write_only = synced.load();
    gr.open();
    t.join();

    REQUIRE_FALSE(synced_early);
    REQUIRE_FALSE(synced_on_write_only);
    REQUIRE(synced.load());
}

TEST_CASE("a copy out to the host waits on the write list and registers a read",
          "[usm_vector][events]") {
    Fixture f;
    warm_up(f);

    usm_vector<int> v{8, 1, alloc_of(f, kind::device)};
    v.sync();

    SECTION("it waits on a registered write") {
        Gate g{f, v.data(), 8};
        v.register_write(g.event);

        std::atomic<bool> returned{false};
        std::vector<int>  seen;
        std::thread       t{[&] {
            const auto view = v.host_view();
            seen.assign(view.begin(), view.end());
            returned = true;
        }};
        settle();
        const bool returned_early = returned.load();
        g.open();
        t.join();

        REQUIRE_FALSE(returned_early);
        REQUIRE(seen.size() == 8);
        for (int x : seen) REQUIRE(x == 100);

        // The copy is now a read: a writer of the container waits for it (and for its construction fill).
        REQUIRE(v.write_deps().size() == 2);
    }

    SECTION("it does not wait on a registered read") {
        if (is_openmp(f)) SKIP(kNoOverlap);
        Gate g{f};
        v.register_read(g.event);

        std::atomic<bool> returned{false};
        std::thread       t{[&] {
            const auto view = v.host_view();
            (void)view;
            returned = true;
        }};
        // Bounded: a copy that wrongly waited on the reader is reported, and the
        // gate opens either way so the thread can finish.
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds{10};
        while (!returned.load() && std::chrono::steady_clock::now() < until)
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        const bool returned_while_reader_ran = returned.load() && !is_done(g.event);
        g.open();
        t.join();

        REQUIRE(returned_while_reader_ran);
    }
}

TEST_CASE("the event lists travel with a move", "[usm_vector][events]") {
    Fixture         f;
    usm_vector<int> v{8, 1, alloc_of(f, kind::device)};
    usm_vector<int> moved{alloc_of(f, kind::device)};  // before the gate, so it outlives it
    warm_up(f);
    v.sync();
    Gate g{f};
    v.register_write(g.event);

    moved = std::move(v);
    REQUIRE(v.read_deps().empty());
    REQUIRE(v.write_deps().empty());

    sycl::event e = moved.fill(5);  // waits on the gate that came with the move
    settle();
    REQUIRE_FALSE(is_done(e));

    g.open();
    REQUIRE(wait_done(e));
}

TEST_CASE("a list is collapsed by the registration that makes it longer than 64 events",
          "[usm_vector][events]") {
    Fixture f;
    warm_up(f);

    usm_vector<int> v{8, 1, alloc_of(f, kind::device)};
    v.sync();

    // Completed events: nothing to wait for, so only the bookkeeping shows.
    std::vector<sycl::event> done;
    for (int i = 0; i < 70; ++i) {
        done.push_back(f.sq.single_task([] {}));
        done.back().wait();
    }
    const std::size_t cap = usm_vector<int>::max_listed_events;
    REQUIRE(cap == 64);

    SECTION("reads, one at a time") {
        REQUIRE(v.read_list_size() == 0);
        for (std::size_t i = 0; i < cap; ++i) v.register_read(done[i]);
        REQUIRE(v.read_list_size() == cap);  // at the cap: not yet collapsed
        v.register_read(done[cap]);
        REQUIRE(v.read_list_size() == 1);  // the 65th collapsed them all
        for (std::size_t i = cap + 1; i < done.size(); ++i) v.register_read(done[i]);
        REQUIRE(v.read_list_size() == 1 + (done.size() - cap - 1));
    }

    SECTION("writes, as one span") {
        // The write list starts with the container's own fill.
        v.register_write(std::span<const sycl::event>{done.data(), 70});
        REQUIRE(v.write_list_size() == 1);  // 71 events collapsed on the spot
    }

    SECTION("the collapsed event still covers the events it replaced") {
        Gate g{f};
        v.register_read(g.event);
        for (std::size_t i = 0; i < cap; ++i) v.register_read(done[i]);  // the 65th event
        REQUIRE(v.read_list_size() == 1);

        const std::vector<sycl::event> deps = v.write_deps();
        REQUIRE(deps.size() == 2);  // the collapsed read list, and the construction fill
        settle();
        REQUIRE_FALSE(is_done(deps.front()));
        g.open();
        REQUIRE(wait_done(deps.front()));
    }
}

TEST_CASE("the extra-dependency overloads order after the given events",
          "[usm_vector][events]") {
    Fixture f;
    warm_up(f);

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

        g.open();
        REQUIRE(wait_done(e));
        for (int x : v.host_view()) REQUIRE(x == 5);
    }

    SECTION("resize(n, deps)") {
        Gate g{f, v.data() + 8, 8};
        sycl::event e = v.resize(16, g.as_deps());
        settle();
        REQUIRE_FALSE(is_done(e));

        g.open();
        REQUIRE(wait_done(e));
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
        REQUIRE(wait_done(e));
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
        REQUIRE(wait_done(e));
        for (int x : v.host_view()) REQUIRE(x == 4);
    }

    SECTION("assign(other, deps)") {
        usm_vector<int> src{8, 3, alloc_of(f, kind::device)};
        src.sync();
        Gate g{f, v.data(), 8};
        sycl::event e = v.assign(src, g.as_deps());
        settle();
        REQUIRE_FALSE(is_done(e));

        g.open();
        REQUIRE(wait_done(e));
        for (int x : v.host_view()) REQUIRE(x == 3);
    }
}
