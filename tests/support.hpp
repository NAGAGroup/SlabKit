#pragma once

// Shared test scaffolding.
//
// The fixture is a slabkit context over the default device and one stream on
// it. The sycl objects beside them are the SAME objects: `ctx` is the backend
// default context the slabkit context retrieved, not a second one, so the
// interop placements in these tests name exactly the memory the primary ones
// do. Constructing a sycl::context here would be the one mistake slabkit
// exists to make unwritable.

#include <cstddef>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/slabkit/context.hpp"
#include "naga/slabkit/queue.hpp"

namespace slabkit_test {

struct Fixture {
    naga::slabkit::context sctx;
    naga::slabkit::queue   q;
    sycl::device           dev;
    sycl::context          ctx;
    sycl::queue            sq;

    Fixture()
        : sctx{sycl::device{sycl::default_selector_v}}
        , q{sctx}
        , dev{sctx.devices().front()}
        , ctx{sctx.context_of(dev)}
        , sq{ctx, dev} {}

    // The test is the user here, and waiting before teardown is the user's
    // job: nothing this fixture's streams started may outlive it. On CUDA
    // under WSL2, work left in flight would also make the next test's host
    // access to shared memory fault.
    ~Fixture() {
        q.wait();
        sq.wait();
    }
};

/// Every device the runtime enumerates, across all platforms.
inline std::vector<sycl::device> all_devices() {
    std::vector<sycl::device> out;
    for (const auto& p : sycl::platform::get_platforms())
        for (const auto& d : p.get_devices()) out.push_back(d);
    return out;
}

/// A device that is not `dev`, if the environment has one.
inline bool second_device(const sycl::device& dev, sycl::device& out) {
    for (const auto& d : all_devices()) {
        if (d != dev) {
            out = d;
            return true;
        }
    }
    return false;
}

template <class T>
std::vector<T> read_back(naga::slabkit::queue q, const T* p, std::size_t n) {
    std::vector<T> host(n);
    if (n > 0) {
        q.submit([&](sycl::handler& h) { h.memcpy(host.data(), p, n * sizeof(T)); });
        q.wait();
    }
    return host;
}

/// Reading from a device other than the stream's default one: the copy has to
/// be placed there, so it runs on that device's backend queue and its context.
template <class T>
std::vector<T> read_back(naga::slabkit::queue q, const sycl::device& dev, const T* p,
                         std::size_t n) {
    std::vector<T> host(n);
    if (n > 0) {
        q.submit(dev, [&](sycl::handler& h) { h.memcpy(host.data(), p, n * sizeof(T)); });
        q.wait();
    }
    return host;
}

template <class T>
std::vector<T> read_back(sycl::queue q, const T* p, std::size_t n) {
    std::vector<T> host(n);
    if (n > 0) q.memcpy(host.data(), p, n * sizeof(T)).wait();
    return host;
}

template <class Q, class T>
T read_one(Q q, const T* p) {
    return read_back(q, p, 1).front();
}

}  // namespace slabkit_test
