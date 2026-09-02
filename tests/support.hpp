#pragma once

// Shared test scaffolding.
//
// Every fixture follows the contract's construction order: the device's
// context is created FIRST and the queue is constructed from it, so the
// context a queue reports is the same object the tests pass to the
// device+context forms.

#include <cstddef>
#include <vector>

#include <sycl/sycl.hpp>

namespace slabkit_test {

struct Fixture {
    sycl::device  dev;
    sycl::context ctx;
    sycl::queue   q;

    Fixture() : dev{sycl::default_selector_v}, ctx{dev}, q{ctx, dev} {}
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
std::vector<T> read_back(sycl::queue q, const T* p, std::size_t n) {
    std::vector<T> host(n);
    if (n > 0) q.memcpy(host.data(), p, n * sizeof(T)).wait();
    return host;
}

template <class T>
T read_one(sycl::queue q, const T* p) {
    return read_back(q, p, 1).front();
}

}  // namespace slabkit_test
