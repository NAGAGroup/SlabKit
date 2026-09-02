// =============================================================================
// make.cpp — the initialization core of the make family
// =============================================================================
//
// Writing device memory needs no element type: a copy is a copy and a fill is
// a repeated pattern. So the whole trivially-copyable half of the make family
// compiles once, here, and the public forms are type plumbing over it. Only
// the placement-new path for non-trivially-copyable types is genuinely
// per-type device code, and that stays a template in make.hpp.

#include <cstddef>
#include <cstring>
#include <new>

#include <sycl/sycl.hpp>

#include "detail/os_alloc.hpp"
#include "naga/slabkit/detail/core.hpp"

namespace naga::slabkit::detail {

sycl::queue init_queue(const sycl::queue& q, const sycl::device& dev,
                       const sycl::context& ctx) {
    if (q.get_device() == dev) return q;

    // A foreign device gets a queue on its OWN context. AdaptiveCpp resolves
    // the device of a memcpy operand by querying the pointer against the
    // queue's context, so initializing through the original queue would have
    // the runtime mistake this device's memory for host memory.
    return sycl::queue{ctx, dev};
}

void copy_from_host(void* dst, const void* src, std::size_t bytes, sycl::queue q) {
    if (dst == nullptr || src == nullptr || bytes == 0) return;
    q.memcpy(dst, src, bytes).wait();
}

void fill_pattern(void* dst, const void* pattern, std::size_t pattern_size, std::size_t count,
                  sycl::queue q) {
    if (dst == nullptr || pattern == nullptr || pattern_size == 0 || count == 0) return;

    const auto* pat = static_cast<const std::byte*>(pattern);

    // A pattern of identical bytes -- which is what value-initialization of
    // anything zeroable looks like -- is a memset, and needs no kernel.
    bool uniform = true;
    for (std::size_t i = 1; i < pattern_size; ++i) {
        if (pat[i] != pat[0]) {
            uniform = false;
            break;
        }
    }
    if (uniform) {
        q.memset(dst, static_cast<int>(std::to_integer<unsigned char>(pat[0])),
                 pattern_size * count)
            .wait();
        return;
    }

    // Otherwise stage the pattern on the device and replicate it with one
    // kernel. The staging allocation is an ordinary slabkit handle, so it is
    // released even if the kernel throws.
    auto stage = os_allocate(0, pattern_size, q.get_device(), q.get_context(), {});
    if (!stage) throw std::bad_alloc{};
    q.memcpy(stage.get(), pattern, pattern_size).wait();

    const std::byte* src   = stage.get();
    auto*            out   = static_cast<std::byte*>(dst);
    const std::size_t psize = pattern_size;
    const std::size_t total = pattern_size * count;

    q.parallel_for(sycl::range<1>{total}, [=](sycl::id<1> idx) {
         const std::size_t i = idx[0];
         out[i]              = src[i % psize];
     }).wait();
}

}  // namespace naga::slabkit::detail
