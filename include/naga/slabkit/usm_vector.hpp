#pragma once

// =============================================================================
// naga/slabkit/usm_vector.hpp — a value-semantic array in USM
// =============================================================================
//
// A contiguous array of trivially-copyable elements over allocator<T>, with
// value semantics. Every mutation is an item of the container's queue and
// returns its event; nothing blocks except the two host-access calls.
//
// Two properties make this safe to use with work still in flight:
//
//   * EVERY COPY IS A ROUTED copy on the container's queue -- between devices,
//     backends or kinds -- ordered as one item. The caller never sees the route
//     except in how long it takes.
//   * OLD STORAGE IS RELEASED THROUGH THE ALLOCATOR'S DEFERRED PATH, so resize,
//     reserve, shrink_to_fit, reassignment and destruction never invalidate
//     memory a kernel may still hold. data()'s value changes at a
//     reallocation; the memory it used to name outlives the stream's use of it.
//
// It is NOT a std::vector: no operator[] on the container, no iterators, no
// push_back/insert/erase, no at(). Element access is a host access, and a host
// access is a synchronization point that has to be asked for by name.

#include <cstddef>
#include <initializer_list>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/slabkit/allocator.hpp"
#include "naga/slabkit/kinds.hpp"
#include "naga/slabkit/mirrors.hpp"
#include "naga/slabkit/queue.hpp"

namespace naga::slabkit {

/// A mutable host view of a container's elements. Only shared and host kinds
/// can hand one out; the container synchronized before it did.
template <class T>
class host_access {
public:
    using size_type = std::size_t;

    host_access(T* p, size_type n) : p_{p}, n_{n} {}

    T&        operator[](size_type i) { return p_[i]; }
    const T&  operator[](size_type i) const { return p_[i]; }
    T*        data() { return p_; }
    size_type size() const { return n_; }
    T*        begin() { return p_; }
    T*        end() { return p_ + n_; }

private:
    T*        p_;
    size_type n_;
};

/// A read-only host view: borrowed for the host-accessible kinds, an owned D2H
/// copy for the device kind.
template <class T>
class host_view {
public:
    using size_type = std::size_t;

    host_view(const T* p, size_type n) : p_{p}, n_{n} {}
    explicit host_view(std::vector<T> owned)
        : owned_{std::move(owned)}, p_{owned_.data()}, n_{owned_.size()} {}

    const T&  operator[](size_type i) const { return p_[i]; }
    const T*  data() const { return p_; }
    size_type size() const { return n_; }
    const T*  begin() const { return p_; }
    const T*  end() const { return p_ + n_; }

private:
    std::vector<T> owned_;
    const T*       p_;
    size_type      n_;
};

template <class T>
class usm_vector {
    static_assert(std::is_trivially_copyable_v<T>,
                  "slabkit::usm_vector: elements are moved by memcpy, so they must be "
                  "trivially copyable");
    static_assert(std::is_trivially_destructible_v<T>,
                  "slabkit::usm_vector: nothing runs a destructor on USM elements");

public:
    using value_type     = T;
    using allocator_type = allocator<T>;
    using size_type      = std::size_t;

    explicit usm_vector(allocator_type alloc)
        : alloc_{std::move(alloc)}, stream_{alloc_.get_queue()} {}

    usm_vector(size_type n, allocator_type alloc)
        : alloc_{std::move(alloc)}, stream_{alloc_.get_queue()} {
        grow_to(n);
        size_ = n;
        if (n > 0) last_ = ::naga::slabkit::fill(queue_(), data_, T{}, n);
    }

    usm_vector(size_type n, const T& value, allocator_type alloc)
        : alloc_{std::move(alloc)}, stream_{alloc_.get_queue()} {
        grow_to(n);
        size_ = n;
        if (n > 0) last_ = ::naga::slabkit::fill(queue_(), data_, value, n);
    }

    usm_vector(std::span<const T> src, allocator_type alloc)
        : alloc_{std::move(alloc)}, stream_{alloc_.get_queue()} {
        grow_to(src.size());
        size_ = src.size();
        if (!src.empty()) last_ = from_host_copy(src);
    }

    usm_vector(std::initializer_list<T> src, allocator_type alloc)
        : usm_vector{std::span<const T>{src.begin(), src.size()}, std::move(alloc)} {}

    usm_vector(const usm_vector& other) : usm_vector{other, other.alloc_} {}

    usm_vector(const usm_vector& other, allocator_type alloc)
        : alloc_{std::move(alloc)}, stream_{alloc_.get_queue()} {
        same_stream(alloc_, other.alloc_);
        grow_to(other.size_);
        size_ = other.size_;
        if (size_ > 0)
            last_ = ::naga::slabkit::copy(queue_(), other.data_, data_, size_, deps_of(other));
    }

    usm_vector& operator=(const usm_vector& other) {
        if (this == &other) return *this;
        same_stream(alloc_, other.alloc_);

        if (other.size_ > capacity_) {
            release_storage();      // deferred: a kernel may still hold it
            grow_to(other.size_);
        }
        size_ = other.size_;
        if (size_ > 0)
            last_ = ::naga::slabkit::copy(queue_(), other.data_, data_, size_, deps_of(other));
        return *this;
    }

    usm_vector(usm_vector&& other) noexcept
        : alloc_{other.alloc_}
        , stream_{other.stream_}
        , data_{other.data_}
        , size_{other.size_}
        , capacity_{other.capacity_}
        , last_{other.last_} {
        other.data_     = nullptr;
        other.size_     = 0;
        other.capacity_ = 0;
    }

    /// Storage is stolen only between EQUAL allocators; anything else would
    /// silently move memory to a placement its owner did not ask for.
    usm_vector& operator=(usm_vector&& other) {
        if (this == &other) return *this;
        if (!(alloc_ == other.alloc_))
            throw std::invalid_argument{
                "naga::slabkit::usm_vector: a move between unequal allocators is refused; "
                "copy, or use slabkit::copy on data()"};

        release_storage();
        data_           = other.data_;
        size_           = other.size_;
        capacity_       = other.capacity_;
        last_           = other.last_;
        other.data_     = nullptr;
        other.size_     = 0;
        other.capacity_ = 0;
        return *this;
    }

    ~usm_vector() { release_storage(); }

    // -------------------------------------------------------------------------
    // Mutations — each one item, each returning its event
    // -------------------------------------------------------------------------

    sycl::event resize(size_type n) { return resize_impl(n, T{}, /*value_init=*/true); }
    sycl::event resize(size_type n, const T& value) {
        return resize_impl(n, value, /*value_init=*/false);
    }

    sycl::event fill(const T& value) {
        if (size_ == 0) return last_;
        last_ = ::naga::slabkit::fill(queue_(), data_, value, size_);
        return last_;
    }

    sycl::event from_host(std::span<const T> src) {
        if (src.size() > capacity_) {
            release_storage();
            grow_to(src.size());
        }
        size_ = src.size();
        if (size_ == 0) return last_;
        last_ = from_host_copy(src);
        return last_;
    }

    void reserve(size_type n) {
        if (n <= capacity_) return;

        T* const        old      = data_;
        const size_type old_size = size_;
        data_                    = nullptr;
        capacity_                = 0;
        grow_to(n);

        if (old_size > 0 && old != nullptr)
            last_ = ::naga::slabkit::copy(queue_(), old, data_, old_size);
        if (old != nullptr) alloc_.deallocate(old, old_size);
        size_ = old_size;
    }

    void shrink_to_fit() {
        if (capacity_ == size_) return;

        T* const        old      = data_;
        const size_type old_size = size_;
        data_                    = nullptr;
        capacity_                = 0;
        grow_to(old_size);

        if (old_size > 0 && old != nullptr)
            last_ = ::naga::slabkit::copy(queue_(), old, data_, old_size);
        if (old != nullptr) alloc_.deallocate(old, old_size);
        size_ = old_size;
    }

    /// Elements are trivially destructible, so this is a size change and
    /// nothing else. The storage stays.
    void clear() { size_ = 0; }

    // -------------------------------------------------------------------------
    // Observation
    // -------------------------------------------------------------------------

    size_type   size() const { return size_; }
    size_type   capacity() const { return capacity_; }
    bool        empty() const { return size_ == 0; }
    sycl::event last_event() const { return last_; }

    T*       data() { return data_; }
    const T* data() const { return data_; }

    allocator_type get_allocator() const { return alloc_; }
    queue          get_queue() const { return alloc_.get_queue(); }
    kind           get_kind() const { return alloc_.get_kind(); }

    /// Syncs, then hands out a mutable host view. Refused for the device kind:
    /// element access there would be a host dereference of device memory.
    host_access<T> get_access() {
        if (get_kind() == kind::device)
            throw std::invalid_argument{
                "naga::slabkit::usm_vector: get_access() on a device-kind container would be a "
                "host dereference of device memory; use host_view() for a read-only copy"};
        queue_().wait();
        return host_access<T>{data_, size_};
    }

    /// Syncs, then hands out a read-only host view: borrowed for the
    /// host-accessible kinds, an owned copy for the device kind.
    ::naga::slabkit::host_view<T> host_view() {
        queue_().wait();
        if (get_kind() != kind::device) return ::naga::slabkit::host_view<T>{data_, size_};

        std::vector<T> owned(size_);
        if (size_ > 0) {
            ::naga::slabkit::copy(queue_(), data_, owned.data(), size_);
            queue_().wait();
        }
        return ::naga::slabkit::host_view<T>{std::move(owned)};
    }

private:
    static void same_stream(const allocator_type& a, const allocator_type& b) {
        if (!(a.get_queue() == b.get_queue()))
            throw std::invalid_argument{
                "naga::slabkit::usm_vector: container-to-container operations require the same "
                "queue; the cross-stream tool is slabkit::copy on data() with explicit events"};
    }

    static std::vector<sycl::event> deps_of(const usm_vector& other) {
        return std::vector<sycl::event>{other.last_};
    }

    /// The stream, as a fresh handle: a queue is a light handle and the
    /// mirrors take it by reference.
    queue& queue_() const { return stream_; }

    void grow_to(size_type n) {
        if (n == 0) return;
        data_     = alloc_.allocate(n);
        capacity_ = n;
    }

    void release_storage() {
        if (data_ != nullptr) alloc_.deallocate(data_, capacity_);
        data_     = nullptr;
        size_     = 0;
        capacity_ = 0;
    }

    sycl::event from_host_copy(std::span<const T> src) {
        // The source is host memory the caller owns and may reuse the moment
        // this returns, so it is copied into an operation-owned buffer first
        // and the routed copy runs from that.
        auto owned = std::make_shared<std::vector<T>>(src.begin(), src.end());
        sycl::event e =
            ::naga::slabkit::copy(queue_(), owned->data(), data_, owned->size());
        queue_().submit_host([e, owned]() mutable { e.wait(); });
        return e;
    }

    sycl::event resize_impl(size_type n, const T& value, bool value_init) {
        if (n > capacity_) {
            T* const        old      = data_;
            const size_type old_size = size_;
            data_                    = nullptr;
            capacity_                = 0;
            grow_to(n);

            const size_type keep = old_size < n ? old_size : n;
            if (keep > 0 && old != nullptr) ::naga::slabkit::copy(queue_(), old, data_, keep);
            if (old != nullptr) alloc_.deallocate(old, old_size);
            size_ = keep;
        }

        if (n > size_) {
            const T pattern = value_init ? T{} : value;
            last_ = ::naga::slabkit::fill(queue_(), data_ + size_, pattern, n - size_);
        }
        size_ = n;
        return last_;
    }

    allocator_type alloc_;
    mutable queue  stream_;
    T*             data_     = nullptr;
    size_type      size_     = 0;
    size_type      capacity_ = 0;
    sycl::event    last_{};
};

}  // namespace naga::slabkit
