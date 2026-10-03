#pragma once

// =============================================================================
// naga/slabkit/usm_vector.hpp — a value-semantic array in USM
// =============================================================================
//
// A contiguous array of trivially-copyable elements over allocator<T>, with
// value semantics. Every mutation is a memory task of the container's queue and
// returns its event; nothing blocks except the three host-access calls.
//
// THE CONTAINER SYNCS WHAT IT DID; THE CALLER SYNCS WHAT THEY DID. Memory tasks
// are asynchronous with each other (queue.hpp), so the container carries its own
// last event and every mutation depends on it: two mutations of one container
// are ordered, and a mutation is ordered against the kernel and group work the
// stream had outstanding. Work the CALLER launched over data() is theirs to
// order -- by passing last_event() to it, and its event back to the next
// mutation.
//
// OR HAND IT OVER. register_event(s) gives the container the events of outside
// work (kernels that read or wrote data()) and the container takes ownership of
// ordering after them: it keeps them beside its own last event, every mutation
// is submitted with them as dependencies, the free waits on them, and sync()
// covers them. A copy out of a container is also ordered after the events
// registered on its source. Nothing waits queue-wide, and a registered event is
// forgotten once a task that depended on it has been submitted (or, if it has
// already completed, at the next registration). Register BEFORE the next
// mutation, destruction or copy that must come after the work. The
// extra-dependency overloads (resize, fill, from_host, assign taking a span of
// events) order that one operation after the given events without keeping them.
//
// Two properties make this safe to use with work still in flight:
//
//   * EVERY COPY IS A ROUTED copy on the container's queue -- between devices,
//     backends or kinds -- ordered as one task. The caller never sees the route
//     except in how long it takes.
//   * OLD STORAGE IS FREED SYNCHRONOUSLY once the container's own last task (the
//     copy that read it) has completed. Kernels the caller launched over data()
//     are the caller's to wait for before resize, reserve, shrink_to_fit,
//     reassignment or destruction.
//
// RESIZE IS std::vector's CONTRACT VERBATIM: the address changes immediately --
// the allocation and the swap are host-side -- and the contents catch up, as an
// enqueued copy and fill. Using the container concurrently with a resize is
// undefined, and scope exit is ordinary scope semantics.
//
// It is NOT a std::vector: no operator[] on the container, no iterators, no
// push_back/insert/erase, no at(). Element access is a host access, and a host
// access is a synchronization point that has to be asked for by name.

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/slabkit/allocator.hpp"
#include "naga/slabkit/kinds.hpp"
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
        if (n > 0) last_ = queue_().fill(mine(), data_, T{}, n);
    }

    usm_vector(size_type n, const T& value, allocator_type alloc)
        : alloc_{std::move(alloc)}, stream_{alloc_.get_queue()} {
        grow_to(n);
        size_ = n;
        if (n > 0) last_ = queue_().fill(mine(), data_, value, n);
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
        if (size_ > 0) copied_from(other);
    }

    usm_vector& operator=(const usm_vector& other) {
        assign(other, std::span<const sycl::event>{});
        return *this;
    }

    /// Copy-assignment as an event: returns the copy's event. The extra events
    /// are dependencies of the copy, combined with this container's own and
    /// the source's; they are never waited on. If nothing is copied (an empty
    /// source, or self-assignment) the returned event is last_event().
    sycl::event assign(const usm_vector& other, std::span<const sycl::event> deps) {
        if (this == &other) return last_;
        same_stream(alloc_, other.alloc_);

        if (other.size_ > capacity_) {
            release_storage();  // waits for this container's own work, then frees
            grow_to(other.size_);
        }
        size_ = other.size_;
        if (size_ > 0) copied_from(other, deps);
        return last_;
    }

    usm_vector(usm_vector&& other) noexcept
        : alloc_{other.alloc_}
        , stream_{other.stream_}
        , data_{other.data_}
        , size_{other.size_}
        , capacity_{other.capacity_}
        , last_{other.last_}
        , registered_{std::move(other.registered_)} {
        other.data_     = nullptr;
        other.size_     = 0;
        other.capacity_ = 0;
        other.registered_.clear();
    }

    /// Storage is stolen only between EQUAL allocators; anything else would
    /// silently move memory to a placement its owner did not ask for.
    usm_vector& operator=(usm_vector&& other) {
        if (this == &other) return *this;
        if (!(alloc_ == other.alloc_))
            throw std::invalid_argument{
                "naga::slabkit::usm_vector: a move between unequal allocators is refused; "
                "copy, or use the queue's copy on data()"};

        release_storage();
        data_           = other.data_;
        size_           = other.size_;
        capacity_       = other.capacity_;
        last_           = other.last_;
        registered_     = std::move(other.registered_);
        other.data_     = nullptr;
        other.size_     = 0;
        other.capacity_ = 0;
        other.registered_.clear();
        return *this;
    }

    ~usm_vector() { release_storage(); }

    // -------------------------------------------------------------------------
    // Mutations — each one task, each returning its event
    // -------------------------------------------------------------------------

    sycl::event resize(size_type n) { return resize(n, std::span<const sycl::event>{}); }
    sycl::event resize(size_type n, const T& value) {
        return resize(n, value, std::span<const sycl::event>{});
    }
    sycl::event fill(const T& value) { return fill(value, std::span<const sycl::event>{}); }
    sycl::event from_host(std::span<const T> src) {
        return from_host(src, std::span<const sycl::event>{});
    }

    // The same mutations with extra dependencies: events the operation is
    // ordered after, combined with the container's own. They are submission
    // dependencies, never waited on. When a call submits nothing (a fill of an
    // empty container, a resize that does not grow) the returned event is
    // last_event() and does not cover `deps`.

    sycl::event resize(size_type n, std::span<const sycl::event> deps) {
        return resize_impl(n, T{}, /*value_init=*/true, deps);
    }
    sycl::event resize(size_type n, const T& value, std::span<const sycl::event> deps) {
        return resize_impl(n, value, /*value_init=*/false, deps);
    }

    sycl::event fill(const T& value, std::span<const sycl::event> deps) {
        if (size_ == 0) return last_;
        set_last(queue_().fill(mine(deps), data_, value, size_));
        return last_;
    }

    sycl::event from_host(std::span<const T> src, std::span<const sycl::event> deps) {
        if (src.size() > capacity_) {
            release_storage();
            grow_to(src.size());
        }
        size_ = src.size();
        if (size_ == 0) return last_;
        set_last(from_host_copy(src, deps));
        return last_;
    }

    // -------------------------------------------------------------------------
    // Registered events — outside work the container orders itself after
    // -------------------------------------------------------------------------

    /// Hands the container events of outside work over data(). Events that have
    /// already completed are dropped first, then these are kept.
    void register_events(std::span<const sycl::event> evs) {
        std::erase_if(registered_, [](const sycl::event& e) { return is_complete(e); });
        registered_.insert(registered_.end(), evs.begin(), evs.end());
    }
    void register_event(const sycl::event& e) {
        register_events(std::span<const sycl::event>{&e, 1});
    }

    /// How many registered events the container is currently keeping.
    size_type registered_event_count() const { return registered_.size(); }

    void reserve(size_type n) {
        if (n <= capacity_) return;
        reallocate(n, size_);
    }

    void shrink_to_fit() {
        if (capacity_ == size_) return;
        reallocate(size_, size_);
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

    /// What a submission that touches this container's data should depend on: the
    /// container's last event plus every event registered on it. A caller that
    /// reads or writes data() on a queue of its own passes this list to that
    /// submission, so the work comes after everything the container is still
    /// waiting for.
    std::vector<sycl::event> events() const { return mine(); }

    T*       data() { return data_; }
    const T* data() const { return data_; }

    allocator_type get_allocator() const { return alloc_; }
    queue          get_queue() const { return alloc_.get_queue(); }
    kind           get_kind() const { return alloc_.get_kind(); }

    /// Waits for what the CONTAINER has enqueued and the events registered on
    /// it, and for nothing else.
    void sync() const { sycl::event::wait(mine()); }

    /// Syncs, then hands out a mutable host view. Refused for the device kind:
    /// element access there would be a host dereference of device memory.
    ///
    /// SHARED KIND ON DEVICES WITHOUT CONCURRENT MANAGED ACCESS (CUDA on WSL2
    /// and on Windows): the host may not touch ANY shared allocation while ANY
    /// work is executing on the device -- including work unrelated to this
    /// container, and work on other streams. get_access() waits for this
    /// container's own work only. On such platforms, wait every stream that
    /// uses the device (queue::wait()) before host access to shared memory,
    /// or prefer device-kind containers with host_view() / from_host().
    host_access<T> get_access() {
        if (get_kind() == kind::device)
            throw std::invalid_argument{
                "naga::slabkit::usm_vector: get_access() on a device-kind container would be a "
                "host dereference of device memory; use host_view() for a read-only copy"};
        sync();
        return host_access<T>{data_, size_};
    }

    /// Syncs, then hands out a read-only host view: borrowed for the
    /// host-accessible kinds, an owned copy for the device kind.
    ::naga::slabkit::host_view<T> host_view() {
        sync();
        if (get_kind() != kind::device) return ::naga::slabkit::host_view<T>{data_, size_};

        std::vector<T> owned(size_);
        if (size_ > 0) queue_().copy(mine(), data_, owned.data(), size_).wait();
        return ::naga::slabkit::host_view<T>{std::move(owned)};
    }

private:
    static void same_stream(const allocator_type& a, const allocator_type& b) {
        if (!(a.get_queue() == b.get_queue()))
            throw std::invalid_argument{
                "naga::slabkit::usm_vector: container-to-container operations require the same "
                "queue; the cross-stream tool is the queue's copy on data() with explicit "
                "events"};
    }

    static bool is_complete(const sycl::event& e) {
        return e.get_info<sycl::info::event::command_execution_status>() ==
               sycl::info::event_command_status::complete;
    }

    /// Everything a task on this container must come after: its own last task
    /// (the self-edge that orders two mutations, which nothing else provides),
    /// the registered events, and any extra dependencies of this one call.
    std::vector<sycl::event> mine(std::span<const sycl::event> extra = {}) const {
        std::vector<sycl::event> all;
        all.reserve(1 + registered_.size() + extra.size());
        all.push_back(last_);
        all.insert(all.end(), registered_.begin(), registered_.end());
        all.insert(all.end(), extra.begin(), extra.end());
        return all;
    }

    /// Records the event of a task submitted with mine(): last_ now depends on
    /// every registered event, so they can be forgotten. Call it ONLY for a
    /// submission that carried the full set.
    void set_last(sycl::event e) {
        last_ = std::move(e);
        registered_.clear();
    }

    /// A copy out of `other`: it wrote here and read there, so both remember it.
    /// It comes after this container's full set and the source's full set. Both
    /// containers' registered events are forgotten: the copy depends on them
    /// and is each container's new last_.
    void copied_from(const usm_vector& other, std::span<const sycl::event> extra = {}) {
        std::vector<sycl::event> deps = mine(extra);
        deps.push_back(other.last_);
        deps.insert(deps.end(), other.registered_.begin(), other.registered_.end());
        set_last(queue_().copy(deps, other.data_, data_, size_));
        other.last_ = last_;
        other.registered_.clear();
    }

    /// The stream, as a fresh handle: a queue is a light handle.
    queue& queue_() const { return stream_; }

    void grow_to(size_type n) {
        if (n == 0) return;
        data_     = alloc_.allocate(n);
        capacity_ = n;
    }

    void release_storage() {
        // The full set: the free waits on the last task and every registered event.
        if (data_ != nullptr) alloc_.deallocate(data_, capacity_, mine());
        registered_.clear();  // waited on above, or there was no storage to protect
        data_     = nullptr;
        size_     = 0;
        capacity_ = 0;
    }

    /// New storage of `n` elements carrying the first `keep` of the old, whose
    /// release waits for the copy that read it.
    void reallocate(size_type n, size_type keep, std::span<const sycl::event> deps = {}) {
        T* const        old      = data_;
        const size_type old_size = capacity_;
        data_                    = nullptr;
        capacity_                = 0;
        grow_to(n);

        // The copy carries the full set, so last_ covers the registered events
        // afterwards; without a copy they stay registered and the free waits on them.
        if (keep > 0 && old != nullptr) set_last(queue_().copy(mine(deps), old, data_, keep));
        if (old != nullptr) alloc_.deallocate(old, old_size, mine());
        size_ = keep;
    }

    sycl::event from_host_copy(std::span<const T> src, std::span<const sycl::event> deps = {}) {
        // The source is host memory the caller owns and may reuse the moment
        // this returns, so it is copied into a task-owned buffer first and the
        // routed copy runs from that.
        auto        owned = std::make_shared<std::vector<T>>(src.begin(), src.end());
        sycl::event e     = queue_().copy(mine(deps), owned->data(), data_, owned->size());
        queue_().async_host(std::vector<sycl::event>{e}, [owned] {});
        return e;
    }

    sycl::event resize_impl(size_type n, const T& value, bool value_init,
                            std::span<const sycl::event> deps) {
        if (n > capacity_) reallocate(n, size_ < n ? size_ : n, deps);

        if (n > size_) {
            const T pattern = value_init ? T{} : value;
            set_last(queue_().fill(mine(deps), data_ + size_, pattern, n - size_));
        }
        size_ = n;
        return last_;
    }

    allocator_type      alloc_;
    mutable queue       stream_;
    T*                  data_     = nullptr;
    size_type           size_     = 0;
    size_type           capacity_ = 0;
    mutable sycl::event last_{};

    // Events of outside work the container orders itself after. Mutable for the
    // same reason as last_: a copy out of a const source depends on them and
    // then supersedes them in the source's last_.
    mutable std::vector<sycl::event> registered_{};
};

}  // namespace naga::slabkit
