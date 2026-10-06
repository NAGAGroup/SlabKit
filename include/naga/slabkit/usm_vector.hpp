#pragma once

// =============================================================================
// naga/slabkit/usm_vector.hpp — a value-semantic array in USM
// =============================================================================
//
// A contiguous array of trivially-copyable elements over allocator<T>, with
// value semantics. Every mutation is a memory task of the container's queue and
// returns its event; nothing blocks except the host-access calls.
//
// THE CONTAINER KEEPS TWO EVENT LISTS, and they are the only event truth:
//
//   * the WRITE list: the events of everything that wrote the data -- the
//     container's own fills, copies-in and reallocation copies, and outside
//     writers handed over with register_write();
//   * the READ list: the events of everything that read the data -- copies out
//     of the container, and outside readers handed over with register_read().
//
// WHAT AN OUTSIDE TASK WAITS ON
//
//   read_deps()   what an outside READER must wait on: the write list.
//   write_deps()  what an outside WRITER must wait on: the read list. Outside
//                 writes never wait on each other.
//
// ALLOCATIONS ARE UNINITIALIZED. resize(n) and reserve(n) never fill; a
// constructor given only a size does not fill. A call or constructor given an
// explicit value fills with it. A reallocation that has elements to keep copies
// them over (the copy is an ordinary write, in the write list) and resize() /
// reserve() return its event; when nothing was copied they return an empty
// optional. Whoever reads the new storage waits on that event (CoSpan keeps it
// in the variable's resize-event slot).
//
// Work the CALLER launches over data() is theirs to order by passing one of
// these to it, and to hand back with register_read / register_write so the
// container orders what it does next after that work.
//
// WHAT THE CONTAINER WAITS ON
//
//   * its own writes (fill, resize and reallocation copies, from_host,
//     copy-in through assignment) wait on BOTH lists;
//   * a copy between containers waits on the source's write list and on both
//     of the destination's lists, then registers a read on the source and a
//     write on the destination;
//   * a copy out to the host waits on the write list and registers a read;
//   * the free (destruction, reassignment, reallocation) and sync() wait on
//     both lists.
//
// NO COLLAPSE. A list is handed out as it is: read_deps() and write_deps() return
// every event in the list, and the caller depends on all of them. Nothing is
// submitted to join them: with a join task (an empty single_task over the list,
// or one that increments a counter) CoSpan's tests saw dependent work read stale
// data about half the time on the OpenMP backend of AdaptiveCpp, and without it
// never. The cause is not established.
//
// PRUNING happens at assignment only: a registration (register_read,
// register_write, or the container's own) that makes a list longer than 8
// events drops the events whose command_execution_status is complete (they
// order nothing); events still in flight stay however many there are. Reading a
// list never changes it. A free, which has waited on everything, empties the
// lists.
//
// Two properties make this safe to use with work still in flight:
//
//   * EVERY COPY IS A ROUTED copy on the container's queue -- between devices,
//     backends or kinds -- ordered as one task. The caller never sees the route
//     except in how long it takes.
//   * OLD STORAGE IS FREED SYNCHRONOUSLY once both lists have completed.
//     Kernels the caller launched over data() and did not register are the
//     caller's to wait for before resize, reserve, shrink_to_fit,
//     reassignment or destruction.
//
// RESIZE IS std::vector's CONTRACT VERBATIM: the address changes immediately --
// the allocation and the swap are host-side -- and the contents catch up, as an
// enqueued copy (and a fill, when a value was given). Using the container concurrently with a resize is
// undefined, and scope exit is ordinary scope semantics.
//
// It is NOT a std::vector: no operator[] on the container, no iterators, no
// push_back/insert/erase, no at(). Element access is a host access, and a host
// access is a synchronization point that has to be asked for by name.

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <optional>
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

    /// UNINITIALIZED: n elements of whatever the allocation holds.
    usm_vector(size_type n, allocator_type alloc)
        : alloc_{std::move(alloc)}, stream_{alloc_.get_queue()} {
        grow_to(n);
        size_ = n;
    }

    usm_vector(size_type n, const T& value, allocator_type alloc)
        : alloc_{std::move(alloc)}, stream_{alloc_.get_queue()} {
        grow_to(n);
        size_ = n;
        if (n > 0) record_write(queue_().fill(all_events(), data_, value, n));
    }

    usm_vector(std::span<const T> src, allocator_type alloc)
        : alloc_{std::move(alloc)}, stream_{alloc_.get_queue()} {
        grow_to(src.size());
        size_ = src.size();
        if (!src.empty()) from_host_copy(src);
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
    /// are dependencies of the copy, combined with this container's two lists
    /// and the source's write list; they are never waited on. The copy is
    /// recorded as a read of the source and a write of this container. If
    /// nothing is copied (an empty source, or self-assignment) the returned
    /// event is this container's last write and does not cover `deps`.
    sycl::event assign(const usm_vector& other, std::span<const sycl::event> deps) {
        if (this == &other) return last_write();
        same_stream(alloc_, other.alloc_);

        if (other.size_ > capacity_) {
            release_storage();  // waits on both of this container's lists, then frees
            grow_to(other.size_);
        }
        size_ = other.size_;
        if (size_ > 0) return copied_from(other, deps);
        return last_write();
    }

    usm_vector(usm_vector&& other) noexcept
        : alloc_{other.alloc_}
        , stream_{other.stream_}
        , data_{other.data_}
        , size_{other.size_}
        , capacity_{other.capacity_}
        , reads_{std::move(other.reads_)}
        , writes_{std::move(other.writes_)} {
        other.data_     = nullptr;
        other.size_     = 0;
        other.capacity_ = 0;
        other.reads_.clear();
        other.writes_.clear();
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
        reads_          = std::move(other.reads_);
        writes_         = std::move(other.writes_);
        other.data_     = nullptr;
        other.size_     = 0;
        other.capacity_ = 0;
        other.reads_.clear();
        other.writes_.clear();
        return *this;
    }

    ~usm_vector() { release_storage(); }

    // -------------------------------------------------------------------------
    // Mutations — each one task, each returning its event
    // -------------------------------------------------------------------------

    /// Size change without initialization. Returns the event of the copy of the
    /// old elements when the storage was reallocated with elements to keep;
    /// otherwise an empty optional (shrinking, growing within capacity, growing
    /// from empty).
    std::optional<sycl::event> resize(size_type n) {
        return resize(n, std::span<const sycl::event>{});
    }
    /// Resizes and fills the NEW elements with `value`.
    sycl::event resize(size_type n, const T& value) {
        return resize(n, value, std::span<const sycl::event>{});
    }
    sycl::event fill(const T& value) { return fill(value, std::span<const sycl::event>{}); }
    sycl::event from_host(std::span<const T> src) {
        return from_host(src, std::span<const sycl::event>{});
    }

    // The same mutations with extra dependencies: events the operation is
    // ordered after, combined with the container's two lists. They are
    // submission dependencies, never waited on. When a call submits nothing (a
    // fill of an empty container, a resize that does not grow) the returned
    // event is the container's last write and does not cover `deps`.

    std::optional<sycl::event> resize(size_type n, std::span<const sycl::event> deps) {
        if (n > capacity_) {
            std::optional<sycl::event> e = reallocate(n, size_, deps);
            size_                        = n;
            return e;
        }
        size_ = n;
        return std::nullopt;
    }
    sycl::event resize(size_type n, const T& value, std::span<const sycl::event> deps) {
        return resize_fill(n, value, deps);
    }

    sycl::event fill(const T& value, std::span<const sycl::event> deps) {
        if (size_ == 0) return last_write();
        return record_write(queue_().fill(all_events(deps), data_, value, size_));
    }

    sycl::event from_host(std::span<const T> src, std::span<const sycl::event> deps) {
        if (src.size() > capacity_) {
            release_storage();
            grow_to(src.size());
        }
        size_ = src.size();
        if (size_ == 0) return last_write();
        return from_host_copy(src, deps);
    }

    // -------------------------------------------------------------------------
    // Event lists — what outside work waits on, and what it hands over
    // -------------------------------------------------------------------------

    /// What an outside READER of data() must wait on: every event of the write
    /// list. Empty when nothing has written.
    std::vector<sycl::event> read_deps() const { return writes_; }

    /// What an outside WRITER of data() must wait on: every event of the read
    /// list. Empty when nothing has read. Outside writes never wait on each other.
    std::vector<sycl::event> write_deps() const { return reads_; }

    /// Hands the container the event of outside work that reads data(). Later
    /// writes to the container come after it.
    void register_read(const sycl::event& e) { push(reads_, e); }
    void register_read(std::span<const sycl::event> evs) {
        reads_.insert(reads_.end(), evs.begin(), evs.end());
        prune(reads_);
    }

    /// Hands the container the event of outside work that writes data(). Later
    /// reads of, and writes to, the container come after it.
    void register_write(const sycl::event& e) { push(writes_, e); }
    void register_write(std::span<const sycl::event> evs) {
        writes_.insert(writes_.end(), evs.begin(), evs.end());
        prune(writes_);
    }

    /// A list that grows past this drops its completed events, at the
    /// assignment that made it so (never at a read).
    static constexpr size_type max_listed_events = 8;

    /// How many events each list currently holds (a diagnostic).
    size_type read_list_size() const { return reads_.size(); }
    size_type write_list_size() const { return writes_.size(); }

    /// Capacity only; the size stays. The event of the copy of the old elements
    /// when there were any, else empty.
    std::optional<sycl::event> reserve(size_type n) {
        if (n <= capacity_) return std::nullopt;
        return reallocate(n, size_);
    }

    std::optional<sycl::event> shrink_to_fit() {
        if (capacity_ == size_) return std::nullopt;
        return reallocate(size_, size_);
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

    T*       data() { return data_; }
    const T* data() const { return data_; }

    allocator_type get_allocator() const { return alloc_; }
    queue          get_queue() const { return alloc_.get_queue(); }
    kind           get_kind() const { return alloc_.get_kind(); }

    /// Waits on both lists, and on nothing else.
    void sync() const { sycl::event::wait(all_events()); }

    /// Syncs, then hands out a mutable host view. Refused for the device kind:
    /// element access there would be a host dereference of device memory.
    ///
    /// SHARED KIND ON DEVICES WITHOUT CONCURRENT MANAGED ACCESS (CUDA on WSL2
    /// and on Windows): the host may not touch ANY shared allocation while ANY
    /// work is executing on the device -- including work unrelated to this
    /// container, and work on other streams. get_access() waits for this
    /// container's own lists only. On such platforms, wait every stream that
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

    /// Waits on the write list, then hands out a read-only host view: borrowed
    /// for the host-accessible kinds, an owned copy for the device kind (whose
    /// copy is registered as a read).
    ::naga::slabkit::host_view<T> host_view() {
        if (get_kind() != kind::device) {
            sycl::event::wait(read_deps());
            return ::naga::slabkit::host_view<T>{data_, size_};
        }

        std::vector<T> owned(size_);
        if (size_ > 0) {
            sycl::event e = queue_().copy(read_deps(), data_, owned.data(), size_);
            register_read(e);
            e.wait();
        }
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

    /// Past max_listed_events, drops the events that have completed: they order
    /// nothing. Events still in flight stay. Called after an assignment only.
    static void prune(std::vector<sycl::event>& list) {
        if (list.size() > max_listed_events) std::erase_if(list, is_complete);
    }

    static void push(std::vector<sycl::event>& list, const sycl::event& e) {
        list.push_back(e);
        prune(list);
    }

    /// Everything the container's own writes, its free and sync() come after:
    /// both lists and the extra dependencies of this one call.
    std::vector<sycl::event> all_events(std::span<const sycl::event> extra = {}) const {
        std::vector<sycl::event> all = writes_;
        const std::vector<sycl::event>& r = reads_;
        all.reserve(all.size() + r.size() + extra.size());
        all.insert(all.end(), r.begin(), r.end());
        all.insert(all.end(), extra.begin(), extra.end());
        return all;
    }

    /// The container's own write, recorded in the write list.
    sycl::event record_write(sycl::event e) {
        push(writes_, e);
        return e;
    }

    /// The newest write, for calls that submit nothing; a default (completed)
    /// event when nothing has written.
    sycl::event last_write() const { return writes_.empty() ? sycl::event{} : writes_.back(); }

    /// A copy out of `other`: it reads there and writes here. It comes after the
    /// source's write list and both of this container's lists, and is then a
    /// read of the source and a write of this container.
    sycl::event copied_from(const usm_vector& other, std::span<const sycl::event> extra = {}) {
        std::vector<sycl::event> deps = all_events(extra);
        const std::vector<sycl::event> src = other.read_deps();
        deps.insert(deps.end(), src.begin(), src.end());
        sycl::event e = queue_().copy(deps, other.data_, data_, size_);
        push(other.reads_, e);
        return record_write(e);
    }

    /// The stream, as a fresh handle: a queue is a light handle.
    queue& queue_() const { return stream_; }

    void grow_to(size_type n) {
        if (n == 0) return;
        data_     = alloc_.allocate(n);
        capacity_ = n;
    }

    /// Frees `p` after both lists, then empties them: the free waited on every
    /// event they held. With no storage to free it still waits, so an event
    /// handed to an empty container is not forgotten unfinished.
    void retire(T* p, size_type cap) {
        const std::vector<sycl::event> all = all_events();
        if (p != nullptr) alloc_.deallocate(p, cap, all);
        else sycl::event::wait(all);
        reads_.clear();
        writes_.clear();
    }

    void release_storage() {
        retire(data_, capacity_);
        data_     = nullptr;
        size_     = 0;
        capacity_ = 0;
    }

    /// New storage of `n` elements carrying the first `keep` of the old, whose
    /// release waits for both lists, the copy that read it among them. Returns the
    /// copy's event, empty when nothing was copied. The size becomes `keep`.
    std::optional<sycl::event> reallocate(size_type n, size_type keep,
                                          std::span<const sycl::event> deps = {}) {
        T* const        old      = data_;
        const size_type old_size = capacity_;
        data_                    = nullptr;
        capacity_                = 0;
        grow_to(n);

        std::optional<sycl::event> copy;
        if (keep > 0 && old != nullptr)
            copy = record_write(queue_().copy(all_events(deps), old, data_, keep));
        if (old != nullptr) retire(old, old_size);
        // retire() waited on everything and emptied the lists, the copy among
        // it. The copy wrote the new storage, so it goes back into the write
        // list (complete by now, but the list is the record of what wrote).
        if (copy) push(writes_, *copy);
        size_ = keep;
        return copy;
    }

    sycl::event from_host_copy(std::span<const T> src, std::span<const sycl::event> deps = {}) {
        // The source is host memory the caller owns and may reuse the moment
        // this returns, so it is copied into a task-owned buffer first and the
        // routed copy runs from that.
        auto        owned = std::make_shared<std::vector<T>>(src.begin(), src.end());
        sycl::event e     = queue_().copy(all_events(deps), owned->data(), data_, owned->size());
        queue_().async_host(std::vector<sycl::event>{e}, [owned] {});
        return record_write(e);
    }

    sycl::event resize_fill(size_type n, const T& value, std::span<const sycl::event> deps) {
        if (n > capacity_) reallocate(n, size_ < n ? size_ : n, deps);

        if (n > size_) {
            sycl::event e =
                record_write(queue_().fill(all_events(deps), data_ + size_, value, n - size_));
            size_ = n;
            return e;
        }
        size_ = n;
        return last_write();
    }

    allocator_type      alloc_;
    mutable queue       stream_;
    T*                  data_     = nullptr;
    size_type           size_     = 0;
    size_type           capacity_ = 0;

    // The two event lists, the container's only event truth. Mutable: reading a
    // list collapses it, and a copy out of a const source is a read of it.
    mutable std::vector<sycl::event> reads_{};
    mutable std::vector<sycl::event> writes_{};
};

}  // namespace naga::slabkit
