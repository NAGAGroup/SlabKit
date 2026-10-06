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
//     container's own fills, copies-in and resizes, and outside writers handed
//     over with register_write();
//   * the READ list: the events of everything that read the data -- copies out
//     of the container, and outside readers handed over with register_read().
//
// WHAT AN OUTSIDE TASK WAITS ON
//
//   read_deps()   what an outside READER must wait on: the write list.
//   write_deps()  what an outside WRITER must wait on: the read list. Writes
//                 never wait on writes.
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
// COLLAPSE ON ACCESS. Whenever a list that holds more than one event is read,
// the container submits an empty single_task on its queue that depends on all
// of them, replaces the list with that one event and uses it. A list of zero or
// one events is used as is. Lists only grow between accesses, except that a
// registration that makes a list longer than 64 events collapses it on the spot;
// nothing else clears them (a free, which has waited on everything, empties both).
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

namespace detail {

/// The empty task that joins an event list into one event. A named type, so
/// every container element type launches the same kernel.
struct collapse_noop {
    void operator()() const {}
};

}  // namespace detail

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
        if (n > 0) record_write(queue_().fill(all_events(), data_, T{}, n));
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

    sycl::event resize(size_type n) { return resize(n, std::span<const sycl::event>{}); }
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

    sycl::event resize(size_type n, std::span<const sycl::event> deps) {
        return resize_impl(n, T{}, /*value_init=*/true, deps);
    }
    sycl::event resize(size_type n, const T& value, std::span<const sycl::event> deps) {
        return resize_impl(n, value, /*value_init=*/false, deps);
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

    /// What an outside READER of data() must wait on: the write list, collapsed
    /// to one event if it held several. Empty when nothing has written.
    std::vector<sycl::event> read_deps() const { return collapsed(writes_); }

    /// What an outside WRITER of data() must wait on: the read list, collapsed
    /// to one event if it held several. Empty when nothing has read. Writes
    /// never wait on writes.
    std::vector<sycl::event> write_deps() const { return collapsed(reads_); }

    /// Hands the container the event of outside work that reads data(). Later
    /// writes to the container come after it.
    void register_read(const sycl::event& e) {
        reads_.push_back(e);
        cap(reads_);
    }
    void register_read(std::span<const sycl::event> evs) {
        reads_.insert(reads_.end(), evs.begin(), evs.end());
        cap(reads_);
    }

    /// Hands the container the event of outside work that writes data(). Later
    /// reads of, and writes to, the container come after it.
    void register_write(const sycl::event& e) {
        writes_.push_back(e);
        cap(writes_);
    }
    void register_write(std::span<const sycl::event> evs) {
        writes_.insert(writes_.end(), evs.begin(), evs.end());
        cap(writes_);
    }

    /// A list longer than this is collapsed by the registration that made it so,
    /// so a container that is registered on again and again but never accessed
    /// (read every step, never written) does not grow without bound.
    static constexpr size_type max_listed_events = 64;

    /// How many events each list currently holds (a diagnostic: the lists
    /// collapse on access and at max_listed_events).
    size_type read_list_size() const { return reads_.size(); }
    size_type write_list_size() const { return writes_.size(); }

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

    /// One event that completes when every event of `deps` has: an empty
    /// single_task on the container's queue, placed on the container's device
    /// (the host kind has none and runs on the stream's default).
    sycl::event join(const std::vector<sycl::event>& deps) const {
        if (get_kind() == kind::host) return queue_().single_task(deps, detail::collapse_noop{});
        return queue_().single_task(alloc_.get_device(), deps, detail::collapse_noop{});
    }

    /// Collapses a list that registration pushed past max_listed_events.
    void cap(std::vector<sycl::event>& list) const {
        if (list.size() > max_listed_events) collapsed(list);
    }

    /// The list, as events to depend on. A list holding more than one event is
    /// first replaced by the one event that joins them.
    std::vector<sycl::event> collapsed(std::vector<sycl::event>& list) const {
        if (list.size() > 1) {
            sycl::event joined = join(list);
            list.assign(1, joined);
        }
        return list;
    }

    /// Everything the container's own writes, its free and sync() come after:
    /// both lists, collapsed, and the extra dependencies of this one call.
    std::vector<sycl::event> all_events(std::span<const sycl::event> extra = {}) const {
        std::vector<sycl::event> all = collapsed(writes_);
        const std::vector<sycl::event> r = collapsed(reads_);
        all.reserve(all.size() + r.size() + extra.size());
        all.insert(all.end(), r.begin(), r.end());
        all.insert(all.end(), extra.begin(), extra.end());
        return all;
    }

    /// The container's own write, recorded in the write list.
    sycl::event record_write(sycl::event e) {
        writes_.push_back(e);
        return e;
    }

    /// The newest write as one event, for calls that submit nothing; a default
    /// (completed) event when nothing has written.
    sycl::event last_write() const {
        const std::vector<sycl::event> w = collapsed(writes_);
        return w.empty() ? sycl::event{} : w.back();
    }

    /// A copy out of `other`: it reads there and writes here. It comes after the
    /// source's write list and both of this container's lists, and is then a
    /// read of the source and a write of this container.
    sycl::event copied_from(const usm_vector& other, std::span<const sycl::event> extra = {}) {
        std::vector<sycl::event> deps = all_events(extra);
        const std::vector<sycl::event> src = other.read_deps();
        deps.insert(deps.end(), src.begin(), src.end());
        sycl::event e = queue_().copy(deps, other.data_, data_, size_);
        other.reads_.push_back(e);
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
    /// release waits for both lists, the copy that read it among them.
    void reallocate(size_type n, size_type keep, std::span<const sycl::event> deps = {}) {
        T* const        old      = data_;
        const size_type old_size = capacity_;
        data_                    = nullptr;
        capacity_                = 0;
        grow_to(n);

        if (keep > 0 && old != nullptr)
            record_write(queue_().copy(all_events(deps), old, data_, keep));
        if (old != nullptr) retire(old, old_size);
        size_ = keep;
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

    sycl::event resize_impl(size_type n, const T& value, bool value_init,
                            std::span<const sycl::event> deps) {
        if (n > capacity_) reallocate(n, size_ < n ? size_ : n, deps);

        if (n > size_) {
            const T     pattern = value_init ? T{} : value;
            sycl::event e =
                record_write(queue_().fill(all_events(deps), data_ + size_, pattern, n - size_));
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
