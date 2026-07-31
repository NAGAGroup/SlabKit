# SlabKit Anti-Patterns

This guide documents ten common mistakes in SlabKit code: from move-semantics errors and lifetime violations to silent memory bugs and unhandled exceptions. Understanding these patterns will help you avoid silent undefined behaviour, pool corruption, and runtime crashes.

## Quick Reference

| Anti-Pattern | Type |
|---|---|
| Capturing `DeviceArray` by value in a kernel lambda | Compile Error |
| Using a `DeviceView` after its backing array is destroyed | Runtime UB |
| Keeping a `shared_ptr<Context>` alive shorter than its arrays | Silent Bug |
| Rewinding scratch while GPU work is in flight | Runtime UB |
| Using `q.submit(handler)` for USM kernels | Unnecessary |
| Indexing `DeviceView` with the wrong type | Compile Error |
| Constructing Context from multiple threads | Runtime UB |
| Assuming device memory is zero-initialized | Silent Bug |
| Ignoring `AllocationError` on pool exhaustion | Silent Bug |

## Anti-Pattern 1: Capturing `DeviceArray` by Value in a Kernel Lambda

### The Problem
```cpp
auto arr = naga::slabkit::make_device_array<float, 1>(sycl::range<1>{1024}, ctx, 0);
// ERROR: DeviceArray is move-only; copy constructor is deleted
q.parallel_for(arr.get_range(), [=](sycl::id<1> id) {
    arr[id] = 0.0f;  // Won't compile: copy of move-only type
}).wait();
```

### Why It's Wrong
`DeviceArray` is move-only — its copy constructor is explicitly deleted. Lambda capture by value (`[=]`) attempts to copy the array, which fails to compile. Even if it compiled, `DeviceArray` is not trivially copyable and cannot be safely passed to the device.

### The Fix
```cpp
auto arr = naga::slabkit::make_device_array<float, 1>(sycl::range<1>{1024}, ctx, 0);
auto view = arr.view();  // Trivially-copyable; safe to capture
q.parallel_for(view.get_range(), [=](sycl::id<1> id) {
    view[id] = 0.0f;
}).wait();
// arr must remain in scope until the kernel completes
```

### Rule
> Always call `.view()` before capturing device data in a kernel lambda; never capture `DeviceArray` directly.

## Anti-Pattern 2: Using a `DeviceView` After Its Backing Array Is Destroyed

### The Problem
```cpp
naga::slabkit::DeviceView<float, 1> view;
{
    auto arr = naga::slabkit::make_device_array<float, 1>(sycl::range<1>{1024}, ctx, 0);
    view = arr.view();
}  // arr destroyed here; pool nodes returned; view now dangling

// Use after free:
q.parallel_for(view.get_range(), [=](sycl::id<1> id) {
    view[id] = 1.0f;  // UB: pointer is dangling
}).wait();
```

### Why It's Wrong
`DeviceView` is non-owning — it holds only a raw pointer into the pool. When the `DeviceArray` is destroyed, its destructor returns the nodes to the pool. Any subsequent access through the view is undefined behaviour. There is no use-after-free detection at the SlabKit layer.

### The Fix
```cpp
auto arr = naga::slabkit::make_device_array<float, 1>(sycl::range<1>{1024}, ctx, 0);
auto view = arr.view();
q.parallel_for(view.get_range(), [=](sycl::id<1> id) {
    view[id] = 1.0f;
}).wait();
// arr must outlive the kernel and the event
```

### Rule
> Ensure the owning `DeviceArray` outlives every kernel and event that references one of its views.

## Anti-Pattern 3: Keeping a `shared_ptr<Context>` Alive Shorter Than Its Arrays

### The Problem
```cpp
naga::slabkit::DeviceArray<float, 1> arr;
{
    auto ctx = naga::slabkit::ContextBuilder().devices({dev}).build();
    arr = naga::slabkit::make_device_array<float, 1>(sycl::range<1>{1024}, ctx, 0);
}  // ctx destroyed here; shared_ptr refcount hits 0

// arr's destructor fires later:
// DeviceUSMPoolDeleter::operator() checks weak_ptr<Context> — it's expired
// Pool return is silently skipped; no assertion, no error
```

### Why It's Wrong
`DeviceArray`'s pool deleter holds a `weak_ptr<Context>`. When the last `shared_ptr<Context>` is destroyed, the weak_ptr expires. Subsequent pool returns in the deleter silently no-op — memory is leaked from the pool's perspective (the underlying USM slab is freed with the Context, so no OS leak, but the pool accounting is wrong).

### The Fix
```cpp
// Keep ctx alive at least as long as any array allocated from it
auto ctx = naga::slabkit::ContextBuilder().devices({dev}).build();
auto arr = naga::slabkit::make_device_array<float, 1>(sycl::range<1>{1024}, ctx, 0);
// ... use arr ...
// arr destroyed before ctx goes out of scope
```

### Rule
> Always keep a `shared_ptr<Context>` alive for at least as long as any `DeviceArray` or `unique_device_ptr` allocated from it.

## Anti-Pattern 4: Rewinding Scratch While GPU Work Is In Flight

### The Problem
```cpp
auto scratch = naga::slabkit::make_scratch(ctx, 0, 65536);
auto mark = scratch.top();
void* buf = scratch.allocate(4096);

// Submit kernel using buf...
auto ev = q.parallel_for(sycl::range<1>{1024}, [=](sycl::id<1> id) {
    auto ptr = static_cast<float*>(buf);
    ptr[id.get(0)] = static_cast<float>(id.get(0));
});

// BUG: unwind before kernel finishes
scratch.unwind(mark);   // Host offset rewound; buf region is now "free"
void* buf2 = scratch.allocate(4096);  // Returns same address as buf
// buf and buf2 now alias; ev is still running — data race
```

### Why It's Wrong
`unwind()` only adjusts the host-side bump pointer; it does not touch device memory or cancel in-flight work. The next `allocate()` after unwind returns the same address range, causing aliasing with active device memory. Any pending kernel using the old pointer now races with the kernel using the new allocation.

### The Fix
```cpp
auto scratch = naga::slabkit::make_scratch(ctx, 0, 65536);
auto mark = scratch.top();
void* buf = scratch.allocate(4096);

auto ev = q.parallel_for(sycl::range<1>{1024}, [=](sycl::id<1> id) {
    auto ptr = static_cast<float*>(buf);
    ptr[id.get(0)] = static_cast<float>(id.get(0));
});
ev.wait();  // Wait before rewinding

scratch.unwind(mark);   // Safe: GPU is done with buf
void* buf2 = scratch.allocate(4096);  // Safe reuse
```

### Rule
> Always wait for all in-flight kernels using scratch memory before calling `unwind()` or `reset()`.

## Anti-Pattern 5: Using `q.submit(handler)` for USM Kernels

### The Problem
```cpp
auto view = arr.view();
// Unnecessary: submit+handler is for buffer/accessor workflows
q.submit([&](sycl::handler& h) {
    h.parallel_for(view.get_range(), [=](sycl::id<2> id) {
        view[id] *= 2.0f;
    });
}).wait();
```

### Why It's Wrong
SlabKit uses USM (Unified Shared Memory). USM pointers and views are accessible directly in kernels without accessors. `q.submit(handler)` is required for the buffer/accessor model, not USM. While it works, it adds boilerplate (the handler closure) with no benefit.

### The Fix
```cpp
auto view = arr.view();
q.parallel_for(view.get_range(), [=](sycl::id<2> id) {
    view[id] *= 2.0f;
}).wait();
```

### Rule
> Use `q.parallel_for()` directly for USM kernels; reserve `q.submit()` for buffer/accessor workflows.

## Anti-Pattern 6: Indexing `DeviceView` with the Wrong Type

### The Problem
```cpp
auto arr = naga::slabkit::make_device_array<float, 2>(sycl::range<2>{64, 64}, ctx, 0);
auto view = arr.view();
q.parallel_for(view.get_range(), [=](sycl::id<2> id) {
    // WRONG: operator() expects 2 ints, not a sycl::id<2>
    view(id) = 1.0f;

    // ALSO WRONG for 2D: only one of the two dimensions is used
    view(id[0]) = 1.0f;
}).wait();
```

### Why It's Wrong
`DeviceView::operator()(idx...)` takes exactly N integer indices (for N-dimensional row-major access), not a `sycl::id<N>`. Passing `id` directly either fails to compile (for N > 1) or accesses only the first dimension (silently wrong). `operator[](sycl::id<N>)` is the correct form for use with a SYCL id.

### The Fix
```cpp
auto arr = naga::slabkit::make_device_array<float, 2>(sycl::range<2>{64, 64}, ctx, 0);
auto view = arr.view();
q.parallel_for(view.get_range(), [=](sycl::id<2> id) {
    view[id] = 1.0f;           // Correct: operator[] takes sycl::id<N>
    // or: view(id[0], id[1])  // Also correct: operator() takes N ints
}).wait();
```

### Rule
> Use `view[id]` in kernels; use `view(i, j, ...)` only when you have individual integer indices.

## Anti-Pattern 7: Constructing Context from Multiple Threads

### The Problem
```cpp
// Two threads each build their own context concurrently — UNSAFE
std::thread t1([&] {
    auto ctx1 = naga::slabkit::ContextBuilder().devices({dev}).build();
    // ...
});
std::thread t2([&] {
    auto ctx2 = naga::slabkit::ContextBuilder().devices({dev}).build();
    // ...
});
t1.join(); t2.join();
```

### Why It's Wrong
`ContextBuilder::build()` is not thread-safe. The internal grouping logic (matching devices to platform contexts) is not protected by a mutex. Concurrent calls race on shared SYCL runtime state. The result is undefined behaviour.

### The Fix
```cpp
// Build once on the main thread, before spawning
auto ctx = naga::slabkit::ContextBuilder().devices({dev}).build();

std::thread t1([ctx] {
    // Pool access and queue creation are thread-safe
    auto q = ctx->make_in_order_queue(ctx->devices[0]);
    // ...
});
std::thread t2([ctx] {
    auto q = ctx->make_out_of_order_queue(ctx->devices[0]);
    // ...
});
t1.join(); t2.join();
```

### Rule
> Build `Context` once on the main thread before spawning; all post-construction access is thread-safe.

## Anti-Pattern 8: Assuming Device Memory Is Zero-Initialized

### The Problem
```cpp
auto arr = naga::slabkit::make_device_array<float, 1>(sycl::range<1>{1024}, ctx, 0);

// BUG: arr contains garbage — pool memory is not zeroed
auto view = arr.view();
q.parallel_for(view.get_range(), [=](sycl::id<1> id) {
    // view[id] may be any value from previous use of this pool region
    view[id] += 1.0f;
}).wait();
```

### Why It's Wrong
`make_device_array()` allocates from a slab pool — the memory may contain data from a previous allocation. Similarly, `MirroredArray` zero-initializes the host side but leaves the device side uninitialized. Any computation that assumes zero initial values will produce incorrect results silently.

### The Fix
```cpp
auto arr = naga::slabkit::make_device_array<float, 1>(sycl::range<1>{1024}, ctx, 0);

// Explicitly initialize if you need deterministic values
std::vector<float> zeros(arr.size(), 0.0f);
naga::slabkit::copy_to_device(zeros.data(), arr, q).wait();

// Now arr is zeroed
auto view = arr.view();
q.parallel_for(view.get_range(), [=](sycl::id<1> id) {
    view[id] += 1.0f;  // Safe: starts from 0.0
}).wait();
```

### Rule
> Never assume pool-allocated device memory is zero-initialized; initialize explicitly if needed.

## Anti-Pattern 9: Ignoring `AllocationError` on Pool Exhaustion

### The Problem
```cpp
// No try/catch — if the pool is full, this throws and terminates
auto arr = naga::slabkit::make_device_array<float, 1>(
    sycl::range<1>{1024 * 1024 * 64},  // 256 MB of float
    ctx,
    0
);
```

### Why It's Wrong
If the slab is exhausted (or the pool is fragmented), `make_device_array()` throws `naga::slabkit::AllocationError`. Without a catch block, this propagates as an uncaught exception and terminates the program. In production code — especially loops that allocate many arrays — silent termination is unacceptable.

### The Fix
```cpp
try {
    auto arr = naga::slabkit::make_device_array<float, 1>(
        sycl::range<1>{1024 * 1024 * 64},
        ctx,
        0
    );
    // ... use arr ...
} catch (const naga::slabkit::AllocationError& e) {
    switch (e.reason()) {
    case naga::slabkit::AllocationError::Reason::PoolExhausted:
        // Slab is full; wait for other arrays to be freed, or resize the slab
        break;
    case naga::slabkit::AllocationError::Reason::Fragmented:
        // Slab has space but no contiguous region; compact or resize
        break;
    case naga::slabkit::AllocationError::Reason::ScratchFull:
        // Scratch slab is full; reset or use a larger scratch
        break;
    }
}
```

### Rule
> Always catch `AllocationError` around pool allocation calls in production code; check `e.reason()` to distinguish exhaustion from fragmentation.

## See Also
- [Getting Started](getting-started.md)
- [Concepts](concepts.md)
- [API Reference](api-reference.md)
- [Patterns](patterns.md)
