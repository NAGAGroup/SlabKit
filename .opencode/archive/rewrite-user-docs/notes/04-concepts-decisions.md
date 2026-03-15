# Subtask 04 — concepts.md Decisions

## Status
Completed and user-approved. Committed as `4a0444b`.

## Key decisions made

### Thread-safety claims (corrected during session)
- Pool lazy init: mutex-protected double-checked locking ✅
- Pool allocation/deallocation: **thread-safe** — `DeviceMemoryPool` holds internal mutex (verified in `device_pool.hpp`)
- `DeviceScratchMemory`: not thread-safe — caller's responsibility
- Context construction: not thread-safe; build once before threads
- Multiple independent Contexts: safe from different threads

### DeviceUSMPoolDeleter uses weak_ptr (confirmed)
- Holds `weak_ptr<Context>` not `shared_ptr<Context>` to avoid reference cycle
- Arrays do not extend Context lifetime
- If Context destroyed before array: deleter is a no-op (memory already freed with slab)

### No AllocatorRegistry mentioned
- Consistent with prior decisions; AllocatorRegistry is deleted and not documented anywhere

### No device-backed overloads
- Only `make_device_array<T,N>(range, ctx, device_index)` documented (3-arg form)

### USM / kernel API
- All kernel examples use `q.parallel_for()` directly — not `q.submit(handler)` pattern
- This is the whole point of USM

## Sections in concepts.md
1. Why These Concepts Matter
2. The Ownership Hierarchy (ASCII diagram)
3. Context: Root of Ownership (heap alloc, platform grouping, node_size/slab_size, no global state, pool access, queue creation)
4. DeviceArray: Move-Only Ownership and RAII (move-only, unique_ptr + custom deleter, weak_ptr rationale, lazy allocation, RAII, array methods)
5. Views: Non-Owning Projections (DeviceView contiguous, DeviceSubView strided, lifetime invariant, implicit conversion)
6. MirroredArray: Paired Host and Device Storage
7. DeviceScratchMemory: Bump Allocation for Temporaries
8. Thread Safety
9. Design Rationale Summary
10. See Also
