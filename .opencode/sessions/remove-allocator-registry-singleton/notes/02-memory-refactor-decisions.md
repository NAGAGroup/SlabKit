# Subtask 02 — memory.hpp Refactor Decisions

## What was done
- Deleted `class AllocatorRegistry` entirely (class declaration + all 12 inline method definitions)
- Deleted registry-backed `malloc_device<T>(count, dev)` overload
- Deleted `make_scratch(dev, size)` registry-backed inline
- Deleted all three queue factory free functions (`make_in_order_queue`, `make_out_of_order_queue`, `make_queue_with_props`)
- Deleted `#ifdef SLABKIT_TESTING` / `reset_for_testing()` block
- Updated `DeviceUSMPoolDeleter::ctx` from `weak_ptr<SlabContext>` → `weak_ptr<SlabKitContext>`
- Updated `malloc_device(count, ctx, device_index)` to use `SlabKitContext`
- Added new `malloc_device(count, ctx, dev)` overload that delegates via `ctx->device_index(dev)`
- Trimmed includes: removed algorithm, cassert, mutex, stdexcept, vector, device_stack.hpp, error.hpp

## Final file shape
83 lines total. Contains only:
- `DeviceUSMPoolDeleter` struct
- `unique_device_ptr<T>` alias
- Two `malloc_device<T>` free function templates

## Notes
- Queue factory functions (make_in_order_queue etc.) are now on `SlabKitContext` directly — callers must use `ctx->make_in_order_queue(dev)` instead of the free function form
- `make_scratch` free function is gone from memory.hpp — it will be defined in scratch.hpp only (subtask 04)
