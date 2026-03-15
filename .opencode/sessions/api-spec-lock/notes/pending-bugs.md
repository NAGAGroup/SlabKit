# Note: Pending Critical Bugs (Subtask 02 Scope)

**Date:** 2026-03-15
**Session:** api-spec-lock, Subtask 02 (not yet started)

## Bugs to Fix in Subtask 02

1. **`find_contiguous_short` double-advance** (`detail/device_pool.hpp` ~line 316)
   - Loop body does `++w; continue` then loop's own `++w` fires → skips word w+1
   - Fix: delete the manual `++w` inside the loop body; keep only `continue`
   - Impact: spurious `AllocationError::Fragmented` on pools with available space

2. **`copy(DeviceArray, DeviceArray)` cross-device unchecked** (`array.hpp` ~line 309-313)
   - No assertion that `src.device_index() == dst.device_index()`
   - SYCL memcpy between different contexts is UB
   - Fix: add `assert(src.device_index() == dst.device_index())` before memcpy

3. **`find_contiguous_long` fragile control flow** (`detail/device_pool.hpp` ~line 344-412)
   - Exit logic is implicit; add explicit `break` to make flow clear

4. **`DeviceArray::subview()` bounds check after pointer arithmetic** (`array.hpp` ~line 192-211)
   - Offset computed before bounds validated; should validate first
   - Fix: move bounds-check loop before offset computation

5. **`const_subview()` uses `const_cast`** (`array.hpp`)
   - Extract a const-correct private helper instead of casting

## Attributes to Add

- `[[nodiscard]]` on: `malloc_device`, `make_device_array`, `make_scratch`, all `copy_*` / `copy` returning `sycl::event`, queue factories
- Delete both move ctor and move assignment on `Context` (make fully non-movable)
