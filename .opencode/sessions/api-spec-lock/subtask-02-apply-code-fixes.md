# Subtask 02 — apply-code-fixes

## Objective
Apply the critical bug fixes and API quality improvements identified in the code review. This covers: two allocator pool logic bugs, a cross-device safety assertion, a bounds-check ordering fix, a const_cast UB fix, and `[[nodiscard]]` annotations across all relevant public API functions.

## Scope

### Edit
- `include/slabkit/detail/device_pool.hpp` — pool logic bugs
- `include/slabkit/array.hpp` — cross-device assertion, bounds-check ordering, const_cast fix, [[nodiscard]] on transfer functions
- `include/slabkit/context.hpp` — move semantics fix, [[nodiscard]] on queue factories
- `include/slabkit/memory.hpp` — [[nodiscard]] on malloc_device
- `include/slabkit/scratch.hpp` — [[nodiscard]] on make_scratch

### Read
- All files being edited (read before editing)

### Excluded
- `tests/` — no test changes in this subtask
- `docs/` — no doc changes in this subtask
- `include/slabkit/detail/device_stack.hpp` — no changes needed

## Constraints
- **Bug 1 — double-advance in find_contiguous_short:** In `device_pool.hpp`, inside the `find_contiguous_short` function, there is a spurious `++w` before a `continue` statement (around line 316). The loop already increments `w` as its loop step. Delete only the explicit `++w` before `continue`; leave the `continue` itself. This causes spurious `Fragmented` errors when pool has space.
- **Bug 2 — find_contiguous_long control flow:** In `device_pool.hpp`, the `find_contiguous_long` function (around lines 344-412) has fragile exit logic. Locate the section where the fully-free word scan fails and add an explicit `break` statement to make the exit path clear and correct. Read the function carefully and understand the logic before editing.
- **Bug 3 — cross-device assertion in copy(DeviceArray, DeviceArray):** In `array.hpp` around lines 309-313, the device-to-device contiguous copy function has no check that src and dst are on the same device. Add `assert(src.device_index() == dst.device_index() && "copy(DeviceArray, DeviceArray): src and dst must be on the same device");` before the memcpy call. Include `<cassert>` if not already included.
- **Bug 4 — bounds check ordering in subview():** In `array.hpp` around lines 192-211, `DeviceArray::subview()` computes the offset pointer before performing bounds validation. Move the bounds-check loop to run BEFORE the offset arithmetic. Read the existing code carefully to understand what to move.
- **Bug 5 — const_cast UB in const_subview():** In `array.hpp`, `const_subview()` uses `const_cast` to delegate to `subview()`. This is UB if the object is const-qualified. Extract a private const-correct implementation helper that both `subview()` and `const_subview()` delegate to (e.g., a private `subview_impl()` function templated on `T` that does the actual work). Read the current implementation before writing the fix.
- **[[nodiscard]] additions:**
  - `context.hpp`: Add `[[nodiscard]]` to `make_in_order_queue`, `make_out_of_order_queue`, `make_queue_with_props`
  - `memory.hpp`: Add `[[nodiscard]]` to both `malloc_device` overloads
  - `scratch.hpp`: Add `[[nodiscard]]` to both `make_scratch` overloads
  - `array.hpp`: Add `[[nodiscard]]` to all `copy_to_host`, `copy_to_device`, and `copy` free functions that return `sycl::event`; also add `[[nodiscard]]` to `make_device_array` overloads
- **Context move semantics:** In `context.hpp`, the `Context` struct has an asymmetry: move constructor is enabled but move assignment is deleted. Since `Context` contains a `std::mutex` which is not movable, the correct fix is to delete BOTH the move constructor and move assignment operator explicitly. Make `Context` fully non-movable (and non-copyable). It is always managed via `shared_ptr` and never moved in practice.
- Do NOT reformat or re-indent code unrelated to the fixes — minimise diff noise
- Do NOT change any function signatures, return types, or semantics beyond what is specified above
- All changes must be correct C++20

## Todolist
- [ ] Read `device_pool.hpp` and fix Bug 1 (double-advance in find_contiguous_short)
- [ ] Read `device_pool.hpp` and fix Bug 2 (explicit break in find_contiguous_long)
- [ ] Read `array.hpp` and fix Bug 3 (cross-device assertion in copy(DeviceArray, DeviceArray))
- [ ] Read `array.hpp` and fix Bug 4 (bounds check ordering in subview())
- [ ] Read `array.hpp` and fix Bug 5 (const_cast UB in const_subview())
- [ ] Add `[[nodiscard]]` to all relevant functions in context.hpp, memory.hpp, scratch.hpp, array.hpp
- [ ] Fix Context move semantics asymmetry in context.hpp

## Delegation
- **Agent:** SlabKitImplAgent
- **Model:** PLACEHOLDER_MODEL_ID
- **Type:** Implementation (bug fixes + API attributes)
