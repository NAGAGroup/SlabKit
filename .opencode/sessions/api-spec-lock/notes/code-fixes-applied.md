# Subtask 02 — Code Fixes Applied

## What Was Done

Applied critical bug fixes and `[[nodiscard]]` API quality improvements across 4 header files.

## Bugs Already Fixed (no change needed)

- **Bug 1** (`find_contiguous_short` double-advance): No spurious `++w` before `continue` exists in current code. Already clean.
- **Bug 2** (`find_contiguous_long` exit path): Already logically correct — inner loop sets `w = ww` before `continue` on `!run_ok`; valid run returns early via `return slot`. No missing `break`.
- **Bug 5** (const_cast UB in `const_subview()`): `subview_impl()` const helper already exists. `const_subview()` delegates to it directly. No UB present.

## Bugs Fixed

### Bug 3 — Cross-device assertion in `copy(DeviceArray, DeviceArray)` (`array.hpp`)
Added `assert(src.device_index() == dst.device_index() && "copy(DeviceArray, DeviceArray): src and dst must be on the same device");` before the `memcpy` call. Previously only a size assert was present.

### Bug 4 — Bounds check ordering in `subview_impl()` (`array.hpp`)
The bounds-check loop was executing AFTER the `linear_offset` computation. Moved bounds check to run BEFORE offset arithmetic so out-of-bounds access is caught before any pointer math is performed.

## `[[nodiscard]]` Additions

Added `[[nodiscard]]` to all functions returning resources/events whose discard is almost certainly a bug:

- **`array.hpp`**: `make_device_array` (both overloads), all `copy_to_host` / `copy_to_device` / `copy` free functions returning `sycl::event`
- **`context.hpp`**: `make_in_order_queue`, `make_out_of_order_queue`, `make_queue_with_props`
- **`memory.hpp`**: both `malloc_device` overloads
- **`scratch.hpp`**: both `make_scratch` overloads

## Context Move Semantics Fix (`context.hpp`)

`Context` previously had: copy ctor deleted, copy assign deleted, move assign deleted, but move ctor USER-DEFINED (problematic with `std::mutex` member). Fixed by:
- Deleting move constructor explicitly
- `ContextBuilder::build()` refactored: was `Context result; ...; new Context(std::move(result))` — now `std::shared_ptr<Context>(new Context())` with field-by-field assignment, so no move constructor is needed.

## Build Note

SlabKit is header-only. No compilation step for headers alone. Test suite requires SYCL/AdaptiveCpp runtime (not available in current environment). Headers verified by manual inspection.
