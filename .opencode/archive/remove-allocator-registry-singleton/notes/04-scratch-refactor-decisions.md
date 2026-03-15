# Subtask 04 — scratch.hpp Refactor Decisions

## Changes made
- Updated file-level banner comment: removed `make_scratch(dev, size)` registry entry, added `make_scratch(ctx, dev, size)` device entry, removed note about factory implementations being in memory.hpp
- Added `#include <sycl/sycl.hpp>` (needed for new `sycl::device` overload parameter)
- `make_scratch(ctx, device_index, size)`: `shared_ptr<SlabContext>` → `shared_ptr<SlabKitContext>`, updated doc comment
- Removed trailing comment about `registry-backed make_scratch(dev, size)` being in memory.hpp
- Added new `make_scratch(ctx, dev, size)` overload using `ctx->pool_for(dev)` directly

## Key decision
Used `pool_for(dev)` in the device overload (rather than `pool_at(device_index(dev))`). This is an equivalent but more concise call path — `SlabKitContext::pool_for` accepts a `sycl::device&` directly.

## Unchanged
- `DeviceScratchBuffer` alias
- `DeviceScratchView` struct (all members, methods)
- `#include "slabkit/detail/device_stack.hpp"` and `#include "slabkit/context.hpp"`
- File line count: 77 (was 78, 1 line removed net)

## Commit
`da91302` — "wip: subtask-04 remove registry-backed make_scratch, add sycl::device overload"
