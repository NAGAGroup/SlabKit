# Subtask 03 — array.hpp Refactor Decisions

## Changes made
- `DeviceArray::from_pool` signature: `shared_ptr<SlabContext>` → `shared_ptr<SlabKitContext>`
- `DeviceUSMPoolDeleter` in from_pool body: `weak_ptr<SlabContext>` → `weak_ptr<SlabKitContext>`
- `make_device_array(range, ctx, device_index)`: updated param type, updated doc comment (removed "Phase 2 API" language)
- Removed registry-backed `make_device_array(range, dev)` that called `AllocatorRegistry::context()`
- Added new `make_device_array(range, ctx, dev)` overload delegating via `ctx->device_index(dev)`
- `MirroredArray::ctx_` member: `shared_ptr<SlabContext>` → `shared_ptr<SlabKitContext>`
- `MirroredArray(range, ctx, device_index)` constructor: updated param type, removed "Phase 2 API" doc
- Removed registry-backed `MirroredArray(range, dev)` constructor that called `AllocatorRegistry::context()` twice

## Unchanged
- All `DeviceView<T,N>` and `DeviceSubView<T,N>` code
- All transfer utilities (Tier 1–4: copy_to_host, copy_to_device, copy, sync_to_host)
- `MirroredArray::clone()` — still uses `ctx_` (now `shared_ptr<SlabKitContext>`)
- `DeviceArray` move-only semantics
- File line count: 609 (was 618, 9 lines removed)

## Commit
`1914832` — "wip: subtask-03 remove registry-backed array overloads, add sycl::device overloads"
